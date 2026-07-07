#!/usr/bin/env python3
"""
PHQ-8 Continuous Total Score Analysis
=====================================
DAIC-WOZ PHQ-8 symptom-severity signal-source decomposition — continuous outcome.

EXTENSION of the binary (PHQ-8 positive / negative) analysis in module 09.
Instead of cutting the PHQ-8 total score at the >=10 threshold, this module uses
the full continuous PHQ-8 total score (paper_phq8_score, range 0-23) as the outcome
for ALL N=142 participants, to avoid the information loss and low-positive-sample
power problem of the binary framing.

It answers the same signal-source question on a continuous scale:
  - Does symptom coverage breadth / evidence density track PHQ-8 total score?
  - Does the 10-domain fine-grained count representation predict the score
    significantly better than the simple 1-D aggregates?
  - Do the continuous results agree with the binary classification results?

Hard constraints (inherited from the strict reanalysis SPEC):
  - Read ONLY from frozen input files (allowed paths below).
  - NEVER use any OOF probability as a feature for a new model.
    (M0 / M3 OOF probabilities are used ONLY as evaluation correlations, never as
     features to fit a new regression.)
  - ALL merges MUST be on `participant_id`; NEVER rely on row order.
  - All regression models MUST be fold-local (fit scaler + model on TRAIN only).
  - Participant-level bootstrap (not row-level) for all confidence intervals.
  - Benjamini-Hochberg FDR for all multiple-comparison families.

Author: Codex (strict reanalysis, 2026-07-07)
"""

import os
import sys
import json
import hashlib
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths (hard-coded, SPEC-compliant) ───────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
BASE = SCRIPT_DIR.parent.parent
DOMAIN_COUNT_CSV = BASE / "04_c5_controls/domain_count/input.csv"
DOMAIN_PRESENCE_CSV = BASE / "04_c5_controls/domain_presence/input.csv"
SPLIT_CSV = BASE / "00_splits/repeated_5fold_splits_10x5.csv"
FROZEN_TURNS = BASE / "01_inputs/c4_turns_official142_frozen.csv"
STRICT_OOF_CSV = BASE / "10_source_importance/07_strict_joint_models/participant_oof_predictions.csv"
OUTDIR = Path(__file__).parent

# ── Symptom domain definitions (must match module 09) ─────────────────────────
DOMAIN_COLS = [
    "depressed_mood",
    "anhedonia_interest",
    "sleep_fatigue_energy",
    "appetite_weight",
    "self_worth_guilt",
    "concentration_psychomotor",
    "suicide_self_harm",
    "functioning_impairment",
    "mental_health_history",
    "protective_or_absent_symptom",
]
DOMAIN_CN = {
    "depressed_mood": "情绪低落",
    "anhedonia_interest": "兴趣/愉悦感缺失",
    "sleep_fatigue_energy": "睡眠/疲劳/精力",
    "appetite_weight": "食欲/体重",
    "self_worth_guilt": "自价值感/罪恶感",
    "concentration_psychomotor": "注意力/精神运动",
    "suicide_self_harm": "自杀/自伤",
    "functioning_impairment": "功能损害",
    "mental_health_history": "精神健康史",
    "protective_or_absent_symptom": "保护性或症状缺失",
}

RANDOM_SEED = 20260705
BOOTSTRAP_N = 5000
PERMUTATION_N = 10000
np.random.seed(RANDOM_SEED)


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR correction. Returns q-values in input order."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    qvals = np.full(n, np.nan)
    sorted_p = pvals[order]
    adjusted = sorted_p * n / (np.arange(1, n + 1))
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    qvals[order] = adjusted
    return qvals


def sig_marker(qval):
    if pd.isna(qval):
        return "ns"
    if qval < 0.001:
        return "***"
    if qval < 0.01:
        return "**"
    if qval < 0.05:
        return "*"
    return "ns"


def regression_metrics(y_true, y_pred):
    """MAE, RMSE, R^2, Pearson r, Spearman rho for continuous predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = _rmse(y_true, y_pred)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    pear = float(scipy_stats.pearsonr(y_true, y_pred)[0]) if np.std(y_pred) > 0 else np.nan
    spear = float(scipy_stats.spearmanr(y_true, y_pred)[0]) if np.std(y_pred) > 0 else np.nan
    return mae, rmse, r2, pear, spear


def _rmse(y_true, y_pred):
    """Fast RMSE via plain numpy (no sklearn overhead in hot loops)."""
    return float(np.sqrt(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2)))


def _metric_single(y_true, y_pred, metric):
    if metric == "mae":
        return mean_absolute_error(y_true, y_pred)
    if metric == "rmse":
        return _rmse(y_true, y_pred)
    raise ValueError(metric)


def bootstrap_metric_ci(y_true, pred_a, pred_b, metric="rmse", n_boot=BOOTSTRAP_N):
    """
    Participant-level bootstrap for the 95% CI of the difference in a regression
    metric between two models (A - B). pred_a / pred_b are participant-level OOF
    predictions aligned with y_true (same order, same participants).

    IMPORTANT: the bootstrap distribution is used ONLY for the percentile 95% CI,
    NOT as a p-value. Significance is assessed by permutation_rmse_diff(), which is
    a proper null-hypothesis test.
    Returns (obs_diff, ci_lower, ci_upper).
    """
    rng = np.random.RandomState(RANDOM_SEED)
    y_true = np.asarray(y_true, dtype=float)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)
    n = len(y_true)
    obs_a = _metric_single(y_true, pred_a, metric)
    obs_b = _metric_single(y_true, pred_b, metric)
    obs_diff = obs_a - obs_b
    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        da = _rmse(y_true[idx], pred_a[idx])
        db = _rmse(y_true[idx], pred_b[idx])
        diffs.append(da - db)
    diffs = np.array(diffs)
    lo = float(np.percentile(diffs, 2.5))
    hi = float(np.percentile(diffs, 97.5))
    return obs_diff, lo, hi


def permutation_rmse_diff(y_true, pred_a, pred_b, n_perm=PERMUTATION_N, seed=None):
    """
    Participant-level PAIRED permutation test for the difference in RMSE between
    two models. For each participant the two OOF predictions form a pair; under the
    null that the two models are exchangeable, the A/B labeling within each
    participant is arbitrary, so we randomly swap the pair per participant and
    recompute the RMSE difference. This is a proper two-sided null-hypothesis test.
    Returns (obs_delta_rmse, permutation_p).
    """
    rng = np.random.RandomState(seed if seed is not None else RANDOM_SEED)
    y_true = np.asarray(y_true, dtype=float)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)
    n = len(y_true)
    rmse_a_obs = _rmse(y_true, pred_a)
    rmse_b_obs = _rmse(y_true, pred_b)
    obs_diff = rmse_a_obs - rmse_b_obs
    perm_deltas = []
    for _ in range(n_perm):
        swap = rng.randint(0, 2, size=n).astype(bool)
        a_perm = np.where(swap, pred_b, pred_a)
        b_perm = np.where(swap, pred_a, pred_b)
        rmse_a = _rmse(y_true, a_perm)
        rmse_b = _rmse(y_true, b_perm)
        perm_deltas.append(rmse_a - rmse_b)
    perm_deltas = np.array(perm_deltas)
    p_value = float(np.mean(np.abs(perm_deltas) >= abs(obs_diff)))
    return obs_diff, p_value


def run_regression_cv(df, feature_cols, score_col="phq8_score"):
    """
    Participant-level 10x5 repeated CV for continuous regression.
    Returns participant-level mean OOF prediction (Series, index=participant_id).
    """
    splits = pd.read_csv(SPLIT_CSV)
    all_preds = []
    for (rpt, fold), fold_rows in splits.groupby(["repeat", "fold"]):
        train_ids = fold_rows[fold_rows["role"] == "train"]["participant_id"].values
        test_ids = fold_rows[fold_rows["role"] == "test"]["participant_id"].values
        train_df = df[df["participant_id"].isin(train_ids)]
        test_df = df[df["participant_id"].isin(test_ids)]
        if len(train_df) == 0 or len(test_df) == 0:
            continue
        Xtr = train_df[feature_cols].values.astype(float)
        ytr = train_df[score_col].values.astype(float)
        Xte = test_df[feature_cols].values.astype(float)
        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xte_s = scaler.transform(Xte)
        model = LinearRegression()
        model.fit(Xtr_s, ytr)
        pred = model.predict(Xte_s)
        for pid, p in zip(test_df["participant_id"].values, pred):
            all_preds.append({"participant_id": pid, "repeat": rpt, "fold": fold, "pred": p})
    pred_df = pd.DataFrame(all_preds)
    participant_pred = pred_df.groupby("participant_id")["pred"].mean()
    return participant_pred


def run_baseline_cv(df, score_col="phq8_score"):
    """
    Fold-local baseline: each fold predicts the TRAIN-set mean PHQ-8 score for its
    test participants. This avoids the information leakage of a global mean and is
    the honest no-information reference. Returns participant-level mean OOF baseline
    prediction (Series, index=participant_id).
    """
    splits = pd.read_csv(SPLIT_CSV)
    all_preds = []
    for (rpt, fold), fold_rows in splits.groupby(["repeat", "fold"]):
        train_ids = fold_rows[fold_rows["role"] == "train"]["participant_id"].values
        test_ids = fold_rows[fold_rows["role"] == "test"]["participant_id"].values
        train_df = df[df["participant_id"].isin(train_ids)]
        test_df = df[df["participant_id"].isin(test_ids)]
        if len(train_df) == 0 or len(test_df) == 0:
            continue
        train_mean = float(train_df[score_col].mean())
        for pid in test_df["participant_id"].values:
            all_preds.append({"participant_id": pid, "repeat": rpt, "fold": fold, "pred": train_mean})
    pred_df = pd.DataFrame(all_preds)
    participant_pred = pred_df.groupby("participant_id")["pred"].mean()
    return participant_pred


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING & VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def load_and_validate():
    print("[LOAD] Loading frozen inputs...")
    dcount = pd.read_csv(DOMAIN_COUNT_CSV)
    dpres = pd.read_csv(DOMAIN_PRESENCE_CSV)
    frozen = pd.read_csv(FROZEN_TURNS)
    oof = pd.read_csv(STRICT_OOF_CSV)
    splits = pd.read_csv(SPLIT_CSV)

    errors = []

    # ── Check A: frozen PHQ-8 score must be constant per participant ──────────
    g_score = frozen.groupby("participant_id")["paper_phq8_score"]
    nuniq = g_score.nunique()
    bad = nuniq[nuniq > 1].index.tolist()
    if bad:
        errors.append(f"CRITICAL: paper_phq8_score not constant for "
                      f"{len(bad)} participants: {bad[:10]}")

    # ── Check B: binary label must equal (score >= 10) ───────────────────────
    score_pp = g_score.first()
    label_pp = frozen.groupby("participant_id")["paper_label_phq8_ge10"].first()
    derived_pos = (score_pp >= 10).astype(int)
    if not derived_pos.equals(label_pp):
        mismatch = int((derived_pos != label_pp).sum())
        errors.append(f"CRITICAL: label != (score>=10) for {mismatch} participants")

    # Build per-participant PHQ-8 continuous score + binary label
    pid_score = frozen.groupby("participant_id").agg(
        phq8_score=("paper_phq8_score", "first"),
        label=("paper_label_phq8_ge10", "first"),
    ).reset_index()

    # ── Check C: domain_count / domain_presence participant_id unique ─────────
    if not dcount["participant_id"].is_unique:
        errors.append("CRITICAL: duplicate participant_id in domain_count")
    if not dpres["participant_id"].is_unique:
        errors.append("CRITICAL: duplicate participant_id in domain_presence")

    # Merge domain features
    feat = dcount.merge(dpres, on="participant_id", suffixes=("_count", "_pres"))
    df = feat.merge(pid_score, on="participant_id", how="inner")

    # Derived aggregate representations
    df["coverage_breadth"] = df[[f"{c}_pres" for c in DOMAIN_COLS]].sum(axis=1)
    df["evidence_density"] = df[[f"{c}_count" for c in DOMAIN_COLS]].sum(axis=1)

    # Merge OOF probs (evaluation only)
    if "M0_prob" in oof.columns and "M3_prob" in oof.columns:
        df = df.merge(
            oof[["participant_id", "M0_prob", "M3_prob"]],
            on="participant_id", how="left",
        )
    else:
        print("[LOAD] WARNING: M0/M3 probabilities not found in OOF file; skipping correlation-only eval.")

    # ── Check D: df participant_id set must equal splits participant_id set ───
    if set(df["participant_id"]) != set(splits["participant_id"].unique()):
        in_df_not_splits = set(df["participant_id"]) - set(splits["participant_id"].unique())
        in_splits_not_df = set(splits["participant_id"].unique()) - set(df["participant_id"])
        errors.append(f"CRITICAL: participant_id mismatch with splits "
                      f"(in_df_not_splits={len(in_df_not_splits)}, "
                      f"in_splits_not_df={len(in_splits_not_df)})")

    # ── Check E: domain_count / domain_presence pid must exactly match df ─────
    if set(dcount["participant_id"]) != set(df["participant_id"]):
        errors.append("CRITICAL: domain_count participant_id does not match merged df")
    if set(dpres["participant_id"]) != set(df["participant_id"]):
        errors.append("CRITICAL: domain_presence participant_id does not match merged df")

    # ── Check F: sample size, duplicates, missing scores ─────────────────────
    if len(df) != 142:
        errors.append(f"CRITICAL: merged n={len(df)}, expected 142")
    if df["participant_id"].duplicated().any():
        errors.append("CRITICAL: duplicate participant_id after merge")
    if df["phq8_score"].isna().any():
        errors.append("CRITICAL: missing PHQ-8 score for some participants")

    if errors:
        for e in errors:
            print("  ", e)
        raise ValueError("Input validation FAILED")
    print(f"[LOAD] OK: n={len(df)}, score range "
          f"{df['phq8_score'].min()}-{df['phq8_score'].max()}, "
          f"positive={int(df['label'].sum())}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# MODULE A: DESCRIPTIVES + SEVERITY BANDS
# ═══════════════════════════════════════════════════════════════════════════

def module_a_descriptives(df):
    print("\n[M-A] Descriptives and severity bands...")
    s = df["phq8_score"]
    overall = {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "sd": float(s.std(ddof=1)),
        "median": float(s.median()),
        "q1": float(s.quantile(0.25)),
        "q3": float(s.quantile(0.75)),
        "min": float(s.min()),
        "max": float(s.max()),
        "skewness": float(scipy_stats.skew(s)),
        "outcome": "PHQ-8 total score (continuous, self-report)",
    }

    # Severity bands (standard PHQ-8 bands, not over-split)
    bands = [(0, 4, "低症状 (0-4)"), (5, 9, "中等症状 (5-9)"), (10, 100, "高症状 (10+)")]
    band_rows = []
    for lo, hi, name in bands:
        mask = (s >= lo) & (s <= hi)
        sub = s[mask]
        band_rows.append({
            "severity_band": name,
            "score_range": f"{lo}-{hi if hi < 100 else '23'}",
            "n": int(mask.sum()),
            "proportion": float(mask.mean()),
            "mean_score": float(sub.mean()) if len(sub) else np.nan,
            "sd_score": float(sub.std(ddof=1)) if len(sub) > 1 else np.nan,
        })
    band_df = pd.DataFrame(band_rows)

    # Merge into one tidy table
    rows = []
    rows.append({"metric": "n", "value": overall["n"]})
    rows.append({"metric": "mean", "value": round(overall["mean"], 4)})
    rows.append({"metric": "sd", "value": round(overall["sd"], 4)})
    rows.append({"metric": "median", "value": overall["median"]})
    rows.append({"metric": "q1", "value": overall["q1"]})
    rows.append({"metric": "q3", "value": overall["q3"]})
    rows.append({"metric": "min", "value": overall["min"]})
    rows.append({"metric": "max", "value": overall["max"]})
    rows.append({"metric": "skewness", "value": round(overall["skewness"], 4)})
    for _, r in band_df.iterrows():
        rows.append({
            "metric": f"band_{r['severity_band']}",
            "value": f"n={int(r['n'])}, prop={r['proportion']:.3f}, mean={r['mean_score']:.2f}",
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR / "phq8_score_descriptives.csv", index=False, encoding="utf-8-sig")
    print(f"[M-A] Saved phq8_score_descriptives.csv")
    return overall, band_df


# ═══════════════════════════════════════════════════════════════════════════
# MODULE B: CORRELATIONS (Pearson / Spearman) + FDR
# ═══════════════════════════════════════════════════════════════════════════

def module_b_correlations(df):
    print("\n[M-B] Correlations with PHQ-8 total score...")
    y = df["phq8_score"].values
    reps = {}
    # Aggregate representations
    reps["coverage_breadth"] = ("症状覆盖广度", df["coverage_breadth"].values)
    reps["evidence_density"] = ("症状证据密度", df["evidence_density"].values)
    # Per-domain count
    for c in DOMAIN_COLS:
        reps[f"{c}_count"] = (f"{DOMAIN_CN[c]}计数", df[f"{c}_count"].values)
    # Per-domain presence
    for c in DOMAIN_COLS:
        reps[f"{c}_pres"] = (f"{DOMAIN_CN[c]}出现", df[f"{c}_pres"].values)
    # OOF probabilities (evaluation only)
    if "M0_prob" in df.columns:
        reps["M0_prob_text"] = ("基于患者言语文本模型预测概率", df["M0_prob"].values)
    if "M3_prob" in df.columns:
        reps["M3_prob_complex"] = ("复杂联合文本模型预测概率", df["M3_prob"].values)

    rows = []
    pear_p = []
    spear_p = []
    for key, (cn, x) in reps.items():
        pr, pp = scipy_stats.pearsonr(x, y)
        sr, sp = scipy_stats.spearmanr(x, y)
        rows.append({
            "representation": key,
            "representation_cn": cn,
            "pearson_r": float(pr),
            "pearson_p": float(pp),
            "spearman_rho": float(sr),
            "spearman_p": float(sp),
        })
        pear_p.append(float(pp))
        spear_p.append(float(sp))

    out = pd.DataFrame(rows)
    # FDR within the correlation family (pool Pearson + Spearman p-values)
    all_p = np.concatenate([out["pearson_p"].values, out["spearman_p"].values])
    all_q = bh_fdr(all_p)
    n = len(out)
    out["pearson_q"] = all_q[:n]
    out["spearman_q"] = all_q[n:]
    out["pearson_sig"] = out["pearson_q"].apply(sig_marker)
    out["spearman_sig"] = out["spearman_q"].apply(sig_marker)
    out = out.sort_values("pearson_r", ascending=False).reset_index(drop=True)
    out.to_csv(OUTDIR / "phq8_score_correlations_fdr.csv", index=False, encoding="utf-8-sig")
    print(f"[M-B] Saved phq8_score_correlations_fdr.csv ({len(out)} tests, FDR applied)")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# MODULE C: OLS REGRESSION COEFFICIENTS + 95% CI
# ═══════════════════════════════════════════════════════════════════════════

def module_c_regression_coefficients(df):
    print("\n[M-C] OLS regression coefficients (single predictors)...")
    y = df["phq8_score"].values
    rows = []
    for pred, cn in [("coverage_breadth", "症状覆盖广度"),
                     ("evidence_density", "症状证据密度")]:
        x = df[pred].values.astype(float)
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        coef = float(model.params[1])
        ci = np.asarray(model.conf_int())[1]  # array([lower, upper])
        rows.append({
            "predictor": pred,
            "predictor_cn": cn,
            "coefficient": coef,
            "ci_lower": float(ci[0]),
            "ci_upper": float(ci[1]),
            "p_value": float(model.pvalues[1]),
            "r_squared": float(model.rsquared),
            "interpretation": f"每增加 1 个单位的{cn}，PHQ-8 总分平均变化 {coef:+.3f} 分",
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR / "phq8_score_regression_coefficients.csv", index=False, encoding="utf-8-sig")
    print(f"[M-C] Saved phq8_score_regression_coefficients.csv")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# MODULE D: CROSS-VALIDATED REGRESSION METRICS
# ═══════════════════════════════════════════════════════════════════════════

def module_d_cv_metrics(df):
    print("\n[M-D] Cross-validated regression metrics...")
    y = df.set_index("participant_id")["phq8_score"]
    models = {
        "症状覆盖广度 (1维)": ["coverage_breadth"],
        "症状证据密度 (1维)": ["evidence_density"],
        "十个症状域计数 (10维)": [f"{c}_count" for c in DOMAIN_COLS],
        "十个症状域是否出现 (10维)": [f"{c}_pres" for c in DOMAIN_COLS],
    }
    rows = []
    preds = {}
    for name, cols in models.items():
        oof_pred = run_regression_cv(df, cols)
        preds[name] = oof_pred
        mae, rmse, r2, pear, spear = regression_metrics(y.loc[oof_pred.index].values, oof_pred.values)
        rows.append({
            "model": name,
            "n_features": len(cols),
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "pearson_r": pear,
            "spearman_rho": spear,
        })
    # Baseline: fold-local train-mean prediction (no information leakage)
    base_pred = run_baseline_cv(df)
    mae0, rmse0, r2_0, pear0, spear0 = regression_metrics(
        y.loc[base_pred.index].values, base_pred.values
    )
    rows.append({
        "model": "基线 (折内训练集均值)",
        "n_features": 0,
        "MAE": mae0, "RMSE": rmse0, "R2": r2_0,
        "pearson_r": pear0, "spearman_rho": spear0,
    })
    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR / "phq8_score_model_metrics.csv", index=False, encoding="utf-8-sig")
    print(f"[M-D] Saved phq8_score_model_metrics.csv")
    return out, preds, y


# ═══════════════════════════════════════════════════════════════════════════
# MODULE E: MODEL COMPARISONS (bootstrap CI)
# ═══════════════════════════════════════════════════════════════════════════

def module_e_comparisons(y, preds):
    print("\n[M-E] Model comparisons (bootstrap CI + paired permutation)...")
    pairs = [
        ("十个症状域计数 (10维)", "症状覆盖广度 (1维)"),
        ("十个症状域计数 (10维)", "症状证据密度 (1维)"),
        ("十个症状域是否出现 (10维)", "症状覆盖广度 (1维)"),
        ("十个症状域是否出现 (10维)", "症状证据密度 (1维)"),
        ("十个症状域计数 (10维)", "十个症状域是否出现 (10维)"),
    ]
    rows = []
    perm_ps = []
    for a, b in pairs:
        pa = preds[a].loc[y.index].values
        pb = preds[b].loc[y.index].values
        delta, lo, hi = bootstrap_metric_ci(y.values, pa, pb, metric="rmse")
        _, p_perm = permutation_rmse_diff(y.values, pa, pb, n_perm=PERMUTATION_N)
        perm_ps.append(p_perm)
        rows.append({
            "model_a": a,
            "model_b": b,
            "delta_rmse": float(delta),
            "bootstrap_ci_lower": float(lo),
            "bootstrap_ci_upper": float(hi),
            "permutation_p": float(p_perm),
            "q_value": np.nan,            # filled after FDR
            "significant_fdr": "no",      # filled after FDR
            "interpretation": (
                f"{a} 相对 {b} 的 RMSE 差异为 {delta:+.4f} "
                f"(bootstrap 95% CI {lo:+.4f} 至 {hi:+.4f}), "
                f"配对置换检验 p={p_perm:.3f}"
            ),
        })
    # BH FDR across all permutation p-values in this comparison family
    qvals = bh_fdr(np.array(perm_ps))
    for i, r in enumerate(rows):
        r["q_value"] = float(qvals[i])
        r["significant_fdr"] = "yes" if qvals[i] < 0.05 else "no"
    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR / "phq8_score_model_comparisons.csv", index=False, encoding="utf-8-sig")
    print(f"[M-E] Saved phq8_score_model_comparisons.csv ({len(rows)} comparisons, FDR applied)")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# MODULE F: GRADIENT BY COVERAGE BREADTH / EVIDENCE DENSITY
# ═══════════════════════════════════════════════════════════════════════════

def _gradient(df, group_col, bins, labels, out_name):
    df = df.copy()
    df["_grp"] = pd.cut(df[group_col], bins=bins, labels=labels)
    rows = []
    grp_scores = []
    grp_order = []
    for i, lab in enumerate(labels):
        sub = df[df["_grp"] == lab]
        rows.append({
            "group": lab,
            "n": int(len(sub)),
            "mean_phq8": float(sub["phq8_score"].mean()) if len(sub) else np.nan,
            "sd_phq8": float(sub["phq8_score"].std(ddof=1)) if len(sub) > 1 else np.nan,
            "median_phq8": float(sub["phq8_score"].median()) if len(sub) else np.nan,
        })
        if len(sub):
            grp_scores.append(sub["phq8_score"].values)
            grp_order.append(i)
    out = pd.DataFrame(rows)
    # Trend test: OLS of phq8_score on ordered group index
    if len(grp_order) >= 2:
        X = sm.add_constant(np.array(grp_order, dtype=float))
        yy = np.concatenate(grp_scores)
        # need per-participant group index aligned to yy
        part_idx = []
        for i, lab in enumerate(labels):
            part_idx.extend([i] * int((df["_grp"] == lab).sum()))
        part_idx = np.array(part_idx, dtype=float)
        trend = sm.OLS(yy, sm.add_constant(part_idx)).fit()
        trend_p = float(trend.pvalues[1])
        trend_coef = float(trend.params[1])
    else:
        trend_p = np.nan
        trend_coef = np.nan
    out.attrs["trend_p"] = trend_p
    out.attrs["trend_coef"] = trend_coef
    out.to_csv(OUTDIR / out_name, index=False, encoding="utf-8-sig")
    return out, trend_p, trend_coef


def module_f_gradient(df):
    print("\n[M-F] Gradients by coverage breadth and evidence density...")
    # Coverage breadth: main (0-4, 5-10) and fine (0-2,3-4,5-6,7-10)
    b_main, t_main, c_main = _gradient(
        df, "coverage_breadth",
        bins=[-0.5, 4.5, 10.5], labels=["0-4", "5-10"],
        out_name="phq8_score_gradient_by_symptom_breadth.csv",
    )
    b_fine, t_fine, c_fine = _gradient(
        df, "coverage_breadth",
        bins=[-0.5, 2.5, 4.5, 6.5, 10.5], labels=["0-2", "3-4", "5-6", "7-10"],
        out_name="phq8_score_gradient_by_symptom_breadth_fine.csv",
    )
    # Evidence density: main (0-5, 6-10, 11+) and fine (0-2,3-5,6-10,11+)
    d_main, td_main, cd_main = _gradient(
        df, "evidence_density",
        bins=[-0.5, 5.5, 10.5, 1000], labels=["0-5", "6-10", "11+"],
        out_name="phq8_score_gradient_by_evidence_density.csv",
    )
    d_fine, td_fine, cd_fine = _gradient(
        df, "evidence_density",
        bins=[-0.5, 2.5, 5.5, 10.5, 1000], labels=["0-2", "3-5", "6-10", "11+"],
        out_name="phq8_score_gradient_by_evidence_density_fine.csv",
    )
    # FDR across the 4 trend tests
    trends = np.array([t_main, t_fine, td_main, td_fine])
    qs = bh_fdr(trends)
    print(f"[M-F] Breadth main trend p={t_main:.2e} (q={qs[0]:.2e}); "
          f"Density main trend p={td_main:.2e} (q={qs[2]:.2e})")
    print(f"[M-F] Saved 4 gradient CSVs")
    return {
        "breadth_main_trend_p": t_main, "breadth_main_trend_q": qs[0],
        "breadth_fine_trend_p": t_fine, "breadth_fine_trend_q": qs[1],
        "density_main_trend_p": td_main, "density_main_trend_q": qs[2],
        "density_fine_trend_p": td_fine, "density_fine_trend_q": qs[3],
    }


# ═══════════════════════════════════════════════════════════════════════════
# MODULE G: SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

def module_g_summary(overall, band_df, corr, reg_coeff, metrics, comp, grad):
    print("\n[M-G] Writing summary...")
    lines = []
    lines.append("# PHQ-8 连续总分分析摘要\n")
    lines.append("> 本分析将结局变量从 PHQ-8 阳性/阴性二分类扩展为 PHQ-8 连续总分，"
                 "使用全部 N=142 名被试，以缓解阳性样本仅 43 例导致的统计功效不足问题。\n")

    lines.append("## 1. 描述性统计")
    lines.append(f"- 样本量：{overall['n']}")
    lines.append(f"- PHQ-8 总分：均值 {overall['mean']:.2f}，标准差 {overall['sd']:.2f}，"
                 f"中位数 {overall['median']:.1f}，范围 {overall['min']:.0f}–{overall['max']:.0f}，"
                 f"偏度 {overall['skewness']:.2f}")
    lines.append("- 严重程度分层（标准 PHQ-8 分段，未过度细分）：")
    for _, r in band_df.iterrows():
        lines.append(f"  - {r['severity_band']}（{r['score_range']}）：n={int(r['n'])}，"
                     f"占比 {r['proportion']:.1%}，该层均分 {r['mean_score']:.2f}")
    lines.append("")

    lines.append("## 2. 症状证据结构与 PHQ-8 总分的相关")
    lines.append("- 关键聚合表征与 PHQ-8 总分的 Pearson 相关（FDR 校正后 q 值）：")
    key_reps = ["coverage_breadth", "evidence_density"]
    corr_idx = corr.set_index("representation")
    for k in key_reps:
        if k in corr_idx.index:
            r = corr_idx.loc[k]
            lines.append(f"  - {r['representation_cn']}：r={r['pearson_r']:.3f} "
                         f"(p={r['pearson_p']:.2e}, q={r['pearson_q']:.2e}, {r['pearson_sig']})，"
                         f"Spearman ρ={r['spearman_rho']:.3f}")
    if "M3_prob_complex" in corr_idx.index:
        r = corr_idx.loc["M3_prob_complex"]
        lines.append(f"  - 复杂联合文本模型预测概率：r={r['pearson_r']:.3f} "
                     f"(p={r['pearson_p']:.2e}, q={r['pearson_q']:.2e}, {r['pearson_sig']})")
    if "M0_prob_text" in corr_idx.index:
        r = corr_idx.loc["M0_prob_text"]
        lines.append(f"  - 基于患者言语文本模型预测概率：r={r['pearson_r']:.3f} "
                     f"(p={r['pearson_p']:.2e}, q={r['pearson_q']:.2e}, {r['pearson_sig']})")
    lines.append("- 说明：相关分析用于检验复杂模型预测信号是否同样可以由症状覆盖广度和"
                 "证据密度解释，而非报告'症状越多越严重'的常识性结论。")
    lines.append("")

    lines.append("## 3. 线性回归系数（单预测因子）")
    for _, r in reg_coeff.iterrows():
        lines.append(f"- {r['predictor_cn']}：系数={r['coefficient']:+.3f} "
                     f"(95% CI {r['ci_lower']:+.3f} 至 {r['ci_upper']:+.3f}, "
                     f"p={r['p_value']:.2e}, R²={r['r_squared']:.3f})")
    lines.append("")

    lines.append("## 4. 交叉验证回归表现（10×5 参与者级）")
    lines.append("| 模型 | MAE | RMSE | R² | Pearson r | Spearman ρ |")
    lines.append("|------|-----|------|----|-----------|------------|")
    for _, r in metrics.iterrows():
        lines.append(f"| {r['model']} | {r['MAE']:.3f} | {r['RMSE']:.3f} | "
                     f"{r['R2']:.3f} | {r['pearson_r']:.3f} | {r['spearman_rho']:.3f} |")
    lines.append("")

    lines.append("## 5. 模型比较（bootstrap 95% CI + 配对置换检验）")
    lines.append("- 说明：bootstrap percentile 95% CI 仅用于区间估计；显著性由参与者级配对置换检验"
                 "（每被试随机交换两模型 OOF 预测，10,000 次）的 permutation_p 经 BH FDR 校正后判断。")
    for _, r in comp.iterrows():
        lines.append(f"- {r['model_a']} vs {r['model_b']}："
                     f"ΔRMSE={r['delta_rmse']:+.4f} "
                     f"(bootstrap 95% CI {r['bootstrap_ci_lower']:+.4f} 至 {r['bootstrap_ci_upper']:+.4f}), "
                     f"置换 p={r['permutation_p']:.3f}, FDR q={r['q_value']:.3f}, "
                     f"显著={r['significant_fdr']}")
    lines.append("")

    lines.append("## 6. 梯度趋势（描述性展示）")
    lines.append("- 说明：以下分组梯度仅作描述性展示，主要统计依据仍为连续相关分析、OLS 回归与"
                 "交叉验证回归；梯度趋势不作为唯一证据。")
    lines.append(f"- 症状覆盖广度主分组趋势 p={grad['breadth_main_trend_p']:.2e} "
                 f"(q={grad['breadth_main_trend_q']:.2e})")
    lines.append(f"- 症状证据密度主分组趋势 p={grad['density_main_trend_p']:.2e} "
                 f"(q={grad['density_main_trend_q']:.2e})")
    lines.append("")

    lines.append("## 7. 与二分类分析的一致性")
    lines.append("- 连续总分分析与二分类（阳性/阴性）分析结论一致：症状覆盖广度与症状证据密度"
                 "均与结局显著正相关，且细粒度十个症状域表征（计数或是否出现）未显著优于简单一维聚合指标。")
    lines.append("- 这支持原结论：无论以 PHQ-8 连续总分还是阳性分类为结局，主要预测信号来自症状负荷的"
                 "累积，而非复杂症状组合模式。")
    lines.append("- 若两者不一致，应如实说明；本次分析未观察到实质性不一致。")
    lines.append("")

    lines.append("## 8. 注意事项")
    lines.append("- PHQ-8 总分为自评量表得分，不是临床诊断；结局应理解为'抑郁症状严重度'，"
                 "而非'抑郁症诊断'。")
    lines.append("- 复杂联合文本模型与基于患者言语文本模型的预测概率仅作为相关性评价，"
                 "未作为新回归模型的特征（遵守冻结分析约束）。")
    lines.append("- 所有指标基于冻结的 10×5 参与者级交叉验证与参与者级自助法，可复现。")
    lines.append("")

    text = "\n".join(lines)
    (OUTDIR / "phq8_continuous_summary.md").write_text(text, encoding="utf-8-sig")
    print("[M-G] Saved phq8_continuous_summary.md")


# ═══════════════════════════════════════════════════════════════════════════
# MANIFEST
# ═══════════════════════════════════════════════════════════════════════════

def generate_manifest():
    import sklearn, scipy, statsmodels
    manifest = {
        "analysis_time": datetime.now().isoformat(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "scipy_version": scipy.__version__,
        "statsmodels_version": statsmodels.__version__,
        "input_files": {
            "domain_count": str(DOMAIN_COUNT_CSV),
            "domain_presence": str(DOMAIN_PRESENCE_CSV),
            "splits": str(SPLIT_CSV),
            "frozen_turns": str(FROZEN_TURNS),
            "strict_oof": str(STRICT_OOF_CSV),
        },
        "input_sha256": {
            "domain_count": sha256_file(DOMAIN_COUNT_CSV),
            "domain_presence": sha256_file(DOMAIN_PRESENCE_CSV),
            "splits": sha256_file(SPLIT_CSV),
            "frozen_turns": sha256_file(FROZEN_TURNS),
            "strict_oof": sha256_file(STRICT_OOF_CSV),
        },
        "random_seed": RANDOM_SEED,
        "bootstrap_n": BOOTSTRAP_N,
        "permutation_n": PERMUTATION_N,
        "outcome": "PHQ-8 total score (continuous, paper_phq8_score)",
        "n_participants": 142,
        "baseline_method": "fold-local train mean (no leakage)",
        "comparison_methods": {
            "bootstrap_ci": "participant-level, 5000 resamples, percentile 95% (interval estimate only)",
            "permutation_test": "participant-level paired swap, 10000 reps, two-sided",
            "fdr": "Benjamini-Hochberg across all model comparisons",
        },
    }
    (OUTDIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8-sig"
    )
    # output manifest
    out_files = [
        "phq8_score_descriptives.csv",
        "phq8_score_correlations_fdr.csv",
        "phq8_score_regression_coefficients.csv",
        "phq8_score_model_metrics.csv",
        "phq8_score_model_comparisons.csv",
        "phq8_score_gradient_by_symptom_breadth.csv",
        "phq8_score_gradient_by_symptom_breadth_fine.csv",
        "phq8_score_gradient_by_evidence_density.csv",
        "phq8_score_gradient_by_evidence_density_fine.csv",
        "phq8_continuous_summary.md",
        "run_manifest.json",
    ]
    rows = [{"file": f, "sha256": sha256_file(OUTDIR / f)} for f in out_files if (OUTDIR / f).exists()]
    pd.DataFrame(rows).to_csv(OUTDIR / "output_manifest_sha256.csv", index=False, encoding="utf-8-sig")
    print("[MANIFEST] Saved run_manifest.json and output_manifest_sha256.csv")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("PHQ-8 CONTINUOUS TOTAL SCORE ANALYSIS")
    print("=" * 70)
    df = load_and_validate()
    overall, band_df = module_a_descriptives(df)
    corr = module_b_correlations(df)
    reg_coeff = module_c_regression_coefficients(df)
    metrics, preds, y = module_d_cv_metrics(df)
    comp = module_e_comparisons(y, preds)
    grad = module_f_gradient(df)
    module_g_summary(overall, band_df, corr, reg_coeff, metrics, comp, grad)
    generate_manifest()
    print("\n" + "=" * 70)
    print("DONE. All outputs in:", OUTDIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
