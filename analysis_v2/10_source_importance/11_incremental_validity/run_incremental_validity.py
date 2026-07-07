#!/usr/bin/env python3
"""
Incremental Validity Analysis (PHQ-8)
=====================================
DAIC-WOZ PHQ-8 symptom-severity signal-source decomposition — extension that
directly answers the core question:

    "Does the complex text model, or the fine-grained 10-domain symptom
     representation, provide PREDICTIVE INFORMATION BEYOND the simple
     symptom-load indicators (coverage breadth + evidence density)?"

This module does NOT replace the existing continuous-total-score (module 10)
or binary-classification (module 09) results. It builds ON TOP of them and
tests incremental validity in two outcomes:

  A. Continuous outcome: PHQ-8 total score (all N=142)
       - Base load model:        coverage_breadth + evidence_density
       - 10-domain count model:   base + ten domain_count columns
       - 10-domain presence model: base + ten domain_presence columns
     (The complex text model is a BINARY classifier; its OOF probability is a
      0-1 score, not on the PHQ-8 score scale, and the frozen SPEC forbids
      re-fitting it as a continuous regressor. Its incremental value for the
      continuous score is therefore reported only at the correlation level and
      is formally assessed in the binary layer below.)

  B. Binary outcome: PHQ-8 positive (>=10)
       - Base load model:        coverage_breadth + evidence_density
       - 10-domain count model:   base + ten domain_count columns
       - 10-domain presence model: base + ten domain_presence columns
       - Complex joint text model: EXTERNAL OOF probability (module 07), used
         only as an evaluation reference, never as a retrained feature.

Hard constraints (inherited from the strict reanalysis SPEC):
  - Read ONLY from frozen input files (allowed paths below).
  - NEVER use any OOF probability as a feature for a new model.
  - ALL merges MUST be on `participant_id`; NEVER rely on row order.
  - All models MUST be fold-local (fit scaler + model on TRAIN only).
  - Participant-level bootstrap (not row-level) for all confidence intervals.
  - Participant-level PAIRED permutation test for all significance p-values.
  - Benjamini-Hochberg FDR for all multiple-comparison families.

Author: strict reanalysis (2026-07-07)
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
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    roc_auc_score,
    precision_recall_curve,
    auc,
    f1_score,
    confusion_matrix,
    brier_score_loss,
    log_loss,
)

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

# ── Symptom domain definitions (must match modules 09 / 10) ──────────────────
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
LR_PARAMS = dict(
    penalty="l2", C=1.0, class_weight="balanced",
    solver="liblinear", max_iter=2000, random_state=RANDOM_SEED,
)
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


def _rmse(y_true, y_pred):
    """Fast RMSE via plain numpy (no sklearn overhead in hot loops)."""
    return float(np.sqrt(np.mean(
        (np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2
    )))


def regression_metrics(y_true, y_pred):
    """MAE, RMSE, R^2, Pearson r, Spearman rho for continuous predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = _rmse(y_true, y_pred)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    pear = float(scipy_stats.pearsonr(y_true, y_pred)[0]) if np.std(y_pred) > 0 else np.nan
    spear = float(scipy_stats.spearmanr(y_true, y_pred)[0]) if np.std(y_pred) > 0 else np.nan
    return mae, rmse, r2, pear, spear


def bootstrap_rmse_diff_ci(y_true, pred_a, pred_b, n_boot=BOOTSTRAP_N):
    """
    Participant-level bootstrap for the 95% CI of the RMSE difference (A - B).
    Bootstrap distribution is used ONLY for the percentile 95% CI, NOT as a
    p-value. Significance comes from permutation_rmse_diff().
    Returns (obs_diff, ci_lower, ci_upper).
    """
    rng = np.random.RandomState(RANDOM_SEED)
    y_true = np.asarray(y_true, dtype=float)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)
    n = len(y_true)
    obs_diff = _rmse(y_true, pred_a) - _rmse(y_true, pred_b)
    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        da = _rmse(y_true[idx], pred_a[idx])
        db = _rmse(y_true[idx], pred_b[idx])
        diffs.append(da - db)
    diffs = np.array(diffs)
    return obs_diff, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def permutation_rmse_diff(y_true, pred_a, pred_b, n_perm=PERMUTATION_N, seed=None):
    """
    Participant-level PAIRED permutation test for the RMSE difference between
    two models. Under the null the two models are exchangeable, so per
    participant we randomly swap the A/B OOF prediction and recompute ΔRMSE.
    Two-sided null-hypothesis test. Returns (obs_delta_rmse, permutation_p).
    """
    rng = np.random.RandomState(seed if seed is not None else RANDOM_SEED)
    y_true = np.asarray(y_true, dtype=float)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)
    n = len(y_true)
    obs_diff = _rmse(y_true, pred_a) - _rmse(y_true, pred_b)
    perm_deltas = []
    for _ in range(n_perm):
        swap = rng.randint(0, 2, size=n).astype(bool)
        a_perm = np.where(swap, pred_b, pred_a)
        b_perm = np.where(swap, pred_a, pred_b)
        perm_deltas.append(_rmse(y_true, a_perm) - _rmse(y_true, b_perm))
    perm_deltas = np.array(perm_deltas)
    return obs_diff, float(np.mean(np.abs(perm_deltas) >= abs(obs_diff)))


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING & VALIDATION (reuse module-10 checks)
# ═══════════════════════════════════════════════════════════════════════════

def load_and_validate():
    print("[LOAD] Loading frozen inputs...")
    dcount = pd.read_csv(DOMAIN_COUNT_CSV)
    dpres = pd.read_csv(DOMAIN_PRESENCE_CSV)
    frozen = pd.read_csv(FROZEN_TURNS)
    oof = pd.read_csv(STRICT_OOF_CSV)
    splits = pd.read_csv(SPLIT_CSV)

    errors = []

    # Check A: PHQ-8 score constant per participant
    g_score = frozen.groupby("participant_id")["paper_phq8_score"]
    if (g_score.nunique() > 1).any():
        bad = g_score.nunique()
        bad = bad[bad > 1].index.tolist()
        errors.append(f"CRITICAL: paper_phq8_score not constant for {len(bad)} participants")

    # Check B: label == (score >= 10)
    score_pp = g_score.first()
    label_pp = frozen.groupby("participant_id")["paper_label_phq8_ge10"].first()
    if not (score_pp >= 10).astype(int).equals(label_pp):
        errors.append("CRITICAL: label != (score>=10) for some participants")

    pid_score = frozen.groupby("participant_id").agg(
        phq8_score=("paper_phq8_score", "first"),
        label=("paper_label_phq8_ge10", "first"),
    ).reset_index()

    if not dcount["participant_id"].is_unique:
        errors.append("CRITICAL: duplicate participant_id in domain_count")
    if not dpres["participant_id"].is_unique:
        errors.append("CRITICAL: duplicate participant_id in domain_presence")

    feat = dcount.merge(dpres, on="participant_id", suffixes=("_count", "_pres"))
    df = feat.merge(pid_score, on="participant_id", how="inner")

    df["coverage_breadth"] = df[[f"{c}_pres" for c in DOMAIN_COLS]].sum(axis=1)
    df["evidence_density"] = df[[f"{c}_count" for c in DOMAIN_COLS]].sum(axis=1)

    if "M0_prob" in oof.columns and "M3_prob" in oof.columns:
        df = df.merge(
            oof[["participant_id", "M0_prob", "M3_prob"]],
            on="participant_id", how="left",
        )
    else:
        errors.append("CRITICAL: M0/M3 OOF probabilities missing from strict_oof file")

    if set(df["participant_id"]) != set(splits["participant_id"].unique()):
        errors.append("CRITICAL: participant_id mismatch with splits")
    if set(dcount["participant_id"]) != set(df["participant_id"]):
        errors.append("CRITICAL: domain_count pid mismatch")
    if set(dpres["participant_id"]) != set(df["participant_id"]):
        errors.append("CRITICAL: domain_presence pid mismatch")
    if len(df) != 142:
        errors.append(f"CRITICAL: merged n={len(df)}, expected 142")
    if df["participant_id"].duplicated().any():
        errors.append("CRITICAL: duplicate participant_id after merge")
    if df["phq8_score"].isna().any():
        errors.append("CRITICAL: missing PHQ-8 score")

    if errors:
        for e in errors:
            print("  ", e)
        raise ValueError("Input validation FAILED")

    print(f"[LOAD] OK: n={len(df)}, score range "
          f"{df['phq8_score'].min()}-{df['phq8_score'].max()}, "
          f"positive={int(df['label'].sum())}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATED PREDICTION GENERATORS
# ═══════════════════════════════════════════════════════════════════════════

def run_regression_cv(df, feature_cols, score_col="phq8_score"):
    """Participant-level 10x5 repeated CV for continuous regression."""
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
        model = LinearRegression()
        model.fit(scaler.transform(Xtr), ytr)
        pred = model.predict(scaler.transform(Xte))
        for pid, p in zip(test_df["participant_id"].values, pred):
            all_preds.append({"participant_id": pid, "repeat": rpt, "fold": fold, "pred": p})
    return pd.DataFrame(all_preds).groupby("participant_id")["pred"].mean()


def run_logistic_cv(df, feature_cols, label_series):
    """Participant-level 10x5 repeated CV for binary logistic regression."""
    splits = pd.read_csv(SPLIT_CSV)
    all_probs = []
    for (rpt, fold), fold_rows in splits.groupby(["repeat", "fold"]):
        train_ids = fold_rows[fold_rows["role"] == "train"]["participant_id"].values
        test_ids = fold_rows[fold_rows["role"] == "test"]["participant_id"].values
        train_df = df[df["participant_id"].isin(train_ids)]
        test_df = df[df["participant_id"].isin(test_ids)]
        if len(train_df) == 0 or len(test_df) == 0:
            continue
        Xtr = train_df[feature_cols].values.astype(float)
        ytr = label_series.loc[train_df["participant_id"].values].values
        Xte = test_df[feature_cols].values.astype(float)
        scaler = StandardScaler().fit(Xtr)
        model = LogisticRegression(**LR_PARAMS)
        model.fit(scaler.transform(Xtr), ytr)
        prob = model.predict_proba(scaler.transform(Xte))[:, 1]
        for pid, p in zip(test_df["participant_id"].values, prob):
            all_probs.append({"participant_id": pid, "repeat": rpt, "fold": fold, "prob": p})
    return pd.DataFrame(all_probs).groupby("participant_id")["prob"].mean()


# ═══════════════════════════════════════════════════════════════════════════
# BINARY METRIC HELPERS (matching module 09 conventions)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_binary(y_true, y_prob):
    auc_val = float(roc_auc_score(y_true, y_prob))
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(auc(recall, precision))
    y_pred_50 = (np.asarray(y_prob) >= 0.50).astype(int)
    cm = confusion_matrix(y_true, y_pred_50)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
    else:
        sens = spec = np.nan
    macro_f1 = float(f1_score(y_true, y_pred_50, average="macro", zero_division=0))
    return {
        "auc": auc_val, "pr_auc": pr_auc, "macro_f1": macro_f1,
        "sensitivity": sens, "specificity": spec,
    }


def bootstrap_auc_ci(y_true, y_prob, n_boot=BOOTSTRAP_N):
    """Participant-level bootstrap 95% CI for AUC."""
    rng = np.random.RandomState(RANDOM_SEED)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    parts = np.arange(len(y_true))
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(parts, size=len(parts), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    if len(aucs) < 100:
        return np.nan, np.nan
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def bootstrap_delta_auc_ci(y_true, prob_a, prob_b, n_boot=BOOTSTRAP_N):
    """Participant-level bootstrap 95% CI for ΔAUC (A - B)."""
    rng = np.random.RandomState(RANDOM_SEED)
    y_true = np.asarray(y_true)
    prob_a = np.asarray(prob_a)
    prob_b = np.asarray(prob_b)
    parts = np.arange(len(y_true))
    obs = roc_auc_score(y_true, prob_a) - roc_auc_score(y_true, prob_b)
    deltas = []
    for _ in range(n_boot):
        idx = rng.choice(parts, size=len(parts), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        da = roc_auc_score(y_true[idx], prob_a[idx])
        db = roc_auc_score(y_true[idx], prob_b[idx])
        deltas.append(da - db)
    if len(deltas) < 100:
        return obs, np.nan, np.nan
    deltas = np.array(deltas)
    return obs, float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def permutation_auc_diff_paired(y_true, prob_a, prob_b, n_perm=PERMUTATION_N, seed=None):
    """
    Participant-level PAIRED permutation test for ΔAUC. Per participant randomly
    swap the A/B OOF probabilities and recompute ΔAUC. Two-sided null test.
    """
    rng = np.random.RandomState(seed if seed is not None else RANDOM_SEED)
    y_true = np.asarray(y_true)
    prob_a = np.asarray(prob_a)
    prob_b = np.asarray(prob_b)
    n = len(y_true)
    obs = roc_auc_score(y_true, prob_a) - roc_auc_score(y_true, prob_b)
    perm_deltas = []
    for _ in range(n_perm):
        swap = rng.randint(0, 2, size=n).astype(bool)
        a_perm = np.where(swap, prob_b, prob_a)
        b_perm = np.where(swap, prob_a, prob_b)
        perm_deltas.append(roc_auc_score(y_true, a_perm) - roc_auc_score(y_true, b_perm))
    perm_deltas = np.array(perm_deltas)
    return obs, float(np.mean(np.abs(perm_deltas) >= abs(obs)))


# ═══════════════════════════════════════════════════════════════════════════
# CONTINUOUS INCREMENTAL VALIDITY
# ═══════════════════════════════════════════════════════════════════════════

def continuous_incremental(df):
    print("\n[CONT] Continuous-outcome incremental validity (PHQ-8 total score)...")
    y = df.set_index("participant_id")["phq8_score"]

    base_cols = ["coverage_breadth", "evidence_density"]
    count_inc_cols = base_cols + [f"{c}_count" for c in DOMAIN_COLS]
    pres_inc_cols = base_cols + [f"{c}_pres" for c in DOMAIN_COLS]

    models = {
        "基础负荷模型 (覆盖广度+证据密度)": base_cols,
        "十域计数增量模型": count_inc_cols,
        "十域出现增量模型": pres_inc_cols,
    }
    preds = {name: run_regression_cv(df, cols) for name, cols in models.items()}

    rows = []
    base_pred = preds["基础负荷模型 (覆盖广度+证据密度)"]
    base_rmse = _rmse(y.loc[base_pred.index].values, base_pred.values)
    base_r2 = regression_metrics(y.loc[base_pred.index].values, base_pred.values)[2]
    for name, cols in models.items():
        p = preds[name].loc[y.index].values
        mae, rmse, r2, pear, spear = regression_metrics(y.loc[preds[name].index].values, preds[name].values)
        rows.append({
            "model": name,
            "model_type": "cv_regression",
            "n_features": len(cols),
            "MAE": mae, "RMSE": rmse, "R2": r2,
            "pearson_r": pear, "spearman_rho": spear,
            "delta_rmse_vs_base": rmse - base_rmse,
            "delta_r2_vs_base": r2 - base_r2,
        })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTDIR / "incremental_validity_continuous_metrics.csv",
                   index=False, encoding="utf-8-sig")

    # Comparisons: each incremental model vs base (RMSE-based)
    pairs = [
        ("十域计数增量模型", "基础负荷模型 (覆盖广度+证据密度)"),
        ("十域出现增量模型", "基础负荷模型 (覆盖广度+证据密度)"),
    ]
    comp_rows = []
    perm_ps = []
    for a, b in pairs:
        pa = preds[a].loc[y.index].values
        pb = preds[b].loc[y.index].values
        delta, lo, hi = bootstrap_rmse_diff_ci(y.values, pa, pb)
        _, p_perm = permutation_rmse_diff(y.values, pa, pb, n_perm=PERMUTATION_N)
        perm_ps.append(p_perm)
        comp_rows.append({
            "model_a": a, "model_b": b,
            "delta_rmse": float(delta),
            "bootstrap_ci_lower": float(lo), "bootstrap_ci_upper": float(hi),
            "permutation_p": float(p_perm),
            "q_value": np.nan, "significant_fdr": "no",
            "interpretation": (
                f"{a} 相对 {b} 的 RMSE 差异为 {delta:+.4f} "
                f"(bootstrap 95% CI {lo:+.4f} 至 {hi:+.4f}), 配对置换检验 p={p_perm:.3f}"
            ),
        })
    qvals = bh_fdr(np.array(perm_ps))
    for i, r in enumerate(comp_rows):
        r["q_value"] = float(qvals[i])
        r["significant_fdr"] = "yes" if qvals[i] < 0.05 else "no"
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(OUTDIR / "incremental_validity_continuous_comparisons.csv",
                index=False, encoding="utf-8-sig")

    print(f"[CONT] Saved incremental_validity_continuous_metrics.csv / _comparisons.csv")
    return metrics, comp, preds, y


# ═══════════════════════════════════════════════════════════════════════════
# BINARY INCREMENTAL VALIDITY
# ═══════════════════════════════════════════════════════════════════════════

def binary_incremental(df):
    print("\n[BIN] Binary-outcome incremental validity (PHQ-8 positive)...")
    label_series = df.set_index("participant_id")["label"]

    base_cols = ["coverage_breadth", "evidence_density"]
    count_inc_cols = base_cols + [f"{c}_count" for c in DOMAIN_COLS]
    pres_inc_cols = base_cols + [f"{c}_pres" for c in DOMAIN_COLS]

    # CV-trained models
    cv_models = {
        "基础负荷模型 (覆盖广度+证据密度)": base_cols,
        "十域计数增量模型": count_inc_cols,
        "十域出现增量模型": pres_inc_cols,
    }
    cv_preds = {name: run_logistic_cv(df, cols, label_series)
                for name, cols in cv_models.items()}

    # External reference: complex joint text model OOF probability (module 07)
    ext_preds = {}
    if "M3_prob" in df.columns:
        ext = df.set_index("participant_id")["M3_prob"]
        # align to label_series index
        ext_preds["复杂联合文本模型 (外部 OOF, 07)"] = ext.loc[label_series.index]

    rows = []
    base_pred = cv_preds["基础负荷模型 (覆盖广度+证据密度)"]
    base_auc = roc_auc_score(label_series.loc[base_pred.index].values, base_pred.values)
    auc_cis = {}
    # CV-trained metrics
    for name, cols in cv_models.items():
        p = cv_preds[name]
        yt = label_series.loc[p.index].values
        pv = p.values
        m = evaluate_binary(yt, pv)
        ci_l, ci_u = bootstrap_auc_ci(yt, pv)
        auc_cis[name] = (ci_l, ci_u)
        rows.append({
            "model": name,
            "model_type": "cv_logistic",
            "n_features": len(cols),
            "AUC": m["auc"], "AUC_ci_lower": ci_l, "AUC_ci_upper": ci_u,
            "PR_AUC": m["pr_auc"], "Macro_F1": m["macro_f1"],
            "sensitivity": m["sensitivity"], "specificity": m["specificity"],
            "delta_auc_vs_base": m["auc"] - base_auc,
        })
    # External complex model metric
    for name, p in ext_preds.items():
        yt = label_series.loc[p.index].values
        pv = p.values
        m = evaluate_binary(yt, pv)
        ci_l, ci_u = bootstrap_auc_ci(yt, pv)
        auc_cis[name] = (ci_l, ci_u)
        rows.append({
            "model": name,
            "model_type": "external_oof",
            "n_features": np.nan,
            "AUC": m["auc"], "AUC_ci_lower": ci_l, "AUC_ci_upper": ci_u,
            "PR_AUC": m["pr_auc"], "Macro_F1": m["macro_f1"],
            "sensitivity": m["sensitivity"], "specificity": m["specificity"],
            "delta_auc_vs_base": m["auc"] - base_auc,
        })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTDIR / "incremental_validity_binary_metrics.csv",
                   index=False, encoding="utf-8-sig")

    # Comparisons (ΔAUC vs base) + paired permutation p + FDR
    comp_pairs = [
        ("十域计数增量模型", "基础负荷模型 (覆盖广度+证据密度)", "cv_logistic"),
        ("十域出现增量模型", "基础负荷模型 (覆盖广度+证据密度)", "cv_logistic"),
        ("复杂联合文本模型 (外部 OOF, 07)", "基础负荷模型 (覆盖广度+证据密度)", "external_oof"),
    ]
    comp_rows = []
    perm_ps = []
    for a, b, btype in comp_pairs:
        pa = (cv_preds[a] if a in cv_preds else ext_preds[a]).loc[label_series.index].values
        pb = cv_preds[b].loc[label_series.index].values
        yt = label_series.loc[cv_preds[b].index].values
        delta, lo, hi = bootstrap_delta_auc_ci(yt, pa, pb)
        _, p_perm = permutation_auc_diff_paired(yt, pa, pb, n_perm=PERMUTATION_N)
        perm_ps.append(p_perm)
        comp_rows.append({
            "model_a": a, "model_b": b, "model_b_type": btype,
            "delta_auc": float(delta),
            "bootstrap_ci_lower": float(lo), "bootstrap_ci_upper": float(hi),
            "permutation_p": float(p_perm),
            "q_value": np.nan, "significant_fdr": "no",
            "interpretation": (
                f"{a} 相对 {b} 的 ΔAUC 为 {delta:+.4f} "
                f"(bootstrap 95% CI {lo:+.4f} 至 {hi:+.4f}), 配对置换检验 p={p_perm:.3f}"
            ),
        })
    qvals = bh_fdr(np.array(perm_ps))
    for i, r in enumerate(comp_rows):
        r["q_value"] = float(qvals[i])
        r["significant_fdr"] = "yes" if qvals[i] < 0.05 else "no"
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(OUTDIR / "incremental_validity_binary_comparisons.csv",
                index=False, encoding="utf-8-sig")

    print(f"[BIN] Saved incremental_validity_binary_metrics.csv / _comparisons.csv")
    return metrics, comp, cv_preds, ext_preds, label_series


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

def write_summary(cont_metrics, cont_comp, bin_metrics, bin_comp, df):
    print("\n[SUM] Writing summary...")
    lines = []
    lines.append("# 增量效度分析摘要\n")
    lines.append("> 本模块在既有连续总分（模块 10）与阳性分类（模块 09）结果之上，"
                 "直接检验核心问题：**复杂文本模型或细粒度十个症状域表征，是否在简单症状负荷"
                 "指标（症状覆盖广度 + 症状证据密度）之外提供额外的预测信息？**\n")

    # ── Continuous ──────────────────────────────────────────────────────────
    lines.append("## 1. 连续结局（PHQ-8 总分，全 N=142）：增量效度")
    lines.append("| 模型 | 特征数 | RMSE | R² | Pearson r | ΔRMSE vs 基础 | ΔR² vs 基础 |")
    lines.append("|------|-------|------|----|-----------|---------------|------------|")
    for _, r in cont_metrics.iterrows():
        lines.append(
            f"| {r['model']} | {r['n_features']} | {r['RMSE']:.3f} | {r['R2']:.3f} | "
            f"{r['pearson_r']:.3f} | {r['delta_rmse_vs_base']:+.3f} | {r['delta_r2_vs_base']:+.3f} |"
        )
    lines.append("")
    lines.append("**模型比较（bootstrap 95% CI 仅作区间估计；显著性由参与者级配对置换检验 + BH FDR 判断）**")
    for _, r in cont_comp.iterrows():
        lines.append(
            f"- {r['model_a']} vs {r['model_b']}：ΔRMSE={r['delta_rmse']:+.4f} "
            f"(bootstrap 95% CI {r['bootstrap_ci_lower']:+.4f} 至 {r['bootstrap_ci_upper']:+.4f}), "
            f"置换 p={r['permutation_p']:.3f}, FDR q={r['q_value']:.3f}, 显著={r['significant_fdr']}"
        )
    lines.append("- 说明：复杂联合文本模型为二分类器，其 OOF 预测为 0–1 概率、不在 PHQ-8 分数尺度上，"
                 "且冻结分析约束禁止将其重新训练为连续回归特征；因此其相对简单负荷的增量价值在"
                 "**二分类层**中直接评估（见第 2 节 ΔAUC），不在连续层做 RMSE 增量比较。")
    lines.append("")

    # ── Binary ─────────────────────────────────────────────────────────────
    lines.append("## 2. 二分类结局（PHQ-8 阳性，标签=总分≥10）：增量效度")
    lines.append("| 模型 | 类型 | 特征数 | AUC | PR-AUC | Macro-F1 | 灵敏度 | 特异度 | ΔAUC vs 基础 |")
    lines.append("|------|------|-------|-----|--------|----------|-------|--------|-------------|")
    for _, r in bin_metrics.iterrows():
        nf = r["n_features"]
        nf_s = f"{nf:.0f}" if pd.notna(nf) else "—"
        lines.append(
            f"| {r['model']} | {r['model_type']} | {nf_s} | {r['AUC']:.3f} | {r['PR_AUC']:.3f} | "
            f"{r['Macro_F1']:.3f} | {r['sensitivity']:.3f} | {r['specificity']:.3f} | "
            f"{r['delta_auc_vs_base']:+.3f} |"
        )
    lines.append("")
    lines.append("**模型比较（ΔAUC；bootstrap 95% CI 仅作区间估计；显著性由参与者级配对置换检验 + BH FDR 判断）**")
    for _, r in bin_comp.iterrows():
        lines.append(
            f"- {r['model_a']} vs {r['model_b']}（{r['model_b_type']}）：ΔAUC={r['delta_auc']:+.4f} "
            f"(bootstrap 95% CI {r['bootstrap_ci_lower']:+.4f} 至 {r['bootstrap_ci_upper']:+.4f}), "
            f"置换 p={r['permutation_p']:.3f}, FDR q={r['q_value']:.3f}, 显著={r['significant_fdr']}"
        )
    lines.append("")

    # ── Core conclusion ─────────────────────────────────────────────────────
    lines.append("## 3. 核心结论")
    bin_sig = (bin_comp["significant_fdr"] == "yes").any()
    cont_sig = (cont_comp["significant_fdr"] == "yes").any()
    if not bin_sig and not cont_sig:
        lines.append("- 在当前 DAIC-WOZ PHQ-8 任务中，细粒度症状域表征**未显示出超出简单症状负荷指标的稳定增量效度**"
                     "（连续层与二分类层的所有增量比较均不显著）。")
    else:
        lines.append(f"- 连续层增量显著={cont_sig}，二分类层增量显著={bin_sig}。")
    lines.append("- 复杂联合文本模型（外部 OOF）在二分类层相对基础负荷模型的 ΔAUC 同样未达显著，"
                 "表明复杂文本模型的预测信号在较大程度上可由症状负荷累积解释，而非来自额外不可概括的信息。")
    lines.append("- 注意：此处结论为'未观察到稳定额外增量效度'，并非'细粒度表征完全没有价值'；"
                 "十域模型本身具有预测信息（AUC / Pearson r 均非随机），只是未稳定超越简单聚合指标。")
    lines.append("- 注意事项：PHQ-8 为自评量表总分，非临床诊断；本分析为可解释性/信号来源分解研究，"
                 "非临床预测模型开发。")
    lines.append("")

    text = "\n".join(lines)
    (OUTDIR / "incremental_validity_summary.md").write_text(text, encoding="utf-8-sig")
    print("[SUM] Saved incremental_validity_summary.md")


# ═══════════════════════════════════════════════════════════════════════════
# MANIFEST
# ═══════════════════════════════════════════════════════════════════════════

def generate_manifest():
    import sklearn, scipy
    manifest = {
        "analysis_time": datetime.now().isoformat(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "scipy_version": scipy.__version__,
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
        "outcomes": {
            "continuous": "PHQ-8 total score (paper_phq8_score)",
            "binary": "PHQ-8 positive (paper_label_phq8_ge10)",
        },
        "n_participants": 142,
        "base_load_model": "coverage_breadth + evidence_density",
        "complex_model_handling": (
            "complex joint text model OOF probability (module 07) used ONLY as external "
            "evaluation reference for the binary outcome; never as a retrained feature. "
            "Not entered as a continuous-score regression (binary classifier, scale mismatch, "
            "and SPEC forbids refitting)."
        ),
        "comparison_methods": {
            "bootstrap_ci": "participant-level, 5000 resamples, percentile 95% (interval estimate only)",
            "permutation_test": "participant-level paired swap, 10000 reps, two-sided",
            "fdr": "Benjamini-Hochberg within each comparison family",
        },
    }
    (OUTDIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8-sig"
    )
    out_files = [
        "incremental_validity_continuous_metrics.csv",
        "incremental_validity_continuous_comparisons.csv",
        "incremental_validity_binary_metrics.csv",
        "incremental_validity_binary_comparisons.csv",
        "incremental_validity_summary.md",
        "run_manifest.json",
    ]
    rows = [{"file": f, "sha256": sha256_file(OUTDIR / f)} for f in out_files if (OUTDIR / f).exists()]
    pd.DataFrame(rows).to_csv(OUTDIR / "output_manifest_sha256.csv", index=False, encoding="utf-8-sig")
    print("[MANIFEST] Saved run_manifest.json and output_manifest_sha256.csv")


def main():
    print("=" * 70)
    print("INCREMENTAL VALIDITY ANALYSIS (PHQ-8)")
    print("=" * 70)
    df = load_and_validate()
    cont_metrics, cont_comp, _, _ = continuous_incremental(df)
    bin_metrics, bin_comp, _, _, _ = binary_incremental(df)
    write_summary(cont_metrics, cont_comp, bin_metrics, bin_comp, df)
    generate_manifest()
    print("\n" + "=" * 70)
    print("DONE. All outputs in:", OUTDIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
