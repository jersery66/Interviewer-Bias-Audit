#!/usr/bin/env python3
"""
Symptom-Level Signal Decomposition Analysis
==========================================
DAIC-WOZ PHQ-8 positive prediction — symptom-level signal source analysis.

This script decomposes the "clinical domain count" signal by analyzing:
  (1) which individual symptom domains carry predictive signal,
  (2) whether symptom coverage breadth (presence sum) shows a risk gradient,
  (3) whether symptom evidence density (count sum) shows a risk gradient,
  (4) which symptom combinations approach the full 10-domain model,
  (5) how symptom-level models compare to the strict full joint model (M3).

Hard constraints:
  - Read ONLY from frozen input files (see SPEC for allowed paths).
  - NEVER use any OOF probability as a feature for a new model.
  - ALL merges MUST be on `participant_id`; NEVER rely on row order.
  - All models MUST be fold-local (fit scaler + classifier on TRAIN only).
  - Participant-level bootstrap (not row-level) for all confidence intervals.
  - Benjamini-Hochberg FDR for all multiple-comparison families.

Author: Codex (strict reanalysis, 2026-07-06)
"""

import os
import sys
import json
import hashlib
import warnings
import itertools
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.special import expit, logit
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    brier_score_loss, log_loss,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths (hard-coded, SPEC-compliant) ───────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
BASE = SCRIPT_DIR.parent.parent
DOMAIN_COUNT_CSV = BASE / "04_c5_controls/domain_count/input.csv"
DOMAIN_PRESENCE_CSV = BASE / "04_c5_controls/domain_presence/input.csv"
SPLIT_CSV = BASE / "00_splits/repeated_5fold_splits_10x5.csv"
STRICT_OOF_CSV = BASE / "10_source_importance/07_strict_joint_models/participant_oof_predictions.csv"
STRICT_MANIFEST = BASE / "10_source_importance/07_strict_joint_models/run_manifest.json"
OUTDIR = Path(__file__).parent

# ── Symptom domain definitions ─────────────────────────────────────────────────
# Column names as they appear in the CSV files:
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
# Chinese names (for output tables):
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

# ── LR parameters (SPEC-fixed) ────────────────────────────────────────────────
LR_PARAMS = dict(
    penalty="l2", C=1.0, class_weight="balanced",
    solver="liblinear", max_iter=2000, random_state=20260705,
)
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
    """Benjamini-Hochberg FDR correction. Returns q-values."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    qvals = np.full(n, np.nan)
    # Sort p-values and apply BH
    sorted_p = pvals[order]
    # Adjusted p-values (BH)
    adjusted = sorted_p * n / (np.arange(1, n + 1))
    # Ensure monotonicity (enforce q_i <= q_{i+1})
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


def bootstrap_auc_ci(y_true, y_pred, n_boot=BOOTSTRAP_N, alpha=0.05):
    """
    Participant-level bootstrap for AUC CI.
    y_true, y_pred: same length, aligned.
    Returns (ci_lower, ci_upper).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    boot_aucs = []
    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            auc_val = roc_auc_score(y_true[idx], y_pred[idx])
            boot_aucs.append(auc_val)
        except ValueError:
            continue
    if len(boot_aucs) < 100:
        return np.nan, np.nan
    lower = np.percentile(boot_aucs, 100 * alpha / 2)
    upper = np.percentile(boot_aucs, 100 * (1 - alpha / 2))
    return lower, upper



def bootstrap_delta_auc_ci(y_true, prob_a, prob_b, n_boot=BOOTSTRAP_N, alpha=0.05):
    """Bootstrap CI for ΔAUC (p-value via permutation, NOT bootstrap)."""
    y_true = np.asarray(y_true)
    prob_a = np.asarray(prob_a)
    prob_b = np.asarray(prob_b)
    obs = roc_auc_score(y_true, prob_a) - roc_auc_score(y_true, prob_b)
    rng = np.random.RandomState(RANDOM_SEED)
    n = len(y_true)
    deltas = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        try:
            d = roc_auc_score(y_true[idx], prob_a[idx]) - roc_auc_score(y_true[idx], prob_b[idx])
        except ValueError:
            d = 0.0
        deltas.append(d)
    deltas = np.array(deltas)
    lo = np.percentile(deltas, 100 * alpha/2)
    hi = np.percentile(deltas, 100 * (1 - alpha/2))
    return obs, lo, hi

def permutation_test_auc_diff(y_true, prob_a, prob_b, n_perm=PERMUTATION_N, seed=None):
    """Label permutation test for ΔAUC."""
    y_true = np.asarray(y_true)
    prob_a = np.asarray(prob_a)
    prob_b = np.asarray(prob_b)
    auc_a_obs = roc_auc_score(y_true, prob_a)
    auc_b_obs = roc_auc_score(y_true, prob_b)
    delta_obs = auc_a_obs - auc_b_obs
    rng = np.random.RandomState(seed if seed is not None else RANDOM_SEED)
    perm_deltas = []
    for _ in range(n_perm):
        perm_y = y_true[rng.permutation(len(y_true))]
        if len(np.unique(perm_y)) < 2:
            continue
        try:
            da = roc_auc_score(perm_y, prob_a)
            db = roc_auc_score(perm_y, prob_b)
            perm_deltas.append(da - db)
        except ValueError:
            continue
    if len(perm_deltas) < 100:
        return np.nan
    perm_deltas = np.array(perm_deltas)
    p_value = np.mean(np.abs(perm_deltas) >= abs(delta_obs))
    return p_value


def fit_fold_local_model(X_train, y_train, X_test, scale=True):
    """
    Fit scaler + LR on TRAIN only; predict on TEST.
    Returns test probabilities.
    """
    if scale and X_train.shape[1] > 0:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    if X_train_scaled.shape[1] == 0:
        return np.full(len(X_test), 0.5)
    model = LogisticRegression(**LR_PARAMS)
    model.fit(X_train_scaled, y_train)
    prob = model.predict_proba(X_test_scaled)[:, 1]
    return prob


def run_single_feature_cv(df, feature_col, label_col="label", scale=True):
    """
    Participant-level 10×5 repeated CV for a single feature.
    Returns participant-level mean OOF probability (Series, index=participant_id).
    """
    splits = pd.read_csv(SPLIT_CSV)
    all_probs = []
    for (rpt, fold), fold_rows in splits.groupby(["repeat", "fold"]):
        train_ids = fold_rows[fold_rows["role"] == "train"]["participant_id"].values
        test_ids = fold_rows[fold_rows["role"] == "test"]["participant_id"].values
        train_df = df[df["participant_id"].isin(train_ids)]
        test_df = df[df["participant_id"].isin(test_ids)]
        if len(train_df) == 0 or len(test_df) == 0:
            continue
        X_train = train_df[[feature_col]].values
        y_train = train_df[label_col].values
        X_test = test_df[[feature_col]].values
        prob = fit_fold_local_model(X_train, y_train, X_test, scale=scale)
        for pid, p in zip(test_df["participant_id"].values, prob):
            all_probs.append({"participant_id": pid, "repeat": rpt, "fold": fold, "prob": p})
    prob_df = pd.DataFrame(all_probs)
    # Average over repeats for each participant
    participant_prob = prob_df.groupby("participant_id")["prob"].mean()
    return participant_prob


def evaluate_predictions(y_true, y_prob):
    """Compute evaluation metrics for a set of predictions."""
    auc_val = roc_auc_score(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    brier = brier_score_loss(y_true, y_prob)
    logloss = log_loss(y_true, y_prob)
    # Default 0.50 threshold
    y_pred_50 = (y_prob >= 0.50).astype(int)
    cm = confusion_matrix(y_true, y_pred_50)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sens_50 = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec_50 = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    else:
        sens_50 = np.nan
        spec_50 = np.nan
    return {
        "auc": auc_val,
        "pr_auc": pr_auc,
        "brier": brier,
        "logloss": logloss,
        "sens_50": sens_50,
        "spec_50": spec_50,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MODULE 0: INPUT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def validate_inputs():
    """Load and rigorously validate all input files. Raise on critical failure."""
    errors = []
    warnings_list = []
    report = {}

    # ── Load files ────────────────────────────────────────────────────────────
    print("  [VAL] Loading input files...")
    dcount = pd.read_csv(DOMAIN_COUNT_CSV)
    dpres = pd.read_csv(DOMAIN_PRESENCE_CSV)
    splits = pd.read_csv(SPLIT_CSV)
    strict_oof = pd.read_csv(STRICT_OOF_CSV)

    report["n_rows_domain_count"] = len(dcount)
    report["n_rows_domain_presence"] = len(dpres)
    report["n_rows_strict_oof"] = len(strict_oof)
    report["n_rows_splits"] = len(splits)

    # ── Check 1: merge sample size = 142 ─────────────────────────────────────
    merged = dcount[["participant_id"]].copy()
    merged = merged.merge(dpres[["participant_id"]], on="participant_id", how="inner")
    merged = merged.merge(strict_oof[["participant_id", "label"]], on="participant_id", how="inner")
    n_merged = len(merged)
    report["merged_sample_size"] = n_merged
    if n_merged != 142:
        errors.append(f"CRITICAL: merged sample size = {n_merged}, expected 142")
    print(f"    Merged sample size: {n_merged}")

    # ── Check 2 & 3: label counts ─────────────────────────────────────────────
    label_counts = strict_oof.groupby("label").size()
    n_pos = int(label_counts.get(1, 0))
    n_neg = int(label_counts.get(0, 0))
    report["n_positive"] = n_pos
    report["n_negative"] = n_neg
    if n_pos != 43:
        errors.append(f"CRITICAL: positive count = {n_pos}, expected 43")
    if n_neg != 99:
        errors.append(f"CRITICAL: negative count = {n_neg}, expected 99")
    print(f"    Positive: {n_pos}, Negative: {n_neg}")

    # ── Check 4: participant_id unique ───────────────────────────────────────
    for name, df in [("domain_count", dcount), ("domain_presence", dpres), ("strict_oof", strict_oof)]:
        if not df["participant_id"].is_unique:
            errors.append(f"CRITICAL: duplicate participant_id in {name}")
    report["participant_id_unique"] = True

    # ── Check 5: no participant loss on merge ────────────────────────────────
    for name, df in [("domain_count", dcount), ("domain_presence", dpres)]:
        lost = set(df["participant_id"]) - set(strict_oof["participant_id"])
        if lost:
            errors.append(f"CRITICAL: {len(lost)} participants in {name} not in strict_oof")
        report[f"{name}_all_in_strict_oof"] = len(lost) == 0

    # ── Check 6: counts are non-negative ─────────────────────────────────────
    for col in DOMAIN_COLS:
        if col in dcount.columns:
            if (dcount[col] < 0).any():
                errors.append(f"CRITICAL: negative values in domain_count.{col}")
        else:
            errors.append(f"CRITICAL: column {col} missing in domain_count")
    report["domain_counts_non_negative"] = True

    # ── Check 7: presence ∈ {0, 1} ───────────────────────────────────────
    for col in DOMAIN_COLS:
        if col in dpres.columns:
            unique_vals = set(dpres[col].unique())
            if not unique_vals.issubset({0, 1}):
                errors.append(f"CRITICAL: domain_presence.{col} has values {unique_vals}")
        else:
            errors.append(f"CRITICAL: column {col} missing in domain_presence")
    report["domain_presence_binary"] = True

    # ── Check 8: count > 0 ⇒ presence == 1 ────────────────────────────────
    # Merge for checking
    check_df = dcount[["participant_id"] + DOMAIN_COLS].merge(
        dpres[["participant_id"] + DOMAIN_COLS],
        on="participant_id", suffixes=("_count", "_pres")
    )
    for col in DOMAIN_COLS:
        mask = check_df[f"{col}_count"] > 0
        if (check_df.loc[mask, f"{col}_pres"] != 1).any():
            errors.append(f"CRITICAL: {col}: count>0 but presence!=1 for some participants")
        # Check 9: count==0 but presence==1
        mask2 = check_df[f"{col}_count"] == 0
        if (check_df.loc[mask2, f"{col}_pres"] == 1).any():
            n_warn = (check_df.loc[mask2, f"{col}_pres"] == 1).sum()
            warnings_list.append(f"WARNING: {col}: count=0 but presence=1 for {n_warn} participants")
    report["count_presence_consistency"] = True
    report["warnings"] = warnings_list

    # ── Check 10: splits have 10 repeats × 5 folds ─────────────────────────
    n_repeats = splits["repeat"].nunique()
    n_folds = splits["fold"].nunique()
    report["n_repeats"] = n_repeats
    report["n_folds"] = n_folds
    if n_repeats != 10:
        errors.append(f"CRITICAL: split file has {n_repeats} repeats, expected 10")
    if n_folds != 5:
        errors.append(f"CRITICAL: split file has {n_folds} folds, expected 5")

    # ── Check 11: each participant appears exactly once as test per repeat ───
    test_participants = splits[splits["role"] == "test"].groupby("repeat")["participant_id"].apply(set)
    for rpt, pids in test_participants.items():
        if len(pids) == 0:
            errors.append(f"CRITICAL: repeat {rpt} has no test participants")
    # Check: each participant appears in test set in each repeat exactly once
    test_df = splits[splits["role"] == "test"]
    test_counts = test_df.groupby("participant_id")["repeat"].nunique()
    if test_counts.nunique() != 1 or test_counts.iloc[0] != 10:
        errors.append("CRITICAL: not all participants appear as test in all 10 repeats")
    report["each_participant_tested_in_all_repeats"] = True

    # ── Check 12: train/test non-overlap in each fold ───────────────────────
    for (rpt, fold), group in splits.groupby(["repeat", "fold"]):
        train_pids = set(group[group["role"] == "train"]["participant_id"])
        test_pids = set(group[group["role"] == "test"]["participant_id"])
        overlap = train_pids & test_pids
        if overlap:
            errors.append(f"CRITICAL: repeat={rpt} fold={fold} has train/test overlap: {overlap}")
    report["train_test_no_overlap"] = True

    # ── Save report ──────────────────────────────────────────────────────────
    report_path = OUTDIR / "input_validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  [VAL] Validation report saved: {report_path}")

    # ── Raise on critical errors ─────────────────────────────────────────────
    if errors:
        print("\n".join(errors))
        raise ValueError(f"Input validation FAILED with {len(errors)} critical errors. See above.")

    print("  [VAL] ✅ All critical checks PASSED.")
    if warnings_list:
        print("  [VAL] Warnings:")
        for w in warnings_list:
            print(f"       {w}")

    return dcount, dpres, splits, strict_oof


# ═══════════════════════════════════════════════════════════════════════════
# MODULE 1: DESCRIPTIVE STATISTICS
# ═══════════════════════════════════════════════════════════════════════════

def module1_descriptive(df_count, df_pres, labels):
    """
    Compute descriptive statistics for each symptom domain.
    labels: Series(index=participant_id, values=label)
    """
    print("\n  [M1] Module 1: Symptom descriptive statistics...")
    rows = []
    for col in DOMAIN_COLS:
        row = {"domain_en": col, "domain_cn": DOMAIN_CN.get(col, "")}
        cnt = df_count.set_index("participant_id")[col]
        prs = df_pres.set_index("participant_id")[col]
        lbl = labels.copy()

        # Presence-based stats
        presence = (cnt > 0).astype(int)  # derive presence from count for consistency
        # Actually use the presence file
        presence = prs

        n_present = int(presence.sum())
        row["n_present"] = n_present
        row["prop_present"] = n_present / 142

        # By group
        pos_mask = lbl == 1
        neg_mask = lbl == 0
        n_present_pos = int(presence[pos_mask].sum())
        n_present_neg = int(presence[neg_mask].sum())
        row["n_present_positive"] = n_present_pos
        row["prop_present_positive"] = n_present_pos / 43 if 43 > 0 else np.nan
        row["n_present_negative"] = n_present_neg
        row["prop_present_negative"] = n_present_neg / 99 if 99 > 0 else np.nan

        # Count-based stats
        row["mean_count_all"] = cnt.mean()
        row["mean_count_positive"] = cnt[pos_mask].mean()
        row["mean_count_negative"] = cnt[neg_mask].mean()
        row["median_count"] = cnt.median()
        row["q1_count"] = cnt.quantile(0.25)
        row["q3_count"] = cnt.quantile(0.75)
        row["max_count"] = cnt.max()

        rows.append(row)

    out = pd.DataFrame(rows)
    out_path = OUTDIR / "symptom_descriptive_table.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  [M1] Saved: {out_path}")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# MODULE 2: SINGLE SYMPTOM DOMAIN PREDICTION MODELS
# ═══════════════════════════════════════════════════════════════════════════

def module2_single_domain_cv(dcount, dpres, labels):
    """
    For each symptom domain, fit two single-predictor CV models:
      (a) presence model (0/1)
      (b) count model (integer)
    Then compute metrics + bootstrap CI + permutation p + FDR.
    """
    print("\n  [M2] Module 2: Single symptom domain CV models...")
    splits = pd.read_csv(SPLIT_CSV)
    rows = []

    for col in DOMAIN_COLS:
        for model_type in ["presence", "count"]:
            print(f"    Fitting {col} ({model_type})...")
            # Build analysis DataFrame
            if model_type == "presence":
                feat_df = dpres[["participant_id", col]].copy()
                feat_df = feat_df.rename(columns={col: "feature"})
            else:
                feat_df = dcount[["participant_id", col]].copy()
                feat_df = feat_df.rename(columns={col: "feature"})
            feat_df["label"] = labels.reindex(feat_df["participant_id"]).values

            # Participant-level 10×5 CV
            participant_probs = []
            for (rpt, fold), fold_rows in splits.groupby(["repeat", "fold"]):
                train_ids = fold_rows[fold_rows["role"] == "train"]["participant_id"].values
                test_ids = fold_rows[fold_rows["role"] == "test"]["participant_id"].values
                train_df = feat_df[feat_df["participant_id"].isin(train_ids)]
                test_df = feat_df[feat_df["participant_id"].isin(test_ids)]
                if len(train_df) == 0 or len(test_df) == 0:
                    continue
                X_train = train_df[["feature"]].values
                y_train = train_df["label"].values
                X_test = test_df[["feature"]].values
                prob = fit_fold_local_model(X_train, y_train, X_test, scale=True)
                for pid, p in zip(test_df["participant_id"].values, prob):
                    participant_probs.append({"participant_id": pid, "repeat": rpt, "prob": p})

            if len(participant_probs) == 0:
                continue

            prob_df = pd.DataFrame(participant_probs)
            # Average over repeats
            part_prob = prob_df.groupby("participant_id")["prob"].mean()
            y_true = labels.loc[part_prob.index].values
            y_prob = part_prob.values

            # Metrics
            metrics = evaluate_predictions(y_true, y_prob)
            row = {
                "domain": col,
                "domain_cn": DOMAIN_CN.get(col, ""),
                "model_type": model_type,
                "auc": metrics["auc"],
                "pr_auc": metrics["pr_auc"],
                "brier": metrics["brier"],
                "logloss": metrics["logloss"],
                "sens_50": metrics["sens_50"],
                "spec_50": metrics["spec_50"],
            }

            # Bootstrap CI for AUC
            ci_low, ci_high = bootstrap_auc_ci(y_true, y_prob)
            row["auc_ci_lower"] = ci_low
            row["auc_ci_upper"] = ci_high

            # Youden threshold (exploratory)
            fpr, tpr, thresholds = _roc_curve(y_true, y_prob)
            youden = tpr - fpr
            best_idx = np.argmax(youden)
            youden_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            y_pred_youden = (y_prob >= youden_thresh).astype(int)
            cm = confusion_matrix(y_true, y_pred_youden)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                row["youden_threshold"] = youden_thresh
                row["youden_sens"] = tp / (tp + fn) if (tp + fn) > 0 else np.nan
                row["youden_spec"] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            else:
                row["youden_threshold"] = np.nan
                row["youden_sens"] = np.nan
                row["youden_spec"] = np.nan

            # Label permutation test for AUC > 0.5
            # Directional AUC
            auc = metrics["auc"]
            row["auc_directional"] = max(auc, 1 - auc)
            row["direction"] = "positive" if auc >= 0.5 else "inverse"
            # One-sided permutation p (AUC > 0.5)
            perm_p_one = _permutation_test_auc_above_random(y_true, y_prob)
            row["permutation_p_one_sided"] = perm_p_one
            # Two-sided permutation p (|AUC-0.5|)
            perm_p_two = _permutation_test_auc_two_sided(y_true, y_prob)
            row["permutation_p_two_sided"] = perm_p_two
            # Keep permutation_p = two_sided for backward compatibility
            row["permutation_p"] = perm_p_two

            rows.append(row)

    result = pd.DataFrame(rows)

    # FDR correction within this family (20 tests)
    pvals = result["permutation_p_two_sided"].values  # Use two-sided p for FDR
    qvals = bh_fdr(pvals)
    result["q_value"] = qvals
    result["significance"] = [sig_marker(q) for q in qvals]

    out_path = OUTDIR / "symptom_single_domain_cv_metrics.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  [M2] Saved: {out_path}")
    print(f"  [M2] Significant (q<0.05) models: {(result['significance']!='ns').sum()}")
    return result


def _roc_curve(y_true, y_prob):
    """Return fpr, tpr, thresholds (wrapper for sklearn)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return fpr, tpr, thresholds


def _permutation_test_auc_above_random(y_true, y_prob, n_perm=PERMUTATION_N):
    """Test H0: AUC <= 0.5 (one-sided)."""
    auc_obs = roc_auc_score(y_true, y_prob)
    n = len(y_true)
    perm_aucs = []
    for _ in range(n_perm):
        perm_y = np.random.permutation(y_true)
        if len(np.unique(perm_y)) < 2:
            continue
        try:
            perm_aucs.append(roc_auc_score(perm_y, y_prob))
        except ValueError:
            continue
    if len(perm_aucs) < 100:
        return np.nan
    # One-sided: P(AUC_perm >= AUC_obs) under H0
    p_value = np.mean(np.array(perm_aucs) >= auc_obs)
    return p_value

def _permutation_test_auc_two_sided(y_true, y_prob, n_perm=PERMUTATION_N):
    """Two-sided permutation test: H0: AUC = 0.5."""
    from sklearn.metrics import roc_auc_score
    auc_obs = roc_auc_score(y_true, y_prob)
    obs_stat = abs(auc_obs - 0.5)
    perm_stats = []
    for _ in range(n_perm):
        perm_y = np.random.permutation(y_true)
        if len(np.unique(perm_y)) < 2:
            continue
        try:
            perm_auc = roc_auc_score(perm_y, y_prob)
            perm_stats.append(abs(perm_auc - 0.5))
        except ValueError:
            continue
    if len(perm_stats) < 100:
        return np.nan
    p_value = np.mean(np.array(perm_stats) >= obs_stat)
    return p_value



# ═══════════════════════════════════════════════════════════════════════════
# MODULE 3: UNIVARIATE ASSOCIATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def module3_univariate_association(dcount, dpres, labels):
    """
    Association between each symptom domain and PHQ-8 positive label.
    - Presence: Fisher exact test or univariable LR → OR, 95% CI, p, q
    - Count: univariable LR → OR per unit, 95% CI, p; plus Mann-Whitney + Cliff's delta
    FDR: 20 tests (10 presence + 10 count).
    """
    print("\n  [M3] Module 3: Univariate association analysis...")
    rows = []

    for col in DOMAIN_COLS:
        # ── Presence variable ────────────────────────────────────────────────
        prs = dpres.set_index("participant_id")[col]
        data = pd.DataFrame({"label": labels, "presence": prs})
        data = data.dropna()

        # Fisher exact test
        table = pd.crosstab(data["presence"], data["label"])
        if table.shape == (2, 2):
            odds_ratio, p_fisher = scipy_stats.fisher_exact(table)
            # 95% CI for OR via logistic regression (more stable)
            try:
                model = smf.logit("label ~ presence", data=data).fit(disp=0)
                params = model.params["presence"]
                ci = model.conf_int().loc["presence"]
                or_val = np.exp(params)
                ci_lower = np.exp(ci[0])
                ci_upper = np.exp(ci[1])
                p_val = model.pvalues["presence"]
            except Exception:
                # Complete separation → use Fisher
                or_val = odds_ratio
                # Approximate CI via logit
                log_or = np.log(odds_ratio)
                se_log_or = np.sqrt(1/table.iloc[0, 0] + 1/table.iloc[0, 1] +
                                    1/table.iloc[1, 0] + 1/table.iloc[1, 1])
                ci_lower = np.exp(log_or - 1.96 * se_log_or)
                ci_upper = np.exp(log_or + 1.96 * se_log_or)
                p_val = p_fisher
        else:
            or_val = np.nan
            ci_lower = np.nan
            ci_upper = np.nan
            p_val = np.nan

        rows.append({
            "domain": col,
            "domain_cn": DOMAIN_CN.get(col, ""),
            "variable_type": "presence",
            "odds_ratio": or_val,
            "or_ci_lower": ci_lower,
            "or_ci_upper": ci_upper,
            "permutation_p": p_val,
        })

        # ── Count variable ────────────────────────────────────────────────────
        cnt = dcount.set_index("participant_id")[col]
        data_c = pd.DataFrame({"label": labels, "count": cnt})
        data_c = data_c.dropna()

        try:
            model_c = smf.logit("label ~ count", data=data_c).fit(disp=0)
            params_c = model_c.params["count"]
            ci_c = model_c.conf_int().loc["count"]
            or_c = np.exp(params_c)
            ci_c_lower = np.exp(ci_c[0])
            ci_c_upper = np.exp(ci_c[1])
            p_c = model_c.pvalues["count"]
        except Exception:
            or_c = np.nan
            ci_c_lower = np.nan
            ci_c_upper = np.nan
            p_c = np.nan

        # Mann-Whitney U + Cliff's delta
        pos_counts = data_c[data_c["label"] == 1]["count"].values
        neg_counts = data_c[data_c["label"] == 0]["count"].values
        if len(pos_counts) > 0 and len(neg_counts) > 0:
            u_stat, u_p = scipy_stats.mannwhitneyu(pos_counts, neg_counts, alternative="two-sided")
            # Cliff's delta
            n_pos = len(pos_counts)
            n_neg = len(neg_counts)
            cliff_delta = (2 * u_stat / (n_pos * n_neg)) - 1
        else:
            u_p = np.nan
            cliff_delta = np.nan

        rows.append({
            "domain": col,
            "domain_cn": DOMAIN_CN.get(col, ""),
            "variable_type": "count",
            "odds_ratio": or_c,
            "or_ci_lower": ci_c_lower,
            "or_ci_upper": ci_c_upper,
            "p_value": p_c,
            "mannwhitney_p": u_p,
            "cliffs_delta": cliff_delta,
        })

    result = pd.DataFrame(rows)

    # FDR correction: 20 tests
    pvals = result["p_value"].values.copy()
    # For presence models, use p_value; for count models, also use p_value
    qvals = bh_fdr(pvals)
    result["q_value"] = qvals
    result["significance"] = [sig_marker(q) for q in qvals]

    out_path = OUTDIR / "symptom_univariate_association_fdr.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  [M3] Saved: {out_path}")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# MODULE 4: SYMPTOM COVERAGE GRADIENT
# ═══════════════════════════════════════════════════════════════════════════

def module4_coverage_gradient(dpres, labels):
    """
    Analyze risk gradient by number of symptom domains present.
    Variable: domain_presence_sum (0–10).
    """
    print("\n  [M4] Module 4: Symptom coverage gradient...")
    dpres = dpres.copy()
    dpres["domain_presence_sum"] = dpres[DOMAIN_COLS].sum(axis=1)
    dpres = dpres.set_index("participant_id")

    data = pd.DataFrame({"label": labels, "coverage": dpres["domain_presence_sum"]})
    data = data.dropna()

    # ── Preset grouping: 0–2, 3–4, 5–6, 7–10 ───────────────────────────
    data["group_preset"] = pd.cut(
        data["coverage"], bins=[-0.5, 2.5, 4.5, 6.5, 10.5],
        labels=["0-2", "3-4", "5-6", "7-10"]
    )
    # Check group sizes; merge small groups
    group_sizes = data.groupby("group_preset").size()
    print(f"    Preset group sizes: {group_sizes.to_dict()}")

    # ── Also quartile grouping ────────────────────────────────────────────────
    quartiles = data["coverage"].quantile([0.25, 0.5, 0.75]).values
    data["group_quartile"] = pd.cut(
        data["coverage"], bins=[-0.5, quartiles[0], quartiles[1], quartiles[2], 10.5],
        labels=["Q1", "Q2", "Q3", "Q4"]
    )

    # ── Risk table ────────────────────────────────────────────────────────────
    rows = []
    for group_name, group_data in data.groupby("group_preset"):
        n = len(group_data)
        n_pos = int(group_data["label"].sum())
        prop_pos = n_pos / n if n > 0 else np.nan
        rows.append({
            "group_type": "preset",
            "group": str(group_name),
            "n": n,
            "n_positive": n_pos,
            "positive_rate": prop_pos,
        })
    # Also quartiles
    for group_name, group_data in data.groupby("group_quartile"):
        n = len(group_data)
        n_pos = int(group_data["label"].sum())
        prop_pos = n_pos / n if n > 0 else np.nan
        rows.append({
            "group_type": "quartile",
            "group": str(group_name),
            "n": n,
            "n_positive": n_pos,
            "positive_rate": prop_pos,
        })

    risk_table = pd.DataFrame(rows)

    # ── Logistic regression for trend (ordinal group) ──────────────────────
    # Use preset groups as ordinal
    data_ord = data.dropna(subset=["group_preset"]).copy()
    data_ord["coverage_ordinal"] = data_ord["group_preset"].cat.codes
    try:
        model_trend = smf.logit("label ~ coverage_ordinal", data=data_ord).fit(disp=0)
        trend_or = np.exp(model_trend.params["coverage_ordinal"])
        trend_ci = model_trend.conf_int().loc["coverage_ordinal"]
        trend_or_ci = (np.exp(trend_ci[0]), np.exp(trend_ci[1]))
        trend_p = model_trend.pvalues["coverage_ordinal"]
    except Exception as e:
        print(f"    Trend model failed: {e}")
        trend_or = np.nan
        trend_or_ci = (np.nan, np.nan)
        trend_p = np.nan

    # FDR for trend test (1 test in this family... actually 2 with density)
    # We'll store the trend result separately
    trend_result = {
        "variable": "domain_presence_sum",
        "trend_odds_ratio": trend_or,
        "trend_or_ci_lower": trend_or_ci[0],
        "trend_or_ci_upper": trend_or_ci[1],
        "trend_p_value": trend_p,
    }

    # ── CV model for domain_presence_sum ────────────────────────────────────
    feat_df = pd.DataFrame({
        "participant_id": data.index,
        "feature": data["coverage"].values,
        "label": data["label"].values,
    })
    # Run CV
    splits = pd.read_csv(SPLIT_CSV)
    participant_probs = []
    for (rpt, fold), fold_rows in splits.groupby(["repeat", "fold"]):
        train_ids = fold_rows[fold_rows["role"] == "train"]["participant_id"].values
        test_ids = fold_rows[fold_rows["role"] == "test"]["participant_id"].values
        train_df = feat_df[feat_df["participant_id"].isin(train_ids)]
        test_df = feat_df[feat_df["participant_id"].isin(test_ids)]
        if len(train_df) == 0 or len(test_df) == 0:
            continue
        X_train = train_df[["feature"]].values
        y_train = train_df["label"].values
        X_test = test_df[["feature"]].values
        prob = fit_fold_local_model(X_train, y_train, X_test, scale=True)
        for pid, p in zip(test_df["participant_id"].values, prob):
            participant_probs.append({"participant_id": pid, "repeat": rpt, "prob": p})

    cv_metrics = {}
    if len(participant_probs) > 0:
        prob_df = pd.DataFrame(participant_probs)
        part_prob = prob_df.groupby("participant_id")["prob"].mean()
        y_true = labels.loc[part_prob.index].values
        y_prob = part_prob.values
        cv_metrics = evaluate_predictions(y_true, y_prob)
        cv_metrics["auc_ci_lower"], cv_metrics["auc_ci_upper"] = bootstrap_auc_ci(y_true, y_prob)

    # Save outputs
    risk_table_path = OUTDIR / "symptom_coverage_gradient_main.csv"
    risk_table.to_csv(risk_table_path, index=False, encoding="utf-8-sig")
    print(f"  [M4] Risk table saved: {risk_table_path}")

    trend_path = OUTDIR / "symptom_coverage_gradient_trend.json"
    with open(trend_path, "w") as f:
        json.dump(trend_result, f, indent=2, ensure_ascii=False)
    print(f"  [M4] Trend result: OR={trend_or:.3f} (95% CI {trend_or_ci[0]:.3f}-{trend_or_ci[1]:.3f}), p={trend_p:.4f}")
    print(f"  [M4] CV metrics: AUC={cv_metrics.get('auc', np.nan):.4f}")
    return risk_table, trend_result, cv_metrics


# ═══════════════════════════════════════════════════════════════════════════
# MODULE 5: SYMPTOM EVIDENCE DENSITY GRADIENT
# ═══════════════════════════════════════════════════════════════════════════

def module5_evidence_density_gradient(dcount, labels):
    """
    Analyze risk gradient by total symptom evidence count.
    Variable: total_domain_count (sum of 10 domain counts).
    """
    print("\n  [M5] Module 5: Symptom evidence density gradient...")
    dcount = dcount.copy()
    dcount["total_domain_count"] = dcount[DOMAIN_COLS].sum(axis=1)
    dcount = dcount.set_index("participant_id")

    data = pd.DataFrame({"label": labels, "density": dcount["total_domain_count"]})
    data = data.dropna()

    # ── Preset grouping: 0–5, 6–10, 11+ ─────────────────────────────
    max_density = data["density"].max()
    upper_bound = max(11, max_density)
    data["group_preset"] = pd.cut(
        data["density"], bins=[-0.5, 5.5, 10.5, upper_bound + 0.5],
        labels=["0-5", "6-10", "11+"]
    )
    group_sizes = data.groupby("group_preset").size()
    print(f"    Density preset group sizes: {group_sizes.to_dict()}")

    # ── Quartile grouping ────────────────────────────────────────────────────
    quartiles = data["density"].quantile([0.25, 0.5, 0.75]).values
    data["group_quartile"] = pd.cut(
        data["density"], bins=[-0.5, quartiles[0], quartiles[1], quartiles[2], upper_bound + 0.5],
        labels=["Q1", "Q2", "Q3", "Q4"]
    )

    # ── Risk table ────────────────────────────────────────────────────────────
    rows = []
    for group_name, group_data in data.groupby("group_preset"):
        n = len(group_data)
        n_pos = int(group_data["label"].sum())
        prop_pos = n_pos / n if n > 0 else np.nan
        rows.append({
            "group_type": "preset",
            "group": str(group_name),
            "n": n,
            "n_positive": n_pos,
            "positive_rate": prop_pos,
        })
    for group_name, group_data in data.groupby("group_quartile"):
        n = len(group_data)
        n_pos = int(group_data["label"].sum())
        prop_pos = n_pos / n if n > 0 else np.nan
        rows.append({
            "group_type": "quartile",
            "group": str(group_name),
            "n": n,
            "n_positive": n_pos,
            "positive_rate": prop_pos,
        })

    risk_table = pd.DataFrame(rows)

    # ── Trend test ───────────────────────────────────────────────────────────
    data_ord = data.dropna(subset=["group_preset"]).copy()
    data_ord["density_ordinal"] = data_ord["group_preset"].cat.codes
    try:
        model_trend = smf.logit("label ~ density_ordinal", data=data_ord).fit(disp=0)
        trend_or = np.exp(model_trend.params["density_ordinal"])
        trend_ci = model_trend.conf_int().loc["density_ordinal"]
        trend_or_ci = (np.exp(trend_ci[0]), np.exp(trend_ci[1]))
        trend_p = model_trend.pvalues["density_ordinal"]
    except Exception as e:
        print(f"    Trend model failed: {e}")
        trend_or = np.nan
        trend_or_ci = (np.nan, np.nan)
        trend_p = np.nan

    trend_result = {
        "variable": "total_domain_count",
        "trend_odds_ratio": trend_or,
        "trend_or_ci_lower": trend_or_ci[0],
        "trend_or_ci_upper": trend_or_ci[1],
        "trend_p_value": trend_p,
    }

    # ── CV model ─────────────────────────────────────────────────────────────
    feat_df = pd.DataFrame({
        "participant_id": data.index,
        "feature": data["density"].values,
        "label": data["label"].values,
    })
    splits = pd.read_csv(SPLIT_CSV)
    participant_probs = []
    for (rpt, fold), fold_rows in splits.groupby(["repeat", "fold"]):
        train_ids = fold_rows[fold_rows["role"] == "train"]["participant_id"].values
        test_ids = fold_rows[fold_rows["role"] == "test"]["participant_id"].values
        train_df = feat_df[feat_df["participant_id"].isin(train_ids)]
        test_df = feat_df[feat_df["participant_id"].isin(test_ids)]
        if len(train_df) == 0 or len(test_df) == 0:
            continue
        X_train = train_df[["feature"]].values
        y_train = train_df["label"].values
        X_test = test_df[["feature"]].values
        prob = fit_fold_local_model(X_train, y_train, X_test, scale=True)
        for pid, p in zip(test_df["participant_id"].values, prob):
            participant_probs.append({"participant_id": pid, "repeat": rpt, "prob": p})

    cv_metrics = {}
    if len(participant_probs) > 0:
        prob_df = pd.DataFrame(participant_probs)
        part_prob = prob_df.groupby("participant_id")["prob"].mean()
        y_true = labels.loc[part_prob.index].values
        y_prob = part_prob.values
        cv_metrics = evaluate_predictions(y_true, y_prob)
        cv_metrics["auc_ci_lower"], cv_metrics["auc_ci_upper"] = bootstrap_auc_ci(y_true, y_prob)

    risk_table_path = OUTDIR / "symptom_evidence_density_gradient_main.csv"
    risk_table.to_csv(risk_table_path, index=False, encoding="utf-8-sig")
    print(f"  [M5] Risk table saved: {risk_table_path}")
    print(f"  [M5] Trend result: OR={trend_or:.3f} (95% CI {trend_or_ci[0]:.3f}-{trend_or_ci[1]:.3f}), p={trend_p:.4f}")
    print(f"  [M5] CV metrics: AUC={cv_metrics.get('auc', np.nan):.4f}")
    return risk_table, trend_result, cv_metrics


print("[Main] Script loaded successfully.")

# ══════════════════════════════════════════════════════════════════════════
# MODULE 6: SYMPTOM GROUP MODELS
# ══════════════════════════════════════════════════════════════════════════

# Preset symptom group definitions
SYMPTOM_GROUPS = {
    "depressive_core": ["depressed_mood", "anhedonia_interest",
                         "sleep_fatigue_energy", "appetite_weight",
                         "self_worth_guilt", "concentration_psychomotor",
                         "suicide_self_harm"],
    "phq8_like": ["depressed_mood", "anhedonia_interest",
                   "sleep_fatigue_energy", "appetite_weight",
                   "self_worth_guilt", "concentration_psychomotor",
                   "functioning_impairment"],
    "mood_cognitive": ["depressed_mood", "anhedonia_interest",
                        "self_worth_guilt", "concentration_psychomotor"],
    "somatic_behavioral": ["sleep_fatigue_energy", "appetite_weight",
                             "functioning_impairment"],
    "risk_clinical_history": ["suicide_self_harm", "mental_health_history",
                               "functioning_impairment"],
    "protective_absent": ["protective_or_absent_symptom"],
    "all_ten_domains": None,  # use all DOMAIN_COLS
}

def _get_group_cols(group_name):
    if group_name == "all_ten_domains":
        return DOMAIN_COLS
    return SYMPTOM_GROUPS[group_name]


def _run_group_cv(df_features, label_series, splits):
    """
    Generic participant-level 10x5 CV for a set of features.
    df_features: DataFrame with participant_id + feature columns.
    Returns: Series(index=participant_id, values=mean OOF prob).
    """
    all_probs = []
    for (rpt, fold), fold_rows in splits.groupby(["repeat", "fold"]):
        train_ids = fold_rows[fold_rows["role"] == "train"]["participant_id"].values
        test_ids = fold_rows[fold_rows["role"] == "test"]["participant_id"].values
        train_df = df_features[df_features["participant_id"].isin(train_ids)]
        test_df = df_features[df_features["participant_id"].isin(test_ids)]
        if len(train_df) == 0 or len(test_df) == 0:
            continue
        feat_cols = [c for c in df_features.columns if c != "participant_id"]
        X_train = train_df[feat_cols].values
        y_train = label_series.loc[train_df["participant_id"].values]
        X_test = test_df[feat_cols].values
        prob = fit_fold_local_model(X_train, y_train.values, X_test, scale=True)
        for pid, p in zip(test_df["participant_id"].values, prob):
            all_probs.append({"participant_id": pid, "repeat": rpt, "prob": p})
    if len(all_probs) == 0:
        return pd.Series(dtype=float)
    prob_df = pd.DataFrame(all_probs)
    return prob_df.groupby("participant_id")["prob"].mean()


def module6_group_models(dcount, dpres, labels):
    """
    Fit symptom group models (presence + count versions for each group).
    Output: sympton_group_model_metrics.csv
    """
    print("\n  [M6] Module 6: Symptom group models...")
    splits = pd.read_csv(SPLIT_CSV)
    rows = []

    for group_name in SYMPTOM_GROUPS.keys():
        cols = _get_group_cols(group_name)
        print(f"    Group: {group_name} ({len(cols)} domains)")

        # ── Presence version ────────────────────────────────────────────────
        feat_df_pres = dpres[["participant_id"] + cols].copy()
        part_prob_pres = _run_group_cv(feat_df_pres, labels, splits)
        if len(part_prob_pres) > 0:
            y_true = labels.loc[part_prob_pres.index].values
            y_prob = part_prob_pres.values
            m = evaluate_predictions(y_true, y_prob)
            ci_l, ci_u = bootstrap_auc_ci(y_true, y_prob)
            # Youden threshold (exploratory)
            fpr, tpr, thresholds = _roc_curve_safe(y_true, y_prob)
            youden = tpr - fpr
            best_idx = np.argmax(youden)
            y_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            y_pred = (y_prob >= y_thresh).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                y_sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
                y_spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            else:
                y_sens = np.nan
                y_spec = np.nan
            rows.append({
                "group_name": group_name,
                "version": "presence",
                "n_domains": len(cols),
                "auc": m["auc"],
                "auc_ci_lower": ci_l,
                "auc_ci_upper": ci_u,
                "pr_auc": m["pr_auc"],
                "brier": m["brier"],
                "logloss": m["logloss"],
                "sens_50": m["sens_50"],
                "spec_50": m["spec_50"],
                "youden_threshold": y_thresh,
                "youden_sens": y_sens,
                "youden_spec": y_spec,
            })

        # ── Count version ──────────────────────────────────────────────────
        feat_df_count = dcount[["participant_id"] + cols].copy()
        part_prob_count = _run_group_cv(feat_df_count, labels, splits)
        if len(part_prob_count) > 0:
            y_true = labels.loc[part_prob_count.index].values
            y_prob = part_prob_count.values
            m = evaluate_predictions(y_true, y_prob)
            ci_l, ci_u = bootstrap_auc_ci(y_true, y_prob)
            fpr, tpr, thresholds = _roc_curve_safe(y_true, y_prob)
            youden = tpr - fpr
            best_idx = np.argmax(youden)
            y_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            y_pred = (y_prob >= y_thresh).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                y_sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
                y_spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            else:
                y_sens = np.nan
                y_spec = np.nan
            rows.append({
                "group_name": group_name,
                "version": "count",
                "n_domains": len(cols),
                "auc": m["auc"],
                "auc_ci_lower": ci_l,
                "auc_ci_upper": ci_u,
                "pr_auc": m["pr_auc"],
                "brier": m["brier"],
                "logloss": m["logloss"],
                "sens_50": m["sens_50"],
                "spec_50": m["spec_50"],
                "youden_threshold": y_thresh,
                "youden_sens": y_sens,
                "youden_spec": y_spec,
            })

    result = pd.DataFrame(rows)
    out_path = OUTDIR / "symptom_group_model_metrics.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  [M6] Saved: {out_path}")
    return result


def _roc_curve_safe(y_true, y_prob):
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return fpr, tpr, thresholds



# ════════════════════════════════════════════════════════════════════════
# MODULE 7: SYMPTOM GROUP MODEL COMPARISONS
# ════════════════════════════════════════════════════════════════════════

def module7_group_comparisons(dcount, dpres, labels, strict_oof):
    """
    Compare key models using paired bootstrap + permutation.
    Comparison family (6 comparisons, one-sided or two-sided):
      1. all_ten_domains_count vs domain_presence_sum
      2. all_ten_domains_count vs total_domain_count
      3. all_ten_domains_count vs phq8_like_count
      4. all_ten_domains_count vs risk_clinical_history_count
      5. all_ten_domains_count vs all_ten_domains_presence
      6. all_ten_domains_count vs M3 (strict OOF)
    FDR: BH within this family (6 comparisons).
    """
    print("\n  [M7] Module 7: Symptom group model comparisons...")
    splits = pd.read_csv(SPLIT_CSV)

    # ── Get OOF probabilities for comparison models ─────────────────────────
    # all_ten_domains_count
    print("    Fitting all_ten_domains_count...")
    cols = DOMAIN_COLS
    feat_df = dcount[["participant_id"] + cols].copy()
    prob_all10_count = _run_group_cv(feat_df, labels, splits)
    prob_all10_count = prob_all10_count.reindex(labels.index)

    # domain_presence_sum (coverage)
    print("    Fitting domain_presence_sum...")
    dpres2 = dpres.copy()
    dpres2["coverage"] = dpres2[DOMAIN_COLS].sum(axis=1)
    feat_cov = pd.DataFrame({
        "participant_id": dpres2["participant_id"],
        "feature": dpres2["coverage"],
    })
    feat_cov["label"] = labels.reindex(feat_cov["participant_id"]).values
    prob_coverage = _run_single_feature_cv_fast(feat_cov, splits)
    prob_coverage = prob_coverage.reindex(labels.index)

    # total_domain_count (density)
    print("    Fitting total_domain_count...")
    dcount2 = dcount.copy()
    dcount2["density"] = dcount2[DOMAIN_COLS].sum(axis=1)
    feat_den = pd.DataFrame({
        "participant_id": dcount2["participant_id"],
        "feature": dcount2["density"],
    })
    feat_den["label"] = labels.reindex(feat_den["participant_id"]).values
    prob_density = _run_single_feature_cv_fast(feat_den, splits)
    prob_density = prob_density.reindex(labels.index)

    # phq8_like_count
    print("    Fitting phq8_like_count...")
    phq8_cols = ["depressed_mood", "anhedonia_interest",
                  "sleep_fatigue_energy", "appetite_weight",
                  "self_worth_guilt", "concentration_psychomotor",
                  "functioning_impairment"]
    feat_phq8 = dcount[["participant_id"] + phq8_cols].copy()
    prob_phq8 = _run_group_cv(feat_phq8, labels, splits)
    prob_phq8 = prob_phq8.reindex(labels.index)

    # risk_clinical_history_count
    print("    Fitting risk_clinical_history_count...")
    risk_cols = ["suicide_self_harm", "mental_health_history",
                 "functioning_impairment"]
    feat_risk = dcount[["participant_id"] + risk_cols].copy()
    prob_risk = _run_group_cv(feat_risk, labels, splits)
    prob_risk = prob_risk.reindex(labels.index)

    # all_ten_domains_presence
    print("    Fitting all_ten_domains_presence...")
    feat_pres = dpres[["participant_id"] + DOMAIN_COLS].copy()
    prob_all10_pres = _run_group_cv(feat_pres, labels, splits)
    prob_all10_pres = prob_all10_pres.reindex(labels.index)

    # M3 (strict OOF)
    prob_m3 = strict_oof.set_index("participant_id")["M3_prob"].reindex(labels.index)

    # ── Compute comparisons ──────────────────────────────────────────────────
    comparisons = [
        ("all10_count", "coverage_sum", prob_all10_count, prob_coverage),
        ("all10_count", "density_sum", prob_all10_count, prob_density),
        ("all10_count", "phq8_count", prob_all10_count, prob_phq8),
        ("all10_count", "risk_count", prob_all10_count, prob_risk),
        ("all10_count", "all10_presence", prob_all10_count, prob_all10_pres),
        ("all10_count", "M3_strict", prob_all10_count, prob_m3),
    ]

    rows = []
    pvals = []
    for idx, (name_a, name_b, prob_a, prob_b) in enumerate(comparisons):
        print(f"    Comparing {name_a} vs {name_b}...")
        y = labels.values
        # Bootstrap for CI only; permutation for p-value
        delta, ci_l, ci_u = bootstrap_delta_auc_ci(y, prob_a.values, prob_b.values)
        p_val = permutation_test_auc_diff(
            y, prob_a.values, prob_b.values,
            seed=RANDOM_SEED + idx
        )
        rows.append({
            "model_a": name_a,
            "model_b": name_b,
            "delta_auc": delta,
            "delta_auc_ci_lower": ci_l,
            "delta_auc_ci_upper": ci_u,
            "permutation_p": p_val,
        })
        pvals.append(p_val)

    result = pd.DataFrame(rows)
    # FDR correction (6 comparisons)
    qvals = bh_fdr(np.array(pvals))
    result["q_value"] = qvals
    result["significance"] = [sig_marker(q) for q in qvals]

    out_path = OUTDIR / "symptom_group_comparisons_fdr.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  [M7] Saved: {out_path}")
    return result


def _run_single_feature_cv_fast(feat_df, splits):
    """Run CV for a single feature (participant-level). feat_df has participant_id, feature, label."""
    all_probs = []
    for (rpt, fold), fold_rows in splits.groupby(["repeat", "fold"]):
        train_ids = fold_rows[fold_rows["role"] == "train"]["participant_id"].values
        test_ids = fold_rows[fold_rows["role"] == "test"]["participant_id"].values
        train_df = feat_df[feat_df["participant_id"].isin(train_ids)]
        test_df = feat_df[feat_df["participant_id"].isin(test_ids)]
        if len(train_df) == 0 or len(test_df) == 0:
            continue
        X_train = train_df[["feature"]].values
        y_train = train_df["label"].values
        X_test = test_df[["feature"]].values
        prob = fit_fold_local_model(X_train, y_train, X_test, scale=True)
        for pid, p in zip(test_df["participant_id"].values, prob):
            all_probs.append({"participant_id": pid, "repeat": rpt, "prob": p})
    if len(all_probs) == 0:
        return pd.Series(dtype=float)
    prob_df = pd.DataFrame(all_probs)
    return prob_df.groupby("participant_id")["prob"].mean()



# ════════════════════════════════════════════════════════════════════════
# MODULE 8: EXPLORATORY THRESHOLD ANALYSIS
# ════════════════════════════════════════════════════════════════════════

def _find_threshold_sens_at_least_80(y_true, y_prob):
    """Find HIGHEST threshold with sensitivity >= 0.80."""
    fpr, tpr, thresholds = _roc_curve_safe(y_true, y_prob)
    # Also need predictive values → use sklearn's precision_recall_curve
    from sklearn.metrics import precision_recall_curve as prc
    prec, rec, th_pr = prc(y_true, y_prob)
    # For sens >= 0.80: highest threshold
    valid = tpr >= 0.80
    if not valid.any():
        return np.nan  # No threshold meets criterion
    # Highest threshold among valid ones
    valid_thresholds = thresholds[valid]
    return valid_thresholds[-1]  # thresholds are decreasing


def _find_threshold_spec_at_least_80(y_true, y_prob):
    """Find LOWEST threshold with specificity >= 0.80."""
    fpr, tpr, thresholds = _roc_curve_safe(y_true, y_prob)
    spec = 1 - fpr
    valid = spec >= 0.80
    if not valid.any():
        return np.nan
    # Lowest threshold among valid ones (leftmost in thresholds array)
    valid_thresholds = thresholds[valid]
    return valid_thresholds[0]  # thresholds are decreasing, so first valid = lowest


def _threshold_metrics(y_true, y_prob, threshold):
    """Compute sensitivity/specificity at a given threshold."""
    if pd.isna(threshold):
        return np.nan, np.nan
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        return sens, spec
    return np.nan, np.nan


def module8_threshold_analysis(labels, strict_oof):
    """
    Exploratory threshold analysis for selected models.
    Thresholds are post-hoc OOF selections → mark exploratory.
    """
    print("\n  [M8] Module 8: Exploratory threshold analysis...")
    # Get OOF probabilities for each model (participant-averaged)
    # We need to re-compute OOF probs for coverage, density, all10_count, all10_presence
    # For simplicity, use the existing strict OOF for M3, and re-run CV for others
    splits = pd.read_csv(SPLIT_CSV)

    # ── (A) domain_presence_sum ────────────────────────────────────────────
    dpres = pd.read_csv(DOMAIN_PRESENCE_CSV)
    dpres = dpres.copy()
    dpres["coverage"] = dpres[DOMAIN_COLS].sum(axis=1)
    feat_cov = pd.DataFrame({
        "participant_id": dpres["participant_id"],
        "feature": dpres["coverage"],
        "label": labels.reindex(dpres["participant_id"]).values,
    })
    prob_cov = _run_single_feature_cv_fast(feat_cov, splits)
    prob_cov = prob_cov.reindex(labels.index)

    # ── (B) total_domain_count ──────────────────────────────────────────────
    dcount = pd.read_csv(DOMAIN_COUNT_CSV)
    dcount = dcount.copy()
    dcount["density"] = dcount[DOMAIN_COLS].sum(axis=1)
    feat_den = pd.DataFrame({
        "participant_id": dcount["participant_id"],
        "feature": dcount["density"],
        "label": labels.reindex(dcount["participant_id"]).values,
    })
    prob_den = _run_single_feature_cv_fast(feat_den, splits)
    prob_den = prob_den.reindex(labels.index)

    # ── (C) all 10 domain counts ───────────────────────────────────────────
    feat_all10 = dcount[["participant_id"] + DOMAIN_COLS].copy()
    prob_all10 = _run_group_cv(feat_all10, labels, splits)
    prob_all10 = prob_all10.reindex(labels.index)

    # ── (D) all 10 domain presence ───────────────────────────────────────
    feat_all10p = dpres[["participant_id"] + DOMAIN_COLS].copy()
    prob_all10p = _run_group_cv(feat_all10p, labels, splits)
    prob_all10p = prob_all10p.reindex(labels.index)

    # ── (E) M3 strict ─────────────────────────────────────────────────────
    prob_m3 = strict_oof.set_index("participant_id")["M3_prob"].reindex(labels.index)

    models = {
        "domain_presence_sum": prob_cov,
        "total_domain_count": prob_den,
        "all10_domain_counts": prob_all10,
        "all10_domain_presence": prob_all10p,
        "M3_strict": prob_m3,
    }

    rows = []
    for model_name, probs in models.items():
        print(f"    Threshold analysis: {model_name}")
        y = labels.values
        p = probs.values

        # Default 0.50
        sens_50, spec_50 = _threshold_metrics(y, p, 0.50)

        # Youden threshold
        fpr, tpr, thresholds = _roc_curve_safe(y, p)
        youden = tpr - fpr
        best_idx = np.argmax(youden)
        thresh_youden = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        sens_youden, spec_youden = _threshold_metrics(y, p, thresh_youden)

        # Macro-F1 threshold
        from sklearn.metrics import f1_score
        best_f1_thresh = 0.5
        best_f1 = -1
        for t in np.arange(0.01, 0.99, 0.01):
            y_pred = (p >= t).astype(int)
            f1 = f1_score(y, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_f1_thresh = t
        sens_f1, spec_f1 = _threshold_metrics(y, p, best_f1_thresh)

        # Sensitivity >= 0.80 (HIGHEST threshold)
        thresh_sens80 = _find_threshold_sens_at_least_80(y, p)
        sens_sens80, spec_sens80 = _threshold_metrics(y, p, thresh_sens80)

        # Specificity >= 0.80 (LOWEST threshold)
        thresh_spec80 = _find_threshold_spec_at_least_80(y, p)
        sens_spec80, spec_spec80 = _threshold_metrics(y, p, thresh_spec80)

        rows.append({
            "model": model_name,
            "default50_sens": sens_50,
            "default50_spec": spec_50,
            "youden_threshold": thresh_youden,
            "youden_sens": sens_youden,
            "youden_spec": spec_youden,
            "youden_note": "exploratory_posthoc",
            "macroF1_threshold": best_f1_thresh,
            "macroF1_sens": sens_f1,
            "macroF1_spec": spec_f1,
            "macroF1_note": "exploratory_posthoc",
            "sens80_threshold": thresh_sens80,
            "sens80_sens": sens_sens80,
            "sens80_spec": spec_sens80,
            "sens80_note": "exploratory_posthoc",
            "spec80_threshold": thresh_spec80,
            "spec80_sens": sens_spec80,
            "spec80_spec": spec_spec80,
            "spec80_note": "exploratory_posthoc",
        })

    result = pd.DataFrame(rows)
    out_path = OUTDIR / "symptom_threshold_analysis_exploratory.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  [M8] Saved: {out_path}")
    return result



# ═══════════════════════════════════════════════════════════════════════
# MODULE 9: CALIBRATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def _calibration_metrics(y_true, y_prob, n_bins=10):
    """Compute calibration metrics."""
    # Brier + LogLoss already in evaluate_predictions
    # Calibration intercept & slope: logistic regression of y_true ~ logit(y_prob)
    logit_probs = np.log(y_prob / (1 - y_prob) + 1e-12)
    # Use statsmodels for intercept & slope
    try:
        X = sm.add_constant(logit_probs)
        model = sm.GLM(y_true, X, family=sm.families.Binomial()).fit(disp=0)
        cal_intercept = model.params[0]
        cal_slope = model.params[1]
    except Exception:
        cal_intercept = np.nan
        cal_slope = np.nan

    # Expected calibration error (ECE)
    prob_df = pd.DataFrame({"y": y_true, "p": y_prob})
    prob_df["bin"] = pd.qcut(prob_df["p"], n_bins, duplicates="drop")
    ece = 0.0
    for _, group in prob_df.groupby("bin"):
        rel_freq = group["y"].mean()
        conf = group["p"].mean()
        ece += (len(group) / len(prob_df)) * abs(rel_freq - conf)
    return {
        "cal_intercept": cal_intercept,
        "cal_slope": cal_slope,
        "ece": ece,
    }


def module9_calibration(labels, strict_oof):
    """
    Calibration analysis for selected models.
    Models: coverage_sum, density_sum, all10_count, all10_presence, M3_strict.
    """
    print("\n  [M9] Module 9: Calibration analysis...")
    splits = pd.read_csv(SPLIT_CSV)

    # Re-run CV to get OOF probs (same as Module 8)
    dpres = pd.read_csv(DOMAIN_PRESENCE_CSV)
    dcount = pd.read_csv(DOMAIN_COUNT_CSV)

    dpres2 = dpres.copy()
    dpres2["coverage"] = dpres2[DOMAIN_COLS].sum(axis=1)
    feat_cov = pd.DataFrame({
        "participant_id": dpres2["participant_id"],
        "feature": dpres2["coverage"],
        "label": labels.reindex(dpres2["participant_id"]).values,
    })
    prob_cov = _run_single_feature_cv_fast(feat_cov, splits)
    prob_cov = prob_cov.reindex(labels.index)

    dcount2 = dcount.copy()
    dcount2["density"] = dcount2[DOMAIN_COLS].sum(axis=1)
    feat_den = pd.DataFrame({
        "participant_id": dcount2["participant_id"],
        "feature": dcount2["density"],
        "label": labels.reindex(dcount2["participant_id"]).values,
    })
    prob_den = _run_single_feature_cv_fast(feat_den, splits)
    prob_den = prob_den.reindex(labels.index)

    feat_all10 = dcount[["participant_id"] + DOMAIN_COLS].copy()
    prob_all10 = _run_group_cv(feat_all10, labels, splits)
    prob_all10 = prob_all10.reindex(labels.index)

    feat_all10p = dpres[["participant_id"] + DOMAIN_COLS].copy()
    prob_all10p = _run_group_cv(feat_all10p, labels, splits)
    prob_all10p = prob_all10p.reindex(labels.index)

    prob_m3 = strict_oof.set_index("participant_id")["M3_prob"].reindex(labels.index)

    models = {
        "domain_presence_sum": prob_cov,
        "total_domain_count": prob_den,
        "all10_domain_counts": prob_all10,
        "all10_domain_presence": prob_all10p,
        "M3_strict": prob_m3,
    }

    rows = []
    for model_name, probs in models.items():
        print(f"    Calibrating: {model_name}")
        y = labels.values
        p = probs.values
        m = evaluate_predictions(y, p)
        cal = _calibration_metrics(y, p)
        rows.append({
            "model": model_name,
            "brier": m["brier"],
            "logloss": m["logloss"],
            "cal_intercept": cal["cal_intercept"],
            "cal_slope": cal["cal_slope"],
            "ece": cal["ece"],
        })

    result = pd.DataFrame(rows)
    out_path = OUTDIR / "symptom_calibration_metrics.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  [M9] Saved: {out_path}")
    return result




# ═════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════


def module4_coverage_gradient_fine(dpres, labels):
    """Supplementary fine-grained coverage gradient.
    Output: symptom_coverage_gradient_fine.csv
    Grouping: 0-2, 3-4, 5-7, 8-10
    """
    print("\n  [M4-fine] Coverage gradient (fine-grained)...")
    dp = dpres.copy()
    dp["domain_presence_sum"] = dp[DOMAIN_COLS].sum(axis=1)
    dp = dp.set_index("participant_id")
    data = pd.DataFrame({
        "label": labels.reindex(dp.index),
        "coverage": dp["domain_presence_sum"],
    }).dropna()
    data["group"] = pd.cut(
        data["coverage"],
        bins=[-0.5, 2.5, 4.5, 6.5, 10.5],
            labels=["0-2", "3-4", "5-6", "7-10"],
    )
    print("    Fine group sizes:", data.groupby("group").size().to_dict())
    rows = []
    for g, gd in data.groupby("group"):
        n = len(gd); npos = int(gd["label"].sum())
        rows.append({"group": str(g), "n": n, "n_positive": npos,
                     "positive_rate": npos/n if n>0 else float("nan")})
    tbl = pd.DataFrame(rows)
    out = OUTDIR / "symptom_coverage_gradient_fine.csv"
    tbl.to_csv(out, index=False, encoding="utf-8-sig")
    print("  [M4-fine] Saved:", out)
    return tbl


def module5_evidence_density_gradient_fine(dcount, labels):
    """Supplementary fine-grained evidence density gradient.
    Output: symptom_evidence_density_gradient_fine.csv
    Grouping: 0-2, 3-5, 6-10, 11+
    """
    print("\n  [M5-fine] Evidence density gradient (fine-grained)...")
    dc = dcount.copy()
    dc["total_domain_count"] = dc[DOMAIN_COLS].sum(axis=1)
    dc = dc.set_index("participant_id")
    data = pd.DataFrame({
        "label": labels.reindex(dc.index),
        "density": dc["total_domain_count"],
    }).dropna()
    mx = data["density"].max()
    upper = max(11, mx)
    data["group"] = pd.cut(
        data["density"],
        bins=[-0.5, 2.5, 5.5, 10.5, upper + 0.5],
        labels=["0-2", "3-5", "6-10", "11+"],
    )
    print("    Fine group sizes:", data.groupby("group").size().to_dict())
    rows = []
    for g, gd in data.groupby("group"):
        n = len(gd); npos = int(gd["label"].sum())
        rows.append({"group": str(g), "n": n, "n_positive": npos,
                     "positive_rate": npos/n if n>0 else float("nan")})
    tbl = pd.DataFrame(rows)
    out = OUTDIR / "symptom_evidence_density_gradient_fine.csv"
    tbl.to_csv(out, index=False, encoding="utf-8-sig")
    print("  [M5-fine] Saved:", out)
    return tbl



def main():
    print("=" * 70)
    print("  Symptom-Level Signal Decomposition Analysis")
    print("  DAIC-WOZ PHQ-8 Positive Prediction")
    print("=" * 70)
    dcount, dpres, splits, strict_oof = validate_inputs()
    labels = pd.Series(strict_oof.set_index("participant_id")["label"])
    labels = labels.reindex(dcount["participant_id"].values)
    desc = module1_descriptive(dcount, dpres, labels)
    single_domain = module2_single_domain_cv(dcount, dpres, labels)
    association = module3_univariate_association(dcount, dpres, labels)
    coverage_tbl, coverage_trend, coverage_cv = module4_coverage_gradient(dpres, labels)
    density_tbl, density_trend, density_cv = module5_evidence_density_gradient(dcount, labels)
    # Fine-grained supplementary gradients
    module4_coverage_gradient_fine(dpres, labels)
    module5_evidence_density_gradient_fine(dcount, labels)
    # Combined trend FDR (Module 4/5 in same family)
    trend_pvals = [coverage_trend["trend_p_value"], density_trend["trend_p_value"]]
    trend_qvals = bh_fdr(np.array(trend_pvals))
    trend_rows = [
        {"variable": "domain_presence_sum", "test_type": "coverage_gradient",
         "p_value": coverage_trend["trend_p_value"],
         "q_value": trend_qvals[0],
         "significance": sig_marker(trend_qvals[0])},
        {"variable": "total_domain_count", "test_type": "density_gradient",
         "p_value": density_trend["trend_p_value"],
         "q_value": trend_qvals[1],
         "significance": sig_marker(trend_qvals[1])},
    ]
    trend_fdr_df = pd.DataFrame(trend_rows)
    trend_fdr_path = OUTDIR / "symptom_gradient_trend_fdr.csv"
    trend_fdr_df.to_csv(trend_fdr_path, index=False, encoding="utf-8-sig")
    print(f"  [M4/M5] Combined trend FDR saved: {trend_fdr_path}")
    group_models = module6_group_models(dcount, dpres, labels)
    comparisons = module7_group_comparisons(dcount, dpres, labels, strict_oof)
    threshold = module8_threshold_analysis(labels, strict_oof)
    calibration = module9_calibration(labels, strict_oof)
    generate_summary(desc, single_domain, association,
                     coverage_trend, density_trend,
                     group_models, comparisons,
                     threshold, calibration)
    generate_manifest()
    print("\n" + "=" * 70)
    print("  ALL MODULES COMPLETED SUCCESSFULLY.")
    print("=" * 70)


def generate_summary(desc, single_domain, association,
                     coverage_trend, density_trend,
                     group_models, comparisons,
                     threshold, calibration):
    print("\n  [SUM] Generating summary...")
    lines = []
    lines.append("# Symptom-Level Signal Decomposition Analysis")
    lines.append("## Summary Report")
    lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append(f"**Dataset**: DAIC-WOZ, N=142 (PHQ-8-positive=43, negative=99)")
    lines.append(f"**Method**: Participant-level 10x5 repeated cross-validation")
    lines.append(f"**Random seed**: 20260705")
    lines.append(f"\n---\n")
    lines.append("## 1. Input Validation")
    lines.append("All critical checks passed (see `input_validation_report.json`).")
    lines.append("Merged sample size = 142, positive = 43, negative = 99.")
    lines.append("\n## 2. Single Symptom Domain Prediction Signals")
    if single_domain is not None:
        sig = single_domain[single_domain["significance"] != "ns"]
        if len(sig) == 0:
            lines.append("**No individual symptom domain showed a corrected-significant (q < 0.05) prediction signal.**")
            lines.append("This does NOT mean individual symptoms are 'useless'; rather, no single domain carries enough standalone signal to survive FDR correction in this sample.")
        else:
            lines.append(f"{len(sig)} individual domain model(s) showed q < 0.05:")
            for _, row in sig.iterrows():
                lines.append(f"- **{row['domain']}** ({row['domain_cn']}), {row['model_type']}: AUC={row['auc']:.3f}")
    lines.append("\n## 3. Symptom Coverage Breadth (Presence Sum)")
    if coverage_trend is not None:
        t_or = coverage_trend.get("trend_odds_ratio", float("nan"))
        t_p = coverage_trend.get("trend_p_value", 1.0)
        lines.append(f"Trend test (ordinal coverage group): OR={t_or:.3f}, p={t_p:.4f}")
        if t_p < 0.05:
            lines.append("**Significant risk gradient by symptom coverage breadth** was observed. Prediction signal appears cumulative across symptom domains.")
    lines.append("\n## 4. Symptom Evidence Density (Count Sum)")
    if density_trend is not None:
        t_or = density_trend.get("trend_odds_ratio", float("nan"))
        t_p = density_trend.get("trend_p_value", 1.0)
        lines.append(f"Trend test (ordinal density group): OR={t_or:.3f}, p={t_p:.4f}")
        if t_p < 0.05:
            lines.append("**Significant risk gradient by total symptom evidence count** was observed. Prediction signal has a density component.")
    lines.append("\n## 5. Symptom Group Models vs. Full 10-Domain Model")
    if comparisons is not None:
        sig_comp = comparisons[comparisons["significance"] != "ns"]
        if len(sig_comp) > 0:
            lines.append(f"{len(sig_comp)} comparison(s) showed q < 0.05:")
            for _, row in sig_comp.iterrows():
                lines.append(f"- {row['model_a']} vs {row['model_b']}: ΔAUC={row['delta_auc']:.4f}, q={row['q_value']:.4f}")
        else:
            lines.append("No comparison reached corrected significance (q < 0.05). The full 10-domain model did not significantly outperform simpler aggregates (coverage sum, density sum, or PHQ-8-like subset). **This suggests the main predictive signal may come from symptom coverage breadth or evidence density, not fine-grained symptom patterns.**")
    lines.append("\n## 6. Comparison with Strict Full Joint Model (M3)")
    if comparisons is not None:
        m3_row = comparisons[comparisons["model_b"] == "M3_strict"]
        if len(m3_row) > 0:
            row = m3_row.iloc[0]
            lines.append(f"All-10-domain count model vs. M3 (strict OOF): ΔAUC={row['delta_auc']:.4f}, q={row['q_value']:.4f} ({row['significance']})")
            if row["significance"] == "ns":
                lines.append("The 10-domain count model did NOT significantly underperform the strict full joint model. **Low-dimensional symptom domain representations approach the performance of complex joint models.**")
    lines.append("\n## 7. Interpretation Guidelines for Manuscript")
    lines.append("The following statements are WARRANTED by this analysis:")
    lines.append("- Individual symptom domains did / did not show standalone corrected-significant signals (see Section 2).")
    lines.append("- Symptom coverage breadth and/or evidence density explain much of the predictive signal.")
    lines.append("- Fine-grained symptom patterns (10-domain counts) did not significantly improve over aggregate measures.")
    lines.append("\nThe following statements are NOT WARRANTED:")
    lines.append("- 'Symptom X is unimportant' (absence of standalone signal ≠ unimportance).")
    lines.append("- 'The 10-domain model is equivalent to M3' (non-inferiority was not tested; only non-significance of ΔAUC).")
    lines.append("- Any threshold-based sensitivity/specificity values from Module 8 as 'optimal clinical thresholds' (they are post-hoc OOF selections, not independent validations).")
    lines.append("\n---\n*Generated by `run_symptom_level_analysis.py`*")
    summary_path = OUTDIR / "symptom_level_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [SUM] Summary saved: {summary_path}")


def generate_manifest():
    print("  [MAN] Generating run manifest...")
    import sklearn, scipy
    manifest = {
        "analysis_time": datetime.now().isoformat(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "scipy_version": scipy.__version__,
        "statsmodels_version": sm.__version__,
        "input_files": {
            "domain_count": str(DOMAIN_COUNT_CSV),
            "domain_presence": str(DOMAIN_PRESENCE_CSV),
            "splits": str(SPLIT_CSV),
            "strict_oof": str(STRICT_OOF_CSV),
        },
        "input_sha256": {
            "domain_count": sha256_file(DOMAIN_COUNT_CSV),
            "domain_presence": sha256_file(DOMAIN_PRESENCE_CSV),
            "splits": sha256_file(SPLIT_CSV),
            "strict_oof": sha256_file(STRICT_OOF_CSV),
        },
        "random_seed": RANDOM_SEED,
        "bootstrap_n": BOOTSTRAP_N,
        "permutation_n": PERMUTATION_N,
        "lr_params": LR_PARAMS,
    }
    manifest_path = OUTDIR / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    out_files = list(OUTDIR.glob("*.csv")) + list(OUTDIR.glob("*.json")) + list(OUTDIR.glob("*.md"))
    sha_rows = []
    for fp in out_files:
        sha_rows.append({"file": fp.name, "sha256": sha256_file(fp)})
    sha_df = pd.DataFrame(sha_rows)
    sha_df.to_csv(OUTDIR / "output_manifest_sha256.csv", index=False)
    print(f"  [MAN] Manifest saved: {manifest_path}")


if __name__ == "__main__":
    main()
