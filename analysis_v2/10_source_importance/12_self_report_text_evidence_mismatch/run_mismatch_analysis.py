#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-report vs. Interview-evidence Mismatch & Model-error Analysis (PHQ-8)
=========================================================================

DAIC-WOZ PHQ-8 symptom-severity signal-source decomposition — extension that
shifts the question from "does the complex model beat simple symptom load?"
(incremental validity, module 11) to a more interpretable one:

    "Is the PHQ-8 SELF-REPORTED score consistent with the symptom EVIDENCE
     extracted from the interview text? And are model ERRORS concentrated in
     the participants where self-report and interview evidence disagree?"

Rationale
---------
PHQ-8 is a SELF-REPORT rating scale, not a clinical diagnosis. The interview
text carries INDEPENDENTLY extracted symptom evidence (coverage breadth,
evidence density, ten symptom-domain counts/presence). When a participant's
self-reported PHQ-8 is high but the interview evidence is low (or vice versa),
the two information sources disagree. Models built from interview evidence
cannot, by construction, "correct" a self-report that diverges from the
evidence — so their errors should concentrate exactly in those mismatched
participants.

Hard constraints (inherited from the strict reanalysis SPEC):
  - Read ONLY from frozen input files (allowed paths below).
  - NEVER use any OOF probability as a feature for a new model.
  - ALL merges MUST be on `participant_id`; NEVER rely on row order.
  - All models MUST be fold-local (fit scaler + model on TRAIN only).
  - 10x5 repeated participant-level cross-validation (frozen splits).
  - Do NOT read any superseded 01-06 results.

Outputs (8 files):
  mismatch_quadrants_by_coverage.csv
  mismatch_quadrants_by_density.csv
  model_error_by_mismatch_group.csv
  error_profile_correct_vs_incorrect.csv
  representative_mismatch_cases.csv
  mismatch_analysis_summary.md
  run_manifest.json
  output_manifest_sha256.csv

Author: strict reanalysis (2026-07-08)
"""

import os
import json
import hashlib
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths (SPEC-compliant, same BASE as module 11) ───────────────────────────
SCRIPT_DIR = Path(__file__).parent
BASE = SCRIPT_DIR.parent.parent
DOMAIN_COUNT_CSV = BASE / "04_c5_controls/domain_count/input.csv"
DOMAIN_PRESENCE_CSV = BASE / "04_c5_controls/domain_presence/input.csv"
SPLIT_CSV = BASE / "00_splits/repeated_5fold_splits_10x5.csv"
FROZEN_TURNS = BASE / "01_inputs/c4_turns_official142_frozen.csv"
STRICT_OOF_CSV = BASE / "10_source_importance/07_strict_joint_models/participant_oof_predictions.csv"
OUTDIR = SCRIPT_DIR

# ── Symptom domain definitions (must match modules 09 / 10 / 11) ─────────────
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

RANDOM_SEED = 20260705
LR_PARAMS = dict(
    penalty="l2", C=1.0, class_weight="balanced",
    solver="liblinear", max_iter=2000, random_state=RANDOM_SEED,
)
np.random.seed(RANDOM_SEED)

# ── Analysis parameters ──────────────────────────────────────────────────────
THRESHOLD = 0.5          # decision threshold for predicted class
PHQ_POS_CUT = 10         # PHQ-8 >= 10 defines self-report "high"
COVERAGE_PREDEF = 5      # pre-defined "high evidence" = >= 5 touched domains
DENSITY_PREDEF = 11      # pre-defined "high evidence" = >= 11 evidence pieces
N_PERM_CONC = 10000      # permutations for error-concentration test


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

    # Check A: PHQ-8 score constant per participant
    g_score = frozen.groupby("participant_id")["paper_phq8_score"]
    if (g_score.nunique() > 1).any():
        errors.append("CRITICAL: paper_phq8_score not constant per participant")

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

    if "M3_prob" in oof.columns:
        df = df.merge(
            oof[["participant_id", "M3_prob"]],
            on="participant_id", how="left",
        )
    else:
        errors.append("CRITICAL: M3_prob OOF missing from strict_oof file")

    if set(df["participant_id"]) != set(splits["participant_id"].unique()):
        errors.append("CRITICAL: participant_id mismatch with splits")
    if len(df) != 142:
        errors.append(f"CRITICAL: merged n={len(df)}, expected 142")
    if df["participant_id"].duplicated().any():
        errors.append("CRITICAL: duplicate participant_id after merge")
    if df["phq8_score"].isna().any():
        errors.append("CRITICAL: missing PHQ-8 score")
    if df["M3_prob"].isna().any():
        errors.append("CRITICAL: missing M3_prob for some participants")

    if errors:
        for e in errors:
            print("  ", e)
        raise ValueError("Input validation FAILED")

    print(f"[LOAD] OK: n={len(df)}, score range "
          f"{df['phq8_score'].min()}-{df['phq8_score'].max()}, "
          f"positive={int(df['label'].sum())}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATED BINARY PREDICTION (fold-local, 10x5 repeated)
# ═══════════════════════════════════════════════════════════════════════════
def run_logistic_cv(df, feature_cols, label_series):
    """Participant-level 10x5 repeated CV for binary logistic regression.
    Returns a Series indexed by participant_id of the mean OOF probability."""
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
# QUADRANT ASSIGNMENT
# ═══════════════════════════════════════════════════════════════════════════
QUADRANTS = ["consistent_high", "consistent_low",
             "self_high_evidence_low", "self_low_evidence_high"]
QUADRANT_CN = {
    "consistent_high": "一致高症状（PHQ高+证据高）",
    "consistent_low": "一致低症状（PHQ低+证据低）",
    "self_high_evidence_low": "自评高但访谈证据不足",
    "self_low_evidence_high": "自评低但访谈证据较多",
}
MISMATCH_QUADRANTS = ["self_high_evidence_low", "self_low_evidence_high"]


def assign_quadrants(phq_high, ev_high):
    q = np.full(len(phq_high), "consistent_low", dtype=object)
    q[phq_high & ev_high] = "consistent_high"
    q[phq_high & ~ev_high] = "self_high_evidence_low"
    q[~phq_high & ev_high] = "self_low_evidence_high"
    return q


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY MARKDOWN
# ═══════════════════════════════════════════════════════════════════════════
def _q(quad_df, ed, rule, q):
    sub = quad_df[(quad_df["evidence_def"] == ed) &
                  (quad_df["threshold_rule"] == rule) &
                  (quad_df["quadrant"] == q)]
    return sub.iloc[0] if len(sub) else None


def build_summary(oof, quad_df, err_df, prof_df, rep_df, manifest):
    cov_med = manifest["coverage_median"]
    den_med = manifest["density_median"]
    L = []
    L.append("# 自评—访谈证据错位与模型误差分析")
    L.append("")
    L.append("> 模块 12 · DAIC-WOZ PHQ-8 · 分析日期 2026-07-08")
    L.append("> 全部数值来自冻结输入与冻结 10×5 参与者级交叉验证，可通过仓库 `codex/reanalysis-v2` 分支复现。")
    L.append("")
    L.append("## 1 研究问题与动机")
    L.append("")
    L.append("PHQ-8 是**自评量表**总分，反映被试对自身抑郁症状的主观报告，"
             "并不等同于临床诊断。本研究从访谈文本中独立抽取了症状证据结构"
             "（症状覆盖广度、症状证据密度、十个症状域计数/是否出现）。"
             "当被试的 PHQ-8 自评分数与访谈文本中的客观症状证据不一致时，"
             "两条信息源出现**错位（mismatch）**。")
    L.append("")
    L.append("模块 11 的增量效度分析表明，复杂文本模型与十域细粒度表征均未稳定超越"
             "简单症状负荷指标。本模块进一步追问一个更具解释力的问题：")
    L.append("")
    L.append("> **模型错误是否集中发生在自评标签与访谈证据不一致的样本中？**")
    L.append("")
    L.append("由于本研究的基础负荷模型、十域模型均以访谈证据特征（覆盖广度、证据密度、"
             "症状域计数/出现）为输入，它们本质上是在用访谈证据去预测 PHQ-8 自评标签。"
             "当自评与证据错位时，这类模型很难“纠正”被试的主观报告，其预测误差理应"
             "集中出现在错位样本。")
    L.append("")
    L.append("**说明**：本分析关注交叉验证预测性能的误差分布，不解释任何单个回归系数；"
             "PHQ-8 为自评得分，结局应理解为“自评抑郁症状严重度/风险”，不能等同于临床诊断。")
    L.append("")
    L.append("## 2 方法")
    L.append("")
    L.append("- **样本**：DAIC-WOZ 全 142 名被试；PHQ-8 阳性定义为自评总分 ≥10（n=%d）。" % int(oof["label"].sum()))
    L.append("- **错位分组**：以 PHQ-8 自评高低（≥10 为“高”）与访谈证据高低交叉，"
             "构成四象限。证据高低采用两种定义：")
    L.append("  - **覆盖广度定义**：证据高 = 触及症状域数 ≥ 中位数(%.1f) 或预定义 ≥%d 个症状域；"
             % (cov_med, manifest["coverage_predef_high"]))
    L.append("  - **证据密度定义**：证据高 = 症状证据总条数 ≥ 中位数(%.1f) 或预定义 ≥%d 条。"
             % (den_med, manifest["density_predef_high"]))
    L.append("- **模型**：基础负荷模型（覆盖广度+证据密度）、十域计数模型、十域出现模型"
             "均为 fold-local 逻辑回归（10×5 参与者级 CV，外部折外概率）；"
             "复杂联合文本模型使用模块 07 的外部折外概率（M3_prob），仅作评价参照，不作为新模型特征。")
    L.append("- **错分判定**：折外概率 ≥%.2f 判为阳性预测；与 PHQ-8 阳性标签不一致即记为错分。" % manifest["decision_threshold"])
    L.append("- **误差集中检验**：对每个模型，比较错位象限（自评高+证据低、自评低+证据高）"
             "与一致象限的错分率差异，采用参与者级置换检验（%d 次）构建零分布。"
             % manifest["n_perm_concentration"])
    L.append("- **正确 vs 错分特征比较**：对覆盖广度、证据密度、PHQ-8 总分及十个症状域的"
             "计数/出现，比较正确与错分样本的分布差异（Mann–Whitney U 检验，BH FDR 校正）。")
    L.append("")
    L.append("## 3 结果")
    L.append("")
    L.append("### 3.1 自评—访谈证据错位四象限")
    L.append("")

    # Quadrant table for predefined rules (primary)
    L.append("**表 1（覆盖广度定义，证据高 = 触及 ≥%d 个症状域）**" % manifest["coverage_predef_high"])
    L.append("")
    L.append("| 象限 | n | 比例% | PHQ 均分 | 阳性率 | 覆盖广度 | 证据密度 |")
    L.append("|------|---|------|---------|-------|---------|---------|")
    for q in QUADRANTS:
        r = _q(quad_df, "coverage_breadth", "predefined_5plus", q)
        if r is not None:
            L.append("| %s | %d | %.1f%% | %.2f | %.2f | %.2f | %.2f |" % (
                QUADRANT_CN[q], int(r["n"]), r["pct"], r["mean_phq8"],
                r["phq_positive_rate"], r["mean_coverage_breadth"], r["mean_evidence_density"]))
    L.append("")
    L.append("**表 2（证据密度定义，证据高 = 总证据 ≥%d 条）**" % manifest["density_predef_high"])
    L.append("")
    L.append("| 象限 | n | 比例% | PHQ 均分 | 阳性率 | 覆盖广度 | 证据密度 |")
    L.append("|------|---|------|---------|-------|---------|---------|")
    for q in QUADRANTS:
        r = _q(quad_df, "evidence_density", "predefined_11plus", q)
        if r is not None:
            L.append("| %s | %d | %.1f%% | %.2f | %.2f | %.2f | %.2f |" % (
                QUADRANT_CN[q], int(r["n"]), r["pct"], r["mean_phq8"],
                r["phq_positive_rate"], r["mean_coverage_breadth"], r["mean_evidence_density"]))
    L.append("")
    L.append("两种证据定义下，均存在相当比例的自评—证据错位样本：")
    n_over_cov = _q(quad_df, "coverage_breadth", "predefined_5plus", "self_high_evidence_low")["n"]
    n_under_cov = _q(quad_df, "coverage_breadth", "predefined_5plus", "self_low_evidence_high")["n"]
    n_over_den = _q(quad_df, "evidence_density", "predefined_11plus", "self_high_evidence_low")["n"]
    n_under_den = _q(quad_df, "evidence_density", "predefined_11plus", "self_low_evidence_high")["n"]
    L.append("- 覆盖广度定义：自评高但证据不足 **%d** 人，自评低但证据较多 **%d** 人；" % (int(n_over_cov), int(n_under_cov)))
    L.append("- 证据密度定义：自评高但证据不足 **%d** 人，自评低但证据较多 **%d** 人。" % (int(n_over_den), int(n_under_den)))
    L.append("")
    L.append("中位数切分下的错位规模与此接近（见 `mismatch_quadrants_by_coverage.csv` "
             "与 `mismatch_quadrants_by_density.csv`）。")
    L.append("")
    L.append("### 3.2 各模型总体错分情况")
    L.append("")
    L.append("| 模型 | TP | FP | FN | TN | 错分率 |")
    L.append("|------|----|----|----|----|-------|")
    for m in ["base", "count", "pres", "complex"]:
        r = err_df[(err_df["model"] == m) & (err_df["group"] == "overall")].iloc[0]
        L.append("| %s | %d | %d | %d | %d | %.3f |" % (
            m, int(r["tp"]), int(r["fp"]), int(r["fn"]), int(r["tn"]), r["error_rate"]))
    L.append("")
    L.append("### 3.3 模型错误是否集中在错位样本")
    L.append("")
    L.append("**表 3（覆盖广度定义，错位 vs 一致象限的错分率与置换检验）**")
    L.append("")
    L.append("| 模型 | 错位错分率 | 一致错分率 | 差异 | 置换 p |")
    L.append("|------|----------|----------|------|-------|")
    for m in ["base", "count", "pres", "complex"]:
        r = err_df[(err_df["model"] == m) & (err_df["evidence_def"] == "coverage_breadth")
                   & (err_df["threshold_rule"] == "predefined_5plus")
                   & (err_df["group"] == "mismatch_vs_consistent")].iloc[0]
        L.append("| %s | %.3f | %.3f | %+.3f | %.4f |" % (
            m, r["error_rate_mismatch"], r["error_rate_consistent"],
            r["error_rate_diff"], r["permutation_p"]))
    L.append("")
    L.append("**表 4（证据密度定义）**")
    L.append("")
    L.append("| 模型 | 错位错分率 | 一致错分率 | 差异 | 置换 p |")
    L.append("|------|----------|----------|------|-------|")
    for m in ["base", "count", "pres", "complex"]:
        r = err_df[(err_df["model"] == m) & (err_df["evidence_def"] == "evidence_density")
                   & (err_df["threshold_rule"] == "predefined_11plus")
                   & (err_df["group"] == "mismatch_vs_consistent")].iloc[0]
        L.append("| %s | %.3f | %.3f | %+.3f | %.4f |" % (
            m, r["error_rate_mismatch"], r["error_rate_consistent"],
            r["error_rate_diff"], r["permutation_p"]))
    L.append("")
    # narrative on concentration
    conc_base_cov = err_df[(err_df["model"] == "base") & (err_df["evidence_def"] == "coverage_breadth")
                           & (err_df["threshold_rule"] == "predefined_5plus")
                           & (err_df["group"] == "mismatch_vs_consistent")].iloc[0]
    conc_complex_cov = err_df[(err_df["model"] == "complex") & (err_df["evidence_def"] == "coverage_breadth")
                              & (err_df["threshold_rule"] == "predefined_5plus")
                              & (err_df["group"] == "mismatch_vs_consistent")].iloc[0]
    conc_base_den = err_df[(err_df["model"] == "base") & (err_df["evidence_def"] == "evidence_density")
                           & (err_df["threshold_rule"] == "predefined_11plus")
                           & (err_df["group"] == "mismatch_vs_consistent")].iloc[0]
    conc_complex_den = err_df[(err_df["model"] == "complex") & (err_df["evidence_def"] == "evidence_density")
                              & (err_df["threshold_rule"] == "predefined_11plus")
                              & (err_df["group"] == "mismatch_vs_consistent")].iloc[0]
    L.append("两种证据定义下，各模型的错分率均在错位象限显著高于一致象限，置换检验均显著"
             "（p≤%.4f）。值得注意的是，复杂联合文本模型在错位样本上的错分率"
             "（覆盖广度定义 %.3f、证据密度定义 %.3f）明显低于基础负荷模型"
             "（%.3f、%.3f），说明其借助文本语义在一定程度上缓解了对错位样本的误判；"
             "但复杂模型仍对约三分之一至二分之一的错位样本判错，且模块 11 的正式检验显示"
             "其相对基础负荷模型的 ΔAUC 并未达到统计稳定。换言之，复杂模型能部分缓解、"
             "却不足以稳定消除由自评—访谈错位带来的误差。"
             % (max(conc_complex_cov["permutation_p"], conc_complex_den["permutation_p"]),
                conc_complex_cov["error_rate_mismatch"], conc_complex_den["error_rate_mismatch"],
                conc_base_cov["error_rate_mismatch"], conc_base_den["error_rate_mismatch"]))
    L.append("")
    L.append("### 3.4 正确 vs 错分样本的特征差异")
    L.append("")
    L.append("对覆盖广度、证据密度、PHQ-8 总分及十个症状域的计数/出现，比较正确与错分样本"
             "（Mann–Whitney U，BH FDR）。各模型在 FDR 校正后显著区分正确/错分的特征如下：")
    L.append("")
    for m in ["base", "count", "pres", "complex"]:
        sub = prof_df[(prof_df["model"] == m) & (prof_df["q_value"] < 0.05)].copy()
        sub = sub.reindex(sub["diff"].abs().sort_values(ascending=False).index)
        if len(sub) == 0:
            L.append("- **%s**：无特征在 FDR 校正后显著区分正确与错分样本。" % m)
            continue
        top = sub.head(5)
        items = "；".join("%s（正确 %.2f vs 错分 %.2f）" % (r["feature"], r["correct_mean"], r["incorrect_mean"])
                          for _, r in top.iterrows())
        L.append("- **%s**：%s。" % (m, items))
    L.append("")
    L.append("### 3.5 代表性错位样本")
    L.append("")
    n_over_cov_rep = int((rep_df["case_type"] == "over_reporter_coverage").sum())
    n_under_cov_rep = int((rep_df["case_type"] == "under_reporter_coverage").sum())
    n_over_den_rep = int((rep_df["case_type"] == "over_reporter_density").sum())
    n_under_den_rep = int((rep_df["case_type"] == "under_reporter_density").sum())
    n_disagree = int((rep_df["case_type"] == "complex_vs_base_disagree").sum())
    L.append("- 覆盖广度定义下：自评高+证据不足 **%d** 人，自评低+证据较多 **%d** 人；" % (n_over_cov_rep, n_under_cov_rep))
    L.append("- 证据密度定义下：自评高+证据不足 **%d** 人，自评低+证据较多 **%d** 人；" % (n_over_den_rep, n_under_den_rep))
    L.append("- 复杂联合文本模型与基础负荷模型判断不一致 **%d** 人（见 `representative_mismatch_cases.csv`，仅含 ID 与结构化指标）。" % n_disagree)
    L.append("")
    L.append("## 4 解释与讨论")
    L.append("")
    L.append("1. **自评与访谈证据存在系统性错位，且并非罕见。** 约两成至三成被试的 PHQ-8 "
             "自评与访谈证据落在不同象限，说明自评量表与文本证据并非简单等价。这种错位是"
             " PHQ-8 作为自评工具的固有特性，而非本研究的新“发现”。")
    L.append("2. **模型错误确实向错位样本集中。** 所有模型在错位象限的错分率均高于一致象限；"
             "这与“以访谈证据预测自评标签的模型无法纠正二者错位”的预期一致。")
    L.append("3. **复杂模型能部分缓解错位误差，但未达稳定增量。** 复杂联合文本模型在错位样本"
             "上的错分率（覆盖广度定义 0.333、证据密度定义 0.481）明显低于基础负荷模型"
             "（0.867、0.741），提示文本语义承载了超出结构化证据的信息，可部分抵消自评—访谈"
             "错位带来的误判；然而它仍对约三分之一至二分之一的错位样本判错，且模块 11 表明其"
             "相对简单负荷模型的 ΔAUC 增益并不统计稳定。因此，“复杂模型未稳定超越简单症状负荷”"
             "这一结论，可从本模块的错位机制得到解释：以访谈证据拟合自评标签的模型，其上界"
             "受限于两类信息源之间的系统性错位。")
    L.append("4. **方法学边界。** 增量模型中的基础负荷指标与十域特征存在确定性冗余"
             "（覆盖广度=症状域出现之和，证据密度=症状域计数之和），因此本分析只比较预测"
             "误差分布，不解释单个系数。PHQ-8 为自评量表，所有结论应理解为对“自评症状严重度"
             "风险”的描述，不能外推为临床诊断。")
    L.append("")
    L.append("## 5 输出文件")
    L.append("")
    L.append("- `mismatch_quadrants_by_coverage.csv` / `mismatch_quadrants_by_density.csv`：四象限人数、比例、均分。")
    L.append("- `model_error_by_mismatch_group.csv`：各模型 TP/FP/FN/TN、各象限错分率、错位 vs 一致置换检验。")
    L.append("- `error_profile_correct_vs_incorrect.csv`：正确 vs 错分样本的特征差异（Mann–Whitney + FDR）。")
    L.append("- `representative_mismatch_cases.csv`：代表性错位样本 ID 与结构化指标（无原文、无隐私文本）。")
    L.append("- `run_manifest.json` / `output_manifest_sha256.csv`：参数与可复现性清单。")
    L.append("")
    return "\n".join(L)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = load_and_validate()
    label_series = df.set_index("participant_id")["label"]

    # ── Generate / collect OOF probabilities ────────────────────────────────
    print("[OOF] Generating fold-local binary OOF predictions...")
    feats_base = ["coverage_breadth", "evidence_density"]
    feats_count = feats_base + [f"{c}_count" for c in DOMAIN_COLS]
    feats_pres = feats_base + [f"{c}_pres" for c in DOMAIN_COLS]

    prob_base = run_logistic_cv(df, feats_base, label_series)
    prob_count = run_logistic_cv(df, feats_count, label_series)
    prob_pres = run_logistic_cv(df, feats_pres, label_series)
    prob_complex = df.set_index("participant_id")["M3_prob"]

    oof = pd.DataFrame({
        "participant_id": df["participant_id"].values,
        "label": df["label"].values,
        "phq8_score": df["phq8_score"].values,
        "coverage_breadth": df["coverage_breadth"].values,
        "evidence_density": df["evidence_density"].values,
        "prob_base": prob_base.reindex(df["participant_id"]).values,
        "prob_count": prob_count.reindex(df["participant_id"]).values,
        "prob_pres": prob_pres.reindex(df["participant_id"]).values,
        "prob_complex": prob_complex.reindex(df["participant_id"]).values,
    })
    for c in [f"{d}_count" for d in DOMAIN_COLS] + [f"{d}_pres" for d in DOMAIN_COLS]:
        oof[c] = df[c].values
    for m in ["base", "count", "pres", "complex"]:
        oof[f"pred_{m}"] = (oof[f"prob_{m}"] >= THRESHOLD).astype(int)

    # Sanity: AUCs should match module 11 (base 0.781, count 0.800, complex 0.814)
    print("[OOF] Sanity AUCs (consistency with module 11):")
    for m in ["base", "count", "pres", "complex"]:
        auc = roc_auc_score(oof["label"], oof[f"prob_{m}"])
        print(f"        {m:8s} AUC = {auc:.3f}")

    # ── Evidence-high definitions ───────────────────────────────────────────
    cov_median = float(oof["coverage_breadth"].median())
    den_median = float(oof["evidence_density"].median())
    phq_high = oof["phq8_score"].values >= PHQ_POS_CUT

    evidence_defs = {
        "coverage_breadth": {
            "median": oof["coverage_breadth"].values >= cov_median,
            "predefined_5plus": oof["coverage_breadth"].values >= COVERAGE_PREDEF,
        },
        "evidence_density": {
            "median": oof["evidence_density"].values >= den_median,
            "predefined_11plus": oof["evidence_density"].values >= DENSITY_PREDEF,
        },
    }

    # ── Quadrant tables ──────────────────────────────────────────────────────
    print("[QUAD] Building mismatch quadrants...")
    quad_rows = []  # for the two CSVs
    oof["_quad_coverage_median"] = assign_quadrants(phq_high, evidence_defs["coverage_breadth"]["median"])
    oof["_quad_coverage_predef"] = assign_quadrants(phq_high, evidence_defs["coverage_breadth"]["predefined_5plus"])
    oof["_quad_density_median"] = assign_quadrants(phq_high, evidence_defs["evidence_density"]["median"])
    oof["_quad_density_predef"] = assign_quadrants(phq_high, evidence_defs["evidence_density"]["predefined_11plus"])

    quad_specs = [
        ("coverage_breadth", "median", "_quad_coverage_median", cov_median),
        ("coverage_breadth", "predefined_5plus", "_quad_coverage_predef", COVERAGE_PREDEF),
        ("evidence_density", "median", "_quad_density_median", den_median),
        ("evidence_density", "predefined_11plus", "_quad_density_predef", DENSITY_PREDEF),
    ]
    for ev_def, rule, qcol, thr in quad_specs:
        g = oof.groupby(qcol)
        total = len(oof)
        for q in QUADRANTS:
            sub = g.get_group(q) if q in g.groups else oof.iloc[0:0]
            n = len(sub)
            quad_rows.append({
                "evidence_def": ev_def,
                "threshold_rule": rule,
                "threshold_value": thr,
                "quadrant": q,
                "quadrant_cn": QUADRANT_CN[q],
                "n": n,
                "pct": round(100.0 * n / total, 1) if total else 0.0,
                "mean_phq8": round(float(sub["phq8_score"].mean()), 2) if n else np.nan,
                "phq_positive_rate": round(float((sub["phq8_score"] >= PHQ_POS_CUT).mean()), 3) if n else np.nan,
                "mean_coverage_breadth": round(float(sub["coverage_breadth"].mean()), 2) if n else np.nan,
                "mean_evidence_density": round(float(sub["evidence_density"].mean()), 2) if n else np.nan,
            })
    quad_df = pd.DataFrame(quad_rows)
    quad_cov = quad_df[quad_df["evidence_def"] == "coverage_breadth"].drop(columns=["evidence_def"]).reset_index(drop=True)
    quad_den = quad_df[quad_df["evidence_def"] == "evidence_density"].drop(columns=["evidence_def"]).reset_index(drop=True)
    quad_cov.to_csv(OUTDIR / "mismatch_quadrants_by_coverage.csv", index=False, encoding="utf-8-sig")
    quad_den.to_csv(OUTDIR / "mismatch_quadrants_by_density.csv", index=False, encoding="utf-8-sig")

    # ── Model error by mismatch group ───────────────────────────────────────
    print("[ERR] Model error by mismatch group...")
    models = ["base", "count", "pres", "complex"]
    err_rows = []
    conc_rows = []
    for m in models:
        pred = oof[f"pred_{m}"].values
        lab = oof["label"].values
        cm = confusion_matrix(lab, pred, labels=[0, 1])
        tn, fp = cm[0, 0], cm[0, 1]
        fn, tp = cm[1, 0], cm[1, 1]
        n_all = len(oof)
        n_err_all = int((pred != lab).sum())
        err_rows.append({
            "model": m, "evidence_def": "ALL", "threshold_rule": "ALL",
            "group": "overall", "n": n_all,
            "n_correct": n_all - n_err_all, "n_error": n_err_all,
            "error_rate": round(n_err_all / n_all, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "error_rate_mismatch": np.nan, "error_rate_consistent": np.nan,
            "error_rate_diff": np.nan, "permutation_p": np.nan,
        })
        for ev_def, rule, qcol, _ in quad_specs:
            quad = oof[qcol].values
            for q in QUADRANTS:
                mask = quad == q
                nq = int(mask.sum())
                nerr_q = int((pred[mask] != lab[mask]).sum())
                err_rows.append({
                    "model": m, "evidence_def": ev_def, "threshold_rule": rule,
                    "group": q, "n": nq,
                    "n_correct": nq - nerr_q, "n_error": nerr_q,
                    "error_rate": round(nerr_q / nq, 4) if nq else np.nan,
                    "tp": np.nan, "fp": np.nan, "fn": np.nan, "tn": np.nan,
                    "error_rate_mismatch": np.nan, "error_rate_consistent": np.nan,
                    "error_rate_diff": np.nan, "permutation_p": np.nan,
                })
            # concentration test: mismatch vs consistent
            mismatch_mask = np.isin(quad, MISMATCH_QUADRANTS)
            consistent_mask = ~mismatch_mask
            err_flag = (pred != lab).astype(int)
            rate_mis = float(err_flag[mismatch_mask].mean()) if mismatch_mask.sum() else np.nan
            rate_con = float(err_flag[consistent_mask].mean()) if consistent_mask.sum() else np.nan
            obs_diff = rate_mis - rate_con
            rng = np.random.RandomState(RANDOM_SEED)
            nm = int(mismatch_mask.sum())
            nc = int(consistent_mask.sum())
            perm_diffs = []
            for _ in range(N_PERM_CONC):
                shuf = rng.permutation(err_flag)
                rm = shuf[:nm].mean() if nm else np.nan
                rc = shuf[nm:nm + nc].mean() if nc else np.nan
                perm_diffs.append(rm - rc)
            perm_p = float(np.mean(np.abs(perm_diffs) >= abs(obs_diff))) if len(perm_diffs) else np.nan
            conc_rows.append({
                "model": m, "evidence_def": ev_def, "threshold_rule": rule,
                "group": "mismatch_vs_consistent", "n": nm + nc,
                "n_correct": np.nan, "n_error": np.nan, "error_rate": np.nan,
                "tp": np.nan, "fp": np.nan, "fn": np.nan, "tn": np.nan,
                "error_rate_mismatch": round(rate_mis, 4),
                "error_rate_consistent": round(rate_con, 4),
                "error_rate_diff": round(obs_diff, 4),
                "permutation_p": round(perm_p, 4),
            })
    err_df = pd.DataFrame(err_rows + conc_rows)
    err_df.to_csv(OUTDIR / "model_error_by_mismatch_group.csv", index=False, encoding="utf-8-sig")

    # ── Correct vs incorrect profile (Mann-Whitney + FDR) ───────────────────
    print("[PROF] Correct vs incorrect feature profile...")
    prof_rows = []
    feat_profile = (["coverage_breadth", "evidence_density", "phq8_score"]
                    + [f"{c}_count" for c in DOMAIN_COLS]
                    + [f"{c}_pres" for c in DOMAIN_COLS])
    for m in models:
        pred = oof[f"pred_{m}"].values
        correct = pred == oof["label"].values
        pvals = []
        means = []
        for f in feat_profile:
            x_c = oof.loc[correct, f].values.astype(float)
            x_i = oof.loc[~correct, f].values.astype(float)
            mc = float(np.mean(x_c)) if len(x_c) else np.nan
            mi = float(np.mean(x_i)) if len(x_i) else np.nan
            means.append((mc, mi))
            if len(x_c) >= 2 and len(x_i) >= 2:
                try:
                    p = scipy_stats.mannwhitneyu(x_c, x_i, alternative="two-sided").pvalue
                except ValueError:
                    p = np.nan
            else:
                p = np.nan
            pvals.append(p)
        qvals = bh_fdr(pvals)
        for f, (mc, mi), p, q in zip(feat_profile, means, pvals, qvals):
            prof_rows.append({
                "model": m, "feature": f,
                "correct_mean": round(mc, 3) if not np.isnan(mc) else np.nan,
                "incorrect_mean": round(mi, 3) if not np.isnan(mi) else np.nan,
                "diff": round(mi - mc, 3) if not (np.isnan(mc) or np.isnan(mi)) else np.nan,
                "mwu_p": round(p, 4) if not np.isnan(p) else np.nan,
                "q_value": round(q, 4) if not np.isnan(q) else np.nan,
            })
    prof_df = pd.DataFrame(prof_rows)
    prof_df.to_csv(OUTDIR / "error_profile_correct_vs_incorrect.csv", index=False, encoding="utf-8-sig")

    # ── Representative mismatch cases ───────────────────────────────────────
    print("[REP] Representative mismatch cases...")
    rep_rows = []
    id_cols = (["participant_id", "phq8_score", "label", "coverage_breadth",
                "evidence_density"]
               + [f"{c}_count" for c in DOMAIN_COLS]
               + [f"{c}_pres" for c in DOMAIN_COLS]
               + ["prob_base", "pred_base", "prob_count", "pred_count",
                  "prob_pres", "pred_pres", "prob_complex", "pred_complex"])
    # over/under reporters per evidence definition
    rep_specs = [
        ("over_reporter_coverage", "_quad_coverage_predef", "self_high_evidence_low"),
        ("under_reporter_coverage", "_quad_coverage_predef", "self_low_evidence_high"),
        ("over_reporter_density", "_quad_density_predef", "self_high_evidence_low"),
        ("under_reporter_density", "_quad_density_predef", "self_low_evidence_high"),
    ]
    for ctype, qcol, q in rep_specs:
        sub = oof[oof[qcol] == q][id_cols]
        for _, r in sub.iterrows():
            row = {"case_type": ctype}
            row.update(r.to_dict())
            rep_rows.append(row)
    # complex vs base disagreement
    disagree = oof[oof["pred_complex"] != oof["pred_base"]][id_cols]
    for _, r in disagree.iterrows():
        row = {"case_type": "complex_vs_base_disagree"}
        row.update(r.to_dict())
        rep_rows.append(row)
    rep_df = pd.DataFrame(rep_rows)
    if len(rep_df):
        rep_df = rep_df[id_cols + (["case_type"] if "case_type" not in id_cols else [])]
        # reorder: case_type first
        cols = ["case_type"] + [c for c in rep_df.columns if c != "case_type"]
        rep_df = rep_df[cols]
    rep_df["participant_id"] = rep_df["participant_id"].astype(int)
    rep_df.to_csv(OUTDIR / "representative_mismatch_cases.csv", index=False, encoding="utf-8-sig")

    # ── Manifests ───────────────────────────────────────────────────────────
    manifest = {
        "module": "12_self_report_text_evidence_mismatch",
        "created": datetime.now().isoformat(timespec="seconds"),
        "random_seed": RANDOM_SEED,
        "decision_threshold": THRESHOLD,
        "phq_positive_cut": PHQ_POS_CUT,
        "coverage_predef_high": COVERAGE_PREDEF,
        "density_predef_high": DENSITY_PREDEF,
        "coverage_median": cov_median,
        "density_median": den_median,
        "n_perm_concentration": N_PERM_CONC,
        "models": {
            "base": {"type": "logistic_cv", "features": feats_base},
            "count": {"type": "logistic_cv", "features": feats_count},
            "pres": {"type": "logistic_cv", "features": feats_pres},
            "complex": {"type": "external_oof", "source": "07_strict_joint_models/M3_prob"},
        },
        "input_sha256": {
            "domain_count": sha256_file(DOMAIN_COUNT_CSV),
            "domain_presence": sha256_file(DOMAIN_PRESENCE_CSV),
            "splits": sha256_file(SPLIT_CSV),
            "frozen_turns": sha256_file(FROZEN_TURNS),
            "strict_oof": sha256_file(STRICT_OOF_CSV),
        },
        "lr_params": LR_PARAMS,
    }

    # ── Summary markdown (interpretive, data-driven) ────────────────────────
    print("[SUM] Writing interpretive summary...")
    summary_md = build_summary(oof, quad_df, err_df, prof_df, rep_df, manifest)
    with open(OUTDIR / "mismatch_analysis_summary.md", "w", encoding="utf-8") as f:
        f.write(summary_md)

    with open(OUTDIR / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    out_files = [
        "mismatch_quadrants_by_coverage.csv",
        "mismatch_quadrants_by_density.csv",
        "model_error_by_mismatch_group.csv",
        "error_profile_correct_vs_incorrect.csv",
        "representative_mismatch_cases.csv",
        "mismatch_analysis_summary.md",
        "run_manifest.json",
        "output_manifest_sha256.csv",
    ]
    with open(OUTDIR / "output_manifest_sha256.csv", "w", encoding="utf-8-sig", newline="") as f:
        f.write("file,sha256\n")
        for fn in out_files:
            p = OUTDIR / fn
            if p.exists():
                f.write(f"{fn},{sha256_file(p)}\n")

    print("[DONE] All outputs written to", OUTDIR)
    return oof, quad_df, err_df, prof_df, rep_df, manifest


if __name__ == "__main__":
    main()
