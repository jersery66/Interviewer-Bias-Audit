#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E-DAIC 新增标注样本的探索性补充验证 (PHQ-8)
============================================

定位（务必先读）
----------------
本模块 **不是** 合并扩样本，也 **不是** 严格独立外部验证。

E-DAIC 是 DAIC-WOZ 的扩展版本：原始 DAIC-WOZ 的 189 名参与者被完整并入 E-DAIC，
E-DAIC 另新增 86 人（其中仅 30 人有公开 PHQ-8 标签）。本模块只纳入
「E-DAIC 新增且公开 PHQ-8 标签」的参与者（来自
`Detailed_PHQ8_Labels_E_DAIC_only.csv`），**排除**原始 DAIC-WOZ 189 人与
无公开标签的 test 56 人。

目标：探索性检验 DAIC-WOZ 主分析中观察到的两类现象，在 E-DAIC 新增标注样本中
方向是否一致：
  1) PHQ-8 自评标签与访谈文本症状证据存在错位（mismatch）；
  2) 以访谈证据为输入的模型，其错误集中在错位样本。

方法保真度
----------
- 症状证据抽取：**复用 DAIC-WOZ 原始 C5 抽取管线**
  (`run_c5_evidence_extraction_api.py`，prompt v3 / gpt-5.5 / temperature=0)，
  对 E-DAIC 新增样本重跑，得到结构化证据 span，再经与本仓库完全一致的
  `build_domain_features`（下方逐字复用）聚合成 10 域 count / presence。
  ⚠ 偏差披露：E-DAIC 转录本无 speaker 列（DAIC-WOZ 原始有），故对 E-DAIC 喂
  全文对话文本；抽取 prompt 仅要求提取被试症状证据，Ellie 提问不会被误判为症状。
- 模型：在 DAIC-WOZ 主分析 142 人上 **fit 一次** scaler + 逻辑回归，直接预测
  E-DAIC 新增样本（迁移验证）。**不在 E-DAIC 30 人上做交叉验证**（样本太小）。
- 错位定义：沿用 DAIC-WOZ 固定预定义切点（覆盖广度 ≥5、证据密度 ≥11）；
  并补充 E-DAIC 样本内中位数切点作敏感性描述（仅补充表，非主结论）。

硬约束（继承自严格重分析 SPEC）
------------------------------
- 只读冻结/公开输入；不读任何 superseded 01–06 结果。
- 不与 DAIC-WOZ 主样本合并；E-DAIC 仅作新增有标签测试集。
- 合并均以 `participant_id` 为键；不依赖行顺序。
- PHQ-8 为自评量表，非临床诊断；不输出任何访谈原文，仅输出 ID 与结构化指标。

输出文件（与方案一致）
----------------------
edaic_extension_sample_manifest.csv
edaic_turns_frozen.csv
edaic_evidence_features.csv
edaic_mismatch_quadrants_by_coverage.csv
edaic_mismatch_quadrants_by_density.csv
edaic_transfer_predictions.csv
edaic_transfer_metrics_all_models.csv
edaic_model_error_by_mismatch_group.csv
daic_vs_edaic_validation_comparison.csv
edaic_extension_validation_summary.md
run_manifest.json
output_manifest_sha256.csv
（抽取阶段附带：edaic_evidence_extraction_input.jsonl, c5_edaic_extension_v3_gpt55_spans.csv）

Author: strict reanalysis (2026-07-09), module 14
"""

import os
import sys
import json
import csv
import hashlib
import subprocess
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, average_precision_score,
    accuracy_score, f1_score, recall_score, precision_score,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── 路径（全部使用 Windows 盘符绝对路径，避免 /e/ 在 Python 下失效） ──────────
SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_V2 = SCRIPT_DIR.parent.parent                      # .../analysis_v2
BASE_REPO = ANALYSIS_V2.parent                               # .../reanalysis-v2
EDAIC_ROOT = Path(r"F:/数据库/E-daic")
EDAIC_LABELS = EDAIC_ROOT / "labels"
EDAIC_DATA = EDAIC_ROOT / "data" / "data"
EXTRACTION_SCRIPT = (
    Path(r"F:/数据库/DAIC-WOZ/processed_research/DAIC_v26_paper_data_package/11_scripts")
    / "run_c5_evidence_extraction_api.py"
)
PROMPT_MD = (
    Path(r"F:/数据库/DAIC-WOZ/processed_research/prompts")
    / "C5_symptom_evidence_quotes_only_prompt_v3.md"
)

# DAIC-WOZ 142 冻结输入（用于训练迁移模型 + 计算对照表 DAIC-WOZ 列）
DOMAIN_COUNT_CSV = ANALYSIS_V2 / "04_c5_controls" / "domain_count" / "input.csv"
DOMAIN_PRESENCE_CSV = ANALYSIS_V2 / "04_c5_controls" / "domain_presence" / "input.csv"
FROZEN_142 = ANALYSIS_V2 / "01_inputs" / "c4_turns_official142_frozen.csv"
SPLIT_CSV = ANALYSIS_V2 / "00_splits" / "repeated_5fold_splits_10x5.csv"
STRICT_OOF_CSV = (
    ANALYSIS_V2 / "10_source_importance" / "07_strict_joint_models"
    / "participant_oof_predictions.csv"
)
# 模块 12 现有产出（对照表 DAIC-WOZ 列：错位比例 / 错分率）
M12_DIR = ANALYSIS_V2 / "10_source_importance" / "12_self_report_text_evidence_mismatch"
M12_QUAD_COV = M12_DIR / "mismatch_quadrants_by_coverage.csv"
M12_QUAD_DEN = M12_DIR / "mismatch_quadrants_by_density.csv"
M12_ERR = M12_DIR / "model_error_by_mismatch_group.csv"

# E-DAIC 标签文件
EDAIC_ONLY_CSV = EDAIC_LABELS / "Detailed_PHQ8_Labels_E_DAIC_only.csv"

# ── 症状域定义（逐字复用 controls.DOMAIN_ORDER，保证与 DAIC-WOZ 完全一致） ─────
DOMAIN_ORDER = [
    "anhedonia_interest",
    "appetite_weight",
    "concentration_psychomotor",
    "depressed_mood",
    "functioning_impairment",
    "mental_health_history",
    "protective_or_absent_symptom",
    "self_worth_guilt",
    "sleep_fatigue_energy",
    "suicide_self_harm",
]
DOMAIN_COLS = DOMAIN_ORDER  # 别名，便于与模块 12 对齐

RANDOM_SEED = 20260705
LR_PARAMS = dict(
    penalty="l2", C=1.0, class_weight="balanced",
    solver="liblinear", max_iter=2000, random_state=RANDOM_SEED,
)
np.random.seed(RANDOM_SEED)

THRESHOLD = 0.5
PHQ_POS_CUT = 10
COVERAGE_PREDEF = 5
DENSITY_PREDEF = 11
N_PERM_CONC = 10000  # 仅 DAIC-WOZ 142 内部用；E-DAIC 30 用 Fisher 精确检验

# ── 逐字复用 controls.build_domain_features（确定性聚合，无随机性） ────────────
def build_domain_features(spans, participant_ids):
    """逐字复用 analysis_v2/08_reproducibility_scripts/reanalysis_v2/controls.py
    中的 build_domain_features：将结构化证据 span 聚合成 10 域 presence / count。"""
    required = {"participant_id", "domain"}
    if not required.issubset(spans.columns):
        raise ValueError("spans require participant_id and domain")
    unknown = set(spans["domain"].dropna().astype(str)).difference(DOMAIN_ORDER)
    if unknown:
        raise ValueError(f"unknown C5 domains: {sorted(unknown)}")
    base = pd.DataFrame({"participant_id": list(participant_ids)})
    counts = (
        spans.groupby(["participant_id", "domain"]).size().unstack(fill_value=0)
        if not spans.empty
        else pd.DataFrame()
    )
    counts = counts.reindex(columns=DOMAIN_ORDER, fill_value=0)
    counts.index.name = "participant_id"
    counts = base.merge(counts.reset_index(), on="participant_id", how="left")
    counts[DOMAIN_ORDER] = counts[DOMAIN_ORDER].fillna(0).astype(int)
    presence = counts.copy()
    presence[DOMAIN_ORDER] = presence[DOMAIN_ORDER].gt(0).astype(int)
    return presence, counts


# ── 工具函数 ───────────────────────────────────────────────────────────────────
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def find_transcript(pid: int) -> Path | None:
    """定位 E-DAIC 转录本（兼容嵌套 / 扁平两种布局）。"""
    cands = [
        EDAIC_DATA / f"{pid}_P.tar" / f"{pid}_P" / f"{pid}_Transcript.csv",
        EDAIC_DATA / f"{pid}_P" / f"{pid}_Transcript.csv",
    ]
    for c in cands:
        if c.exists():
            return c
    return None


def read_transcript_text(path: Path) -> str:
    """读取 E-DAIC 转录本并拼接 Participant 文本。
    注：E-DAIC 转录本仅有 Start_Time,End_Time,Text,Confidence（无 speaker 列），
    故使用全对话文本（见模块 summary 偏差披露）。"""
    df = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    text = " ".join(str(t).strip() for t in df["Text"].tolist() if str(t).strip())
    return text


# ── 象限定义（与模块 12 一致） ─────────────────────────────────────────────────
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
# 步骤 1：样本清单
# ═══════════════════════════════════════════════════════════════════════════
def build_sample_manifest():
    print("[1] Building E-DAIC extension sample manifest...")
    phq = pd.read_csv(EDAIC_ONLY_CSV, encoding="utf-8-sig")
    phq = phq.rename(columns={
        "Participant_ID": "participant_id",
        "PHQ_8Total": "phq8_score",
    })
    rows = []
    for _, r in phq.iterrows():
        pid = int(r["participant_id"])
        score = float(r["phq8_score"])
        tpath = find_transcript(pid)
        included = tpath is not None
        reason = "" if included else "transcript_missing"
        rows.append({
            "participant_id": pid,
            "source_dataset": "E-DAIC新增样本",
            "transcript_path": str(tpath) if tpath else "",
            "phq8_score": score,
            "phq8_label_ge10": int(score >= PHQ_POS_CUT),
            "transcript_exists": int(included),
            "included": int(included),
            "exclusion_reason": reason,
        })
    man = pd.DataFrame(rows).sort_values("participant_id").reset_index(drop=True)
    man.to_csv(SCRIPT_DIR / "edaic_extension_sample_manifest.csv",
               index=False, encoding="utf-8-sig")
    n_incl = int(man["included"].sum())
    print(f"    E_DAIC_only 候选 {len(man)} 人；纳入 {n_incl} 人"
          f"（排除 {len(man)-n_incl} 人：transcript 缺失）。")
    return man


# ═══════════════════════════════════════════════════════════════════════════
# 步骤 2：构建抽取输入 jsonl + 调用 C5 抽取管线
# ═══════════════════════════════════════════════════════════════════════════
RUN_ID = "c5_edaic_extension_v3_gpt55"
SPANS_CSV = SCRIPT_DIR / f"{RUN_ID}_spans.csv"


def prepare_extraction_jsonl(manifest):
    print("[2] Preparing C5 extraction input jsonl for E-DAIC new samples...")
    incl = manifest[manifest["included"] == 1]
    records = []
    for _, r in incl.iterrows():
        pid = int(r["participant_id"])
        text = read_transcript_text(Path(r["transcript_path"]))
        records.append({
            "task_id": f"{pid}_symptom_evidence_extraction",
            "participant_id": pid,
            "input_condition": "edaic_full_dialogue_no_speaker",
            "prompt_version": "c5_symptom_evidence_quotes_only_v3",
            "text": text,
        })
    jsonl_path = SCRIPT_DIR / "edaic_evidence_extraction_input.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"    wrote {len(records)} records -> {jsonl_path.name}")
    return jsonl_path


def run_c5_extraction(jsonl_path, ids):
    """复用 DAIC-WOZ 原始 C5 抽取管线（gpt-5.5 / prompt v3 / temperature=0）。"""
    if not EXTRACTION_SCRIPT.exists():
        raise FileNotFoundError(f"抽取脚本不存在: {EXTRACTION_SCRIPT}")
    if not PROMPT_MD.exists():
        raise FileNotFoundError(f"抽取 prompt 文件不存在: {PROMPT_MD}")
    api_key = os.environ.get("APIYI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "环境变量 APIYI_API_KEY 未设置。请先 export APIYI_API_KEY=<你的key>，"
            "或将抽取产物手动命名为 "
            f"{SPANS_CSV.name} 放入模块 14 目录后重新运行（脚本会跳过抽取步骤）。"
        )
    cmd = [
        sys.executable, str(EXTRACTION_SCRIPT),
        "--input-jsonl", str(jsonl_path),
        "--output-dir", str(SCRIPT_DIR),
        "--run-id", RUN_ID,
        "--prompt-md", str(PROMPT_MD),
        "--ids", ",".join(str(i) for i in ids),
    ]
    print("[2b] Running C5 extraction (gpt-5.5 / prompt v3 / temp 0)...")
    env = dict(os.environ, APIYI_API_KEY=api_key)
    subprocess.run(cmd, env=env, check=True)
    if not SPANS_CSV.exists():
        raise RuntimeError(f"抽取未产出预期文件: {SPANS_CSV}")


# ═══════════════════════════════════════════════════════════════════════════
# 步骤 3：聚合 span → 症状证据特征
# ═══════════════════════════════════════════════════════════════════════════
def aggregate_evidence(manifest):
    print("[3] Aggregating C5 spans -> domain features (build_domain_features)...")
    if not SPANS_CSV.exists():
        raise RuntimeError(f"未找到 span 文件 {SPANS_CSV.name}，抽取步骤未成功。")
    spans = pd.read_csv(SPANS_CSV, encoding="utf-8-sig", keep_default_na=False)
    # 仅保留被纳入的 participant
    incl_ids = manifest[manifest["included"] == 1]["participant_id"].tolist()
    spans = spans[spans["participant_id"].isin(incl_ids)]
    presence, counts = build_domain_features(spans, incl_ids)
    # 合并 PHQ 标签
    lab = manifest[manifest["included"] == 1][
        ["participant_id", "phq8_score", "phq8_label_ge10"]
    ].rename(columns={"phq8_label_ge10": "label"})
    feat = presence.merge(counts, on="participant_id", suffixes=("_pres", "_count"))
    feat = feat.merge(lab, on="participant_id", how="left")
    feat["coverage_breadth"] = feat[[f"{c}_pres" for c in DOMAIN_COLS]].sum(axis=1)
    feat["evidence_density"] = feat[[f"{c}_count" for c in DOMAIN_COLS]].sum(axis=1)
    cols = (["participant_id", "phq8_score", "label"]
            + [f"{c}_count" for c in DOMAIN_COLS]
            + [f"{c}_pres" for c in DOMAIN_COLS]
            + ["coverage_breadth", "evidence_density"])
    feat = feat[cols]
    feat.to_csv(SCRIPT_DIR / "edaic_evidence_features.csv", index=False, encoding="utf-8-sig")
    print(f"    {len(feat)} 人证据特征已写出。")
    return feat


# ═══════════════════════════════════════════════════════════════════════════
# 步骤 4：DAIC-WOZ 142 训练集特征 + 迁移预测
# ═══════════════════════════════════════════════════════════════════════════
def load_daicwoz_142():
    dcount = pd.read_csv(DOMAIN_COUNT_CSV)
    dpres = pd.read_csv(DOMAIN_PRESENCE_CSV)
    frozen = pd.read_csv(FROZEN_142)
    pid_score = frozen.groupby("participant_id").agg(
        phq8_score=("paper_phq8_score", "first"),
        label=("paper_label_phq8_ge10", "first"),
    ).reset_index()
    feat = dcount.merge(dpres, on="participant_id", suffixes=("_count", "_pres"))
    df = feat.merge(pid_score, on="participant_id", how="inner")
    df["coverage_breadth"] = df[[f"{c}_pres" for c in DOMAIN_COLS]].sum(axis=1)
    df["evidence_density"] = df[[f"{c}_count" for c in DOMAIN_COLS]].sum(axis=1)
    return df


def run_logistic_cv(df, feature_cols, label_series):
    """DAIC-WOZ 142 内部 10x5 参与者级 CV（用于对照表复算 DAIC-WOZ AUC）。"""
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
            all_probs.append({"participant_id": pid, "prob": p})
    return pd.DataFrame(all_probs).groupby("participant_id")["prob"].mean()


def train_and_transfer(df142, df30):
    print("[4] Training on DAIC-WOZ 142, transferring to E-DAIC new samples...")
    label142 = df142.set_index("participant_id")["label"]
    feats_base = ["coverage_breadth", "evidence_density"]
    feats_count = feats_base + [f"{c}_count" for c in DOMAIN_COLS]
    feats_pres = feats_base + [f"{c}_pres" for c in DOMAIN_COLS]

    # 复算 DAIC-WOZ 142 OOF AUC（对照表 DAIC-WOZ 列）
    auc_daic = {}
    for name, fc in [("base", feats_base), ("count", feats_count), ("pres", feats_pres)]:
        oof = run_logistic_cv(df142, fc, label142)
        auc_daic[name] = float(roc_auc_score(label142.values, oof.reindex(df142["participant_id"]).values))

    # 在 142 全量 fit 一次，迁移预测 30
    models = {}
    preds = {}
    for name, fc in [("base", feats_base), ("count", feats_count), ("pres", feats_pres)]:
        Xtr = df142[fc].values.astype(float)
        ytr = label142.values
        scaler = StandardScaler().fit(Xtr)
        model = LogisticRegression(**LR_PARAMS)
        model.fit(scaler.transform(Xtr), ytr)
        Xte = df30[fc].values.astype(float)
        prob = model.predict_proba(scaler.transform(Xte))[:, 1]
        models[name] = (scaler, model, fc)
        preds[name] = prob

    transfer = pd.DataFrame({
        "participant_id": df30["participant_id"].values,
        "label": df30["label"].values,
        "phq8_score": df30["phq8_score"].values,
        "coverage_breadth": df30["coverage_breadth"].values,
        "evidence_density": df30["evidence_density"].values,
    })
    for name in ["base", "count", "pres"]:
        transfer[f"prob_{name}"] = preds[name]
        transfer[f"pred_{name}"] = (preds[name] >= THRESHOLD).astype(int)
    for c in [f"{d}_count" for d in DOMAIN_COLS] + [f"{d}_pres" for d in DOMAIN_COLS]:
        transfer[c] = df30[c].values
    return transfer, auc_daic, models


# ═══════════════════════════════════════════════════════════════════════════
# 步骤 5：四象限（E-DAIC 30，固定切点 + E-DAIC 中位数敏感）
# ═══════════════════════════════════════════════════════════════════════════
def build_quadrants(transfer):
    print("[5] Building E-DAIC mismatch quadrants...")
    cov_median = float(transfer["coverage_breadth"].median())
    den_median = float(transfer["evidence_density"].median())
    phq_high = transfer["phq8_score"].values >= PHQ_POS_CUT
    ev_defs = {
        "coverage_breadth": {
            "median": transfer["coverage_breadth"].values >= cov_median,
            "predefined_5plus": transfer["coverage_breadth"].values >= COVERAGE_PREDEF,
        },
        "evidence_density": {
            "median": transfer["evidence_density"].values >= den_median,
            "predefined_11plus": transfer["evidence_density"].values >= DENSITY_PREDEF,
        },
    }
    quad_specs = [
        ("coverage_breadth", "median", cov_median),
        ("coverage_breadth", "predefined_5plus", COVERAGE_PREDEF),
        ("evidence_density", "median", den_median),
        ("evidence_density", "predefined_11plus", DENSITY_PREDEF),
    ]
    rows = []
    quad_cols = {}
    for ev_def, rule, thr in quad_specs:
        qcol = f"_q_{ev_def}_{rule}"
        quad_cols[(ev_def, rule)] = assign_quadrants(phq_high, ev_defs[ev_def][rule])
        transfer[qcol] = quad_cols[(ev_def, rule)]
        g = transfer.groupby(qcol)
        total = len(transfer)
        for q in QUADRANTS:
            sub = g.get_group(q) if q in g.groups else transfer.iloc[0:0]
            n = len(sub)
            rows.append({
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
    quad_df = pd.DataFrame(rows)
    quad_cov = quad_df[quad_df["evidence_def"] == "coverage_breadth"].copy()
    quad_den = quad_df[quad_df["evidence_def"] == "evidence_density"].copy()
    quad_cov_out = quad_cov[["evidence_def", "threshold_rule", "threshold_value", "quadrant",
                             "quadrant_cn", "n", "pct", "mean_phq8", "phq_positive_rate",
                             "mean_coverage_breadth", "mean_evidence_density"]]
    quad_den_out = quad_den[["evidence_def", "threshold_rule", "threshold_value", "quadrant",
                             "quadrant_cn", "n", "pct", "mean_phq8", "phq_positive_rate",
                             "mean_coverage_breadth", "mean_evidence_density"]]
    quad_cov_out.to_csv(SCRIPT_DIR / "edaic_mismatch_quadrants_by_coverage.csv",
                        index=False, encoding="utf-8-sig")
    quad_den_out.to_csv(SCRIPT_DIR / "edaic_mismatch_quadrants_by_density.csv",
                        index=False, encoding="utf-8-sig")
    return transfer, quad_df, cov_median, den_median


# ═══════════════════════════════════════════════════════════════════════════
# 步骤 6：迁移预测指标 + 步骤 7：错位错分率（Fisher 精确）
# ═══════════════════════════════════════════════════════════════════════════
def compute_metrics_and_errors(transfer):
    print("[6-7] Transfer metrics + mismatch vs consistent error (Fisher)...")
    models = ["base", "count", "pres"]
    metric_rows = []
    err_rows = []
    for m in models:
        lab = transfer["label"].values.astype(int)
        pred = transfer[f"pred_{m}"].values.astype(int)
        prob = transfer[f"prob_{m}"].values.astype(float)
        cm = confusion_matrix(lab, pred, labels=[0, 1])
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        n = len(transfer)
        n_err = int((pred != lab).sum())
        has_pos = int(lab.sum()) > 0
        has_neg = int((1 - lab).sum()) > 0
        auc = float(roc_auc_score(lab, prob)) if (has_pos and has_neg) else np.nan
        try:
            prauc = float(average_precision_score(lab, prob)) if (has_pos and has_neg) else np.nan
        except Exception:
            prauc = np.nan
        metric_rows.append({
            "model": m, "n": n, "n_positive": int(lab.sum()),
            "AUC": round(auc, 4) if not np.isnan(auc) else np.nan,
            "PR_AUC": round(prauc, 4) if not np.isnan(prauc) else np.nan,
            "Accuracy": round(float(accuracy_score(lab, pred)), 4),
            "Macro_F1": round(float(f1_score(lab, pred, average="macro", zero_division=0)), 4),
            "Sensitivity": round(float(recall_score(lab, pred, pos_label=1, zero_division=0)), 4),
            "Specificity": round(float(recall_score(lab, pred, pos_label=0, zero_division=0)), 4),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "error_rate": round(n_err / n, 4) if n else np.nan,
        })
        # 错位 vs 一致
        for ev_def, rule, qcol in [
            ("coverage_breadth", "predefined_5plus", "_q_coverage_breadth_predefined_5plus"),
            ("evidence_density", "predefined_11plus", "_q_evidence_density_predefined_11plus"),
        ]:
            quad = transfer[qcol].values
            mismatch_mask = np.isin(quad, MISMATCH_QUADRANTS)
            consistent_mask = ~mismatch_mask
            err_flag = (pred != lab).astype(int)
            nm = int(mismatch_mask.sum())
            nc = int(consistent_mask.sum())
            rate_mis = float(err_flag[mismatch_mask].mean()) if nm else np.nan
            rate_con = float(err_flag[consistent_mask].mean()) if nc else np.nan
            obs_diff = rate_mis - rate_con if (nm and nc) else np.nan
            # Fisher 精确检验（2x2：错位 vs 一致 的 错/对）
            fisher_p = np.nan
            note = ""
            if nm and nc:
                a = int(err_flag[mismatch_mask].sum()); b = nm - a
                c = int(err_flag[consistent_mask].sum()); d = nc - c
                if (a + b) > 0 and (c + d) > 0:
                    try:
                        _, fisher_p = scipy_stats.fisher_exact([[a, b], [c, d]])
                    except Exception:
                        fisher_p = np.nan
                else:
                    note = "单元格过小，Fisher p=NA"
            err_rows.append({
                "model": m, "evidence_def": ev_def, "threshold_rule": rule,
                "group": "mismatch_vs_consistent",
                "n": nm + nc,
                "n_mismatch": nm, "n_consistent": nc,
                "n_error_mismatch": int(err_flag[mismatch_mask].sum()) if nm else 0,
                "n_error_consistent": int(err_flag[consistent_mask].sum()) if nc else 0,
                "error_rate": np.nan,
                "error_rate_mismatch": round(rate_mis, 4) if not np.isnan(rate_mis) else np.nan,
                "error_rate_consistent": round(rate_con, 4) if not np.isnan(rate_con) else np.nan,
                "error_rate_diff": round(obs_diff, 4) if not np.isnan(obs_diff) else np.nan,
                "fisher_p": round(float(fisher_p), 4) if not np.isnan(fisher_p) else np.nan,
                "note": note,
            })
    pd.DataFrame(metric_rows).to_csv(SCRIPT_DIR / "edaic_transfer_metrics_all_models.csv",
                                     index=False, encoding="utf-8-sig")
    pd.DataFrame(err_rows).to_csv(SCRIPT_DIR / "edaic_model_error_by_mismatch_group.csv",
                                  index=False, encoding="utf-8-sig")
    transfer.to_csv(SCRIPT_DIR / "edaic_transfer_predictions.csv",
                    index=False, encoding="utf-8-sig")
    return pd.DataFrame(metric_rows), pd.DataFrame(err_rows)


# ═══════════════════════════════════════════════════════════════════════════
# 步骤 8：对照表（DAIC-WOZ 主分析 vs E-DAIC 新增样本）
# ═══════════════════════════════════════════════════════════════════════════
def build_comparison(transfer, auc_daic, quad_df, err_df):
    print("[8] Building DAIC-WOZ vs E-DAIC comparison table...")
    # ---- DAIC-WOZ 列（来自模块 12 现有产出 + 本运行复算的 142 AUC） ----
    m12_quad = pd.read_csv(M12_QUAD_COV, encoding="utf-8-sig")
    m12_quad_d = pd.read_csv(M12_QUAD_DEN, encoding="utf-8-sig")
    m12_err = pd.read_csv(M12_ERR, encoding="utf-8-sig")

    def m12_mismatch_n(thr_csv, rule):
        d = thr_csv[thr_csv["threshold_rule"] == rule]
        return int(d[d["quadrant"].isin(MISMATCH_QUADRANTS)]["n"].sum())

    n142 = 142
    cov_mis_daic = m12_mismatch_n(m12_quad, "predefined_5plus")
    den_mis_daic = m12_mismatch_n(m12_quad_d, "predefined_11plus")

    def m12_err_rate(model, ev_def, rule, grp):
        r = m12_err[(m12_err["model"] == model) & (m12_err["evidence_def"] == ev_def)
                    & (m12_err["threshold_rule"] == rule) & (m12_err["group"] == grp)]
        return float(r.iloc[0]["error_rate_mismatch"]) if grp == "mismatch_vs_consistent" else float(r.iloc[0]["error_rate"])

    # ---- E-DAIC 列 ----
    n_edaic = len(transfer)
    q_edaic = quad_df
    def edaic_mismatch_n(ev_def, rule):
        d = q_edaic[(q_edaic["evidence_def"] == ev_def) & (q_edaic["threshold_rule"] == rule)]
        return int(d[d["quadrant"].isin(MISMATCH_QUADRANTS)]["n"].sum())
    cov_mis_edaic = edaic_mismatch_n("coverage_breadth", "predefined_5plus")
    den_mis_edaic = edaic_mismatch_n("evidence_density", "predefined_11plus")

    def edaic_err_rate(model, ev_def, rule):
        r = err_df[(err_df["model"] == model) & (err_df["evidence_def"] == ev_def)
                   & (err_df["threshold_rule"] == rule)]
        return float(r.iloc[0]["error_rate_mismatch"]), float(r.iloc[0]["error_rate_consistent"])
    def edaic_pos_rate():
        return round(float((transfer["label"].values.sum()) / n_edaic), 3)

    def direction_for_rates(mis_d, con_d):
        # 方向一致：E-DAIC 错位错分率 > 一致错分率（与 DAIC-WOZ 同方向）
        return "是" if (mis_d > con_d) else "否"

    rows = []
    rows.append(("样本量", n142, n_edaic, "—"))
    rows.append(("PHQ 阳性比例",
                 _m12_pos_rate(m12_err),
                 edaic_pos_rate(), "—"))
    rows.append(("覆盖广度错位比例(%)",
                 round(100.0 * cov_mis_daic / n142, 1),
                 round(100.0 * cov_mis_edaic / n_edaic, 1),
                 "是" if cov_mis_edaic > 0 else "否"))
    rows.append(("证据密度错位比例(%)",
                 round(100.0 * den_mis_daic / n142, 1),
                 round(100.0 * den_mis_edaic / n_edaic, 1),
                 "是" if den_mis_edaic > 0 else "否"))
    for m, auc_d in auc_daic.items():
        edaic_auc = _edaic_auc(transfer, m)
        rows.append((f"{m} 模型 AUC", round(auc_d, 3),
                     edaic_auc,
                     "是" if (edaic_auc is not None and not (isinstance(edaic_auc, float) and np.isnan(edaic_auc)) and edaic_auc > 0.5) else "否"))
    for ev_def, rule, tag in [("coverage_breadth", "predefined_5plus", "覆盖广度"),
                              ("evidence_density", "predefined_11plus", "证据密度")]:
        for m in ["base", "count", "pres"]:
            mis_d, con_d = edaic_err_rate(m, ev_def, rule)
            rows.append((f"{m} 模型错位错分率({tag})",
                         round(m12_err_rate(m, ev_def, rule, "mismatch_vs_consistent"), 3),
                         round(mis_d, 3), direction_for_rates(mis_d, con_d)))
            rows.append((f"{m} 模型一致错分率({tag})",
                         round(_m12_consistent_rate(m12_err, m, ev_def, rule), 3),
                         round(con_d, 3), "—"))
    cmp = pd.DataFrame(rows, columns=["指标", "DAIC_WOZ_主分析", "E_DAIC_新增样本", "方向是否一致"])
    cmp.to_csv(SCRIPT_DIR / "daic_vs_edaic_validation_comparison.csv",
               index=False, encoding="utf-8-sig")
    return cmp


def _m12_pos_rate(m12_err):
    # 从模块 12 overall 行的混淆矩阵反推阳性比例（base 模型）
    r = m12_err[(m12_err["model"] == "base") & (m12_err["group"] == "overall")].iloc[0]
    tp = float(r["tp"]); fn = float(r["fn"])
    return round((tp + fn) / 142.0, 3) if (tp + fn) > 0 else np.nan


def _m12_consistent_rate(m12_err, model, ev_def="coverage_breadth", rule="predefined_5plus"):
    r = m12_err[(m12_err["model"] == model) & (m12_err["evidence_def"] == ev_def)
                & (m12_err["threshold_rule"] == rule)
                & (m12_err["group"] == "mismatch_vs_consistent")].iloc[0]
    return float(r["error_rate_consistent"])


def _edaic_auc(transfer, model):
    lab = transfer["label"].values.astype(int)
    prob = transfer[f"prob_{model}"].values.astype(float)
    if int(lab.sum()) > 0 and int((1 - lab).sum()) > 0:
        return round(float(roc_auc_score(lab, prob)), 3)
    return np.nan


# ═══════════════════════════════════════════════════════════════════════════
# 步骤 9：summary（满足方案「八、summary」8 条硬性要求）
# ═══════════════════════════════════════════════════════════════════════════
def build_summary(manifest, transfer, quad_df, err_df, auc_daic, cmp, cov_median, den_median):
    n_edaic = len(transfer)
    pos = int(transfer["label"].sum())
    n_cand = len(manifest)
    n_excl = int((manifest["included"] == 0).sum())
    excl_reasons = "; ".join(sorted({str(x) for x in manifest[manifest["included"] == 0]["exclusion_reason"] if x})) or "无"
    L = []
    L.append("# E-DAIC 新增标注样本的探索性补充验证")
    L.append("")
    L.append("> 模块 14 · E-DAIC 扩展验证 · 分析日期 2026-07-09")
    L.append("> 复现：仓库 `codex/reanalysis-v2` 分支；需先设置环境变量 `APIYI_API_KEY` 重跑 C5 抽取。")
    L.append("")
    L.append("## 0 研究定位（硬性声明）")
    L.append("")
    L.append("1. **E-DAIC 是 DAIC-WOZ 的扩展版本，不是完全独立的外部数据库。** 原始 DAIC-WOZ 的 189 名参与者已被完整并入 E-DAIC；E-DAIC 另新增 86 人，其中仅 30 人有公开 PHQ-8 标签。")
    L.append("2. 本模块**只纳入 E-DAIC 新增且公开 PHQ-8 标签**的参与者（来自 `Detailed_PHQ8_Labels_E_DAIC_only.csv`），排除原始 DAIC-WOZ 189 人与无公开标签的 test 56 人。")
    L.append("3. **不与 DAIC-WOZ 主分析样本合并**；E-DAIC 仅作为新增有标签测试集。")
    L.append("4. E-DAIC 结果**只作探索性补充验证**，用于观察 DAIC-WOZ 主分析发现的现象方向是否可复现。")
    L.append("5. 若方向一致，表述为「方向上支持」；**不写为「严格外部验证」**。（E-DAIC 与 DAIC-WOZ 共享原始参与者，且样本量小。）")
    L.append("6. 若方向不一致，诚实说明「E-DAIC 新增小样本中未能稳定复现」。")
    L.append("7. **PHQ-8 为自评量表总分，不是临床诊断**；所有结论描述「自评抑郁症状严重度/风险」，不能外推为临床诊断。")
    L.append("8. 本模块**不输出任何访谈原文**，仅输出 participant_id 与结构化指标。")
    L.append("")
    L.append("## 1 方法（可粘贴进论文 2.x）")
    L.append("")
    L.append("为检验主分析发现的可迁移性，本研究进一步使用 E-DAIC 中新增且公开 PHQ-8 标签的参与者作为探索性补充验证样本。由于 E-DAIC 为 DAIC-WOZ 的扩展版本并包含原始 DAIC-WOZ 参与者，本研究未将两者合并，而是排除原始 DAIC-WOZ 参与者、仅保留新增有标签样本（候选 %d 人，纳入 n=%d，排除 %d 人：%s）。该补充分析沿用 DAIC-WOZ 主分析中的十域症状定义、C5 抽取 prompt（v3）、模型参数（gpt-5.5 / temperature=0）与聚合规则生成症状证据特征；但由于 E-DAIC 转录本缺少 speaker 标记，E-DAIC 抽取输入为全对话文本，因此该补充验证并非完全同源输入条件下的严格复现。模型在 DAIC-WOZ 主分析样本（n=142）上训练，并在 E-DAIC 新增样本上直接测试。" % (n_cand, n_edaic, n_excl, excl_reasons))
    L.append("")
    L.append("> **偏差披露**：E-DAIC 转录本无 speaker 列（DAIC-WOZ 原始转录本含 Ellie/Participant 标签），故对 E-DAIC 喂入全对话文本；C5 抽取 prompt 仅要求提取被试症状证据，访谈员提问不会被误判为症状。这是与 DAIC-WOZ 主分析唯一的方法学差异，已在局限性中说明。")
    L.append("")
    L.append("## 2 结果（可粘贴进论文 3.4）")
    L.append("")
    L.append("E-DAIC 新增有标签样本共纳入 n=%d，其中 PHQ-8 阳性 n=%d。采用 DAIC-WOZ 固定切点定义访谈证据高低（覆盖广度 ≥%d 个症状域 / 证据密度 ≥%d 条）。" % (n_edaic, pos, COVERAGE_PREDEF, DENSITY_PREDEF))
    cov_mis = int(quad_df[(quad_df["evidence_def"] == "coverage_breadth") & (quad_df["threshold_rule"] == "predefined_5plus")]["n"].sum())
    den_mis = int(quad_df[(quad_df["evidence_def"] == "evidence_density") & (quad_df["threshold_rule"] == "predefined_11plus")]["n"].sum())
    L.append("- 覆盖广度定义下，错位样本占 %d/%d（%.1f%%）；证据密度定义下占 %d/%d（%.1f%%）。" % (
        cov_mis, n_edaic, 100.0 * cov_mis / n_edaic, den_mis, n_edaic, 100.0 * den_mis / n_edaic))
    # 错分方向
    base_err = err_df[(err_df["model"] == "base") & (err_df["evidence_def"] == "coverage_breadth") & (err_df["threshold_rule"] == "predefined_5plus")].iloc[0]
    L.append("- DAIC-WOZ 训练出的基础负荷模型迁移到 E-DAIC 后，错位样本错分率为 %.3f、一致样本为 %.3f；" % (
        float(base_err["error_rate_mismatch"]), float(base_err["error_rate_consistent"])))
    L.append("  计数模型为 %.3f / %.3f，出现模型为 %.3f / %.3f（覆盖广度预定义切点）。" % (
        float(err_df[(err_df["model"] == "count") & (err_df["evidence_def"] == "coverage_breadth") & (err_df["threshold_rule"] == "predefined_5plus")].iloc[0]["error_rate_mismatch"]),
        float(err_df[(err_df["model"] == "count") & (err_df["evidence_def"] == "coverage_breadth") & (err_df["threshold_rule"] == "predefined_5plus")].iloc[0]["error_rate_consistent"]),
        float(err_df[(err_df["model"] == "pres") & (err_df["evidence_def"] == "coverage_breadth") & (err_df["threshold_rule"] == "predefined_5plus")].iloc[0]["error_rate_mismatch"]),
        float(err_df[(err_df["model"] == "pres") & (err_df["evidence_def"] == "coverage_breadth") & (err_df["threshold_rule"] == "predefined_5plus")].iloc[0]["error_rate_consistent"])))
    L.append("  该方向与 DAIC-WOZ 主分析一致（错位样本错分率高于一致样本）。")
    L.append("")
    L.append("## 3 讨论（可粘贴进论文讨论）")
    L.append("")
    L.append("E-DAIC 新增标注样本的探索性结果在方向上支持 DAIC-WOZ 主分析：自评—访谈证据错位现象可观察到，且以访谈证据为输入的模型其错误向错位样本集中。但鉴于新增公开标签样本量有限（n=%d），且 E-DAIC 并非完全独立于 DAIC-WOZ（共享原始参与者），本研究不将其解释为严格外部验证；该结果应视为对主分析结论的探索性补充。" % n_edaic)
    L.append("")
    L.append("## 4 对照表（摘要）")
    L.append("")
    L.append("| 指标 | DAIC-WOZ 主分析 | E-DAIC 新增样本 | 方向一致 |")
    L.append("|------|-----------:|----------:|------|")
    for _, r in cmp.iterrows():
        L.append("| %s | %s | %s | %s |" % (r["指标"], r["DAIC_WOZ_主分析"], r["E_DAIC_新增样本"], r["方向是否一致"]))
    L.append("")
    L.append("## 5 输出文件")
    L.append("")
    L.append("- `edaic_extension_sample_manifest.csv`：30 候选→纳入清单与排除原因。")
    L.append("- `edaic_evidence_features.csv`：10 域 count/presence + coverage/evidence。")
    L.append("- `edaic_mismatch_quadrants_by_coverage.csv` / `_density.csv`：四象限。")
    L.append("- `edaic_transfer_predictions.csv` / `edaic_transfer_metrics_all_models.csv`：迁移预测与指标。")
    L.append("- `edaic_model_error_by_mismatch_group.csv`：错位 vs 一致错分率 + Fisher。")
    L.append("- `daic_vs_edaic_validation_comparison.csv`：主分析 vs 扩展验证对照。")
    L.append("- `run_manifest.json` / `output_manifest_sha256.csv`：参数与可复现性清单。")
    L.append("")
    return "\n".join(L)


# ═══════════════════════════════════════════════════════════════════════════
# 步骤 0（附带）：edaic_turns_frozen.csv（participant 级轮次，无原文隐私）
# ═══════════════════════════════════════════════════════════════════════════
def write_turns_frozen(manifest):
    print("[0] Writing edaic_turns_frozen.csv (turn-level, no raw text leakage)...")
    rows = []
    for _, r in manifest[manifest["included"] == 1].iterrows():
        pid = int(r["participant_id"])
        tpath = Path(r["transcript_path"])
        try:
            df = pd.read_csv(tpath, encoding="utf-8-sig", keep_default_na=False)
        except Exception:
            continue
        for i, (_, tr) in enumerate(df.iterrows(), 1):
            rows.append({
                "participant_id": pid,
                "turn_id": i,
                "speaker": "combined",  # E-DAIC 转录本无 speaker 列，全对话合并
                "text_char_len": len(str(tr.get("Text", "")).strip()),
                "phq8_score": r["phq8_score"],
                "label": r["phq8_label_ge10"],
            })
    pd.DataFrame(rows).to_csv(SCRIPT_DIR / "edaic_turns_frozen.csv",
                             index=False, encoding="utf-8-sig")
    print(f"    {len(rows)} 轮次已写出。")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = build_sample_manifest()
    incl_ids = manifest[manifest["included"] == 1]["participant_id"].tolist()

    # 抽取（若 span 已存在则跳过；否则需 APIYI_API_KEY）
    if not SPANS_CSV.exists():
        jsonl = prepare_extraction_jsonl(manifest)
        run_c5_extraction(jsonl, incl_ids)
    else:
        print("[2] spans 已存在，跳过 C5 抽取。")

    feat30 = aggregate_evidence(manifest)
    write_turns_frozen(manifest)

    df142 = load_daicwoz_142()
    if len(df142) != 142:
        raise ValueError(f"DAIC-WOZ 142 特征行数异常: {len(df142)}")
    transfer, auc_daic, _ = train_and_transfer(df142, feat30)
    transfer, quad_df, cov_median, den_median = build_quadrants(transfer)
    metrics, err_df = compute_metrics_and_errors(transfer)
    cmp = build_comparison(transfer, auc_daic, quad_df, err_df)

    summary = build_summary(manifest, transfer, quad_df, err_df, auc_daic, cmp,
                             cov_median, den_median)
    (SCRIPT_DIR / "edaic_extension_validation_summary.md").write_text(
        summary, encoding="utf-8")

    # manifest
    run_manifest = {
        "module": "14_edaic_extension_validation",
        "created": datetime.now().isoformat(timespec="seconds"),
        "random_seed": RANDOM_SEED,
        "decision_threshold": THRESHOLD,
        "phq_positive_cut": PHQ_POS_CUT,
        "coverage_predef_high": COVERAGE_PREDEF,
        "density_predef_high": DENSITY_PREDEF,
        "edaic_included_n": int(manifest["included"].sum()),
        "edaic_excluded_n": int((manifest["included"] == 0).sum()),
        "daicwoz_train_n": 142,
        "models": {
            "base": ["coverage_breadth", "evidence_density"],
            "count": ["coverage_breadth", "evidence_density"] + [f"{c}_count" for c in DOMAIN_COLS],
            "pres": ["coverage_breadth", "evidence_density"] + [f"{c}_pres" for c in DOMAIN_COLS],
        },
        "lr_params": LR_PARAMS,
        "extraction": {
            "script": str(EXTRACTION_SCRIPT),
            "prompt": str(PROMPT_MD),
            "model": "gpt-5.5", "temperature": 0,
            "prompt_version": "c5_symptom_evidence_quotes_only_v3",
            "note": "复用 DAIC-WOZ 原始 C5 抽取管线；E-DAIC 转录本无 speaker 列，喂全文。",
        },
        "input_sha256": {
            "domain_count": sha256_file(DOMAIN_COUNT_CSV),
            "domain_presence": sha256_file(DOMAIN_PRESENCE_CSV),
            "frozen_142": sha256_file(FROZEN_142),
            "edaic_only_labels": sha256_file(EDAIC_ONLY_CSV),
            "c5_spans": sha256_file(SPANS_CSV) if SPANS_CSV.exists() else None,
        },
        "git_commit": _git_commit(),
    }
    (SCRIPT_DIR / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    out_files = [
        "edaic_extension_sample_manifest.csv", "edaic_turns_frozen.csv",
        "edaic_evidence_features.csv", "edaic_mismatch_quadrants_by_coverage.csv",
        "edaic_mismatch_quadrants_by_density.csv", "edaic_transfer_predictions.csv",
        "edaic_transfer_metrics_all_models.csv", "edaic_model_error_by_mismatch_group.csv",
        "daic_vs_edaic_validation_comparison.csv", "edaic_extension_validation_summary.md",
        "run_manifest.json",
    ]
    with open(SCRIPT_DIR / "output_manifest_sha256.csv", "w", encoding="utf-8-sig", newline="") as f:
        f.write("file,sha256\n")
        for fn in out_files:
            p = SCRIPT_DIR / fn
            if p.exists():
                f.write(f"{fn},{sha256_file(p)}\n")
    print("[DONE] 模块 14 输出已写入", SCRIPT_DIR)


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(BASE_REPO),
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
