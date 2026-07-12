from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = (
    REPO_ROOT
    / "analysis_v2"
    / "06_tables_figures"
    / "main_text"
    / "manuscript_restructured_zh.md"
)
AUDIT_DIR = REPO_ROOT / "analysis_v2" / "12_submission_audit"


def test_restructured_manuscript_has_reader_facing_architecture() -> None:
    text = MANUSCRIPT.read_text(encoding="utf-8")

    for heading in (
        "## 摘要",
        "## 引言",
        "## 方法",
        "## 结果",
        "## 讨论",
        "## 数据与代码可用性",
        "## 伦理声明",
        "## 补充材料与统计说明",
        "## 参考文献",
    ):
        assert heading in text

    for heading in (
        "### 1 不同文本来源的预测表现",
        "### 2 访谈者语言的条件增量与结构信息",
        "### 3 文本预算、窗口位置与症状证据的敏感性",
    ):
        assert heading in text

    result_h3 = [line for line in text.splitlines() if line.startswith("### ")]
    assert len(result_h3) == 3


def test_restructured_manuscript_removes_internal_audit_vocabulary() -> None:
    text = MANUSCRIPT.read_text(encoding="utf-8")

    for internal_term in (
        "RQ1",
        "RQ2",
        "RQ3",
        "M3",
        "M4",
        "M5",
        "C5",
        "S_length",
        "S_interaction",
        "S_protocol",
        "S_all",
        "half-min",
        "fixed draw",
        "分析锁定",
        "投稿审查前",
        "唯一主张",
        "解释边界",
    ):
        assert internal_term not in text

    for stale_or_overclaimed in (
        "纯长度不足以解释",
        "已证明访谈者偏差",
        "标签泄漏已被证明",
        "完整审核前后比较",
        "外部验证结果表明",
    ):
        assert stale_or_overclaimed not in text


def test_restructured_manuscript_numbers_match_frozen_outputs() -> None:
    text = MANUSCRIPT.read_text(encoding="utf-8")
    source = pd.read_csv(AUDIT_DIR / "locked_source_metrics.csv")
    comparisons = pd.read_csv(AUDIT_DIR / "locked_core_paired_comparisons.csv")
    half_min = pd.read_csv(AUDIT_DIR / "symmetric_half_min_paired_comparison.csv")
    c5 = pd.read_csv(AUDIT_DIR / "c5_retained_alignment_paired_differences.csv")
    trace = json.loads((AUDIT_DIR / "c5_traceability_summary.json").read_text(encoding="utf-8"))

    for condition in (
        "participant_speech",
        "interviewer_speech",
        "full_transcript",
        "participant_symptom_evidence",
    ):
        auc = float(source.loc[source["condition"] == condition, "roc_auc"].iloc[0])
        assert f"{auc:.3f}" in text

    increment = float(
        comparisons.loc[comparisons["comparison"] == "M5 vs M3", "delta_auc"].iloc[0]
    )
    assert f"{increment:.4f}" in text

    delta = float(half_min.loc[0, "mean_delta_auc"])
    ci_low = float(half_min.loc[0, "participant_random_draw_ci_low"])
    ci_high = float(half_min.loc[0, "participant_random_draw_ci_high"])
    assert f"{delta:.4f}" in text
    assert f"{ci_low:.4f}" in text or f"−{abs(ci_low):.4f}" in text
    assert f"{ci_high:.4f}" in text or f"−{abs(ci_high):.4f}" in text

    c5_auc_delta = float(c5.loc[c5["metric"] == "roc_auc", "aligned_minus_original"].iloc[0])
    assert f"{c5_auc_delta:.4f}" in text or f"−{abs(c5_auc_delta):.4f}" in text
    assert f"{int(trace['candidate_span_n'])}" in text
    assert f"{int(trace['final_span_n'])}" in text
    assert f"{int(trace['manually_corrected_to_source_n'])}" in text


def test_restructured_manuscript_states_calibration_and_reproducibility_boundaries() -> None:
    text = MANUSCRIPT.read_text(encoding="utf-8")

    for required in (
        "外层训练折基率",
        "Brier skill score",
        "不把 isotonic 宣称为最优",
        "不能证明访谈者偏差",
        "不能宣称从原始数据完整复现",
        "PHQ-8 是自评症状风险指标，不是临床诊断",
    ):
        assert required in text


def test_analysis_entry_point_distinguishes_clean_text_from_audit_draft() -> None:
    readme = (REPO_ROOT / "analysis_v2" / "README.md").read_text(encoding="utf-8")
    log = (
        REPO_ROOT
        / "analysis_v2"
        / "06_tables_figures"
        / "manuscript_review_and_revision_log.md"
    ).read_text(encoding="utf-8")

    assert "main_text/manuscript_restructured_zh.md" in readme
    assert "manuscript_submission_zh.md" in readme
    assert "reader-facing manuscript restructure" in log
