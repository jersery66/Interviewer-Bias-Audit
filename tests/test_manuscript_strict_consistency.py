from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = REPO_ROOT / "analysis_v2" / "06_tables_figures" / "manuscript_submission_zh.md"
SUBMISSION_DIR = REPO_ROOT / "analysis_v2" / "12_submission_audit"
FORMAL_DIR = (
    REPO_ROOT
    / "analysis_v2"
    / "10_source_importance"
    / "08_identification_sensitivity"
    / "run_formal_10x"
)


def test_manuscript_uses_current_four_group_architecture() -> None:
    text = MANUSCRIPT.read_text(encoding="utf-8")

    assert text.startswith("# ")
    for heading in ("### 3.1", "### 3.2", "### 3.3", "### 3.4"):
        assert heading in text
    assert "锁定五折" in text
    assert "不是预注册" in text
    assert "5,000" in text
    assert "BH-FDR" in text
    assert "training_match_quality_limited" in text
    assert "code/test reproducibility" in text
    assert "原始数据完整复现" in text

    source = pd.read_csv(SUBMISSION_DIR / "locked_source_metrics.csv")
    for condition in (
        "participant_speech",
        "interviewer_speech",
        "full_transcript",
        "participant_symptom_evidence",
    ):
        auc = float(source.loc[source["condition"] == condition, "roc_auc"].iloc[0])
        assert f"{auc:.3f}" in text


def test_manuscript_has_paper_sections_and_downgraded_claims() -> None:
    text = MANUSCRIPT.read_text(encoding="utf-8")

    for stale_value in ("0.6956", "0.8252", "0.1294", "0.1810", "0.1272"):
        assert stale_value not in text
    for section in (
        "## 摘要",
        "## 1 引言",
        "## 2 方法",
        "## 3 结果",
        "## 4 讨论",
        "## 5 局限",
        "## 6 结论",
        "## 数据与代码可用性",
        "## 伦理声明",
        "## 参考文献",
    ):
        assert section in text

    assert "条件增量" in text
    assert "访谈者语言整体没有价值" in text
    assert "不证明访谈者造成了偏差" in text
    assert "不等同于临床诊断" in text
    assert "Fake-D" in text
    assert "不能把访谈者增量下降归因于 D-P 已经特异性吸收了患者症状内容" in text


def test_claim_matrix_and_submission_readme_lock_interpretation_boundaries() -> None:
    readme = (REPO_ROOT / "analysis_v2" / "README.md").read_text(encoding="utf-8")
    matrix = (SUBMISSION_DIR / "claim_evidence_matrix.md").read_text(encoding="utf-8")

    assert "Single-source performance measures predictive sufficiency" in readme
    assert "interviewer bias as an identified causal effect" in readme
    assert "Real pairing retains extra ranking information" in matrix
    assert "Fake-D negative control" in matrix
    assert "External validity" in matrix
    assert "Cohen’s κ" in matrix
    assert "no agreement or consensus metrics are available" in matrix


def test_submission_abstract_and_claims_do_not_overreach() -> None:
    text = MANUSCRIPT.read_text(encoding="utf-8")
    abstract = text.split("## 1 引言", maxsplit=1)[0]

    assert "E-DAIC" not in abstract
    for unsupported in (
        "访谈者偏差导致",
        "证明了访谈者偏差",
        "模型具有临床筛查",
        "C5 证明",
        "外部验证结果表明",
        "泛化性能得到验证",
    ):
        assert unsupported not in text


def test_submission_numbers_match_current_frozen_outputs() -> None:
    text = MANUSCRIPT.read_text(encoding="utf-8")
    source = pd.read_csv(SUBMISSION_DIR / "locked_source_metrics.csv")
    c5 = pd.read_csv(SUBMISSION_DIR / "c5_retained_alignment_paired_differences.csv")
    trace = json.loads((SUBMISSION_DIR / "c5_traceability_summary.json").read_text(encoding="utf-8"))
    primary = pd.read_csv(FORMAL_DIR / "primary_contrasts.csv")

    for condition in (
        "participant_speech",
        "interviewer_speech",
        "full_transcript",
        "participant_symptom_evidence",
    ):
        auc = float(source.loc[source["condition"] == condition, "roc_auc"].iloc[0])
        assert f"{auc:.3f}" in text
    for row in primary.itertuples(index=False):
        assert f"{row.mean:.4f}".replace("-", "−") in text
        assert f"{row.ci_low:.4f}".replace("-", "−") in text
        assert f"{row.ci_high:.4f}".replace("-", "−") in text
    auc_delta = float(c5.loc[c5["metric"] == "roc_auc", "aligned_minus_original"].iloc[0])
    assert f"{auc_delta:.4f}".replace("-", "−") in text
    assert f"{int(trace['final_span_n']):,}" in text
    assert str(int(trace["manually_corrected_to_source_n"])) in text


def test_post_audit_results_and_interpretive_downgrades_are_integrated() -> None:
    text = MANUSCRIPT.read_text(encoding="utf-8")
    matrix = (SUBMISSION_DIR / "claim_evidence_matrix.md").read_text(encoding="utf-8")

    for required in (
        "外层训练折阳性率",
        "Brier skill",
        "−0.092",
        "对称 half-min",
        "−0.0691 至 0.2953",
        "窗口",
        "来源对齐敏感性分析",
        "1,137",
        "24 名受试者",
        "−0.0042",
        "盲法",
    ):
        assert required in text

    for overclaim in (
        "文本量不能解释访谈者侧表现",
        "协议和互动结构具有明确预测信号",
        "完整审核前后比较",
        "D-P 特异性吸收",
    ):
        assert overclaim not in text

    assert "symmetric half-min" in matrix
    assert "Complete pre/post review comparison" in matrix
    assert "does not support a simple" in matrix


def test_related_work_claim_is_tied_to_a_targeted_search_log() -> None:
    text = MANUSCRIPT.read_text(encoding="utf-8")
    search_log = (SUBMISSION_DIR / "targeted_literature_search_log.md").read_text(
        encoding="utf-8"
    )

    assert "定向检索" in text
    assert "不据此提出“首次”" in text
    assert "数据库与来源" in search_log
    assert "检索截止日期" in search_log
    assert "纳入标准" in search_log
    assert "排除标准" in search_log
    assert "非系统综述" in search_log


def test_submission_tiffs_are_losslessly_compressed_for_github() -> None:
    tiffs = sorted((SUBMISSION_DIR / "figures").glob("figure_*.tiff"))

    assert len(tiffs) == 4
    for path in tiffs:
        with Image.open(path) as image:
            assert image.tag_v2.get(259) in {5, 8}
        assert path.stat().st_size < 100 * 1024 * 1024


def test_submission_manifest_records_tiff_dependency() -> None:
    manifest = json.loads((SUBMISSION_DIR / "run_manifest.json").read_text(encoding="utf-8"))

    assert manifest["runtime"]["tifffile"]
