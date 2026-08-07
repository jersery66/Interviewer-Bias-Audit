from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = (
    REPO_ROOT
    / "analysis_v2"
    / "06_tables_figures"
    / "main_text"
    / "manuscript_restructured_zh.md"
)
AUDIT_DIR = REPO_ROOT / "analysis_v2" / "12_submission_audit"


def test_restructured_manuscript_is_a_reader_facing_architecture_index() -> None:
    text = MANUSCRIPT.read_text(encoding="utf-8")

    assert text.startswith("# 当前论文骨架")
    for heading in ("## 核心问题", "## 正文四组结果", "## 补充材料", "## 结论句", "## 未完成的人工作业"):
        assert heading in text
    for result_group in ("单来源可预测性", "逐级条件增量", "访谈者信号可恢复性", "配对破坏与 Fake-D"):
        assert result_group in text
    assert "manuscript_submission_zh.md" in text
    assert "training_match_quality_limited" in text


def test_restructured_manuscript_removes_internal_audit_vocabulary() -> None:
    text = MANUSCRIPT.read_text(encoding="utf-8")

    for internal_term in ("RQ1", "RQ2", "RQ3", "M3", "M4", "M5", "S_length", "fixed draw"):
        assert internal_term not in text
    for stale_or_overclaimed in (
        "已经证明访谈者偏差",
        "标签泄漏已被证明",
        "完整审核前后比较",
        "外部验证结果表明",
    ):
        assert stale_or_overclaimed not in text
    assert "C5" in text
    assert "half-min" in text


def test_restructured_manuscript_defers_numbers_to_canonical_submission() -> None:
    text = MANUSCRIPT.read_text(encoding="utf-8")

    assert "唯一权威版本" in text
    assert "补充材料" in text
    assert "完整 ledger" in text


def test_restructured_manuscript_states_calibration_and_reproducibility_boundaries() -> None:
    text = MANUSCRIPT.read_text(encoding="utf-8")

    assert "Brier" in text
    assert "不进入主叙事" in text
    assert "不报告域级一致率" in text
    assert "PHQ" in text


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
