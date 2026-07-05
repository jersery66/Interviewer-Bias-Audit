from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = REPO_ROOT / "analysis_v2" / "06_tables_figures" / "manuscript_draft_zh.md"
STRICT_DIR = REPO_ROOT / "analysis_v2" / "10_source_importance" / "07_strict_joint_models"


def test_manuscript_uses_strict_results_and_embeds_all_figures() -> None:
    text = MANUSCRIPT.read_text(encoding="utf-8")
    metrics = pd.read_csv(STRICT_DIR / "model_metrics.csv")

    assert text.startswith("# ")
    assert "10 × 5" in text
    assert "5000" in text
    assert "10,000" in text
    assert "Benjamini–Hochberg" in text
    assert "均未通过 FDR 校正" in text
    assert "*q* = 0.0039" in text

    for row in metrics.itertuples(index=False):
        assert f"{row.roc_auc:.4f}" in text

    for figure_number, stem in (
        (1, "source_decomposition_framework"),
        (2, "incremental_model_auc"),
        (3, "permutation_importance"),
        (4, "domain_leave_one_out"),
    ):
        assert f"![图 {figure_number}]" in text
        assert f"figures/figure_{figure_number}_{stem}.png" in text


def test_manuscript_removes_superseded_claims_and_has_paper_sections() -> None:
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

    assert "条件性关联" in text
    assert "外部独立测试集" in text
    assert "固定的重复 OOF 预测" in text
