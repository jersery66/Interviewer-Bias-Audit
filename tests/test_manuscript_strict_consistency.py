from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = REPO_ROOT / "analysis_v2" / "06_tables_figures" / "manuscript_submission_zh.md"
SUBMISSION_DIR = REPO_ROOT / "analysis_v2" / "12_submission_audit"


def test_manuscript_uses_strict_results_and_embeds_all_figures() -> None:
    text = MANUSCRIPT.read_text(encoding="utf-8")
    metrics = pd.read_csv(SUBMISSION_DIR / "locked_source_metrics.csv", encoding="utf-8-sig")

    assert text.startswith("# ")
    assert "分析锁定划分" in text
    assert "不构成正式预注册" in text
    assert "预设重复划分" not in text
    assert "每名被试仅产生一次 cross-fitted 预测" in text
    assert "5000" in text
    assert "10,000" in text
    assert "Benjamini–Hochberg" in text
    assert "10×5 重复交叉验证仅用于补充" in text

    for row in metrics.itertuples(index=False):
        assert f"{row.roc_auc:.3f}" in text

    for figure_number, stem in (
        (1, "source_signal_paths"),
        (2, "locked_source_performance"),
        (3, "calibration_threshold_locked"),
        (4, "structure_controls_locked"),
    ):
        assert f"![图 {figure_number}]" in text
        assert f"../12_submission_audit/figures/figure_{figure_number}_{stem}.png" in text


def test_manuscript_removes_superseded_claims_and_has_paper_sections() -> None:
    text = MANUSCRIPT.read_text(encoding="utf-8")

    for stale_value in ("0.6956", "0.8252", "0.1294", "0.1810", "0.1272", "0.0039"):
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

    assert "条件关联" in text
    assert "独立外部测试集" in text
    assert "来源单独可预测不等于其对联合模型具有独立贡献" in text
    assert "不证明访谈者造成因果偏差或标签泄漏" in text
    assert "不是精神科诊断" in text


def test_claim_matrix_and_submission_readme_lock_interpretation_boundaries() -> None:
    readme = (REPO_ROOT / "analysis_v2" / "README.md").read_text(encoding="utf-8")
    matrix = (SUBMISSION_DIR / "claim_evidence_matrix.md").read_text(encoding="utf-8")

    assert "Single-source performance measures predictive sufficiency" in readme
    assert "interviewer bias as an identified causal effect" in readme
    assert "M5-M3 Delta AUC 0.003" in matrix
    assert "Outcome blinding cannot be demonstrated" in matrix
    assert "External validation" in matrix


def test_submission_abstract_and_claims_do_not_overreach() -> None:
    text = MANUSCRIPT.read_text(encoding="utf-8")
    abstract = text.split("## 1 引言", maxsplit=1)[0]

    assert "E-DAIC" not in abstract
    assert text.count("![图 ") == 4
    for unsupported in (
        "访谈者偏差导致",
        "证明了访谈者偏差",
        "模型具有临床筛查",
        "C5 证明",
        "外部验证结果表明",
        "泛化性能得到验证",
    ):
        assert unsupported not in text


def test_submission_numbers_match_control_outputs() -> None:
    text = MANUSCRIPT.read_text(encoding="utf-8")
    structural = pd.read_csv(
        SUBMISSION_DIR / "structural_baseline_metrics.csv", encoding="utf-8-sig"
    )
    token = pd.read_csv(SUBMISSION_DIR / "token_matched_metrics.csv", encoding="utf-8-sig")
    comparisons = pd.read_csv(
        SUBMISSION_DIR / "locked_core_paired_comparisons.csv", encoding="utf-8-sig"
    )
    c5 = pd.read_json(SUBMISSION_DIR / "c5_traceability_summary.json", typ="series")

    for row in structural.itertuples(index=False):
        assert f"{row.roc_auc:.3f}" in text
    for row in token.itertuples(index=False):
        assert f"{row.mean_auc:.3f}" in text
    for row in comparisons.itertuples(index=False):
        assert f"{row.delta_auc:.4f}" in text
        assert f"q={row.q_value:.4f}" in text
    assert str(int(c5["final_span_n"])) in text
    assert str(int(c5["manually_corrected_to_source_n"])) in text


def test_post_audit_results_and_interpretive_downgrades_are_integrated() -> None:
    text = MANUSCRIPT.read_text(encoding="utf-8")
    matrix = (SUBMISSION_DIR / "claim_evidence_matrix.md").read_text(encoding="utf-8")

    for required in (
        "外层训练折基率",
        "Brier skill score",
        "−0.092",
        "对称半最小预算",
        "−0.069–0.295",
        "位置敏感",
        "来源对齐修订敏感性分析",
        "1102/1143",
        "96.4%",
        "−0.0042",
    ):
        assert required in text

    for overclaim in (
        "纯长度不足以解释访谈者侧表现",
        "协议和互动结构具有明显预测信号",
        "最终 1137 条证据均可回溯到被试原文，其中 35 条经人工纠正",
    ):
        assert overclaim not in text

    assert "symmetric half-min Delta AUC 0.117" in matrix
    assert "1137/1137 is post-review traceability, not extraction accuracy" in matrix


def test_related_work_claim_is_tied_to_a_targeted_search_log() -> None:
    text = MANUSCRIPT.read_text(encoding="utf-8")
    search_log = (SUBMISSION_DIR / "targeted_literature_search_log.md").read_text(
        encoding="utf-8"
    )

    assert "在本次定向检索纳入的代表性研究中" in text
    assert "首次" not in text
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
