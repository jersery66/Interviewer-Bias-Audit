from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY = REPO_ROOT / "analysis_v2" / "12_submission_audit" / "frozen_data_summary.md"
AUDIT_DIR = REPO_ROOT / "analysis_v2" / "12_submission_audit"


def test_frozen_data_summary_contains_key_frozen_values() -> None:
    text = SUMMARY.read_text(encoding="utf-8")
    source = pd.read_csv(AUDIT_DIR / "locked_source_metrics.csv")
    core = pd.read_csv(AUDIT_DIR / "locked_core_paired_comparisons.csv")
    half_min = pd.read_csv(AUDIT_DIR / "symmetric_half_min_paired_comparison.csv")
    c5 = pd.read_csv(AUDIT_DIR / "c5_retained_alignment_paired_differences.csv")
    trace = json.loads((AUDIT_DIR / "c5_traceability_summary.json").read_text(encoding="utf-8"))

    assert "n | 142" in text
    assert "n_positive | 43" in text
    for condition in (
        "participant_speech",
        "interviewer_speech",
        "full_transcript",
        "participant_symptom_evidence",
    ):
        auc = float(source.loc[source["condition"] == condition, "roc_auc"].iloc[0])
        assert f"{auc:.3f}" in text

    delta = float(core.loc[core["comparison"] == "M5 vs M3", "delta_auc"].iloc[0])
    assert f"{delta:.4f}" in text
    for column in (
        "mean_delta_auc",
        "participant_random_draw_ci_low",
        "participant_random_draw_ci_high",
    ):
        value = float(half_min.loc[0, column])
        assert f"{value:.4f}" in text or f"−{abs(value):.4f}" in text

    c5_delta = float(c5.loc[c5["metric"] == "roc_auc", "aligned_minus_original"].iloc[0])
    assert f"{c5_delta:.4f}" in text or f"−{abs(c5_delta):.4f}" in text
    for key in ("candidate_span_n", "final_span_n", "unchanged_source_matched_n", "manually_corrected_to_source_n"):
        assert str(int(trace[key])) in text

    for expected_file in (
        "locked_brier_fold_metrics.csv",
        "locked_calibration_curve_data.csv",
        "locked_threshold_curve_data.csv",
        "locked_template_control_metrics.csv",
        "locked_joint_predictions.csv",
        "symmetric_half_min_draw_predictions.csv",
        "c5_traceability_audit_no_quotes.csv",
        "output_manifest_sha256.csv",
    ):
        assert expected_file in text

    assert "T_template" in text
    assert "local_pytest_passed | 101" in text


def test_frozen_data_summary_has_no_interpretive_directives() -> None:
    text = SUMMARY.read_text(encoding="utf-8")

    for forbidden in (
        "显著",
        "更高",
        "优于",
        "改善",
        "支持",
        "偏差",
        "泄漏",
        "结论",
        "解释",
        "因此",
        "应当",
        "不能",
    ):
        assert forbidden not in text


def test_frozen_data_summary_links_from_analysis_readme() -> None:
    readme = (REPO_ROOT / "analysis_v2" / "README.md").read_text(encoding="utf-8")
    assert "12_submission_audit/frozen_data_summary.md" in readme
