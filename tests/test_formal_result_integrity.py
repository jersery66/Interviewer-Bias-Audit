from __future__ import annotations

from pathlib import Path

from scripts.verify_submission_artifacts import verify_formal_identification_output


FORMAL_DIR = (
    Path(__file__).resolve().parents[1]
    / "analysis_v2"
    / "10_source_importance"
    / "08_identification_sensitivity"
    / "run_formal_10x"
)


def test_committed_formal_identification_results_pass_integrity_gate() -> None:
    assert verify_formal_identification_output(FORMAL_DIR) == []
