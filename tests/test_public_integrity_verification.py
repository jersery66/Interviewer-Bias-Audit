from __future__ import annotations

import subprocess
import sys
import hashlib
from pathlib import Path

from scripts.verify_submission_artifacts import hash_candidates, plan_status_is_acceptable


def test_public_hash_candidates_tolerate_git_newline_conversion(tmp_path: Path) -> None:
    path = tmp_path / "artifact.csv"
    path.write_bytes(b"a,b\r\n1,2\r\n")
    expected_lf = hashlib.sha256(b"a,b\n1,2\n").hexdigest()

    assert expected_lf in hash_candidates(path)


def test_plan_status_accepts_closeout_amendment_after_original_freeze() -> None:
    assert plan_status_is_acceptable("original_plan_frozen_before_sensitivity_results")
    assert plan_status_is_acceptable("original_plan_frozen_before_sensitivity_results;closeout_amendment_frozen_before_closeout_inference_rerun")


def test_public_integrity_verifier_passes_committed_outputs() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "verify_submission_artifacts.py")],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Result integrity verification: PASS" in result.stdout
    assert "Full data-to-result reproduction: RESTRICTED INPUTS REQUIRED" in result.stdout
