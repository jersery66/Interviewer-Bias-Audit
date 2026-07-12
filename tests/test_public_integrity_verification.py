from __future__ import annotations

import subprocess
import sys
from pathlib import Path


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
