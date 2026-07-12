from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "analysis_v2" / "12_submission_audit"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify() -> list[str]:
    errors: list[str] = []
    inventory_path = AUDIT_DIR / "output_manifest_sha256.csv"
    inventory = pd.read_csv(inventory_path)
    for row in inventory.itertuples(index=False):
        path = AUDIT_DIR / str(row.file)
        if not path.is_file():
            errors.append(f"missing: {row.file}")
        elif sha256(path) != str(row.sha256):
            errors.append(f"hash mismatch: {row.file}")

    run_manifest = json.loads((AUDIT_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    plan = AUDIT_DIR / run_manifest["post_audit_plan"]["file"]
    if sha256(plan) != run_manifest["post_audit_plan"]["sha256"]:
        errors.append("post-audit plan hash mismatch")
    if run_manifest["post_audit_plan"]["status"] != "frozen_before_sensitivity_results":
        errors.append("post-audit plan was not recorded as frozen before results")
    if run_manifest["interpretation_guardrails"]["ci"] != (
        "code_tests_and_derived_integrity_not_full_restricted_data_reproduction"
    ):
        errors.append("CI interpretation boundary missing")

    trace = pd.read_csv(AUDIT_DIR / "c5_traceability_audit_no_quotes.csv")
    forbidden_columns = {"quote", "text", "source_text", "original_quote", "corrected_quote"}
    leaked = forbidden_columns.intersection(column.lower() for column in trace.columns)
    if leaked:
        errors.append(f"public C5 audit contains text columns: {sorted(leaked)}")

    required = {
        "locked_brier_skill_metrics.csv",
        "symmetric_half_min_paired_comparison.csv",
        "symmetric_position_paired_comparisons.csv",
        "c5_retained_alignment_paired_differences.csv",
        "model_configuration.md",
        "submission_audit_summary.md",
    }
    missing_required = sorted(name for name in required if not (AUDIT_DIR / name).is_file())
    if missing_required:
        errors.append(f"required outputs missing: {missing_required}")
    return errors


def main() -> int:
    errors = verify()
    if errors:
        print("Result integrity verification: FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Code/test reproducibility: run pytest in the public clone")
    print("Result integrity verification: PASS")
    print("Full data-to-result reproduction: RESTRICTED INPUTS REQUIRED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
