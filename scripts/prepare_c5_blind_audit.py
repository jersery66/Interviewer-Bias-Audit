"""Prepare the blinded D-P manual quality-audit packet and owner key."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reanalysis_v2.c5_blind_audit import (  # noqa: E402
    build_blind_cases,
    build_review_form,
    build_scoring_reference,
    make_stratified_sample,
    validate_blind_packet,
)


SEED = 20260714
N_PER_CELL = 5


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def prepare(*, output_root: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    participant_path = output_root / "01_inputs" / "c2_participant_speech.csv"
    domain_path = output_root / "04_c5_controls" / "domain_count" / "input.csv"
    participants = pd.read_csv(participant_path)
    domains = pd.read_csv(domain_path)
    sample = make_stratified_sample(participants, n_per_cell=N_PER_CELL, seed=SEED)
    cases = build_blind_cases(participants, sample)
    reviewer_a = build_review_form(sample, reviewer_id="reviewer_a")
    reviewer_b = build_review_form(sample, reviewer_id="reviewer_b")
    scoring_reference = build_scoring_reference(sample, domains)
    validate_blind_packet(cases, reviewer_a)
    validate_blind_packet(cases, reviewer_b)

    sample.to_csv(output_dir / "sample_key_owner_only.csv", index=False, lineterminator="\n")
    cases.to_csv(output_dir / "blind_cases.csv", index=False, lineterminator="\n")
    reviewer_a.to_csv(output_dir / "reviewer_a_blank.csv", index=False, lineterminator="\n")
    reviewer_b.to_csv(output_dir / "reviewer_b_blank.csv", index=False, lineterminator="\n")
    scoring_reference.to_csv(
        output_dir / "scoring_reference_owner_only.csv", index=False, lineterminator="\n"
    )
    readme = """# D-P blind manual audit packet

`blind_cases.csv` and the reviewer blank forms contain no labels, PHQ-8 scores, participant IDs, model results, interviewer language, or existing D-P counts. Give the cases file and exactly one blank form to each independent reviewer.

`sample_key_owner_only.csv` and `scoring_reference_owner_only.csv` are owner-only files and must not be shared with reviewers. The packet is awaiting human judgments; no agreement or validity statistic is populated yet.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    output_hashes = {
        path.name: _sha256(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "packet_manifest.json"
    }
    manifest = {
        "status": "pending_reviewer_input",
        "analysis": "d_p_blind_manual_quality_audit",
        "sample_n": int(len(sample)),
        "n_per_label_length_cell": N_PER_CELL,
        "strata": "paper label x participant word-count tertile",
        "seed": SEED,
        "reviewers": ["reviewer_a", "reviewer_b"],
        "fallback_if_reviewer_b_missing": "single-reviewer content-validity sample only; no kappa or consensus metrics",
        "inputs": {
            "participant_speech": {"path": str(participant_path), "sha256": _sha256(participant_path)},
            "domain_count": {"path": str(domain_path), "sha256": _sha256(domain_path)},
        },
        "code_commit": _code_commit(),
        "blind_case_columns": list(cases.columns),
        "output_file_hashes": output_hashes,
    }
    (output_dir / "packet_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the D-P blind manual audit packet")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "analysis_v2")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "analysis_v2" / "04_c5_controls" / "blind_quality_audit" / "packet_1",
    )
    args = parser.parse_args()
    manifest = prepare(output_root=args.output_root, output_dir=args.output_dir)
    print(json.dumps({"status": manifest["status"], "sample_n": manifest["sample_n"], "output_dir": str(args.output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
