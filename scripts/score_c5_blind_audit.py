"""Score the D-P blind audit in locked stages.

``validate`` uses only the blinded cases and the two completed independent
forms.  ``compare`` is the explicit unblinding gate and is the first stage that
reads the owner-only D-P scoring reference.  ``evidence`` is a separate second
round for the retained D-P evidence snippets.
"""

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

from reanalysis_v2.c5_blind_workflow import (  # noqa: E402
    build_consensus_template,
    build_disagreement_sheet,
    build_domain_confusion_tables,
    compute_dp_vs_consensus,
    compute_evidence_validity,
    compute_interrater_metrics,
    validate_consensus_labels,
    validate_reviewer_form,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, keep_default_na=True)


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def _expected_ids(cases: pd.DataFrame) -> list[str]:
    if list(cases.columns) != ["blind_id", "participant_text"]:
        raise ValueError("cases must contain only blind_id and participant_text")
    return cases["blind_id"].astype(str).tolist()


def run_validate(*, cases_path: Path, reviewer_a_path: Path, reviewer_b_path: Path, output_dir: Path, force: bool) -> dict[str, object]:
    cases = _read(cases_path)
    reviewer_a = _read(reviewer_a_path)
    reviewer_b = _read(reviewer_b_path)
    expected_ids = _expected_ids(cases)
    validate_reviewer_form(reviewer_a, expected_blind_ids=expected_ids, require_completed=True)
    validate_reviewer_form(reviewer_b, expected_blind_ids=expected_ids, require_completed=True)
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_interrater_metrics(reviewer_a, reviewer_b)
    confusion = build_domain_confusion_tables(reviewer_a, reviewer_b)
    disagreement = build_disagreement_sheet(reviewer_a, reviewer_b, cases)
    consensus_template = build_consensus_template(reviewer_a, reviewer_b)
    _write(metrics, output_dir / "interrater_metrics.csv")
    _write(confusion, output_dir / "domain_confusion_tables.csv")
    _write(disagreement, output_dir / "disagreement_sheet.csv")
    _write(consensus_template, output_dir / "consensus_labels_template.csv")
    manifest = {
        "status": "independent_agreement_scored_consensus_pending",
        "stage": "validate",
        "training_excluded": True,
        "reviewer_A": {"path": str(reviewer_a_path), "sha256": _sha256(reviewer_a_path)},
        "reviewer_B": {"path": str(reviewer_b_path), "sha256": _sha256(reviewer_b_path)},
        "blind_cases": {"path": str(cases_path), "sha256": _sha256(cases_path)},
        "output_hashes": {path.name: _sha256(path) for path in sorted(output_dir.iterdir()) if path.is_file()},
        "code_commit": _code_commit(),
        "next_gate": "complete consensus_labels.csv without opening owner-only D-P files",
    }
    (output_dir / "blind_audit_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def run_compare(*, consensus_path: Path, reviewer_a_path: Path, reviewer_b_path: Path, scoring_reference_path: Path, output_dir: Path, force: bool) -> dict[str, object]:
    reviewer_a = _read(reviewer_a_path)
    reviewer_b = _read(reviewer_b_path)
    consensus = _read(consensus_path)
    scoring_reference = _read(scoring_reference_path)
    validate_consensus_labels(consensus, reviewer_a, reviewer_b)
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = pd.concat(
        [compute_dp_vs_consensus(consensus, scoring_reference, ns_policy=policy) for policy in ("exclude", "as_0", "as_1")],
        ignore_index=True,
    )
    _write(outputs, output_dir / "dp_vs_consensus_metrics.csv")
    manifest = {
        "status": "consensus_scored_after_unblinding",
        "stage": "compare",
        "ns_policies": ["exclude", "as_0", "as_1"],
        "consensus": {"path": str(consensus_path), "sha256": _sha256(consensus_path)},
        "reviewer_A": {"path": str(reviewer_a_path), "sha256": _sha256(reviewer_a_path)},
        "reviewer_B": {"path": str(reviewer_b_path), "sha256": _sha256(reviewer_b_path)},
        "scoring_reference": {"path": str(scoring_reference_path), "sha256": _sha256(scoring_reference_path)},
        "output_hashes": {path.name: _sha256(path) for path in sorted(output_dir.iterdir()) if path.is_file()},
        "code_commit": _code_commit(),
    }
    (output_dir / "blind_audit_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def run_evidence(*, review_path: Path, output_dir: Path, force: bool) -> dict[str, object]:
    review = _read(review_path)
    output = compute_evidence_validity(review)
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write(output, output_dir / "evidence_validity_results.csv")
    manifest = {
        "status": "evidence_round_scored",
        "stage": "evidence",
        "review": {"path": str(review_path), "sha256": _sha256(review_path)},
        "output_hashes": {path.name: _sha256(path) for path in sorted(output_dir.iterdir()) if path.is_file()},
        "code_commit": _code_commit(),
    }
    (output_dir / "blind_audit_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Score the staged D-P blinded audit")
    parser.add_argument("--stage", choices=("validate", "compare", "evidence"), required=True)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--reviewer-a", type=Path)
    parser.add_argument("--reviewer-b", type=Path)
    parser.add_argument("--consensus", type=Path)
    parser.add_argument("--scoring-reference", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.stage == "validate":
        required = {"--cases": args.cases, "--reviewer-a": args.reviewer_a, "--reviewer-b": args.reviewer_b}
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error(f"validate requires {', '.join(missing)}")
        manifest = run_validate(cases_path=args.cases, reviewer_a_path=args.reviewer_a, reviewer_b_path=args.reviewer_b, output_dir=args.output_dir, force=args.force)
    elif args.stage == "compare":
        required = {"--consensus": args.consensus, "--reviewer-a": args.reviewer_a, "--reviewer-b": args.reviewer_b, "--scoring-reference": args.scoring_reference}
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error(f"compare requires {', '.join(missing)}")
        manifest = run_compare(consensus_path=args.consensus, reviewer_a_path=args.reviewer_a, reviewer_b_path=args.reviewer_b, scoring_reference_path=args.scoring_reference, output_dir=args.output_dir, force=args.force)
    else:
        if args.review is None:
            parser.error("evidence requires --review")
        manifest = run_evidence(review_path=args.review, output_dir=args.output_dir, force=args.force)
    print(json.dumps({"status": manifest["status"], "stage": manifest["stage"], "output_dir": str(args.output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

