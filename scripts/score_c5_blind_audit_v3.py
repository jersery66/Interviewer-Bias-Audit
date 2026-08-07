"""Score packet-3 D-P validation in locked stages."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import cohen_kappa_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reanalysis_v2.c5_blind_workflow_v3 import (
    DOMAIN_FEATURES,
    build_stage1_consensus_template,
    compute_count_validity,
    compute_dp_validity,
    compute_stage1_interrater,
    compute_stage2_quality,
    freeze_consensus,
    validate_stage1_consensus,
    validate_stage1_reviewer_form,
    verify_consensus_freeze,
)
from scripts.prepare_c5_blind_audit_v3 import DEFAULT_PRIVATE_PACKET, DEFAULT_PUBLIC_PACKET


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def _prepare_output(output_dir: Path, *, force: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def _manifest(output_dir: Path, payload: dict[str, object]) -> dict[str, object]:
    files = [
        path
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "stage_manifest.json"
    ]
    payload["output_hashes"] = {path.name: _sha256(path) for path in files}
    (output_dir / "stage_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload


def run_validate(
    *,
    cases_path: Path,
    reviewer_a_path: Path,
    reviewer_b_path: Path,
    output_dir: Path,
    force: bool = False,
) -> dict[str, object]:
    cases = _read(cases_path)
    reviewer_a = _read(reviewer_a_path)
    reviewer_b = _read(reviewer_b_path)
    expected_ids = cases["blind_id"].astype(str).tolist()
    validate_stage1_reviewer_form(
        reviewer_a, expected_blind_ids=expected_ids, require_completed=True
    )
    validate_stage1_reviewer_form(
        reviewer_b, expected_blind_ids=expected_ids, require_completed=True
    )
    _prepare_output(output_dir, force=force)
    agreement = compute_stage1_interrater(reviewer_a, reviewer_b)
    consensus_template = build_stage1_consensus_template(reviewer_a, reviewer_b)
    disagreement_mask = (
        consensus_template["reviewer_A_manual_status"].fillna("").astype(str)
        != consensus_template["reviewer_B_manual_status"].fillna("").astype(str)
    ) | (
        consensus_template["reviewer_A_count_bin"].fillna("").astype(str)
        != consensus_template["reviewer_B_count_bin"].fillna("").astype(str)
    )
    disagreement = consensus_template.loc[disagreement_mask].merge(
        cases[["blind_id", "participant_text"]],
        on="blind_id",
        how="left",
        validate="many_to_one",
    )
    _write(agreement, output_dir / "stage1_interrater_metrics.csv")
    _write(disagreement, output_dir / "stage1_disagreement_sheet.csv")
    _write(consensus_template, output_dir / "stage1_consensus_template.csv")
    return _manifest(
        output_dir,
        {
            "status": "stage1_independent_agreement_scored",
            "stage": "validate",
            "c5_or_dp_opened": False,
            "reviewer_A_sha256": _sha256(reviewer_a_path),
            "reviewer_B_sha256": _sha256(reviewer_b_path),
            "cases_sha256": _sha256(cases_path),
            "next_gate": "adjudicate and freeze human consensus without opening D-P or C5",
        },
    )


def run_freeze(
    *,
    consensus_path: Path,
    reviewer_a_path: Path,
    reviewer_b_path: Path,
    freeze_record_path: Path,
) -> dict[str, object]:
    consensus = _read(consensus_path)
    reviewer_a = _read(reviewer_a_path)
    reviewer_b = _read(reviewer_b_path)
    validate_stage1_consensus(consensus, reviewer_a, reviewer_b)
    record = freeze_consensus(consensus_path, freeze_record_path)
    record["reviewer_A_sha256"] = _sha256(reviewer_a_path)
    record["reviewer_B_sha256"] = _sha256(reviewer_b_path)
    freeze_record_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return record


def _sentinel_table(consensus: pd.DataFrame, scoring_reference: pd.DataFrame) -> pd.DataFrame:
    sentinel = scoring_reference.loc[scoring_reference["sample_type"].eq("sentinel")]
    rows: list[dict[str, object]] = []
    for row in sentinel.itertuples(index=False):
        for domain in DOMAIN_FEATURES:
            rows.append(
                {
                    "blind_id": str(row.blind_id),
                    "participant_id": int(row.participant_id),
                    "sample_type": "sentinel",
                    "domain": domain,
                    "dp_count": float(getattr(row, domain)),
                    "dp_presence": int(float(getattr(row, domain)) > 0),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.merge(
        consensus[
            [
                "blind_id",
                "domain",
                "consensus_status",
                "consensus_indeterminate_reason",
                "consensus_count_bin",
                "consensus_evidence_quote",
                "consensus_line_number",
            ]
        ],
        on=["blind_id", "domain"],
        how="left",
        validate="one_to_one",
    )


def run_compare(
    *,
    consensus_path: Path,
    freeze_record_path: Path,
    scoring_reference_path: Path,
    output_dir: Path,
    bootstrap_n: int = 5000,
    seed: int = 20260721,
    force: bool = False,
) -> dict[str, object]:
    verify_consensus_freeze(consensus_path, freeze_record_path)
    consensus = _read(consensus_path)
    scoring_reference = _read(scoring_reference_path)
    _prepare_output(output_dir, force=force)
    presence = compute_dp_validity(
        consensus,
        scoring_reference,
        bootstrap_n=bootstrap_n,
        seed=seed,
    )
    counts = compute_count_validity(consensus, scoring_reference)
    sentinel = _sentinel_table(consensus, scoring_reference)
    _write(presence, output_dir / "dp_presence_validity_metrics.csv")
    _write(counts, output_dir / "dp_count_validity_secondary.csv")
    _write(sentinel, output_dir / "sentinel_case_review.csv")
    return _manifest(
        output_dir,
        {
            "status": "dp_validity_scored_main_enriched_separate",
            "stage": "compare",
            "main_sample_is_primary": True,
            "enriched_sample_pooled_into_primary": False,
            "sentinel_included_in_aggregate_metrics": False,
            "participant_cluster_bootstrap_n": int(bootstrap_n),
            "bootstrap_seed": int(seed),
            "consensus_sha256": _sha256(consensus_path),
            "freeze_record_sha256": _sha256(freeze_record_path),
            "scoring_reference_sha256": _sha256(scoring_reference_path),
        },
    )


def run_evidence(
    *,
    evidence_review_path: Path,
    omission_review_path: Path,
    blind_key_path: Path,
    output_dir: Path,
    force: bool = False,
) -> dict[str, object]:
    evidence = _read(evidence_review_path)
    omissions = _read(omission_review_path)
    blind_key = _read(blind_key_path)
    _prepare_output(output_dir, force=force)
    quality = compute_stage2_quality(evidence, omissions, blind_key)
    _write(quality, output_dir / "stage2_evidence_quality.csv")
    return _manifest(
        output_dir,
        {
            "status": "stage2_evidence_quality_scored",
            "stage": "evidence",
            "main_and_enriched_reported_separately": True,
            "sentinel_included_in_aggregate_metrics": False,
            "evidence_review_sha256": _sha256(evidence_review_path),
            "omission_review_sha256": _sha256(omission_review_path),
            "blind_key_sha256": _sha256(blind_key_path),
        },
    )


STAGE2_EVIDENCE_FIXED_COLUMNS = (
    "blind_id",
    "domain",
    "dp_evidence_id",
    "dp_quote",
    "dp_line_number",
    "dp_polarity",
)
STAGE2_EVIDENCE_CORE_COLUMNS = (
    "source_supported",
    "speaker_correct",
    "domain_correct",
    "negation_preserved",
    "temporality_correct",
    "subject_correct",
    "context_preserved",
)
STAGE2_OMISSION_FIXED_COLUMNS = (
    "blind_id",
    "domain",
    "human_evidence_id",
    "human_evidence_quote",
    "human_line_number",
    "same_cell_dp_evidence_ids",
)
STAGE2_OMISSION_CORE_COLUMNS = ("covered_by_dp", "important_omission")


def _stage2_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _stage2_merge_reviewers(
    reviewer_a: pd.DataFrame,
    reviewer_b: pd.DataFrame,
    *,
    key_column: str,
    fixed_columns: tuple[str, ...],
    core_columns: tuple[str, ...],
    optional_columns: tuple[str, ...],
    review_name: str,
) -> pd.DataFrame:
    required = set(fixed_columns) | set(core_columns) | set(optional_columns)
    for reviewer_name, frame in (("A", reviewer_a), ("B", reviewer_b)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{review_name} reviewer {reviewer_name} missing columns: {missing}")
        if frame[key_column].duplicated().any():
            raise ValueError(f"{review_name} reviewer {reviewer_name} has duplicate {key_column}")
        for column in core_columns:
            values = _stage2_text(frame[column]).str.lower()
            invalid = sorted(set(values) - {"yes", "no"})
            if invalid:
                raise ValueError(
                    f"{review_name} reviewer {reviewer_name} invalid {column}: {invalid}"
                )

    keys_a = set(_stage2_text(reviewer_a[key_column]))
    keys_b = set(_stage2_text(reviewer_b[key_column]))
    if keys_a != keys_b:
        raise ValueError(f"{review_name} reviewer record sets differ")

    columns = list(dict.fromkeys((*fixed_columns, *core_columns, *optional_columns)))
    a = reviewer_a[columns].copy()
    b = reviewer_b[columns].copy()
    a[key_column] = _stage2_text(a[key_column])
    b[key_column] = _stage2_text(b[key_column])
    merged = a.merge(b, on=key_column, suffixes=("_A", "_B"), validate="one_to_one")
    for column in fixed_columns:
        if column == key_column:
            continue
        left = _stage2_text(merged[f"{column}_A"])
        right = _stage2_text(merged[f"{column}_B"])
        if not left.equals(right):
            count = int((left != right).sum())
            raise ValueError(f"{review_name} fixed column {column} differs in {count} rows")

    output = pd.DataFrame({key_column: merged[key_column]})
    for column in fixed_columns:
        if column != key_column:
            output[column] = merged[f"{column}_A"]
    for column in (*core_columns, *optional_columns):
        output[f"reviewer_A_{column}"] = merged[f"{column}_A"]
        output[f"reviewer_B_{column}"] = merged[f"{column}_B"]
    return output.sort_values(key_column, kind="stable").reset_index(drop=True)


def _stage2_safe_kappa(a: pd.Series, b: pd.Series) -> float:
    values = set(a.tolist()) | set(b.tolist())
    if len(values) < 2:
        return float("nan")
    return float(cohen_kappa_score(a, b))


def _stage2_interrater_rows(
    frame: pd.DataFrame,
    *,
    review_type: str,
    fields: tuple[str, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    scopes: list[tuple[str, pd.DataFrame]] = [("all", frame)]
    scopes.extend(
        (str(sample_type), part)
        for sample_type, part in frame.groupby("sample_type", sort=False)
    )
    for sample_type, part in scopes:
        for field in fields:
            a = _stage2_text(part[f"reviewer_A_{field}"]).str.lower()
            b = _stage2_text(part[f"reviewer_B_{field}"]).str.lower()
            rows.append(
                {
                    "review_type": review_type,
                    "sample_type": sample_type,
                    "field": field,
                    "n_records": int(len(part)),
                    "n_disagreement": int((a != b).sum()),
                    "exact_agreement": float((a == b).mean()) if len(part) else float("nan"),
                    "cohen_kappa": _stage2_safe_kappa(a, b),
                    "reviewer_A_yes_rate": float(a.eq("yes").mean()) if len(part) else float("nan"),
                    "reviewer_B_yes_rate": float(b.eq("yes").mean()) if len(part) else float("nan"),
                }
            )
    return rows


def _stage2_consensus_template(
    frame: pd.DataFrame,
    *,
    consensus_fields: tuple[str, ...],
) -> pd.DataFrame:
    output = frame.copy()
    disagreement_columns = []
    for field in consensus_fields:
        a = _stage2_text(output[f"reviewer_A_{field}"])
        b = _stage2_text(output[f"reviewer_B_{field}"])
        agreed = a.eq(b)
        output[f"consensus_{field}"] = a.where(agreed, "")
        disagreement_columns.append((field, ~agreed))
    output["disagreement_fields"] = [
        ";".join(field for field, mask in disagreement_columns if bool(mask.iloc[index]))
        for index in range(len(output))
    ]
    output["adjudicator_note"] = ""
    return output


def run_stage2_validate(
    *,
    evidence_reviewer_a_path: Path,
    evidence_reviewer_b_path: Path,
    omission_reviewer_a_path: Path,
    omission_reviewer_b_path: Path,
    blind_key_path: Path,
    output_dir: Path,
    force: bool = False,
) -> dict[str, object]:
    evidence_a = _read(evidence_reviewer_a_path)
    evidence_b = _read(evidence_reviewer_b_path)
    omission_a = _read(omission_reviewer_a_path)
    omission_b = _read(omission_reviewer_b_path)
    blind_key = _read(blind_key_path)
    if not {"blind_id", "sample_type"}.issubset(blind_key.columns):
        raise ValueError("blind key must contain blind_id and sample_type")
    if blind_key["blind_id"].duplicated().any():
        raise ValueError("blind key has duplicate blind_id")

    evidence = _stage2_merge_reviewers(
        evidence_a,
        evidence_b,
        key_column="dp_evidence_id",
        fixed_columns=STAGE2_EVIDENCE_FIXED_COLUMNS,
        core_columns=STAGE2_EVIDENCE_CORE_COLUMNS,
        optional_columns=(
            "duplicate_of_evidence_id",
            "human_evidence_match",
            "error_notes",
        ),
        review_name="evidence",
    )
    omissions = _stage2_merge_reviewers(
        omission_a,
        omission_b,
        key_column="human_evidence_id",
        fixed_columns=STAGE2_OMISSION_FIXED_COLUMNS,
        core_columns=STAGE2_OMISSION_CORE_COLUMNS,
        optional_columns=("matched_dp_evidence_ids", "omission_reason"),
        review_name="omission",
    )
    key = blind_key[["blind_id", "sample_type"]].copy()
    evidence = evidence.merge(key, on="blind_id", how="left", validate="many_to_one")
    omissions = omissions.merge(key, on="blind_id", how="left", validate="many_to_one")
    if evidence["sample_type"].isna().any() or omissions["sample_type"].isna().any():
        raise ValueError("stage2 review contains blind_id absent from blind key")

    metrics = pd.DataFrame(
        _stage2_interrater_rows(
            evidence,
            review_type="evidence",
            fields=STAGE2_EVIDENCE_CORE_COLUMNS,
        )
        + _stage2_interrater_rows(
            omissions,
            review_type="omission",
            fields=STAGE2_OMISSION_CORE_COLUMNS,
        )
    )
    evidence_template = _stage2_consensus_template(
        evidence,
        consensus_fields=(*STAGE2_EVIDENCE_CORE_COLUMNS, "duplicate_of_evidence_id"),
    )
    omission_template = _stage2_consensus_template(
        omissions,
        consensus_fields=STAGE2_OMISSION_CORE_COLUMNS,
    )
    evidence_disagreements = evidence_template.loc[
        evidence_template["disagreement_fields"].ne("")
    ].copy()
    omission_disagreements = omission_template.loc[
        omission_template["disagreement_fields"].ne("")
    ].copy()

    _prepare_output(output_dir, force=force)
    _write(metrics, output_dir / "stage2_interrater_metrics.csv")
    _write(evidence_template, output_dir / "stage2_evidence_consensus_template.csv")
    _write(omission_template, output_dir / "stage2_omission_consensus_template.csv")
    _write(evidence_disagreements, output_dir / "stage2_evidence_disagreements.csv")
    _write(omission_disagreements, output_dir / "stage2_omission_disagreements.csv")
    return _manifest(
        output_dir,
        {
            "status": "stage2_interrater_scored_requires_adjudication",
            "stage": "stage2-validate",
            "automatic_adjudication_performed": False,
            "evidence_disagreement_rows": int(len(evidence_disagreements)),
            "omission_disagreement_rows": int(len(omission_disagreements)),
            "main_and_enriched_reported_separately": True,
            "sentinel_included_in_primary_inference": False,
            "evidence_reviewer_A_sha256": _sha256(evidence_reviewer_a_path),
            "evidence_reviewer_B_sha256": _sha256(evidence_reviewer_b_path),
            "omission_reviewer_A_sha256": _sha256(omission_reviewer_a_path),
            "omission_reviewer_B_sha256": _sha256(omission_reviewer_b_path),
            "blind_key_sha256": _sha256(blind_key_path),
            "next_gate": "adjudicate only blank consensus fields before final evidence-quality scoring",
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Score packet-3 D-P validation")
    parser.add_argument(
        "--stage",
        choices=("validate", "freeze", "compare", "stage2-validate", "evidence"),
        required=True,
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_PUBLIC_PACKET / "blind_cases.csv")
    parser.add_argument("--reviewer-a", type=Path)
    parser.add_argument("--reviewer-b", type=Path)
    parser.add_argument("--consensus", type=Path)
    parser.add_argument("--freeze-record", type=Path)
    parser.add_argument("--scoring-reference", type=Path)
    parser.add_argument("--evidence-review", type=Path)
    parser.add_argument("--omission-review", type=Path)
    parser.add_argument("--evidence-reviewer-a", type=Path)
    parser.add_argument("--evidence-reviewer-b", type=Path)
    parser.add_argument("--omission-reviewer-a", type=Path)
    parser.add_argument("--omission-reviewer-b", type=Path)
    parser.add_argument("--blind-key", type=Path, default=DEFAULT_PRIVATE_PACKET / "blind_id_key.csv")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-n", type=int, default=5000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.stage == "validate":
        if args.reviewer_a is None or args.reviewer_b is None or args.output_dir is None:
            parser.error("validate requires --reviewer-a, --reviewer-b, and --output-dir")
        manifest = run_validate(
            cases_path=args.cases,
            reviewer_a_path=args.reviewer_a,
            reviewer_b_path=args.reviewer_b,
            output_dir=args.output_dir,
            force=args.force,
        )
    elif args.stage == "freeze":
        if (
            args.consensus is None
            or args.reviewer_a is None
            or args.reviewer_b is None
            or args.freeze_record is None
        ):
            parser.error(
                "freeze requires --consensus, --reviewer-a, --reviewer-b, and --freeze-record"
            )
        manifest = run_freeze(
            consensus_path=args.consensus,
            reviewer_a_path=args.reviewer_a,
            reviewer_b_path=args.reviewer_b,
            freeze_record_path=args.freeze_record,
        )
    elif args.stage == "compare":
        if (
            args.consensus is None
            or args.freeze_record is None
            or args.scoring_reference is None
            or args.output_dir is None
        ):
            parser.error(
                "compare requires --consensus, --freeze-record, --scoring-reference, and --output-dir"
            )
        manifest = run_compare(
            consensus_path=args.consensus,
            freeze_record_path=args.freeze_record,
            scoring_reference_path=args.scoring_reference,
            output_dir=args.output_dir,
            bootstrap_n=args.bootstrap_n,
            force=args.force,
        )
    elif args.stage == "stage2-validate":
        if (
            args.evidence_reviewer_a is None
            or args.evidence_reviewer_b is None
            or args.omission_reviewer_a is None
            or args.omission_reviewer_b is None
            or args.output_dir is None
        ):
            parser.error(
                "stage2-validate requires both reviewers' evidence and omission files plus --output-dir"
            )
        manifest = run_stage2_validate(
            evidence_reviewer_a_path=args.evidence_reviewer_a,
            evidence_reviewer_b_path=args.evidence_reviewer_b,
            omission_reviewer_a_path=args.omission_reviewer_a,
            omission_reviewer_b_path=args.omission_reviewer_b,
            blind_key_path=args.blind_key,
            output_dir=args.output_dir,
            force=args.force,
        )
    else:
        if (
            args.evidence_review is None
            or args.omission_review is None
            or args.output_dir is None
        ):
            parser.error(
                "evidence requires --evidence-review, --omission-review, and --output-dir"
            )
        manifest = run_evidence(
            evidence_review_path=args.evidence_review,
            omission_review_path=args.omission_review,
            blind_key_path=args.blind_key,
            output_dir=args.output_dir,
            force=args.force,
        )
    print(json.dumps({"status": manifest["status"], "stage": args.stage}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
