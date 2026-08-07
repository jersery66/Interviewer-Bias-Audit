from pathlib import Path

import pandas as pd
import pytest

from reanalysis_v2.c5_blind_workflow_v3 import (
    build_stage1_consensus_template,
    build_stage1_review_form,
)
from reanalysis_v2.embedding_incremental import DOMAIN_FEATURES
from scripts.score_c5_blind_audit_v3 import (
    run_compare,
    run_evidence,
    run_freeze,
    run_stage2_validate,
    run_validate,
)


def _completed_form(reviewer_id: str, blind_ids: list[str]) -> pd.DataFrame:
    form = build_stage1_review_form(blind_ids, reviewer_id=reviewer_id, order_seed=1)
    form["manual_status"] = "indeterminate"
    form["indeterminate_reason"] = "not_mentioned"
    form["count_bin"] = "0"
    form["temporality"] = "not_applicable"
    form["negation"] = "not_applicable"
    form["subject"] = "not_applicable"
    target = (form["blind_id"] == "V001") & (form["domain"] == "depressed_mood")
    form.loc[target, "manual_status"] = "current_present"
    form.loc[target, "indeterminate_reason"] = pd.NA
    form.loc[target, "count_bin"] = "1"
    form.loc[target, "evidence_quote"] = "I feel sad"
    form.loc[target, "line_number"] = "1"
    form.loc[target, "temporality"] = "current"
    form.loc[target, "negation"] = "affirmed"
    form.loc[target, "subject"] = "participant"
    return form


@pytest.mark.filterwarnings("error")
def test_scoring_v3_enforces_validate_freeze_compare_and_separate_outputs(tmp_path):
    blind_ids = ["V001", "V002", "V003", "V004"]
    cases = pd.DataFrame(
        {"blind_id": blind_ids, "participant_text": ["I feel sad", "fine", "none", "fine"]}
    )
    reviewer_a = _completed_form("A", blind_ids)
    reviewer_b = _completed_form("B", blind_ids)
    cases_path = tmp_path / "cases.csv"
    a_path = tmp_path / "a.csv"
    b_path = tmp_path / "b.csv"
    cases.to_csv(cases_path, index=False)
    reviewer_a.to_csv(a_path, index=False)
    reviewer_b.to_csv(b_path, index=False)

    validate_dir = tmp_path / "validate"
    validate_manifest = run_validate(
        cases_path=cases_path,
        reviewer_a_path=a_path,
        reviewer_b_path=b_path,
        output_dir=validate_dir,
    )
    assert validate_manifest["status"] == "stage1_independent_agreement_scored"
    assert (validate_dir / "stage1_interrater_metrics.csv").exists()

    consensus = build_stage1_consensus_template(reviewer_a, reviewer_b)
    consensus["consensus_status"] = consensus["reviewer_A_manual_status"]
    consensus["consensus_indeterminate_reason"] = consensus["reviewer_A_indeterminate_reason"]
    consensus["consensus_count_bin"] = consensus["reviewer_A_count_bin"]
    consensus["consensus_evidence_quote"] = consensus["reviewer_A_evidence_quote"]
    consensus["consensus_line_number"] = consensus["reviewer_A_line_number"]
    consensus["consensus_temporality"] = consensus["reviewer_A_temporality"]
    consensus["consensus_negation"] = consensus["reviewer_A_negation"]
    consensus["consensus_subject"] = consensus["reviewer_A_subject"]
    consensus["adjudication_reason"] = "reviewers agreed"
    consensus_path = tmp_path / "consensus.csv"
    freeze_path = tmp_path / "consensus_freeze.json"
    consensus.to_csv(consensus_path, index=False)
    run_freeze(
        consensus_path=consensus_path,
        reviewer_a_path=a_path,
        reviewer_b_path=b_path,
        freeze_record_path=freeze_path,
    )
    assert freeze_path.exists()

    reference_rows = []
    for blind_id, sample_type in (
        ("V001", "main"),
        ("V002", "main"),
        ("V003", "sentinel"),
        ("V004", "enriched"),
    ):
        row = {
            "blind_id": blind_id,
            "participant_id": int(blind_id[-1]),
            "sample_type": sample_type,
        }
        for domain in DOMAIN_FEATURES:
            row[domain] = 0
        if blind_id == "V001":
            row["depressed_mood"] = 1
        reference_rows.append(row)
    reference_path = tmp_path / "reference.csv"
    pd.DataFrame(reference_rows).to_csv(reference_path, index=False)
    compare_dir = tmp_path / "compare"
    compare_manifest = run_compare(
        consensus_path=consensus_path,
        freeze_record_path=freeze_path,
        scoring_reference_path=reference_path,
        output_dir=compare_dir,
        bootstrap_n=20,
    )
    assert compare_manifest["status"] == "dp_validity_scored_main_enriched_separate"
    metrics = pd.read_csv(compare_dir / "dp_presence_validity_metrics.csv")
    assert set(metrics["sample_type"]) == {"main", "enriched"}
    sentinel = pd.read_csv(compare_dir / "sentinel_case_review.csv")
    assert set(sentinel["sample_type"]) == {"sentinel"}

    key_path = tmp_path / "blind_key.csv"
    pd.DataFrame(reference_rows)[["blind_id", "participant_id", "sample_type"]].to_csv(
        key_path, index=False
    )
    evidence = pd.DataFrame(
        {
            "blind_id": ["V001", "V004"],
            "source_supported": ["yes", "yes"],
            "speaker_correct": ["yes", "yes"],
            "domain_correct": ["yes", "no"],
            "negation_preserved": ["yes", "yes"],
            "temporality_correct": ["yes", "yes"],
            "subject_correct": ["yes", "yes"],
            "context_preserved": ["yes", "yes"],
            "duplicate_of_evidence_id": [pd.NA, pd.NA],
        }
    )
    omissions = pd.DataFrame(
        {
            "blind_id": ["V001", "V004"],
            "covered_by_dp": ["yes", "no"],
            "important_omission": ["no", "yes"],
        }
    )
    evidence_path = tmp_path / "evidence.csv"
    omissions_path = tmp_path / "omissions.csv"
    evidence.to_csv(evidence_path, index=False)
    omissions.to_csv(omissions_path, index=False)
    evidence_dir = tmp_path / "evidence_results"
    evidence_manifest = run_evidence(
        evidence_review_path=evidence_path,
        omission_review_path=omissions_path,
        blind_key_path=key_path,
        output_dir=evidence_dir,
    )
    assert evidence_manifest["status"] == "stage2_evidence_quality_scored"
    quality = pd.read_csv(evidence_dir / "stage2_evidence_quality.csv")
    assert set(quality["sample_type"]) == {"main", "enriched"}


def test_stage2_validate_scores_interrater_and_leaves_disagreements_unadjudicated(
    tmp_path,
):
    evidence_base = {
        "blind_id": ["V001"],
        "domain": ["depressed_mood"],
        "dp_evidence_id": ["V001-DP-001"],
        "dp_quote": ["I feel sad"],
        "dp_line_number": ["1"],
        "dp_polarity": ["positive"],
        "source_supported": ["yes"],
        "speaker_correct": ["yes"],
        "domain_correct": ["yes"],
        "negation_preserved": ["yes"],
        "temporality_correct": ["yes"],
        "subject_correct": ["yes"],
        "context_preserved": ["yes"],
        "duplicate_of_evidence_id": [pd.NA],
        "human_evidence_match": [pd.NA],
        "error_notes": [pd.NA],
    }
    evidence_a = pd.DataFrame({"reviewer_id": ["A"], **evidence_base})
    evidence_b = pd.DataFrame({"reviewer_id": ["B"], **evidence_base})
    evidence_b.loc[0, "domain_correct"] = "no"

    omission_base = {
        "blind_id": ["V001"],
        "domain": ["depressed_mood"],
        "human_evidence_id": ["V001-H-depr-01"],
        "human_evidence_quote": ["I feel sad"],
        "human_line_number": ["1"],
        "same_cell_dp_evidence_ids": ["V001-DP-001"],
        "covered_by_dp": ["no"],
        "matched_dp_evidence_ids": [pd.NA],
        "important_omission": ["yes"],
        "omission_reason": ["changes presence"],
    }
    omission_a = pd.DataFrame({"reviewer_id": ["A"], **omission_base})
    omission_b = pd.DataFrame({"reviewer_id": ["B"], **omission_base})
    omission_b.loc[0, "important_omission"] = "no"
    omission_b.loc[0, "omission_reason"] = pd.NA

    paths = {}
    for name, frame in (
        ("evidence_a", evidence_a),
        ("evidence_b", evidence_b),
        ("omission_a", omission_a),
        ("omission_b", omission_b),
    ):
        paths[name] = tmp_path / f"{name}.csv"
        frame.to_csv(paths[name], index=False)
    key_path = tmp_path / "blind_key.csv"
    pd.DataFrame(
        {"blind_id": ["V001"], "participant_id": [1], "sample_type": ["main"]}
    ).to_csv(key_path, index=False)

    output_dir = tmp_path / "stage2_validate"
    manifest = run_stage2_validate(
        evidence_reviewer_a_path=paths["evidence_a"],
        evidence_reviewer_b_path=paths["evidence_b"],
        omission_reviewer_a_path=paths["omission_a"],
        omission_reviewer_b_path=paths["omission_b"],
        blind_key_path=key_path,
        output_dir=output_dir,
    )

    assert manifest["status"] == "stage2_interrater_scored_requires_adjudication"
    metrics = pd.read_csv(output_dir / "stage2_interrater_metrics.csv")
    domain_row = metrics.loc[
        metrics["review_type"].eq("evidence")
        & metrics["field"].eq("domain_correct")
        & metrics["sample_type"].eq("main")
    ].iloc[0]
    assert domain_row["n_disagreement"] == 1
    omission_row = metrics.loc[
        metrics["review_type"].eq("omission")
        & metrics["field"].eq("important_omission")
        & metrics["sample_type"].eq("main")
    ].iloc[0]
    assert omission_row["n_disagreement"] == 1

    evidence_template = pd.read_csv(
        output_dir / "stage2_evidence_consensus_template.csv", keep_default_na=False
    )
    assert evidence_template.loc[0, "consensus_source_supported"] == "yes"
    assert evidence_template.loc[0, "consensus_domain_correct"] == ""
    omission_template = pd.read_csv(
        output_dir / "stage2_omission_consensus_template.csv", keep_default_na=False
    )
    assert omission_template.loc[0, "consensus_covered_by_dp"] == "no"
    assert omission_template.loc[0, "consensus_important_omission"] == ""
    assert len(pd.read_csv(output_dir / "stage2_evidence_disagreements.csv")) == 1
    assert len(pd.read_csv(output_dir / "stage2_omission_disagreements.csv")) == 1
    assert not (output_dir / "stage2_evidence_quality.csv").exists()
