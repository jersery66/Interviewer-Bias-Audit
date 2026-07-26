import json

import pandas as pd
import pytest

from reanalysis_v2.embedding_incremental import DOMAIN_FEATURES
from reanalysis_v2.c5_blind_workflow_v3 import (
    COMMON_SYMPTOM_STATUSES,
    MENTAL_HEALTH_HISTORY_STATUSES,
    PROTECTIVE_STATUSES,
    build_stage1_consensus_template,
    build_stage1_review_form,
    build_stage2_evidence_forms,
    build_validation_sample,
    compute_count_validity,
    compute_dp_validity,
    compute_stage1_interrater,
    compute_stage2_quality,
    freeze_consensus,
    make_opaque_blind_key,
    manual_status_to_binary,
    select_enrichment_sample,
    validate_stage1_consensus,
    validate_stage1_reviewer_form,
    verify_consensus_freeze,
)


def _domain_counts() -> pd.DataFrame:
    rows = []
    for participant_id in range(1, 61):
        row = {"participant_id": participant_id}
        for domain in DOMAIN_FEATURES:
            row[domain] = 0
        if participant_id in {1, 11, 12, 13, 14, 15, 16, 17}:
            row["appetite_weight"] = 1
        if participant_id in {2, 11, 18, 19, 20, 21, 22}:
            row["suicide_self_harm"] = 1
        if participant_id in {3, 12, 18, 23, 24, 25, 26}:
            row["anhedonia_interest"] = 1
        row["sleep_fatigue_energy"] = 0 if participant_id in {4, 13, 19, 23, 27, 28, 29} else 1
        row["protective_or_absent_symptom"] = 0 if participant_id in {5, 14, 20, 24, 30} else 1
        rows.append(row)
    return pd.DataFrame(rows)


def test_enrichment_is_separate_deterministic_and_covers_rare_states():
    domains = _domain_counts()
    main_ids = list(range(1, 7))
    training_ids = list(range(7, 11))

    first = select_enrichment_sample(
        domains,
        excluded_participant_ids={*main_ids, *training_ids, 60},
        target_quota=4,
        min_n=10,
        max_n=15,
    )
    second = select_enrichment_sample(
        domains,
        excluded_participant_ids={*main_ids, *training_ids, 60},
        target_quota=4,
        min_n=10,
        max_n=15,
    )

    assert first.equals(second)
    assert 10 <= len(first) <= 15
    assert not set(first["participant_id"]) & {*main_ids, *training_ids, 60}
    for target in (
        "appetite_weight_positive",
        "suicide_self_harm_positive",
        "anhedonia_interest_positive",
        "sleep_fatigue_energy_zero",
        "protective_or_absent_symptom_zero",
    ):
        assert first[target].sum() >= 4


def test_validation_sample_preserves_main_and_separates_enriched_and_sentinel():
    main = pd.DataFrame({"participant_id": [1, 2, 3], "audit_case_id": ["old1", "old2", "old3"]})
    enriched = pd.DataFrame({"participant_id": [11, 12]})
    sample = build_validation_sample(main, enriched, sentinel_participant_id=60)

    assert set(sample.loc[sample["sample_type"] == "main", "participant_id"]) == {1, 2, 3}
    assert set(sample.loc[sample["sample_type"] == "enriched", "participant_id"]) == {11, 12}
    assert sample.loc[sample["sample_type"] == "sentinel", "participant_id"].tolist() == [60]
    key = make_opaque_blind_key(sample, seed=20260721)
    assert key["blind_id"].str.match(r"V\d{3}").all()
    assert key["blind_id"].is_unique


def _completed_stage1_form() -> pd.DataFrame:
    form = build_stage1_review_form(["V001"], reviewer_id="A", order_seed=1)
    form["manual_status"] = "indeterminate"
    form["indeterminate_reason"] = "not_mentioned"
    form["count_bin"] = "0"
    form["temporality"] = "not_applicable"
    form["negation"] = "not_applicable"
    form["subject"] = "not_applicable"

    symptom = form["domain"] == "depressed_mood"
    form.loc[symptom, "manual_status"] = "current_present"
    form.loc[symptom, "indeterminate_reason"] = pd.NA
    form.loc[symptom, "count_bin"] = "1"
    form.loc[symptom, "evidence_quote"] = "I feel sad"
    form.loc[symptom, "line_number"] = "2"
    form.loc[symptom, "temporality"] = "current"
    form.loc[symptom, "negation"] = "affirmed"
    form.loc[symptom, "subject"] = "participant"

    history = form["domain"] == "mental_health_history"
    form.loc[history, "manual_status"] = "history_present"
    form.loc[history, "indeterminate_reason"] = pd.NA
    form.loc[history, "count_bin"] = "1"
    form.loc[history, "evidence_quote"] = "I saw a therapist"
    form.loc[history, "line_number"] = "4"
    form.loc[history, "temporality"] = "past"
    form.loc[history, "negation"] = "affirmed"
    form.loc[history, "subject"] = "participant"

    protective = form["domain"] == "protective_or_absent_symptom"
    form.loc[protective, "manual_status"] = "protective_denial_evidence_present"
    form.loc[protective, "indeterminate_reason"] = pd.NA
    form.loc[protective, "count_bin"] = "1"
    form.loc[protective, "evidence_quote"] = "I would never hurt myself"
    form.loc[protective, "line_number"] = "6"
    form.loc[protective, "temporality"] = "current"
    form.loc[protective, "negation"] = "explicitly_negated"
    form.loc[protective, "subject"] = "participant"
    return form


def test_stage1_uses_domain_specific_statuses_and_contains_no_dp_fields():
    form = _completed_stage1_form()
    assert "participant_id" not in form.columns
    assert "sample_type" not in form.columns
    assert not set(DOMAIN_FEATURES).intersection(form.columns)
    validate_stage1_reviewer_form(form, expected_blind_ids=["V001"], require_completed=True)
    assert manual_status_to_binary("depressed_mood", "current_present") == 1
    assert manual_status_to_binary("depressed_mood", "past_only") == 0
    assert manual_status_to_binary("mental_health_history", "history_present") == 1
    assert manual_status_to_binary("protective_or_absent_symptom", "protective_denial_evidence_present") == 1
    assert manual_status_to_binary("depressed_mood", "conflicting") is None
    assert "current_present" in COMMON_SYMPTOM_STATUSES
    assert "history_present" in MENTAL_HEALTH_HISTORY_STATUSES
    assert "protective_denial_evidence_present" in PROTECTIVE_STATUSES


def test_consensus_must_freeze_before_stage2_and_actual_span_rows_are_expanded(tmp_path):
    reviewer_a = _completed_stage1_form()
    reviewer_b = reviewer_a.copy()
    reviewer_b["reviewer_id"] = "B"
    consensus = build_stage1_consensus_template(reviewer_a, reviewer_b)
    consensus["consensus_status"] = consensus["reviewer_A_manual_status"]
    consensus["consensus_count_bin"] = consensus["reviewer_A_count_bin"]
    consensus["consensus_evidence_quote"] = consensus["reviewer_A_evidence_quote"]
    consensus["consensus_line_number"] = consensus["reviewer_A_line_number"]
    consensus["consensus_temporality"] = consensus["reviewer_A_temporality"]
    consensus["consensus_negation"] = consensus["reviewer_A_negation"]
    consensus["consensus_subject"] = consensus["reviewer_A_subject"]
    consensus["consensus_indeterminate_reason"] = consensus["reviewer_A_indeterminate_reason"]
    consensus["adjudication_reason"] = "reviewers agreed"
    validate_stage1_consensus(consensus, reviewer_a, reviewer_b)

    consensus_path = tmp_path / "consensus.csv"
    freeze_path = tmp_path / "consensus_freeze.json"
    consensus.to_csv(consensus_path, index=False)
    with pytest.raises(FileNotFoundError):
        verify_consensus_freeze(consensus_path, freeze_path)
    freeze_consensus(consensus_path, freeze_path)
    verify_consensus_freeze(consensus_path, freeze_path)

    key = pd.DataFrame({"blind_id": ["V001"], "participant_id": [101], "sample_type": ["main"]})
    spans = pd.DataFrame(
        {
            "participant_id": [101, 101, 999],
            "source_order": [1, 2, 1],
            "domain": ["depressed_mood", "depressed_mood", "depressed_mood"],
            "polarity": ["present", "present", "present"],
            "exact_quote": ["I feel sad", "Nothing is fun", "not selected"],
        }
    )
    lines = pd.DataFrame(
        {
            "blind_id": ["V001", "V001"],
            "line_number": [1, 2],
            "participant_text_line": ["Nothing is fun", "I feel sad"],
        }
    )
    evidence, omissions = build_stage2_evidence_forms(
        consensus,
        key,
        spans,
        lines,
        reviewer_id="A",
    )
    assert len(evidence) == 2
    assert evidence["dp_evidence_id"].is_unique
    assert set(evidence["dp_line_number"].astype(str)) == {"1", "2"}
    assert not (evidence["dp_quote"] == "not selected").any()
    assert len(omissions) >= 1


def test_primary_metrics_are_main_only_and_count_is_secondary():
    consensus_rows = []
    reference_rows = []
    for blind_id, sample_type, human, dp in (
        ("V001", "main", 1, 1),
        ("V002", "main", 0, 0),
        ("V005", "main", "conflicting", 1),
        ("V003", "enriched", 1, 0),
        ("V004", "sentinel", 0, 0),
    ):
        status = (
            "conflicting"
            if human == "conflicting"
            else ("current_present" if human else "indeterminate")
        )
        consensus_rows.append(
            {
                "blind_id": blind_id,
                "domain": "depressed_mood",
                "consensus_status": status,
                "consensus_count_bin": "1" if human in {1, "conflicting"} else "0",
            }
        )
        row = {"blind_id": blind_id, "sample_type": sample_type, "participant_id": int(blind_id[-1])}
        for domain in DOMAIN_FEATURES:
            row[domain] = 0
        row["depressed_mood"] = dp
        reference_rows.append(row)
    consensus = pd.DataFrame(consensus_rows)
    reference = pd.DataFrame(reference_rows)

    metrics = compute_dp_validity(consensus, reference, bootstrap_n=50, seed=7)
    assert set(metrics["conflict_policy"]) == {"exclude", "as_0", "as_1"}
    main_overall = metrics.loc[
        (metrics["sample_type"] == "main")
        & (metrics["scope"] == "overall")
        & (metrics["conflict_policy"] == "exclude")
    ].iloc[0]
    enriched_overall = metrics.loc[
        (metrics["sample_type"] == "enriched")
        & (metrics["scope"] == "overall")
        & (metrics["conflict_policy"] == "exclude")
    ].iloc[0]
    assert main_overall["n_cells"] == 2
    assert main_overall["sensitivity"] == 1.0
    assert main_overall["specificity"] == 1.0
    assert enriched_overall["sensitivity"] == 0.0
    assert not (metrics["sample_type"] == "sentinel").any()

    count = compute_count_validity(consensus, reference)
    assert {"exact_agreement", "weighted_kappa", "mean_absolute_difference", "spearman_rho"} <= set(count.columns)


def test_stage1_interrater_and_stage2_error_types_are_reported_separately():
    reviewer_a = _completed_stage1_form()
    reviewer_b = reviewer_a.copy()
    reviewer_b["reviewer_id"] = "B"
    target = reviewer_b["domain"] == "depressed_mood"
    reviewer_b.loc[target, "manual_status"] = "explicitly_absent"
    reviewer_b.loc[target, "temporality"] = "current"
    reviewer_b.loc[target, "negation"] = "explicitly_negated"
    agreement = compute_stage1_interrater(reviewer_a, reviewer_b)
    assert {"status_cohen_kappa", "binary_cohen_kappa", "count_weighted_kappa"} <= set(agreement.columns)
    assert "overall" in set(agreement["scope"])

    key = pd.DataFrame(
        {
            "blind_id": ["V001", "V002"],
            "sample_type": ["main", "enriched"],
            "participant_id": [1, 2],
        }
    )
    evidence = pd.DataFrame(
        {
            "blind_id": ["V001", "V001", "V002"],
            "source_supported": ["yes", "no", "yes"],
            "speaker_correct": ["yes", "no", "yes"],
            "domain_correct": ["yes", "yes", "no"],
            "negation_preserved": ["yes", "no", "yes"],
            "temporality_correct": ["yes", "no", "yes"],
            "subject_correct": ["yes", "yes", "no"],
            "context_preserved": ["yes", "no", "yes"],
            "duplicate_of_evidence_id": [pd.NA, "V001-DP-001", pd.NA],
        }
    )
    omissions = pd.DataFrame(
        {
            "blind_id": ["V001", "V002"],
            "covered_by_dp": ["no", "yes"],
            "important_omission": ["yes", "no"],
        }
    )
    quality = compute_stage2_quality(evidence, omissions, key)
    main = quality.loc[quality["sample_type"] == "main"].iloc[0]
    enriched = quality.loc[quality["sample_type"] == "enriched"].iloc[0]
    assert main["speaker_mismatch_rate"] == 0.5
    assert main["unsupported_evidence_rate"] == 0.5
    assert main["important_omission_rate"] == 1.0
    assert enriched["domain_mismatch_rate"] == 1.0
