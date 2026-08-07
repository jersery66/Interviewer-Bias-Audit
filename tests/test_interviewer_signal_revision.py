from __future__ import annotations

import hashlib
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import UndefinedMetricWarning

import reanalysis_v2.interviewer_signal_revision as revision

from reanalysis_v2.interviewer_signal_revision import (
    C5_HUMAN_FIELDS,
    build_block_increment_contrasts,
    build_c5_human_validation_dataset,
    build_feature_block_source_table,
    apply_stratified_fake_domain_permutation,
    run_r_absorption_sensitivity,
    select_c5_validation_participants,
    source_model_performance_by_repeat,
    source_model_performance_summary,
)
from reanalysis_v2.interviewer_signal_explanation import (
    D_P_FEATURES,
    ExplanationInputs,
    SourceFoldCache,
)
from reanalysis_v2.splits import make_repeated_split_membership, split_indices
from scripts.compute_c5_human_machine_agreement import main as compute_c5_agreement_main


def _synthetic_c5_spans() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "participant_id": [101, 101, 102],
            "exact_quote": ["I cannot sleep", "I feel guilty", "I am okay"],
            "domain": ["sleep_fatigue_energy", "self_worth_guilt", "protective_or_absent_symptom"],
            "polarity": ["present", "present", "absent"],
            "review_status": ["reviewed"] * 3,
            "original_model_quote": ["I cannot sleep", "I feel guilty", "I am okay"],
            "source_text_sha256": ["a", "b", "c"],
            "quote_sha256": ["qa", "qb", "qc"],
            "source_turn_index": [1, 2, 1],
            "source_char_start": [0, 5, 0],
            "source_char_end": [13, 17, 9],
        }
    )


def _synthetic_sampling_inputs(n: int = 36) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = np.arange(1000, 1000 + n)
    labels = np.asarray([0, 1] * (n // 2), dtype=int)
    structural = pd.DataFrame(
        {
            "participant_id": ids,
            "label": labels,
            "length_only__participant_word_count": np.arange(10, 10 + n),
        }
    )
    domains = pd.DataFrame({"participant_id": ids})
    for index, name in enumerate(
        [
            "anhedonia_interest",
            "appetite_weight",
            "concentration_psychomotor",
            "depressed_mood",
            "functioning_impairment",
            "mental_health_history",
            "protective_or_absent_symptom",
            "self_worth_guilt",
            "sleep_fatigue_energy",
            "suicide_self_harm",
        ]
    ):
        domains[name] = ((ids + index) % (index + 2) == 0).astype(float)
    return structural, domains


def _synthetic_stratified_fake_oof() -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = np.asarray([11, 12, 13, 14], dtype=int)
    labels = np.asarray([0, 0, 1, 1], dtype=int)
    explanation_rows: list[dict[str, object]] = []
    residual_rows: list[dict[str, object]] = []
    for domain_type, draw, prediction_shift, enhanced_shift in (
        ("real", 0, 0.0, 0.25),
        ("fake", 1, 0.35, 0.06),
        ("fake", 2, -0.25, 0.02),
    ):
        for participant_id, label in zip(ids, labels):
            interviewer_logit = float(label) + participant_id / 100.0
            explanation_rows.append(
                {
                    "participant_id": int(participant_id),
                    "label": int(label),
                    "representation": "tfidf",
                    "repeat": 1,
                    "domain_type": domain_type,
                    "draw": draw,
                    "interviewer_logit": interviewer_logit,
                    "predicted_interviewer_logit": interviewer_logit - prediction_shift,
                    "outer_train_mean_null_prediction": 0.5,
                }
            )
            base_probability = 0.25 + 0.5 * float(label)
            residual_rows.append(
                {
                    "participant_id": int(participant_id),
                    "label": int(label),
                    "representation": "tfidf",
                    "repeat": 1,
                    "domain_type": domain_type,
                    "draw": draw,
                    "base_probability": base_probability,
                    "enhanced_probability": min(0.99, base_probability + enhanced_shift * (1 if label else -1)),
                }
            )
    return pd.DataFrame(explanation_rows), pd.DataFrame(residual_rows)


def test_c5_candidate_contains_model_fields_and_blank_human_fields() -> None:
    roster = pd.DataFrame({"participant_id": [101, 102]})
    candidate = build_c5_human_validation_dataset(
        _synthetic_c5_spans(), roster, source_sha256="source-hash"
    )

    assert set(C5_HUMAN_FIELDS).issubset(candidate.columns)
    assert candidate["model_exact_quote"].notna().all()
    for field in C5_HUMAN_FIELDS:
        assert candidate[field].fillna("").eq("").all(), field
    assert "transcript" not in " ".join(candidate.columns).lower()
    assert candidate["source_sha256"].eq("source-hash").all()
    assert list(candidate.loc[:, ["participant_id", "label", "span_id", "model_exact_quote", "model_domain", "model_polarity", *C5_HUMAN_FIELDS]].columns) == [
        "participant_id",
        "label",
        "span_id",
        "model_exact_quote",
        "model_domain",
        "model_polarity",
        *C5_HUMAN_FIELDS,
    ]
    assert not set(candidate.columns).intersection(
        {"exact_quote", "original_model_quote", "speaker_text", "transcript_text"}
    )


def test_c5_candidate_is_deterministic_and_does_not_infer_human_labels() -> None:
    roster = pd.DataFrame({"participant_id": [101, 102]})
    first = build_c5_human_validation_dataset(
        _synthetic_c5_spans(), roster, source_sha256="source-hash"
    )
    second = build_c5_human_validation_dataset(
        _synthetic_c5_spans(), roster, source_sha256="source-hash"
    )
    pd.testing.assert_frame_equal(first, second)
    assert first["human_domain"].eq("").all()
    assert first["human_polarity"].eq("").all()
    assert first["human_span_valid"].eq("").all()


def test_c5_sampling_is_deterministic_and_label_stratified_without_model_fields() -> None:
    structural, domains = _synthetic_sampling_inputs()
    evidence_counts = pd.Series(np.arange(len(structural)) % 7 + 1, index=structural["participant_id"])
    first = select_c5_validation_participants(structural, domains, n=12, evidence_counts=evidence_counts)
    second = select_c5_validation_participants(structural, domains, n=12, evidence_counts=evidence_counts)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 12
    assert set(first["label"]) == {0, 1}
    assert set(first["evidence_count_tertile"]) >= {1, 2, 3}
    assert first["selection_rule"].str.contains("evidence-count tertile", regex=False).all()
    assert "model_correct" not in first.columns
    assert "source_score" not in first.columns


def test_block_source_table_locks_r_as_protocol_structure() -> None:
    class Inputs:
        block_columns = {
            "P": ("participant_logit",),
            "Q": ("q_length",),
            "R": ("clinical_question_count", "prompt_presence_x"),
            "D": ("sleep_fatigue_energy",),
        }

    table = build_feature_block_source_table(Inputs())
    r = table.loc[table["block"].eq("R")]
    assert set(r["feature"]) == {"clinical_question_count", "prompt_presence_x"}
    assert r["source_side"].eq("interviewer-side protocol structure").all()
    assert not table.astype(str).apply(lambda col: col.str.contains("causal", case=False).any()).any()


def test_block_increment_contrasts_are_derived_from_locked_subset_r2() -> None:
    rows = []
    values = {
        "P": 0.10,
        "Q": 0.20,
        "R": 0.30,
        "D": 0.40,
        "P+R": 0.50,
        "R+D": 0.60,
        "P+R+D": 0.70,
        "P+Q+R+D": 0.80,
    }
    for subset, r2 in values.items():
        rows.append({"representation": "tfidf", "repeat": 1, "subset": subset, "r2": r2})
    result = build_block_increment_contrasts(pd.DataFrame(rows))
    row = result.iloc[0]
    assert row["full_r2"] == pytest.approx(0.80)
    assert row["full_minus_R_r2"] == pytest.approx(0.50)
    assert row["P+R_minus_R_r2"] == pytest.approx(0.20)
    assert row["R+D_minus_R_r2"] == pytest.approx(0.30)
    assert row["P+R+D_minus_R_r2"] == pytest.approx(0.40)
    assert row["P+Q+R+D_minus_R_r2"] == pytest.approx(0.50)


def test_block_increment_requires_all_locked_contrasts() -> None:
    with pytest.raises(ValueError, match="missing locked subset"):
        build_block_increment_contrasts(
            pd.DataFrame(
                {
                    "representation": ["tfidf"],
                    "repeat": [1],
                    "subset": ["P+Q+R+D"],
                    "r2": [0.1],
                }
            )
        )


def test_r_absorption_uses_fold_local_residuals_and_reports_locked_models() -> None:
    rng = np.random.default_rng(19)
    n = 30
    ids = np.arange(2000, 2000 + n)
    labels = np.asarray([0] * 15 + [1] * 15)
    q = rng.normal(size=(n, 2))
    r = rng.normal(size=(n, 2))
    d = rng.normal(size=(n, 3))
    structural = pd.DataFrame({"participant_id": ids, "label": labels, "phq8_score": labels})
    domains = pd.DataFrame({"participant_id": ids})
    for index, name in enumerate(D_P_FEATURES):
        domains[name] = d[:, index % 3]
    inputs = ExplanationInputs(
        participant_ids=ids,
        labels=labels,
        structural=structural,
        domains=domains,
        path_features=pd.DataFrame({"participant_id": ids}),
        block_matrices={"Q": q, "R": r, "D": d},
        block_columns={"P": ("participant_logit",), "Q": ("q0", "q1"), "R": ("r0", "r1"), "D": ("d0", "d1", "d2")},
        structural_audit=pd.DataFrame(),
        structural_audit_summary={},
        path_dictionary=(),
        path_audit=pd.DataFrame(),
        path_audit_summary={},
        domain_audit=pd.DataFrame(),
        domain_audit_summary={},
    )
    membership = make_repeated_split_membership(ids, labels, n_repeats=1, n_splits=5, base_seed=41)
    cache = {}
    for fold in range(1, 6):
        train_idx, test_idx = split_indices(membership, 1, fold, ids)
        p_train = labels[train_idx].astype(float) + rng.normal(scale=0.3, size=len(train_idx))
        p_test = labels[test_idx].astype(float) + rng.normal(scale=0.3, size=len(test_idx))
        i_train = 0.4 * labels[train_idx].astype(float) + rng.normal(scale=0.3, size=len(train_idx))
        i_test = 0.4 * labels[test_idx].astype(float) + rng.normal(scale=0.3, size=len(test_idx))
        cache[("tfidf", 1, fold)] = SourceFoldCache(
            train_idx=train_idx,
            test_idx=test_idx,
            participant_train_logit=p_train,
            participant_test_logit=p_test,
            interviewer_train_logit=i_train,
            interviewer_test_logit=i_test,
            participant_c=1.0,
            interviewer_c=1.0,
            tuning=(),
        )

    oof, metrics, _ = run_r_absorption_sensitivity(
        inputs,
        membership,
        Path("."),
        source_cache=cache,
        alpha_grid=(1.0,),
        label_c_grid=(1.0,),
        base_seed=7,
    )
    assert set(metrics.loc[metrics["metric_scope"].eq("label_prediction"), "model_name"]) == {
        "P+Q+D",
        "P+Q+D+residual_without_R",
        "P+Q+R+D",
        "P+Q+R+D+residual_full",
    }
    assert np.allclose(
        oof["residual_without_R"],
        oof["interviewer_logit"] - oof["predicted_interviewer_logit_without_R"],
    )
    assert np.allclose(
        oof["residual_full"],
        oof["interviewer_logit"] - oof["predicted_interviewer_logit_full"],
    )


def test_source_model_performance_uses_complete_outer_test_cohort_and_stratified_ci() -> None:
    ids = np.arange(142)
    labels = np.asarray([0] * 99 + [1] * 43)
    rows = []
    for representation in ("tfidf", "mpnet", "bge"):
        for repeat in range(1, 11):
            for participant_id, label in zip(ids, labels):
                base = float(label) * 0.8 + (participant_id % 11) / 1000.0
                rows.append(
                    {
                        "participant_id": int(participant_id),
                        "label": int(label),
                        "representation": representation,
                        "repeat": repeat,
                        "participant_probability": base,
                        "interviewer_probability": min(0.99, base + 0.05),
                    }
                )
    source_scores = pd.DataFrame(rows)
    by_repeat = source_model_performance_by_repeat(
        source_scores, n_bootstrap=20, expected_n=142, expected_repeats=10, seed=5
    )
    assert len(by_repeat) == 60
    repeat_one = by_repeat.loc[by_repeat["repeat"].eq(1)]
    assert repeat_one["n_participants"].eq(142).all()
    assert repeat_one["n_positive"].eq(43).all()
    assert repeat_one["n_negative"].eq(99).all()
    assert repeat_one["roc_auc_ci_low"].notna().all()
    summary = source_model_performance_summary(by_repeat)
    assert len(summary) == 6
    assert summary["repeat_n"].eq(10).all()
    assert summary["roc_auc_mean"].notna().all()


def test_stratified_fake_d_is_label_safe_partition_safe_rowwise_and_reproducible() -> None:
    ids = np.arange(20)
    labels = np.asarray([0, 1] * 10)
    domains = np.arange(60, dtype=float).reshape(20, 3)
    train_idx = np.arange(0, 12)
    test_idx = np.arange(12, 20)
    first_domains, first_ledger = apply_stratified_fake_domain_permutation(
        participant_ids=ids,
        labels=labels,
        domains=domains,
        train_idx=train_idx,
        test_idx=test_idx,
        draw=7,
        seed=101,
    )
    second_domains, second_ledger = apply_stratified_fake_domain_permutation(
        participant_ids=ids,
        labels=labels,
        domains=domains,
        train_idx=train_idx,
        test_idx=test_idx,
        draw=7,
        seed=101,
    )
    np.testing.assert_array_equal(first_domains, second_domains)
    pd.testing.assert_frame_equal(first_ledger, second_ledger)
    assert not first_ledger["self_match"].any()
    assert first_ledger["same_label"].all()
    assert first_ledger["one_to_one"].all()
    train_donors = set(first_ledger.loc[first_ledger["partition"].eq("train"), "donor_id"])
    test_donors = set(first_ledger.loc[first_ledger["partition"].eq("test"), "donor_id"])
    assert train_donors.isdisjoint(test_donors)
    for row in first_ledger.itertuples(index=False):
        np.testing.assert_array_equal(first_domains[int(row.recipient_index)], domains[int(row.donor_index)])


def test_stratified_fake_d_summary_retains_bootstrap_intervals_without_p_q_or_fdr() -> None:
    explanation_oof, residual_oof = _synthetic_stratified_fake_oof()
    summary = revision._stratified_fake_summary(
        explanation_oof,
        residual_oof,
        non_stratified_explanation_results=None,
        non_stratified_residual_results=None,
        n_bootstrap=20,
        seed=7,
    )
    assert summary.loc[0, "n_draws"] == 2
    assert summary.loc[0, "n_bootstrap"] == 20
    assert {"stratified_fake_D_r2_q025", "real_minus_stratified_fake_D_r2_ci_low"}.issubset(summary.columns)
    assert not any("p_value" in column or "q_value" in column for column in summary.columns)
    assert "fdr_family" not in summary.columns
    contract = revision.targeted_fake_d_manifest_contract()
    assert contract["targeted_negative_control_sensitivity"] is True
    assert contract["modifies_frozen_primary_FDR_family"] is False
    assert contract["new_targeted_sensitivity_FDR_family"] is False


def test_locked_revision_hash_validation_rejects_mutated_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_root = tmp_path / "analysis_v2"
    project_root = tmp_path / "project"
    metric_path = output_root / "frozen" / "metrics.csv"
    canonical_path = project_root / "canonical" / "c5_spans.csv"
    metric_path.parent.mkdir(parents=True)
    canonical_path.parent.mkdir(parents=True)
    metric_path.write_text("metric-original", encoding="utf-8")
    canonical_path.write_text("canonical-original", encoding="utf-8")
    expected = {
        "formal_metrics": hashlib.sha256(metric_path.read_bytes()).hexdigest(),
        "canonical_c5_spans": hashlib.sha256(canonical_path.read_bytes()).hexdigest(),
    }
    monkeypatch.setattr(revision, "LOCKED_REVISION_INPUT_HASHES", expected, raising=False)
    monkeypatch.setattr(
        revision,
        "LOCKED_REVISION_INPUT_PATHS",
        {
            "formal_metrics": ("output_root", Path("frozen/metrics.csv")),
            "canonical_c5_spans": ("project_root", Path("canonical/c5_spans.csv")),
        },
        raising=False,
    )

    assert revision.validate_locked_revision_inputs(output_root, project_root) == expected
    metric_path.write_text("metric-mutated", encoding="utf-8")
    with pytest.raises(ValueError, match="locked revision input hash mismatch.*formal_metrics"):
        revision.validate_locked_revision_inputs(output_root, project_root)


def test_locked_revision_validation_happens_before_run_directory_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_dir = tmp_path / "analysis_v2" / "16_interviewer_signal_revision" / "run_1"

    def fail_lock(*_args: object, **_kwargs: object) -> dict[str, str]:
        raise ValueError("locked revision input hash mismatch")

    monkeypatch.setattr(revision, "validate_locked_revision_inputs", fail_lock, raising=False)
    monkeypatch.setattr(
        revision,
        "resolve_revision_project_root",
        lambda *_args, **_kwargs: tmp_path,
    )
    with pytest.raises(ValueError, match="locked revision input hash mismatch"):
        revision.run_revision_analysis(
            tmp_path / "analysis_v2",
            result_dir,
            project_root=tmp_path,
        )
    assert not result_dir.exists()


def test_revision_project_root_prefers_an_ancestor_with_canonical_c5_spans(tmp_path: Path) -> None:
    repository_root = tmp_path / "repository"
    output_root = repository_root / ".workbuddy" / "worktrees" / "submission-audit" / "analysis_v2"
    canonical = repository_root / revision.C5_CANONICAL_RELATIVE
    output_root.mkdir(parents=True)
    canonical.parent.mkdir(parents=True)
    canonical.write_text("canonical", encoding="utf-8")
    assert revision.resolve_revision_project_root(output_root) == repository_root.resolve()


def test_c5_blinded_packet_excludes_model_and_participant_fields_but_owner_crosswalk_retains_linkage() -> None:
    roster = pd.DataFrame({"participant_id": [101, 102]})
    owner_crosswalk = revision.build_c5_owner_crosswalk(
        _synthetic_c5_spans(), roster, source_sha256="source-hash"
    )
    packet = revision.build_c5_blinded_reviewer_packet(owner_crosswalk)
    assert list(packet.columns) == [
        "review_case_id",
        "deidentified_quote",
        "local_context_reference",
        "human_span_valid",
        "human_domain",
        "human_polarity",
        "rater_id",
        "notes",
    ]
    prohibited = {
        "participant_id",
        "label",
        "model_domain",
        "model_polarity",
        "model_review_status",
        "model_match_status",
        "model_exact_quote",
        "span_id",
    }
    assert not prohibited.intersection(packet.columns)
    assert owner_crosswalk["participant_id"].notna().all()
    assert owner_crosswalk["model_domain"].notna().all()
    assert packet["review_case_id"].str.startswith("c5-review-").all()
    assert packet.loc[:, ["human_span_valid", "human_domain", "human_polarity", "rater_id", "notes"]].fillna("").eq("").all().all()


def test_c5_missing_evidence_audit_is_participant_level_and_separate_from_candidate_spans() -> None:
    roster = pd.DataFrame({"participant_id": [101, 102]})
    owner_crosswalk = revision.build_c5_owner_crosswalk(
        _synthetic_c5_spans(), roster, source_sha256="source-hash"
    )
    missing_evidence = revision.build_c5_missing_evidence_audit_form(owner_crosswalk)
    assert list(missing_evidence.columns) == [
        "fulltext_review_case_id",
        "local_context_reference",
        "human_missing_evidence",
        "missing_evidence_quote",
        "rater_id",
        "notes",
    ]
    assert len(missing_evidence) == 2
    assert "participant_id" not in missing_evidence.columns
    assert missing_evidence.loc[:, ["human_missing_evidence", "missing_evidence_quote", "rater_id", "notes"]].fillna("").eq("").all().all()


def test_c5_blinded_review_package_writes_separate_reviewer_and_owner_artifacts(tmp_path: Path) -> None:
    roster = pd.DataFrame({"participant_id": [101, 102]})
    owner_crosswalk = revision.build_c5_owner_crosswalk(
        _synthetic_c5_spans(), roster, source_sha256="source-hash"
    )
    reviewer_dir = tmp_path / "reviewers"
    owner_dir = tmp_path / "owner"
    written = revision.write_c5_blinded_review_package(
        owner_crosswalk,
        reviewer_dir=reviewer_dir,
        owner_dir=owner_dir,
    )
    assert set(written) == {
        "reviewer_a_csv",
        "reviewer_b_csv",
        "reviewer_a_xlsx",
        "reviewer_b_xlsx",
        "missing_evidence_csv",
        "owner_crosswalk_csv",
        "owner_crosswalk_xlsx",
    }
    reviewer_a = pd.read_csv(written["reviewer_a_csv"])
    reviewer_b = pd.read_csv(written["reviewer_b_csv"])
    assert list(reviewer_a.columns) == list(revision.C5_REVIEWER_FIELDS)
    pd.testing.assert_frame_equal(reviewer_a, reviewer_b)
    assert not {"participant_id", "label", "model_domain", "model_polarity"}.intersection(reviewer_a.columns)
    owner = pd.read_csv(written["owner_crosswalk_csv"])
    assert {"participant_id", "label", "model_domain", "model_polarity"}.issubset(owner.columns)
    assert (reviewer_dir / "c5_missing_evidence_audit_blank.csv").exists()
    assert (owner_dir / "c5_owner_crosswalk.xlsx").exists()


def test_c5_agreement_allows_blank_domain_on_invalid_spans_and_separates_model_comparisons() -> None:
    roster = pd.DataFrame({"participant_id": [101, 102]})
    owner_crosswalk = revision.build_c5_owner_crosswalk(
        _synthetic_c5_spans(), roster, source_sha256="source-hash"
    )
    packet = revision.build_c5_blinded_reviewer_packet(owner_crosswalk)
    with pytest.raises(ValueError, match="cannot compute kappa or F1"):
        revision.compute_c5_human_machine_agreement(packet, packet, owner_crosswalk=owner_crosswalk)
    completed_a = packet.copy()
    completed_b = packet.copy()
    completed_a["human_span_valid"] = ["yes", "yes", "no"]
    completed_b["human_span_valid"] = ["yes", "no", "no"]
    completed_a["human_domain"] = ["sleep_fatigue_energy", "self_worth_guilt", ""]
    completed_b["human_domain"] = ["sleep_fatigue_energy", "", ""]
    completed_a["human_polarity"] = ["present", "present", ""]
    completed_b["human_polarity"] = ["present", "", ""]
    metrics = revision.compute_c5_human_machine_agreement(
        completed_a,
        completed_b,
        owner_crosswalk=owner_crosswalk,
    )
    assert {"model_vs_rater_a", "model_vs_rater_b", "rater_a_vs_rater_b"}.issubset(metrics["comparison"])
    assert "model_vs_consensus" not in set(metrics["comparison"])
    joint_valid = metrics.loc[
        metrics["comparison"].eq("rater_a_vs_rater_b") & metrics["metric"].isin(["domain_cohen_kappa", "polarity_cohen_kappa"])
    ]
    assert joint_valid["n"].eq(1).all()
    with_consensus = revision.compute_c5_human_machine_agreement(
        completed_a,
        completed_b,
        owner_crosswalk=owner_crosswalk,
        consensus=completed_a,
    )
    assert "model_vs_consensus" in set(with_consensus["comparison"])


def test_c5_agreement_does_not_warn_when_jointly_valid_domain_scope_has_one_span() -> None:
    roster = pd.DataFrame({"participant_id": [101, 102]})
    owner_crosswalk = revision.build_c5_owner_crosswalk(
        _synthetic_c5_spans(), roster, source_sha256="source-hash"
    )
    packet = revision.build_c5_blinded_reviewer_packet(owner_crosswalk)
    completed_a = packet.copy()
    completed_b = packet.copy()
    completed_a["human_span_valid"] = ["yes", "yes", "no"]
    completed_b["human_span_valid"] = ["yes", "no", "no"]
    completed_a["human_domain"] = ["sleep_fatigue_energy", "self_worth_guilt", ""]
    completed_b["human_domain"] = ["sleep_fatigue_energy", "", ""]
    completed_a["human_polarity"] = ["present", "present", ""]
    completed_b["human_polarity"] = ["present", "", ""]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        revision.compute_c5_human_machine_agreement(
            completed_a,
            completed_b,
            owner_crosswalk=owner_crosswalk,
        )
    assert not any(issubclass(item.category, UndefinedMetricWarning) for item in caught)


def test_c5_agreement_cli_uses_owner_crosswalk_and_optional_consensus(tmp_path: Path) -> None:
    roster = pd.DataFrame({"participant_id": [101, 102]})
    owner_crosswalk = revision.build_c5_owner_crosswalk(
        _synthetic_c5_spans(), roster, source_sha256="source-hash"
    )
    rater_a = revision.build_c5_blinded_reviewer_packet(owner_crosswalk)
    rater_b = rater_a.copy()
    for frame in (rater_a, rater_b):
        frame["human_span_valid"] = ["yes", "yes", "no"]
        frame["human_domain"] = ["sleep_fatigue_energy", "self_worth_guilt", ""]
        frame["human_polarity"] = ["present", "present", ""]
    rater_a_path = tmp_path / "rater_a.csv"
    rater_b_path = tmp_path / "rater_b.csv"
    owner_path = tmp_path / "owner.csv"
    consensus_path = tmp_path / "consensus.csv"
    output_path = tmp_path / "agreement.csv"
    rater_a.to_csv(rater_a_path, index=False)
    rater_b.to_csv(rater_b_path, index=False)
    owner_crosswalk.to_csv(owner_path, index=False)
    rater_a.to_csv(consensus_path, index=False)
    assert compute_c5_agreement_main(
        [
            "--rater-a",
            str(rater_a_path),
            "--rater-b",
            str(rater_b_path),
            "--owner-crosswalk",
            str(owner_path),
            "--consensus",
            str(consensus_path),
            "--output",
            str(output_path),
        ]
    ) == 0
    metrics = pd.read_csv(output_path)
    assert "model_vs_consensus" in set(metrics["comparison"])
