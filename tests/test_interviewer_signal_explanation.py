from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reanalysis_v2.interviewer_signal_explanation import (
    ALL_BLOCK_SUBSETS,
    D_P_FEATURES,
    ExplanationInputs,
    SourceFoldCache,
    PAIRING_MATCH_FEATURES,
    apply_fake_domain_permutation,
    audit_numeric_features,
    block_subset_columns,
    compute_shapley_values,
    derive_interviewer_path_features,
    fake_domain_summary,
    fit_ridge_pipeline,
    joint_bootstrap_indices,
    make_pairing_assignment,
    nested_dense_source_logits,
    public_output_columns_are_safe,
    run_direct_explanation,
    run_fake_domain_controls,
    run_pairing_dependence,
    run_residual_signal,
    validate_dp_frame,
    verify_explanation_reruns,
)
from reanalysis_v2.splits import make_repeated_split_membership
from reanalysis_v2.splits import split_indices


def _synthetic_explanation_inputs_and_cache() -> tuple[ExplanationInputs, pd.DataFrame, dict[tuple[str, int, int], SourceFoldCache]]:
    rng = np.random.default_rng(123)
    n = 20
    ids = np.arange(1000, 1000 + n)
    labels = np.array([0] * 10 + [1] * 10)
    structural = pd.DataFrame({"participant_id": ids, "label": labels, "phq8_score": labels * 10})
    q = rng.normal(size=(n, 2))
    r = rng.normal(size=(n, 2))
    d = rng.integers(0, 3, size=(n, 3)).astype(float)
    structural["q0"], structural["q1"] = q[:, 0], q[:, 1]
    structural["r0"], structural["r1"] = r[:, 0], r[:, 1]
    structural["length_only__interviewer_word_count"] = rng.integers(10, 100, size=n)
    structural["interaction_structure__interviewer_turn_count"] = rng.integers(2, 20, size=n)
    structural["protocol_structure__clinical_question_count"] = rng.integers(0, 5, size=n)
    structural["protocol_structure__clinical_prompt_coverage_count"] = rng.integers(0, 5, size=n)
    structural["protocol_structure__non_explicit_interviewer_turn_count"] = rng.integers(0, 20, size=n)
    domains = pd.DataFrame({"participant_id": ids})
    for index, feature in enumerate(D_P_FEATURES):
        domains[feature] = d[:, index % 3]
    audit = pd.DataFrame({"feature": ["q0", "q1", "r0", "r1"], "model_eligible": [True] * 4})
    inputs = ExplanationInputs(
        participant_ids=ids,
        labels=labels,
        structural=structural,
        domains=domains,
        path_features=pd.DataFrame({"participant_id": ids}),
        block_matrices={"Q": q, "R": r, "D": d},
        block_columns={"P": ("participant_logit",), "Q": ("q0", "q1"), "R": ("r0", "r1"), "D": ("d0", "d1", "d2")},
        structural_audit=audit,
        structural_audit_summary={"dropped_features": []},
        path_dictionary=(),
        path_audit=pd.DataFrame(),
        path_audit_summary={"dropped_features": []},
        domain_audit=pd.DataFrame(),
        domain_audit_summary={"dropped_features": []},
    )
    membership = make_repeated_split_membership(ids, labels, n_repeats=1, n_splits=5, base_seed=33)
    cache = {}
    for fold in range(1, 6):
        train_idx, test_idx = split_indices(membership, 1, fold, ids)
        participant_train = labels[train_idx].astype(float) + rng.normal(scale=0.2, size=len(train_idx))
        participant_test = labels[test_idx].astype(float) + rng.normal(scale=0.2, size=len(test_idx))
        interviewer_train = 0.5 * labels[train_idx].astype(float) + rng.normal(scale=0.2, size=len(train_idx))
        interviewer_test = 0.5 * labels[test_idx].astype(float) + rng.normal(scale=0.2, size=len(test_idx))
        cache[("tfidf", 1, fold)] = SourceFoldCache(
            train_idx=train_idx,
            test_idx=test_idx,
            participant_train_logit=participant_train,
            participant_test_logit=participant_test,
            interviewer_train_logit=interviewer_train,
            interviewer_test_logit=interviewer_test,
            participant_c=1.0,
            interviewer_c=1.0,
            tuning=(),
        )
    return inputs, membership, cache


def test_all_sixteen_block_subsets_are_locked_in_order() -> None:
    assert ALL_BLOCK_SUBSETS == (
        "null",
        "P",
        "Q",
        "R",
        "D",
        "P+Q",
        "P+R",
        "P+D",
        "Q+R",
        "Q+D",
        "R+D",
        "P+Q+R",
        "P+Q+D",
        "P+R+D",
        "Q+R+D",
        "P+Q+R+D",
    )
    assert block_subset_columns("P+Q+R+D", {"P": ["p"], "Q": ["q"], "R": ["r"], "D": ["d"]}) == [
        "p",
        "q",
        "r",
        "d",
    ]


def test_ridge_pipeline_scales_inside_the_training_pipeline() -> None:
    pipeline = fit_ridge_pipeline(alpha=10.0)

    assert list(pipeline.named_steps) == ["scale", "ridge"]
    assert pipeline.named_steps["scale"].__class__.__name__ == "StandardScaler"
    assert pipeline.named_steps["ridge"].alpha == 10.0


def test_numeric_audit_marks_constant_and_near_zero_features_without_dropping_input() -> None:
    frame = pd.DataFrame(
        {
            "constant": [1.0] * 20,
            "near_zero": [0.0] * 19 + [1.0],
            "usable": list(range(20)),
        }
    )
    audit, summary = audit_numeric_features(frame, list(frame.columns))

    assert audit.loc[audit["feature"].eq("constant"), "zero_variance"].item()
    assert audit.loc[audit["feature"].eq("near_zero"), "near_zero_variance"].item()
    assert audit.loc[audit["feature"].eq("usable"), "model_eligible"].item()
    assert summary["dropped_features"] == ["constant", "near_zero"]


def test_dp_contract_keeps_the_frozen_ten_domain_order_and_rejects_unknown_columns() -> None:
    ids = np.arange(4)
    frame = pd.DataFrame({"participant_id": ids})
    for index, feature in enumerate(D_P_FEATURES):
        frame[feature] = ids + index

    validated = validate_dp_frame(frame, expected_ids=ids)

    assert list(validated.columns) == ["participant_id", *D_P_FEATURES]
    assert validated[list(D_P_FEATURES)].to_numpy(dtype=float).shape == (4, 10)
    with pytest.raises(ValueError, match="unknown"):
        validate_dp_frame(frame.assign(D_All=1), expected_ids=ids)


def test_path_features_use_frozen_turn_order_and_do_not_emit_labels_or_text() -> None:
    turns = pd.DataFrame(
        {
            "participant_id": [1, 1, 1, 2, 2],
            "turn_index": [1, 2, 3, 1, 2],
            "speaker_norm": ["interviewer", "participant", "interviewer", "interviewer", "participant"],
            "is_prompt_annotated": [1, 0, 1, 0, 0],
            "is_explicit_clinical_prompt": [1, 0, 0, 0, 0],
            "prompt_id": ["P1", np.nan, "P2", np.nan, np.nan],
            "normalized_spoken_text": ["redacted", "redacted", "redacted", "redacted", "redacted"],
            "paper_label_phq8_ge10": [0, 0, 0, 1, 1],
        }
    )
    features, dictionary = derive_interviewer_path_features(turns)

    assert features["participant_id"].tolist() == [1, 2]
    assert features.loc[features["participant_id"].eq(1), "protocol_path_depth"].item() == 2
    assert features.loc[features["participant_id"].eq(1), "clinical_nonclinical_transition_count"].item() == 1
    assert not any(column in features.columns for column in ("paper_label_phq8_ge10", "normalized_spoken_text"))
    assert all("raw_prompt_id" in row and "feature" in row for row in dictionary)


def test_fake_domain_moves_whole_rows_and_has_no_self_matches() -> None:
    ids = np.arange(6)
    domains = np.arange(60, dtype=float).reshape(6, 10)
    fake, ledger = apply_fake_domain_permutation(
        participant_ids=ids,
        domains=domains,
        train_idx=np.array([0, 1, 2, 3]),
        test_idx=np.array([4, 5]),
        draw=3,
        seed=17,
    )

    assert np.array_equal(fake[0:4], domains[ledger.loc[ledger["partition"].eq("train"), "donor_index"].to_numpy()])
    assert np.array_equal(fake[4:6], domains[ledger.loc[ledger["partition"].eq("test"), "donor_index"].to_numpy()])
    assert not (ledger["recipient_id"] == ledger["donor_id"]).any()
    assert set(ledger.loc[ledger["partition"].eq("train"), "donor_id"]).isdisjoint(
        set(ledger.loc[ledger["partition"].eq("test"), "donor_id"])
    )


def test_pairing_assignment_is_one_to_one_nonself_and_uses_only_matching_features() -> None:
    ids = np.arange(6)
    features = pd.DataFrame(
        {
            "participant_id": ids,
            **{name: np.arange(6, dtype=float) + index for index, name in enumerate(PAIRING_MATCH_FEATURES)},
        }
    )
    result = make_pairing_assignment(
        recipient=features,
        donor=features,
        fit_reference=features.iloc[:4],
        seed=9,
    )

    assert result["recipient_id"].nunique() == 6
    assert result["donor_id"].nunique() == 6
    assert not (result["recipient_id"] == result["donor_id"]).any()
    assert set(result.columns) >= {"recipient_id", "donor_id", "distance", "self_match"}


def test_nested_dense_source_logits_produces_cross_fitted_train_and_outer_test_scores() -> None:
    rng = np.random.default_rng(4)
    ids = np.arange(20)
    labels = np.array([0] * 10 + [1] * 10)
    membership = make_repeated_split_membership(
        ids, labels, n_repeats=1, n_splits=5, base_seed=19
    )
    train_idx = np.arange(16)
    test_idx = np.arange(16, 20)
    train_logits, test_logits, tuning = nested_dense_source_logits(
        rng.normal(size=(16, 5)),
        labels[train_idx],
        rng.normal(size=(4, 5)),
        inner_folds=2,
        c_grid=(0.1, 1.0),
        seed=21,
    )

    assert train_logits.shape == (16,)
    assert test_logits.shape == (4,)
    assert np.isfinite(train_logits).all()
    assert np.isfinite(test_logits).all()
    assert any(row["layer"] == "source_outer_full" for row in tuning)
    assert test_idx.tolist() == [16, 17, 18, 19]
    assert membership["participant_id"].nunique() == 20


def test_shapley_values_have_symmetry_and_efficiency() -> None:
    blocks = ("P", "Q", "R", "D")
    values = {
        "null": 0.0,
        "P": 0.2,
        "Q": 0.2,
        "R": 0.0,
        "D": 0.0,
        "P+Q": 0.4,
        "P+R": 0.2,
        "P+D": 0.2,
        "Q+R": 0.2,
        "Q+D": 0.2,
        "R+D": 0.0,
        "P+Q+R": 0.4,
        "P+Q+D": 0.4,
        "P+R+D": 0.2,
        "Q+R+D": 0.2,
        "P+Q+R+D": 0.4,
    }
    shapley = compute_shapley_values(values, blocks=blocks)

    assert shapley["P"] == pytest.approx(shapley["Q"])
    assert shapley["R"] == pytest.approx(shapley["D"])
    assert sum(shapley.values()) == pytest.approx(values["P+Q+R+D"] - values["null"])


def test_joint_bootstrap_uses_one_participant_index_vector_per_resample() -> None:
    first = joint_bootstrap_indices(8, n_bootstrap=5, seed=23)
    second = joint_bootstrap_indices(8, n_bootstrap=5, seed=23)

    assert np.array_equal(first, second)
    assert first.shape == (5, 8)
    assert np.all((first >= 0) & (first < 8))


def test_public_output_guard_rejects_interview_source_columns() -> None:
    public_output_columns_are_safe(["participant_id", "interviewer_logit"])
    with pytest.raises(ValueError, match="source text"):
        public_output_columns_are_safe(["participant_id", "text"])


def test_rerun_verification_requires_identical_core_hashes_and_code_contract(tmp_path: Path) -> None:
    base = tmp_path / "analysis"
    for run_name in ("run_1", "run_2"):
        run = base / run_name
        run.mkdir(parents=True)
        output = run / "metrics.csv"
        output.write_text("a,b\n1,2\n", encoding="utf-8")
        digest = __import__("hashlib").sha256(output.read_bytes()).hexdigest()
        (run / "run_manifest.json").write_text(
            json.dumps({"code_commit": "stage-a", "plan_sha256": "plan", "output_file_hashes": {"metrics.csv": digest}}),
            encoding="utf-8",
        )

    verified = verify_explanation_reruns(base)
    assert verified["status"] == "verified"
    (base / "run_2" / "metrics.csv").write_text("a,b\n9,9\n", encoding="utf-8")
    failed = verify_explanation_reruns(base)
    assert failed["status"] == "failed"


def test_direct_explanation_runs_all_subsets_and_crossfits_full_training_predictions() -> None:
    inputs, membership, cache = _synthetic_explanation_inputs_and_cache()

    predictions, metrics, tuning, full_cache = run_direct_explanation(
        inputs,
        cache,
        membership,
        representations=("tfidf",),
        repeats=(1,),
        inner_folds=2,
        alpha_grid=(0.1, 1.0),
        base_seed=41,
    )

    assert len(predictions) == len(inputs.participant_ids) * len(ALL_BLOCK_SUBSETS)
    assert set(predictions["subset"]) == set(ALL_BLOCK_SUBSETS)
    assert len(metrics) == len(ALL_BLOCK_SUBSETS)
    assert set(full_cache) == {("tfidf", 1, fold) for fold in range(1, 6)}
    assert all(np.isfinite(result.train_prediction).all() for result in full_cache.values())
    assert not tuning.empty


def test_residual_fake_d_and_pairing_smoke_keep_separate_contracts() -> None:
    inputs, membership, cache = _synthetic_explanation_inputs_and_cache()
    predictions, _, _, full_cache = run_direct_explanation(
        inputs,
        cache,
        membership,
        representations=("tfidf",),
        repeats=(1,),
        inner_folds=2,
        alpha_grid=(0.1, 1.0),
        base_seed=43,
    )
    residual_predictions, residual_metrics, residual_deltas = run_residual_signal(
        inputs,
        cache,
        full_cache,
        membership,
        representations=("tfidf",),
        repeats=(1,),
        inner_folds=2,
        c_grid=(0.1, 1.0),
        base_seed=43,
        n_bootstrap=10,
        n_permutations=10,
    )
    (
        fake_assignments,
        fake_explanation,
        fake_residual,
        fake_explanation_oof,
        fake_residual_oof,
    ) = run_fake_domain_controls(
        inputs,
        cache,
        membership,
        representations=("tfidf",),
        main_repeat=1,
        n_draws=1,
        inner_folds=2,
        label_c_grid=(0.1, 1.0),
        alpha_grid=(0.1, 1.0),
        base_seed=43,
        return_oof=True,
    )
    pairing_assignments, pairing_predictions, pairing_balance, pairing_metrics, pairing_deltas, _ = run_pairing_dependence(
        inputs,
        cache,
        membership,
        representations=("tfidf",),
        repeats=(1,),
        inner_folds=2,
        c_grid=(0.1, 1.0),
        n_draws=1,
        base_seed=43,
        n_bootstrap=10,
        n_permutations=10,
    )

    assert len(residual_predictions) == len(inputs.participant_ids)
    assert set(residual_metrics["model_name"]) == {"base", "enhanced"}
    assert residual_deltas.loc[0, "n_bootstrap"] == 10
    assert set(fake_assignments["partition"]) == {"train", "test"}
    assert fake_assignments["self_match"].eq(False).all()
    assert set(fake_explanation["domain_type"]) == {"real", "fake"}
    assert set(fake_residual["domain_type"]) == {"real", "fake"}
    assert len(fake_explanation_oof) == len(inputs.participant_ids) * 2
    assert len(fake_residual_oof) == len(inputs.participant_ids) * 2
    fake_summary = fake_domain_summary(
        fake_explanation,
        fake_residual,
        fake_explanation_oof,
        fake_residual_oof,
        n_bootstrap=10,
        seed=43,
    )
    assert fake_summary.loc[0, "double_bootstrap_n"] == 10
    assert len(pairing_assignments) == len(inputs.participant_ids) * 5
    assert pairing_assignments.groupby(["outer_fold", "partition"]) ["recipient_id"].nunique().min() >= 2
    assert pairing_assignments["self_match"].eq(False).all()
    assert pairing_balance["matching_adequate"].all()
    assert not pairing_metrics.empty
    assert not pairing_deltas.empty
