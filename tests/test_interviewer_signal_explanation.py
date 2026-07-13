from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reanalysis_v2.interviewer_signal_explanation import (
    ALL_BLOCK_SUBSETS,
    D_P_FEATURES,
    ExplanationInputs,
    SourceRepresentationData,
    SourceFoldCache,
    _pairing_draw_inference,
    _write_global_fdr,
    PAIRING_MATCH_FEATURES,
    add_label_conditioned_interpretation_flags,
    apply_fake_domain_permutation,
    audit_numeric_features,
    block_subset_columns,
    build_fold_contained_matched_source_logits,
    compute_shapley_values,
    derive_interviewer_path_features,
    fake_domain_summary,
    fit_ridge_pipeline,
    joint_bootstrap_indices,
    make_random_derangement_assignment,
    make_pairing_assignment,
    nested_dense_source_logits,
    paired_prediction_swap_permutation,
    public_output_columns_are_safe,
    pairing_quality_summary,
    recoverability_fold_metrics,
    recoverability_inference,
    recoverability_scale_sensitivity,
    is_source_score_scale_sensitive,
    run_label_conditioned_recoverability,
    run_label_only_recoverability,
    summarise_inner_training_quality,
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


def test_pairing_quality_gate_rejects_bad_one_to_one_assignment_despite_zero_marginal_smd() -> None:
    ids = np.arange(8)
    features = pd.DataFrame(
        {
            "participant_id": ids,
            **{name: ids.astype(float) * (index + 1) for index, name in enumerate(PAIRING_MATCH_FEATURES)},
        }
    )
    bad_assignment = pd.DataFrame(
        {
            "recipient_id": ids,
            "donor_id": np.roll(ids, 4),
            "self_match": False,
            "one_to_one": True,
        }
    )
    quality = pairing_quality_summary(
        recipient=features,
        donor=features,
        assignment=bad_assignment,
        fit_reference=features,
        random_distance_reference=np.full(100, 1e6),
        assignment_type="optimal_matched",
    )

    assert quality["max_abs_smd"] == pytest.approx(0.0)
    assert quality["matching_adequate"] is False
    assert quality["n_features_mean_abs_paired_diff_lt_0_50"] < 5


def test_fold_contained_mismatch_swaps_source_before_prediction_and_keeps_donors_in_validation_partition() -> None:
    n = 20
    ids = np.arange(2000, 2000 + n)
    labels = np.array([0] * 10 + [1] * 10)
    texts = np.array(
        [
            ("calm baseline " if label == 0 else "sad concern ") + "shared symptom interview token"
            for label in labels
        ],
        dtype=object,
    )
    source_data = SourceRepresentationData(
        participant=np.asarray(texts, dtype=object),
        interviewer=np.asarray(texts, dtype=object),
        feature_type="text",
    )
    match_frame = pd.DataFrame(
        {
            "participant_id": ids,
            **{name: np.arange(n, dtype=float) + index for index, name in enumerate(PAIRING_MATCH_FEATURES)},
        }
    )
    outer_train = np.arange(16)
    outer_test = np.arange(16, 20)
    train_assignment = make_pairing_assignment(
        recipient=match_frame.iloc[outer_train].reset_index(drop=True),
        donor=match_frame.iloc[outer_train].reset_index(drop=True),
        fit_reference=match_frame.iloc[outer_train].reset_index(drop=True),
        seed=7,
    )
    test_assignment = make_pairing_assignment(
        recipient=match_frame.iloc[outer_test].reset_index(drop=True),
        donor=match_frame.iloc[outer_test].reset_index(drop=True),
        fit_reference=match_frame.iloc[outer_train].reset_index(drop=True),
        seed=7,
    )

    matched_train, matched_test, audit, _ = build_fold_contained_matched_source_logits(
        representation="tfidf",
        source_data=source_data,
        labels=labels,
        participant_ids=ids,
        match_frame=match_frame,
        train_idx=outer_train,
        test_idx=outer_test,
        outer_test_assignment=test_assignment,
        outer_train_assignment=train_assignment,
        inner_folds=2,
        c_grid=(0.1, 1.0),
        seed=19,
    )

    assert np.isfinite(matched_train).all()
    assert np.isfinite(matched_test).all()
    assert set(audit["prediction_stage"]) == {"after_mismatch"}
    inner_rows = audit.loc[audit["scope"].eq("outer_train_inner_validation")]
    assert not inner_rows.empty
    assert (inner_rows["donor_partition"] == "inner_validation").all()
    for row in inner_rows.itertuples(index=False):
        assert set(row.source_train_ids).isdisjoint({row.recipient_id, row.donor_id})
    test_rows = audit.loc[audit["scope"].eq("outer_test")]
    assert set(test_rows["donor_partition"]) == {"outer_test"}
    assert set(test_rows["source_train_ids"].map(tuple).explode()) <= set(ids[outer_train])


def test_random_derangement_assignment_is_not_an_optimal_matched_draw() -> None:
    ids = np.arange(10)
    features = pd.DataFrame(
        {
            "participant_id": ids,
            **{name: ids.astype(float) + index for index, name in enumerate(PAIRING_MATCH_FEATURES)},
        }
    )
    optimal = make_pairing_assignment(
        recipient=features,
        donor=features,
        fit_reference=features,
        seed=1,
    )
    random = make_random_derangement_assignment(
        recipient=features,
        donor=features,
        fit_reference=features,
        seed=2,
    )

    assert set(random["assignment_type"]) == {"random_mismatch"}
    assert set(optimal["assignment_type"]) == {"optimal_matched"}
    assert not random["donor_id"].equals(optimal["donor_id"])


def test_recoverability_permutation_p_is_one_when_null_and_full_predictions_match() -> None:
    target = np.linspace(-1.0, 1.0, 20)
    null = np.linspace(-1.0, 1.0, 20)
    result = paired_prediction_swap_permutation(
        target,
        null,
        null.copy(),
        n_permutations=200,
        seed=3,
    )

    assert result["observed_delta_mse"] == pytest.approx(0.0)
    assert result["p_value"] == pytest.approx(1.0)


def test_recoverability_inference_reports_permutation_p_and_bootstrap_delta_mse_ci() -> None:
    ids = np.arange(8)
    target = np.linspace(-1.0, 1.0, 8)
    null = np.zeros(8)
    full = target * 0.5
    frame = pd.DataFrame(
        {
            "participant_id": ids,
            "representation": "tfidf",
            "repeat": 1,
            "subset": "P+Q+R+D",
            "interviewer_logit": target,
            "predicted_interviewer_logit": full,
            "outer_train_mean_null_prediction": null,
        }
    )
    result = recoverability_inference(
        frame,
        representation="tfidf",
        repeat=1,
        n_bootstrap=20,
        n_permutations=50,
        seed=4,
    )

    assert "observed_delta_mse" in result
    assert "permutation_p_value" in result
    assert result["n_permutations"] == 50


def test_primary_pairing_fdr_uses_full_control_p_value_even_when_directions_reverse() -> None:
    explanation = pd.DataFrame(
        {
            "participant_id": [1, 2, 3, 4],
            "label": [0, 0, 1, 1],
            "representation": "tfidf",
            "repeat": 1,
            "subset": "P+Q+R+D",
            "interviewer_logit": [0.0, 0.1, 0.9, 1.0],
            "predicted_interviewer_logit": [0.0, 0.1, 0.9, 1.0],
            "outer_train_mean_null_prediction": 0.5,
        }
    )
    residual = pd.DataFrame({"representation": ["tfidf"], "repeat": [1], "p_value": [0.2]})
    fake = pd.DataFrame({"representation": ["tfidf"], "repeat": [1], "p_value": [0.3]})
    pairing = pd.DataFrame(
        {
            "representation": ["tfidf"],
            "repeat": [1],
            "draw": [0],
            "full_real_minus_optimal_delta_auc": [0.25],
            "weak_real_minus_optimal_delta_auc": [-0.25],
            "full_real_minus_optimal_p_value": [0.037],
            "weak_real_minus_optimal_p_value": [0.991],
        }
    )

    result = _write_global_fdr(
        explanation,
        residual,
        fake,
        pairing,
        main_repeat=1,
        n_bootstrap=10,
        n_permutations=10,
    )

    assert result.loc[result["test"].eq("pairing_delta_auc"), "p_value"].item() == pytest.approx(0.037)


def test_pairing_draw_inference_reports_full_primary_and_weak_secondary_fields() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    rows = []
    model_values = {
        "P": np.array([0.10, 0.20, 0.30, 0.40, 0.60, 0.70, 0.80, 0.90]),
        "P+I-real": np.array([0.08, 0.18, 0.28, 0.38, 0.62, 0.72, 0.82, 0.92]),
        "P+I-optimal_matched": np.array([0.12, 0.22, 0.32, 0.42, 0.58, 0.68, 0.78, 0.88]),
        "full": np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]),
        "full+I-real": np.array([0.05, 0.15, 0.25, 0.35, 0.65, 0.75, 0.85, 0.95]),
        "full+I-optimal_matched": np.array([0.16, 0.26, 0.36, 0.46, 0.54, 0.64, 0.74, 0.84]),
    }
    for model_name, probabilities in model_values.items():
        rows.extend(
            {
                "participant_id": int(index),
                "label": int(labels[index]),
                "representation": "tfidf",
                "repeat": 1,
                "outer_fold": 1,
                "draw": 0,
                "draw_type": "optimal_matched",
                "model_name": model_name,
                "probability": float(probabilities[index]),
            }
            for index in range(len(labels))
        )
    predictions = pd.DataFrame(rows)
    deltas = pd.DataFrame(
        {
            "representation": ["tfidf"],
            "repeat": [1],
            "draw": [0],
            "draw_type": ["optimal_matched"],
            "weak_real_delta_auc": [0.1],
            "weak_matched_delta_auc": [0.05],
            "full_real_delta_auc": [0.2],
            "full_matched_delta_auc": [0.1],
        }
    )
    result = _pairing_draw_inference(
        predictions,
        deltas,
        main_repeat=1,
        n_bootstrap=20,
        n_permutations=20,
        seed=13,
    )

    assert {
        "full_real_minus_optimal_delta_auc",
        "full_real_minus_optimal_ci_low",
        "full_real_minus_optimal_ci_high",
        "full_real_minus_optimal_p_value",
        "full_optimal_minus_no_i_delta_auc",
        "full_optimal_minus_no_i_ci_low",
        "full_optimal_minus_no_i_ci_high",
        "weak_real_minus_optimal_delta_auc",
        "weak_real_minus_optimal_p_value",
    }.issubset(result.columns)
    assert result.loc[0, "primary_pairing_control"].startswith("P+Q+R+D")
    assert result.loc[0, "weak_control_scope"] == "secondary_weak_control_pairing"


def test_label_only_outer_test_prediction_uses_outer_training_class_mean() -> None:
    inputs, membership, cache = _synthetic_explanation_inputs_and_cache()
    oof, metrics = run_label_only_recoverability(
        cache,
        labels=inputs.labels,
        participant_ids=inputs.participant_ids,
        representations=("tfidf",),
        repeats=(1,),
        inner_folds=2,
        base_seed=31,
    )

    source = cache[("tfidf", 1, 1)]
    test_rows = oof.loc[(oof["outer_fold"] == 1) & oof["prediction_scope"].eq("outer_test")]
    expected = {
        int(label): float(source.interviewer_train_logit[inputs.labels[source.train_idx] == label].mean())
        for label in (0, 1)
    }
    assert np.allclose(test_rows["label_only_prediction"], [expected[int(label)] for label in test_rows["label"]])
    assert set(metrics["prediction_scope"]) == {"outer_test"}


def test_label_only_outer_training_predictions_are_inner_cross_fitted_without_self() -> None:
    inputs, membership, cache = _synthetic_explanation_inputs_and_cache()
    oof, _ = run_label_only_recoverability(
        cache,
        labels=inputs.labels,
        participant_ids=inputs.participant_ids,
        representations=("tfidf",),
        repeats=(1,),
        inner_folds=2,
        base_seed=31,
    )

    train_rows = oof.loc[oof["prediction_scope"].eq("outer_train_inner_validation")]
    assert not train_rows.empty
    assert train_rows["class_mean_excludes_recipient"].eq(True).all()
    assert train_rows["class_mean_training_n"].gt(0).all()
    for fold in range(1, 6):
        source = cache[("tfidf", 1, fold)]
        full_class_counts = pd.Series(inputs.labels[source.train_idx]).value_counts().to_dict()
        fold_rows = train_rows.loc[train_rows["outer_fold"].eq(fold)]
        assert all(
            int(row.class_mean_training_n) < int(full_class_counts[int(row.label)])
            for row in fold_rows.itertuples(index=False)
        )


def test_label_conditioned_outer_test_residual_uses_outer_training_class_mean() -> None:
    inputs, membership, cache = _synthetic_explanation_inputs_and_cache()
    oof, _ = run_label_conditioned_recoverability(
        inputs,
        cache,
        membership,
        representations=("tfidf",),
        repeats=(1,),
        inner_folds=2,
        alpha_grid=(0.1, 1.0),
        base_seed=31,
    )

    source = cache[("tfidf", 1, 1)]
    row = oof.loc[(oof["outer_fold"] == 1) & oof["prediction_scope"].eq("outer_test")].iloc[0]
    global_index = int(np.flatnonzero(inputs.participant_ids == int(row["participant_id"])).item())
    label = int(inputs.labels[global_index])
    expected_mean = source.interviewer_train_logit[inputs.labels[source.train_idx] == label].mean()
    assert row["interviewer_within"] == pytest.approx(source.interviewer_test_logit[np.flatnonzero(source.test_idx == global_index).item()] - expected_mean)
    assert bool(row["outer_test_mean_source_excludes_test"])


def test_label_conditioned_recoverability_removes_shared_label_alignment() -> None:
    inputs, membership, cache = _synthetic_explanation_inputs_and_cache()
    shared_cache: dict[tuple[str, int, int], SourceFoldCache] = {}
    for key, source in cache.items():
        train_labels = inputs.labels[source.train_idx].astype(float)
        test_labels = inputs.labels[source.test_idx].astype(float)
        shared_cache[key] = replace(
            source,
            participant_train_logit=2.0 * train_labels,
            participant_test_logit=2.0 * test_labels,
            interviewer_train_logit=10.0 * train_labels,
            interviewer_test_logit=10.0 * test_labels,
        )

    _, label_only_metrics = run_label_only_recoverability(
        shared_cache,
        labels=inputs.labels,
        participant_ids=inputs.participant_ids,
        representations=("tfidf",),
        repeats=(1,),
        inner_folds=2,
        base_seed=31,
    )
    _, conditioned_metrics = run_label_conditioned_recoverability(
        inputs,
        shared_cache,
        membership,
        representations=("tfidf",),
        repeats=(1,),
        inner_folds=2,
        alpha_grid=(0.1, 1.0),
        base_seed=31,
    )

    assert label_only_metrics.loc[label_only_metrics["metric"].eq("r2"), "estimate"].item() > 0.9
    full = conditioned_metrics.loc[conditioned_metrics["subset"].eq("P_within+Q+R+D")].iloc[0]
    assert full["target_mse"] == pytest.approx(0.0)
    assert full["r2"] == pytest.approx(0.0)

    explanation_metrics = pd.DataFrame(
        {
            "representation": ["tfidf"],
            "repeat": [1],
            "subset": ["P+Q+R+D"],
            "r2": [1.0],
        }
    )
    flagged = add_label_conditioned_interpretation_flags(conditioned_metrics, explanation_metrics)
    assert set(flagged["interpretation_flag"]) == {"recoverability_largely_shared_outcome_alignment"}


def test_label_conditioned_recoverability_retains_within_label_shared_variation() -> None:
    inputs, membership, cache = _synthetic_explanation_inputs_and_cache()
    within = np.linspace(-1.0, 1.0, len(inputs.labels))
    varied_cache: dict[tuple[str, int, int], SourceFoldCache] = {}
    for key, source in cache.items():
        varied_cache[key] = replace(
            source,
            participant_train_logit=within[source.train_idx],
            participant_test_logit=within[source.test_idx],
            interviewer_train_logit=3.0 * within[source.train_idx],
            interviewer_test_logit=3.0 * within[source.test_idx],
        )
    conditioned_inputs = replace(
        inputs,
        block_matrices={**inputs.block_matrices, "Q": within.reshape(-1, 1)},
        block_columns={**inputs.block_columns, "Q": ("within_q",)},
    )

    _, metrics = run_label_conditioned_recoverability(
        conditioned_inputs,
        varied_cache,
        membership,
        representations=("tfidf",),
        repeats=(1,),
        inner_folds=2,
        alpha_grid=(0.1, 1.0),
        base_seed=31,
    )

    full = metrics.loc[metrics["subset"].eq("P_within+Q+R+D")].iloc[0]
    assert full["r2"] > 0.5


def test_inner_validation_matching_quality_is_returned_with_random_reference() -> None:
    n = 20
    ids = np.arange(3000, 3000 + n)
    labels = np.array([0] * 10 + [1] * 10)
    texts = np.asarray([("calm " if label == 0 else "sad ") + "shared token" for label in labels], dtype=object)
    source_data = SourceRepresentationData(texts, texts, "text")
    frame = pd.DataFrame({"participant_id": ids, **{name: np.arange(n, dtype=float) + index for index, name in enumerate(PAIRING_MATCH_FEATURES)}})
    train_idx = np.arange(16)
    test_idx = np.arange(16, 20)
    test_assignment = make_pairing_assignment(
        recipient=frame.iloc[test_idx].reset_index(drop=True),
        donor=frame.iloc[test_idx].reset_index(drop=True),
        fit_reference=frame.iloc[train_idx].reset_index(drop=True),
        seed=3,
    )

    matched_train, matched_test, audit, quality = build_fold_contained_matched_source_logits(
        representation="tfidf",
        source_data=source_data,
        labels=labels,
        participant_ids=ids,
        match_frame=frame,
        train_idx=train_idx,
        test_idx=test_idx,
        outer_test_assignment=test_assignment,
        inner_folds=2,
        c_grid=(0.1, 1.0),
        seed=7,
    )

    assert np.isfinite(matched_train).all() and np.isfinite(matched_test).all()
    assert set(quality["assignment_type"]) == {"optimal_matched"}
    assert quality["random_reference_n"].eq(100).all()
    assert {"mean_distance", "random_distance_q05", "matching_adequate"}.issubset(quality.columns)
    assert set(audit["prediction_stage"]) == {"after_mismatch"}


def test_inner_training_quality_gate_marks_below_eighty_percent_pass_rate() -> None:
    quality = pd.DataFrame(
        {
            "assignment_type": ["optimal_matched"] * 5,
            "matching_adequate": [True, True, True, True, False],
            "no_self": [True] * 5,
            "one_to_one": [True] * 5,
        }
    )
    summary = summarise_inner_training_quality(quality)
    assert summary["pass_rate"] == pytest.approx(0.8)
    assert summary["training_match_quality_limited"] is False

    quality.loc[3, "matching_adequate"] = False
    summary = summarise_inner_training_quality(quality)
    assert summary["pass_rate"] == pytest.approx(0.6)
    assert summary["training_match_quality_limited"] is True


def test_scale_difference_below_substantive_threshold_is_not_sensitive() -> None:
    assert not is_source_score_scale_sensitive(
        raw_r2=0.10,
        standardized_r2=0.13,
        raw_spearman=0.20,
        standardized_spearman=0.22,
    )


def test_scale_r2_direction_flip_is_sensitive() -> None:
    assert is_source_score_scale_sensitive(
        raw_r2=0.10,
        standardized_r2=-0.01,
        raw_spearman=0.20,
        standardized_spearman=0.22,
    )


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
    fold_scale = recoverability_fold_metrics(predictions)
    pooled_scale = recoverability_scale_sensitivity(predictions)
    assert {"raw_r2", "outer_train_standardized_r2", "outer_train_participant_scale_sd"}.issubset(
        fold_scale.columns
    )
    assert set(pooled_scale["scale"]) == {"raw", "outer_train_standardized"}


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
    texts = np.asarray(
        [
            ("calm baseline " if label == 0 else "sad concern ") + "shared symptom interview token"
            for label in inputs.labels
        ],
        dtype=object,
    )
    source_data = {
        "tfidf": SourceRepresentationData(
            participant=texts,
            interviewer=texts,
            feature_type="text",
        )
    }
    (
        pairing_assignments,
        pairing_predictions,
        pairing_balance,
        pairing_pairwise_balance,
        pairing_random_reference,
        pairing_metrics,
        pairing_deltas,
        pairing_stability,
        pairing_inner_quality,
    ) = run_pairing_dependence(
        inputs,
        cache,
        membership,
        representations=("tfidf",),
        repeats=(1,),
        source_data=source_data,
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
    assert not pairing_assignments.empty
    assert {"outer_train_inner_validation", "outer_test"}.issubset(set(pairing_assignments["scope"]))
    outer_assignments = pairing_assignments.loc[pairing_assignments["record_type"].eq("outer_test_assignment")]
    assert outer_assignments["self_match"].eq(False).all()
    assert not pairing_balance.empty
    assert {"optimal_matched", "random_mismatch"}.issubset(set(pairing_balance["assignment_type"]))
    assert not pairing_pairwise_balance.empty
    assert not pairing_random_reference.empty
    assert not pairing_inner_quality.empty
    assert {"optimal_real_minus_matched_delta_auc", "random_real_minus_matched_delta_auc_mean"}.issubset(
        pairing_stability.columns
    )
    assert not pairing_metrics.empty
    assert not pairing_deltas.empty
