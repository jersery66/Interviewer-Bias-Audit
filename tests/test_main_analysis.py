import numpy as np
import pandas as pd

from reanalysis_v2.analysis import (
    PRIMARY_CONDITIONS,
    build_primary_paired_tests,
    run_dense_condition,
    run_dense_outer_fold,
    run_text_condition,
    run_text_outer_fold,
    summarize_participant_results,
    summarize_threshold_stability,
)
from reanalysis_v2.splits import make_repeated_split_membership
from reanalysis_v2.modeling import make_inner_splits


def test_outer_text_fold_returns_train_only_nested_and_calibrated_predictions():
    train_texts = [
        "alpha stable",
        "alpha alpha stable",
        "beta stable",
        "beta beta stable",
        "gamma stable",
        "gamma gamma stable",
        "delta stable",
        "delta delta stable",
        "epsilon stable",
        "epsilon epsilon stable",
        "zeta stable",
        "zeta zeta stable",
    ]
    y_train = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    test_texts = ["outertestonly alpha", "outertestonly zeta"]
    inner = make_inner_splits(y_train, n_splits=3, seed=12)

    result = run_text_outer_fold(
        train_texts,
        y_train,
        test_texts,
        inner_splits=inner,
        seed=99,
    )

    assert result.raw_probability.shape == (2,)
    assert result.platt_probability.shape == (2,)
    assert result.isotonic_probability.shape == (2,)
    assert set(result.thresholds) == {
        "max_macro_f1",
        "max_youden_j",
        "sensitivity_at_spec80",
    }
    assert result.inner_oof_probability.shape == (12,)
    assert np.isfinite(result.inner_oof_probability).all()
    assert all(0.0 <= value <= 1.0 for value in result.thresholds.values())
    assert "outertestonly" not in result.fitted_raw.named_steps["tfidf"].vocabulary_


def test_outer_dense_fold_tunes_c_and_calibrates_inside_outer_train():
    rng = np.random.default_rng(77)
    X_train = rng.normal(size=(30, 6))
    y_train = np.array([0, 1] * 15)
    X_train[y_train == 1, 0] += 1.0
    X_test = rng.normal(size=(4, 6))
    inner = make_inner_splits(y_train, n_splits=3, seed=18)
    result = run_dense_outer_fold(
        X_train,
        y_train,
        X_test,
        inner_splits=inner,
        c_grid=[0.01, 0.1, 1.0],
        seed=19,
    )
    assert result.selected_c in {0.01, 0.1, 1.0}
    assert set(result.c_scores) == {0.01, 0.1, 1.0}
    assert result.raw_probability.shape == (4,)
    assert result.platt_probability.shape == (4,)
    assert result.isotonic_probability.shape == (4,)
    assert result.inner_oof_probability.shape == (30,)
    assert set(result.thresholds) == {
        "max_macro_f1",
        "max_youden_j",
        "sensitivity_at_spec80",
    }


def test_summary_has_all_preregistered_ranking_decision_and_calibration_metrics():
    participant = pd.DataFrame(
        {
            "participant_id": np.arange(8),
            "label": [0, 0, 0, 0, 1, 1, 1, 1],
            "raw_probability": [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
            "platt_probability": [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85],
            "isotonic_probability": [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0],
            "prediction_at_0_5": [0, 0, 0, 0, 1, 1, 1, 1],
            "nested_prediction_max_macro_f1": [0, 0, 0, 1, 1, 1, 1, 1],
            "platt_prediction_at_0_5": [0, 0, 0, 0, 1, 1, 1, 1],
            "isotonic_prediction_at_0_5": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    summary, distributions = summarize_participant_results(participant, "demo")
    required = {
        "roc_auc",
        "pr_auc",
        "macro_f1_at_0_5",
        "macro_f1_nested",
        "sensitivity_at_0_5",
        "sensitivity_nested",
        "specificity_at_0_5",
        "specificity_nested",
        "raw_brier",
        "raw_ece_5bin",
        "raw_calibration_intercept",
        "raw_calibration_slope",
        "platt_brier",
        "isotonic_brier",
    }
    assert required.issubset(summary)
    assert summary["condition"] == "demo"
    assert set(distributions.probability_type) == {"raw", "platt", "isotonic"}
    assert set(distributions.label) == {0, 1}


def test_primary_paired_family_is_all_ten_pairs_times_four_endpoints():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    tables = {}
    for offset, condition in enumerate(PRIMARY_CONDITIONS):
        probability = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        probability = np.clip(probability + offset * 0.001, 0, 1)
        tables[condition] = pd.DataFrame(
            {
                "participant_id": np.arange(8),
                "label": y,
                "raw_probability": probability,
                "prediction_at_0_5": (probability >= 0.5).astype(int),
                "nested_prediction_max_macro_f1": (probability >= 0.45).astype(int),
            }
        )
    result = build_primary_paired_tests(tables, n_permutations=19, base_seed=30)
    assert len(result) == 10 * 4
    assert result.family.eq("primary_input_source_comparisons").all()
    assert set(result.metric) == {
        "roc_auc",
        "pr_auc",
        "macro_f1_at_0_5",
        "macro_f1_nested",
    }
    assert result.fdr_bh_q.between(0, 1).all()


def test_text_condition_uses_every_frozen_outer_test_membership_once():
    participant_ids = np.arange(100, 120)
    labels = np.array([0, 1] * 10)
    texts = np.array(
        [
            ("low stable common " if label == 0 else "high stable common ")
            + f"participant{participant_id}"
            for participant_id, label in zip(participant_ids, labels)
        ],
        dtype=object,
    )
    membership = make_repeated_split_membership(
        participant_ids,
        labels,
        n_repeats=1,
        n_splits=5,
        base_seed=20260624,
    )
    result = run_text_condition(
        participant_ids,
        labels,
        texts,
        membership,
        condition="demo",
        expected_repeats=1,
        inner_splits_count=2,
        base_seed=20260624,
    )
    fold_predictions = result["fold_predictions"]
    participant = result["participant_predictions"]
    thresholds = result["thresholds"]
    assert len(fold_predictions) == 20
    assert fold_predictions.participant_id.value_counts().eq(1).all()
    assert len(participant) == 20
    assert len(thresholds) == 5 * 3
    assert thresholds.groupby(["repeat", "fold"]).size().eq(3).all()
    assert fold_predictions.train_ids_sha256.str.len().eq(64).all()
    assert fold_predictions.test_ids_sha256.str.len().eq(64).all()


def test_threshold_stability_reports_all_frozen_statistics():
    thresholds = pd.DataFrame(
        {
            "condition": ["c"] * 4,
            "rule": ["max_macro_f1"] * 4,
            "threshold": [0.2, 0.3, 0.4, 0.5],
        }
    )
    summary = summarize_threshold_stability(thresholds).iloc[0]
    assert summary["n_folds"] == 4
    assert summary["mean"] == 0.35
    assert summary["median"] == 0.35
    assert summary["min"] == 0.2
    assert summary["max"] == 0.5
    assert summary["iqr"] == summary["q3"] - summary["q1"]


def test_dense_condition_reuses_frozen_outer_splits_and_records_c_selection():
    rng = np.random.default_rng(55)
    participant_ids = np.arange(100, 120)
    labels = np.array([0, 1] * 10)
    X = rng.normal(size=(20, 5))
    X[labels == 1, 0] += 0.8
    membership = make_repeated_split_membership(
        participant_ids,
        labels,
        n_repeats=1,
        n_splits=5,
        base_seed=20260624,
    )
    result = run_dense_condition(
        participant_ids,
        labels,
        X,
        membership,
        condition="demo_dense",
        expected_repeats=1,
        inner_splits_count=2,
        c_grid=[0.01, 0.1],
        base_seed=20260624,
    )
    assert len(result["fold_predictions"]) == 20
    assert len(result["participant_predictions"]) == 20
    assert len(result["thresholds"]) == 5 * 3
    tuning = result["c_tuning"]
    assert len(tuning) == 5 * 2
    assert tuning.groupby(["repeat", "fold"]).selected.sum().eq(1).all()
