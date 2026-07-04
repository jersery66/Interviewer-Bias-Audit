import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from reanalysis_v2.inference import (
    apply_bh_by_family,
    benjamini_hochberg,
    paired_permutation_test,
)
from reanalysis_v2.metrics import (
    aggregate_repeated_predictions,
    calibration_slope_intercept,
    quantile_ece,
)
from reanalysis_v2.thresholds import select_threshold, threshold_candidates


def test_threshold_candidates_and_tie_breaking_are_frozen():
    candidates = threshold_candidates(np.array([0.1, 0.4, 0.4, 0.8]))
    np.testing.assert_allclose(candidates, [0.0, 0.25, 0.6, 1.0])

    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.3, 0.4, 0.9])
    assert select_threshold(y, p, "max_macro_f1") == pytest.approx(0.35)
    assert select_threshold(y, p, "max_youden_j") == pytest.approx(0.35)
    assert select_threshold(y, p, "sensitivity_at_spec80") == pytest.approx(0.35)


def test_quantile_ece_and_calibration_parameters_are_finite():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    ece, effective_bins = quantile_ece(y, p, n_bins=2)
    assert ece == pytest.approx(0.15)
    assert effective_bins == 2

    intercept, slope = calibration_slope_intercept(y, p)
    assert np.isfinite(intercept)
    assert np.isfinite(slope)
    assert slope > 0


def test_participant_aggregation_uses_probabilities_not_fold_means():
    rows = []
    for repeat, values in [(1, [(1, 0.7, 0.6), (2, 0.4, 0.3)]), (2, [(1, 0.5, 0.4), (2, 0.2, 0.3)])]:
        for participant_id, probability, threshold in values:
            rows.append(
                {
                    "repeat": repeat,
                    "participant_id": participant_id,
                    "label": 1 if participant_id == 1 else 0,
                    "raw_probability": probability,
                    "platt_probability": probability,
                    "isotonic_probability": probability,
                    "threshold_max_macro_f1": threshold,
                    "threshold_max_youden_j": threshold,
                    "threshold_sensitivity_at_spec80": threshold,
                }
            )
    aggregated = aggregate_repeated_predictions(pd.DataFrame(rows), expected_repeats=2)
    first = aggregated.set_index("participant_id").loc[1]
    second = aggregated.set_index("participant_id").loc[2]
    assert first.raw_probability == pytest.approx(0.6)
    assert first.nested_margin_max_macro_f1 == pytest.approx(0.1)
    assert first.nested_prediction_max_macro_f1 == 1
    assert second.nested_margin_max_macro_f1 == pytest.approx(0.0)
    assert second.nested_prediction_max_macro_f1 == 1


def test_paired_permutation_and_fdr_are_participant_level_and_family_local():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    result = paired_permutation_test(
        y, p, p.copy(), roc_auc_score, n_permutations=99, seed=7
    )
    assert result["observed_difference"] == pytest.approx(0.0)
    assert result["p_value"] == pytest.approx(1.0)

    q = benjamini_hochberg(np.array([0.01, 0.04, 0.03]))
    np.testing.assert_allclose(q, [0.03, 0.04, 0.04])

    table = pd.DataFrame(
        {"family": ["a", "a", "b"], "p_value": [0.01, 0.04, 0.04]}
    )
    corrected = apply_bh_by_family(table)
    np.testing.assert_allclose(corrected.fdr_bh_q, [0.02, 0.04, 0.04])

    with pytest.raises(ValueError, match="same length"):
        paired_permutation_test(y, p[:-1], p, roc_auc_score, 99, 7)

