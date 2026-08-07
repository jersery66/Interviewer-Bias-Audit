from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .thresholds import THRESHOLD_RULES


def quantile_ece(
    y_true: np.ndarray, probabilities: np.ndarray, *, n_bins: int = 5
) -> tuple[float, int]:
    y_true = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(y_true) != len(probabilities):
        raise ValueError("labels and probabilities must have the same length")
    if len(y_true) == 0:
        raise ValueError("cannot compute ECE on an empty array")
    bins = pd.qcut(probabilities, q=n_bins, labels=False, duplicates="drop")
    bins = np.asarray(bins, dtype=int)
    total = len(y_true)
    ece = 0.0
    unique_bins = np.unique(bins)
    for bin_id in unique_bins:
        mask = bins == bin_id
        ece += mask.sum() / total * abs(y_true[mask].mean() - probabilities[mask].mean())
    return float(ece), int(len(unique_bins))


def calibration_slope_intercept(
    y_true: np.ndarray, probabilities: np.ndarray
) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(y_true) != len(probabilities):
        raise ValueError("labels and probabilities must have the same length")
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    estimator = LogisticRegression(
        C=1_000_000.0, solver="lbfgs", max_iter=5000, random_state=0
    )
    estimator.fit(logits, y_true)
    return float(estimator.intercept_[0]), float(estimator.coef_[0, 0])


def aggregate_repeated_predictions(
    predictions: pd.DataFrame, *, expected_repeats: int = 10
) -> pd.DataFrame:
    required = {
        "repeat",
        "participant_id",
        "label",
        "raw_probability",
        "platt_probability",
        "isotonic_probability",
        *(f"threshold_{rule}" for rule in THRESHOLD_RULES),
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"missing prediction columns: {sorted(missing)}")
    counts = predictions.groupby("participant_id")["repeat"].nunique()
    if not counts.eq(expected_repeats).all():
        raise ValueError("each participant must have exactly the expected repeated OOF predictions")
    if predictions.duplicated(["participant_id", "repeat"]).any():
        raise ValueError("duplicate participant-repeat predictions")
    if predictions.groupby("participant_id")["label"].nunique().gt(1).any():
        raise ValueError("participant labels vary across repeats")

    aggregation = {
        "label": "first",
        "raw_probability": "mean",
        "platt_probability": "mean",
        "isotonic_probability": "mean",
        **{f"threshold_{rule}": "mean" for rule in THRESHOLD_RULES},
    }
    result = predictions.groupby("participant_id", as_index=False).agg(aggregation)
    for rule in THRESHOLD_RULES:
        margin = result["raw_probability"] - result[f"threshold_{rule}"]
        result[f"nested_margin_{rule}"] = margin
        result[f"nested_prediction_{rule}"] = margin.ge(0).astype(int)
    result["prediction_at_0_5"] = result["raw_probability"].ge(0.5).astype(int)
    result["platt_prediction_at_0_5"] = result["platt_probability"].ge(0.5).astype(int)
    result["isotonic_prediction_at_0_5"] = result["isotonic_probability"].ge(0.5).astype(int)
    return result.sort_values("participant_id").reset_index(drop=True)
