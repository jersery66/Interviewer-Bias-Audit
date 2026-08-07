from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


THRESHOLD_RULES = (
    "max_macro_f1",
    "max_youden_j",
    "sensitivity_at_spec80",
)


def threshold_candidates(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1 or len(probabilities) == 0:
        raise ValueError("probabilities must be a non-empty one-dimensional array")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain non-finite values")
    unique = np.unique(np.clip(probabilities, 0.0, 1.0))
    midpoints = (unique[:-1] + unique[1:]) / 2.0
    return np.unique(np.concatenate(([0.0], midpoints, [1.0])))


def _decision_statistics(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> tuple[float, float, float]:
    predictions = (probabilities >= threshold).astype(int)
    macro_f1 = float(f1_score(y_true, predictions, average="macro", zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if tp + fn else float("nan")
    specificity = float(tn / (tn + fp)) if tn + fp else float("nan")
    return macro_f1, sensitivity, specificity


def select_threshold(
    y_true: np.ndarray, probabilities: np.ndarray, rule: str
) -> float:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(y_true) != len(probabilities):
        raise ValueError("labels and probabilities must have the same length")
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("threshold selection requires both classes")
    if rule not in THRESHOLD_RULES:
        raise ValueError(f"unknown threshold rule: {rule}")

    scored = []
    for threshold in threshold_candidates(probabilities):
        macro_f1, sensitivity, specificity = _decision_statistics(
            y_true, probabilities, float(threshold)
        )
        if rule == "max_macro_f1":
            primary = macro_f1
            secondary = 0.0
        elif rule == "max_youden_j":
            primary = sensitivity + specificity - 1.0
            secondary = 0.0
        else:
            if specificity < 0.80:
                continue
            primary = sensitivity
            secondary = specificity
        scored.append(
            (primary, secondary, -abs(float(threshold) - 0.5), -float(threshold), threshold)
        )
    if not scored:
        raise RuntimeError(f"no valid threshold for rule {rule}")
    return float(max(scored)[-1])
