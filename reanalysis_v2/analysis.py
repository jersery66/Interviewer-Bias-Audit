from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from collections.abc import Sequence
import hashlib

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from .inference import apply_bh_by_family, paired_permutation_test
from .metrics import (
    aggregate_repeated_predictions,
    calibration_slope_intercept,
    quantile_ece,
)
from .modeling import (
    fit_predict_dense,
    fit_predict_tfidf,
    make_inner_splits,
    make_dense_classifier,
    make_tfidf_classifier,
    tune_dense_c,
)
from .splits import split_indices
from .thresholds import THRESHOLD_RULES, select_threshold


PRIMARY_CONDITIONS = [
    "full_transcript",
    "participant_speech",
    "interviewer_speech",
    "non_explicit_interviewer_speech",
    "participant_symptom_evidence",
]


@dataclass(frozen=True)
class OuterFoldResult:
    raw_probability: np.ndarray
    platt_probability: np.ndarray
    isotonic_probability: np.ndarray
    inner_oof_probability: np.ndarray
    thresholds: dict[str, float]
    fitted_raw: object


@dataclass(frozen=True)
class DenseOuterFoldResult:
    raw_probability: np.ndarray
    platt_probability: np.ndarray
    isotonic_probability: np.ndarray
    inner_oof_probability: np.ndarray
    thresholds: dict[str, float]
    selected_c: float
    c_scores: dict[float, float]
    fitted_raw: object


def _participant_ids_sha256(participant_ids: np.ndarray) -> str:
    digest = hashlib.sha256()
    for participant_id in sorted(np.asarray(participant_ids).tolist()):
        digest.update(f"{participant_id}\n".encode("utf-8"))
    return digest.hexdigest()


def _fit_calibrated_text(
    train_texts: Sequence[str],
    y_train: np.ndarray,
    test_texts: Sequence[str],
    *,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    method: str,
    seed: int,
) -> np.ndarray:
    estimator = make_tfidf_classifier(seed=seed)
    calibrated = CalibratedClassifierCV(
        estimator=estimator,
        method=method,
        cv=inner_splits,
        ensemble=False,
    )
    calibrated.fit(list(train_texts), np.asarray(y_train, dtype=int))
    return calibrated.predict_proba(list(test_texts))[:, 1]


def _fit_calibrated_dense(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    method: str,
    c: float,
    seed: int,
) -> np.ndarray:
    estimator = make_dense_classifier(c=c, seed=seed)
    calibrated = CalibratedClassifierCV(
        estimator=estimator,
        method=method,
        cv=inner_splits,
        ensemble=False,
    )
    calibrated.fit(np.asarray(X_train), np.asarray(y_train, dtype=int))
    return calibrated.predict_proba(np.asarray(X_test))[:, 1]


def run_text_outer_fold(
    train_texts: Sequence[str],
    y_train: np.ndarray,
    test_texts: Sequence[str],
    *,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> OuterFoldResult:
    train_texts = np.asarray(list(train_texts), dtype=object)
    test_texts = np.asarray(list(test_texts), dtype=object)
    y_train = np.asarray(y_train, dtype=int)
    if len(train_texts) != len(y_train):
        raise ValueError("outer training texts and labels must have the same length")
    inner_oof = np.full(len(y_train), np.nan, dtype=float)
    for inner_fold, (inner_train, inner_validation) in enumerate(inner_splits, start=1):
        probability, _ = fit_predict_tfidf(
            train_texts[inner_train],
            y_train[inner_train],
            train_texts[inner_validation],
            seed=seed + inner_fold,
        )
        if np.isfinite(inner_oof[inner_validation]).any():
            raise ValueError("inner validation records received duplicate predictions")
        inner_oof[inner_validation] = probability
    if not np.isfinite(inner_oof).all():
        raise ValueError("inner OOF predictions are incomplete")
    thresholds = {
        rule: select_threshold(y_train, inner_oof, rule) for rule in THRESHOLD_RULES
    }

    raw_probability, fitted_raw = fit_predict_tfidf(
        train_texts, y_train, test_texts, seed=seed
    )
    platt_probability = _fit_calibrated_text(
        train_texts,
        y_train,
        test_texts,
        inner_splits=inner_splits,
        method="sigmoid",
        seed=seed,
    )
    isotonic_probability = _fit_calibrated_text(
        train_texts,
        y_train,
        test_texts,
        inner_splits=inner_splits,
        method="isotonic",
        seed=seed,
    )
    return OuterFoldResult(
        raw_probability=np.asarray(raw_probability),
        platt_probability=np.asarray(platt_probability),
        isotonic_probability=np.asarray(isotonic_probability),
        inner_oof_probability=inner_oof,
        thresholds=thresholds,
        fitted_raw=fitted_raw,
    )


def run_dense_outer_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    c_grid: Sequence[float],
    seed: int,
) -> DenseOuterFoldResult:
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train, dtype=int)
    if len(X_train) != len(y_train):
        raise ValueError("outer training vectors and labels must have the same length")
    selected_c, c_scores = tune_dense_c(
        X_train,
        y_train,
        inner_splits,
        c_grid=c_grid,
        seed=seed,
    )
    inner_oof = np.full(len(y_train), np.nan, dtype=float)
    for inner_fold, (inner_train, inner_validation) in enumerate(inner_splits, start=1):
        probability, _ = fit_predict_dense(
            X_train[inner_train],
            y_train[inner_train],
            X_train[inner_validation],
            c=selected_c,
            seed=seed + inner_fold,
        )
        if np.isfinite(inner_oof[inner_validation]).any():
            raise ValueError("inner validation records received duplicate predictions")
        inner_oof[inner_validation] = probability
    if not np.isfinite(inner_oof).all():
        raise ValueError("inner OOF predictions are incomplete")
    thresholds = {
        rule: select_threshold(y_train, inner_oof, rule) for rule in THRESHOLD_RULES
    }
    raw_probability, fitted_raw = fit_predict_dense(
        X_train, y_train, X_test, c=selected_c, seed=seed
    )
    platt_probability = _fit_calibrated_dense(
        X_train,
        y_train,
        X_test,
        inner_splits=inner_splits,
        method="sigmoid",
        c=selected_c,
        seed=seed,
    )
    isotonic_probability = _fit_calibrated_dense(
        X_train,
        y_train,
        X_test,
        inner_splits=inner_splits,
        method="isotonic",
        c=selected_c,
        seed=seed,
    )
    return DenseOuterFoldResult(
        raw_probability=np.asarray(raw_probability),
        platt_probability=np.asarray(platt_probability),
        isotonic_probability=np.asarray(isotonic_probability),
        inner_oof_probability=inner_oof,
        thresholds=thresholds,
        selected_c=selected_c,
        c_scores=c_scores,
        fitted_raw=fitted_raw,
    )


def run_text_condition(
    participant_ids: np.ndarray,
    labels: np.ndarray,
    texts: np.ndarray,
    membership: pd.DataFrame,
    *,
    condition: str,
    expected_repeats: int = 10,
    inner_splits_count: int = 3,
    base_seed: int = 20260624,
) -> dict[str, object]:
    participant_ids = np.asarray(participant_ids)
    labels = np.asarray(labels, dtype=int)
    texts = np.asarray(texts, dtype=object)
    if not (len(participant_ids) == len(labels) == len(texts)):
        raise ValueError("participant IDs, labels, and texts must have the same length")
    if len(np.unique(participant_ids)) != len(participant_ids):
        raise ValueError("participant IDs must be unique")
    repeats = sorted(pd.to_numeric(membership["repeat"]).astype(int).unique())
    if repeats != list(range(1, expected_repeats + 1)):
        raise ValueError("split membership repeats do not match expected repeats")
    fold_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    for repeat in repeats:
        folds = sorted(
            pd.to_numeric(
                membership.loc[membership["repeat"].astype(int).eq(repeat), "fold"]
            )
            .astype(int)
            .unique()
        )
        for fold in folds:
            train_idx, test_idx = split_indices(
                membership, repeat, fold, participant_ids
            )
            inner_seed = base_seed + 1000 + 100 * repeat + fold
            inner_splits = make_inner_splits(
                labels[train_idx], n_splits=inner_splits_count, seed=inner_seed
            )
            model_seed = base_seed + repeat
            fold_result = run_text_outer_fold(
                texts[train_idx],
                labels[train_idx],
                texts[test_idx],
                inner_splits=inner_splits,
                seed=model_seed,
            )
            train_hash = _participant_ids_sha256(participant_ids[train_idx])
            test_hash = _participant_ids_sha256(participant_ids[test_idx])
            for local_index, global_index in enumerate(test_idx):
                fold_rows.append(
                    {
                        "condition": condition,
                        "repeat": repeat,
                        "fold": fold,
                        "participant_id": participant_ids[global_index],
                        "label": int(labels[global_index]),
                        "raw_probability": float(
                            fold_result.raw_probability[local_index]
                        ),
                        "platt_probability": float(
                            fold_result.platt_probability[local_index]
                        ),
                        "isotonic_probability": float(
                            fold_result.isotonic_probability[local_index]
                        ),
                        **{
                            f"threshold_{rule}": float(value)
                            for rule, value in fold_result.thresholds.items()
                        },
                        "model_seed": model_seed,
                        "inner_seed": inner_seed,
                        "train_ids_sha256": train_hash,
                        "test_ids_sha256": test_hash,
                    }
                )
            for rule, value in fold_result.thresholds.items():
                threshold_rows.append(
                    {
                        "condition": condition,
                        "repeat": repeat,
                        "fold": fold,
                        "rule": rule,
                        "threshold": float(value),
                        "n_outer_train": int(len(train_idx)),
                        "n_outer_test": int(len(test_idx)),
                        "inner_seed": inner_seed,
                        "model_seed": model_seed,
                        "inner_oof_mean": float(
                            fold_result.inner_oof_probability.mean()
                        ),
                        "inner_oof_sd": float(
                            fold_result.inner_oof_probability.std(ddof=1)
                        ),
                        "train_ids_sha256": train_hash,
                        "test_ids_sha256": test_hash,
                    }
                )
    fold_predictions = pd.DataFrame(fold_rows).sort_values(
        ["repeat", "fold", "participant_id"], kind="stable"
    ).reset_index(drop=True)
    participant_predictions = aggregate_repeated_predictions(
        fold_predictions, expected_repeats=expected_repeats
    )
    thresholds = pd.DataFrame(threshold_rows).sort_values(
        ["repeat", "fold", "rule"], kind="stable"
    ).reset_index(drop=True)
    summary, distributions = summarize_participant_results(
        participant_predictions, condition
    )
    return {
        "fold_predictions": fold_predictions,
        "participant_predictions": participant_predictions,
        "thresholds": thresholds,
        "summary": summary,
        "distributions": distributions,
    }


def run_dense_condition(
    participant_ids: np.ndarray,
    labels: np.ndarray,
    embeddings: np.ndarray,
    membership: pd.DataFrame,
    *,
    condition: str,
    expected_repeats: int = 10,
    inner_splits_count: int = 3,
    c_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    base_seed: int = 20260624,
) -> dict[str, object]:
    participant_ids = np.asarray(participant_ids)
    labels = np.asarray(labels, dtype=int)
    embeddings = np.asarray(embeddings)
    if not (len(participant_ids) == len(labels) == len(embeddings)):
        raise ValueError("participant IDs, labels, and embeddings must have the same length")
    if len(np.unique(participant_ids)) != len(participant_ids):
        raise ValueError("participant IDs must be unique")
    repeats = sorted(pd.to_numeric(membership["repeat"]).astype(int).unique())
    if repeats != list(range(1, expected_repeats + 1)):
        raise ValueError("split membership repeats do not match expected repeats")

    fold_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    for repeat in repeats:
        folds = sorted(
            pd.to_numeric(
                membership.loc[membership["repeat"].astype(int).eq(repeat), "fold"]
            )
            .astype(int)
            .unique()
        )
        for fold in folds:
            train_idx, test_idx = split_indices(
                membership, repeat, fold, participant_ids
            )
            inner_seed = base_seed + 1000 + 100 * repeat + fold
            inner_splits = make_inner_splits(
                labels[train_idx], n_splits=inner_splits_count, seed=inner_seed
            )
            model_seed = base_seed + repeat
            fold_result = run_dense_outer_fold(
                embeddings[train_idx],
                labels[train_idx],
                embeddings[test_idx],
                inner_splits=inner_splits,
                c_grid=c_grid,
                seed=model_seed,
            )
            train_hash = _participant_ids_sha256(participant_ids[train_idx])
            test_hash = _participant_ids_sha256(participant_ids[test_idx])
            for local_index, global_index in enumerate(test_idx):
                fold_rows.append(
                    {
                        "condition": condition,
                        "repeat": repeat,
                        "fold": fold,
                        "participant_id": participant_ids[global_index],
                        "label": int(labels[global_index]),
                        "raw_probability": float(
                            fold_result.raw_probability[local_index]
                        ),
                        "platt_probability": float(
                            fold_result.platt_probability[local_index]
                        ),
                        "isotonic_probability": float(
                            fold_result.isotonic_probability[local_index]
                        ),
                        **{
                            f"threshold_{rule}": float(value)
                            for rule, value in fold_result.thresholds.items()
                        },
                        "selected_c": float(fold_result.selected_c),
                        "model_seed": model_seed,
                        "inner_seed": inner_seed,
                        "train_ids_sha256": train_hash,
                        "test_ids_sha256": test_hash,
                    }
                )
            for rule, value in fold_result.thresholds.items():
                threshold_rows.append(
                    {
                        "condition": condition,
                        "repeat": repeat,
                        "fold": fold,
                        "rule": rule,
                        "threshold": float(value),
                        "selected_c": float(fold_result.selected_c),
                        "n_outer_train": int(len(train_idx)),
                        "n_outer_test": int(len(test_idx)),
                        "inner_seed": inner_seed,
                        "model_seed": model_seed,
                        "inner_oof_mean": float(
                            fold_result.inner_oof_probability.mean()
                        ),
                        "inner_oof_sd": float(
                            fold_result.inner_oof_probability.std(ddof=1)
                        ),
                        "train_ids_sha256": train_hash,
                        "test_ids_sha256": test_hash,
                    }
                )
            for c, score in sorted(fold_result.c_scores.items()):
                tuning_rows.append(
                    {
                        "condition": condition,
                        "repeat": repeat,
                        "fold": fold,
                        "C": float(c),
                        "mean_inner_roc_auc": float(score),
                        "selected": bool(c == fold_result.selected_c),
                        "inner_seed": inner_seed,
                        "train_ids_sha256": train_hash,
                    }
                )
    fold_predictions = pd.DataFrame(fold_rows).sort_values(
        ["repeat", "fold", "participant_id"], kind="stable"
    ).reset_index(drop=True)
    participant_predictions = aggregate_repeated_predictions(
        fold_predictions, expected_repeats=expected_repeats
    )
    thresholds = pd.DataFrame(threshold_rows).sort_values(
        ["repeat", "fold", "rule"], kind="stable"
    ).reset_index(drop=True)
    c_tuning = pd.DataFrame(tuning_rows).sort_values(
        ["repeat", "fold", "C"], kind="stable"
    ).reset_index(drop=True)
    summary, distributions = summarize_participant_results(
        participant_predictions, condition
    )
    return {
        "fold_predictions": fold_predictions,
        "participant_predictions": participant_predictions,
        "thresholds": thresholds,
        "c_tuning": c_tuning,
        "summary": summary,
        "distributions": distributions,
    }


def _sensitivity_specificity(
    y_true: np.ndarray, predictions: np.ndarray
) -> tuple[float, float]:
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()
    sensitivity = float(tp / (tp + fn)) if tp + fn else float("nan")
    specificity = float(tn / (tn + fp)) if tn + fp else float("nan")
    return sensitivity, specificity


def summarize_participant_results(
    participant_predictions: pd.DataFrame, condition: str
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    y = participant_predictions["label"].to_numpy(dtype=int)
    raw = participant_predictions["raw_probability"].to_numpy(dtype=float)
    raw_prediction = participant_predictions["prediction_at_0_5"].to_numpy(dtype=int)
    nested_prediction = participant_predictions[
        "nested_prediction_max_macro_f1"
    ].to_numpy(dtype=int)
    sensitivity_raw, specificity_raw = _sensitivity_specificity(y, raw_prediction)
    sensitivity_nested, specificity_nested = _sensitivity_specificity(
        y, nested_prediction
    )

    summary: dict[str, float | int | str] = {
        "condition": condition,
        "n": int(len(y)),
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
        "roc_auc": float(roc_auc_score(y, raw)),
        "pr_auc": float(average_precision_score(y, raw)),
        "macro_f1_at_0_5": float(
            f1_score(y, raw_prediction, average="macro", zero_division=0)
        ),
        "macro_f1_nested": float(
            f1_score(y, nested_prediction, average="macro", zero_division=0)
        ),
        "sensitivity_at_0_5": sensitivity_raw,
        "sensitivity_nested": sensitivity_nested,
        "specificity_at_0_5": specificity_raw,
        "specificity_nested": specificity_nested,
    }
    probability_columns = {
        "raw": "raw_probability",
        "platt": "platt_probability",
        "isotonic": "isotonic_probability",
    }
    for name, column in probability_columns.items():
        probability = participant_predictions[column].to_numpy(dtype=float)
        ece, effective_bins = quantile_ece(y, probability, n_bins=5)
        intercept, slope = calibration_slope_intercept(y, probability)
        summary[f"{name}_brier"] = float(brier_score_loss(y, probability))
        summary[f"{name}_ece_5bin"] = ece
        summary[f"{name}_ece_effective_bins"] = effective_bins
        summary[f"{name}_calibration_intercept"] = intercept
        summary[f"{name}_calibration_slope"] = slope
        predictions = participant_predictions[
            "prediction_at_0_5" if name == "raw" else f"{name}_prediction_at_0_5"
        ].to_numpy(dtype=int)
        sensitivity, specificity = _sensitivity_specificity(y, predictions)
        summary[f"{name}_sensitivity_at_0_5"] = sensitivity
        summary[f"{name}_specificity_at_0_5"] = specificity
        summary[f"{name}_macro_f1_at_0_5"] = float(
            f1_score(y, predictions, average="macro", zero_division=0)
        )

    distribution_rows = []
    for name, column in probability_columns.items():
        for label in (0, 1):
            values = participant_predictions.loc[
                participant_predictions["label"].eq(label), column
            ].to_numpy(dtype=float)
            q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
            distribution_rows.append(
                {
                    "condition": condition,
                    "probability_type": name,
                    "label": label,
                    "n": int(len(values)),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "median": float(median),
                    "q1": float(q1),
                    "q3": float(q3),
                    "iqr": float(q3 - q1),
                }
            )
    return summary, pd.DataFrame(distribution_rows)


def build_primary_paired_tests(
    participant_tables: dict[str, pd.DataFrame],
    *,
    n_permutations: int = 10_000,
    base_seed: int = 20260624,
) -> pd.DataFrame:
    if set(participant_tables) != set(PRIMARY_CONDITIONS):
        raise ValueError("paired tests require exactly the five primary conditions")
    aligned = {}
    reference_ids = None
    reference_labels = None
    for condition in PRIMARY_CONDITIONS:
        table = participant_tables[condition].sort_values("participant_id", kind="stable")
        ids = table["participant_id"].to_numpy()
        labels = table["label"].to_numpy(dtype=int)
        if reference_ids is None:
            reference_ids, reference_labels = ids, labels
        elif not np.array_equal(ids, reference_ids) or not np.array_equal(
            labels, reference_labels
        ):
            raise ValueError("condition participant IDs or labels are not aligned")
        aligned[condition] = table

    metrics = {
        "roc_auc": ("raw_probability", roc_auc_score),
        "pr_auc": ("raw_probability", average_precision_score),
        "macro_f1_at_0_5": (
            "prediction_at_0_5",
            lambda y, pred: f1_score(y, pred, average="macro", zero_division=0),
        ),
        "macro_f1_nested": (
            "nested_prediction_max_macro_f1",
            lambda y, pred: f1_score(y, pred, average="macro", zero_division=0),
        ),
    }
    rows = []
    test_index = 0
    for condition_a, condition_b in combinations(PRIMARY_CONDITIONS, 2):
        for metric_name, (column, metric) in metrics.items():
            test_index += 1
            result = paired_permutation_test(
                reference_labels,
                aligned[condition_a][column].to_numpy(),
                aligned[condition_b][column].to_numpy(),
                metric,
                n_permutations=n_permutations,
                seed=base_seed + test_index,
            )
            rows.append(
                {
                    "family": "primary_input_source_comparisons",
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    "metric": metric_name,
                    "n": int(len(reference_labels)),
                    **result,
                }
            )
    return apply_bh_by_family(pd.DataFrame(rows))


def summarize_threshold_stability(thresholds: pd.DataFrame) -> pd.DataFrame:
    required = {"condition", "rule", "threshold"}
    if not required.issubset(thresholds.columns):
        raise ValueError(f"threshold table missing columns: {sorted(required-set(thresholds.columns))}")
    rows = []
    for (condition, rule), group in thresholds.groupby(
        ["condition", "rule"], sort=False
    ):
        values = group["threshold"].to_numpy(dtype=float)
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        rows.append(
            {
                "condition": condition,
                "rule": rule,
                "n_folds": int(len(values)),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "median": float(median),
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(q3 - q1),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)
