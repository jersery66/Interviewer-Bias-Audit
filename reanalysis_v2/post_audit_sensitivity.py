from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .source_importance_strict import (
    BASE_SEED,
    bh_fdr,
    paired_auc_inference,
    run_repeated_cv,
    significance_marker,
    summarize_model_metrics,
)
from .submission_audit import select_locked_split, summarize_token_matched_draws


PROBABILITY_COLUMNS = {
    "raw": "raw_probability",
    "platt": "platt_probability",
    "isotonic": "isotonic_probability",
}
NO_EVIDENCE_SENTINEL = "NO_EXPLICIT_SYMPTOM_EVIDENCE"
SYMMETRIC_MODEL_SPECS = OrderedDict(
    {
        "participant_matched": ("participant_matched_text",),
        "interviewer_matched": ("interviewer_matched_text",),
    }
)
RETAINED_ALIGNMENT_MODEL_SPECS = OrderedDict(
    {
        "original_retained": ("original_retained_text",),
        "aligned_retained": ("aligned_retained_text",),
    }
)


def _validate_probability(values: np.ndarray, name: str) -> None:
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError(f"{name} must contain finite probabilities within [0, 1]")


def _stratified_sample_indices(
    labels: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if not len(positive) or not len(negative):
        raise ValueError("Both label classes are required")
    return np.concatenate(
        [
            rng.choice(positive, size=len(positive), replace=True),
            rng.choice(negative, size=len(negative), replace=True),
        ]
    )


def build_cross_fitted_null_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"condition", "repeat", "fold", "participant_id", "label"}
    if missing := required.difference(predictions.columns):
        raise ValueError(f"Missing source prediction columns: {sorted(missing)}")
    if predictions.empty:
        raise ValueError("Source predictions are empty")

    key_columns = ["repeat", "fold", "participant_id", "label"]
    conditions = sorted(predictions["condition"].astype(str).unique())
    reference = (
        predictions.loc[predictions["condition"].eq(conditions[0]), key_columns]
        .sort_values(["repeat", "participant_id"])
        .reset_index(drop=True)
    )
    if reference.duplicated(["repeat", "participant_id"]).any():
        raise ValueError("Each participant must have one locked prediction per repeat")
    for condition in conditions[1:]:
        candidate = (
            predictions.loc[predictions["condition"].eq(condition), key_columns]
            .sort_values(["repeat", "participant_id"])
            .reset_index(drop=True)
        )
        if not reference.equals(candidate):
            raise ValueError("Conditions do not share identical participants, labels, and folds")

    rows: list[dict[str, float | int | str]] = []
    for repeat, repeat_rows in reference.groupby("repeat", sort=True):
        for fold, test_rows in repeat_rows.groupby("fold", sort=True):
            train_rows = repeat_rows.loc[~repeat_rows["fold"].eq(fold)]
            if train_rows.empty:
                raise ValueError("Outer training fold is empty")
            train_positive = int(train_rows["label"].sum())
            prevalence = float(train_rows["label"].mean())
            for row in test_rows.itertuples(index=False):
                rows.append(
                    {
                        "repeat": int(repeat),
                        "fold": int(fold),
                        "participant_id": int(row.participant_id),
                        "label": int(row.label),
                        "null_probability": prevalence,
                        "outer_train_n": int(len(train_rows)),
                        "outer_train_positive_n": train_positive,
                        "null_source": "outer_training_fold_prevalence",
                    }
                )
    return pd.DataFrame(rows).sort_values(["repeat", "participant_id"]).reset_index(drop=True)


def summarize_brier_skill(
    predictions: pd.DataFrame,
    null_predictions: pd.DataFrame,
    *,
    n_bootstrap: int = 5000,
    seed: int = BASE_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "condition",
        "repeat",
        "fold",
        "participant_id",
        "label",
        *PROBABILITY_COLUMNS.values(),
    }
    if missing := required.difference(predictions.columns):
        raise ValueError(f"Missing Brier prediction columns: {sorted(missing)}")
    null_required = {
        "repeat",
        "fold",
        "participant_id",
        "label",
        "null_probability",
    }
    if missing := null_required.difference(null_predictions.columns):
        raise ValueError(f"Missing null prediction columns: {sorted(missing)}")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")

    merged = predictions.merge(
        null_predictions[list(null_required)],
        on=["repeat", "fold", "participant_id", "label"],
        how="left",
        validate="many_to_one",
    )
    if merged["null_probability"].isna().any():
        raise ValueError("Some model predictions lack a cross-fitted null probability")

    labels_all = null_predictions["label"].to_numpy(dtype=int)
    cohort_prevalence = float(labels_all.mean())
    cohort_null_brier = float(np.mean((labels_all - cohort_prevalence) ** 2))
    metric_rows: list[dict[str, float | int | str]] = []
    fold_rows: list[dict[str, float | int | str]] = []
    for condition_index, (condition, group) in enumerate(
        merged.groupby("condition", sort=True)
    ):
        group = group.sort_values("participant_id").reset_index(drop=True)
        labels = group["label"].to_numpy(dtype=int)
        null_probability = group["null_probability"].to_numpy(dtype=float)
        _validate_probability(null_probability, "null_probability")
        null_error = (labels - null_probability) ** 2
        trainfold_null_brier = float(null_error.mean())
        for variant_index, (variant, column) in enumerate(PROBABILITY_COLUMNS.items()):
            probability = group[column].to_numpy(dtype=float)
            _validate_probability(probability, column)
            model_error = (labels - probability) ** 2
            model_brier = float(model_error.mean())
            bss_trainfold = float(1.0 - model_brier / trainfold_null_brier)
            bss_cohort = float(1.0 - model_brier / cohort_null_brier)

            rng = np.random.default_rng(
                seed + 101 * condition_index + 10_007 * variant_index
            )
            boot_brier = np.empty(n_bootstrap, dtype=float)
            boot_bss = np.empty(n_bootstrap, dtype=float)
            for bootstrap_index in range(n_bootstrap):
                sampled = _stratified_sample_indices(labels, rng)
                sampled_model = float(model_error[sampled].mean())
                sampled_null = float(null_error[sampled].mean())
                boot_brier[bootstrap_index] = sampled_model
                boot_bss[bootstrap_index] = 1.0 - sampled_model / sampled_null

            metric_rows.append(
                {
                    "condition": str(condition),
                    "probability_variant": variant,
                    "n": int(len(group)),
                    "n_positive": int(labels.sum()),
                    "model_brier": model_brier,
                    "model_brier_ci_low": float(np.quantile(boot_brier, 0.025)),
                    "model_brier_ci_high": float(np.quantile(boot_brier, 0.975)),
                    "trainfold_null_brier": trainfold_null_brier,
                    "bss_trainfold": bss_trainfold,
                    "bss_trainfold_ci_low": float(np.quantile(boot_bss, 0.025)),
                    "bss_trainfold_ci_high": float(np.quantile(boot_bss, 0.975)),
                    "cohort_prevalence": cohort_prevalence,
                    "cohort_null_brier_descriptive": cohort_null_brier,
                    "bss_cohort_descriptive": bss_cohort,
                    "bootstrap_resamples": int(n_bootstrap),
                    "formal_null_source": "outer_training_fold_prevalence",
                    "cohort_null_scope": "descriptive_full_cohort_only",
                }
            )

            for fold, fold_group in group.groupby("fold", sort=True):
                fold_labels = fold_group["label"].to_numpy(dtype=int)
                fold_probability = fold_group[column].to_numpy(dtype=float)
                fold_null = fold_group["null_probability"].to_numpy(dtype=float)
                fold_model_brier = float(brier_score_loss(fold_labels, fold_probability))
                fold_null_brier = float(brier_score_loss(fold_labels, fold_null))
                fold_rows.append(
                    {
                        "condition": str(condition),
                        "probability_variant": variant,
                        "repeat": int(fold_group["repeat"].iloc[0]),
                        "fold": int(fold),
                        "n": int(len(fold_group)),
                        "n_positive": int(fold_labels.sum()),
                        "model_brier": fold_model_brier,
                        "trainfold_null_brier": fold_null_brier,
                        "bss_trainfold": float(1.0 - fold_model_brier / fold_null_brier),
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(fold_rows)


def _token_words(text: object) -> list[str]:
    if pd.isna(text):
        return []
    return str(text).split()


def _merge_source_texts(
    participant: pd.DataFrame,
    interviewer: pd.DataFrame,
) -> pd.DataFrame:
    required = {"participant_id", "paper_label_phq8_ge10", "text"}
    for name, frame in (("participant", participant), ("interviewer", interviewer)):
        if missing := required.difference(frame.columns):
            raise ValueError(f"Missing {name} columns: {sorted(missing)}")
        if frame["participant_id"].duplicated().any():
            raise ValueError(f"Duplicate IDs in {name} input")
    merged = participant[list(required)].merge(
        interviewer[list(required)],
        on="participant_id",
        how="inner",
        suffixes=("_participant", "_interviewer"),
        validate="one_to_one",
    )
    if len(merged) != len(participant) or len(merged) != len(interviewer):
        raise ValueError("Participant and interviewer ID sets differ")
    if not merged["paper_label_phq8_ge10_participant"].equals(
        merged["paper_label_phq8_ge10_interviewer"]
    ):
        raise ValueError("Participant and interviewer labels disagree")
    return merged.sort_values("participant_id").reset_index(drop=True)


def _random_window(
    words: Sequence[str],
    target: int,
    rng: np.random.Generator,
) -> list[str]:
    if target == 0:
        return []
    if target < 0 or target > len(words):
        raise ValueError("Invalid token target")
    start = int(rng.integers(0, len(words) - target + 1))
    return list(words[start : start + target])


def _position_window(words: Sequence[str], target: int, position: str) -> list[str]:
    if position not in {"early", "middle", "late"}:
        raise ValueError("position must be early, middle, or late")
    if target == 0:
        return []
    if target < 0 or target > len(words):
        raise ValueError("Invalid token target")
    available = len(words) - target
    if position == "early":
        start = 0
    elif position == "middle":
        start = available // 2
    else:
        start = available
    return list(words[start : start + target])


def _symmetric_audit_row(
    participant_id: int,
    participant_count: int,
    interviewer_count: int,
    target: int,
) -> dict[str, float | int | bool]:
    return {
        "participant_id": participant_id,
        "participant_token_count": participant_count,
        "interviewer_token_count": interviewer_count,
        "target_token_count": target,
        "participant_retained_fraction": (
            target / participant_count if participant_count else 0.0
        ),
        "interviewer_retained_fraction": (
            target / interviewer_count if interviewer_count else 0.0
        ),
        "participant_truncated": target < participant_count,
        "interviewer_truncated": target < interviewer_count,
        "zero_target": target == 0,
        "target_below_10": target < 10,
    }


def build_symmetric_half_min_draw(
    participant: pd.DataFrame,
    interviewer: pd.DataFrame,
    *,
    draw: int,
    seed: int = BASE_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if draw <= 0:
        raise ValueError("draw must be positive")
    merged = _merge_source_texts(participant, interviewer)
    matched_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, float | int | bool]] = []
    for row in merged.itertuples(index=False):
        participant_words = _token_words(row.text_participant)
        interviewer_words = _token_words(row.text_interviewer)
        target = int(np.floor(0.5 * min(len(participant_words), len(interviewer_words))))
        local_seed = int(seed + 2_000_003 * draw + 97 * int(row.participant_id))
        rng = np.random.default_rng(local_seed)
        participant_matched = _random_window(participant_words, target, rng)
        interviewer_matched = _random_window(interviewer_words, target, rng)
        matched_rows.append(
            {
                "participant_id": int(row.participant_id),
                "label": int(row.paper_label_phq8_ge10_participant),
                "participant_matched_text": " ".join(participant_matched),
                "interviewer_matched_text": " ".join(interviewer_matched),
            }
        )
        audit_rows.append(
            _symmetric_audit_row(
                int(row.participant_id),
                len(participant_words),
                len(interviewer_words),
                target,
            )
        )
    return pd.DataFrame(matched_rows), pd.DataFrame(audit_rows)


def build_symmetric_position_dataset(
    participant: pd.DataFrame,
    interviewer: pd.DataFrame,
    *,
    position: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = _merge_source_texts(participant, interviewer)
    matched_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, float | int | bool | str]] = []
    for row in merged.itertuples(index=False):
        participant_words = _token_words(row.text_participant)
        interviewer_words = _token_words(row.text_interviewer)
        target = int(np.floor(0.5 * min(len(participant_words), len(interviewer_words))))
        participant_matched = _position_window(participant_words, target, position)
        interviewer_matched = _position_window(interviewer_words, target, position)
        matched_rows.append(
            {
                "participant_id": int(row.participant_id),
                "label": int(row.paper_label_phq8_ge10_participant),
                "participant_matched_text": " ".join(participant_matched),
                "interviewer_matched_text": " ".join(interviewer_matched),
            }
        )
        audit = _symmetric_audit_row(
            int(row.participant_id),
            len(participant_words),
            len(interviewer_words),
            target,
        )
        audit["position"] = position
        audit_rows.append(audit)
    return pd.DataFrame(matched_rows), pd.DataFrame(audit_rows)


def _quartiles(values: np.ndarray) -> tuple[float, float, float]:
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    return float(q1), float(median), float(q3)


def summarize_token_budget_audit(
    audit: pd.DataFrame,
    *,
    control: str,
) -> pd.DataFrame:
    target_column = (
        "target_token_count" if "target_token_count" in audit else "matched_token_count"
    )
    required = {
        "participant_id",
        "participant_token_count",
        "interviewer_token_count",
        target_column,
    }
    if missing := required.difference(audit.columns):
        raise ValueError(f"Missing token audit columns: {sorted(missing)}")
    target = audit[target_column].to_numpy(dtype=float)
    target_q1, target_median, target_q3 = _quartiles(target)
    rows: list[dict[str, float | int | str]] = []
    for source in ("participant", "interviewer"):
        original = audit[f"{source}_token_count"].to_numpy(dtype=float)
        nonempty = original > 0
        retained = target[nonempty] / original[nonempty]
        retained_q1, retained_median, retained_q3 = _quartiles(retained)
        original_q1, original_median, original_q3 = _quartiles(original)
        rows.append(
            {
                "control": control,
                "source": source,
                "n": int(len(audit)),
                "original_token_q1": original_q1,
                "original_token_median": original_median,
                "original_token_q3": original_q3,
                "truncated_n": int(np.count_nonzero(target < original)),
                "mean_retained_fraction_nonempty": float(retained.mean()),
                "retained_fraction_q1_nonempty": retained_q1,
                "retained_fraction_median_nonempty": retained_median,
                "retained_fraction_q3_nonempty": retained_q3,
                "zero_source_n": int(np.count_nonzero(original == 0)),
                "target_token_min": float(target.min()),
                "target_token_q1": target_q1,
                "target_token_median": target_median,
                "target_token_q3": target_q3,
                "target_token_max": float(target.max()),
                "zero_target_n": int(np.count_nonzero(target == 0)),
                "target_below_10_n": int(np.count_nonzero(target < 10)),
            }
        )
    rows.append(
        {
            "control": control,
            "source": "shared_target",
            "n": int(len(audit)),
            "original_token_q1": float("nan"),
            "original_token_median": float("nan"),
            "original_token_q3": float("nan"),
            "truncated_n": 0,
            "mean_retained_fraction_nonempty": float("nan"),
            "retained_fraction_q1_nonempty": float("nan"),
            "retained_fraction_median_nonempty": float("nan"),
            "retained_fraction_q3_nonempty": float("nan"),
            "zero_source_n": 0,
            "target_token_min": float(target.min()),
            "target_token_q1": target_q1,
            "target_token_median": target_median,
            "target_token_q3": target_q3,
            "target_token_max": float(target.max()),
            "zero_target_n": int(np.count_nonzero(target == 0)),
            "target_below_10_n": int(np.count_nonzero(target < 10)),
        }
    )
    return pd.DataFrame(rows)


def _normalize_retained_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text == NO_EVIDENCE_SENTINEL:
        return ""
    return " ".join(text.split())


def build_retained_alignment_frames(
    spans: pd.DataFrame,
    frozen_c5: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    span_required = {
        "participant_id",
        "source_order",
        "domain",
        "polarity",
        "original_model_quote",
        "exact_quote",
        "review_status",
    }
    if missing := span_required.difference(spans.columns):
        raise ValueError(f"Missing retained span columns: {sorted(missing)}")
    frozen_required = {"participant_id", "paper_label_phq8_ge10", "text"}
    if missing := frozen_required.difference(frozen_c5.columns):
        raise ValueError(f"Missing frozen C5 columns: {sorted(missing)}")
    if frozen_c5["participant_id"].duplicated().any():
        raise ValueError("Frozen C5 participant IDs must be unique")
    if spans[["original_model_quote", "exact_quote"]].isna().any().any():
        raise ValueError("Retained span text fields cannot be missing")

    ordered = spans.sort_values(["participant_id", "source_order"]).copy()
    ordered["is_alignment_revision"] = ordered["review_status"].eq(
        "manual_pass_corrected_quote_added"
    )
    original_grouped = ordered.groupby("participant_id")["original_model_quote"].agg(
        lambda values: "\n".join(str(value) for value in values)
    )
    aligned_grouped = ordered.groupby("participant_id")["exact_quote"].agg(
        lambda values: "\n".join(str(value) for value in values)
    )
    span_counts = ordered.groupby("participant_id").size()
    corrected_counts = ordered.groupby("participant_id")["is_alignment_revision"].sum()

    original_rows: list[dict[str, object]] = []
    aligned_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for row in frozen_c5.sort_values("participant_id").itertuples(index=False):
        participant_id = int(row.participant_id)
        original_text = str(original_grouped.get(participant_id, ""))
        aligned_text = str(aligned_grouped.get(participant_id, ""))
        frozen_normalized = _normalize_retained_text(row.text)
        if _normalize_retained_text(aligned_text) != frozen_normalized:
            raise ValueError(
                f"Aligned retained spans do not reconstruct frozen C5 for participant {participant_id}"
            )
        original_normalized = _normalize_retained_text(original_text)
        aligned_normalized = _normalize_retained_text(aligned_text)
        label = int(row.paper_label_phq8_ge10)
        original_rows.append(
            {
                "participant_id": participant_id,
                "label": label,
                "retained_text": original_text,
            }
        )
        aligned_rows.append(
            {
                "participant_id": participant_id,
                "label": label,
                "retained_text": aligned_text,
            }
        )
        original_tokens = len(original_text.split())
        aligned_tokens = len(aligned_text.split())
        audit_rows.append(
            {
                "participant_id": participant_id,
                "retained_span_count": int(span_counts.get(participant_id, 0)),
                "corrected_span_count": int(corrected_counts.get(participant_id, 0)),
                "original_token_count": original_tokens,
                "aligned_token_count": aligned_tokens,
                "token_count_delta": aligned_tokens - original_tokens,
                "representation_changed": original_normalized != aligned_normalized,
                "domain_held_fixed": True,
                "polarity_held_fixed": True,
                "source_order_held_fixed": True,
                "retained_status_held_fixed": True,
            }
        )
    return (
        pd.DataFrame(original_rows),
        pd.DataFrame(aligned_rows),
        pd.DataFrame(audit_rows),
    )


def _metric_functions() -> dict[str, Callable[[np.ndarray, np.ndarray], float]]:
    return {
        "roc_auc": lambda labels, probability: float(
            roc_auc_score(labels, probability)
        ),
        "pr_auc": lambda labels, probability: float(
            average_precision_score(labels, probability)
        ),
        "brier": lambda labels, probability: float(
            brier_score_loss(labels, probability)
        ),
    }


def paired_metric_sensitivity(
    labels: Sequence[int] | np.ndarray,
    aligned_probability: Sequence[float] | np.ndarray,
    original_probability: Sequence[float] | np.ndarray,
    *,
    n_bootstrap: int = 5000,
    n_permutations: int = 10_000,
    seed: int = BASE_SEED,
) -> pd.DataFrame:
    labels = np.asarray(labels, dtype=int)
    aligned = np.asarray(aligned_probability, dtype=float)
    original = np.asarray(original_probability, dtype=float)
    if not (len(labels) == len(aligned) == len(original)):
        raise ValueError("Labels and paired probabilities must have equal length")
    if n_bootstrap <= 0 or n_permutations <= 0:
        raise ValueError("Resampling counts must be positive")
    _validate_probability(aligned, "aligned_probability")
    _validate_probability(original, "original_probability")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Both binary label classes are required")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    for metric_index, (metric_name, metric_function) in enumerate(
        _metric_functions().items()
    ):
        aligned_value = metric_function(labels, aligned)
        original_value = metric_function(labels, original)
        observed = aligned_value - original_value
        boot_values = np.empty(n_bootstrap, dtype=float)
        for bootstrap_index in range(n_bootstrap):
            sampled = _stratified_sample_indices(labels, rng)
            boot_values[bootstrap_index] = metric_function(
                labels[sampled], aligned[sampled]
            ) - metric_function(labels[sampled], original[sampled])

        permutation_values = np.empty(n_permutations, dtype=float)
        metric_rng = np.random.default_rng(seed + 1009 * (metric_index + 1))
        for permutation_index in range(n_permutations):
            swap = metric_rng.integers(0, 2, size=len(labels), dtype=np.int8).astype(bool)
            permuted_aligned = np.where(swap, original, aligned)
            permuted_original = np.where(swap, aligned, original)
            permutation_values[permutation_index] = metric_function(
                labels, permuted_aligned
            ) - metric_function(labels, permuted_original)
        p_value = float(
            (1 + np.count_nonzero(np.abs(permutation_values) >= abs(observed)))
            / (n_permutations + 1)
        )
        rows.append(
            {
                "metric": metric_name,
                "aligned_value": aligned_value,
                "original_value": original_value,
                "aligned_minus_original": observed,
                "ci_lower": float(np.quantile(boot_values, 0.025)),
                "ci_upper": float(np.quantile(boot_values, 0.975)),
                "p_value": p_value,
                "n_bootstrap_valid": int(len(boot_values)),
                "n_permutations": int(n_permutations),
                "difference_direction": (
                    "aligned_minus_original; negative_is_better"
                    if metric_name == "brier"
                    else "aligned_minus_original; positive_is_better"
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["q_value"] = bh_fdr(result["p_value"].to_numpy(dtype=float))
    result["significance"] = result["q_value"].map(significance_marker)
    return result


def run_symmetric_half_min_control(
    participant_text: pd.DataFrame,
    interviewer_text: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    n_draws: int = 50,
    n_bootstrap: int = 5000,
    seed: int = BASE_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if n_draws <= 0:
        raise ValueError("n_draws must be positive")
    locked = select_locked_split(splits)
    prediction_rows: list[pd.DataFrame] = []
    draw_metric_rows: list[dict[str, float | int | str]] = []
    length_audit: pd.DataFrame | None = None
    for draw in range(1, n_draws + 1):
        matched, audit = build_symmetric_half_min_draw(
            participant_text,
            interviewer_text,
            draw=draw,
            seed=seed,
        )
        if length_audit is None:
            length_audit = audit
        elif not length_audit.equals(audit):
            raise AssertionError("Symmetric token budget audit changed across draws")
        _, participant_oof = run_repeated_cv(
            matched,
            locked,
            model_specs=SYMMETRIC_MODEL_SPECS,
            base_seed=seed + 20_000 * draw,
        )
        participant_oof.insert(0, "draw", draw)
        prediction_rows.append(participant_oof)
        labels = participant_oof["label"].to_numpy(dtype=int)
        for model in SYMMETRIC_MODEL_SPECS:
            probability = participant_oof[f"{model}_prob"].to_numpy(dtype=float)
            draw_metric_rows.append(
                {
                    "draw": draw,
                    "condition": model,
                    "roc_auc": float(roc_auc_score(labels, probability)),
                    "pr_auc": float(average_precision_score(labels, probability)),
                    "brier": float(brier_score_loss(labels, probability)),
                }
            )
    if length_audit is None:
        raise AssertionError("Symmetric token audit was not generated")
    predictions = pd.concat(prediction_rows, ignore_index=True)
    draw_metrics = pd.DataFrame(draw_metric_rows)
    metrics, comparison = summarize_token_matched_draws(
        predictions,
        draw_metrics,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    comparison["interpretation_scope"] = "symmetric_half_min_information_budget"
    return predictions, draw_metrics, metrics, comparison, length_audit


def run_symmetric_position_control(
    participant_text: pd.DataFrame,
    interviewer_text: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    n_bootstrap: int = 5000,
    n_permutations: int = 10_000,
    seed: int = BASE_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    locked = select_locked_split(splits)
    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[pd.DataFrame] = []
    comparison_rows: list[dict[str, float | int | str]] = []
    budget_audit: pd.DataFrame | None = None
    for position_index, position in enumerate(("early", "middle", "late")):
        matched, audit = build_symmetric_position_dataset(
            participant_text,
            interviewer_text,
            position=position,
        )
        comparable_audit = audit.drop(columns="position")
        if budget_audit is None:
            budget_audit = comparable_audit
        elif not budget_audit.equals(comparable_audit):
            raise AssertionError("Symmetric positional budgets differ by position")
        _, participant_oof = run_repeated_cv(
            matched,
            locked,
            model_specs=SYMMETRIC_MODEL_SPECS,
            base_seed=seed + 30_000 * (position_index + 1),
        )
        participant_oof.insert(0, "position", position)
        prediction_rows.append(participant_oof)
        metrics, auc_ci = summarize_model_metrics(
            participant_oof,
            n_bootstrap=n_bootstrap,
            seed=seed + 101 * position_index,
        )
        metrics = metrics.merge(auc_ci, on="model", how="left", validate="one_to_one")
        metrics.insert(0, "position", position)
        metric_rows.append(metrics)
        inference = paired_auc_inference(
            participant_oof["label"],
            participant_oof["interviewer_matched_prob"],
            participant_oof["participant_matched_prob"],
            n_bootstrap=n_bootstrap,
            n_permutations=n_permutations,
            seed=seed + 1009 * position_index,
        )
        comparison_rows.append(
            {
                "position": position,
                "comparison": "interviewer_matched vs participant_matched",
                "comparison_family": "symmetric_position_source_auc",
                **inference,
            }
        )
    if budget_audit is None:
        raise AssertionError("Symmetric positional audit was not generated")
    comparisons = pd.DataFrame(comparison_rows)
    comparisons["q_value"] = bh_fdr(comparisons["p_value"].to_numpy(dtype=float))
    comparisons["significance"] = comparisons["q_value"].map(significance_marker)
    return (
        pd.concat(prediction_rows, ignore_index=True),
        pd.concat(metric_rows, ignore_index=True),
        comparisons,
        budget_audit,
    )


def run_retained_alignment_sensitivity(
    original: pd.DataFrame,
    aligned: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    n_bootstrap: int = 5000,
    n_permutations: int = 10_000,
    seed: int = BASE_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {"participant_id", "label", "retained_text"}
    for name, frame in (("original", original), ("aligned", aligned)):
        if missing := required.difference(frame.columns):
            raise ValueError(f"Missing {name} retained-alignment columns: {sorted(missing)}")
        if frame["participant_id"].duplicated().any():
            raise ValueError(f"Duplicate IDs in {name} retained-alignment frame")
    frame = original[list(required)].rename(
        columns={"label": "label_original", "retained_text": "original_retained_text"}
    ).merge(
        aligned[list(required)].rename(
            columns={"label": "label_aligned", "retained_text": "aligned_retained_text"}
        ),
        on="participant_id",
        how="inner",
        validate="one_to_one",
    )
    if len(frame) != len(original) or len(frame) != len(aligned):
        raise ValueError("Original and aligned participant sets differ")
    if not frame["label_original"].equals(frame["label_aligned"]):
        raise ValueError("Original and aligned labels differ")
    frame["label"] = frame.pop("label_original")
    frame = frame.drop(columns="label_aligned")
    locked = select_locked_split(splits)
    _, participant_oof = run_repeated_cv(
        frame,
        locked,
        model_specs=RETAINED_ALIGNMENT_MODEL_SPECS,
        base_seed=seed,
    )
    metrics, auc_ci = summarize_model_metrics(
        participant_oof,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    metrics = metrics.merge(auc_ci, on="model", how="left", validate="one_to_one")
    differences = paired_metric_sensitivity(
        participant_oof["label"],
        participant_oof["aligned_retained_prob"],
        participant_oof["original_retained_prob"],
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
        seed=seed,
    )
    absolute_change = np.abs(
        participant_oof["aligned_retained_prob"].to_numpy(dtype=float)
        - participant_oof["original_retained_prob"].to_numpy(dtype=float)
    )
    change_summary = pd.DataFrame(
        [
            {
                "n": int(len(absolute_change)),
                "absolute_change_mean": float(absolute_change.mean()),
                "absolute_change_median": float(np.median(absolute_change)),
                "absolute_change_q1": float(np.quantile(absolute_change, 0.25)),
                "absolute_change_q3": float(np.quantile(absolute_change, 0.75)),
                "absolute_change_q95": float(np.quantile(absolute_change, 0.95)),
                "absolute_change_max": float(absolute_change.max()),
                "absolute_change_ge_0_01_n": int(np.count_nonzero(absolute_change >= 0.01)),
                "absolute_change_ge_0_05_n": int(np.count_nonzero(absolute_change >= 0.05)),
            }
        ]
    )
    return participant_oof, metrics, differences, change_summary
