from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    recall_score,
    roc_auc_score,
)

from .metrics import calibration_slope_intercept, quantile_ece
from .source_importance_strict import (
    BASE_SEED,
    bh_fdr,
    paired_auc_inference,
    run_repeated_cv,
    significance_marker,
    summarize_model_metrics,
)


LOCKED_REPEAT = 1
TOKEN_MATCH_DRAWS = 50

STRUCTURAL_MODEL_SPECS: "OrderedDict[str, tuple[str, ...]]" = OrderedDict(
    {
        "S_length": ("length_only",),
        "S_interaction": ("interaction_structure",),
        "S_protocol": ("protocol_structure",),
        "S_all": ("length_only", "interaction_structure", "protocol_structure"),
    }
)

TOKEN_MATCH_MODEL_SPECS: "OrderedDict[str, tuple[str, ...]]" = OrderedDict(
    {
        "participant_matched": ("participant_matched_text",),
        "interviewer_matched": ("interviewer_matched_text",),
    }
)

SOURCE_CONDITIONS: "OrderedDict[str, str]" = OrderedDict(
    {
        "full_transcript": "full_transcript_fold_oof.csv",
        "participant_speech": "participant_speech_fold_oof.csv",
        "interviewer_speech": "interviewer_speech_fold_oof.csv",
        "non_explicit_interviewer_speech": "non_explicit_interviewer_speech_fold_oof.csv",
        "participant_symptom_evidence": "participant_symptom_evidence_fold_oof.csv",
    }
)


def select_locked_split(splits: pd.DataFrame, *, repeat: int = LOCKED_REPEAT) -> pd.DataFrame:
    required = {"repeat", "fold", "participant_id", "role"}
    if missing := required.difference(splits.columns):
        raise ValueError(f"Missing split columns: {sorted(missing)}")
    locked = splits.loc[splits["repeat"].eq(repeat)].copy()
    if locked.empty:
        raise ValueError(f"Locked repeat {repeat} is absent")
    test = locked.loc[locked["role"].eq("test")]
    counts = test.groupby("participant_id").size()
    participants = locked["participant_id"].nunique()
    if len(counts) != participants or not counts.eq(1).all():
        raise ValueError("Locked split must test each participant exactly once")
    return locked.sort_values(["fold", "participant_id", "role"]).reset_index(drop=True)


def build_structural_features(turns: pd.DataFrame) -> pd.DataFrame:
    required = {
        "participant_id",
        "paper_label_phq8_ge10",
        "paper_phq8_score",
        "turn_index",
        "start_time",
        "stop_time",
        "speaker_norm",
        "word_count",
        "is_prompt_annotated",
        "is_explicit_clinical_prompt",
        "prompt_id",
        "normalized_spoken_text",
    }
    if missing := required.difference(turns.columns):
        raise ValueError(f"Missing turn columns: {sorted(missing)}")
    if turns.empty:
        raise ValueError("Turn table is empty")

    rows: list[dict[str, float | int]] = []
    for participant_id, group in turns.groupby("participant_id", sort=True):
        labels = group["paper_label_phq8_ge10"].dropna().astype(int).unique()
        scores = group["paper_phq8_score"].dropna().astype(float).unique()
        if len(labels) != 1 or len(scores) != 1:
            raise ValueError(f"Non-constant outcome for participant {participant_id}")

        participant_mask = group["speaker_norm"].eq("participant")
        interviewer_mask = group["speaker_norm"].eq("interviewer")
        prompt_mask = interviewer_mask & group["is_prompt_annotated"].fillna(0).astype(int).eq(1)
        clinical_mask = (
            interviewer_mask
            & group["is_explicit_clinical_prompt"].fillna(0).astype(int).eq(1)
        )
        participant_turns = int(participant_mask.sum())
        interviewer_turns = int(interviewer_mask.sum())
        participant_words = float(group.loc[participant_mask, "word_count"].fillna(0).sum())
        interviewer_words = float(group.loc[interviewer_mask, "word_count"].fillna(0).sum())
        start = float(group["start_time"].min())
        stop = float(group["stop_time"].max())
        protocol_ids = group.loc[prompt_mask, "prompt_id"].dropna().astype(str)
        clinical_prompts = (
            group.loc[clinical_mask, "normalized_spoken_text"].dropna().astype(str)
        )

        rows.append(
            {
                "participant_id": int(participant_id),
                "label": int(labels[0]),
                "phq8_score": float(scores[0]),
                "length_only__participant_word_count": participant_words,
                "length_only__interviewer_word_count": interviewer_words,
                "length_only__total_word_count": participant_words + interviewer_words,
                "length_only__interview_duration_sec": stop - start,
                "interaction_structure__participant_turn_count": participant_turns,
                "interaction_structure__interviewer_turn_count": interviewer_turns,
                "interaction_structure__total_turn_count": int(len(group)),
                "interaction_structure__participant_mean_words_per_turn": (
                    participant_words / participant_turns if participant_turns else 0.0
                ),
                "interaction_structure__interviewer_mean_words_per_turn": (
                    interviewer_words / interviewer_turns if interviewer_turns else 0.0
                ),
                "protocol_structure__interviewer_question_count": int(prompt_mask.sum()),
                "protocol_structure__clinical_question_count": int(clinical_mask.sum()),
                "protocol_structure__unique_protocol_prompt_count": int(protocol_ids.nunique()),
                "protocol_structure__clinical_prompt_coverage_count": int(
                    clinical_prompts.nunique()
                ),
                "protocol_structure__non_explicit_interviewer_turn_count": int(
                    (interviewer_mask & ~clinical_mask).sum()
                ),
            }
        )
    result = pd.DataFrame(rows)
    if result["participant_id"].duplicated().any():
        raise AssertionError("Structural output must have one row per participant")
    return result


def _words(text: object) -> list[str]:
    if pd.isna(text):
        return []
    return str(text).split()


def _contiguous_window(words: Sequence[str], target: int, rng: np.random.Generator) -> list[str]:
    if target < 0 or target > len(words):
        raise ValueError("Invalid token target")
    if target == 0:
        return []
    if target == len(words):
        return list(words)
    start = int(rng.integers(0, len(words) - target + 1))
    return list(words[start : start + target])


def build_token_matched_draw(
    participant: pd.DataFrame,
    interviewer: pd.DataFrame,
    *,
    draw: int,
    seed: int = BASE_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    if not merged["paper_label_phq8_ge10_participant"].equals(
        merged["paper_label_phq8_ge10_interviewer"]
    ):
        raise ValueError("Participant and interviewer labels disagree")

    matched_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, float | int]] = []
    for row in merged.sort_values("participant_id").itertuples(index=False):
        participant_words = _words(row.text_participant)
        interviewer_words = _words(row.text_interviewer)
        target = min(len(participant_words), len(interviewer_words))
        local_seed = int(seed + 1_000_003 * int(draw) + 97 * int(row.participant_id))
        rng = np.random.default_rng(local_seed)
        participant_matched = _contiguous_window(participant_words, target, rng)
        interviewer_matched = _contiguous_window(interviewer_words, target, rng)
        matched_rows.append(
            {
                "participant_id": int(row.participant_id),
                "label": int(row.paper_label_phq8_ge10_participant),
                "participant_matched_text": " ".join(participant_matched),
                "interviewer_matched_text": " ".join(interviewer_matched),
            }
        )
        audit_rows.append(
            {
                "participant_id": int(row.participant_id),
                "participant_token_count": len(participant_words),
                "interviewer_token_count": len(interviewer_words),
                "matched_token_count": target,
                "participant_retained_fraction": (
                    target / len(participant_words) if participant_words else 0.0
                ),
                "interviewer_retained_fraction": (
                    target / len(interviewer_words) if interviewer_words else 0.0
                ),
            }
        )
    return pd.DataFrame(matched_rows), pd.DataFrame(audit_rows)


def stratified_auc_ci(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    n_bootstrap: int = 5000,
    seed: int = BASE_SEED,
    level: float = 0.95,
) -> tuple[float, float]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    positive = probabilities[labels == 1]
    negative = probabilities[labels == 0]
    if positive.size == 0 or negative.size == 0:
        return np.nan, np.nan
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    rng = np.random.default_rng(seed)
    pos = positive[rng.integers(0, positive.size, size=(n_bootstrap, positive.size))]
    neg = negative[rng.integers(0, negative.size, size=(n_bootstrap, negative.size))]
    differences = pos[:, :, None] - neg[:, None, :]
    samples = (
        np.count_nonzero(differences > 0, axis=(1, 2))
        + 0.5 * np.count_nonzero(differences == 0, axis=(1, 2))
    ) / (positive.size * negative.size)
    alpha = 1.0 - level
    low, high = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def summarize_locked_source_predictions(
    predictions: pd.DataFrame,
    *,
    n_bootstrap: int = 5000,
    seed: int = BASE_SEED,
) -> pd.DataFrame:
    required = {
        "condition",
        "participant_id",
        "label",
        "raw_probability",
        "platt_probability",
        "threshold_max_macro_f1",
    }
    if missing := required.difference(predictions.columns):
        raise ValueError(f"Missing locked prediction columns: {sorted(missing)}")
    rows: list[dict[str, float | int | str]] = []
    for index, (condition, group) in enumerate(predictions.groupby("condition", sort=False)):
        if group["participant_id"].duplicated().any():
            raise ValueError(f"Duplicate locked predictions for {condition}")
        y = group["label"].to_numpy(dtype=int)
        raw = group["raw_probability"].to_numpy(dtype=float)
        platt = group["platt_probability"].to_numpy(dtype=float)
        fixed_prediction = (raw >= 0.5).astype(int)
        nested_prediction = (
            raw >= group["threshold_max_macro_f1"].to_numpy(dtype=float)
        ).astype(int)
        auc_low, auc_high = stratified_auc_ci(
            y,
            raw,
            n_bootstrap=n_bootstrap,
            seed=seed + index,
        )
        raw_ece, raw_bins = quantile_ece(y, raw, n_bins=5)
        raw_intercept, raw_slope = calibration_slope_intercept(y, raw)
        platt_ece, platt_bins = quantile_ece(y, platt, n_bins=5)
        platt_intercept, platt_slope = calibration_slope_intercept(y, platt)
        rows.append(
            {
                "condition": condition,
                "n": len(group),
                "n_positive": int(y.sum()),
                "roc_auc": float(roc_auc_score(y, raw)),
                "auc_ci_low": auc_low,
                "auc_ci_high": auc_high,
                "pr_auc": float(average_precision_score(y, raw)),
                "raw_brier": float(brier_score_loss(y, raw)),
                "raw_ece_5bin": raw_ece,
                "raw_ece_effective_bins": raw_bins,
                "raw_calibration_intercept": raw_intercept,
                "raw_calibration_slope": raw_slope,
                "platt_brier": float(brier_score_loss(y, platt)),
                "platt_ece_5bin": platt_ece,
                "platt_ece_effective_bins": platt_bins,
                "platt_calibration_intercept": platt_intercept,
                "platt_calibration_slope": platt_slope,
                "macro_f1_at_0_5": float(f1_score(y, fixed_prediction, average="macro")),
                "sensitivity_at_0_5": float(recall_score(y, fixed_prediction)),
                "specificity_at_0_5": float(
                    recall_score(y, fixed_prediction, pos_label=0)
                ),
                "macro_f1_nested": float(f1_score(y, nested_prediction, average="macro")),
                "sensitivity_nested": float(recall_score(y, nested_prediction)),
                "specificity_nested": float(
                    recall_score(y, nested_prediction, pos_label=0)
                ),
                "nested_threshold_mean": float(
                    group["threshold_max_macro_f1"].mean()
                ),
                "nested_threshold_min": float(group["threshold_max_macro_f1"].min()),
                "nested_threshold_max": float(group["threshold_max_macro_f1"].max()),
            }
        )
    return pd.DataFrame(rows)


def load_locked_source_predictions(
    analysis_root: Path,
    *,
    repeat: int = LOCKED_REPEAT,
    conditions: Mapping[str, str] = SOURCE_CONDITIONS,
) -> pd.DataFrame:
    oof_root = analysis_root / "02_tfidf_main" / "oof_probs"
    rows: list[pd.DataFrame] = []
    keep = [
        "repeat",
        "fold",
        "participant_id",
        "label",
        "raw_probability",
        "platt_probability",
        "isotonic_probability",
        "threshold_max_macro_f1",
        "threshold_max_youden_j",
        "threshold_sensitivity_at_spec80",
        "train_ids_sha256",
        "test_ids_sha256",
    ]
    for condition, filename in conditions.items():
        frame = pd.read_csv(oof_root / filename, encoding="utf-8-sig")
        frame = frame.loc[frame["repeat"].eq(repeat), keep].copy()
        if len(frame) != frame["participant_id"].nunique():
            raise ValueError(f"{condition} does not have one locked OOF row per participant")
        frame.insert(0, "condition", condition)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def run_structural_locked_cv(
    features: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    n_bootstrap: int = 5000,
    seed: int = BASE_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    locked = select_locked_split(splits)
    fold, participant = run_repeated_cv(
        features,
        locked,
        model_specs=STRUCTURAL_MODEL_SPECS,
        base_seed=seed,
    )
    metrics, auc_ci = summarize_model_metrics(
        participant,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    return fold, participant, metrics, auc_ci


def run_token_matched_control(
    participant_text: pd.DataFrame,
    interviewer_text: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    n_draws: int = TOKEN_MATCH_DRAWS,
    n_bootstrap: int = 5000,
    seed: int = BASE_SEED,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    if n_draws <= 0:
        raise ValueError("n_draws must be positive")
    locked = select_locked_split(splits)
    prediction_rows: list[pd.DataFrame] = []
    draw_metric_rows: list[dict[str, float | int | str]] = []
    length_audit: pd.DataFrame | None = None
    for draw in range(1, n_draws + 1):
        matched, audit = build_token_matched_draw(
            participant_text,
            interviewer_text,
            draw=draw,
            seed=seed,
        )
        if length_audit is None:
            length_audit = audit
        _, participant_oof = run_repeated_cv(
            matched,
            locked,
            model_specs=TOKEN_MATCH_MODEL_SPECS,
            base_seed=seed + 10_000 * draw,
        )
        participant_oof.insert(0, "draw", draw)
        prediction_rows.append(participant_oof)
        y = participant_oof["label"].to_numpy(dtype=int)
        for model in TOKEN_MATCH_MODEL_SPECS:
            probability = participant_oof[f"{model}_prob"].to_numpy(dtype=float)
            draw_metric_rows.append(
                {
                    "draw": draw,
                    "condition": model,
                    "roc_auc": float(roc_auc_score(y, probability)),
                    "pr_auc": float(average_precision_score(y, probability)),
                    "brier": float(brier_score_loss(y, probability)),
                }
            )
    predictions = pd.concat(prediction_rows, ignore_index=True)
    averaged = (
        predictions.groupby(["participant_id", "label"], as_index=False)[
            ["participant_matched_prob", "interviewer_matched_prob"]
        ]
        .mean()
        .sort_values("participant_id")
        .reset_index(drop=True)
    )
    metrics, auc_ci = summarize_model_metrics(
        averaged,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    inference = paired_auc_inference(
        averaged["label"],
        averaged["interviewer_matched_prob"],
        averaged["participant_matched_prob"],
        n_bootstrap=n_bootstrap,
        n_permutations=10_000,
        seed=seed,
    )
    comparison = pd.DataFrame(
        [
            {
                "comparison": "interviewer_matched vs participant_matched",
                "model_a": "interviewer_matched",
                "model_b": "participant_matched",
                **inference,
            }
        ]
    )
    if length_audit is None:
        raise AssertionError("Token audit was not generated")
    return (
        predictions,
        pd.DataFrame(draw_metric_rows),
        averaged,
        metrics.merge(auc_ci, on="model", how="left", validate="one_to_one"),
        comparison,
        length_audit,
    )


def summarize_token_matched_draws(
    predictions: pd.DataFrame,
    draw_metrics: pd.DataFrame,
    *,
    n_bootstrap: int = 5000,
    seed: int = BASE_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_predictions = {
        "draw",
        "participant_id",
        "label",
        "participant_matched_prob",
        "interviewer_matched_prob",
    }
    required_metrics = {"draw", "condition", "roc_auc", "pr_auc", "brier"}
    if missing := required_predictions.difference(predictions.columns):
        raise ValueError(f"Missing token prediction columns: {sorted(missing)}")
    if missing := required_metrics.difference(draw_metrics.columns):
        raise ValueError(f"Missing token metric columns: {sorted(missing)}")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")

    draws = sorted(predictions["draw"].unique())
    participant_ids = sorted(predictions["participant_id"].unique())
    expected = len(draws) * len(participant_ids)
    if len(predictions) != expected:
        raise ValueError("Token predictions must contain every draw-participant combination")
    labels = (
        predictions[["participant_id", "label"]]
        .drop_duplicates()
        .set_index("participant_id")
        .loc[participant_ids, "label"]
        .to_numpy(dtype=int)
    )
    if predictions.groupby("participant_id")["label"].nunique().gt(1).any():
        raise ValueError("Participant labels vary across token draws")
    matrices = {
        condition: predictions.pivot(
            index="draw", columns="participant_id", values=f"{condition}_prob"
        )
        .loc[draws, participant_ids]
        .to_numpy(dtype=float)
        for condition in ("participant_matched", "interviewer_matched")
    }
    positive_indices = np.flatnonzero(labels == 1)
    negative_indices = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(seed)
    bootstrap_auc = {
        condition: np.empty(n_bootstrap, dtype=float) for condition in matrices
    }
    bootstrap_delta = np.empty(n_bootstrap, dtype=float)
    for bootstrap_index in range(n_bootstrap):
        draw_index = int(rng.integers(0, len(draws)))
        sampled_positive = rng.choice(
            positive_indices, size=len(positive_indices), replace=True
        )
        sampled_negative = rng.choice(
            negative_indices, size=len(negative_indices), replace=True
        )
        draw_aucs: dict[str, float] = {}
        for condition, matrix in matrices.items():
            positive = matrix[draw_index, sampled_positive]
            negative = matrix[draw_index, sampled_negative]
            difference = positive[:, None] - negative[None, :]
            auc = (
                np.count_nonzero(difference > 0)
                + 0.5 * np.count_nonzero(difference == 0)
            ) / difference.size
            bootstrap_auc[condition][bootstrap_index] = auc
            draw_aucs[condition] = float(auc)
        bootstrap_delta[bootstrap_index] = (
            draw_aucs["interviewer_matched"] - draw_aucs["participant_matched"]
        )

    rows: list[dict[str, float | int | str]] = []
    for condition in matrices:
        metrics = draw_metrics.loc[draw_metrics["condition"].eq(condition)]
        rows.append(
            {
                "condition": condition,
                "n_draws": len(draws),
                "mean_auc": float(metrics["roc_auc"].mean()),
                "sd_auc": float(metrics["roc_auc"].std(ddof=1)),
                "draw_auc_q025": float(metrics["roc_auc"].quantile(0.025)),
                "draw_auc_q975": float(metrics["roc_auc"].quantile(0.975)),
                "participant_random_draw_ci_low": float(
                    np.quantile(bootstrap_auc[condition], 0.025)
                ),
                "participant_random_draw_ci_high": float(
                    np.quantile(bootstrap_auc[condition], 0.975)
                ),
                "mean_pr_auc": float(metrics["pr_auc"].mean()),
                "mean_brier": float(metrics["brier"].mean()),
                "bootstrap_resamples": n_bootstrap,
                "bootstrap_draw_sampling": "one random matched dataset draw per participant bootstrap",
            }
        )
    participant_draw_summary = pd.DataFrame(rows)
    draw_delta = draw_metrics.pivot(
        index="draw", columns="condition", values="roc_auc"
    )
    draw_delta = draw_delta["interviewer_matched"] - draw_delta["participant_matched"]
    comparison = pd.DataFrame(
        [
            {
                "comparison": "interviewer_matched vs participant_matched",
                "mean_delta_auc": float(draw_delta.mean()),
                "sd_delta_auc": float(draw_delta.std(ddof=1)),
                "draw_delta_q025": float(draw_delta.quantile(0.025)),
                "draw_delta_q975": float(draw_delta.quantile(0.975)),
                "participant_random_draw_ci_low": float(
                    np.quantile(bootstrap_delta, 0.025)
                ),
                "participant_random_draw_ci_high": float(
                    np.quantile(bootstrap_delta, 0.975)
                ),
                "bootstrap_resamples": n_bootstrap,
                "interpretation_scope": "text_quantity_control_not_semantic_mechanism",
            }
        ]
    )
    return participant_draw_summary, comparison


def build_core_paired_comparisons(
    source_predictions: pd.DataFrame,
    joint_predictions: pd.DataFrame,
    *,
    n_bootstrap: int = 5000,
    n_permutations: int = 10_000,
    seed: int = BASE_SEED,
) -> pd.DataFrame:
    source_wide = source_predictions.pivot(
        index=["participant_id", "label"], columns="condition", values="raw_probability"
    ).reset_index()
    required_joint = {"participant_id", "label", "M3_prob", "M4_prob", "M5_prob"}
    if missing := required_joint.difference(joint_predictions.columns):
        raise ValueError(f"Missing joint prediction columns: {sorted(missing)}")
    merged = source_wide.merge(
        joint_predictions[list(required_joint)],
        on=["participant_id", "label"],
        validate="one_to_one",
    )
    specs = [
        (
            "full_transcript vs participant_speech",
            "full_transcript",
            "participant_speech",
            "source_sufficiency_not_independent_contribution",
        ),
        (
            "interviewer_speech vs participant_speech",
            "interviewer_speech",
            "participant_speech",
            "source_sufficiency_not_independent_contribution",
        ),
        (
            "participant_symptom_evidence vs participant_speech",
            "participant_symptom_evidence",
            "participant_speech",
            "derived_representation_vs_raw_source",
        ),
        (
            "M5 vs M3",
            "M5_prob",
            "M3_prob",
            "conditional_interviewer_increment",
        ),
        (
            "M4 vs M3",
            "M4_prob",
            "M3_prob",
            "conditional_evidence_representation_increment",
        ),
    ]
    rows: list[dict[str, float | int | str]] = []
    y = merged["label"].to_numpy(dtype=int)
    for index, (name, model_a, model_b, scope) in enumerate(specs):
        probability_a = merged[model_a].to_numpy(dtype=float)
        probability_b = merged[model_b].to_numpy(dtype=float)
        inference = paired_auc_inference(
            y,
            probability_a,
            probability_b,
            n_bootstrap=n_bootstrap,
            n_permutations=n_permutations,
            seed=seed + index,
        )
        rows.append(
            {
                "comparison": name,
                "model_a": model_a,
                "model_b": model_b,
                "interpretation_scope": scope,
                "auc_a": float(roc_auc_score(y, probability_a)),
                "auc_b": float(roc_auc_score(y, probability_b)),
                **inference,
            }
        )
    result = pd.DataFrame(rows)
    result["q_value"] = bh_fdr(result["p_value"].to_numpy(dtype=float))
    result["significance"] = result["q_value"].map(significance_marker)
    return result


def calibration_curve_data(
    predictions: pd.DataFrame,
    *,
    probability_column: str = "raw_probability",
    n_bins: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for condition, group in predictions.groupby("condition", sort=False):
        observed, predicted = calibration_curve(
            group["label"].to_numpy(dtype=int),
            group[probability_column].to_numpy(dtype=float),
            n_bins=n_bins,
            strategy="quantile",
        )
        for bin_index, (mean_predicted, fraction_positive) in enumerate(
            zip(predicted, observed), start=1
        ):
            rows.append(
                {
                    "condition": condition,
                    "probability": probability_column,
                    "bin": bin_index,
                    "mean_predicted_probability": float(mean_predicted),
                    "observed_positive_fraction": float(fraction_positive),
                }
            )
    return pd.DataFrame(rows)


def threshold_curve_data(
    predictions: pd.DataFrame,
    *,
    thresholds: Sequence[float] = tuple(np.linspace(0.05, 0.95, 19)),
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for condition, group in predictions.groupby("condition", sort=False):
        y = group["label"].to_numpy(dtype=int)
        probability = group["raw_probability"].to_numpy(dtype=float)
        for threshold in thresholds:
            predicted = (probability >= float(threshold)).astype(int)
            rows.append(
                {
                    "condition": condition,
                    "threshold": float(threshold),
                    "sensitivity": float(recall_score(y, predicted)),
                    "specificity": float(recall_score(y, predicted, pos_label=0)),
                    "macro_f1": float(f1_score(y, predicted, average="macro")),
                }
            )
    return pd.DataFrame(rows)


def _all_occurrences(text: str, quote: str) -> list[int]:
    if not quote:
        return []
    positions: list[int] = []
    start = 0
    while True:
        index = text.find(quote, start)
        if index < 0:
            break
        positions.append(index)
        start = index + max(1, len(quote))
    return positions


def audit_c5_traceability(spans: pd.DataFrame, participant_text: pd.DataFrame) -> pd.DataFrame:
    span_required = {
        "participant_id",
        "source_order",
        "domain",
        "polarity",
        "exact_quote",
        "review_status",
        "match_status",
    }
    source_required = {"participant_id", "text", "text_sha256"}
    if missing := span_required.difference(spans.columns):
        raise ValueError(f"Missing span columns: {sorted(missing)}")
    if missing := source_required.difference(participant_text.columns):
        raise ValueError(f"Missing participant source columns: {sorted(missing)}")
    if participant_text["participant_id"].duplicated().any():
        raise ValueError("Participant source must have unique IDs")

    source = participant_text.set_index("participant_id")
    rows: list[dict[str, float | int | str]] = []
    for span in spans.sort_values(["participant_id", "source_order"]).itertuples(index=False):
        participant_id = int(span.participant_id)
        if participant_id not in source.index:
            raise ValueError(f"Missing participant source for {participant_id}")
        source_row = source.loc[participant_id]
        normalized_source = " ".join(str(source_row["text"]).split())
        normalized_quote = " ".join(str(span.exact_quote).split())
        positions = _all_occurrences(normalized_source, normalized_quote)
        if len(positions) == 1:
            location_status = "unique_exact_match"
        elif len(positions) > 1:
            location_status = "multiple_exact_matches"
        else:
            location_status = "not_found"
        start = positions[0] if positions else np.nan
        end = positions[0] + len(normalized_quote) if positions else np.nan
        rows.append(
            {
                "participant_id": participant_id,
                "source_order": int(span.source_order),
                "domain": str(span.domain),
                "polarity": str(span.polarity),
                "review_status": str(span.review_status),
                "match_status": str(span.match_status),
                "manual_correction": int("corrected" in str(span.review_status)),
                "source_text_sha256": str(source_row["text_sha256"]),
                "quote_sha256": hashlib.sha256(normalized_quote.encode("utf-8")).hexdigest(),
                "normalized_char_start": start,
                "normalized_char_end": end,
                "exact_match_count": len(positions),
                "location_status": location_status,
                "traceable_to_participant_source": int(bool(positions)),
            }
        )
    result = pd.DataFrame(rows)
    if "exact_quote" in result.columns:
        raise AssertionError("Public traceability audit must not expose quote text")
    return result
