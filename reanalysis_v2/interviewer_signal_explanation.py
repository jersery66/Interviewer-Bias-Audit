"""Leakage-resistant explanation analysis for the interviewer-side signal.

The module is intentionally separate from the frozen conditional-increment and
post-audit identification runners.  It exposes small, testable contracts for
feature auditing, strict cross-fitting, Fake-D controls, pairing dependence,
and result hashing.  The formal orchestration is at the bottom of this file and
does not alter any upstream frozen artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .embedding_incremental import (
    DEFAULT_C_GRID,
    _nested_source_logits,
    load_frozen_embedding_pair,
    make_inner_splits,
)
from .io import atomic_write_csv, atomic_write_json, sha256_file
from .modeling import make_tfidf_classifier
from .splits import split_indices


EXPECTED_N = 142
EXPECTED_POSITIVE = 43
EXPECTED_NEGATIVE = 99
BASE_SEED = 20260713
INNER_FOLDS = 3
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
LABEL_C_GRID = (0.01, 0.1, 1.0, 10.0)
TFIDF_SOURCE_C = 1.0
FAKE_D_DRAWS = 100
PAIRING_DRAWS = 100
N_BOOTSTRAP = 5000
N_PERMUTATIONS = 10_000
NEAR_ZERO_DOMINANT_FRACTION = 0.95
RECOVERABILITY_HIGH_R2_THRESHOLD = 0.50
RECOVERABILITY_NEAR_ZERO_R2_THRESHOLD = 0.05

D_P_FEATURES = (
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
)

Q_FEATURES = (
    "length_only__participant_word_count",
    "length_only__interviewer_word_count",
    "length_only__total_word_count",
    "length_only__interview_duration_sec",
    "interaction_structure__participant_turn_count",
    "interaction_structure__interviewer_turn_count",
    "interaction_structure__total_turn_count",
    "interaction_structure__participant_mean_words_per_turn",
    "interaction_structure__interviewer_mean_words_per_turn",
)

R_BASE_FEATURES = (
    "protocol_structure__interviewer_question_count",
    "protocol_structure__clinical_question_count",
    "protocol_structure__unique_protocol_prompt_count",
    "protocol_structure__clinical_prompt_coverage_count",
    "protocol_structure__non_explicit_interviewer_turn_count",
)

PAIRING_MATCH_FEATURES = (
    "interviewer_word_count",
    "interviewer_turn_count",
    "clinical_question_count",
    "clinical_prompt_coverage_count",
    "non_explicit_interviewer_turn_count",
    "d_p_domain_breadth",
    "d_p_total_evidence_count",
)

ALL_BLOCK_SUBSETS = (
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

LABEL_CONDITIONED_SUBSETS = (
    "P_within",
    "Q",
    "R",
    "D",
    "P_within+Q+R+D",
)

BLOCK_ORDER = ("P", "Q", "R", "D")
_FORBIDDEN_PUBLIC_TOKENS = (
    "spoken_text",
    "verified_text",
    "exact_quote",
    "original_model_quote",
)


@dataclass(frozen=True)
class SourceScoreResult:
    train_logits: np.ndarray
    test_logits: np.ndarray
    train_probabilities: np.ndarray
    test_probabilities: np.ndarray
    selected_c: float
    tuning: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class ExplanationFoldResult:
    train_prediction: np.ndarray
    test_prediction: np.ndarray
    selected_alpha: float
    tuning: tuple[dict[str, object], ...]


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _as_numeric(frame: pd.DataFrame, columns: Sequence[str], name: str) -> pd.DataFrame:
    _require_columns(frame, columns, name)
    result = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    if result.isna().any().any() or not np.isfinite(result.to_numpy(dtype=float)).all():
        bad = result.columns[result.isna().any()].tolist()
        raise ValueError(f"{name} contains missing or non-finite numeric values: {bad}")
    return result


def block_subset_columns(
    subset: str, block_columns: Mapping[str, Sequence[str]]
) -> list[str]:
    """Expand a locked subset name into columns without changing block order."""

    if subset not in ALL_BLOCK_SUBSETS:
        raise ValueError(f"unknown block subset: {subset}")
    if subset == "null":
        return []
    blocks = subset.split("+")
    if blocks != sorted(blocks, key=BLOCK_ORDER.index):
        raise ValueError(f"block subset is not in locked order: {subset}")
    columns: list[str] = []
    for block in blocks:
        if block not in BLOCK_ORDER:
            raise ValueError(f"unknown block: {block}")
        columns.extend(str(column) for column in block_columns.get(block, ()))
    if len(columns) != len(set(columns)):
        raise ValueError(f"duplicate feature columns in subset {subset}")
    return columns


def audit_numeric_features(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    near_zero_dominant_fraction: float = NEAR_ZERO_DOMINANT_FRACTION,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Audit numeric columns and return an explicit model-eligibility table."""

    if not 0.5 < float(near_zero_dominant_fraction) <= 1.0:
        raise ValueError("near-zero dominant fraction must be in (0.5, 1]")
    rows: list[dict[str, object]] = []
    duplicate_groups: list[list[str]] = []
    numeric: dict[str, pd.Series] = {}
    for feature in feature_columns:
        if feature not in frame.columns:
            raise ValueError(f"feature is missing from audit input: {feature}")
        series = pd.to_numeric(frame[feature], errors="coerce")
        numeric[feature] = series
        finite = series.notna() & np.isfinite(series.to_numpy(dtype=float, na_value=np.nan))
        values = series.loc[finite].to_numpy(dtype=float)
        unique_count = int(pd.Series(values).nunique(dropna=True))
        mean = float(values.mean()) if len(values) else float("nan")
        sd = float(values.std(ddof=0)) if len(values) else float("nan")
        minimum = float(values.min()) if len(values) else float("nan")
        maximum = float(values.max()) if len(values) else float("nan")
        counts = pd.Series(values).value_counts(dropna=False)
        dominant_fraction = float(counts.iloc[0] / len(values)) if len(values) else float("nan")
        zero_variance = bool(len(values) > 0 and (unique_count <= 1 or sd <= 1e-12))
        near_zero = bool(
            zero_variance
            or (
                len(values) > 0
                and unique_count <= 2
                and dominant_fraction >= near_zero_dominant_fraction
            )
        )
        rows.append(
            {
                "feature": feature,
                "dtype": str(frame[feature].dtype),
                "missing": int(series.isna().sum()),
                "finite": int(finite.sum()),
                "unique_count": unique_count,
                "mean": mean,
                "sd": sd,
                "min": minimum,
                "max": maximum,
                "dominant_fraction": dominant_fraction,
                "zero_variance": zero_variance,
                "near_zero_variance": near_zero,
                "model_eligible": bool(not near_zero and finite.all()),
            }
        )
    for left_index, left in enumerate(feature_columns):
        for right in feature_columns[left_index + 1 :]:
            if numeric[left].equals(numeric[right]):
                group = next((g for g in duplicate_groups if left in g), None)
                if group is None:
                    duplicate_groups.append([left, right])
                else:
                    group.append(right)
    duplicate_lookup = {
        feature: index + 1
        for index, group in enumerate(duplicate_groups)
        for feature in group
    }
    audit = pd.DataFrame(rows)
    audit["exact_duplicate_group"] = audit["feature"].map(duplicate_lookup).fillna("")
    summary = {
        "near_zero_dominant_fraction_threshold": float(near_zero_dominant_fraction),
        "dropped_features": audit.loc[~audit["model_eligible"], "feature"].tolist(),
        "exact_duplicate_groups": duplicate_groups,
    }
    return audit, summary


def validate_dp_frame(
    frame: pd.DataFrame,
    *,
    expected_ids: Sequence[object] | None = None,
) -> pd.DataFrame:
    """Validate the frozen D-P schema and return only its locked columns."""

    required = ["participant_id", *D_P_FEATURES]
    _require_columns(frame, required, "D-P input")
    extras = [column for column in frame.columns if column not in required]
    if any("d_all" in str(column).lower() or "dall" in str(column).lower() for column in extras):
        raise ValueError("unknown D-All columns are not allowed in the D-P input")
    result = frame.loc[:, required].copy()
    if result["participant_id"].duplicated().any():
        raise ValueError("D-P participant IDs must be unique")
    if expected_ids is not None and set(result["participant_id"]) != set(expected_ids):
        raise ValueError("D-P participant IDs do not match the expected cohort")
    result[list(D_P_FEATURES)] = _as_numeric(result, D_P_FEATURES, "D-P input")
    return result.sort_values("participant_id", kind="stable").reset_index(drop=True)


def _prompt_feature_slug(raw_prompt_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", raw_prompt_id).strip("_").lower()
    slug = slug or "empty"
    digest = hashlib.sha1(raw_prompt_id.encode("utf-8")).hexdigest()[:8]
    return f"{slug}_{digest}"


def derive_interviewer_path_features(
    turns: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Derive locked protocol-path features from the frozen turn table only."""

    required = (
        "participant_id",
        "turn_index",
        "speaker_norm",
        "is_prompt_annotated",
        "is_explicit_clinical_prompt",
        "prompt_id",
    )
    _require_columns(turns, required, "frozen turn table")
    if turns.empty:
        raise ValueError("frozen turn table is empty")

    work = turns.loc[:, list(required)].copy()
    work["_turn_index"] = pd.to_numeric(work["turn_index"], errors="raise")
    work["_annotated"] = pd.to_numeric(work["is_prompt_annotated"], errors="coerce").fillna(0).astype(int).eq(1)
    work["_clinical"] = pd.to_numeric(work["is_explicit_clinical_prompt"], errors="coerce").fillna(0).astype(int).eq(1)
    work["_interviewer"] = work["speaker_norm"].astype(str).str.lower().eq("interviewer")
    annotated_ids = sorted(
        {
            str(value)
            for value in work.loc[work["_interviewer"] & work["_annotated"], "prompt_id"]
            if pd.notna(value) and str(value).strip()
        }
    )
    prompt_slugs = {raw: _prompt_feature_slug(raw) for raw in annotated_ids}
    dictionary: list[dict[str, str]] = []
    for raw, slug in prompt_slugs.items():
        dictionary.extend(
            [
                {"raw_prompt_id": raw, "feature": f"protocol_prompt__{slug}__count"},
                {"raw_prompt_id": raw, "feature": f"protocol_prompt__{slug}__presence"},
            ]
        )

    rows: list[dict[str, float | int]] = []
    for participant_id, group in work.groupby("participant_id", sort=True):
        ordered = group.sort_values(["_turn_index"], kind="stable").reset_index(drop=True)
        n_turns = len(ordered)
        positions = np.zeros(n_turns, dtype=float) if n_turns <= 1 else np.arange(n_turns, dtype=float) / (n_turns - 1)
        interviewer = ordered["_interviewer"].to_numpy(dtype=bool)
        clinical = ordered["_clinical"].to_numpy(dtype=bool)
        clinical_positions = positions[interviewer & clinical]
        interviewer_rows = ordered.loc[interviewer].copy()
        interviewer_clinical = clinical[interviewer]
        prompt_rows = interviewer_rows.loc[interviewer_rows["_annotated"]]
        prompt_ids = [str(value) for value in prompt_rows["prompt_id"] if pd.notna(value) and str(value).strip()]
        row: dict[str, float | int] = {
            "participant_id": participant_id,
            "first_clinical_prompt_position_normalized": float(clinical_positions[0]) if len(clinical_positions) else 0.0,
            "last_clinical_prompt_position_normalized": float(clinical_positions[-1]) if len(clinical_positions) else 0.0,
            "clinical_prompt_density": float(np.sum(interviewer & clinical) / max(int(interviewer.sum()), 1)),
            "clinical_prompt_count_early_third": int(np.sum((clinical_positions >= 0) & (clinical_positions < 1 / 3))),
            "clinical_prompt_count_middle_third": int(np.sum((clinical_positions >= 1 / 3) & (clinical_positions < 2 / 3))),
            "clinical_prompt_count_late_third": int(np.sum(clinical_positions >= 2 / 3)),
            "clinical_nonclinical_transition_count": int(np.sum(interviewer_clinical[1:] != interviewer_clinical[:-1])) if len(interviewer_clinical) > 1 else 0,
            "max_consecutive_clinical_prompt_run": 0,
            "protocol_path_depth": int(len(set(prompt_ids))),
        }
        current_run = 0
        for value in interviewer_clinical:
            current_run = current_run + 1 if bool(value) else 0
            row["max_consecutive_clinical_prompt_run"] = max(int(row["max_consecutive_clinical_prompt_run"]), current_run)
        for raw, slug in prompt_slugs.items():
            count = prompt_ids.count(raw)
            row[f"protocol_prompt__{slug}__count"] = int(count)
            row[f"protocol_prompt__{slug}__presence"] = int(count > 0)
        rows.append(row)
    result = pd.DataFrame(rows).sort_values("participant_id", kind="stable").reset_index(drop=True)
    public_output_columns_are_safe(result.columns)
    return result, dictionary


def _make_derangement(n: int, *, seed: int) -> np.ndarray:
    if int(n) < 2:
        raise ValueError("a derangement requires at least two rows")
    rng = np.random.default_rng(int(seed))
    base = rng.permutation(int(n))
    for shift in range(int(n)):
        candidate = np.roll(base, shift)
        if not np.any(candidate == np.arange(int(n))):
            return candidate.astype(int)
    return np.roll(np.arange(int(n)), 1).astype(int)


def apply_fake_domain_permutation(
    *,
    participant_ids: Sequence[object],
    domains: np.ndarray,
    train_idx: Sequence[int],
    test_idx: Sequence[int],
    draw: int,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Move complete D-P rows within each outer partition and record donors."""

    ids = np.asarray(participant_ids)
    matrix = np.asarray(domains, dtype=float)
    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    if matrix.shape[0] != len(ids) or matrix.ndim != 2:
        raise ValueError("domain matrix and participant IDs do not agree")
    if set(train_idx).intersection(test_idx):
        raise ValueError("Fake-D train/test indices overlap")
    result = matrix.copy()
    rows: list[dict[str, object]] = []
    for partition_index, (partition, indices) in enumerate((("train", train_idx), ("test", test_idx))):
        if len(indices) < 2:
            raise ValueError(f"Fake-D {partition} partition needs at least two rows")
        local_permutation = _make_derangement(len(indices), seed=int(seed) + int(draw) * 1000 + partition_index)
        donor_indices = indices[local_permutation]
        for recipient_index, donor_index in zip(indices, donor_indices):
            rows.append(
                {
                    "draw": int(draw),
                    "partition": partition,
                    "recipient_index": int(recipient_index),
                    "donor_index": int(donor_index),
                    "recipient_id": ids[recipient_index],
                    "donor_id": ids[donor_index],
                    "self_match": bool(recipient_index == donor_index),
                }
            )
        result[indices] = matrix[donor_indices]
    ledger = pd.DataFrame(rows)
    if ledger["self_match"].any():
        raise AssertionError("Fake-D produced a self assignment")
    train_donors = set(ledger.loc[ledger["partition"].eq("train"), "donor_id"])
    test_donors = set(ledger.loc[ledger["partition"].eq("test"), "donor_id"])
    if train_donors.intersection(test_donors):
        raise AssertionError("Fake-D train and test donor IDs overlap")
    return result, ledger.sort_values(["partition", "recipient_index"], kind="stable").reset_index(drop=True)


def make_pairing_assignment(
    *,
    recipient: pd.DataFrame,
    donor: pd.DataFrame,
    fit_reference: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """Create the one deterministic label-blind Hungarian assignment.

    ``seed`` is retained for call-site compatibility but deliberately does not
    perturb the optimum. Random mismatch references use
    :func:`make_random_derangement_assignment` instead.
    """

    for frame, name in ((recipient, "recipient"), (donor, "donor"), (fit_reference, "fit reference")):
        _require_columns(frame, ["participant_id", *PAIRING_MATCH_FEATURES], name)
        _as_numeric(frame, PAIRING_MATCH_FEATURES, name)
    if len(recipient) != len(donor) or len(recipient) < 2:
        raise ValueError("recipient and donor partitions must have equal size >= 2")
    if recipient["participant_id"].duplicated().any() or donor["participant_id"].duplicated().any():
        raise ValueError("pairing IDs must be unique")
    _ = seed
    recipient_z, donor_z, _ = _pairing_z_values(recipient, donor, fit_reference)
    cost = np.square(recipient_z[:, None, :] - donor_z[None, :, :]).sum(axis=2)
    recipient_ids = recipient["participant_id"].to_numpy()
    donor_ids = donor["participant_id"].to_numpy()
    for row_index, recipient_id in enumerate(recipient_ids):
        for donor_index, donor_id in enumerate(donor_ids):
            if recipient_id == donor_id:
                cost[row_index, donor_index] = np.inf
    if not np.isfinite(cost).any(axis=1).all():
        raise ValueError("pairing assignment has no feasible non-self donor")
    row_index, donor_index = linear_sum_assignment(cost)
    if len(row_index) != len(recipient):
        raise ValueError("Hungarian assignment did not cover every recipient")
    pairwise_distance = np.sqrt(cost[row_index, donor_index])
    result = pd.DataFrame(
        {
            "recipient_index": row_index.astype(int),
            "donor_index": donor_index.astype(int),
            "recipient_id": recipient_ids[row_index],
            "donor_id": donor_ids[donor_index],
            "distance": pairwise_distance,
            "standardized_euclidean_distance": pairwise_distance,
            "assignment_type": "optimal_matched",
        }
    )
    result["self_match"] = result["recipient_id"].eq(result["donor_id"])
    result["donor_reuse_count"] = result.groupby("donor_id")["donor_id"].transform("size").astype(int)
    result["one_to_one"] = bool(result["donor_id"].nunique() == len(result))
    if result["self_match"].any() or not result["one_to_one"].all():
        raise AssertionError("pairing assignment violated donor contract")
    return result.sort_values("recipient_id", kind="stable").reset_index(drop=True)


def _pairing_z_values(
    recipient: pd.DataFrame,
    donor: pd.DataFrame,
    fit_reference: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference_values = fit_reference[list(PAIRING_MATCH_FEATURES)].to_numpy(dtype=float)
    recipient_values = recipient[list(PAIRING_MATCH_FEATURES)].to_numpy(dtype=float)
    donor_values = donor[list(PAIRING_MATCH_FEATURES)].to_numpy(dtype=float)
    mean = reference_values.mean(axis=0)
    scale = reference_values.std(axis=0, ddof=0)
    scale[scale <= 1e-12] = 1.0
    return (recipient_values - mean) / scale, (donor_values - mean) / scale, scale


def make_random_derangement_assignment(
    *,
    recipient: pd.DataFrame,
    donor: pd.DataFrame,
    fit_reference: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """Create one fully random label-free non-self donor assignment."""

    for frame, name in ((recipient, "recipient"), (donor, "donor"), (fit_reference, "fit reference")):
        _require_columns(frame, ["participant_id", *PAIRING_MATCH_FEATURES], name)
        _as_numeric(frame, PAIRING_MATCH_FEATURES, name)
    if len(recipient) != len(donor) or len(recipient) < 2:
        raise ValueError("recipient and donor partitions must have equal size >= 2")
    if recipient["participant_id"].duplicated().any() or donor["participant_id"].duplicated().any():
        raise ValueError("pairing IDs must be unique")
    recipient_z, donor_z, _ = _pairing_z_values(recipient, donor, fit_reference)
    donor_order = _make_derangement(len(recipient), seed=int(seed))
    recipient_ids = recipient["participant_id"].to_numpy()
    donor_ids = donor["participant_id"].to_numpy()
    for recipient_id, donor_id in zip(recipient_ids, donor_ids[donor_order]):
        if recipient_id == donor_id:
            raise AssertionError("random mismatch produced self assignment")
    pairwise_difference = recipient_z - donor_z[donor_order]
    pairwise_distance = np.sqrt(np.square(pairwise_difference).sum(axis=1))
    result = pd.DataFrame(
        {
            "recipient_index": np.arange(len(recipient), dtype=int),
            "donor_index": donor_order.astype(int),
            "recipient_id": recipient_ids,
            "donor_id": donor_ids[donor_order],
            "distance": pairwise_distance,
            "standardized_euclidean_distance": pairwise_distance,
            "assignment_type": "random_mismatch",
        }
    )
    result["self_match"] = result["recipient_id"].eq(result["donor_id"])
    result["donor_reuse_count"] = result.groupby("donor_id")["donor_id"].transform("size").astype(int)
    result["one_to_one"] = bool(result["donor_id"].nunique() == len(result))
    if result["self_match"].any() or not result["one_to_one"].all():
        raise AssertionError("random mismatch violated donor contract")
    return result


def pairing_quality_summary(
    *,
    recipient: pd.DataFrame,
    donor: pd.DataFrame,
    assignment: pd.DataFrame,
    fit_reference: pd.DataFrame,
    random_distance_reference: Sequence[float] | None = None,
    assignment_type: str = "optimal_matched",
    max_mean_abs_paired_difference: float = 0.50,
    random_quantile: float = 0.05,
) -> dict[str, object]:
    """Measure pairwise matching quality and apply the pre-locked gate."""

    _require_columns(assignment, ["recipient_id", "donor_id"], "pairing assignment")
    recipient_lookup = recipient.set_index("participant_id")
    donor_lookup = donor.set_index("participant_id")
    if set(assignment["recipient_id"]) != set(recipient["participant_id"]):
        raise ValueError("pairing assignment does not cover all recipients")
    if set(assignment["donor_id"]) != set(donor["participant_id"]):
        raise ValueError("pairing assignment does not cover all donors")
    ordered_recipient = recipient_lookup.loc[assignment["recipient_id"].tolist(), list(PAIRING_MATCH_FEATURES)]
    ordered_donor = donor_lookup.loc[assignment["donor_id"].tolist(), list(PAIRING_MATCH_FEATURES)]
    _, _, scale = _pairing_z_values(recipient, donor, fit_reference)
    signed_standardized_difference = (
        ordered_recipient.to_numpy(dtype=float) - ordered_donor.to_numpy(dtype=float)
    ) / scale
    absolute_standardized_difference = np.abs(signed_standardized_difference)
    distances = np.sqrt(np.square(signed_standardized_difference).sum(axis=1))
    random_reference = (
        np.asarray(random_distance_reference, dtype=float)
        if random_distance_reference is not None
        else np.asarray([], dtype=float)
    )
    random_reference = random_reference[np.isfinite(random_reference)]
    random_q05 = float(np.quantile(random_reference, random_quantile)) if len(random_reference) else float("nan")
    mean_distance = float(np.mean(distances))
    random_position = (
        float((np.count_nonzero(random_reference <= mean_distance) + 1) / (len(random_reference) + 1))
        if len(random_reference)
        else float("nan")
    )
    feature_means = absolute_standardized_difference.mean(axis=0)
    feature_medians = np.median(absolute_standardized_difference, axis=0)
    feature_q95 = np.quantile(absolute_standardized_difference, 0.95, axis=0)
    feature_pass_count = int(np.count_nonzero(feature_means < float(max_mean_abs_paired_difference)))
    if "self_match" in assignment.columns:
        no_self = not assignment["self_match"].astype(bool).any()
    else:
        no_self = not assignment["recipient_id"].eq(assignment["donor_id"]).any()
    if "one_to_one" in assignment.columns:
        one_to_one = bool(assignment["one_to_one"].astype(bool).all())
    else:
        one_to_one = bool(assignment["donor_id"].nunique() == len(assignment))
    optimal_distance_pass = bool(
        assignment_type != "optimal_matched"
        or (len(random_reference) > 0 and mean_distance < random_q05)
    )
    matching_adequate = bool(no_self and one_to_one and optimal_distance_pass and feature_pass_count >= 5)
    return {
        "assignment_type": assignment_type,
        "n_pairs": int(len(assignment)),
        "no_self": bool(no_self),
        "one_to_one": bool(one_to_one),
        "mean_distance": mean_distance,
        "median_distance": float(np.median(distances)),
        "q95_distance": float(np.quantile(distances, 0.95)),
        "max_distance": float(np.max(distances)),
        "total_distance": float(np.sum(distances)),
        "total_cost": float(np.sum(np.square(distances))),
        "max_abs_smd": float(np.max(np.abs(signed_standardized_difference.mean(axis=0)))),
        "random_distance_q05": random_q05,
        "optimal_distance_random_percentile": random_position,
        "n_features_mean_abs_paired_diff_lt_0_50": feature_pass_count,
        "matching_adequate": matching_adequate,
        "feature_mean_abs_standardized_paired_difference": {
            feature: float(feature_means[index]) for index, feature in enumerate(PAIRING_MATCH_FEATURES)
        },
        "feature_median_abs_standardized_paired_difference": {
            feature: float(feature_medians[index]) for index, feature in enumerate(PAIRING_MATCH_FEATURES)
        },
        "feature_q95_abs_standardized_paired_difference": {
            feature: float(feature_q95[index]) for index, feature in enumerate(PAIRING_MATCH_FEATURES)
        },
        "feature_mean_signed_standardized_paired_difference": {
            feature: float(signed_standardized_difference[:, index].mean())
            for index, feature in enumerate(PAIRING_MATCH_FEATURES)
        },
    }


def nested_dense_source_logits(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    inner_folds: int,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Expose the existing nested dense source-score contract for this module."""

    train, test, tuning = _nested_source_logits(
        np.asarray(X_train, dtype=float),
        np.asarray(y_train, dtype=int),
        np.asarray(X_test, dtype=float),
        inner_folds=int(inner_folds),
        c_grid=tuple(float(value) for value in c_grid),
        seed=int(seed),
    )
    return train, test, tuning


def fit_ridge_pipeline(*, alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha), fit_intercept=True)),
        ]
    )


def compute_shapley_values(
    subset_values: Mapping[str, float],
    *,
    blocks: Sequence[str] = BLOCK_ORDER,
) -> dict[str, float]:
    """Compute the exact average marginal value over every block ordering."""

    blocks = tuple(str(block) for block in blocks)
    if set(blocks) != set(BLOCK_ORDER) or len(blocks) != len(set(blocks)):
        raise ValueError("Shapley blocks must be exactly P, Q, R and D")
    missing = set(ALL_BLOCK_SUBSETS).difference(subset_values)
    if missing:
        raise ValueError(f"missing subset values: {sorted(missing)}")
    contributions = {block: [] for block in blocks}
    for order in itertools.permutations(blocks):
        current: list[str] = []
        previous = float(subset_values["null"])
        for block in order:
            current.append(block)
            subset = "+".join(sorted(current, key=BLOCK_ORDER.index))
            value = float(subset_values[subset])
            contributions[block].append(value - previous)
            previous = value
    return {block: float(np.mean(values)) for block, values in contributions.items()}


def joint_bootstrap_indices(n: int, *, n_bootstrap: int, seed: int) -> np.ndarray:
    if int(n) <= 0 or int(n_bootstrap) <= 0:
        raise ValueError("bootstrap dimensions must be positive")
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, int(n), size=(int(n_bootstrap), int(n)), dtype=np.int64)


def paired_prediction_swap_permutation(
    target: Sequence[float],
    null_prediction: Sequence[float],
    full_prediction: Sequence[float],
    *,
    n_permutations: int,
    seed: int,
) -> dict[str, float | int]:
    """Test paired null/full prediction improvement by within-participant swaps."""

    if int(n_permutations) <= 0:
        raise ValueError("n_permutations must be positive")
    target = np.asarray(target, dtype=float)
    null_prediction = np.asarray(null_prediction, dtype=float)
    full_prediction = np.asarray(full_prediction, dtype=float)
    if not (len(target) == len(null_prediction) == len(full_prediction)) or len(target) == 0:
        raise ValueError("target, null and full predictions must have equal non-zero length")
    observed = float(
        mean_squared_error(target, null_prediction)
        - mean_squared_error(target, full_prediction)
    )
    rng = np.random.default_rng(int(seed))
    null_as_null = np.empty(len(null_prediction), dtype=float)
    full_as_full = np.empty(len(full_prediction), dtype=float)
    statistics = np.empty(int(n_permutations), dtype=float)
    for index in range(int(n_permutations)):
        swap = rng.integers(0, 2, size=len(null_prediction), dtype=np.int8).astype(bool)
        null_as_null[:] = np.where(swap, full_prediction, null_prediction)
        full_as_full[:] = np.where(swap, null_prediction, full_prediction)
        statistics[index] = float(
            mean_squared_error(target, null_as_null)
            - mean_squared_error(target, full_as_full)
        )
    extreme = int(np.count_nonzero(np.abs(statistics) >= abs(observed) - 1e-15))
    return {
        "observed_delta_mse": observed,
        "p_value": float((extreme + 1) / (int(n_permutations) + 1)),
        "n_permutations": int(n_permutations),
    }


def public_output_columns_are_safe(columns: Sequence[object]) -> None:
    """Reject columns that would expose interview source text or quotes."""

    unsafe: list[str] = []
    for value in columns:
        column = str(value).lower()
        if column == "text" or column.endswith("_text") or any(token in column for token in _FORBIDDEN_PUBLIC_TOKENS):
            unsafe.append(str(value))
    if unsafe:
        raise ValueError(f"public output contains source text columns: {unsafe}")


def _manifest_hashes(run_dir: Path) -> dict[str, str]:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes = manifest.get("output_file_hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError(f"run manifest has no output_file_hashes: {manifest_path}")
    result = {str(name): str(value) for name, value in hashes.items()}
    for name, expected in result.items():
        path = run_dir / name
        if not path.exists():
            raise FileNotFoundError(f"manifest output is missing: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(
                f"manifest output hash mismatch for {path.name}: "
                f"expected={expected}, observed={observed}"
            )
    return result


def verify_explanation_reruns(
    output_base: str | Path,
    *,
    run_1: str = "run_1",
    run_2: str = "run_2",
    promote_final: bool = False,
) -> dict[str, object]:
    """Verify two immutable runs and optionally copy only an exact match."""

    output_base = Path(output_base)
    first_dir = output_base / run_1
    second_dir = output_base / run_2
    mismatches: list[str] = []
    try:
        first_hashes = _manifest_hashes(first_dir)
        second_hashes = _manifest_hashes(second_dir)
        first_manifest = json.loads((first_dir / "run_manifest.json").read_text(encoding="utf-8"))
        second_manifest = json.loads((second_dir / "run_manifest.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        record = {"status": "failed", "mismatches": [str(exc)]}
        atomic_write_json(record, output_base / "rerun_verification.json")
        return record
    for name in sorted(set(first_hashes) | set(second_hashes)):
        if first_hashes.get(name) != second_hashes.get(name):
            mismatches.append(f"output:{name}")
    for field in ("code_commit", "plan_sha256", "analysis"):
        if first_manifest.get(field) != second_manifest.get(field):
            mismatches.append(field)
    status = "verified" if not mismatches else "failed"
    record: dict[str, object] = {
        "status": status,
        "run_1": run_1,
        "run_2": run_2,
        "mismatches": mismatches,
        "run_1_output_file_hashes": first_hashes,
        "run_2_output_file_hashes": second_hashes,
    }
    if status == "verified" and promote_final:
        final_dir = output_base / "final"
        if final_dir.exists():
            raise FileExistsError(f"final output already exists: {final_dir}")
        final_dir.mkdir(parents=True)
        for path in second_dir.iterdir():
            if path.is_file():
                shutil.copy2(path, final_dir / path.name)
        record["promoted_final"] = str(final_dir)
    atomic_write_json(record, output_base / "rerun_verification.json")
    return record


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def _selected_outer_c(tuning: Sequence[Mapping[str, object]], default: float) -> float:
    selected = [
        float(row["C"])
        for row in tuning
        if row.get("layer") == "source_outer_full" and bool(row.get("selected"))
    ]
    return selected[-1] if selected else float(default)


def nested_text_source_logits(
    train_texts: Sequence[str],
    y_train: np.ndarray,
    test_texts: Sequence[str],
    *,
    inner_folds: int,
    seed: int,
    c: float = TFIDF_SOURCE_C,
) -> SourceScoreResult:
    """Generate strict inner-CF training and outer-test TF-IDF logits."""

    train_texts = np.asarray([str(value or "") for value in train_texts], dtype=object)
    test_texts = np.asarray([str(value or "") for value in test_texts], dtype=object)
    y_train = np.asarray(y_train, dtype=int)
    if len(train_texts) != len(y_train):
        raise ValueError("source training texts and labels must have the same length")
    inner_splits = make_inner_splits(y_train, n_splits=int(inner_folds), seed=int(seed))
    train_logits = np.full(len(y_train), np.nan, dtype=float)
    tuning: list[dict[str, object]] = []
    for inner_fold, (meta_train, meta_validation) in enumerate(inner_splits, start=1):
        estimator = make_tfidf_classifier(c=float(c), seed=int(seed) + inner_fold)
        estimator.fit(train_texts[meta_train].tolist(), y_train[meta_train])
        train_logits[meta_validation] = estimator.decision_function(
            train_texts[meta_validation].tolist()
        )
        tuning.append(
            {
                "layer": "source_inner_cf",
                "inner_fold": int(inner_fold),
                "C": float(c),
                "selected": True,
            }
        )
    if not np.isfinite(train_logits).all():
        raise ValueError("TF-IDF source cross-fitted training logits are incomplete")
    estimator = make_tfidf_classifier(c=float(c), seed=int(seed) + 1000)
    estimator.fit(train_texts.tolist(), y_train)
    test_logits = np.asarray(estimator.decision_function(test_texts.tolist()), dtype=float)
    tuning.append(
        {
            "layer": "source_outer_full",
            "inner_fold": 0,
            "C": float(c),
            "selected": True,
        }
    )
    return SourceScoreResult(
        train_logits=train_logits,
        test_logits=test_logits,
        train_probabilities=_sigmoid(train_logits),
        test_probabilities=_sigmoid(test_logits),
        selected_c=float(c),
        tuning=tuple(tuning),
    )


def tune_ridge_alpha(
    X_train: np.ndarray,
    target: np.ndarray,
    inner_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    alpha_grid: Sequence[float] = RIDGE_ALPHAS,
) -> tuple[float, dict[float, float]]:
    """Select ridge alpha by inner MSE, preferring stronger regularization on ties."""

    X_train = np.asarray(X_train, dtype=float)
    target = np.asarray(target, dtype=float)
    if len(X_train) != len(target):
        raise ValueError("ridge features and target have different lengths")
    scores: dict[float, float] = {}
    for alpha in sorted(float(value) for value in alpha_grid):
        fold_errors: list[float] = []
        for train_idx, validation_idx in inner_splits:
            estimator = fit_ridge_pipeline(alpha=alpha)
            estimator.fit(X_train[train_idx], target[train_idx])
            prediction = estimator.predict(X_train[validation_idx])
            fold_errors.append(float(mean_squared_error(target[validation_idx], prediction)))
        scores[alpha] = float(np.mean(fold_errors))
    selected = min(scores, key=lambda value: (scores[value], -value))
    return float(selected), scores


def fit_explanation_fold(
    X_train: np.ndarray,
    target_train: np.ndarray,
    X_test: np.ndarray,
    *,
    stratify_labels: np.ndarray,
    inner_folds: int,
    alpha_grid: Sequence[float] = RIDGE_ALPHAS,
    seed: int,
) -> ExplanationFoldResult:
    """Fit one outer explanation fold and produce CF outer-training predictions."""

    X_train = np.asarray(X_train, dtype=float)
    target_train = np.asarray(target_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    stratify_labels = np.asarray(stratify_labels, dtype=int)
    if len(X_train) != len(target_train) or len(X_train) != len(stratify_labels):
        raise ValueError("explanation training arrays have different lengths")
    outer_inner = make_inner_splits(stratify_labels, n_splits=int(inner_folds), seed=int(seed))
    selected, scores = tune_ridge_alpha(
        X_train,
        target_train,
        outer_inner,
        alpha_grid=alpha_grid,
    )
    train_prediction = np.full(len(target_train), np.nan, dtype=float)
    for inner_fold, (inner_train, inner_validation) in enumerate(outer_inner, start=1):
        estimator = fit_ridge_pipeline(alpha=selected)
        estimator.fit(X_train[inner_train], target_train[inner_train])
        train_prediction[inner_validation] = estimator.predict(X_train[inner_validation])
    if not np.isfinite(train_prediction).all():
        raise ValueError("explanation cross-fitted training predictions are incomplete")
    estimator = fit_ridge_pipeline(alpha=selected)
    estimator.fit(X_train, target_train)
    test_prediction = np.asarray(estimator.predict(X_test), dtype=float)
    tuning = tuple(
        {
            "inner_fold": int(index),
            "alpha": float(alpha),
            "mean_inner_mse": float(score),
            "selected": bool(alpha == selected),
        }
        for index, (alpha, score) in enumerate(sorted(scores.items()), start=1)
    )
    return ExplanationFoldResult(
        train_prediction=train_prediction,
        test_prediction=test_prediction,
        selected_alpha=float(selected),
        tuning=tuning,
    )


def relative_r2(
    target: Sequence[float], prediction: Sequence[float], null_prediction: Sequence[float]
) -> float:
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    null_prediction = np.asarray(null_prediction, dtype=float)
    if not (len(target) == len(prediction) == len(null_prediction)):
        raise ValueError("target and prediction arrays have different lengths")
    null_mse = float(mean_squared_error(target, null_prediction))
    if null_mse <= 1e-15:
        return float("nan")
    return float(1.0 - mean_squared_error(target, prediction) / null_mse)


def explanation_metrics(
    target: Sequence[float], prediction: Sequence[float], null_prediction: Sequence[float]
) -> dict[str, float]:
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    null_prediction = np.asarray(null_prediction, dtype=float)
    rho = spearmanr(target, prediction).statistic
    return {
        "r2": relative_r2(target, prediction, null_prediction),
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(target, prediction))),
        "spearman_rho": float(rho) if np.isfinite(rho) else float("nan"),
        "mse_improvement": relative_r2(target, prediction, null_prediction),
        "target_mse": float(mean_squared_error(target, prediction)),
        "null_mse": float(mean_squared_error(target, null_prediction)),
    }


def _stratified_bootstrap_indices(
    labels: np.ndarray, *, n_bootstrap: int, seed: int
) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("stratified bootstrap requires both classes")
    rng = np.random.default_rng(int(seed))
    rows = []
    for _ in range(int(n_bootstrap)):
        rows.append(
            np.concatenate(
                [
                    rng.choice(positive, size=len(positive), replace=True),
                    rng.choice(negative, size=len(negative), replace=True),
                ]
            )
        )
    return np.asarray(rows, dtype=int)


def paired_auc_delta_inference(
    labels: Sequence[int],
    base_probability: Sequence[float],
    enhanced_probability: Sequence[float],
    *,
    n_bootstrap: int,
    n_permutations: int,
    seed: int,
) -> dict[str, float | int]:
    """Participant-paired bootstrap CI and paired swap p-value for delta AUC."""

    labels = np.asarray(labels, dtype=int)
    base = np.asarray(base_probability, dtype=float)
    enhanced = np.asarray(enhanced_probability, dtype=float)
    observed = float(roc_auc_score(labels, enhanced) - roc_auc_score(labels, base))
    bootstrap = _stratified_bootstrap_indices(labels, n_bootstrap=n_bootstrap, seed=seed)
    bootstrap_values = np.asarray(
        [
            roc_auc_score(labels[index], enhanced[index])
            - roc_auc_score(labels[index], base[index])
            for index in bootstrap
        ],
        dtype=float,
    )
    rng = np.random.default_rng(int(seed) + 1_000_003)
    extreme = 0
    for _ in range(int(n_permutations)):
        swap = rng.integers(0, 2, size=len(labels), dtype=np.int8).astype(bool)
        left = np.where(swap, enhanced, base)
        right = np.where(swap, base, enhanced)
        statistic = roc_auc_score(labels, left) - roc_auc_score(labels, right)
        extreme += int(abs(statistic) >= abs(observed) - 1e-15)
    return {
        "delta_auc": observed,
        "ci_low": float(np.quantile(bootstrap_values, 0.025)),
        "ci_high": float(np.quantile(bootstrap_values, 0.975)),
        "n_bootstrap": int(n_bootstrap),
        "n_permutations": int(n_permutations),
        "p_value": float((extreme + 1) / (int(n_permutations) + 1)),
    }


LOCKED_INPUT_HASHES = {
    "splits": "ceaddff172bf9582987cf76ff0522dcfdd0fe90c5b3014565bcfaacf003703ae",
    "structural": "70c439c4a78ab7bea2b1106f59d06214ae0fa9bf09540bd24c859350cc2fce4a",
    "structural_dictionary": "fe3c56b0ebd51118bffee4ef2091c5088fad0d60a3537ac67c3242f52f97f821",
    "domain_count": "7687c70d3770b7e8902900ec3c117d8a679fc6d718abf1c2a7f8d1361b07ae33",
    "domain_source_audit": "8a9b2694bfbaa17d67e75e60545b87acda21e37da871bf3bac7dd3a180d5a864",
    "participant_text": "c520c713922e671431b40382238d64d0b07779ad5e77d905c9224a6f66b4ec9b",
    "interviewer_text": "cf1278335dd4e7ee27dbff13bb792a7c1875341dd9412900c0f72a69f06c5e48",
    "turns": "763cd3857a430f7a8dc6ee997d2820b07787f5e30ddceaee8bf83329fedc1a9a",
}

TEXT_INPUTS = {
    "tfidf_participant": "01_inputs/c2_participant_speech.csv",
    "tfidf_interviewer": "01_inputs/c3_interviewer_speech.csv",
}


@dataclass(frozen=True)
class ExplanationInputs:
    participant_ids: np.ndarray
    labels: np.ndarray
    structural: pd.DataFrame
    domains: pd.DataFrame
    path_features: pd.DataFrame
    block_matrices: dict[str, np.ndarray]
    block_columns: dict[str, tuple[str, ...]]
    structural_audit: pd.DataFrame
    structural_audit_summary: dict[str, object]
    path_dictionary: tuple[dict[str, str], ...]
    path_audit: pd.DataFrame
    path_audit_summary: dict[str, object]
    domain_audit: pd.DataFrame
    domain_audit_summary: dict[str, object]


@dataclass(frozen=True)
class SourceFoldCache:
    train_idx: np.ndarray
    test_idx: np.ndarray
    participant_train_logit: np.ndarray
    participant_test_logit: np.ndarray
    interviewer_train_logit: np.ndarray
    interviewer_test_logit: np.ndarray
    participant_c: float
    interviewer_c: float
    tuning: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class SourceRepresentationData:
    """Frozen participant/interviewer source material used for source refits."""

    participant: np.ndarray
    interviewer: np.ndarray
    feature_type: str


def _hash_variants(path: str | Path) -> dict[str, str]:
    raw = Path(path).read_bytes()
    return {
        "raw": hashlib.sha256(raw).hexdigest(),
        "lf": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),
    }


def _stable_seed_offset(value: object) -> int:
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 10_000


def _validate_expected_hash(path: Path, expected: str, *, label: str) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"locked {label} file not found: {path}")
    observed = _hash_variants(path)
    if expected.lower() not in {value.lower() for value in observed.values()}:
        raise ValueError(
            f"locked {label} hash mismatch: expected={expected}, observed={observed}"
        )
    return observed


def _input_paths(output_root: Path) -> dict[str, Path]:
    return {
        "splits": output_root / "00_splits" / "repeated_5fold_splits_10x5.csv",
        "structural": output_root / "12_submission_audit" / "structural_baseline_features.csv",
        "structural_dictionary": output_root / "12_submission_audit" / "structural_feature_dictionary.md",
        "domain_count": output_root / "04_c5_controls" / "domain_count" / "input.csv",
        "domain_source_audit": output_root / "10_source_importance" / "07_strict_joint_models" / "domain_source_audit.json",
        "participant_text": output_root / "01_inputs" / "c2_participant_speech.csv",
        "interviewer_text": output_root / "01_inputs" / "c3_interviewer_speech.csv",
        "turns": output_root / "01_inputs" / "c4_turns_official142_frozen.csv",
    }


def build_input_availability(
    output_root: str | Path,
    *,
    embedding_root: str | Path | None = None,
    representations: Sequence[str] = ("tfidf", "mpnet", "bge"),
) -> dict[str, object]:
    """Report frozen input availability without creating substitute inputs."""

    output_root = Path(output_root).resolve()
    paths = _input_paths(output_root)
    entries: dict[str, object] = {}
    for name, path in paths.items():
        expected = LOCKED_INPUT_HASHES.get(name)
        entry: dict[str, object] = {
            "path": str(path),
            "required": name in {"splits", "structural", "structural_dictionary", "domain_count", "domain_source_audit"},
        }
        if not path.exists():
            entry.update({"status": "missing", "reason": "file_not_found"})
        else:
            observed = _hash_variants(path)
            entry["status"] = "available"
            entry["sha256"] = observed
            if expected is not None:
                entry["hash_match"] = bool(expected.lower() in {value.lower() for value in observed.values()})
                if not entry["hash_match"]:
                    entry["status"] = "hash_mismatch"
        entries[name] = entry

    embedding_root = Path(embedding_root or output_root / "03_embedding_robustness").resolve()
    embedding_entries: dict[str, object] = {}
    model_paths = {
        "mpnet": embedding_root / "model_1_all_mpnet_base_v2" / "run_manifest.json",
        "bge": embedding_root / "model_2_bge_large_en_v1_5" / "run_manifest.json",
    }
    for representation in representations:
        if representation == "tfidf":
            available = all(entries[key].get("status") == "available" for key in ("participant_text", "interviewer_text"))
            embedding_entries[representation] = {"status": "available" if available else "not_run_data_unavailable", "source": "frozen text tables"}
            continue
        path = model_paths.get(str(representation))
        if path is None or not path.exists():
            embedding_entries[representation] = {"status": "not_run_data_unavailable", "reason": "embedding_manifest_not_found", "path": str(path) if path else None}
        else:
            embedding_entries[representation] = {"status": "available", "path": str(path), "sha256": _hash_variants(path)}
    return {
        "status": "available" if all(
            entry.get("status") in {"available", "not_run_data_unavailable"}
            for entry in entries.values()
        ) else "input_contract_incomplete",
        "output_root": str(output_root),
        "inputs": entries,
        "representations": embedding_entries,
        "d_all": {"status": "not_run_data_unavailable", "reason": "no_frozen_full_transcript_D_All_input"},
    }


def _normalise_membership(
    membership: pd.DataFrame,
    participant_ids: np.ndarray,
    labels: np.ndarray,
    *,
    expected_repeats: int,
) -> pd.DataFrame:
    required = {"repeat", "fold", "role", "participant_id", "label"}
    _require_columns(membership, sorted(required), "split membership")
    result = membership.copy()
    for column in ("repeat", "fold", "participant_id", "label"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype(int)
    result["role"] = result["role"].astype(str)
    label_lookup = dict(zip(participant_ids.tolist(), labels.tolist()))
    if set(result["participant_id"]) != set(participant_ids):
        raise ValueError("split membership participant IDs do not match the cohort")
    if any(int(row.label) != int(label_lookup[int(row.participant_id)]) for row in result[["participant_id", "label"]].drop_duplicates().itertuples(index=False)):
        raise ValueError("split membership labels do not match frozen labels")
    if not set(range(1, int(expected_repeats) + 1)).issubset(set(result["repeat"])):
        raise ValueError("split membership is missing a requested repeat")
    return result


def load_explanation_inputs(
    output_root: str | Path,
    *,
    expected_n: int = EXPECTED_N,
    expected_repeats: int = 10,
    validate_hashes: bool = True,
) -> tuple[ExplanationInputs, pd.DataFrame]:
    """Load and audit the frozen participant, structure, path and D-P inputs."""

    output_root = Path(output_root).resolve()
    paths = _input_paths(output_root)
    if validate_hashes:
        core_keys = (
            "splits",
            "structural",
            "structural_dictionary",
            "domain_count",
            "domain_source_audit",
        )
        for key in core_keys:
            expected = LOCKED_INPUT_HASHES[key]
            _validate_expected_hash(paths[key], expected, label=key)
    structural = pd.read_csv(paths["structural"])
    _require_columns(structural, ["participant_id", "label", "phq8_score", *Q_FEATURES, *R_BASE_FEATURES], "structural baseline")
    structural["participant_id"] = pd.to_numeric(structural["participant_id"], errors="raise").astype(int)
    structural["label"] = pd.to_numeric(structural["label"], errors="raise").astype(int)
    structural["phq8_score"] = pd.to_numeric(structural["phq8_score"], errors="raise")
    structural = structural.sort_values("participant_id", kind="stable").reset_index(drop=True)
    if len(structural) != int(expected_n) or structural["participant_id"].duplicated().any():
        raise ValueError("structural baseline does not contain the expected unique cohort")
    participant_ids = structural["participant_id"].to_numpy(dtype=int)
    labels = structural["label"].to_numpy(dtype=int)
    if set(labels) != {0, 1}:
        raise ValueError("cohort labels must contain both classes")

    structural_feature_columns = [
        column
        for column in structural.columns
        if column not in {"participant_id", "label", "phq8_score"}
    ]
    structural_audit, structural_summary = audit_numeric_features(structural, structural_feature_columns)
    structural_eligible = set(structural_audit.loc[structural_audit["model_eligible"], "feature"])
    q_columns = tuple(column for column in Q_FEATURES if column in structural_eligible)
    r_base_columns = tuple(column for column in R_BASE_FEATURES if column in structural_eligible)

    turns_available = paths["turns"].exists()
    if validate_hashes and turns_available:
        observed_turns = _hash_variants(paths["turns"])
        turns_available = LOCKED_INPUT_HASHES["turns"].lower() in {
            value.lower() for value in observed_turns.values()
        }
    if turns_available:
        turns = pd.read_csv(paths["turns"])
        path_features, path_dictionary = derive_interviewer_path_features(turns)
        path_columns = [column for column in path_features.columns if column != "participant_id"]
        path_audit, path_summary = audit_numeric_features(path_features, path_columns)
        path_eligible = set(path_audit.loc[path_audit["model_eligible"], "feature"])
        path_features = path_features[["participant_id", *path_columns]].copy()
    else:
        path_features = pd.DataFrame({"participant_id": participant_ids})
        path_dictionary = []
        path_columns = []
        path_audit = pd.DataFrame(columns=["feature", "model_eligible"])
        path_summary = {"status": "not_run_data_unavailable", "dropped_features": []}
        path_eligible = set()
    if set(path_features["participant_id"]) != set(participant_ids):
        raise ValueError("path feature participant IDs do not match the structural cohort")
    path_features = path_features.sort_values("participant_id", kind="stable").reset_index(drop=True)

    domains = validate_dp_frame(pd.read_csv(paths["domain_count"]), expected_ids=participant_ids)
    domain_audit, domain_summary = audit_numeric_features(domains, D_P_FEATURES)
    domain_eligible = set(domain_audit.loc[domain_audit["model_eligible"], "feature"])
    d_columns = tuple(column for column in D_P_FEATURES if column in domain_eligible)

    combined = structural[["participant_id", *Q_FEATURES, *R_BASE_FEATURES]].merge(
        path_features, on="participant_id", how="left", validate="one_to_one"
    )
    r_columns = tuple(
        column
        for column in (*r_base_columns, *path_columns)
        if column in combined.columns and (column in r_base_columns or column in path_eligible)
    )
    block_columns = {
        "P": ("participant_logit",),
        "Q": q_columns,
        "R": r_columns,
        "D": d_columns,
    }
    block_matrices = {
        "Q": combined[list(q_columns)].to_numpy(dtype=float) if q_columns else np.empty((len(participant_ids), 0)),
        "R": combined[list(r_columns)].to_numpy(dtype=float) if r_columns else np.empty((len(participant_ids), 0)),
        "D": domains[list(d_columns)].to_numpy(dtype=float) if d_columns else np.empty((len(participant_ids), 0)),
    }
    split = pd.read_csv(paths["splits"])
    split = _normalise_membership(split, participant_ids, labels, expected_repeats=expected_repeats)
    return (
        ExplanationInputs(
            participant_ids=participant_ids,
            labels=labels,
            structural=structural,
            domains=domains,
            path_features=path_features,
            block_matrices=block_matrices,
            block_columns=block_columns,
            structural_audit=structural_audit,
            structural_audit_summary=structural_summary,
            path_dictionary=tuple(path_dictionary),
            path_audit=path_audit,
            path_audit_summary=path_summary,
            domain_audit=domain_audit,
            domain_audit_summary=domain_summary,
        ),
        split,
    )


def _load_source_text(
    path: Path,
    participant_ids: np.ndarray,
    labels: np.ndarray,
    *,
    condition: str,
) -> np.ndarray:
    table = pd.read_csv(path, keep_default_na=False)
    _require_columns(table, ["participant_id", "paper_label_phq8_ge10", "condition_id", "text"], condition)
    table["participant_id"] = pd.to_numeric(table["participant_id"], errors="raise").astype(int)
    table["paper_label_phq8_ge10"] = pd.to_numeric(table["paper_label_phq8_ge10"], errors="raise").astype(int)
    table = table.sort_values("participant_id", kind="stable").reset_index(drop=True)
    if not np.array_equal(table["participant_id"].to_numpy(dtype=int), participant_ids):
        raise ValueError(f"{condition} text participant IDs are not aligned")
    if not np.array_equal(table["paper_label_phq8_ge10"].to_numpy(dtype=int), labels):
        raise ValueError(f"{condition} text labels are not aligned")
    if not table["condition_id"].eq(condition).all():
        raise ValueError(f"{condition} text condition identity is not frozen")
    return table["text"].fillna("").astype(str).to_numpy(dtype=object)


def load_source_representation_data(
    output_root: str | Path,
    participant_ids: np.ndarray,
    labels: np.ndarray,
    *,
    representations: Sequence[str],
    embedding_root: str | Path | None = None,
) -> dict[str, SourceRepresentationData]:
    """Load the already frozen source material without re-encoding it."""

    output_root = Path(output_root).resolve()
    paths = _input_paths(output_root)
    result: dict[str, SourceRepresentationData] = {}
    if "tfidf" in representations:
        result["tfidf"] = SourceRepresentationData(
            participant=_load_source_text(
                paths["participant_text"], participant_ids, labels, condition="participant_speech"
            ),
            interviewer=_load_source_text(
                paths["interviewer_text"], participant_ids, labels, condition="interviewer_speech"
            ),
            feature_type="text",
        )
    embedding_root = Path(embedding_root or output_root / "03_embedding_robustness").resolve()
    embedding_specs = {
        "mpnet": ("model_1_all_mpnet_base_v2", 768),
        "bge": ("model_2_bge_large_en_v1_5", 1024),
    }
    for representation in representations:
        if representation not in embedding_specs:
            continue
        slug, dimension = embedding_specs[representation]
        pair = load_frozen_embedding_pair(
            embedding_root,
            slug,
            participant_ids,
            expected_dim=dimension,
            allow_subset=False,
        )
        result[representation] = SourceRepresentationData(
            participant=pair.participant,
            interviewer=pair.interviewer,
            feature_type="embedding",
        )
    return result


def _source_result_from_dense(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    inner_folds: int,
    c_grid: Sequence[float],
    seed: int,
) -> SourceScoreResult:
    train, test, tuning = nested_dense_source_logits(
        X_train,
        y_train,
        X_test,
        inner_folds=inner_folds,
        c_grid=c_grid,
        seed=seed,
    )
    selected_c = _selected_outer_c(tuning, default=float(c_grid[0]))
    return SourceScoreResult(
        train_logits=train,
        test_logits=test,
        train_probabilities=_sigmoid(train),
        test_probabilities=_sigmoid(test),
        selected_c=selected_c,
        tuning=tuple(tuning),
    )


def _fit_source_on_training_predict_donors(
    *,
    representation: str,
    source_data: SourceRepresentationData,
    labels: np.ndarray,
    source_train_idx: np.ndarray,
    donor_idx: np.ndarray,
    inner_folds: int,
    c_grid: Sequence[float],
    seed: int,
) -> np.ndarray:
    """Fit a source model on source_train_idx and predict donor_idx only."""

    if len(donor_idx) == 0:
        return np.empty(0, dtype=float)
    if representation == "tfidf":
        result = nested_text_source_logits(
            source_data.interviewer[source_train_idx],
            labels[source_train_idx],
            source_data.interviewer[donor_idx],
            inner_folds=inner_folds,
            seed=seed,
        )
    elif representation in {"mpnet", "bge"}:
        result = _source_result_from_dense(
            source_data.interviewer[source_train_idx],
            labels[source_train_idx],
            source_data.interviewer[donor_idx],
            inner_folds=inner_folds,
            c_grid=c_grid,
            seed=seed,
        )
    else:
        raise ValueError(f"unknown representation: {representation}")
    return np.asarray(result.test_logits, dtype=float)


def build_fold_contained_matched_source_logits(
    *,
    representation: str,
    source_data: SourceRepresentationData,
    labels: np.ndarray,
    participant_ids: np.ndarray,
    match_frame: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    outer_test_assignment: pd.DataFrame,
    outer_train_assignment: pd.DataFrame | None = None,
    inner_folds: int = INNER_FOLDS,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    seed: int = BASE_SEED,
    assignment_mode: str = "optimal_matched",
    random_reference_draws: int = PAIRING_DRAWS,
    collect_inner_quality: bool = True,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Mismatch source data inside folds before predicting matched logits."""

    labels = np.asarray(labels, dtype=int)
    participant_ids = np.asarray(participant_ids)
    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    if source_data.interviewer.shape[0] != len(participant_ids):
        raise ValueError("source data and participant IDs are not aligned")
    if set(train_idx).intersection(test_idx):
        raise ValueError("outer train and test indices overlap")
    if assignment_mode not in {"optimal_matched", "random_mismatch"}:
        raise ValueError("assignment_mode must be optimal_matched or random_mismatch")
    if set(outer_test_assignment["recipient_id"]) != set(participant_ids[test_idx]):
        raise ValueError("outer-test assignment does not cover the outer-test recipients")
    if set(outer_test_assignment["donor_id"]) != set(participant_ids[test_idx]):
        raise ValueError("outer-test assignment donors must stay inside outer-test")
    frame = match_frame.reset_index(drop=True)
    matched_train = np.full(len(train_idx), np.nan, dtype=float)
    matched_test = np.full(len(test_idx), np.nan, dtype=float)
    audit_rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    inner_splits = make_inner_splits(labels[train_idx], n_splits=int(inner_folds), seed=int(seed))
    for inner_fold, (inner_train_local, inner_validation_local) in enumerate(inner_splits, start=1):
        inner_train_global = train_idx[inner_train_local]
        inner_validation_global = train_idx[inner_validation_local]
        validation_frame = frame.iloc[inner_validation_global].reset_index(drop=True)
        fit_frame = frame.iloc[inner_train_global].reset_index(drop=True)
        if assignment_mode == "random_mismatch":
            assignment = make_random_derangement_assignment(
                recipient=validation_frame,
                donor=validation_frame,
                fit_reference=fit_frame,
                seed=int(seed) + inner_fold,
            )
        else:
            assignment = make_pairing_assignment(
                recipient=validation_frame,
                donor=validation_frame,
                fit_reference=fit_frame,
                seed=int(seed) + inner_fold,
            )
        if collect_inner_quality:
            random_distances: list[float] = []
            for random_draw in range(1, int(random_reference_draws) + 1):
                random_assignment = make_random_derangement_assignment(
                    recipient=validation_frame,
                    donor=validation_frame,
                    fit_reference=fit_frame,
                    seed=int(seed) + inner_fold * 10_000 + random_draw,
                )
                random_quality = pairing_quality_summary(
                    recipient=validation_frame,
                    donor=validation_frame,
                    assignment=random_assignment,
                    fit_reference=fit_frame,
                    assignment_type="random_mismatch",
                )
                random_distances.append(float(random_quality["mean_distance"]))
            quality = pairing_quality_summary(
                recipient=validation_frame,
                donor=validation_frame,
                assignment=assignment,
                fit_reference=fit_frame,
                random_distance_reference=random_distances,
                assignment_type=assignment_mode,
            )
            quality_rows.append(
                {
                    "partition": "outer_train_inner_validation",
                    "inner_fold": int(inner_fold),
                    "assignment_type": assignment_mode,
                    "recipient_n": int(len(validation_frame)),
                    "random_reference_n": int(len(random_distances)),
                    "mean_distance": float(quality["mean_distance"]),
                    "median_distance": float(quality["median_distance"]),
                    "q95_distance": float(quality["q95_distance"]),
                    "max_distance": float(quality["max_distance"]),
                    "total_cost": float(quality["total_cost"]),
                    "random_distance_q05": float(quality["random_distance_q05"]),
                    "optimal_distance_random_percentile": float(quality["optimal_distance_random_percentile"]),
                    "n_features_mean_abs_paired_diff_lt_0_50": int(quality["n_features_mean_abs_paired_diff_lt_0_50"]),
                    "no_self": bool(quality["no_self"]),
                    "one_to_one": bool(quality["one_to_one"]),
                    "matching_adequate": bool(quality["matching_adequate"]),
                    "max_abs_smd_descriptive": float(quality["max_abs_smd"]),
                    **{
                        f"mean_abs_standardized_paired_difference__{feature}": float(value)
                        for feature, value in quality["feature_mean_abs_standardized_paired_difference"].items()
                    },
                }
            )
        global_lookup = {participant_id: int(index) for index, participant_id in enumerate(participant_ids.tolist())}
        donor_global = np.asarray([global_lookup[value] for value in assignment["donor_id"]], dtype=int)
        predicted = _fit_source_on_training_predict_donors(
            representation=representation,
            source_data=source_data,
            labels=labels,
            source_train_idx=inner_train_global,
            donor_idx=donor_global,
            inner_folds=inner_folds,
            c_grid=c_grid,
            seed=int(seed) + 10_000 + inner_fold,
        )
        predicted_by_recipient = {
            recipient_id: float(logit)
            for recipient_id, logit in zip(assignment["recipient_id"], predicted)
        }
        for local_position, global_index in zip(inner_validation_local, inner_validation_global):
            recipient_id = participant_ids[global_index]
            donor_id = assignment.loc[assignment["recipient_id"].eq(recipient_id), "donor_id"].iloc[0]
            matched_train[int(local_position)] = predicted_by_recipient[recipient_id]
            audit_rows.append(
                {
                    "scope": "outer_train_inner_validation",
                    "inner_fold": int(inner_fold),
                    "recipient_id": recipient_id,
                    "donor_id": donor_id,
                    "donor_partition": "inner_validation",
                    "source_train_ids": tuple(participant_ids[inner_train_global].tolist()),
                    "source_feature_type": source_data.feature_type,
                    "prediction_stage": "after_mismatch",
                    "source_model_scope": "inner_training_only",
                    "assignment_type": assignment_mode,
                }
            )
    if not np.isfinite(matched_train).all():
        raise ValueError("fold-contained matched training logits are incomplete")

    global_lookup = {participant_id: int(index) for index, participant_id in enumerate(participant_ids.tolist())}
    test_donor_global = np.asarray([global_lookup[value] for value in outer_test_assignment["donor_id"]], dtype=int)
    outer_test_logits = _fit_source_on_training_predict_donors(
        representation=representation,
        source_data=source_data,
        labels=labels,
        source_train_idx=train_idx,
        donor_idx=test_donor_global,
        inner_folds=inner_folds,
        c_grid=c_grid,
        seed=int(seed) + 20_000,
    )
    test_local_by_id = {participant_ids[global_index]: local for local, global_index in enumerate(test_idx)}
    for assignment_row, logit in zip(outer_test_assignment.to_dict("records"), outer_test_logits):
        matched_test[test_local_by_id[assignment_row["recipient_id"]]] = float(logit)
        audit_rows.append(
            {
                "scope": "outer_test",
                "inner_fold": 0,
                "recipient_id": assignment_row["recipient_id"],
                "donor_id": assignment_row["donor_id"],
                "donor_partition": "outer_test",
                "source_train_ids": tuple(participant_ids[train_idx].tolist()),
                "source_feature_type": source_data.feature_type,
                "prediction_stage": "after_mismatch",
                "source_model_scope": "outer_training_only",
                "assignment_type": str(assignment_row.get("assignment_type", assignment_mode)),
            }
        )
    if not np.isfinite(matched_test).all():
        raise ValueError("fold-contained matched outer-test logits are incomplete")
    return matched_train, matched_test, pd.DataFrame(audit_rows), pd.DataFrame(quality_rows)


def _source_fold_cache(
    representation: str,
    *,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    labels: np.ndarray,
    participant_text: np.ndarray | None,
    interviewer_text: np.ndarray | None,
    participant_embedding: np.ndarray | None,
    interviewer_embedding: np.ndarray | None,
    inner_folds: int,
    c_grid: Sequence[float],
    seed: int,
) -> SourceFoldCache:
    if representation == "tfidf":
        if participant_text is None or interviewer_text is None:
            raise ValueError("TF-IDF source text is unavailable")
        participant = nested_text_source_logits(
            participant_text[train_idx],
            labels[train_idx],
            participant_text[test_idx],
            inner_folds=inner_folds,
            seed=seed + 11,
        )
        interviewer = nested_text_source_logits(
            interviewer_text[train_idx],
            labels[train_idx],
            interviewer_text[test_idx],
            inner_folds=inner_folds,
            seed=seed + 22,
        )
    elif representation in {"mpnet", "bge"}:
        if participant_embedding is None or interviewer_embedding is None:
            raise ValueError(f"{representation} embedding cache is unavailable")
        participant = _source_result_from_dense(
            participant_embedding[train_idx],
            labels[train_idx],
            participant_embedding[test_idx],
            inner_folds=inner_folds,
            c_grid=c_grid,
            seed=seed + 11,
        )
        interviewer = _source_result_from_dense(
            interviewer_embedding[train_idx],
            labels[train_idx],
            interviewer_embedding[test_idx],
            inner_folds=inner_folds,
            c_grid=c_grid,
            seed=seed + 22,
        )
    else:
        raise ValueError(f"unknown representation: {representation}")
    tuning = tuple(
        {
            "source": source,
            **row,
        }
        for source, result in (("P", participant), ("I", interviewer))
        for row in result.tuning
    )
    return SourceFoldCache(
        train_idx=train_idx.copy(),
        test_idx=test_idx.copy(),
        participant_train_logit=participant.train_logits,
        participant_test_logit=participant.test_logits,
        interviewer_train_logit=interviewer.train_logits,
        interviewer_test_logit=interviewer.test_logits,
        participant_c=participant.selected_c,
        interviewer_c=interviewer.selected_c,
        tuning=tuning,
    )


def run_source_score_crossfit(
    inputs: ExplanationInputs,
    membership: pd.DataFrame,
    output_root: str | Path,
    *,
    representations: Sequence[str],
    repeats: Sequence[int],
    inner_folds: int = INNER_FOLDS,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    embedding_root: str | Path | None = None,
    base_seed: int = BASE_SEED,
) -> tuple[pd.DataFrame, dict[tuple[str, int, int], SourceFoldCache], pd.DataFrame]:
    """Run strict source cross-fitting and retain fold-local caches in memory."""

    output_root = Path(output_root).resolve()
    source_data = load_source_representation_data(
        output_root,
        inputs.participant_ids,
        inputs.labels,
        representations=representations,
        embedding_root=embedding_root,
    )

    rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    cache: dict[tuple[str, int, int], SourceFoldCache] = {}
    for representation in representations:
        for repeat in sorted(int(value) for value in repeats):
            fold_values = sorted(membership.loc[membership["repeat"].eq(repeat), "fold"].unique().tolist())
            if fold_values != [1, 2, 3, 4, 5]:
                raise ValueError(f"repeat {repeat} does not contain exactly five outer folds")
            for fold in fold_values:
                train_idx, test_idx = split_indices(
                    membership, repeat, int(fold), inputs.participant_ids
                )
                seed = int(base_seed) + repeat * 100_000 + int(fold) * 1_000
                participant_text = interviewer_text = None
                participant_embedding = interviewer_embedding = None
                representation_data = source_data[representation]
                if representation_data.feature_type == "text":
                    participant_text = representation_data.participant
                    interviewer_text = representation_data.interviewer
                else:
                    participant_embedding = representation_data.participant
                    interviewer_embedding = representation_data.interviewer
                fold_cache = _source_fold_cache(
                    representation,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    labels=inputs.labels,
                    participant_text=participant_text,
                    interviewer_text=interviewer_text,
                    participant_embedding=participant_embedding,
                    interviewer_embedding=interviewer_embedding,
                    inner_folds=inner_folds,
                    c_grid=c_grid,
                    seed=seed,
                )
                cache[(representation, repeat, int(fold))] = fold_cache
                tuning_rows.extend(
                    {
                        "representation": representation,
                        "repeat": int(repeat),
                        "outer_fold": int(fold),
                        **row,
                    }
                    for row in fold_cache.tuning
                )
                c_descriptor = f"participant={fold_cache.participant_c:g};interviewer={fold_cache.interviewer_c:g}"
                for local_index, global_index in enumerate(test_idx):
                    rows.append(
                        {
                            "participant_id": int(inputs.participant_ids[global_index]),
                            "label": int(inputs.labels[global_index]),
                            "repeat": int(repeat),
                            "outer_fold": int(fold),
                            "representation": representation,
                            "participant_logit": float(fold_cache.participant_test_logit[local_index]),
                            "interviewer_logit": float(fold_cache.interviewer_test_logit[local_index]),
                            "participant_probability": float(_sigmoid(fold_cache.participant_test_logit[local_index])),
                            "interviewer_probability": float(_sigmoid(fold_cache.interviewer_test_logit[local_index])),
                            "source_model_C": c_descriptor,
                            "participant_source_model_C": float(fold_cache.participant_c),
                            "interviewer_source_model_C": float(fold_cache.interviewer_c),
                            "source_score_scope": "outer_test",
                        }
                    )
    result = pd.DataFrame(rows).sort_values(
        ["representation", "repeat", "outer_fold", "participant_id"], kind="stable"
    ).reset_index(drop=True)
    public_output_columns_are_safe(result.columns)
    return result, cache, pd.DataFrame(tuning_rows)


def _subset_design(
    subset: str,
    *,
    participant_logits: np.ndarray,
    block_matrices: Mapping[str, np.ndarray],
    indices: np.ndarray,
) -> np.ndarray:
    columns: list[np.ndarray] = []
    if subset == "null":
        return np.empty((len(indices), 0), dtype=float)
    for block in subset.split("+"):
        if block == "P":
            columns.append(np.asarray(participant_logits, dtype=float).reshape(-1, 1))
        else:
            matrix = np.asarray(block_matrices[block], dtype=float)
            if matrix.ndim != 2 or matrix.shape[0] < int(indices.max(initial=-1)) + 1:
                raise ValueError(f"block matrix {block} is not aligned with the cohort")
            columns.append(matrix[indices])
    if not columns:
        return np.empty((len(indices), 0), dtype=float)
    return np.column_stack(columns)


def run_direct_explanation(
    inputs: ExplanationInputs,
    source_cache: Mapping[tuple[str, int, int], SourceFoldCache],
    membership: pd.DataFrame,
    *,
    representations: Sequence[str],
    repeats: Sequence[int],
    inner_folds: int = INNER_FOLDS,
    alpha_grid: Sequence[float] = RIDGE_ALPHAS,
    base_seed: int = BASE_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[tuple[str, int, int], ExplanationFoldResult]]:
    """Fit all 16 ridge recoverability subsets with fold-local scaling."""

    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    full_cache: dict[tuple[str, int, int], ExplanationFoldResult] = {}
    for representation in representations:
        for repeat in sorted(int(value) for value in repeats):
            for fold in range(1, 6):
                key = (representation, repeat, fold)
                if key not in source_cache:
                    raise ValueError(f"source-score cache is missing {key}")
                source = source_cache[key]
                train_idx, test_idx = source.train_idx, source.test_idx
                target_train = source.interviewer_train_logit
                target_test = source.interviewer_test_logit
                null_value = float(np.mean(target_train))
                target_scale = float(np.std(target_train, ddof=0))
                if target_scale <= 1e-12:
                    target_scale = 1.0
                participant_mean = float(np.mean(source.participant_train_logit))
                participant_scale = float(np.std(source.participant_train_logit, ddof=0))
                if participant_scale <= 1e-12:
                    participant_scale = 1.0
                for subset_index, subset in enumerate(ALL_BLOCK_SUBSETS):
                    X_train = _subset_design(
                        subset,
                        participant_logits=source.participant_train_logit,
                        block_matrices=inputs.block_matrices,
                        indices=train_idx,
                    )
                    X_test = _subset_design(
                        subset,
                        participant_logits=source.participant_test_logit,
                        block_matrices=inputs.block_matrices,
                        indices=test_idx,
                    )
                    if subset == "null":
                        prediction = np.full(len(test_idx), null_value, dtype=float)
                        selected_alpha = float("nan")
                        tuning = ()
                        fold_result = None
                    else:
                        fold_result = fit_explanation_fold(
                            X_train,
                            target_train,
                            X_test,
                            stratify_labels=inputs.labels[train_idx],
                            inner_folds=inner_folds,
                            alpha_grid=alpha_grid,
                            seed=int(base_seed) + repeat * 100_000 + fold * 1_000 + subset_index,
                        )
                        prediction = fold_result.test_prediction
                        selected_alpha = fold_result.selected_alpha
                        tuning = fold_result.tuning
                    if subset == "P+Q+R+D":
                        if fold_result is None:
                            raise AssertionError("full explanation cannot be null")
                        full_cache[key] = fold_result
                    for tuning_row in tuning:
                        tuning_rows.append(
                            {
                                "representation": representation,
                                "repeat": int(repeat),
                                "outer_fold": int(fold),
                                "subset": subset,
                                **tuning_row,
                            }
                        )
                    for local_index, global_index in enumerate(test_idx):
                        prediction_rows.append(
                            {
                                "participant_id": int(inputs.participant_ids[global_index]),
                                "label": int(inputs.labels[global_index]),
                                "representation": representation,
                                "repeat": int(repeat),
                                "outer_fold": int(fold),
                                "subset": subset,
                                "participant_logit": float(source.participant_test_logit[local_index]),
                                "interviewer_logit": float(target_test[local_index]),
                                "predicted_interviewer_logit": float(prediction[local_index]),
                                "outer_train_mean_null_prediction": null_value,
                                "outer_train_target_scale_mean": null_value,
                                "outer_train_target_scale_sd": target_scale,
                                "outer_train_participant_scale_mean": participant_mean,
                                "outer_train_participant_scale_sd": participant_scale,
                                "participant_logit_outer_train_standardized": float(
                                    (source.participant_test_logit[local_index] - participant_mean) / participant_scale
                                ),
                                "interviewer_logit_outer_train_standardized": float(
                                    (target_test[local_index] - null_value) / target_scale
                                ),
                                "predicted_interviewer_logit_outer_train_standardized": float(
                                    (prediction[local_index] - null_value) / target_scale
                                ),
                                "outer_train_mean_null_prediction_standardized": float(
                                    (null_value - null_value) / target_scale
                                ),
                                "selected_alpha": selected_alpha,
                            }
                        )
    predictions = pd.DataFrame(prediction_rows).sort_values(
        ["representation", "repeat", "subset", "participant_id"], kind="stable"
    ).reset_index(drop=True)
    for (representation, repeat, subset), group in predictions.groupby(
        ["representation", "repeat", "subset"], sort=True
    ):
        metrics = explanation_metrics(
            group["interviewer_logit"].to_numpy(dtype=float),
            group["predicted_interviewer_logit"].to_numpy(dtype=float),
            group["outer_train_mean_null_prediction"].to_numpy(dtype=float),
        )
        metric_rows.append(
            {
                "representation": representation,
                "repeat": int(repeat),
                "subset": subset,
                **metrics,
                "interpretation_scope": "statistical_score_recoverability",
            }
        )
    metrics = pd.DataFrame(metric_rows)
    tuning = pd.DataFrame(tuning_rows)
    public_output_columns_are_safe(predictions.columns)
    return predictions, metrics, tuning, full_cache


def is_source_score_scale_sensitive(
    *,
    raw_r2: float,
    standardized_r2: float,
    raw_spearman: float,
    standardized_spearman: float,
    substantive_threshold: float = 0.05,
) -> bool:
    """Apply the pre-locked substantive source-score scale sensitivity rule."""

    threshold = float(substantive_threshold)
    if threshold <= 0:
        raise ValueError("substantive_threshold must be positive")
    r2_difference = abs(float(raw_r2) - float(standardized_r2))
    spearman_difference = abs(float(raw_spearman) - float(standardized_spearman))
    direction_flip = bool(
        np.isfinite(raw_r2)
        and np.isfinite(standardized_r2)
        and (
            (float(raw_r2) > 0 and float(standardized_r2) < 0)
            or (float(raw_r2) < 0 and float(standardized_r2) > 0)
        )
    )
    return bool(r2_difference >= threshold or spearman_difference >= threshold or direction_flip)


def recoverability_fold_metrics(explanation_predictions: pd.DataFrame) -> pd.DataFrame:
    """Report raw and outer-training-standardized metrics for every outer fold."""

    required = {
        "representation",
        "repeat",
        "outer_fold",
        "subset",
        "interviewer_logit",
        "predicted_interviewer_logit",
        "outer_train_mean_null_prediction",
        "outer_train_target_scale_mean",
        "outer_train_target_scale_sd",
        "outer_train_participant_scale_mean",
        "outer_train_participant_scale_sd",
    }
    missing = sorted(required.difference(explanation_predictions.columns))
    if missing:
        raise ValueError(f"explanation predictions are missing scale fields: {missing}")
    rows: list[dict[str, object]] = []
    grouping = ["representation", "repeat", "outer_fold", "subset"]
    for keys, group in explanation_predictions.groupby(grouping, sort=True):
        representation, repeat, outer_fold, subset = keys
        target = group["interviewer_logit"].to_numpy(dtype=float)
        prediction = group["predicted_interviewer_logit"].to_numpy(dtype=float)
        null_prediction = group["outer_train_mean_null_prediction"].to_numpy(dtype=float)
        target_mean = float(group["outer_train_target_scale_mean"].iloc[0])
        target_sd = float(group["outer_train_target_scale_sd"].iloc[0])
        if target_sd <= 1e-12:
            target_sd = 1.0
        target_standardized = (target - target_mean) / target_sd
        prediction_standardized = (prediction - target_mean) / target_sd
        null_standardized = (null_prediction - target_mean) / target_sd
        raw = explanation_metrics(target, prediction, null_prediction)
        standardized = explanation_metrics(
            target_standardized,
            prediction_standardized,
            null_standardized,
        )
        rows.append(
            {
                "representation": representation,
                "repeat": int(repeat),
                "outer_fold": int(outer_fold),
                "subset": subset,
                "n_participants": int(len(group)),
                "outer_train_target_scale_mean": target_mean,
                "outer_train_target_scale_sd": target_sd,
                "outer_train_participant_scale_mean": float(group["outer_train_participant_scale_mean"].iloc[0]),
                "outer_train_participant_scale_sd": float(group["outer_train_participant_scale_sd"].iloc[0]),
                **{f"raw_{key}": value for key, value in raw.items()},
                **{f"outer_train_standardized_{key}": value for key, value in standardized.items()},
                "standardization_scope": "outer_training_source_scores_only",
                "interpretation_scope": "recoverability_fold_scale_sensitivity",
            }
        )
    return pd.DataFrame(rows)


def recoverability_scale_sensitivity(explanation_predictions: pd.DataFrame) -> pd.DataFrame:
    """Compare pooled raw and fold-standardized recoverability metrics."""

    fold_metrics = recoverability_fold_metrics(explanation_predictions)
    rows: list[dict[str, object]] = []
    grouping = ["representation", "repeat", "subset"]
    for keys, group in explanation_predictions.groupby(grouping, sort=True):
        representation, repeat, subset = keys
        group = group.sort_values(["outer_fold", "participant_id"], kind="stable")
        target = group["interviewer_logit"].to_numpy(dtype=float)
        prediction = group["predicted_interviewer_logit"].to_numpy(dtype=float)
        null_prediction = group["outer_train_mean_null_prediction"].to_numpy(dtype=float)
        raw = explanation_metrics(target, prediction, null_prediction)
        standardized_target: list[np.ndarray] = []
        standardized_prediction: list[np.ndarray] = []
        standardized_null: list[np.ndarray] = []
        for _, fold in group.groupby("outer_fold", sort=True):
            target_mean = float(fold["outer_train_target_scale_mean"].iloc[0])
            target_sd = float(fold["outer_train_target_scale_sd"].iloc[0])
            if target_sd <= 1e-12:
                target_sd = 1.0
            fold_target = fold["interviewer_logit"].to_numpy(dtype=float)
            fold_prediction = fold["predicted_interviewer_logit"].to_numpy(dtype=float)
            fold_null = fold["outer_train_mean_null_prediction"].to_numpy(dtype=float)
            standardized_target.append((fold_target - target_mean) / target_sd)
            standardized_prediction.append((fold_prediction - target_mean) / target_sd)
            standardized_null.append((fold_null - target_mean) / target_sd)
        standardized = explanation_metrics(
            np.concatenate(standardized_target),
            np.concatenate(standardized_prediction),
            np.concatenate(standardized_null),
        )
        scale_sensitive = is_source_score_scale_sensitive(
            raw_r2=float(raw["r2"]),
            standardized_r2=float(standardized["r2"]),
            raw_spearman=float(raw["spearman_rho"]),
            standardized_spearman=float(standardized["spearman_rho"]),
        )
        for scale_name, metrics in (("raw", raw), ("outer_train_standardized", standardized)):
            rows.append(
                {
                    "representation": representation,
                    "repeat": int(repeat),
                    "subset": subset,
                    "scale": scale_name,
                    "n_participants": int(len(group)),
                    **metrics,
                    "source_score_scale_sensitive": scale_sensitive,
                    "standardization_scope": "none" if scale_name == "raw" else "outer_training_source_scores_only",
                    "interpretation_scope": "recoverability_scale_sensitivity",
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty and not fold_metrics.empty:
        result["n_outer_folds"] = result.apply(
            lambda row: int(
                fold_metrics.loc[
                    fold_metrics["representation"].eq(row["representation"])
                    & fold_metrics["repeat"].eq(int(row["repeat"]))
                    & fold_metrics["subset"].eq(row["subset"]),
                    "outer_fold",
                ].nunique()
            ),
            axis=1,
        )
    return result


def _class_mean_predictions(
    training_values: np.ndarray,
    training_labels: np.ndarray,
    recipient_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    training_values = np.asarray(training_values, dtype=float)
    training_labels = np.asarray(training_labels, dtype=int)
    recipient_labels = np.asarray(recipient_labels, dtype=int)
    means: dict[int, float] = {}
    counts: dict[int, int] = {}
    for label in np.unique(training_labels):
        mask = training_labels == int(label)
        if not mask.any():
            continue
        means[int(label)] = float(training_values[mask].mean())
        counts[int(label)] = int(mask.sum())
    missing = sorted(set(int(value) for value in recipient_labels).difference(means))
    if missing:
        raise ValueError(f"label-only class means are unavailable for labels: {missing}")
    return (
        np.asarray([means[int(label)] for label in recipient_labels], dtype=float),
        np.asarray([counts[int(label)] for label in recipient_labels], dtype=int),
    )


def _cross_fitted_class_mean_predictions(
    values: np.ndarray,
    labels: np.ndarray,
    inner_splits: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=int)
    prediction = np.full(len(values), np.nan, dtype=float)
    counts = np.zeros(len(values), dtype=int)
    for inner_train, inner_validation in inner_splits:
        fold_prediction, fold_counts = _class_mean_predictions(
            values[inner_train],
            labels[inner_train],
            labels[inner_validation],
        )
        prediction[inner_validation] = fold_prediction
        counts[inner_validation] = fold_counts
    if not np.isfinite(prediction).all() or not (counts > 0).all():
        raise ValueError("cross-fitted label-only predictions are incomplete")
    return prediction, counts


def _recoverability_metric_summary(
    target: Sequence[float],
    prediction: Sequence[float],
    null_prediction: Sequence[float],
) -> dict[str, float | bool]:
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    null_prediction = np.asarray(null_prediction, dtype=float)
    target_mse = float(mean_squared_error(target, prediction))
    null_mse = float(mean_squared_error(target, null_prediction))
    r2_defined = bool(null_mse > 1e-15)
    r2 = float(1.0 - target_mse / null_mse) if r2_defined else 0.0
    rho = spearmanr(target, prediction).statistic
    return {
        "r2": r2,
        "delta_mse": float(null_mse - target_mse),
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": float(np.sqrt(target_mse)),
        "spearman_rho": float(rho) if np.isfinite(rho) else float("nan"),
        "target_mse": target_mse,
        "null_mse": null_mse,
        "r2_defined": r2_defined,
    }


def run_label_only_recoverability(
    source_cache: Mapping[tuple[str, int, int], SourceFoldCache],
    *,
    labels: np.ndarray,
    participant_ids: np.ndarray,
    representations: Sequence[str],
    repeats: Sequence[int],
    inner_folds: int = INNER_FOLDS,
    base_seed: int = BASE_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Benchmark interviewer-score recoverability from labels alone."""

    labels = np.asarray(labels, dtype=int)
    participant_ids = np.asarray(participant_ids)
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for representation in representations:
        for repeat in sorted(int(value) for value in repeats):
            for fold in range(1, 6):
                source = source_cache[(representation, repeat, fold)]
                train_idx, test_idx = source.train_idx, source.test_idx
                train_labels = labels[train_idx]
                split_seed = int(base_seed) + repeat * 100_000 + fold * 1_000 + 71
                inner_splits = make_inner_splits(train_labels, n_splits=int(inner_folds), seed=split_seed)
                train_prediction, train_counts = _cross_fitted_class_mean_predictions(
                    source.interviewer_train_logit,
                    train_labels,
                    inner_splits,
                )
                outer_test_prediction, outer_test_counts = _class_mean_predictions(
                    source.interviewer_train_logit,
                    train_labels,
                    labels[test_idx],
                )
                overall_null = float(np.mean(source.interviewer_train_logit))
                for local_index, global_index in enumerate(train_idx):
                    prediction_rows.append(
                        {
                            "participant_id": int(participant_ids[global_index]),
                            "label": int(labels[global_index]),
                            "representation": representation,
                            "repeat": int(repeat),
                            "outer_fold": int(fold),
                            "interviewer_logit": float(source.interviewer_train_logit[local_index]),
                            "label_only_prediction": float(train_prediction[local_index]),
                            "prediction_scope": "outer_train_inner_validation",
                            "class_mean_training_n": int(train_counts[local_index]),
                            "class_mean_training_scope": "inner_training_only",
                            "class_mean_excludes_recipient": True,
                            "outer_train_overall_mean_null_prediction": overall_null,
                        }
                    )
                for local_index, global_index in enumerate(test_idx):
                    prediction_rows.append(
                        {
                            "participant_id": int(participant_ids[global_index]),
                            "label": int(labels[global_index]),
                            "representation": representation,
                            "repeat": int(repeat),
                            "outer_fold": int(fold),
                            "interviewer_logit": float(source.interviewer_test_logit[local_index]),
                            "label_only_prediction": float(outer_test_prediction[local_index]),
                            "prediction_scope": "outer_test",
                            "class_mean_training_n": int(outer_test_counts[local_index]),
                            "class_mean_training_scope": "outer_training_only",
                            "class_mean_excludes_recipient": True,
                            "outer_train_overall_mean_null_prediction": overall_null,
                        }
                    )
    predictions = pd.DataFrame(prediction_rows).sort_values(
        ["representation", "repeat", "prediction_scope", "participant_id"], kind="stable"
    ).reset_index(drop=True)
    for (representation, repeat), group in predictions.loc[
        predictions["prediction_scope"].eq("outer_test")
    ].groupby(["representation", "repeat"], sort=True):
        group = group.sort_values("participant_id", kind="stable")
        summary = _recoverability_metric_summary(
            group["interviewer_logit"].to_numpy(dtype=float),
            group["label_only_prediction"].to_numpy(dtype=float),
            group["outer_train_overall_mean_null_prediction"].to_numpy(dtype=float),
        )
        for metric in ("r2", "delta_mse", "mae", "rmse", "spearman_rho"):
            metric_rows.append(
                {
                    "representation": representation,
                    "repeat": int(repeat),
                    "prediction_scope": "outer_test",
                    "metric": metric,
                    "estimate": summary[metric],
                    **summary,
                    "r2_defined": summary["r2_defined"],
                    "n_participants": int(len(group)),
                    "interpretation_scope": "label_only_common_outcome_alignment_benchmark",
                }
            )
    public_output_columns_are_safe(predictions.columns)
    return predictions, pd.DataFrame(metric_rows)


def _label_conditioned_design(
    subset: str,
    *,
    p_within: np.ndarray,
    block_matrices: Mapping[str, np.ndarray],
    block_indices: np.ndarray,
) -> np.ndarray:
    p_within = np.asarray(p_within, dtype=float).reshape(-1, 1)
    if subset == "P_within":
        return p_within
    if subset in {"Q", "R", "D"}:
        return np.asarray(block_matrices[subset], dtype=float)[block_indices]
    if subset == "P_within+Q+R+D":
        return np.column_stack(
            [
                p_within,
                np.asarray(block_matrices["Q"], dtype=float)[block_indices],
                np.asarray(block_matrices["R"], dtype=float)[block_indices],
                np.asarray(block_matrices["D"], dtype=float)[block_indices],
            ]
        )
    raise ValueError(f"unknown label-conditioned subset: {subset}")


def run_label_conditioned_recoverability(
    inputs: ExplanationInputs,
    source_cache: Mapping[tuple[str, int, int], SourceFoldCache],
    membership: pd.DataFrame,
    *,
    representations: Sequence[str],
    repeats: Sequence[int],
    inner_folds: int = INNER_FOLDS,
    alpha_grid: Sequence[float] = RIDGE_ALPHAS,
    base_seed: int = BASE_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recover interviewer-score variation after removing label alignment."""

    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for representation in representations:
        for repeat in sorted(int(value) for value in repeats):
            for fold in range(1, 6):
                source = source_cache[(representation, repeat, fold)]
                train_idx, test_idx = source.train_idx, source.test_idx
                train_labels = inputs.labels[train_idx]
                split_seed = int(base_seed) + repeat * 100_000 + fold * 1_000 + 83
                inner_splits = make_inner_splits(train_labels, n_splits=int(inner_folds), seed=split_seed)
                interviewer_class_mean_train, _ = _cross_fitted_class_mean_predictions(
                    source.interviewer_train_logit,
                    train_labels,
                    inner_splits,
                )
                participant_class_mean_train, _ = _cross_fitted_class_mean_predictions(
                    source.participant_train_logit,
                    train_labels,
                    inner_splits,
                )
                interviewer_within_train = source.interviewer_train_logit - interviewer_class_mean_train
                participant_within_train = source.participant_train_logit - participant_class_mean_train
                interviewer_class_mean_test, interviewer_test_counts = _class_mean_predictions(
                    source.interviewer_train_logit,
                    train_labels,
                    inputs.labels[test_idx],
                )
                participant_class_mean_test, _ = _class_mean_predictions(
                    source.participant_train_logit,
                    train_labels,
                    inputs.labels[test_idx],
                )
                interviewer_within_test = source.interviewer_test_logit - interviewer_class_mean_test
                participant_within_test = source.participant_test_logit - participant_class_mean_test
                null_value = float(np.mean(interviewer_within_train))
                for subset_index, subset in enumerate(LABEL_CONDITIONED_SUBSETS):
                    X_train = _label_conditioned_design(
                        subset,
                        p_within=participant_within_train,
                        block_matrices=inputs.block_matrices,
                        block_indices=train_idx,
                    )
                    X_test = _label_conditioned_design(
                        subset,
                        p_within=participant_within_test,
                        block_matrices=inputs.block_matrices,
                        block_indices=test_idx,
                    )
                    if X_train.shape[1] == 0:
                        raise ValueError(f"label-conditioned subset has no eligible features: {subset}")
                    result = fit_explanation_fold(
                        X_train,
                        interviewer_within_train,
                        X_test,
                        stratify_labels=train_labels,
                        inner_folds=inner_folds,
                        alpha_grid=alpha_grid,
                        seed=int(base_seed) + repeat * 100_000 + fold * 1_000 + subset_index,
                    )
                    for local_index, global_index in enumerate(test_idx):
                        prediction_rows.append(
                            {
                                "participant_id": int(inputs.participant_ids[global_index]),
                                "label": int(inputs.labels[global_index]),
                                "representation": representation,
                                "repeat": int(repeat),
                                "outer_fold": int(fold),
                                "subset": subset,
                                "interviewer_within": float(interviewer_within_test[local_index]),
                                "participant_within": float(participant_within_test[local_index]),
                                "predicted_interviewer_within": float(result.test_prediction[local_index]),
                                "label_conditioned_null_prediction": null_value,
                                "outer_train_class_mean_I": float(interviewer_class_mean_test[local_index]),
                                "outer_train_class_mean_P": float(participant_class_mean_test[local_index]),
                                "class_mean_training_n": int(interviewer_test_counts[local_index]),
                                "outer_test_mean_source_excludes_test": True,
                                "prediction_scope": "outer_test",
                                "selected_alpha": float(result.selected_alpha),
                            }
                        )
    predictions = pd.DataFrame(prediction_rows).sort_values(
        ["representation", "repeat", "subset", "participant_id"], kind="stable"
    ).reset_index(drop=True)
    for (representation, repeat, subset), group in predictions.groupby(
        ["representation", "repeat", "subset"], sort=True
    ):
        summary = _recoverability_metric_summary(
            group["interviewer_within"].to_numpy(dtype=float),
            group["predicted_interviewer_within"].to_numpy(dtype=float),
            group["label_conditioned_null_prediction"].to_numpy(dtype=float),
        )
        metric_rows.append(
            {
                "representation": representation,
                "repeat": int(repeat),
                "subset": subset,
                "prediction_scope": "outer_test",
                **summary,
                "interpretation_scope": "label_conditioned_recoverability",
            }
        )
    public_output_columns_are_safe(predictions.columns)
    return predictions, pd.DataFrame(metric_rows)


def add_label_conditioned_interpretation_flags(
    conditioned_metrics: pd.DataFrame,
    explanation_metrics: pd.DataFrame,
    *,
    high_r2_threshold: float = RECOVERABILITY_HIGH_R2_THRESHOLD,
    near_zero_r2_threshold: float = RECOVERABILITY_NEAR_ZERO_R2_THRESHOLD,
) -> pd.DataFrame:
    """Attach the pre-registered label-alignment interpretation flag."""

    if float(high_r2_threshold) <= 0 or float(near_zero_r2_threshold) < 0:
        raise ValueError("recoverability interpretation thresholds must be non-negative")
    result = conditioned_metrics.copy()
    if result.empty:
        result["total_recoverability_r2"] = pd.Series(dtype=float)
        result["interpretation_flag"] = pd.Series(dtype=object)
        return result
    total = explanation_metrics.loc[
        explanation_metrics["subset"].eq("P+Q+R+D"),
        ["representation", "repeat", "r2"],
    ].rename(columns={"r2": "total_recoverability_r2"})
    full_conditioned = result.loc[
        result["subset"].eq("P_within+Q+R+D"),
        ["representation", "repeat", "r2"],
    ].rename(columns={"r2": "full_label_conditioned_r2"})
    result = result.merge(total, on=["representation", "repeat"], how="left", validate="many_to_one")
    result = result.merge(full_conditioned, on=["representation", "repeat"], how="left", validate="many_to_one")

    def flag(row: pd.Series) -> str:
        total_r2 = float(row["total_recoverability_r2"])
        conditioned_r2 = float(row["full_label_conditioned_r2"])
        if not np.isfinite(total_r2) or not np.isfinite(conditioned_r2):
            return "inconclusive_label_conditioned_recoverability"
        if total_r2 >= float(high_r2_threshold) and abs(conditioned_r2) <= float(near_zero_r2_threshold):
            return "recoverability_largely_shared_outcome_alignment"
        if conditioned_r2 > float(near_zero_r2_threshold):
            return "observable_blocks_reconstructed_interviewer_score_variation_beyond_binary_label_alignment"
        return "inconclusive_label_conditioned_recoverability"

    result["interpretation_flag"] = result.apply(flag, axis=1)
    result["high_total_r2_threshold"] = float(high_r2_threshold)
    result["near_zero_conditioned_r2_threshold"] = float(near_zero_r2_threshold)
    return result


def _shapley_from_prediction_groups(
    groups: Mapping[str, pd.DataFrame],
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for representation, frame in groups.items():
        subset_frames = {
            subset: frame.loc[frame["subset"].eq(subset)].sort_values("participant_id", kind="stable")
            for subset in ALL_BLOCK_SUBSETS
        }
        participant_ids = subset_frames["null"]["participant_id"].to_numpy()
        for subset, subset_frame in subset_frames.items():
            if not np.array_equal(subset_frame["participant_id"].to_numpy(), participant_ids):
                raise ValueError(f"Shapley subset participant alignment failed for {representation}/{subset}")
        target = subset_frames["null"]["interviewer_logit"].to_numpy(dtype=float)
        null_prediction = subset_frames["null"]["outer_train_mean_null_prediction"].to_numpy(dtype=float)
        observed_values = {
            subset: relative_r2(
                target,
                subset_frames[subset]["predicted_interviewer_logit"].to_numpy(dtype=float),
                null_prediction,
            )
            for subset in ALL_BLOCK_SUBSETS
        }
        estimate = compute_shapley_values(observed_values)
        bootstrap_indices = joint_bootstrap_indices(
            len(participant_ids), n_bootstrap=n_bootstrap, seed=seed + _stable_seed_offset(representation)
        )
        bootstrap_values = {block: [] for block in BLOCK_ORDER}
        for index in bootstrap_indices:
            sampled_values = {
                subset: relative_r2(
                    target[index],
                    subset_frames[subset]["predicted_interviewer_logit"].to_numpy(dtype=float)[index],
                    null_prediction[index],
                )
                for subset in ALL_BLOCK_SUBSETS
            }
            sample_shapley = compute_shapley_values(sampled_values)
            for block in BLOCK_ORDER:
                bootstrap_values[block].append(sample_shapley[block])
        for block in BLOCK_ORDER:
            distribution = np.asarray(bootstrap_values[block], dtype=float)
            rows.append(
                {
                    "representation": representation,
                    "block": block,
                    "estimate_delta_r2": float(estimate[block]),
                    "ci_low": float(np.quantile(distribution, 0.025)),
                    "ci_high": float(np.quantile(distribution, 0.975)),
                    "n_bootstrap": int(n_bootstrap),
                    "interpretation_scope": "statistical_explanation_not_causal_contribution",
                }
            )
    return pd.DataFrame(rows)


def run_block_shapley(
    explanation_predictions: pd.DataFrame,
    *,
    main_repeat: int,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BASE_SEED,
) -> pd.DataFrame:
    main = explanation_predictions.loc[explanation_predictions["repeat"].eq(int(main_repeat))].copy()
    groups = {
        str(representation): group
        for representation, group in main.groupby("representation", sort=True)
    }
    return _shapley_from_prediction_groups(groups, n_bootstrap=n_bootstrap, seed=seed)


def make_label_pipeline(*, C: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(C),
                    class_weight="balanced",
                    solver="liblinear",
                    fit_intercept=True,
                    max_iter=2000,
                    random_state=int(seed),
                ),
            ),
        ]
    )


def tune_label_C(
    X_train: np.ndarray,
    labels: np.ndarray,
    inner_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    c_grid: Sequence[float] = LABEL_C_GRID,
    seed: int,
) -> tuple[float, dict[float, float]]:
    X_train = np.asarray(X_train, dtype=float)
    labels = np.asarray(labels, dtype=int)
    scores: dict[float, float] = {}
    for C in sorted(float(value) for value in c_grid):
        fold_scores: list[float] = []
        for inner_fold, (train_idx, validation_idx) in enumerate(inner_splits, start=1):
            estimator = make_label_pipeline(C=C, seed=int(seed) + inner_fold)
            estimator.fit(X_train[train_idx], labels[train_idx])
            probability = estimator.predict_proba(X_train[validation_idx])[:, 1]
            fold_scores.append(float(roc_auc_score(labels[validation_idx], probability)))
        scores[C] = float(np.mean(fold_scores))
    # A smaller C is the stronger-regularization tie breaker.
    selected = min(scores, key=lambda value: (-scores[value], value))
    return float(selected), scores


def fit_label_outer(
    X_train: np.ndarray,
    labels_train: np.ndarray,
    X_test: np.ndarray,
    *,
    inner_folds: int,
    c_grid: Sequence[float] = LABEL_C_GRID,
    seed: int,
) -> tuple[np.ndarray, float, dict[float, float]]:
    labels_train = np.asarray(labels_train, dtype=int)
    inner_splits = make_inner_splits(labels_train, n_splits=int(inner_folds), seed=int(seed))
    selected, scores = tune_label_C(
        X_train,
        labels_train,
        inner_splits,
        c_grid=c_grid,
        seed=int(seed) + 101,
    )
    estimator = make_label_pipeline(C=selected, seed=int(seed) + 202)
    estimator.fit(np.asarray(X_train, dtype=float), labels_train)
    probability = estimator.predict_proba(np.asarray(X_test, dtype=float))[:, 1]
    return np.asarray(probability, dtype=float), selected, scores


def _classification_metrics(
    labels: Sequence[int], probability: Sequence[float]
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probability = np.asarray(probability, dtype=float)
    return {
        "roc_auc": float(roc_auc_score(labels, probability)),
        "pr_auc": float(average_precision_score(labels, probability)),
        "brier": float(brier_score_loss(labels, probability)),
        "log_loss": float(log_loss(labels, probability, labels=[0, 1])),
        "sensitivity_at_0_50": float(np.sum((probability >= 0.5) & (labels == 1)) / max(np.sum(labels == 1), 1)),
        "specificity_at_0_50": float(np.sum((probability < 0.5) & (labels == 0)) / max(np.sum(labels == 0), 1)),
    }


def _paired_probability_delta_metrics(
    labels: np.ndarray,
    base_probability: np.ndarray,
    enhanced_probability: np.ndarray,
) -> dict[str, float]:
    base_metrics = _classification_metrics(labels, base_probability)
    enhanced_metrics = _classification_metrics(labels, enhanced_probability)
    return {
        "base_roc_auc": base_metrics["roc_auc"],
        "enhanced_roc_auc": enhanced_metrics["roc_auc"],
        "base_pr_auc": base_metrics["pr_auc"],
        "enhanced_pr_auc": enhanced_metrics["pr_auc"],
        "base_brier": base_metrics["brier"],
        "enhanced_brier": enhanced_metrics["brier"],
        "base_log_loss": base_metrics["log_loss"],
        "enhanced_log_loss": enhanced_metrics["log_loss"],
        "delta_auc": enhanced_metrics["roc_auc"] - base_metrics["roc_auc"],
        "delta_pr_auc": enhanced_metrics["pr_auc"] - base_metrics["pr_auc"],
        "delta_brier": base_metrics["brier"] - enhanced_metrics["brier"],
        "delta_log_loss": base_metrics["log_loss"] - enhanced_metrics["log_loss"],
    }


def run_residual_signal(
    inputs: ExplanationInputs,
    source_cache: Mapping[tuple[str, int, int], SourceFoldCache],
    full_explanation_cache: Mapping[tuple[str, int, int], ExplanationFoldResult],
    membership: pd.DataFrame,
    *,
    representations: Sequence[str],
    repeats: Sequence[int],
    inner_folds: int = INNER_FOLDS,
    c_grid: Sequence[float] = LABEL_C_GRID,
    base_seed: int = BASE_SEED,
    main_repeat: int = 1,
    n_bootstrap: int = N_BOOTSTRAP,
    n_permutations: int = N_PERMUTATIONS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Test whether the unexplained interviewer logit retains label value."""

    prediction_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    for representation in representations:
        for repeat in sorted(int(value) for value in repeats):
            for fold in range(1, 6):
                key = (representation, repeat, fold)
                source = source_cache[key]
                explanation = full_explanation_cache[key]
                train_idx, test_idx = source.train_idx, source.test_idx
                residual_train = source.interviewer_train_logit - explanation.train_prediction
                residual_test = source.interviewer_test_logit - explanation.test_prediction
                X_base_train = _subset_design(
                    "P+Q+R+D",
                    participant_logits=source.participant_train_logit,
                    block_matrices=inputs.block_matrices,
                    indices=train_idx,
                )
                X_base_test = _subset_design(
                    "P+Q+R+D",
                    participant_logits=source.participant_test_logit,
                    block_matrices=inputs.block_matrices,
                    indices=test_idx,
                )
                X_enhanced_train = np.column_stack([X_base_train, residual_train])
                X_enhanced_test = np.column_stack([X_base_test, residual_test])
                base_probability, base_c, base_scores = fit_label_outer(
                    X_base_train,
                    inputs.labels[train_idx],
                    X_base_test,
                    inner_folds=inner_folds,
                    c_grid=c_grid,
                    seed=int(base_seed) + repeat * 100_000 + fold * 1_000 + 1,
                )
                enhanced_probability, enhanced_c, enhanced_scores = fit_label_outer(
                    X_enhanced_train,
                    inputs.labels[train_idx],
                    X_enhanced_test,
                    inner_folds=inner_folds,
                    c_grid=c_grid,
                    seed=int(base_seed) + repeat * 100_000 + fold * 1_000 + 2,
                )
                for model_name, selected, scores in (
                    ("base", base_c, base_scores),
                    ("enhanced", enhanced_c, enhanced_scores),
                ):
                    for C, score in sorted(scores.items()):
                        tuning_rows.append(
                            {
                                "representation": representation,
                                "repeat": int(repeat),
                                "outer_fold": int(fold),
                                "model_name": model_name,
                                "C": float(C),
                                "mean_inner_roc_auc": float(score),
                                "selected": bool(C == selected),
                            }
                        )
                for local_index, global_index in enumerate(test_idx):
                    prediction_rows.append(
                        {
                            "participant_id": int(inputs.participant_ids[global_index]),
                            "label": int(inputs.labels[global_index]),
                            "representation": representation,
                            "repeat": int(repeat),
                            "outer_fold": int(fold),
                            "interviewer_logit": float(source.interviewer_test_logit[local_index]),
                            "predicted_interviewer_logit": float(explanation.test_prediction[local_index]),
                            "interviewer_residual": float(residual_test[local_index]),
                            "base_probability": float(base_probability[local_index]),
                            "enhanced_probability": float(enhanced_probability[local_index]),
                            "base_selected_C": float(base_c),
                            "enhanced_selected_C": float(enhanced_c),
                        }
                    )
    predictions = pd.DataFrame(prediction_rows).sort_values(
        ["representation", "repeat", "participant_id"], kind="stable"
    ).reset_index(drop=True)
    metric_rows: list[dict[str, object]] = []
    delta_rows: list[dict[str, object]] = []
    for (representation, repeat), group in predictions.groupby(["representation", "repeat"], sort=True):
        labels = group["label"].to_numpy(dtype=int)
        base_probability = group["base_probability"].to_numpy(dtype=float)
        enhanced_probability = group["enhanced_probability"].to_numpy(dtype=float)
        base_metrics = _classification_metrics(labels, base_probability)
        enhanced_metrics = _classification_metrics(labels, enhanced_probability)
        metric_rows.extend(
            {
                "representation": representation,
                "repeat": int(repeat),
                "model_name": model_name,
                **metrics,
                "interpretation_scope": "residual_label_increment",
            }
            for model_name, metrics in (("base", base_metrics), ("enhanced", enhanced_metrics))
        )
        delta = _paired_probability_delta_metrics(labels, base_probability, enhanced_probability)
        if int(repeat) == int(main_repeat):
            delta.update(
                paired_auc_delta_inference(
                    labels,
                    base_probability,
                    enhanced_probability,
                    n_bootstrap=n_bootstrap,
                    n_permutations=n_permutations,
                    seed=int(base_seed) + 800_000 + _stable_seed_offset(representation),
                )
            )
        else:
            delta.update(
                {
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "n_bootstrap": 0,
                    "n_permutations": 0,
                    "p_value": float("nan"),
                }
            )
        delta.update(
            {
                "representation": representation,
                "repeat": int(repeat),
                "interpretation_scope": "residual_signal_not_causal_mediation",
            }
        )
        delta_rows.append(delta)
    public_output_columns_are_safe(predictions.columns)
    return predictions, pd.DataFrame(metric_rows), pd.DataFrame(delta_rows)


def _fake_result_metrics(
    target_rows: Sequence[dict[str, object]],
    *,
    representation: str,
    repeat: int,
    draw: int,
    domain_type: str,
) -> dict[str, object]:
    frame = pd.DataFrame(target_rows)
    metrics = explanation_metrics(
        frame["interviewer_logit"].to_numpy(dtype=float),
        frame["predicted_interviewer_logit"].to_numpy(dtype=float),
        frame["outer_train_mean_null_prediction"].to_numpy(dtype=float),
    )
    return {
        "representation": representation,
        "repeat": int(repeat),
        "draw": int(draw),
        "domain_type": domain_type,
        **metrics,
        "interpretation_scope": "real_D_vs_fake_D_negative_control",
    }


def run_fake_domain_controls(
    inputs: ExplanationInputs,
    source_cache: Mapping[tuple[str, int, int], SourceFoldCache],
    membership: pd.DataFrame,
    *,
    representations: Sequence[str],
    main_repeat: int,
    repeats: Sequence[int] | None = None,
    n_draws: int = FAKE_D_DRAWS,
    inner_folds: int = INNER_FOLDS,
    label_c_grid: Sequence[float] = LABEL_C_GRID,
    alpha_grid: Sequence[float] = RIDGE_ALPHAS,
    base_seed: int = BASE_SEED,
    return_oof: bool = False,
) -> tuple[pd.DataFrame, ...]:
    """Run intact-row Fake-D explanation and residual negative controls.

    The main repeat is used for primary inference, while ``repeats`` controls
    the repeat-stability rows.  Every repeat receives its own real-D baseline
    and its own partition-local fake-D draws.
    """

    selected_repeats = tuple(sorted(int(value) for value in (repeats if repeats is not None else (main_repeat,))))
    if not selected_repeats or int(main_repeat) not in selected_repeats:
        raise ValueError("Fake-D repeats must be non-empty and include main_repeat")
    assignment_rows: list[dict[str, object]] = []
    explanation_rows: list[dict[str, object]] = []
    residual_rows: list[dict[str, object]] = []
    explanation_oof_rows: list[dict[str, object]] = []
    residual_oof_rows: list[dict[str, object]] = []
    real_explanation_rows: dict[tuple[str, int], list[dict[str, object]]] = {
        (rep, repeat): [] for rep in representations for repeat in selected_repeats
    }
    real_residual_rows: dict[tuple[str, int], list[dict[str, object]]] = {
        (rep, repeat): [] for rep in representations for repeat in selected_repeats
    }
    for representation in representations:
        for repeat in selected_repeats:
            repeat_offset = int(repeat) * 100_000
            for fold in range(1, 6):
                source = source_cache[(representation, repeat, fold)]
                train_idx, test_idx = source.train_idx, source.test_idx
                target_train = source.interviewer_train_logit
                target_test = source.interviewer_test_logit
                null_value = float(np.mean(target_train))
                X_real_train = _subset_design(
                    "P+Q+R+D",
                    participant_logits=source.participant_train_logit,
                    block_matrices=inputs.block_matrices,
                    indices=train_idx,
                )
                X_real_test = _subset_design(
                    "P+Q+R+D",
                    participant_logits=source.participant_test_logit,
                    block_matrices=inputs.block_matrices,
                    indices=test_idx,
                )
                real_explanation = fit_explanation_fold(
                    X_real_train,
                    target_train,
                    X_real_test,
                    stratify_labels=inputs.labels[train_idx],
                    inner_folds=inner_folds,
                    alpha_grid=alpha_grid,
                    seed=int(base_seed) + 900_000 + repeat_offset + fold,
                )
                for local_index, global_index in enumerate(test_idx):
                    real_explanation_rows[(representation, repeat)].append(
                        {
                            "participant_id": int(inputs.participant_ids[global_index]),
                            "label": int(inputs.labels[global_index]),
                            "interviewer_logit": float(target_test[local_index]),
                            "predicted_interviewer_logit": float(real_explanation.test_prediction[local_index]),
                            "outer_train_mean_null_prediction": null_value,
                        }
                    )
                    explanation_oof_rows.append(
                        {
                            "representation": representation,
                            "repeat": int(repeat),
                            "draw": 0,
                            "domain_type": "real",
                            "participant_id": int(inputs.participant_ids[global_index]),
                            "label": int(inputs.labels[global_index]),
                            "interviewer_logit": float(target_test[local_index]),
                            "predicted_interviewer_logit": float(real_explanation.test_prediction[local_index]),
                            "outer_train_mean_null_prediction": null_value,
                        }
                    )
                real_residual_train = target_train - real_explanation.train_prediction
                real_residual_test = target_test - real_explanation.test_prediction
                real_base_probability, _, _ = fit_label_outer(
                    X_real_train,
                    inputs.labels[train_idx],
                    X_real_test,
                    inner_folds=inner_folds,
                    c_grid=label_c_grid,
                    seed=int(base_seed) + 910_000 + repeat_offset + fold,
                )
                real_enhanced_probability, _, _ = fit_label_outer(
                    np.column_stack([X_real_train, real_residual_train]),
                    inputs.labels[train_idx],
                    np.column_stack([X_real_test, real_residual_test]),
                    inner_folds=inner_folds,
                    c_grid=label_c_grid,
                    seed=int(base_seed) + 920_000 + repeat_offset + fold,
                )
                for local_index, global_index in enumerate(test_idx):
                    residual_row = {
                        "participant_id": int(inputs.participant_ids[global_index]),
                        "label": int(inputs.labels[global_index]),
                        "base_probability": float(real_base_probability[local_index]),
                        "enhanced_probability": float(real_enhanced_probability[local_index]),
                    }
                    real_residual_rows[(representation, repeat)].append(residual_row)
                    residual_oof_rows.append(
                        {
                            "representation": representation,
                            "repeat": int(repeat),
                            "draw": 0,
                            "domain_type": "real",
                            **residual_row,
                        }
                    )

            for draw in range(1, int(n_draws) + 1):
                draw_explanation_rows: list[dict[str, object]] = []
                draw_residual_rows: list[dict[str, object]] = []
                for fold in range(1, 6):
                    source = source_cache[(representation, repeat, fold)]
                    train_idx, test_idx = source.train_idx, source.test_idx
                    fake_domains_full, ledger = apply_fake_domain_permutation(
                        participant_ids=inputs.participant_ids,
                        domains=inputs.domains[list(D_P_FEATURES)].to_numpy(dtype=float),
                        train_idx=train_idx,
                        test_idx=test_idx,
                        draw=draw,
                        seed=int(base_seed) + repeat_offset + 100_000 * (representations.index(representation) + 1),
                    )
                    ledger.insert(0, "representation", representation)
                    ledger.insert(1, "repeat", int(repeat))
                    ledger.insert(2, "outer_fold", int(fold))
                    assignment_rows.extend(ledger.to_dict("records"))
                    fake_blocks = dict(inputs.block_matrices)
                    if all(column in D_P_FEATURES for column in inputs.block_columns["D"]):
                        eligible_d_indices = [D_P_FEATURES.index(column) for column in inputs.block_columns["D"]]
                    else:
                        eligible_d_indices = list(range(inputs.block_matrices["D"].shape[1]))
                    fake_blocks["D"] = fake_domains_full[:, eligible_d_indices]
                    X_train = _subset_design(
                        "P+Q+R+D",
                        participant_logits=source.participant_train_logit,
                        block_matrices=fake_blocks,
                        indices=train_idx,
                    )
                    X_test = _subset_design(
                        "P+Q+R+D",
                        participant_logits=source.participant_test_logit,
                        block_matrices=fake_blocks,
                        indices=test_idx,
                    )
                    explanation = fit_explanation_fold(
                        X_train,
                        source.interviewer_train_logit,
                        X_test,
                        stratify_labels=inputs.labels[train_idx],
                        inner_folds=inner_folds,
                        alpha_grid=alpha_grid,
                        seed=int(base_seed) + 930_000 + repeat_offset + draw * 1_000 + fold,
                    )
                    null_value = float(np.mean(source.interviewer_train_logit))
                    for local_index, global_index in enumerate(test_idx):
                        draw_explanation_rows.append(
                            {
                                "participant_id": int(inputs.participant_ids[global_index]),
                                "label": int(inputs.labels[global_index]),
                                "interviewer_logit": float(source.interviewer_test_logit[local_index]),
                                "predicted_interviewer_logit": float(explanation.test_prediction[local_index]),
                                "outer_train_mean_null_prediction": null_value,
                            }
                        )
                    residual_train = source.interviewer_train_logit - explanation.train_prediction
                    residual_test = source.interviewer_test_logit - explanation.test_prediction
                    base_probability, _, _ = fit_label_outer(
                        X_train,
                        inputs.labels[train_idx],
                        X_test,
                        inner_folds=inner_folds,
                        c_grid=label_c_grid,
                        seed=int(base_seed) + 940_000 + repeat_offset + draw * 1_000 + fold,
                    )
                    enhanced_probability, _, _ = fit_label_outer(
                        np.column_stack([X_train, residual_train]),
                        inputs.labels[train_idx],
                        np.column_stack([X_test, residual_test]),
                        inner_folds=inner_folds,
                        c_grid=label_c_grid,
                        seed=int(base_seed) + 950_000 + repeat_offset + draw * 1_000 + fold,
                    )
                    for local_index, global_index in enumerate(test_idx):
                        draw_residual_rows.append(
                            {
                                "participant_id": int(inputs.participant_ids[global_index]),
                                "label": int(inputs.labels[global_index]),
                                "base_probability": float(base_probability[local_index]),
                                "enhanced_probability": float(enhanced_probability[local_index]),
                            }
                        )
                explanation_oof_rows.extend(
                    {
                        "representation": representation,
                        "repeat": int(repeat),
                        "draw": int(draw),
                        "domain_type": "fake",
                        **row,
                    }
                    for row in draw_explanation_rows
                )
                residual_oof_rows.extend(
                    {
                        "representation": representation,
                        "repeat": int(repeat),
                        "draw": int(draw),
                        "domain_type": "fake",
                        **row,
                    }
                    for row in draw_residual_rows
                )
                explanation_rows.append(
                    _fake_result_metrics(
                        draw_explanation_rows,
                        representation=representation,
                        repeat=repeat,
                        draw=draw,
                        domain_type="fake",
                    )
                )
                draw_residual_frame = pd.DataFrame(draw_residual_rows).sort_values("participant_id", kind="stable")
                residual_metric = _paired_probability_delta_metrics(
                    draw_residual_frame["label"].to_numpy(dtype=int),
                    draw_residual_frame["base_probability"].to_numpy(dtype=float),
                    draw_residual_frame["enhanced_probability"].to_numpy(dtype=float),
                )
                residual_rows.append(
                    {
                        "representation": representation,
                        "repeat": int(repeat),
                        "draw": int(draw),
                        "domain_type": "fake",
                        **residual_metric,
                        "interpretation_scope": "real_D_vs_fake_D_negative_control",
                    }
                )
            explanation_rows.append(
                _fake_result_metrics(
                    real_explanation_rows[(representation, repeat)],
                    representation=representation,
                    repeat=repeat,
                    draw=0,
                    domain_type="real",
                )
            )
            real_residual_frame = pd.DataFrame(real_residual_rows[(representation, repeat)]).sort_values("participant_id", kind="stable")
            real_residual_metric = _paired_probability_delta_metrics(
                real_residual_frame["label"].to_numpy(dtype=int),
                real_residual_frame["base_probability"].to_numpy(dtype=float),
                real_residual_frame["enhanced_probability"].to_numpy(dtype=float),
            )
            residual_rows.append(
                {
                    "representation": representation,
                    "repeat": int(repeat),
                    "draw": 0,
                    "domain_type": "real",
                    **real_residual_metric,
                    "interpretation_scope": "real_D_vs_fake_D_negative_control",
                }
            )
    result = (pd.DataFrame(assignment_rows), pd.DataFrame(explanation_rows), pd.DataFrame(residual_rows))
    if return_oof:
        return (*result, pd.DataFrame(explanation_oof_rows), pd.DataFrame(residual_oof_rows))
    return result


def build_pairing_match_frame(inputs: ExplanationInputs) -> pd.DataFrame:
    """Build the pre-locked, label-free pairing vector."""

    structural = inputs.structural.set_index("participant_id")
    domains = inputs.domains.set_index("participant_id")[list(D_P_FEATURES)]
    frame = pd.DataFrame({"participant_id": inputs.participant_ids})
    frame["interviewer_word_count"] = structural.loc[inputs.participant_ids, "length_only__interviewer_word_count"].to_numpy(dtype=float)
    frame["interviewer_turn_count"] = structural.loc[inputs.participant_ids, "interaction_structure__interviewer_turn_count"].to_numpy(dtype=float)
    frame["clinical_question_count"] = structural.loc[inputs.participant_ids, "protocol_structure__clinical_question_count"].to_numpy(dtype=float)
    frame["clinical_prompt_coverage_count"] = structural.loc[inputs.participant_ids, "protocol_structure__clinical_prompt_coverage_count"].to_numpy(dtype=float)
    frame["non_explicit_interviewer_turn_count"] = structural.loc[inputs.participant_ids, "protocol_structure__non_explicit_interviewer_turn_count"].to_numpy(dtype=float)
    frame["d_p_domain_breadth"] = (domains.to_numpy(dtype=float) > 0).sum(axis=1).astype(float)
    frame["d_p_total_evidence_count"] = domains.to_numpy(dtype=float).sum(axis=1)
    return frame


def pairing_balance_rows(
    *,
    recipient: pd.DataFrame,
    donor: pd.DataFrame,
    assignment: pd.DataFrame,
    representation: str,
    repeat: int,
    outer_fold: int,
    draw: int,
    partition: str,
    fit_reference: pd.DataFrame,
    assignment_type: str = "optimal_matched",
    random_distance_reference: Sequence[float] | None = None,
) -> list[dict[str, object]]:
    """Report pairwise balance; marginal SMD is descriptive only."""

    quality = pairing_quality_summary(
        recipient=recipient,
        donor=donor,
        assignment=assignment,
        fit_reference=fit_reference,
        random_distance_reference=random_distance_reference,
        assignment_type=assignment_type,
    )
    recipient_lookup = recipient.set_index("participant_id")
    donor_lookup = donor.set_index("participant_id")
    ordered_recipient = recipient_lookup.loc[assignment["recipient_id"].tolist(), list(PAIRING_MATCH_FEATURES)]
    ordered_donor = donor_lookup.loc[assignment["donor_id"].tolist(), list(PAIRING_MATCH_FEATURES)]
    _, _, scale = _pairing_z_values(recipient, donor, fit_reference)
    signed_difference = (ordered_recipient.to_numpy(dtype=float) - ordered_donor.to_numpy(dtype=float)) / scale
    absolute_difference = np.abs(signed_difference)
    feature_means = quality["feature_mean_abs_standardized_paired_difference"]
    feature_medians = quality["feature_median_abs_standardized_paired_difference"]
    feature_q95 = quality["feature_q95_abs_standardized_paired_difference"]
    rows = []
    for index, feature in enumerate(PAIRING_MATCH_FEATURES):
        rows.append(
            {
                "representation": representation,
                "repeat": int(repeat),
                "outer_fold": int(outer_fold),
                "draw": int(draw),
                "partition": partition,
                "feature": feature,
                "assignment_type": assignment_type,
                "mean_signed_standardized_paired_difference": float(signed_difference[:, index].mean()),
                "mean_abs_standardized_paired_difference": float(feature_means[feature]),
                "median_abs_standardized_paired_difference": float(feature_medians[feature]),
                "q95_abs_standardized_paired_difference": float(feature_q95[feature]),
                "before_standardized_mean_difference": float(
                    (recipient[list(PAIRING_MATCH_FEATURES)].to_numpy(dtype=float).mean(axis=0)[index]
                     - donor[list(PAIRING_MATCH_FEATURES)].to_numpy(dtype=float).mean(axis=0)[index]) / scale[index]
                ),
                "after_standardized_mean_difference": float(signed_difference[:, index].mean()),
                "max_abs_smd": float(quality["max_abs_smd"]),
                "mean_distance": float(quality["mean_distance"]),
                "median_distance": float(quality["median_distance"]),
                "q95_distance": float(quality["q95_distance"]),
                "max_distance": float(quality["max_distance"]),
                "total_cost": float(quality["total_cost"]),
                "random_distance_q05": float(quality["random_distance_q05"]),
                "optimal_distance_random_percentile": float(quality["optimal_distance_random_percentile"]),
                "n_features_mean_abs_paired_diff_lt_0_50": int(quality["n_features_mean_abs_paired_diff_lt_0_50"]),
                "matching_adequate": bool(quality["matching_adequate"]),
                "self_match_count": int(assignment["self_match"].sum()),
                "max_donor_reuse_count": int(assignment["donor_reuse_count"].max()),
                "one_to_one": bool(assignment["one_to_one"].all()),
            }
        )
    return rows


def pairing_pairwise_rows(
    *,
    recipient: pd.DataFrame,
    donor: pd.DataFrame,
    assignment: pd.DataFrame,
    fit_reference: pd.DataFrame,
    representation: str,
    repeat: int,
    outer_fold: int,
    draw: int,
    partition: str,
    assignment_type: str,
) -> list[dict[str, object]]:
    """Return one auditable row per recipient-donor pair."""

    recipient_lookup = recipient.set_index("participant_id")
    donor_lookup = donor.set_index("participant_id")
    _, _, scale = _pairing_z_values(recipient, donor, fit_reference)
    rows: list[dict[str, object]] = []
    for assignment_row in assignment.to_dict("records"):
        recipient_values = recipient_lookup.loc[assignment_row["recipient_id"], list(PAIRING_MATCH_FEATURES)].to_numpy(dtype=float)
        donor_values = donor_lookup.loc[assignment_row["donor_id"], list(PAIRING_MATCH_FEATURES)].to_numpy(dtype=float)
        standardized_difference = (recipient_values - donor_values) / scale
        row: dict[str, object] = {
            "record_type": "pairwise",
            "representation": representation,
            "repeat": int(repeat),
            "outer_fold": int(outer_fold),
            "draw": int(draw),
            "partition": partition,
            "assignment_type": assignment_type,
            "recipient_id": assignment_row["recipient_id"],
            "donor_id": assignment_row["donor_id"],
            "self_match": bool(assignment_row.get("self_match", False)),
            "one_to_one": bool(assignment_row.get("one_to_one", False)),
            "standardized_euclidean_distance": float(np.sqrt(np.square(standardized_difference).sum())),
        }
        row.update(
            {
                f"abs_standardized_pair_difference__{feature}": float(abs(standardized_difference[index]))
                for index, feature in enumerate(PAIRING_MATCH_FEATURES)
            }
        )
        rows.append(row)
    return rows


def summarise_inner_training_quality(quality: pd.DataFrame) -> dict[str, object]:
    """Apply the locked 80% inner-training matching-quality threshold."""

    required = {"assignment_type", "matching_adequate", "no_self", "one_to_one"}
    missing = sorted(required.difference(quality.columns))
    if missing:
        raise ValueError(f"inner matching quality is missing columns: {missing}")
    all_assignments = quality.copy()
    optimal = all_assignments.loc[all_assignments["assignment_type"].eq("optimal_matched")].copy()
    if optimal.empty:
        return {
            "n_optimal_inner_assignments": 0,
            "n_adequate_optimal_inner_assignments": 0,
            "pass_rate": 0.0,
            "all_no_self": bool(all_assignments["no_self"].astype(bool).all()) if not all_assignments.empty else False,
            "all_one_to_one": bool(all_assignments["one_to_one"].astype(bool).all()) if not all_assignments.empty else False,
            "training_match_quality_limited": True,
            "interpretation_scope": "inner_training_pairing_quality_gate",
        }
    pass_rate = float(optimal["matching_adequate"].astype(bool).mean())
    all_no_self = bool(all_assignments["no_self"].astype(bool).all())
    all_one_to_one = bool(all_assignments["one_to_one"].astype(bool).all())
    return {
        "n_optimal_inner_assignments": int(len(optimal)),
        "n_adequate_optimal_inner_assignments": int(optimal["matching_adequate"].astype(bool).sum()),
        "pass_rate": pass_rate,
        "all_no_self": all_no_self,
        "all_one_to_one": all_one_to_one,
        "training_match_quality_limited": bool(pass_rate < 0.80 or not all_no_self or not all_one_to_one),
        "interpretation_scope": "inner_training_pairing_quality_gate",
    }


def _pairing_draw_inference(
    predictions: pd.DataFrame,
    deltas: pd.DataFrame,
    *,
    main_repeat: int,
    n_bootstrap: int,
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    if int(n_bootstrap) <= 0 or int(n_permutations) <= 0:
        raise ValueError("pairing inference resample counts must be positive")
    rows: list[dict[str, object]] = []
    for (representation, repeat), group in predictions.groupby(["representation", "repeat"], sort=True):
        if int(repeat) != int(main_repeat):
            continue
        optimal = group.loc[group["draw_type"].eq("optimal_matched")]
        pivot = optimal.pivot_table(
            index=["participant_id", "label"], columns="model_name", values="probability"
        ).reset_index()
        required = {"P", "P+I-real", "P+I-optimal_matched", "full", "full+I-real", "full+I-optimal_matched"}
        if not required.issubset(pivot.columns):
            raise ValueError(f"pairing predictions are missing model columns: {sorted(required - set(pivot.columns))}")
        pivot = pivot.sort_values("participant_id", kind="stable")
        labels = pivot["label"].to_numpy(dtype=int)
        positive = np.flatnonzero(labels == 1)
        negative = np.flatnonzero(labels == 0)
        rng = np.random.default_rng(int(seed) + _stable_seed_offset(representation))
        bootstrap_full_real_minus_optimal: list[float] = []
        bootstrap_full_optimal_minus_no_i: list[float] = []
        bootstrap_weak_real_minus_optimal: list[float] = []
        bootstrap_weak_optimal_minus_no_i: list[float] = []
        weak_real = pivot["P+I-real"].to_numpy(dtype=float)
        weak_optimal = pivot["P+I-optimal_matched"].to_numpy(dtype=float)
        weak_no_i = pivot["P"].to_numpy(dtype=float)
        full_real = pivot["full+I-real"].to_numpy(dtype=float)
        full_optimal = pivot["full+I-optimal_matched"].to_numpy(dtype=float)
        full_no_i = pivot["full"].to_numpy(dtype=float)
        for _ in range(int(n_bootstrap)):
            index = np.concatenate(
                [
                    rng.choice(positive, size=len(positive), replace=True),
                    rng.choice(negative, size=len(negative), replace=True),
                ]
            )
            weak_base_auc = roc_auc_score(labels[index], weak_no_i[index])
            full_base_auc = roc_auc_score(labels[index], full_no_i[index])
            weak_optimal_delta = roc_auc_score(labels[index], weak_optimal[index]) - weak_base_auc
            full_optimal_delta = roc_auc_score(labels[index], full_optimal[index]) - full_base_auc
            bootstrap_weak_real_minus_optimal.append(
                (roc_auc_score(labels[index], weak_real[index]) - weak_base_auc) - weak_optimal_delta
            )
            bootstrap_weak_optimal_minus_no_i.append(weak_optimal_delta)
            bootstrap_full_real_minus_optimal.append(
                (roc_auc_score(labels[index], full_real[index]) - full_base_auc) - full_optimal_delta
            )
            bootstrap_full_optimal_minus_no_i.append(full_optimal_delta)

        observed_row = deltas.loc[
            deltas["representation"].eq(representation)
            & deltas["repeat"].eq(int(repeat))
            & deltas["draw_type"].eq("optimal_matched")
        ]
        if len(observed_row) != 1:
            raise ValueError(f"pairing delta table must contain one optimal row for {representation}/{repeat}")
        observed_row = observed_row.iloc[0]
        observed_weak = float(observed_row["weak_real_delta_auc"] - observed_row["weak_matched_delta_auc"])
        observed_weak_optimal = float(observed_row["weak_matched_delta_auc"])
        observed_full = float(observed_row["full_real_delta_auc"] - observed_row["full_matched_delta_auc"])
        observed_full_optimal = float(observed_row["full_matched_delta_auc"])

        weak_extreme = 0
        full_extreme = 0
        for _ in range(int(n_permutations)):
            swap = rng.integers(0, 2, size=len(labels), dtype=np.int8).astype(bool)
            weak_left = np.where(swap, weak_optimal, weak_real)
            weak_right = np.where(swap, weak_real, weak_optimal)
            full_left = np.where(swap, full_optimal, full_real)
            full_right = np.where(swap, full_real, full_optimal)
            weak_statistic = roc_auc_score(labels, weak_left) - roc_auc_score(labels, weak_right)
            full_statistic = roc_auc_score(labels, full_left) - roc_auc_score(labels, full_right)
            weak_extreme += int(abs(weak_statistic) >= abs(observed_weak) - 1e-15)
            full_extreme += int(abs(full_statistic) >= abs(observed_full) - 1e-15)
        weak_p_value = float((weak_extreme + 1) / (int(n_permutations) + 1))
        full_p_value = float((full_extreme + 1) / (int(n_permutations) + 1))
        rows.append(
            {
                "representation": representation,
                "repeat": int(repeat),
                "draw": 0,
                "full_real_minus_optimal_delta_auc": observed_full,
                "full_real_minus_optimal_ci_low": float(np.quantile(bootstrap_full_real_minus_optimal, 0.025)),
                "full_real_minus_optimal_ci_high": float(np.quantile(bootstrap_full_real_minus_optimal, 0.975)),
                "full_real_minus_optimal_p_value": full_p_value,
                "full_optimal_minus_no_i_delta_auc": observed_full_optimal,
                "full_optimal_minus_no_i_ci_low": float(np.quantile(bootstrap_full_optimal_minus_no_i, 0.025)),
                "full_optimal_minus_no_i_ci_high": float(np.quantile(bootstrap_full_optimal_minus_no_i, 0.975)),
                "weak_real_minus_optimal_delta_auc": observed_weak,
                "weak_real_minus_optimal_ci_low": float(np.quantile(bootstrap_weak_real_minus_optimal, 0.025)),
                "weak_real_minus_optimal_ci_high": float(np.quantile(bootstrap_weak_real_minus_optimal, 0.975)),
                "weak_real_minus_optimal_p_value": weak_p_value,
                "weak_optimal_minus_no_i_delta_auc": observed_weak_optimal,
                "weak_optimal_minus_no_i_ci_low": float(np.quantile(bootstrap_weak_optimal_minus_no_i, 0.025)),
                "weak_optimal_minus_no_i_ci_high": float(np.quantile(bootstrap_weak_optimal_minus_no_i, 0.975)),
                "real_minus_matched_delta_auc_observed": observed_weak,
                "matched_minus_no_i_delta_auc_observed": observed_weak_optimal,
                "ci_low": float(np.quantile(bootstrap_weak_real_minus_optimal, 0.025)),
                "ci_high": float(np.quantile(bootstrap_weak_real_minus_optimal, 0.975)),
                "optimal_minus_no_i_ci_low": float(np.quantile(bootstrap_weak_optimal_minus_no_i, 0.025)),
                "optimal_minus_no_i_ci_high": float(np.quantile(bootstrap_weak_optimal_minus_no_i, 0.975)),
                "p_value": weak_p_value,
                "n_bootstrap": int(n_bootstrap),
                "n_permutations": int(n_permutations),
                "primary_pairing_control": "P+Q+R+D+I-real_vs_P+Q+R+D+I-optimal_matched",
                "secondary_pairing_control": "P+I-real_vs_P+I-optimal_matched",
                "weak_control_scope": "secondary_weak_control_pairing",
                "interpretation_scope": "pairing_dependence_full_control_primary",
            }
        )
    return pd.DataFrame(rows)


def run_pairing_dependence(
    inputs: ExplanationInputs,
    source_cache: Mapping[tuple[str, int, int], SourceFoldCache],
    membership: pd.DataFrame,
    *,
    representations: Sequence[str],
    repeats: Sequence[int],
    source_data: Mapping[str, SourceRepresentationData],
    inner_folds: int = INNER_FOLDS,
    c_grid: Sequence[float] = LABEL_C_GRID,
    n_draws: int = PAIRING_DRAWS,
    base_seed: int = BASE_SEED,
    main_repeat: int = 1,
    n_bootstrap: int = N_BOOTSTRAP,
    n_permutations: int = N_PERMUTATIONS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare real, one optimal mismatch and random mismatch references."""

    match_frame = build_pairing_match_frame(inputs)
    assignment_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    pairwise_rows: list[dict[str, object]] = []
    random_reference_rows: list[dict[str, object]] = []
    inner_quality_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    delta_rows: list[dict[str, object]] = []
    for representation in representations:
        if representation not in source_data:
            raise ValueError(f"source-level pairing data is missing {representation}")
        for repeat in sorted(int(value) for value in repeats):
            for fold in range(1, 6):
                source = source_cache[(representation, repeat, fold)]
                train_idx, test_idx = source.train_idx, source.test_idx
                train_frame = match_frame.iloc[train_idx].reset_index(drop=True)
                test_frame = match_frame.iloc[test_idx].reset_index(drop=True)
                fold_seed = int(base_seed) + repeat * 100_000 + fold * 1_000 + _stable_seed_offset(representation)
                random_test_assignments: dict[int, pd.DataFrame] = {}
                random_distances: list[float] = []
                for draw in range(1, int(n_draws) + 1):
                    random_assignment = make_random_derangement_assignment(
                        recipient=test_frame,
                        donor=test_frame,
                        fit_reference=train_frame,
                        seed=fold_seed + draw,
                    )
                    random_test_assignments[draw] = random_assignment
                    random_quality = pairing_quality_summary(
                        recipient=test_frame,
                        donor=test_frame,
                        assignment=random_assignment,
                        fit_reference=train_frame,
                        random_distance_reference=None,
                        assignment_type="random_mismatch",
                    )
                    random_distances.append(float(random_quality["mean_distance"]))
                    random_reference_rows.append(
                        {
                            "record_type": "random_reference",
                            "representation": representation,
                            "repeat": int(repeat),
                            "outer_fold": int(fold),
                            "partition": "outer_test",
                            "draw": int(draw),
                            "assignment_type": "random_mismatch",
                            **{key: value for key, value in random_quality.items() if key not in {"feature_mean_abs_standardized_paired_difference", "feature_median_abs_standardized_paired_difference", "feature_q95_abs_standardized_paired_difference", "feature_mean_signed_standardized_paired_difference"}},
                        }
                    )
                optimal_test_assignment = make_pairing_assignment(
                    recipient=test_frame,
                    donor=test_frame,
                    fit_reference=train_frame,
                    seed=fold_seed,
                )
                draw_specs: list[tuple[int, str, pd.DataFrame]] = [(0, "optimal_matched", optimal_test_assignment)]
                draw_specs.extend((draw, "random_mismatch", assignment) for draw, assignment in random_test_assignments.items())
                for draw, draw_type, test_assignment in draw_specs:
                    matched_train_i, matched_test_i, source_audit, inner_quality = build_fold_contained_matched_source_logits(
                        representation=representation,
                        source_data=source_data[representation],
                        labels=inputs.labels,
                        participant_ids=inputs.participant_ids,
                        match_frame=match_frame,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        outer_test_assignment=test_assignment,
                        inner_folds=inner_folds,
                        c_grid=DEFAULT_C_GRID,
                        seed=fold_seed + draw * 10_000,
                        assignment_mode=draw_type,
                        collect_inner_quality=(draw == 0),
                    )
                    for audit_row in source_audit.to_dict("records"):
                        assignment_rows.append(
                            {
                                "record_type": "source_level_assignment",
                                "representation": representation,
                                "repeat": int(repeat),
                                "outer_fold": int(fold),
                                "draw": int(draw),
                                "draw_type": draw_type,
                                **audit_row,
                            }
                        )
                    if not inner_quality.empty:
                        inner_quality_rows.extend(
                            {
                                "representation": representation,
                                "repeat": int(repeat),
                                "outer_fold": int(fold),
                                "draw": int(draw),
                                "draw_type": draw_type,
                                **quality_row,
                            }
                            for quality_row in inner_quality.to_dict("records")
                        )
                    quality = pairing_quality_summary(
                        recipient=test_frame,
                        donor=test_frame,
                        assignment=test_assignment,
                        fit_reference=train_frame,
                        random_distance_reference=random_distances,
                        assignment_type=draw_type,
                    )
                    quality_rows.extend(
                        {
                            "record_type": "quality",
                            **row,
                        }
                        for row in pairing_balance_rows(
                            recipient=test_frame,
                            donor=test_frame,
                            assignment=test_assignment,
                            representation=representation,
                            repeat=repeat,
                            outer_fold=fold,
                            draw=draw,
                            partition="outer_test",
                            fit_reference=train_frame,
                            assignment_type=draw_type,
                            random_distance_reference=random_distances,
                        )
                    )
                    pairwise_rows.extend(
                        pairing_pairwise_rows(
                            recipient=test_frame,
                            donor=test_frame,
                            assignment=test_assignment,
                            fit_reference=train_frame,
                            representation=representation,
                            repeat=repeat,
                            outer_fold=fold,
                            draw=draw,
                            partition="outer_test",
                            assignment_type=draw_type,
                        )
                    )
                    for assignment_row in test_assignment.to_dict("records"):
                        assignment_rows.append(
                            {
                                "record_type": "outer_test_assignment",
                                "representation": representation,
                                "repeat": int(repeat),
                                "outer_fold": int(fold),
                                "draw": int(draw),
                                "draw_type": draw_type,
                                "partition": "outer_test",
                                **assignment_row,
                            }
                        )
                    p_train = _subset_design("P", participant_logits=source.participant_train_logit, block_matrices=inputs.block_matrices, indices=train_idx)
                    p_test = _subset_design("P", participant_logits=source.participant_test_logit, block_matrices=inputs.block_matrices, indices=test_idx)
                    full_train = _subset_design("P+Q+R+D", participant_logits=source.participant_train_logit, block_matrices=inputs.block_matrices, indices=train_idx)
                    full_test = _subset_design("P+Q+R+D", participant_logits=source.participant_test_logit, block_matrices=inputs.block_matrices, indices=test_idx)
                    matched_prefix = "optimal_matched" if draw_type == "optimal_matched" else "random_mismatch"
                    designs: dict[str, tuple[np.ndarray, np.ndarray]] = {
                        "P": (p_train, p_test),
                        "P+I-real": (np.column_stack([p_train, source.interviewer_train_logit]), np.column_stack([p_test, source.interviewer_test_logit])),
                        f"P+I-{matched_prefix}": (np.column_stack([p_train, matched_train_i]), np.column_stack([p_test, matched_test_i])),
                        "full": (full_train, full_test),
                        "full+I-real": (np.column_stack([full_train, source.interviewer_train_logit]), np.column_stack([full_test, source.interviewer_test_logit])),
                        f"full+I-{matched_prefix}": (np.column_stack([full_train, matched_train_i]), np.column_stack([full_test, matched_test_i])),
                    }
                    for model_index, (model_name, (X_train, X_test)) in enumerate(designs.items()):
                        probability, selected, _ = fit_label_outer(
                            X_train,
                            inputs.labels[train_idx],
                            X_test,
                            inner_folds=inner_folds,
                            c_grid=c_grid,
                            seed=fold_seed + draw * 100 + model_index,
                        )
                        for local_index, global_index in enumerate(test_idx):
                            prediction_rows.append(
                                {
                                    "participant_id": int(inputs.participant_ids[global_index]),
                                    "label": int(inputs.labels[global_index]),
                                    "representation": representation,
                                    "repeat": int(repeat),
                                    "outer_fold": int(fold),
                                    "draw": int(draw),
                                    "draw_type": draw_type,
                                    "model_name": model_name,
                                    "probability": float(probability[local_index]),
                                    "selected_C": float(selected),
                                }
                            )
    predictions = pd.DataFrame(prediction_rows)
    for (representation, repeat, draw), draw_frame in predictions.groupby(["representation", "repeat", "draw"], sort=True):
        wide = draw_frame.pivot(index=["participant_id", "label"], columns="model_name", values="probability")
        draw_type = str(draw_frame["draw_type"].iloc[0])
        matched_name = f"P+I-{draw_type}"
        full_matched_name = f"full+I-{draw_type}"
        labels = wide.index.get_level_values("label").to_numpy(dtype=int)
        metrics_by_model = {
            model: _classification_metrics(labels, wide[model].to_numpy(dtype=float))
            for model in wide.columns
        }
        for model_name, model_metrics in metrics_by_model.items():
            metric_rows.append(
                {
                    "representation": representation,
                    "repeat": int(repeat),
                    "draw": int(draw),
                    "draw_type": draw_type,
                    "model_name": model_name,
                    **model_metrics,
                    "interpretation_scope": "pairing_dependence",
                }
            )
        weak_real = metrics_by_model["P+I-real"]["roc_auc"] - metrics_by_model["P"]["roc_auc"]
        weak_matched = metrics_by_model[matched_name]["roc_auc"] - metrics_by_model["P"]["roc_auc"]
        full_real = metrics_by_model["full+I-real"]["roc_auc"] - metrics_by_model["full"]["roc_auc"]
        full_matched = metrics_by_model[full_matched_name]["roc_auc"] - metrics_by_model["full"]["roc_auc"]
        delta_rows.append(
            {
                "representation": representation,
                "repeat": int(repeat),
                "draw": int(draw),
                "draw_type": draw_type,
                "matched_model_name": matched_name,
                "weak_real_delta_auc": weak_real,
                "weak_matched_delta_auc": weak_matched,
                "full_real_delta_auc": full_real,
                "full_matched_delta_auc": full_matched,
                "real_minus_matched_delta_auc": weak_real - weak_matched,
                "matched_minus_no_i_delta_auc": weak_matched,
                "real_minus_optimal_delta_auc": weak_real - weak_matched if draw_type == "optimal_matched" else float("nan"),
                "real_minus_random_delta_auc": weak_real - weak_matched if draw_type == "random_mismatch" else float("nan"),
                "optimal_minus_no_i_delta_auc": weak_matched if draw_type == "optimal_matched" else float("nan"),
                "random_minus_no_i_delta_auc": weak_matched if draw_type == "random_mismatch" else float("nan"),
                "real_minus_matched_full_delta_auc": full_real - full_matched,
                "mean_abs_probability_change_weak": float(np.mean(np.abs(wide["P+I-real"] - wide[matched_name]))),
                "mean_abs_probability_change_full": float(np.mean(np.abs(wide["full+I-real"] - wide[full_matched_name]))),
                "interpretation_scope": "pairing_dependence_not_identified_interaction_effect",
            }
        )
    predictions = predictions.sort_values(["representation", "repeat", "draw", "participant_id", "model_name"], kind="stable").reset_index(drop=True)
    metrics = pd.DataFrame(metric_rows)
    deltas = pd.DataFrame(delta_rows)
    draw_inference = _pairing_draw_inference(
        predictions,
        deltas,
        main_repeat=main_repeat,
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
        seed=base_seed + 960_000,
    )
    if not draw_inference.empty:
        deltas = deltas.merge(draw_inference, on=["representation", "repeat", "draw"], how="left")
    quality = pd.DataFrame(quality_rows)
    pairwise = pd.DataFrame(pairwise_rows)
    random_reference = pd.DataFrame(random_reference_rows)
    inner_quality = pd.DataFrame(inner_quality_rows)
    stability_rows: list[dict[str, object]] = []
    for (representation, repeat), group in deltas.groupby(["representation", "repeat"], sort=True):
        optimal = group.loc[group["draw_type"].eq("optimal_matched")]
        random = group.loc[group["draw_type"].eq("random_mismatch")]
        quality_group = quality.loc[
            quality["representation"].eq(representation)
            & quality["repeat"].eq(int(repeat))
            & quality["assignment_type"].eq("optimal_matched")
        ]
        inner_quality_group = inner_quality.loc[
            inner_quality["representation"].eq(representation)
            & inner_quality["repeat"].eq(int(repeat))
            & inner_quality["assignment_type"].eq("optimal_matched")
        ] if not inner_quality.empty else inner_quality
        inner_quality_summary = summarise_inner_training_quality(inner_quality_group)
        stability_rows.append(
            {
                "representation": representation,
                "repeat": int(repeat),
                "n_random_draws": int(random["draw"].nunique()),
                "optimal_real_minus_matched_delta_auc": float(optimal["real_minus_matched_delta_auc"].iloc[0]),
                "optimal_full_real_minus_matched_delta_auc": float(optimal["real_minus_matched_full_delta_auc"].iloc[0]),
                "random_real_minus_matched_delta_auc_mean": float(random["real_minus_matched_delta_auc"].mean()),
                "random_real_minus_matched_delta_auc_sd": float(random["real_minus_matched_delta_auc"].std(ddof=1)) if len(random) > 1 else 0.0,
                "random_real_minus_matched_delta_auc_q025": float(random["real_minus_matched_delta_auc"].quantile(0.025)),
                "random_real_minus_matched_delta_auc_q975": float(random["real_minus_matched_delta_auc"].quantile(0.975)),
                "random_full_real_minus_matched_delta_auc_mean": float(random["real_minus_matched_full_delta_auc"].mean()),
                "random_full_real_minus_matched_delta_auc_sd": float(random["real_minus_matched_full_delta_auc"].std(ddof=1)) if len(random) > 1 else 0.0,
                "random_full_real_minus_matched_delta_auc_q025": float(random["real_minus_matched_full_delta_auc"].quantile(0.025)),
                "random_full_real_minus_matched_delta_auc_q975": float(random["real_minus_matched_full_delta_auc"].quantile(0.975)),
                "matching_adequate": bool(quality_group["matching_adequate"].all()),
                "inner_matching_pass_rate": float(inner_quality_summary["pass_rate"]),
                "training_match_quality_limited": bool(inner_quality_summary["training_match_quality_limited"]),
                "interpretation_scope": "pairing_dependence_optimal_vs_random_reference",
            }
        )
    return (
        pd.DataFrame(assignment_rows),
        predictions,
        quality,
        pairwise,
        random_reference,
        metrics,
        deltas,
        pd.DataFrame(stability_rows),
        pd.DataFrame(inner_quality_rows),
    )


def recoverability_inference(
    explanation_predictions: pd.DataFrame,
    *,
    representation: str,
    repeat: int,
    subset: str = "P+Q+R+D",
    n_bootstrap: int = N_BOOTSTRAP,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = BASE_SEED,
) -> dict[str, float | int]:
    frame = explanation_predictions.loc[
        explanation_predictions["representation"].eq(representation)
        & explanation_predictions["repeat"].eq(int(repeat))
        & explanation_predictions["subset"].eq(subset)
    ].sort_values("participant_id", kind="stable")
    if frame.empty:
        raise ValueError(f"recoverability predictions are missing {representation}/{repeat}/{subset}")
    target = frame["interviewer_logit"].to_numpy(dtype=float)
    prediction = frame["predicted_interviewer_logit"].to_numpy(dtype=float)
    null_prediction = frame["outer_train_mean_null_prediction"].to_numpy(dtype=float)
    null_mse = float(mean_squared_error(target, null_prediction))
    full_mse = float(mean_squared_error(target, prediction))
    observed_delta_mse = float(null_mse - full_mse)
    observed_r2 = float(1.0 - full_mse / null_mse) if null_mse > 1e-15 else float("nan")
    bootstrap = joint_bootstrap_indices(len(frame), n_bootstrap=n_bootstrap, seed=seed + _stable_seed_offset(representation))
    distribution = np.asarray(
        [
            mean_squared_error(target[index], null_prediction[index])
            - mean_squared_error(target[index], prediction[index])
            for index in bootstrap
        ],
        dtype=float,
    )
    permutation = paired_prediction_swap_permutation(
        target,
        null_prediction,
        prediction,
        n_permutations=n_permutations,
        seed=seed + 1_000_003,
    )
    return {
        "estimate_mse_improvement": observed_delta_mse,
        "estimate_r2": observed_r2,
        "observed_delta_mse": observed_delta_mse,
        "ci_low": float(np.quantile(distribution, 0.025)),
        "ci_high": float(np.quantile(distribution, 0.975)),
        "p_value": float(permutation["p_value"]),
        "permutation_p_value": float(permutation["p_value"]),
        "n_bootstrap": int(n_bootstrap),
        "n_permutations": int(n_permutations),
        "ci_metric": "delta_mse",
    }


def summarise_repeat_stability(
    explanation_predictions: pd.DataFrame,
    residual_deltas: pd.DataFrame,
    pairing_deltas: pd.DataFrame,
    fake_explanation_results: pd.DataFrame,
    *,
    seed: int = BASE_SEED,
) -> pd.DataFrame:
    """Return long-form repeat stability without treating repeats as samples."""

    rows: list[dict[str, object]] = []
    for (representation, repeat), frame in explanation_predictions.groupby(["representation", "repeat"], sort=True):
        full = frame.loc[frame["subset"].eq("P+Q+R+D")].sort_values("participant_id", kind="stable")
        null = frame.loc[frame["subset"].eq("null")].sort_values("participant_id", kind="stable")
        rows.append(
            {
                "representation": representation,
                "repeat": int(repeat),
                "metric": "full_explanation_r2",
                "block": "",
                "estimate": relative_r2(
                    full["interviewer_logit"].to_numpy(dtype=float),
                    full["predicted_interviewer_logit"].to_numpy(dtype=float),
                    null["outer_train_mean_null_prediction"].to_numpy(dtype=float),
                ),
                "interpretation_scope": "repeat_stability_not_independent_samples",
            }
        )
        target = null["interviewer_logit"].to_numpy(dtype=float)
        null_prediction = null["outer_train_mean_null_prediction"].to_numpy(dtype=float)
        subset_values = {
            subset: relative_r2(
                target,
                frame.loc[frame["subset"].eq(subset)].sort_values("participant_id", kind="stable")["predicted_interviewer_logit"].to_numpy(dtype=float),
                null_prediction,
            )
            for subset in ALL_BLOCK_SUBSETS
        }
        shapley = compute_shapley_values(subset_values)
        for block, estimate in shapley.items():
            rows.append(
                {
                    "representation": representation,
                    "repeat": int(repeat),
                    "metric": "block_shapley_delta_r2",
                    "block": block,
                    "estimate": float(estimate),
                    "interpretation_scope": "statistical_explanation_not_causal_contribution",
                }
            )
    for _, row in residual_deltas.iterrows():
        rows.append(
            {
                "representation": row["representation"],
                "repeat": int(row["repeat"]),
                "metric": "residual_delta_auc",
                "block": "",
                "estimate": float(row["delta_auc"]),
                "interpretation_scope": "repeat_stability_not_independent_samples",
            }
        )
    if not pairing_deltas.empty:
        pairing_summary = (
            pairing_deltas.loc[pairing_deltas["draw_type"].eq("optimal_matched")]
            .groupby(["representation", "repeat"], as_index=False)["real_minus_matched_full_delta_auc"]
            .mean()
        )
        for _, row in pairing_summary.iterrows():
            rows.append(
                {
                    "representation": row["representation"],
                    "repeat": int(row["repeat"]),
                    "metric": "pairing_full_control_delta_auc",
                    "block": "",
                    "estimate": float(row["real_minus_matched_full_delta_auc"]),
                    "interpretation_scope": "pairing_dependence_full_control_primary",
                }
            )
    if not fake_explanation_results.empty:
        fake = fake_explanation_results.loc[fake_explanation_results["domain_type"].eq("fake")]
        real = fake_explanation_results.loc[fake_explanation_results["domain_type"].eq("real")]
        for (representation, repeat), group in fake.groupby(["representation", "repeat"], sort=True):
            real_rows = real.loc[
                real["representation"].eq(representation) & real["repeat"].eq(int(repeat))
            ]
            if real_rows.empty:
                continue
            real_value = float(real_rows["r2"].iloc[0])
            rows.append(
                {
                    "representation": representation,
                    "repeat": int(repeat),
                    "metric": "real_D_minus_fake_D_r2",
                    "block": "D",
                    "estimate": real_value - float(group["r2"].mean()),
                    "interpretation_scope": "real_D_vs_fake_D_negative_control",
                }
            )
    return pd.DataFrame(rows)


def benjamini_hochberg(values: Sequence[float]) -> np.ndarray:
    p_values = np.asarray(values, dtype=float)
    if p_values.ndim != 1:
        raise ValueError("p-values must be one-dimensional")
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def fake_domain_summary(
    explanation_results: pd.DataFrame,
    residual_results: pd.DataFrame,
    explanation_oof: pd.DataFrame | None = None,
    residual_oof: pd.DataFrame | None = None,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BASE_SEED,
) -> pd.DataFrame:
    """Summarise Fake-D draws with participant-and-draw bootstrap inference."""

    def double_bootstrap(
        real_explanation_frame: pd.DataFrame,
        fake_explanation_frame: pd.DataFrame,
        real_residual_frame: pd.DataFrame,
        fake_residual_frame: pd.DataFrame,
        *,
        local_seed: int,
    ) -> dict[str, float | int | str]:
        real_explanation_frame = real_explanation_frame.sort_values("participant_id", kind="stable").reset_index(drop=True)
        real_residual_frame = real_residual_frame.sort_values("participant_id", kind="stable").reset_index(drop=True)
        participant_ids = real_explanation_frame["participant_id"].to_numpy()
        labels = real_explanation_frame["label"].to_numpy(dtype=int)
        if len(np.unique(labels)) != 2:
            raise ValueError("Fake-D double bootstrap requires both label classes")
        fake_draws = sorted(int(value) for value in fake_explanation_frame["draw"].unique())
        if not fake_draws:
            raise ValueError("Fake-D double bootstrap has no fake draws")
        fake_explanation_by_draw = {}
        fake_residual_by_draw = {}
        for draw in fake_draws:
            current_explanation = fake_explanation_frame.loc[
                fake_explanation_frame["draw"].eq(draw)
            ].sort_values("participant_id", kind="stable").reset_index(drop=True)
            current_residual = fake_residual_frame.loc[
                fake_residual_frame["draw"].eq(draw)
            ].sort_values("participant_id", kind="stable").reset_index(drop=True)
            if not np.array_equal(current_explanation["participant_id"].to_numpy(), participant_ids):
                raise ValueError("Fake-D explanation participant alignment failed")
            if not np.array_equal(current_residual["participant_id"].to_numpy(), participant_ids):
                raise ValueError("Fake-D residual participant alignment failed")
            fake_explanation_by_draw[draw] = current_explanation
            fake_residual_by_draw[draw] = current_residual
        real_r2 = relative_r2(
            real_explanation_frame["interviewer_logit"],
            real_explanation_frame["predicted_interviewer_logit"],
            real_explanation_frame["outer_train_mean_null_prediction"],
        )
        fake_r2_draws = np.asarray(
            [
                relative_r2(
                    frame["interviewer_logit"],
                    frame["predicted_interviewer_logit"],
                    frame["outer_train_mean_null_prediction"],
                )
                for frame in fake_explanation_by_draw.values()
            ],
            dtype=float,
        )
        real_residual_delta = _paired_probability_delta_metrics(
            real_residual_frame["label"].to_numpy(dtype=int),
            real_residual_frame["base_probability"].to_numpy(dtype=float),
            real_residual_frame["enhanced_probability"].to_numpy(dtype=float),
        )["delta_auc"]
        fake_residual_delta_draws = np.asarray(
            [
                _paired_probability_delta_metrics(
                    frame["label"].to_numpy(dtype=int),
                    frame["base_probability"].to_numpy(dtype=float),
                    frame["enhanced_probability"].to_numpy(dtype=float),
                )["delta_auc"]
                for frame in fake_residual_by_draw.values()
            ],
            dtype=float,
        )
        rng = np.random.default_rng(int(local_seed))
        bootstrap_indices = _stratified_bootstrap_indices(
            labels, n_bootstrap=int(n_bootstrap), seed=int(local_seed) + 1
        )
        r2_distribution: list[float] = []
        residual_distribution: list[float] = []
        for index in bootstrap_indices:
            draw = fake_draws[int(rng.integers(0, len(fake_draws)))]
            fake_explanation = fake_explanation_by_draw[draw]
            fake_residual = fake_residual_by_draw[draw]
            r2_distribution.append(
                relative_r2(
                    real_explanation_frame["interviewer_logit"].to_numpy(dtype=float)[index],
                    real_explanation_frame["predicted_interviewer_logit"].to_numpy(dtype=float)[index],
                    real_explanation_frame["outer_train_mean_null_prediction"].to_numpy(dtype=float)[index],
                )
                - relative_r2(
                    fake_explanation["interviewer_logit"].to_numpy(dtype=float)[index],
                    fake_explanation["predicted_interviewer_logit"].to_numpy(dtype=float)[index],
                    fake_explanation["outer_train_mean_null_prediction"].to_numpy(dtype=float)[index],
                )
            )
            real_delta = _paired_probability_delta_metrics(
                labels[index],
                real_residual_frame["base_probability"].to_numpy(dtype=float)[index],
                real_residual_frame["enhanced_probability"].to_numpy(dtype=float)[index],
            )["delta_auc"]
            fake_delta = _paired_probability_delta_metrics(
                labels[index],
                fake_residual["base_probability"].to_numpy(dtype=float)[index],
                fake_residual["enhanced_probability"].to_numpy(dtype=float)[index],
            )["delta_auc"]
            residual_distribution.append(float(real_delta - fake_delta))
        r2_distribution_array = np.asarray(r2_distribution, dtype=float)
        residual_distribution_array = np.asarray(residual_distribution, dtype=float)
        observed_r2 = float(real_r2 - fake_r2_draws.mean())
        observed_residual = float(real_residual_delta - fake_residual_delta_draws.mean())
        return {
            "real_D_minus_fake_D_r2_double_bootstrap_ci_low": float(np.quantile(r2_distribution_array, 0.025)),
            "real_D_minus_fake_D_r2_double_bootstrap_ci_high": float(np.quantile(r2_distribution_array, 0.975)),
            "real_D_minus_fake_D_r2_double_bootstrap_p_value": float(
                (np.count_nonzero(np.abs(r2_distribution_array) >= abs(observed_r2)) + 1)
                / (len(r2_distribution_array) + 1)
            ),
            "real_D_minus_fake_D_residual_delta_auc_double_bootstrap_ci_low": float(np.quantile(residual_distribution_array, 0.025)),
            "real_D_minus_fake_D_residual_delta_auc_double_bootstrap_ci_high": float(np.quantile(residual_distribution_array, 0.975)),
            "real_D_minus_fake_D_residual_delta_auc_double_bootstrap_p_value": float(
                (np.count_nonzero(np.abs(residual_distribution_array) >= abs(observed_residual)) + 1)
                / (len(residual_distribution_array) + 1)
            ),
            "double_bootstrap_n": int(n_bootstrap),
            "double_bootstrap_scope": "participant_stratified_resampling_with_fake_draw_resampling",
        }

    rows: list[dict[str, object]] = []
    real_explanations = explanation_results.loc[explanation_results["domain_type"].eq("real")]
    for (representation, repeat), real_explanation in real_explanations.groupby(["representation", "repeat"], sort=True):
        fake_explanation = explanation_results.loc[
            explanation_results["representation"].eq(representation)
            & explanation_results["repeat"].eq(int(repeat))
            & explanation_results["domain_type"].eq("fake")
        ]
        real_residual = residual_results.loc[
            residual_results["representation"].eq(representation)
            & residual_results["repeat"].eq(int(repeat))
            & residual_results["domain_type"].eq("real")
        ]
        fake_residual = residual_results.loc[
            residual_results["representation"].eq(representation)
            & residual_results["repeat"].eq(int(repeat))
            & residual_results["domain_type"].eq("fake")
        ]
        if fake_explanation.empty or real_residual.empty or fake_residual.empty:
            continue
        real_r2 = float(real_explanation["r2"].iloc[0])
        fake_r2 = fake_explanation["r2"].to_numpy(dtype=float)
        real_delta = float(real_residual["delta_auc"].iloc[0])
        fake_delta = fake_residual["delta_auc"].to_numpy(dtype=float)
        fake_r2_sd = float(np.std(fake_r2, ddof=1)) if len(fake_r2) > 1 else 0.0
        fake_delta_sd = float(np.std(fake_delta, ddof=1)) if len(fake_delta) > 1 else 0.0
        double_bootstrap_fields: dict[str, object] = {
            "real_D_minus_fake_D_r2_double_bootstrap_ci_low": float("nan"),
            "real_D_minus_fake_D_r2_double_bootstrap_ci_high": float("nan"),
            "real_D_minus_fake_D_r2_double_bootstrap_p_value": float("nan"),
            "real_D_minus_fake_D_residual_delta_auc_double_bootstrap_ci_low": float("nan"),
            "real_D_minus_fake_D_residual_delta_auc_double_bootstrap_ci_high": float("nan"),
            "real_D_minus_fake_D_residual_delta_auc_double_bootstrap_p_value": float("nan"),
            "double_bootstrap_n": 0,
            "double_bootstrap_scope": "not_run_oof_unavailable",
        }
        if explanation_oof is not None and residual_oof is not None and not explanation_oof.empty and not residual_oof.empty:
            real_explanation_oof = explanation_oof.loc[
                explanation_oof["representation"].eq(representation)
                & explanation_oof["repeat"].eq(int(repeat))
                & explanation_oof["domain_type"].eq("real")
            ]
            fake_explanation_oof = explanation_oof.loc[
                explanation_oof["representation"].eq(representation)
                & explanation_oof["repeat"].eq(int(repeat))
                & explanation_oof["domain_type"].eq("fake")
            ]
            real_residual_oof = residual_oof.loc[
                residual_oof["representation"].eq(representation)
                & residual_oof["repeat"].eq(int(repeat))
                & residual_oof["domain_type"].eq("real")
            ]
            fake_residual_oof = residual_oof.loc[
                residual_oof["representation"].eq(representation)
                & residual_oof["repeat"].eq(int(repeat))
                & residual_oof["domain_type"].eq("fake")
            ]
            double_bootstrap_fields = double_bootstrap(
                real_explanation_oof,
                fake_explanation_oof,
                real_residual_oof,
                fake_residual_oof,
                local_seed=int(seed) + _stable_seed_offset(f"{representation}:{repeat}"),
            )
        rows.append(
            {
                "representation": representation,
                "repeat": int(repeat),
                "real_D_r2": real_r2,
                "fake_D_r2_mean": float(fake_r2.mean()),
                "fake_D_r2_sd": fake_r2_sd,
                "fake_D_r2_q025": float(np.quantile(fake_r2, 0.025)),
                "fake_D_r2_q975": float(np.quantile(fake_r2, 0.975)),
                "real_D_minus_fake_D_r2": real_r2 - float(fake_r2.mean()),
                "real_D_residual_delta_auc": real_delta,
                "fake_D_residual_delta_auc_mean": float(fake_delta.mean()),
                "fake_D_residual_delta_auc_sd": fake_delta_sd,
                "fake_D_residual_delta_auc_q025": float(np.quantile(fake_delta, 0.025)),
                "fake_D_residual_delta_auc_q975": float(np.quantile(fake_delta, 0.975)),
                "real_D_minus_fake_D_residual_delta_auc": real_delta - float(fake_delta.mean()),
                "p_value": float(
                    double_bootstrap_fields["real_D_minus_fake_D_r2_double_bootstrap_p_value"]
                    if np.isfinite(double_bootstrap_fields["real_D_minus_fake_D_r2_double_bootstrap_p_value"])
                    else (np.count_nonzero(fake_r2 >= real_r2) + 1) / (len(fake_r2) + 1)
                ),
                "interpretation_scope": "real_D_vs_fake_D_negative_control",
                **double_bootstrap_fields,
            }
        )
    return pd.DataFrame(rows)


def _save_figure_all_formats(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Creator": "DAIC-WOZ interviewer signal explanation",
        "Title": stem.name,
        "CreationDate": None,
        "ModDate": None,
    }
    for extension in ("png", "svg", "pdf"):
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=300, metadata=metadata)
    plt.close(fig)


def generate_analysis_figures(
    output_dir: str | Path,
    *,
    explanation_metrics_table: pd.DataFrame,
    shapley_table: pd.DataFrame,
    residual_deltas: pd.DataFrame,
    pairing_metrics: pd.DataFrame,
    fake_explanation_results: pd.DataFrame,
    main_repeat: int,
) -> list[str]:
    """Create the five locked figure families without source text."""

    output_dir = Path(output_dir)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    main_metrics = explanation_metrics_table.loc[explanation_metrics_table["repeat"].eq(int(main_repeat))]
    fig, ax = plt.subplots(figsize=(8, 6))
    for representation, group in main_metrics.groupby("representation", sort=True):
        y = np.arange(len(group)) + 0.12 * list(main_metrics["representation"].unique()).index(representation)
        ax.scatter(group["r2"], y, label=representation, s=28)
    labels = list(main_metrics.loc[main_metrics["representation"].eq(main_metrics["representation"].iloc[0]), "subset"])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Null-relative OOF R2")
    ax.set_title("Interviewer score recoverability")
    ax.legend()
    stem = figure_dir / "interviewer_score_recoverability_forest"
    _save_figure_all_formats(fig, stem)
    generated.extend(str(stem.with_suffix(f".{ext}")) for ext in ("png", "svg", "pdf"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    shapley_main = shapley_table.loc[shapley_table["representation"].eq("tfidf")]
    if shapley_main.empty:
        shapley_main = shapley_table
    for index, row in shapley_main.reset_index(drop=True).iterrows():
        ax.errorbar(
            row["estimate_delta_r2"],
            index,
            xerr=[[row["estimate_delta_r2"] - row["ci_low"]], [row["ci_high"] - row["estimate_delta_r2"]]],
            fmt="o",
        )
    ax.set_yticks(np.arange(len(shapley_main)))
    ax.set_yticklabels(shapley_main["block"].tolist())
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Shapley ΔR2")
    ax.set_title("Block statistical explanation")
    stem = figure_dir / "explanation_block_shapley_forest"
    _save_figure_all_formats(fig, stem)
    generated.extend(str(stem.with_suffix(f".{ext}")) for ext in ("png", "svg", "pdf"))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    residual_main = residual_deltas.loc[residual_deltas["repeat"].eq(int(main_repeat))]
    if not residual_main.empty:
        ax.errorbar(
            np.arange(len(residual_main)),
            residual_main["delta_auc"],
            yerr=[residual_main["delta_auc"] - residual_main["ci_low"], residual_main["ci_high"] - residual_main["delta_auc"]],
            fmt="o",
        )
        ax.set_xticks(np.arange(len(residual_main)))
        ax.set_xticklabels(residual_main["representation"].tolist())
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Residual ΔAUC")
    ax.set_title("Residual interviewer signal")
    stem = figure_dir / "residual_signal_forest"
    _save_figure_all_formats(fig, stem)
    generated.extend(str(stem.with_suffix(f".{ext}")) for ext in ("png", "svg", "pdf"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if not pairing_metrics.empty:
        pairing_main = pairing_metrics.loc[pairing_metrics["repeat"].eq(int(main_repeat))]
        for index, (model_name, group) in enumerate(pairing_main.groupby("model_name", sort=True)):
            ax.scatter(np.full(len(group), index), group["roc_auc"], s=8, alpha=0.4)
        ax.set_xticks(np.arange(pairing_main["model_name"].nunique()))
        ax.set_xticklabels(sorted(pairing_main["model_name"].unique()), rotation=35, ha="right")
    ax.set_ylabel("OOF AUC")
    ax.set_title("Pairing dependence")
    stem = figure_dir / "pairing_dependence_plot"
    _save_figure_all_formats(fig, stem)
    generated.extend(str(stem.with_suffix(f".{ext}")) for ext in ("png", "svg", "pdf"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if not fake_explanation_results.empty:
        for index, (representation, group) in enumerate(fake_explanation_results.groupby("representation", sort=True)):
            fake = group.loc[group["domain_type"].eq("fake"), "r2"]
            real = group.loc[group["domain_type"].eq("real"), "r2"]
            ax.scatter(np.full(len(fake), index), fake, s=8, alpha=0.35, label=f"{representation} fake")
            if not real.empty:
                ax.scatter([index], [real.iloc[0]], marker="D", s=35, label=f"{representation} real")
        ax.set_xticks(np.arange(fake_explanation_results["representation"].nunique()))
        ax.set_xticklabels(sorted(fake_explanation_results["representation"].unique()))
    ax.set_ylabel("Null-relative OOF R2")
    ax.set_title("Fake-D negative control")
    ax.legend(fontsize=8)
    stem = figure_dir / "fake_domain_negative_control_plot"
    _save_figure_all_formats(fig, stem)
    generated.extend(str(stem.with_suffix(f".{ext}")) for ext in ("png", "svg", "pdf"))
    return generated


def _write_path_dictionary(path: Path, dictionary: Sequence[Mapping[str, str]], *, status: str) -> None:
    lines = [
        "# Interviewer path feature dictionary",
        "",
        f"Status: `{status}`.",
        "",
        "Features are derived only from the frozen turn/prompt annotation table.",
        "No PHQ label, score, or transcript text is used in feature derivation.",
        "",
        "| Frozen prompt ID | Derived feature |",
        "|---|---|",
    ]
    for row in dictionary:
        lines.append(f"| `{row['raw_prompt_id']}` | `{row['feature']}` |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_feature_audits(run_dir: Path, inputs: ExplanationInputs) -> None:
    atomic_write_csv(inputs.structural_audit, run_dir / "structural_feature_audit.csv")
    atomic_write_json(
        {
            "summary": inputs.structural_audit_summary,
            "audit_rows": inputs.structural_audit.to_dict("records"),
        },
        run_dir / "structural_feature_audit.json",
    )
    atomic_write_csv(inputs.path_features, run_dir / "interviewer_path_coverage_features.csv")
    _write_path_dictionary(
        run_dir / "interviewer_path_feature_dictionary.md",
        inputs.path_dictionary,
        status="available" if len(inputs.path_dictionary) else "available_no_prompt_id_rows",
    )
    atomic_write_json(
        {
            "summary": inputs.path_audit_summary,
            "audit_rows": inputs.path_audit.to_dict("records"),
            "dictionary_rows": list(inputs.path_dictionary),
        },
        run_dir / "interviewer_path_feature_audit.json",
    )


def _resolve_code_commit(repo_root: Path, explicit: str | None = None) -> str:
    if explicit:
        return str(explicit)
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _output_file_hashes(run_dir: Path) -> dict[str, str]:
    excluded = {"run_manifest.json", "rerun_verification.json"}
    return {
        path.relative_to(run_dir).as_posix(): sha256_file(path)
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name not in excluded
    }


def _write_global_fdr(
    explanation_predictions: pd.DataFrame,
    residual_deltas: pd.DataFrame,
    fake_summary: pd.DataFrame,
    pairing_deltas: pd.DataFrame,
    *,
    main_repeat: int,
    n_bootstrap: int = N_BOOTSTRAP,
    n_permutations: int = N_PERMUTATIONS,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    representations = sorted(
        set(explanation_predictions["representation"])
        | set(residual_deltas["representation"])
        | set(fake_summary["representation"])
        | (set(pairing_deltas["representation"]) if not pairing_deltas.empty else set())
    )
    for representation in representations:
        recoverability = recoverability_inference(
            explanation_predictions,
            representation=representation,
            repeat=main_repeat,
            n_bootstrap=n_bootstrap,
            n_permutations=n_permutations,
            seed=BASE_SEED + _stable_seed_offset(representation),
        )
        residual = residual_deltas.loc[
            residual_deltas["representation"].eq(representation) & residual_deltas["repeat"].eq(int(main_repeat))
        ]
        fake = fake_summary.loc[
            fake_summary["representation"].eq(representation)
            & fake_summary["repeat"].eq(int(main_repeat))
        ]
        pairing = pairing_deltas.loc[
            pairing_deltas["representation"].eq(representation) & pairing_deltas["repeat"].eq(int(main_repeat))
        ]
        if not pairing.empty and "draw" in pairing.columns:
            pairing_primary = pairing.loc[pairing["draw"].eq(0)]
        else:
            pairing_primary = pairing
        if not pairing_primary.empty and "full_real_minus_optimal_p_value" in pairing_primary.columns:
            pairing_p_value = float(pairing_primary["full_real_minus_optimal_p_value"].dropna().iloc[0]) if pairing_primary["full_real_minus_optimal_p_value"].notna().any() else float("nan")
        else:
            pairing_p_value = float("nan")
        tests = [
            ("recoverability_mse_improvement", float(recoverability["p_value"])),
            ("real_D_vs_fake_D", float(fake["p_value"].iloc[0]) if not fake.empty else float("nan")),
            ("residual_delta_auc", float(residual["p_value"].iloc[0]) if not residual.empty else float("nan")),
            ("pairing_delta_auc", pairing_p_value),
        ]
        for test_name, p_value in tests:
            rows.append(
                {
                    "representation": representation,
                    "test": test_name,
                    "p_value": p_value,
                    "fdr_family": "primary_tfidf_four_tests" if representation == "tfidf" else "representation_robustness_not_primary",
                }
            )
    result = pd.DataFrame(rows)
    primary_mask = result["representation"].eq("tfidf")
    result["q_value"] = np.nan
    if primary_mask.any():
        values = result.loc[primary_mask, "p_value"].to_numpy(dtype=float)
        result.loc[primary_mask, "q_value"] = benjamini_hochberg(values)
    valid_global = result["p_value"].notna()
    result["global_q_value"] = np.nan
    if valid_global.any():
        result.loc[valid_global, "global_q_value"] = benjamini_hochberg(result.loc[valid_global, "p_value"].to_numpy(dtype=float))
    return result


def resolve_run_directory(output_root: str | Path, *, mode: str, run_id: int) -> Path:
    if mode not in {"formal", "smoke"}:
        raise ValueError("mode must be formal or smoke")
    if int(run_id) <= 0:
        raise ValueError("run_id must be positive")
    base = Path(output_root).resolve() / "15_interviewer_signal_explanation"
    if mode == "smoke":
        base = base / "smoke"
    return base / f"run_{int(run_id)}"


def _run_smoke(
    output_root: Path,
    run_dir: Path,
    *,
    code_commit: str | None,
    seed: int,
) -> dict[str, object]:
    run_dir.mkdir(parents=True, exist_ok=True)
    availability = {
        "status": "smoke_test",
        "mode": "smoke",
        "reason": "synthetic contract path; no formal data or result is produced",
        "seed": int(seed),
    }
    atomic_write_json(availability, run_dir / "input_availability.json")
    smoke_summary = {
        "status": "smoke_test",
        "formal_output_root": str(output_root / "15_interviewer_signal_explanation"),
        "source_text_written": False,
        "final_promotion_allowed": False,
    }
    atomic_write_json(smoke_summary, run_dir / "smoke_summary.json")
    hashes = _output_file_hashes(run_dir)
    manifest = {
        "status": "smoke_test",
        "mode": "smoke",
        "analysis": "interviewer-side predictive signal explanation",
        "code_commit": _resolve_code_commit(output_root.parent, code_commit),
        "plan_sha256": _hash_variants(output_root / "15_interviewer_signal_explanation" / "interviewer_signal_explanation_plan.md").get("raw"),
        "cohort": {"n": 0, "positive": 0, "negative": 0},
        "output_file_hashes": hashes,
        "output_inventory_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode("utf-8")).hexdigest(),
        "completion_status": "smoke_only_no_formal_results",
    }
    atomic_write_json(manifest, run_dir / "run_manifest.json")
    return manifest


def run_interviewer_signal_explanation(
    output_root: str | Path,
    *,
    mode: str = "formal",
    run_id: int = 1,
    code_commit: str | None = None,
    embedding_root: str | Path | None = None,
    representations: Sequence[str] = ("tfidf", "mpnet", "bge"),
    expected_n: int = EXPECTED_N,
    expected_repeats: int = 10,
    main_repeat: int = 1,
    stability_repeats: int = 10,
    inner_folds: int = INNER_FOLDS,
    alpha_grid: Sequence[float] = RIDGE_ALPHAS,
    label_c_grid: Sequence[float] = LABEL_C_GRID,
    n_bootstrap: int = N_BOOTSTRAP,
    n_permutations: int = N_PERMUTATIONS,
    fake_draws: int = FAKE_D_DRAWS,
    pairing_draws: int = PAIRING_DRAWS,
    base_seed: int = BASE_SEED,
) -> dict[str, object]:
    """Run the new analysis into one immutable run directory.

    This function is intentionally not called at import time.  The CLI is
    required to request either a formal run or an explicitly separate smoke
    run, so writing result artifacts cannot happen accidentally during tests.
    """

    output_root = Path(output_root).resolve()
    if mode == "smoke":
        run_dir = resolve_run_directory(output_root, mode=mode, run_id=run_id)
        if run_dir.exists():
            raise FileExistsError(f"immutable run directory already exists: {run_dir}")
        return _run_smoke(output_root, run_dir, code_commit=code_commit, seed=base_seed)
    if mode != "formal":
        raise ValueError("mode must be formal or smoke")
    if int(expected_n) != EXPECTED_N or int(expected_repeats) != 10:
        raise ValueError("formal mode requires the locked 142-participant 10-repeat cohort")
    if int(main_repeat) < 1 or int(main_repeat) > int(expected_repeats):
        raise ValueError("main_repeat is outside the locked repeat range")
    if int(stability_repeats) < int(main_repeat) or int(stability_repeats) > int(expected_repeats):
        raise ValueError("stability_repeats must include main_repeat and not exceed expected_repeats")
    representations = tuple(str(value).lower() for value in representations)
    if not representations or any(value not in {"tfidf", "mpnet", "bge"} for value in representations):
        raise ValueError("representations must be selected from tfidf, mpnet and bge")
    if int(fake_draws) <= 0 or int(pairing_draws) <= 0:
        raise ValueError("fake and pairing draw counts must be positive")
    run_dir = resolve_run_directory(output_root, mode=mode, run_id=run_id)
    if run_dir.exists():
        raise FileExistsError(f"immutable run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    availability = build_input_availability(
        output_root,
        embedding_root=embedding_root,
        representations=representations,
    )
    atomic_write_json(availability, run_dir / "input_availability.json")
    required_names = (
        "splits",
        "structural",
        "structural_dictionary",
        "domain_count",
        "domain_source_audit",
    )
    unavailable_required = [
        name for name in required_names if availability["inputs"][name].get("status") != "available"
    ]
    if unavailable_required:
        hashes = _output_file_hashes(run_dir)
        manifest = {
            "status": "not_run_data_unavailable",
            "mode": "formal",
            "analysis": "interviewer-side predictive signal explanation",
            "code_commit": _resolve_code_commit(output_root.parent, code_commit),
            "plan_sha256": _hash_variants(output_root / "15_interviewer_signal_explanation" / "interviewer_signal_explanation_plan.md").get("raw"),
            "missing_required_inputs": unavailable_required,
            "input_availability": availability,
            "output_file_hashes": hashes,
            "completion_status": "not_run_data_unavailable",
        }
        atomic_write_json(manifest, run_dir / "run_manifest.json")
        return manifest

    inputs, membership = load_explanation_inputs(
        output_root,
        expected_n=expected_n,
        expected_repeats=expected_repeats,
        validate_hashes=True,
    )
    _write_feature_audits(run_dir, inputs)
    path_dictionary_status = "available" if availability["inputs"]["turns"].get("status") == "available" else "not_run_data_unavailable"
    _write_path_dictionary(
        run_dir / "interviewer_path_feature_dictionary.md",
        inputs.path_dictionary,
        status=path_dictionary_status,
    )

    available_representations = [
        representation
        for representation in representations
        if availability["representations"].get(representation, {}).get("status") == "available"
    ]
    if not available_representations:
        hashes = _output_file_hashes(run_dir)
        manifest = {
            "status": "not_run_data_unavailable",
            "mode": "formal",
            "analysis": "interviewer-side predictive signal explanation",
            "code_commit": _resolve_code_commit(output_root.parent, code_commit),
            "plan_sha256": _hash_variants(output_root / "15_interviewer_signal_explanation" / "interviewer_signal_explanation_plan.md").get("raw"),
            "missing_representations": list(representations),
            "input_availability": availability,
            "output_file_hashes": hashes,
            "completion_status": "not_run_data_unavailable",
        }
        atomic_write_json(manifest, run_dir / "run_manifest.json")
        return manifest

    repeats = tuple(range(1, int(stability_repeats) + 1))
    source_scores, source_cache, source_tuning = run_source_score_crossfit(
        inputs,
        membership,
        output_root,
        representations=available_representations,
        repeats=repeats,
        inner_folds=inner_folds,
        c_grid=DEFAULT_C_GRID,
        embedding_root=embedding_root,
        base_seed=base_seed,
    )
    explanation_predictions, explanation_metrics_table, explanation_tuning, full_explanation_cache = run_direct_explanation(
        inputs,
        source_cache,
        membership,
        representations=available_representations,
        repeats=repeats,
        inner_folds=inner_folds,
        alpha_grid=alpha_grid,
        base_seed=base_seed,
    )
    recoverability_fold_metrics_table = recoverability_fold_metrics(explanation_predictions)
    recoverability_scale_sensitivity_table = recoverability_scale_sensitivity(explanation_predictions)
    label_only_predictions, label_only_metrics = run_label_only_recoverability(
        source_cache,
        labels=inputs.labels,
        participant_ids=inputs.participant_ids,
        representations=available_representations,
        repeats=repeats,
        inner_folds=inner_folds,
        base_seed=base_seed,
    )
    label_conditioned_predictions, label_conditioned_metrics = run_label_conditioned_recoverability(
        inputs,
        source_cache,
        membership,
        representations=available_representations,
        repeats=repeats,
        inner_folds=inner_folds,
        alpha_grid=alpha_grid,
        base_seed=base_seed,
    )
    label_conditioned_metrics = add_label_conditioned_interpretation_flags(
        label_conditioned_metrics,
        explanation_metrics_table,
    )
    shapley = run_block_shapley(
        explanation_predictions,
        main_repeat=main_repeat,
        n_bootstrap=n_bootstrap,
        seed=base_seed,
    )
    residual_predictions, residual_metrics, residual_deltas = run_residual_signal(
        inputs,
        source_cache,
        full_explanation_cache,
        membership,
        representations=available_representations,
        repeats=repeats,
        inner_folds=inner_folds,
        c_grid=label_c_grid,
        base_seed=base_seed,
        main_repeat=main_repeat,
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
    )
    (
        fake_assignments,
        fake_explanation_results,
        fake_residual_results,
        fake_explanation_oof,
        fake_residual_oof,
    ) = run_fake_domain_controls(
        inputs,
        source_cache,
        membership,
        representations=available_representations,
        main_repeat=main_repeat,
        repeats=repeats,
        n_draws=fake_draws,
        inner_folds=inner_folds,
        label_c_grid=label_c_grid,
        alpha_grid=alpha_grid,
        base_seed=base_seed,
        return_oof=True,
    )
    fake_summary = fake_domain_summary(
        fake_explanation_results,
        fake_residual_results,
        fake_explanation_oof,
        fake_residual_oof,
        n_bootstrap=n_bootstrap,
        seed=base_seed,
    )
    pairing_source_data = load_source_representation_data(
        output_root,
        inputs.participant_ids,
        inputs.labels,
        representations=available_representations,
        embedding_root=embedding_root,
    )
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
        source_cache,
        membership,
        representations=available_representations,
        repeats=repeats,
        source_data=pairing_source_data,
        inner_folds=inner_folds,
        c_grid=label_c_grid,
        n_draws=pairing_draws,
        base_seed=base_seed,
        main_repeat=main_repeat,
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
    )
    repeat_stability = summarise_repeat_stability(
        explanation_predictions,
        residual_deltas,
        pairing_deltas,
        fake_explanation_results,
        seed=base_seed,
    )
    global_fdr = _write_global_fdr(
        explanation_predictions,
        residual_deltas,
        fake_summary,
        pairing_deltas,
        main_repeat=main_repeat,
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
    )

    tables = {
        "source_score_crossfit.csv": source_scores,
        "explanation_oof_predictions.csv": explanation_predictions,
        "explanation_subset_metrics.csv": explanation_metrics_table,
        "explanation_tuning.csv": explanation_tuning,
        "recoverability_fold_metrics.csv": recoverability_fold_metrics_table,
        "recoverability_scale_sensitivity.csv": recoverability_scale_sensitivity_table,
        "label_only_recoverability.csv": label_only_predictions,
        "label_only_recoverability_metrics.csv": label_only_metrics,
        "label_conditioned_recoverability_oof.csv": label_conditioned_predictions,
        "label_conditioned_recoverability_metrics.csv": label_conditioned_metrics,
        "explanation_block_shapley.csv": shapley,
        "residual_signal_oof_predictions.csv": residual_predictions,
        "residual_signal_metrics.csv": residual_metrics,
        "residual_signal_deltas.csv": residual_deltas,
        "fake_domain_assignments.csv": fake_assignments,
        "fake_domain_explanation_results.csv": fake_explanation_results,
        "fake_domain_residual_results.csv": fake_residual_results,
        "fake_domain_explanation_oof_predictions.csv": fake_explanation_oof,
        "fake_domain_residual_oof_predictions.csv": fake_residual_oof,
        "fake_domain_summary.csv": fake_summary,
        "pairing_assignments.csv": pairing_assignments,
        "pairing_balance.csv": pairing_balance,
        "pairing_pairwise_balance.csv": pairing_pairwise_balance,
        "pairing_random_reference.csv": pairing_random_reference,
        "pairing_oof_predictions.csv": pairing_predictions,
        "pairing_metrics.csv": pairing_metrics,
        "pairing_deltas.csv": pairing_deltas,
        "pairing_draw_stability.csv": pairing_stability,
        "pairing_inner_training_quality.csv": pairing_inner_quality,
        "explanation_repeat_stability.csv": repeat_stability,
        "global_fdr_sensitivity.csv": global_fdr,
    }
    for name, table in tables.items():
        public_output_columns_are_safe(table.columns)
        atomic_write_csv(table, run_dir / name)

    generate_analysis_figures(
        run_dir,
        explanation_metrics_table=explanation_metrics_table,
        shapley_table=shapley,
        residual_deltas=residual_deltas,
        pairing_metrics=pairing_metrics,
        fake_explanation_results=fake_explanation_results,
        main_repeat=main_repeat,
    )

    optimal_balance = pairing_balance.loc[pairing_balance["assignment_type"].eq("optimal_matched")]
    matching_adequate = bool(optimal_balance["matching_adequate"].all()) if not optimal_balance.empty else False
    training_match_quality_limited = bool(
        pairing_stability["training_match_quality_limited"].any()
    ) if not pairing_stability.empty and "training_match_quality_limited" in pairing_stability.columns else True
    if training_match_quality_limited:
        status = "training_match_quality_limited"
    elif not matching_adequate:
        status = "matching_not_adequate_for_primary_interpretation"
    else:
        status = "complete"
    input_hashes = {
        name: availability["inputs"][name].get("sha256", {})
        for name in availability["inputs"]
    }
    plan_path = output_root / "15_interviewer_signal_explanation" / "interviewer_signal_explanation_plan.md"
    output_hashes = _output_file_hashes(run_dir)
    manifest = {
        "status": status,
        "mode": "formal",
        "analysis": "interviewer-side predictive signal explanation",
        "code_commit": _resolve_code_commit(output_root.parent, code_commit),
        "plan_sha256": _hash_variants(plan_path).get("raw"),
        "cohort": {
            "n": int(len(inputs.labels)),
            "positive": int(inputs.labels.sum()),
            "negative": int((1 - inputs.labels).sum()),
            "participant_ids_sha256": hashlib.sha256("\n".join(str(value) for value in inputs.participant_ids).encode("utf-8")).hexdigest(),
        },
        "split_hash": availability["inputs"]["splits"].get("sha256", {}),
        "input_hashes": input_hashes,
        "input_availability": availability,
        "feature_blocks": {
            block: list(columns)
            for block, columns in inputs.block_columns.items()
        },
        "dropped_constant_features": {
            "structural": inputs.structural_audit_summary.get("dropped_features", []),
            "path": inputs.path_audit_summary.get("dropped_features", []),
            "D-P": inputs.domain_audit_summary.get("dropped_features", []),
        },
        "models": {
            "source": {"tfidf_C": float(TFIDF_SOURCE_C), "dense_C_grid": list(DEFAULT_C_GRID)},
            "explanation": {"type": "Ridge", "alpha_grid": list(alpha_grid), "inner_folds": int(inner_folds), "scaling": "training_fold_only"},
            "label": {"type": "LogisticRegression", "C_grid": list(label_c_grid), "class_weight": "balanced", "solver": "liblinear", "max_iter": 2000, "scaling": "training_fold_only"},
        },
        "recoverability_scale": {
            "raw_and_outer_training_standardized_fold_metrics": True,
            "substantive_threshold": 0.05,
            "sensitive_if": [
                "abs(raw_r2-standardized_r2)>=0.05",
                "abs(raw_spearman-standardized_spearman)>=0.05",
                "raw_r2_and_standardized_r2_have_opposite_signs",
            ],
            "participant_logit_scale_fields": [
                "outer_train_participant_scale_mean",
                "outer_train_participant_scale_sd",
                "participant_logit_outer_train_standardized",
            ],
            "standardization_fit_scope": "outer_training_source_scores_only",
        },
        "label_alignment_recoverability": {
            "label_only_class_mean_scope": "outer_training_only_for_outer_test; inner_training_only_for_outer_train",
            "label_conditioned_class_mean_scope": "outer_training_only_for_outer_test; inner_training_only_for_outer_train",
            "high_total_r2_threshold": float(RECOVERABILITY_HIGH_R2_THRESHOLD),
            "near_zero_conditioned_r2_threshold": float(RECOVERABILITY_NEAR_ZERO_R2_THRESHOLD),
            "interpretation_scope": "common_outcome_alignment_sensitivity_not_deployment",
        },
        "inference": {"bootstrap_resamples": int(n_bootstrap), "permutation_resamples": int(n_permutations), "primary_fdr_family": "four fixed TF-IDF tests", "main_repeat": int(main_repeat), "stability_repeats": list(repeats)},
        "fake_D": {"draws": int(fake_draws), "rowwise_partition_permutation": True, "label_free": True, "D_definition": "D-P"},
        "matching": {
            "optimal_draw": 0,
            "random_reference_draws": int(pairing_draws),
            "method": "Hungarian standardized Euclidean with forbidden diagonal",
            "pca_components": 0,
            "gate": {
                "optimal_mean_distance_lt_random_q05": True,
                "minimum_features_mean_abs_pair_difference_lt_0_50": 5,
                "required_no_self": True,
                "required_one_to_one": True,
                "minimum_optimal_inner_pass_rate": 0.80,
            },
            "training_match_quality_limited": training_match_quality_limited,
            "source_mismatch_stage": "before_source_prediction",
            "inner_training_source_fit": "inner_training_only",
            "outer_test_source_fit": "outer_training_only",
            "precomputed_oof_logit_reassignment": False,
            "status": "adequate" if matching_adequate else "matching_not_adequate_for_primary_interpretation",
        },
        "seeds": {"base_seed": int(base_seed), "source_seed_formula": "base + repeat*100000 + fold*1000 + source_offset", "bootstrap": int(base_seed), "permutation": int(base_seed) + 1_000_003},
        "output_file_hashes": output_hashes,
        "output_inventory_sha256": hashlib.sha256(json.dumps(output_hashes, sort_keys=True).encode("utf-8")).hexdigest(),
        "completion_status": status,
    }
    atomic_write_json(manifest, run_dir / "run_manifest.json")
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run interviewer-side predictive signal explanation analysis")
    parser.add_argument("--output-root", type=Path, default=Path("analysis_v2"))
    parser.add_argument("--embedding-root", type=Path, default=None)
    parser.add_argument("--mode", choices=("formal", "smoke"), default="formal")
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--code-commit", default=None)
    parser.add_argument("--representations", nargs="+", choices=("tfidf", "mpnet", "bge"), default=("tfidf", "mpnet", "bge"))
    parser.add_argument("--expected-n", type=int, default=EXPECTED_N)
    parser.add_argument("--expected-repeats", type=int, default=10)
    parser.add_argument("--main-repeat", type=int, default=1)
    parser.add_argument("--stability-repeats", type=int, default=10)
    parser.add_argument("--inner-folds", type=int, default=INNER_FOLDS)
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--permutations", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--fake-draws", type=int, default=FAKE_D_DRAWS)
    parser.add_argument("--pairing-draws", type=int, default=PAIRING_DRAWS)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--verify-reruns", action="store_true")
    parser.add_argument("--promote-final", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output_root = Path(args.output_root).resolve()
    if args.verify_reruns:
        output_base = output_root / "15_interviewer_signal_explanation"
        record = verify_explanation_reruns(
            output_base,
            promote_final=args.promote_final,
        )
        print(json.dumps(record, ensure_ascii=False, sort_keys=True))
        return 0 if record["status"] == "verified" else 1
    manifest = run_interviewer_signal_explanation(
        output_root,
        mode=args.mode,
        run_id=args.run_id,
        code_commit=args.code_commit,
        embedding_root=args.embedding_root,
        representations=args.representations,
        expected_n=args.expected_n,
        expected_repeats=args.expected_repeats,
        main_repeat=args.main_repeat,
        stability_repeats=args.stability_repeats,
        inner_folds=args.inner_folds,
        n_bootstrap=args.bootstrap,
        n_permutations=args.permutations,
        fake_draws=args.fake_draws,
        pairing_draws=args.pairing_draws,
        base_seed=args.seed,
    )
    print(json.dumps({"status": manifest["status"], "mode": manifest["mode"], "run_id": args.run_id}, ensure_ascii=False, sort_keys=True))
    return 0 if manifest["status"] in {
        "complete",
        "smoke_test",
        "matching_not_adequate_for_primary_interpretation",
        "training_match_quality_limited",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
