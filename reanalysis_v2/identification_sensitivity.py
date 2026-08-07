"""Post-audit identification sensitivities for the interviewer increment.

This module is deliberately separate from :mod:`embedding_incremental`.  The
formal four-level result files are immutable; these utilities create new,
explicitly labelled sensitivity inputs and run one requested level at a time.
No labels are used to construct interviewer permutations or Fake-D controls.
"""

from __future__ import annotations

from dataclasses import replace
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from .embedding_incremental import (
    DEFAULT_C_GRID,
    DOMAIN_FEATURES,
    FUSION_METHODS,
    MODEL_SPECS,
    PROTOCOL_FEATURES,
    FrozenEmbeddingPair,
    FrozenIncrementalInputs,
    _early_design,
    _fit_classifier_probability,
    _late_design,
    _nested_source_logits,
    _tune_classifier_c,
    make_early_concat_classifier,
    make_inner_splits,
    make_late_fusion_classifier,
    split_indices,
)


D_PHQ_FEATURES = (
    "anhedonia_interest",
    "appetite_weight",
    "concentration_psychomotor",
    "depressed_mood",
    "self_worth_guilt",
    "sleep_fatigue_energy",
    "suicide_self_harm",
)
D_EXTRA_FEATURES = (
    "functioning_impairment",
    "mental_health_history",
    "protective_or_absent_symptom",
)
PAIRING_MATCH_COLUMNS = (
    "length_only__interviewer_word_count",
    "protocol_structure__interviewer_question_count",
    "protocol_structure__clinical_question_count",
    "protocol_structure__unique_protocol_prompt_count",
    "protocol_structure__clinical_prompt_coverage_count",
    "protocol_structure__non_explicit_interviewer_turn_count",
)


def partition_domains(columns: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the pre-locked PHQ-aligned and extra D-P domain groups."""

    values = [str(column) for column in columns]
    if len(values) != len(set(values)):
        raise ValueError("domain columns contain a duplicate")
    unknown = sorted(set(values) - set(DOMAIN_FEATURES))
    if unknown:
        raise ValueError(f"unknown domain columns: {unknown}")
    if set(values) != set(DOMAIN_FEATURES):
        missing = sorted(set(DOMAIN_FEATURES) - set(values))
        raise ValueError(f"domain columns are incomplete; missing={missing}")
    return D_PHQ_FEATURES, D_EXTRA_FEATURES


def _validate_permutation(permutation: Sequence[int], n: int, *, name: str) -> np.ndarray:
    result = np.asarray(permutation, dtype=int)
    if result.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},), got {result.shape}")
    if sorted(result.tolist()) != list(range(n)):
        raise ValueError(f"{name} must be a permutation of 0..{n - 1}")
    return result


def make_derangement(n: int, *, seed: int) -> np.ndarray:
    """Create a deterministic, label-blind permutation with no fixed points."""

    n = int(n)
    if n < 2:
        raise ValueError("a derangement requires at least two rows")
    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(n)
    for shift in range(n):
        candidate = np.roll(permutation, shift)
        if not np.any(candidate == np.arange(n)):
            return candidate.astype(int)
    # A cyclic shift is a guaranteed fallback if every random cyclic shift
    # happened to contain a fixed point.
    return np.roll(np.arange(n), 1).astype(int)


def make_fake_domain_permutation(n: int, *, seed: int) -> np.ndarray:
    """Return the fixed non-identity row permutation used by Fake-D."""

    return make_derangement(n, seed=seed)


def make_structure_matched_permutation(
    structural: pd.DataFrame,
    *,
    columns: Sequence[str] = PAIRING_MATCH_COLUMNS,
    seed: int,
) -> np.ndarray:
    """Match interviewer donors one-to-one using frozen structural variables.

    The assignment minimizes standardized Euclidean distance with the
    recipient/donor diagonal forbidden.  A tiny seeded jitter only resolves
    ties; labels and text outcomes are never read.
    """

    frame = structural.copy()
    if "participant_id" in frame.columns and frame["participant_id"].duplicated().any():
        raise ValueError("structural participant IDs must be unique")
    columns = tuple(str(column) for column in columns)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"structure matching columns are missing: {missing}")
    values = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("at least two structural rows are required")
    if not np.isfinite(values).all():
        raise ValueError("structure matching columns contain missing or non-finite values")
    scale = values.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    standardized = (values - values.mean(axis=0)) / scale
    cost = np.square(standardized[:, None, :] - standardized[None, :, :]).sum(axis=2)
    np.fill_diagonal(cost, np.inf)
    rng = np.random.default_rng(int(seed))
    finite = np.isfinite(cost)
    cost[finite] += rng.uniform(0.0, 1e-9, size=int(finite.sum()))
    row_index, donor_index = linear_sum_assignment(cost)
    result = np.empty(len(values), dtype=int)
    result[row_index] = donor_index
    return _validate_permutation(result, len(values), name="structure-matched donor mapping")


def permute_interviewer_pair(
    pair: FrozenEmbeddingPair, permutation: Sequence[int]
) -> FrozenEmbeddingPair:
    """Return a pair with participant rows fixed and interviewer rows reordered."""

    permutation = _validate_permutation(permutation, len(pair.participant), name="interviewer permutation")
    if len(pair.interviewer) != len(pair.participant):
        raise ValueError("participant and interviewer embedding rows must agree")
    return replace(pair, interviewer=np.asarray(pair.interviewer)[permutation].copy())


def replace_domains(
    inputs: FrozenIncrementalInputs,
    columns: Sequence[str],
    *,
    matrix: np.ndarray | None = None,
) -> FrozenIncrementalInputs:
    """Return frozen inputs with a selected domain matrix and column contract."""

    columns = tuple(str(column) for column in columns)
    unknown = sorted(set(columns) - set(DOMAIN_FEATURES))
    if unknown or len(columns) != len(set(columns)) or not columns:
        raise ValueError(f"invalid domain subset: {columns}")
    indices = [DOMAIN_FEATURES.index(column) for column in columns]
    selected = inputs.domains[:, indices] if matrix is None else np.asarray(matrix, dtype=float)
    if selected.shape != (len(inputs.participant_ids), len(columns)):
        raise ValueError("replacement domain matrix does not match the participant/domain shape")
    if not np.isfinite(selected).all():
        raise ValueError("replacement domain matrix contains non-finite values")
    return replace(inputs, domains=selected.copy(), domain_columns=columns)


def _validate_level_and_inputs(
    pair: FrozenEmbeddingPair,
    inputs: FrozenIncrementalInputs,
    membership: pd.DataFrame,
    *,
    level: str,
    fusion_method: str,
) -> None:
    if level not in MODEL_SPECS:
        raise ValueError(f"unknown model level: {level}")
    if fusion_method not in FUSION_METHODS:
        raise ValueError(f"unknown fusion method: {fusion_method}")
    n = len(inputs.participant_ids)
    if pair.participant.shape[0] != n or pair.interviewer.shape[0] != n:
        raise ValueError("embedding and input row counts do not agree")
    if pair.participant.shape[1] != pair.embedding_dim or pair.interviewer.shape[1] != pair.embedding_dim:
        raise ValueError("embedding dimensions do not agree with embedding_dim")
    if not np.isfinite(pair.participant).all() or not np.isfinite(pair.interviewer).all():
        raise ValueError("embedding matrices contain non-finite values")
    if sorted(pd.to_numeric(membership.loc[membership["repeat"].eq(1), "fold"]).unique().tolist()) != [1, 2, 3, 4, 5]:
        raise ValueError("membership must contain five folds")


def run_single_level_oof(
    *,
    pair: FrozenEmbeddingPair,
    inputs: FrozenIncrementalInputs,
    membership: pd.DataFrame,
    repeat: int,
    level: str,
    augmented: bool,
    fusion_method: str,
    inner_folds: int,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    base_seed: int,
) -> pd.DataFrame:
    """Run one locked model level and return one OOF prediction per participant."""

    _validate_level_and_inputs(pair, inputs, membership, level=level, fusion_method=fusion_method)
    fold_values = sorted(
        pd.to_numeric(membership.loc[membership["repeat"].eq(int(repeat)), "fold"])
        .astype(int)
        .unique()
        .tolist()
    )
    if fold_values != [1, 2, 3, 4, 5]:
        raise ValueError(f"repeat {repeat} must contain exactly five outer folds")
    rows: list[dict[str, object]] = []
    for fold in fold_values:
        train_idx, test_idx = split_indices(membership, int(repeat), fold, inputs.participant_ids)
        y_train = inputs.labels[train_idx]
        seed = int(base_seed) + int(repeat) * 100_000 + fold * 1_000
        train_protocol = inputs.protocol[train_idx]
        test_protocol = inputs.protocol[test_idx]
        train_domains = inputs.domains[train_idx]
        test_domains = inputs.domains[test_idx]
        if fusion_method == "late_fusion":
            p_train, p_test, _ = _nested_source_logits(
                pair.participant[train_idx],
                y_train,
                pair.participant[test_idx],
                inner_folds=inner_folds,
                c_grid=c_grid,
                seed=seed + 11,
            )
            i_train = i_test = None
            if augmented:
                i_train, i_test, _ = _nested_source_logits(
                    pair.interviewer[train_idx],
                    y_train,
                    pair.interviewer[test_idx],
                    inner_folds=inner_folds,
                    c_grid=c_grid,
                    seed=seed + 22,
                )
            X_train = _late_design(
                p_train,
                i_train,
                train_protocol,
                train_domains,
                level,
                augmented,
            )
            X_test = _late_design(
                p_test,
                i_test,
                test_protocol,
                test_domains,
                level,
                augmented,
            )
            classifier_factory = lambda c, seed: make_late_fusion_classifier(c=c, seed=seed)
        else:
            X_train = _early_design(
                pair.participant[train_idx],
                pair.interviewer[train_idx] if augmented else None,
                train_protocol,
                train_domains,
                level,
                augmented,
            )
            X_test = _early_design(
                pair.participant[test_idx],
                pair.interviewer[test_idx] if augmented else None,
                test_protocol,
                test_domains,
                level,
                augmented,
            )
            embedding_count = pair.embedding_dim * (2 if augmented else 1)
            numeric_count = (
                (inputs.protocol.shape[1] if level in {"L2", "L4"} else 0)
                + (inputs.domains.shape[1] if level in {"L3", "L4"} else 0)
            )
            classifier_factory = lambda c, seed, embedding_count=embedding_count, numeric_count=numeric_count: make_early_concat_classifier(
                embedding_columns=list(range(embedding_count)),
                numeric_columns=list(range(embedding_count, embedding_count + numeric_count)),
                c=c,
                seed=seed,
            )
        inner_splits = make_inner_splits(y_train, n_splits=inner_folds, seed=seed + 33)
        # Importing the private tuner is intentional: this module must use the
        # same fold-contained C selection as the frozen formal runner.
        from .embedding_incremental import _tune_classifier_c

        selected, _ = _tune_classifier_c(
            X_train,
            y_train,
            inner_splits,
            c_grid=c_grid,
            seed=seed + 100 + int(level[1]) * 10 + int(augmented),
            classifier_factory=classifier_factory,
        )
        probability, _ = _fit_classifier_probability(
            X_train,
            y_train,
            X_test,
            c=selected,
            seed=seed + 500 + int(level[1]) * 10 + int(augmented),
            classifier_factory=classifier_factory,
        )
        for local_index, global_index in enumerate(test_idx):
            rows.append(
                {
                    "participant_id": inputs.participant_ids[global_index],
                    "label": int(inputs.labels[global_index]),
                    "repeat": int(repeat),
                    "outer_fold": int(fold),
                    "embedding_model": pair.slug,
                    "fusion_method": fusion_method,
                    "model_level": level,
                    "model_name": "augmented" if augmented else "base",
                    "probability": float(probability[local_index]),
                    "selected_c": float(selected),
                }
            )
    result = pd.DataFrame(rows).sort_values("participant_id", kind="stable").reset_index(drop=True)
    if result["participant_id"].nunique() != len(inputs.participant_ids):
        raise ValueError("single-level OOF output does not cover every participant")
    return result


def run_level_pair_oof(
    *,
    pair: FrozenEmbeddingPair,
    inputs: FrozenIncrementalInputs,
    membership: pd.DataFrame,
    repeat: int,
    level: str,
    fusion_method: str,
    inner_folds: int,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    base_seed: int,
) -> pd.DataFrame:
    """Run base and interviewer-augmented models while sharing each fold's logits."""

    _validate_level_and_inputs(pair, inputs, membership, level=level, fusion_method=fusion_method)
    fold_values = sorted(
        pd.to_numeric(membership.loc[membership["repeat"].eq(int(repeat)), "fold"])
        .astype(int)
        .unique()
        .tolist()
    )
    if fold_values != [1, 2, 3, 4, 5]:
        raise ValueError(f"repeat {repeat} must contain exactly five outer folds")
    rows: list[dict[str, object]] = []
    for fold in fold_values:
        train_idx, test_idx = split_indices(membership, int(repeat), fold, inputs.participant_ids)
        y_train = inputs.labels[train_idx]
        seed = int(base_seed) + int(repeat) * 100_000 + fold * 1_000
        train_protocol = inputs.protocol[train_idx]
        test_protocol = inputs.protocol[test_idx]
        train_domains = inputs.domains[train_idx]
        test_domains = inputs.domains[test_idx]
        if fusion_method == "late_fusion":
            p_train, p_test, _ = _nested_source_logits(
                pair.participant[train_idx],
                y_train,
                pair.participant[test_idx],
                inner_folds=inner_folds,
                c_grid=c_grid,
                seed=seed + 11,
            )
            i_train, i_test, _ = _nested_source_logits(
                pair.interviewer[train_idx],
                y_train,
                pair.interviewer[test_idx],
                inner_folds=inner_folds,
                c_grid=c_grid,
                seed=seed + 22,
            )
        inner_splits = make_inner_splits(y_train, n_splits=inner_folds, seed=seed + 33)
        for augmented in (False, True):
            if fusion_method == "late_fusion":
                X_train = _late_design(
                    p_train,
                    i_train if augmented else None,
                    train_protocol,
                    train_domains,
                    level,
                    augmented,
                )
                X_test = _late_design(
                    p_test,
                    i_test if augmented else None,
                    test_protocol,
                    test_domains,
                    level,
                    augmented,
                )
                classifier_factory = lambda c, seed: make_late_fusion_classifier(c=c, seed=seed)
            else:
                X_train = _early_design(
                    pair.participant[train_idx],
                    pair.interviewer[train_idx] if augmented else None,
                    train_protocol,
                    train_domains,
                    level,
                    augmented,
                )
                X_test = _early_design(
                    pair.participant[test_idx],
                    pair.interviewer[test_idx] if augmented else None,
                    test_protocol,
                    test_domains,
                    level,
                    augmented,
                )
                embedding_count = pair.embedding_dim * (2 if augmented else 1)
                numeric_count = (
                    (inputs.protocol.shape[1] if level in {"L2", "L4"} else 0)
                    + (inputs.domains.shape[1] if level in {"L3", "L4"} else 0)
                )
                classifier_factory = lambda c, seed, embedding_count=embedding_count, numeric_count=numeric_count: make_early_concat_classifier(
                    embedding_columns=list(range(embedding_count)),
                    numeric_columns=list(range(embedding_count, embedding_count + numeric_count)),
                    c=c,
                    seed=seed,
                )
            selected, _ = _tune_classifier_c(
                X_train,
                y_train,
                inner_splits,
                c_grid=c_grid,
                seed=seed + 100 + int(level[1]) * 10 + int(augmented),
                classifier_factory=classifier_factory,
            )
            probability, _ = _fit_classifier_probability(
                X_train,
                y_train,
                X_test,
                c=selected,
                seed=seed + 500 + int(level[1]) * 10 + int(augmented),
                classifier_factory=classifier_factory,
            )
            for local_index, global_index in enumerate(test_idx):
                rows.append(
                    {
                        "participant_id": inputs.participant_ids[global_index],
                        "label": int(inputs.labels[global_index]),
                        "repeat": int(repeat),
                        "outer_fold": int(fold),
                        "embedding_model": pair.slug,
                        "fusion_method": fusion_method,
                        "model_level": level,
                        "model_name": "augmented" if augmented else "base",
                        "probability": float(probability[local_index]),
                        "selected_c": float(selected),
                    }
                )
    result = pd.DataFrame(rows).sort_values(
        ["model_name", "participant_id"], kind="stable"
    ).reset_index(drop=True)
    expected = len(inputs.participant_ids) * 2
    if len(result) != expected or result.groupby("model_name")["participant_id"].nunique().min() != len(inputs.participant_ids):
        raise ValueError("level-pair OOF output does not cover every model and participant")
    return result


def prediction_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    """Compute the four locked participant-level prediction metrics."""

    y = predictions["label"].to_numpy(dtype=int)
    probability = predictions["probability"].to_numpy(dtype=float)
    return {
        "auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, np.column_stack([1 - probability, probability]), labels=[0, 1])),
    }
