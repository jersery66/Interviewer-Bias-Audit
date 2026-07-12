"""Locked conditional-increment analysis for frozen MPNet/BGE embeddings.

The runner in this module deliberately consumes already materialised embedding
cache files.  It never downloads a model or re-encodes text.  This keeps the
post-submission analysis auditable and prevents a silent change in the text
representation from being mixed with the conditional-increment question.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from .inference import apply_bh_by_family, paired_permutation_test
from .io import atomic_write_csv, atomic_write_json, sha256_file
from .modeling import make_dense_classifier, make_inner_splits, tune_dense_c
from .splits import split_indices


EXPECTED_N = 142
BASE_SEED = 20260624
DEFAULT_C_GRID = (0.01, 0.1, 1.0, 10.0)
PROTOCOL_FEATURES = (
    "protocol_structure__interviewer_question_count",
    "protocol_structure__clinical_question_count",
    "protocol_structure__unique_protocol_prompt_count",
    "protocol_structure__clinical_prompt_coverage_count",
    "protocol_structure__non_explicit_interviewer_turn_count",
)
DOMAIN_FEATURES = (
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
MODEL_SPECS = {
    "L1": ("P", "P+I"),
    "L2": ("P+R", "P+R+I"),
    "L3": ("P+D", "P+D+I"),
    "L4": ("P+R+D", "P+R+D+I"),
}
FUSION_METHODS = ("late_fusion", "early_concat")
SOURCE_CONDITIONS = ("participant_speech", "interviewer_speech")


@dataclass(frozen=True)
class FrozenIncrementalInputs:
    participant_ids: np.ndarray
    labels: np.ndarray
    protocol: np.ndarray
    domains: np.ndarray
    protocol_columns: tuple[str, ...] = PROTOCOL_FEATURES
    domain_columns: tuple[str, ...] = DOMAIN_FEATURES


@dataclass(frozen=True)
class FrozenEmbeddingPair:
    slug: str
    model_name: str
    revision: str
    embedding_dim: int
    participant: np.ndarray
    interviewer: np.ndarray
    cache_files: tuple[dict[str, str], ...]


def build_model_specifications() -> dict[str, tuple[str, str]]:
    """Return the frozen base/augmented design names in analysis order."""

    return dict(MODEL_SPECS)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _check_unique_ids(frame: pd.DataFrame, name: str) -> np.ndarray:
    _require_columns(frame, ("participant_id",), name)
    ids = pd.to_numeric(frame["participant_id"], errors="raise").to_numpy()
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{name} participant IDs must be unique")
    return ids


def validate_incremental_inputs(
    participant_frame: pd.DataFrame,
    protocol_frame: pd.DataFrame,
    domain_frame: pd.DataFrame,
    *,
    expected_n: int = EXPECTED_N,
) -> pd.DataFrame:
    """Validate and merge the frozen cohort, protocol and domain tables.

    No rows are imputed or selected here.  Missing values are rejected because
    the locked plan defines complete frozen process/domain inputs; callers that
    intentionally want another missing-value policy must lock a new plan.
    """

    participant_ids = _check_unique_ids(participant_frame, "participant table")
    protocol_ids = _check_unique_ids(protocol_frame, "protocol table")
    domain_ids = _check_unique_ids(domain_frame, "domain table")
    if not np.array_equal(np.sort(participant_ids), np.sort(protocol_ids)):
        raise ValueError("participant IDs do not match protocol IDs")
    if not np.array_equal(np.sort(participant_ids), np.sort(domain_ids)):
        raise ValueError("participant IDs do not match domain IDs")
    if expected_n is not None and len(participant_ids) != int(expected_n):
        raise ValueError(f"expected {expected_n} participants, found {len(participant_ids)}")

    _require_columns(participant_frame, ("label",), "participant table")
    _require_columns(protocol_frame, PROTOCOL_FEATURES, "protocol table")
    _require_columns(domain_frame, DOMAIN_FEATURES, "domain table")

    participant = participant_frame[["participant_id", "label"]].copy()
    participant["participant_id"] = pd.to_numeric(participant["participant_id"], errors="raise")
    participant["label"] = pd.to_numeric(participant["label"], errors="raise").astype(int)
    if not participant["label"].isin([0, 1]).all():
        raise ValueError("labels must be binary 0/1")

    protocol = protocol_frame[["participant_id", *PROTOCOL_FEATURES]].copy()
    domains = domain_frame[["participant_id", *DOMAIN_FEATURES]].copy()
    result = participant.merge(protocol, on="participant_id", validate="one_to_one")
    result = result.merge(domains, on="participant_id", validate="one_to_one")
    result = result.sort_values("participant_id", kind="stable").reset_index(drop=True)
    numeric = result[[*PROTOCOL_FEATURES, *DOMAIN_FEATURES]].apply(
        pd.to_numeric, errors="coerce"
    )
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        bad = numeric.columns[numeric.isna().any()].tolist()
        raise ValueError(f"protocol/domain features contain missing or non-finite values: {bad}")
    result[[*PROTOCOL_FEATURES, *DOMAIN_FEATURES]] = numeric
    return result


def load_frozen_incremental_inputs(
    output_root: str | Path, *, expected_n: int = EXPECTED_N
) -> FrozenIncrementalInputs:
    """Load the frozen structural baseline and ten-domain count tables."""

    output_root = Path(output_root)
    structural_path = output_root / "12_submission_audit" / "structural_baseline_features.csv"
    domain_path = output_root / "04_c5_controls" / "domain_count" / "input.csv"
    if not structural_path.exists():
        raise FileNotFoundError(f"frozen structural input not found: {structural_path}")
    if not domain_path.exists():
        raise FileNotFoundError(f"frozen domain-count input not found: {domain_path}")
    structural = pd.read_csv(structural_path)
    domains = pd.read_csv(domain_path)
    merged = validate_incremental_inputs(
        structural[["participant_id", "label", *PROTOCOL_FEATURES]],
        structural[["participant_id", *PROTOCOL_FEATURES]],
        domains,
        expected_n=expected_n,
    )
    return FrozenIncrementalInputs(
        participant_ids=merged["participant_id"].to_numpy(),
        labels=merged["label"].to_numpy(dtype=int),
        protocol=merged[list(PROTOCOL_FEATURES)].to_numpy(dtype=float),
        domains=merged[list(DOMAIN_FEATURES)].to_numpy(dtype=float),
    )


def _validate_matrix(name: str, matrix: np.ndarray, n: int, dim: int | None = None) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != n:
        raise ValueError(f"{name} must have shape ({n}, d), got {matrix.shape}")
    if dim is not None and matrix.shape[1] != int(dim):
        raise ValueError(f"{name} must have dimension {dim}, got {matrix.shape[1]}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def _manifest_entry(manifest: Mapping[str, object], condition: str) -> dict[str, str]:
    for raw in manifest.get("cache_files", []):
        entry = dict(raw)
        if entry.get("condition") == condition:
            return {str(key): str(value) for key, value in entry.items()}
    raise ValueError(f"embedding manifest has no cache entry for {condition}")


def load_frozen_embedding_pair(
    embedding_root: str | Path,
    model_slug: str,
    participant_ids: np.ndarray,
    *,
    expected_dim: int,
    expected_manifest: str | Path | None = None,
) -> FrozenEmbeddingPair:
    """Load participant/interviewer arrays from the existing cache only."""

    embedding_root = Path(embedding_root)
    model_root = embedding_root / model_slug
    manifest_path = Path(expected_manifest) if expected_manifest else model_root / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"frozen embedding manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model = dict(manifest.get("model", {}))
    if int(model.get("embedding_dim", expected_dim)) != int(expected_dim):
        raise ValueError(f"{model_slug} manifest embedding dimension is not {expected_dim}")
    ids = np.asarray(participant_ids)
    matrices: dict[str, np.ndarray] = {}
    cache_records: list[dict[str, str]] = []
    for condition in SOURCE_CONDITIONS:
        entry = _manifest_entry(manifest, condition)
        cache_path = model_root / entry["path"]
        if not cache_path.exists():
            raise FileNotFoundError(
                f"required frozen embedding cache is unavailable: {cache_path}; "
                "the runner does not re-encode text"
            )
        observed_hash = sha256_file(cache_path)
        if entry.get("sha256") and observed_hash != entry["sha256"]:
            raise ValueError(f"embedding cache SHA-256 mismatch: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as cached:
            cached_ids = np.asarray(cached["participant_ids"])
            if not np.array_equal(cached_ids, ids):
                raise ValueError(f"embedding cache participant IDs do not match: {cache_path}")
            matrix = _validate_matrix(
                condition,
                cached["embeddings"],
                len(ids),
                expected_dim,
            )
            norms = np.linalg.norm(matrix, axis=1)
            nonempty = norms > 0
            if nonempty.any() and not np.allclose(norms[nonempty], 1.0, atol=1e-4):
                raise ValueError(f"{condition} embeddings are not L2-normalized")
        matrices[condition] = matrix
        cache_records.append(
            {
                "condition": condition,
                "path": entry["path"],
                "sha256": observed_hash,
                "cache_key": entry.get("cache_key", ""),
            }
        )
    return FrozenEmbeddingPair(
        slug=model_slug,
        model_name=str(model.get("name", model_slug)),
        revision=str(model.get("revision", "")),
        embedding_dim=int(expected_dim),
        participant=matrices["participant_speech"],
        interviewer=matrices["interviewer_speech"],
        cache_files=tuple(cache_records),
    )


def _metric_values(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-8, 1 - 1e-8)
    if len(y_true) != len(probabilities):
        raise ValueError("labels and probabilities must have the same length")
    if len(np.unique(y_true)) < 2:
        raise ValueError("AUC metrics require both outcome classes")
    predictions = probabilities >= 0.5
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if tp + fn else float("nan")
    specificity = float(tn / (tn + fp)) if tn + fp else float("nan")
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "sensitivity_050": sensitivity,
        "specificity_050": specificity,
    }


def paired_metric_deltas(
    y_true: np.ndarray,
    base_probability: np.ndarray,
    augmented_probability: np.ndarray,
) -> dict[str, float]:
    """Return deltas with positive values meaning improved performance."""

    base = _metric_values(y_true, base_probability)
    augmented = _metric_values(y_true, augmented_probability)
    return {
        "base_auc": base["roc_auc"],
        "augmented_auc": augmented["roc_auc"],
        "delta_auc": augmented["roc_auc"] - base["roc_auc"],
        "base_pr_auc": base["pr_auc"],
        "augmented_pr_auc": augmented["pr_auc"],
        "delta_pr_auc": augmented["pr_auc"] - base["pr_auc"],
        "base_brier": base["brier"],
        "augmented_brier": augmented["brier"],
        "delta_brier": base["brier"] - augmented["brier"],
        "base_log_loss": base["log_loss"],
        "augmented_log_loss": augmented["log_loss"],
        "delta_log_loss": base["log_loss"] - augmented["log_loss"],
    }


def _bootstrap_statistic(
    y_true: np.ndarray,
    base_probability: np.ndarray,
    augmented_probability: np.ndarray,
    statistic: str,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=int)
    base_probability = np.asarray(base_probability, dtype=float)
    augmented_probability = np.asarray(augmented_probability, dtype=float)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    attempts = 0
    max_attempts = max(n_bootstrap * 20, 100)
    while len(values) < n_bootstrap and attempts < max_attempts:
        attempts += 1
        sample = rng.integers(0, len(y_true), size=len(y_true))
        if len(np.unique(y_true[sample])) < 2:
            continue
        delta = paired_metric_deltas(
            y_true[sample], base_probability[sample], augmented_probability[sample]
        )
        values.append(float(delta[statistic]))
    if len(values) < max(100, n_bootstrap // 10):
        raise ValueError(f"bootstrap could not obtain enough valid resamples for {statistic}")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def bootstrap_paired_deltas(
    y_true: np.ndarray,
    base_probability: np.ndarray,
    augmented_probability: np.ndarray,
    *,
    n_bootstrap: int = 5000,
    seed: int = BASE_SEED,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for offset, statistic in enumerate(
        ("delta_auc", "delta_pr_auc", "delta_brier", "delta_log_loss")
    ):
        low, high = _bootstrap_statistic(
            y_true,
            base_probability,
            augmented_probability,
            statistic,
            n_bootstrap=n_bootstrap,
            seed=seed + offset,
        )
        result[f"{statistic}_ci_low"] = low
        result[f"{statistic}_ci_high"] = high
    return result


def apply_incremental_fdr(table: pd.DataFrame) -> pd.DataFrame:
    """Apply BH within the explicitly supplied incremental FDR families."""

    required = {"fdr_family", "raw_p"}
    if not required.issubset(table.columns):
        raise ValueError(f"delta table is missing columns: {sorted(required - set(table.columns))}")
    result = table.copy()
    result["bh_q"] = np.nan
    for family, indices in result.groupby("fdr_family", sort=False).groups.items():
        if str(family).startswith("exploratory"):
            continue
        subset = result.loc[list(indices), ["raw_p"]].rename(columns={"raw_p": "p_value"})
        adjusted = apply_bh_by_family(
            subset.assign(family=str(family)), family_column="family", p_column="p_value"
        )["fdr_bh_q"].to_numpy()
        result.loc[list(indices), "bh_q"] = adjusted
    return result


def _fit_dense_probability(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    c: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    estimator = make_dense_classifier(c=float(c), seed=seed)
    estimator.fit(np.asarray(X_train, dtype=float), np.asarray(y_train, dtype=int))
    probability = estimator.predict_proba(np.asarray(X_test, dtype=float))[:, 1]
    score = np.asarray(estimator.decision_function(np.asarray(X_test, dtype=float)), dtype=float)
    return probability, score


def _nested_source_logits(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    inner_folds: int,
    c_grid: Sequence[float],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Generate CF train logits and outer-test logits for one source."""

    inner_splits = make_inner_splits(y_train, n_splits=inner_folds, seed=seed)
    train_logits = np.full(len(y_train), np.nan, dtype=float)
    tuning: list[dict[str, object]] = []
    for inner_fold, (meta_train, meta_validation) in enumerate(inner_splits, start=1):
        nested = make_inner_splits(
            y_train[meta_train], n_splits=inner_folds, seed=seed + 100 + inner_fold
        )
        selected, scores = tune_dense_c(
            X_train[meta_train],
            y_train[meta_train],
            nested,
            c_grid=c_grid,
            seed=seed + 1000 + inner_fold,
        )
        estimator = make_dense_classifier(c=selected, seed=seed + inner_fold)
        estimator.fit(X_train[meta_train], y_train[meta_train])
        train_logits[meta_validation] = estimator.decision_function(
            X_train[meta_validation]
        )
        for c, score in sorted(scores.items()):
            tuning.append(
                {
                    "layer": "source",
                    "inner_fold": inner_fold,
                    "C": float(c),
                    "mean_inner_roc_auc": float(score),
                    "selected": bool(c == selected),
                }
            )
    if not np.isfinite(train_logits).all():
        raise ValueError("source cross-fitted logits are incomplete")
    selected, scores = tune_dense_c(
        X_train,
        y_train,
        inner_splits,
        c_grid=c_grid,
        seed=seed + 2000,
    )
    estimator = make_dense_classifier(c=selected, seed=seed + 3000)
    estimator.fit(X_train, y_train)
    test_logits = estimator.decision_function(X_test)
    tuning.append(
        {
            "layer": "source_outer_full",
            "inner_fold": 0,
            "C": float(selected),
            "mean_inner_roc_auc": float(scores[selected]),
            "selected": True,
        }
    )
    return np.asarray(train_logits, dtype=float), np.asarray(test_logits, dtype=float), tuning


def _scaled_numeric(
    train_protocol: np.ndarray,
    test_protocol: np.ndarray,
    train_domains: np.ndarray,
    test_domains: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    protocol_scaler = StandardScaler().fit(train_protocol)
    domain_scaler = StandardScaler().fit(train_domains)
    return (
        protocol_scaler.transform(train_protocol),
        protocol_scaler.transform(test_protocol),
        domain_scaler.transform(train_domains),
        domain_scaler.transform(test_domains),
    )


def _late_design(
    participant_logits: np.ndarray,
    interviewer_logits: np.ndarray | None,
    protocol: np.ndarray,
    domains: np.ndarray,
    level: str,
    augmented: bool,
) -> np.ndarray:
    columns = [np.asarray(participant_logits).reshape(-1, 1)]
    if level in {"L2", "L4"}:
        columns.append(np.asarray(protocol))
    if level in {"L3", "L4"}:
        columns.append(np.asarray(domains))
    if augmented:
        if interviewer_logits is None:
            raise ValueError("interviewer logits are required for an augmented design")
        columns.append(np.asarray(interviewer_logits).reshape(-1, 1))
    return np.column_stack(columns)


def _early_design(
    participant_embedding: np.ndarray,
    interviewer_embedding: np.ndarray | None,
    protocol: np.ndarray,
    domains: np.ndarray,
    level: str,
    augmented: bool,
) -> np.ndarray:
    columns = [np.asarray(participant_embedding)]
    if level in {"L2", "L4"}:
        columns.append(np.asarray(protocol))
    if level in {"L3", "L4"}:
        columns.append(np.asarray(domains))
    if augmented:
        if interviewer_embedding is None:
            raise ValueError("interviewer embeddings are required for an augmented design")
        columns.append(np.asarray(interviewer_embedding))
    return np.column_stack(columns)


def _run_repeat(
    *,
    pair: FrozenEmbeddingPair,
    inputs: FrozenIncrementalInputs,
    membership: pd.DataFrame,
    repeat: int,
    fusion_method: str,
    inner_folds: int,
    c_grid: Sequence[float],
    base_seed: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    if fusion_method not in FUSION_METHODS:
        raise ValueError(f"unknown fusion method: {fusion_method}")
    rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    fold_values = sorted(
        pd.to_numeric(membership.loc[membership["repeat"].eq(repeat), "fold"])
        .astype(int)
        .unique()
    )
    if fold_values != [1, 2, 3, 4, 5]:
        raise ValueError(f"repeat {repeat} must contain exactly five outer folds")
    for fold in fold_values:
        train_idx, test_idx = split_indices(
            membership, repeat, fold, inputs.participant_ids
        )
        y_train = inputs.labels[train_idx]
        seed = base_seed + repeat * 100_000 + fold * 1_000
        train_protocol, test_protocol, train_domains, test_domains = _scaled_numeric(
            inputs.protocol[train_idx],
            inputs.protocol[test_idx],
            inputs.domains[train_idx],
            inputs.domains[test_idx],
        )
        if fusion_method == "late_fusion":
            p_train, p_test, p_tuning = _nested_source_logits(
                pair.participant[train_idx],
                y_train,
                pair.participant[test_idx],
                inner_folds=inner_folds,
                c_grid=c_grid,
                seed=seed + 11,
            )
            i_train, i_test, i_tuning = _nested_source_logits(
                pair.interviewer[train_idx],
                y_train,
                pair.interviewer[test_idx],
                inner_folds=inner_folds,
                c_grid=c_grid,
                seed=seed + 22,
            )
            for item in p_tuning:
                tuning_rows.append({"repeat": repeat, "fold": fold, "source": "P", **item})
            for item in i_tuning:
                tuning_rows.append({"repeat": repeat, "fold": fold, "source": "I", **item})
            design_builder = lambda level, augmented, train: _late_design(
                p_train if train else p_test,
                i_train if train else i_test,
                train_protocol if train else test_protocol,
                train_domains if train else test_domains,
                level,
                augmented,
            )
        else:
            design_builder = lambda level, augmented, train: _early_design(
                pair.participant[train_idx] if train else pair.participant[test_idx],
                pair.interviewer[train_idx] if train else pair.interviewer[test_idx],
                train_protocol if train else test_protocol,
                train_domains if train else test_domains,
                level,
                augmented,
            )
        second_inner = make_inner_splits(y_train, n_splits=inner_folds, seed=seed + 33)
        for level in MODEL_SPECS:
            for model_name, augmented in (("base", False), ("augmented", True)):
                X_train = design_builder(level, augmented, True)
                X_test = design_builder(level, augmented, False)
                selected, scores = tune_dense_c(
                    X_train,
                    y_train,
                    second_inner,
                    c_grid=c_grid,
                    seed=seed + 100 + list(MODEL_SPECS).index(level) * 10 + int(augmented),
                )
                probability, _ = _fit_dense_probability(
                    X_train,
                    y_train,
                    X_test,
                    c=selected,
                    seed=seed + 500 + list(MODEL_SPECS).index(level) * 10 + int(augmented),
                )
                tuning_rows.extend(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "source": "second_layer",
                        "level": level,
                        "model_name": model_name,
                        "C": float(c),
                        "mean_inner_roc_auc": float(score),
                        "selected": bool(c == selected),
                    }
                    for c, score in sorted(scores.items())
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
                            "model_name": model_name,
                            "probability": float(probability[local_index]),
                            "selected_c": float(selected),
                            "outer_train_n": int(len(train_idx)),
                            "outer_test_n": int(len(test_idx)),
                        }
                    )
    return pd.DataFrame(rows), tuning_rows


def _main_oof_table(predictions: pd.DataFrame, *, repeat: int) -> pd.DataFrame:
    result = predictions.loc[predictions["repeat"].eq(repeat)].copy()
    keys = ["embedding_model", "fusion_method", "model_level", "model_name", "participant_id"]
    if result.duplicated(keys).any():
        raise ValueError("main OOF predictions contain duplicate participant/model rows")
    counts = result.groupby(
        ["embedding_model", "fusion_method", "model_level", "model_name"]
    )["participant_id"].nunique()
    if not counts.eq(len(result["participant_id"].unique())).all():
        raise ValueError("main OOF predictions do not cover every participant")
    return result.sort_values(
        ["embedding_model", "fusion_method", "model_level", "model_name", "participant_id"],
        kind="stable",
    ).reset_index(drop=True)


def _paired_predictions(main_oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = ["embedding_model", "fusion_method", "model_level", "participant_id"]
    base = main_oof.loc[main_oof["model_name"].eq("base")].rename(
        columns={"probability": "base_probability"}
    )
    augmented = main_oof.loc[main_oof["model_name"].eq("augmented")].rename(
        columns={"probability": "augmented_probability"}
    )
    merged = base[key + ["label", "outer_fold", "base_probability"]].merge(
        augmented[key + ["augmented_probability"]], on=key, validate="one_to_one"
    )
    return merged.sort_values(key, kind="stable").reset_index(drop=True), base


def _delta_table(
    paired: pd.DataFrame,
    *,
    n_bootstrap: int,
    n_permutations: int,
    base_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    test_index = 0
    for (embedding, fusion, level), group in paired.groupby(
        ["embedding_model", "fusion_method", "model_level"], sort=False
    ):
        y = group["label"].to_numpy(dtype=int)
        base = group["base_probability"].to_numpy(dtype=float)
        augmented = group["augmented_probability"].to_numpy(dtype=float)
        point = paired_metric_deltas(y, base, augmented)
        ci = bootstrap_paired_deltas(
            y,
            base,
            augmented,
            n_bootstrap=n_bootstrap,
            seed=base_seed + test_index * 100,
        )
        permutation = paired_permutation_test(
            y,
            augmented,
            base,
            roc_auc_score,
            n_permutations=n_permutations,
            seed=base_seed + 50_000 + test_index,
        )
        fdr_family = (
            "primary_l4"
            if fusion == "late_fusion" and level == "L4"
            else "secondary_all_late"
            if fusion == "late_fusion"
            else "exploratory_early_concat"
        )
        rows.append(
            {
                "embedding": embedding,
                "fusion": fusion,
                "level": level,
                "delta_auc": point["delta_auc"],
                "auc_ci_low": ci["delta_auc_ci_low"],
                "auc_ci_high": ci["delta_auc_ci_high"],
                "raw_p": float(permutation["p_value"]),
                "bh_q": np.nan,
                "delta_pr_auc": point["delta_pr_auc"],
                "delta_brier": point["delta_brier"],
                "delta_log_loss": point["delta_log_loss"],
                "pr_auc_ci_low": ci["delta_pr_auc_ci_low"],
                "pr_auc_ci_high": ci["delta_pr_auc_ci_high"],
                "brier_ci_low": ci["delta_brier_ci_low"],
                "brier_ci_high": ci["delta_brier_ci_high"],
                "log_loss_ci_low": ci["delta_log_loss_ci_low"],
                "log_loss_ci_high": ci["delta_log_loss_ci_high"],
                "fdr_family": fdr_family,
                "n": int(len(group)),
                "n_bootstrap": int(n_bootstrap),
                "n_permutations": int(n_permutations),
                "seed": int(base_seed + test_index * 100),
            }
        )
        test_index += 1
    result = apply_incremental_fdr(pd.DataFrame(rows))
    late = result.loc[result["fusion"].eq("late_fusion")].copy()
    late["fdr_family"] = "all_late_delta_auc"
    all_late_q = apply_bh_by_family(
        late.rename(columns={"raw_p": "p_value"}),
        family_column="fdr_family",
        p_column="p_value",
    )[["embedding", "fusion", "level", "fdr_bh_q"]].rename(
        columns={"fdr_bh_q": "bh_q_all_late"}
    )
    return result.merge(all_late_q, on=["embedding", "fusion", "level"], how="left")


def _bootstrap_attenuation(
    y: np.ndarray,
    prediction_by_level: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    metric_key: str,
    left: str,
    right: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    def estimate(indices: np.ndarray) -> float:
        first = paired_metric_deltas(
            y[indices],
            prediction_by_level[left][0][indices],
            prediction_by_level[left][1][indices],
        )[metric_key]
        second = paired_metric_deltas(
            y[indices],
            prediction_by_level[right][0][indices],
            prediction_by_level[right][1][indices],
        )[metric_key]
        return float(first - second)

    point = estimate(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    values: list[float] = []
    attempts = 0
    while len(values) < n_bootstrap and attempts < n_bootstrap * 20:
        attempts += 1
        indices = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[indices])) < 2:
            continue
        values.append(estimate(indices))
    if len(values) < max(100, n_bootstrap // 10):
        raise ValueError(f"bootstrap could not obtain attenuation resamples for {metric_key}")
    return point, float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def _attenuation_table(
    paired: pd.DataFrame,
    *,
    n_bootstrap: int,
    base_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (embedding, fusion), subset in paired.groupby(
        ["embedding_model", "fusion_method"], sort=False
    ):
        levels = {
            level: (
                group["base_probability"].to_numpy(dtype=float),
                group["augmented_probability"].to_numpy(dtype=float),
            )
            for level, group in subset.groupby("model_level", sort=False)
        }
        y = subset.loc[subset["model_level"].eq("L1"), "label"].to_numpy(dtype=int)
        index_by_level = {
            level: group.set_index("participant_id")
            for level, group in subset.groupby("model_level", sort=False)
        }
        common_ids = index_by_level["L1"].index
        for level in ("L2", "L3", "L4"):
            common_ids = common_ids.intersection(index_by_level[level].index)
        common_ids = np.asarray(sorted(common_ids))
        y = index_by_level["L1"].loc[common_ids, "label"].to_numpy(dtype=int)
        predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for level in ("L1", "L2", "L3", "L4"):
            frame = index_by_level[level].loc[common_ids]
            predictions[level] = (
                frame["base_probability"].to_numpy(dtype=float),
                frame["augmented_probability"].to_numpy(dtype=float),
            )
        for metric_key, metric_name in (
            ("delta_auc", "auc"),
            ("delta_pr_auc", "pr_auc"),
            ("delta_brier", "brier"),
            ("delta_log_loss", "log_loss"),
        ):
            for attenuation_type, right in (
                ("protocol", "L2"),
                ("domain", "L3"),
                ("protocol_plus_domain", "L4"),
            ):
                estimate, low, high = _bootstrap_attenuation(
                    y,
                    predictions,
                    metric_key=metric_key,
                    left="L1",
                    right=right,
                    n_bootstrap=n_bootstrap,
                    seed=base_seed + len(rows) * 17,
                )
                rows.append(
                    {
                        "embedding": embedding,
                        "fusion": fusion,
                        "metric": metric_name,
                        "attenuation_type": attenuation_type,
                        "estimate": estimate,
                        "ci_low": low,
                        "ci_high": high,
                        "n": int(len(y)),
                        "n_bootstrap": int(n_bootstrap),
                    }
                )
    return pd.DataFrame(rows)


def _model_metrics(main_oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in main_oof.groupby(
        ["embedding_model", "fusion_method", "model_level", "model_name"], sort=False
    ):
        embedding, fusion, level, model_name = keys
        metrics = _metric_values(
            group["label"].to_numpy(dtype=int),
            group["probability"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "embedding": embedding,
                "fusion": fusion,
                "level": level,
                "model_name": model_name,
                **metrics,
                "n": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def _repeat_stability(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in predictions.groupby(
        ["embedding_model", "fusion_method", "repeat", "model_level"], sort=False
    ):
        embedding, fusion, repeat, level = keys
        merged, _ = _paired_predictions(group)
        point = paired_metric_deltas(
            merged["label"].to_numpy(dtype=int),
            merged["base_probability"].to_numpy(dtype=float),
            merged["augmented_probability"].to_numpy(dtype=float),
        )
        base_metrics = _metric_values(
            merged["label"].to_numpy(dtype=int), merged["base_probability"].to_numpy(dtype=float)
        )
        augmented_metrics = _metric_values(
            merged["label"].to_numpy(dtype=int), merged["augmented_probability"].to_numpy(dtype=float)
        )
        rows.append(
            {
                "embedding": embedding,
                "fusion": fusion,
                "repeat": int(repeat),
                "level": level,
                "base_auc": base_metrics["roc_auc"],
                "augmented_auc": augmented_metrics["roc_auc"],
                "delta_auc": point["delta_auc"],
                "base_pr_auc": base_metrics["pr_auc"],
                "augmented_pr_auc": augmented_metrics["pr_auc"],
                "delta_pr_auc": point["delta_pr_auc"],
                "base_brier": base_metrics["brier"],
                "augmented_brier": augmented_metrics["brier"],
                "delta_brier": point["delta_brier"],
                "base_log_loss": base_metrics["log_loss"],
                "augmented_log_loss": augmented_metrics["log_loss"],
                "delta_log_loss": point["delta_log_loss"],
                "n": int(len(merged)),
            }
        )
    return pd.DataFrame(rows)


def _sha256_ids(ids: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in np.sort(np.asarray(ids)).tolist():
        digest.update(f"{value}\n".encode("utf-8"))
    return digest.hexdigest()


def run_embedding_incremental_analysis(
    output_root: str | Path,
    *,
    embedding_root: str | Path | None = None,
    model_specs: Mapping[str, tuple[int, str]] | None = None,
    expected_n: int = EXPECTED_N,
    expected_repeats: int = 10,
    main_repeat: int = 1,
    inner_folds: int = 3,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    stability_repeats: int | None = 10,
    n_bootstrap: int = 5000,
    n_permutations: int = 10000,
    base_seed: int = BASE_SEED,
) -> dict[str, object]:
    """Run both frozen embeddings and both fusion strategies.

    The function writes only derived results.  It intentionally raises before
    modeling when a cache is absent or fails its manifest hash/ID checks.
    """

    output_root = Path(output_root).resolve()
    analysis_root = output_root / "03_embedding_robustness"
    embedding_root = Path(embedding_root or analysis_root).resolve()
    inputs = load_frozen_incremental_inputs(output_root, expected_n=expected_n)
    membership_path = output_root / "00_splits" / "repeated_5fold_splits_10x5.csv"
    if not membership_path.exists():
        raise FileNotFoundError(f"frozen split membership not found: {membership_path}")
    membership = pd.read_csv(membership_path)
    membership["repeat"] = pd.to_numeric(membership["repeat"], errors="raise").astype(int)
    membership["fold"] = pd.to_numeric(membership["fold"], errors="raise").astype(int)
    membership["participant_id"] = pd.to_numeric(
        membership["participant_id"], errors="raise"
    )
    membership["label"] = pd.to_numeric(membership["label"], errors="raise").astype(int)
    label_lookup = dict(zip(inputs.participant_ids.tolist(), inputs.labels.tolist()))
    unknown_ids = sorted(set(membership["participant_id"]) - set(label_lookup))
    if unknown_ids:
        raise ValueError(f"split membership contains unknown participant IDs: {unknown_ids[:5]}")
    mismatched_labels = [
        participant_id
        for participant_id, label in membership[["participant_id", "label"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
        if int(label) != int(label_lookup[participant_id])
    ]
    if mismatched_labels:
        raise ValueError(
            "split membership labels do not match frozen labels: "
            f"{mismatched_labels[:5]}"
        )
    if sorted(membership["repeat"].unique()) != list(range(1, expected_repeats + 1)):
        raise ValueError("split membership repeats do not match expected_repeats")
    if main_repeat not in set(membership["repeat"]):
        raise ValueError(f"main repeat {main_repeat} is not present in split membership")

    model_specs = dict(
        model_specs
        or {
            "model_1_all_mpnet_base_v2": (768, "all-mpnet-base-v2"),
            "model_2_bge_large_en_v1_5": (1024, "bge-large-en-v1.5"),
        }
    )
    repeat_count = expected_repeats if stability_repeats is None else int(stability_repeats)
    if repeat_count < main_repeat or repeat_count > expected_repeats:
        raise ValueError("stability_repeats must include main_repeat and not exceed expected_repeats")

    prediction_tables: list[pd.DataFrame] = []
    tuning_rows: list[dict[str, object]] = []
    embedding_manifests: list[dict[str, object]] = []
    for model_slug, (dimension, short_name) in model_specs.items():
        pair = load_frozen_embedding_pair(
            embedding_root,
            model_slug,
            inputs.participant_ids,
            expected_dim=int(dimension),
        )
        embedding_manifests.append(
            {
                "slug": pair.slug,
                "name": pair.model_name,
                "short_name": short_name,
                "revision": pair.revision,
                "embedding_dim": pair.embedding_dim,
                "cache_files": list(pair.cache_files),
            }
        )
        for fusion_method in FUSION_METHODS:
            for repeat in range(1, repeat_count + 1):
                table, tuning = _run_repeat(
                    pair=pair,
                    inputs=inputs,
                    membership=membership,
                    repeat=repeat,
                    fusion_method=fusion_method,
                    inner_folds=inner_folds,
                    c_grid=c_grid,
                    base_seed=base_seed,
                )
                prediction_tables.append(table)
                tuning_rows.extend(
                    {"embedding_model": model_slug, "fusion_method": fusion_method, **row}
                    for row in tuning
                )
    predictions = pd.concat(prediction_tables, ignore_index=True)
    main_oof = _main_oof_table(predictions, repeat=main_repeat)
    paired, _ = _paired_predictions(main_oof)
    metrics = _model_metrics(main_oof)
    deltas = _delta_table(
        paired,
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
        base_seed=base_seed,
    )
    attenuation = _attenuation_table(
        paired,
        n_bootstrap=n_bootstrap,
        base_seed=base_seed + 900_000,
    )
    stability = _repeat_stability(predictions)
    result_root = analysis_root / "embedding_incremental"
    oof_output = main_oof.rename(columns={"probability": "model_probability"})
    oof_pairs = paired.copy()
    oof_pairs["repeat"] = int(main_repeat)
    oof_pairs = oof_pairs[
        [
            "participant_id",
            "label",
            "outer_fold",
            "embedding_model",
            "fusion_method",
            "model_level",
            "base_probability",
            "augmented_probability",
            "repeat",
        ]
    ].rename(columns={"embedding_model": "embedding", "fusion_method": "fusion"})
    atomic_write_csv(oof_pairs, result_root / "embedding_incremental_oof_predictions.csv")
    atomic_write_csv(metrics, result_root / "embedding_incremental_model_metrics.csv")
    atomic_write_csv(deltas, result_root / "embedding_incremental_deltas.csv")
    atomic_write_csv(attenuation, result_root / "embedding_incremental_attenuation.csv")
    atomic_write_csv(stability, result_root / "embedding_incremental_repeat_stability.csv")
    atomic_write_csv(pd.DataFrame(tuning_rows), result_root / "embedding_incremental_tuning.csv")

    manifest = {
        "status": "complete",
        "analysis_status": "post-submission locked supplementary conditional-increment analysis",
        "cohort": {
            "n": int(len(inputs.labels)),
            "positive": int(inputs.labels.sum()),
            "negative": int((1 - inputs.labels).sum()),
            "participant_ids_sha256": _sha256_ids(inputs.participant_ids),
        },
        "models": embedding_manifests,
        "fusion_methods": list(FUSION_METHODS),
        "model_specifications": MODEL_SPECS,
        "cv": {
            "main_repeat": int(main_repeat),
            "stability_repeats": int(repeat_count),
            "outer_folds": 5,
            "inner_folds": int(inner_folds),
            "base_seed": int(base_seed),
            "C_grid": [float(value) for value in c_grid],
        },
        "inference": {
            "bootstrap_resamples": int(n_bootstrap),
            "permutation_resamples": int(n_permutations),
            "delta_direction": {
                "auc": "augmented - base",
                "pr_auc": "augmented - base",
                "brier": "base - augmented",
                "log_loss": "base - augmented",
            },
            "primary_fdr_family": "late_fusion L4 across MPNet and BGE",
            "secondary_fdr_family": "all late_fusion L1-L4 delta AUC",
        },
        "inputs": {
            "structural_features": str(
                output_root / "12_submission_audit" / "structural_baseline_features.csv"
            ),
            "domain_count": str(output_root / "04_c5_controls" / "domain_count" / "input.csv"),
            "split_membership": str(membership_path),
            "split_membership_sha256": sha256_file(membership_path),
            "protocol_features": list(PROTOCOL_FEATURES),
            "domain_features": list(DOMAIN_FEATURES),
        },
        "output_files": sorted(path.name for path in result_root.glob("*.csv")),
    }
    atomic_write_json(manifest, result_root / "run_manifest.json")
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run locked conditional-increment analysis on frozen MPNet/BGE embeddings"
    )
    parser.add_argument("--output-root", default="analysis_v2")
    parser.add_argument("--embedding-root", default=None)
    parser.add_argument("--expected-n", type=int, default=EXPECTED_N)
    parser.add_argument("--stability-repeats", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_embedding_incremental_analysis(
        args.output_root,
        embedding_root=args.embedding_root,
        expected_n=args.expected_n,
        stability_repeats=args.stability_repeats,
        n_bootstrap=args.bootstrap,
        n_permutations=args.permutations,
        base_seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
