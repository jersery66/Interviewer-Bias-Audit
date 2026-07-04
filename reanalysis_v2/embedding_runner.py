from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from .analysis import (
    PRIMARY_CONDITIONS,
    build_primary_paired_tests,
    run_dense_condition,
    summarize_threshold_stability,
)
from .embeddings import embed_documents
from .io import atomic_write_csv, atomic_write_json, sha256_file
from .inference import apply_bh_by_family, paired_permutation_test
from .runner import _load_frozen_conditions, _write_output_inventory


@dataclass(frozen=True)
class EmbeddingModelConfig:
    name: str
    revision: str
    slug: str
    chunk_words: int = 200
    overlap: int = 50
    batch_size: int = 32


def build_representation_family_tests(
    tfidf_tables: dict[str, pd.DataFrame],
    embedding_tables: dict[str, dict[str, pd.DataFrame]],
    *,
    precomputed_within: dict[str, pd.DataFrame] | None = None,
    n_permutations: int = 10_000,
    base_seed: int = 20260624,
) -> pd.DataFrame:
    if set(tfidf_tables) != set(PRIMARY_CONDITIONS):
        raise ValueError("TF-IDF tables must contain the five primary conditions")
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
    rows: list[dict[str, object]] = []
    test_index = 0
    for model_slug, model_tables in embedding_tables.items():
        if set(model_tables) != set(PRIMARY_CONDITIONS):
            raise ValueError(f"{model_slug} tables do not contain the five conditions")
        if precomputed_within and model_slug in precomputed_within:
            within = precomputed_within[model_slug].copy()
        else:
            within = build_primary_paired_tests(
                model_tables,
                n_permutations=n_permutations,
                base_seed=base_seed + 10_000 * (len(rows) + 1),
            )
        for record in within.to_dict("records"):
            rows.append(
                {
                    "family": "representation_robustness",
                    "comparison_type": "within_embedding_input_source",
                    "representation_a": model_slug,
                    "representation_b": model_slug,
                    "condition_a": record["condition_a"],
                    "condition_b": record["condition_b"],
                    "metric": record["metric"],
                    "n": int(record["n"]),
                    "observed_difference": float(record["observed_difference"]),
                    "p_value": float(record["p_value"]),
                    "n_permutations": int(record["n_permutations"]),
                    "seed": int(record["seed"]),
                }
            )

        for condition in PRIMARY_CONDITIONS:
            tfidf = tfidf_tables[condition].sort_values("participant_id", kind="stable")
            embedding = model_tables[condition].sort_values(
                "participant_id", kind="stable"
            )
            if not np.array_equal(
                tfidf["participant_id"].to_numpy(),
                embedding["participant_id"].to_numpy(),
            ) or not np.array_equal(
                tfidf["label"].to_numpy(dtype=int),
                embedding["label"].to_numpy(dtype=int),
            ):
                raise ValueError(
                    f"TF-IDF and {model_slug}/{condition} participants are not aligned"
                )
            y = tfidf["label"].to_numpy(dtype=int)
            for metric_name, (column, metric) in metrics.items():
                test_index += 1
                result = paired_permutation_test(
                    y,
                    embedding[column].to_numpy(),
                    tfidf[column].to_numpy(),
                    metric,
                    n_permutations=n_permutations,
                    seed=base_seed + 500_000 + test_index,
                )
                rows.append(
                    {
                        "family": "representation_robustness",
                        "comparison_type": "same_condition_cross_representation",
                        "representation_a": model_slug,
                        "representation_b": "tfidf",
                        "condition_a": condition,
                        "condition_b": condition,
                        "metric": metric_name,
                        "n": int(len(y)),
                        **result,
                    }
                )
    result = pd.DataFrame(rows)
    return apply_bh_by_family(result)


def _embedding_cache_key(
    config: EmbeddingModelConfig,
    condition: str,
    participant_ids: np.ndarray,
    texts: Sequence[str],
) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(asdict(config), sort_keys=True).encode("utf-8"))
    digest.update(condition.encode("utf-8"))
    for participant_id, text in zip(participant_ids, texts):
        digest.update(f"{participant_id}\0".encode("utf-8"))
        digest.update(str(text).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _atomic_write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_or_create_embeddings(
    model_root: Path,
    config: EmbeddingModelConfig,
    condition: str,
    participant_ids: np.ndarray,
    texts: Sequence[str],
    encoder,
    embedding_dim: int,
) -> tuple[np.ndarray, pd.DataFrame, Path, str]:
    cache_key = _embedding_cache_key(
        config, condition, participant_ids, texts
    )
    cache_path = (
        model_root / "embedding_cache" / f"{condition}_{cache_key[:16]}.npz"
    )
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            if str(cached["cache_key"].item()) != cache_key:
                raise ValueError(f"embedding cache key mismatch: {cache_path}")
            cached_ids = cached["participant_ids"]
            if not np.array_equal(cached_ids, participant_ids):
                raise ValueError(f"embedding cache participant mismatch: {cache_path}")
            matrix = cached["embeddings"].astype(np.float32)
            audit = pd.DataFrame(
                {
                    "document_index": np.arange(len(participant_ids)),
                    "word_count": cached["word_count"].astype(int),
                    "chunk_count": cached["chunk_count"].astype(int),
                    "is_empty": cached["is_empty"].astype(bool),
                }
            )
    else:
        matrix, audit = embed_documents(
            list(texts),
            encoder,
            dim=embedding_dim,
            chunk_words=config.chunk_words,
            overlap=config.overlap,
            batch_size=config.batch_size,
        )
        _atomic_write_npz(
            cache_path,
            cache_key=np.asarray(cache_key),
            participant_ids=np.asarray(participant_ids),
            embeddings=matrix,
            word_count=audit["word_count"].to_numpy(dtype=int),
            chunk_count=audit["chunk_count"].to_numpy(dtype=int),
            is_empty=audit["is_empty"].to_numpy(dtype=bool),
            model_name=np.asarray(config.name),
            model_revision=np.asarray(config.revision),
        )
    if matrix.shape != (len(participant_ids), embedding_dim):
        raise ValueError(
            f"embedding matrix shape {matrix.shape} does not match "
            f"{(len(participant_ids), embedding_dim)}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("embedding matrix contains non-finite values")
    audit = audit.copy()
    audit.insert(0, "participant_id", participant_ids)
    audit.insert(0, "condition", condition)
    audit["cache_key"] = cache_key
    audit["cache_path"] = cache_path.name
    return matrix, audit, cache_path, cache_key


def run_embedding_model_analysis(
    output_root: str | Path,
    config: EmbeddingModelConfig,
    *,
    encoder=None,
    embedding_dim: int | None = None,
    device: str = "cpu",
    expected_n: int = 142,
    expected_repeats: int = 10,
    inner_splits_count: int = 3,
    c_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    n_permutations: int = 10_000,
    base_seed: int = 20260624,
) -> dict:
    output_root = Path(output_root).resolve()
    model_root = output_root / "03_embedding_robustness" / config.slug
    started = datetime.now(timezone.utc)
    tables, participant_ids, labels = _load_frozen_conditions(
        output_root, expected_n=expected_n
    )
    membership_path = output_root / "00_splits" / "repeated_5fold_splits_10x5.csv"
    membership = pd.read_csv(membership_path)

    loaded_here = encoder is None
    if loaded_here:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(
            config.name,
            revision=config.revision,
            device=device,
        )
    if embedding_dim is None:
        if hasattr(encoder, "get_embedding_dimension"):
            embedding_dim = int(encoder.get_embedding_dimension())
        else:
            embedding_dim = int(encoder.get_sentence_embedding_dimension())
    if embedding_dim <= 0:
        raise ValueError("embedding dimension must be positive")

    participant_results: dict[str, pd.DataFrame] = {}
    summaries = []
    distributions = []
    threshold_tables = []
    tuning_tables = []
    embedding_audits = []
    cache_records = []
    for condition in PRIMARY_CONDITIONS:
        print(f"[{config.slug}] embedding {condition}", flush=True)
        texts = tables[condition]["text"].fillna("").astype(str).to_numpy()
        matrix, audit, cache_path, cache_key = _load_or_create_embeddings(
            model_root,
            config,
            condition,
            participant_ids,
            texts,
            encoder,
            embedding_dim,
        )
        embedding_audits.append(audit)
        cache_records.append(
            {
                "condition": condition,
                "path": cache_path.relative_to(model_root).as_posix(),
                "sha256": sha256_file(cache_path),
                "cache_key": cache_key,
            }
        )
        print(f"[{config.slug}] modeling {condition}", flush=True)
        result = run_dense_condition(
            participant_ids,
            labels,
            matrix,
            membership,
            condition=condition,
            expected_repeats=expected_repeats,
            inner_splits_count=inner_splits_count,
            c_grid=c_grid,
            base_seed=base_seed,
        )
        atomic_write_csv(
            result["fold_predictions"],
            model_root / "oof_probs" / f"{condition}_fold_oof.csv",
        )
        atomic_write_csv(
            result["participant_predictions"],
            model_root / "oof_probs" / f"{condition}_participant_oof.csv",
        )
        participant_results[condition] = result["participant_predictions"]
        summaries.append(result["summary"])
        distributions.append(result["distributions"])
        threshold_tables.append(result["thresholds"])
        tuning_tables.append(result["c_tuning"])

    summary_table = pd.DataFrame(summaries)
    distribution_table = pd.concat(distributions, ignore_index=True)
    threshold_table = pd.concat(threshold_tables, ignore_index=True)
    tuning_table = pd.concat(tuning_tables, ignore_index=True)
    embedding_audit = pd.concat(embedding_audits, ignore_index=True)
    paired = build_primary_paired_tests(
        participant_results,
        n_permutations=n_permutations,
        base_seed=base_seed,
    )
    paired["family"] = f"{config.slug}_input_source_descriptive"
    paired["fdr_scope_note"] = (
        "model-local descriptive; final representation-family FDR is computed jointly"
    )

    atomic_write_csv(summary_table, model_root / "metrics" / "primary_metrics.csv")
    atomic_write_csv(
        distribution_table,
        model_root / "calibration" / "probability_distributions.csv",
    )
    atomic_write_csv(
        threshold_table,
        model_root / "nested_thresholds" / "selected_thresholds_50folds.csv",
    )
    atomic_write_csv(
        summarize_threshold_stability(threshold_table),
        model_root / "nested_thresholds" / "threshold_stability_summary.csv",
    )
    atomic_write_csv(tuning_table, model_root / "tuning" / "c_selection.csv")
    atomic_write_csv(
        embedding_audit, model_root / "embedding_audit" / "document_chunk_audit.csv"
    )
    atomic_write_csv(
        paired, model_root / "paired_tests" / "input_source_paired_tests_descriptive.csv"
    )

    inventory, inventory_hash = _write_output_inventory(model_root)
    completed = datetime.now(timezone.utc)
    manifest = {
        "status": "complete",
        "started_at_utc": started.isoformat(),
        "completed_at_utc": completed.isoformat(),
        "elapsed_seconds": (completed - started).total_seconds(),
        "model": {
            **asdict(config),
            "embedding_dim": embedding_dim,
            "device": device,
            "revision_explicitly_passed": bool(loaded_here),
        },
        "cohort": {
            "n": int(len(labels)),
            "positive": int(labels.sum()),
            "negative": int((1 - labels).sum()),
        },
        "cv": {
            "outer_repeats": expected_repeats,
            "outer_folds": int(membership["fold"].nunique()),
            "inner_folds": inner_splits_count,
            "base_seed": base_seed,
            "C_grid": [float(value) for value in c_grid],
        },
        "pooling": "unweighted mean of normalized chunk embeddings, then document L2 normalization",
        "empty_document_policy": "all-zero vector",
        "cache_files": cache_records,
        "split_file_sha256": sha256_file(membership_path),
        "output_file_count": int(len(inventory)),
        "output_inventory_sha256": inventory_hash,
    }
    atomic_write_json(manifest, model_root / "run_manifest.json")
    return manifest
