from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from .analysis import (
    PRIMARY_CONDITIONS,
    build_primary_paired_tests,
    run_text_condition,
    summarize_threshold_stability,
)
from .io import atomic_write_csv, atomic_write_json, sha256_file
from .prepare import CONDITION_OUTPUT_NAMES


def _load_frozen_conditions(
    output_root: Path, *, expected_n: int
) -> tuple[dict[str, pd.DataFrame], np.ndarray, np.ndarray]:
    tables: dict[str, pd.DataFrame] = {}
    reference_ids = None
    reference_labels = None
    for condition in PRIMARY_CONDITIONS:
        path = output_root / "01_inputs" / CONDITION_OUTPUT_NAMES[condition]
        table = pd.read_csv(path, keep_default_na=False).sort_values(
            "participant_id", kind="stable"
        )
        if len(table) != expected_n or table["participant_id"].nunique() != expected_n:
            raise ValueError(f"{condition} does not contain {expected_n} unique participants")
        if not table["condition_id"].eq(condition).all():
            raise ValueError(f"condition identity mismatch in {path}")
        ids = table["participant_id"].to_numpy(dtype=int)
        labels = table["paper_label_phq8_ge10"].to_numpy(dtype=int)
        if reference_ids is None:
            reference_ids, reference_labels = ids, labels
        elif not np.array_equal(ids, reference_ids) or not np.array_equal(
            labels, reference_labels
        ):
            raise ValueError("frozen input participant IDs or labels are not aligned")
        tables[condition] = table.reset_index(drop=True)
    return tables, reference_ids, reference_labels


def _write_output_inventory(result_root: Path) -> tuple[pd.DataFrame, str]:
    inventory_path = result_root / "output_manifest_sha256.csv"
    rows = []
    for path in sorted(result_root.rglob("*")):
        if not path.is_file() or path in {inventory_path, result_root / "run_manifest.json"}:
            continue
        rows.append(
            {
                "relative_path": path.relative_to(result_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    inventory = pd.DataFrame(rows)
    atomic_write_csv(inventory, inventory_path)
    return inventory, sha256_file(inventory_path)


def run_tfidf_analysis(
    output_root: str | Path,
    *,
    expected_n: int = 142,
    expected_repeats: int = 10,
    inner_splits_count: int = 3,
    n_permutations: int = 10_000,
    base_seed: int = 20260624,
) -> dict:
    output_root = Path(output_root).resolve()
    result_root = output_root / "02_tfidf_main"
    started = datetime.now(timezone.utc)
    tables, participant_ids, labels = _load_frozen_conditions(
        output_root, expected_n=expected_n
    )
    membership_path = output_root / "00_splits" / "repeated_5fold_splits_10x5.csv"
    membership = pd.read_csv(membership_path)

    participant_results: dict[str, pd.DataFrame] = {}
    summaries = []
    distributions = []
    selected_thresholds = []
    for condition in PRIMARY_CONDITIONS:
        print(f"[TFIDF] starting {condition}", flush=True)
        result = run_text_condition(
            participant_ids,
            labels,
            tables[condition]["text"].fillna("").astype(str).to_numpy(),
            membership,
            condition=condition,
            expected_repeats=expected_repeats,
            inner_splits_count=inner_splits_count,
            base_seed=base_seed,
        )
        fold_predictions = result["fold_predictions"]
        participant_predictions = result["participant_predictions"]
        thresholds = result["thresholds"]
        atomic_write_csv(
            fold_predictions,
            result_root / "oof_probs" / f"{condition}_fold_oof.csv",
        )
        atomic_write_csv(
            participant_predictions,
            result_root / "oof_probs" / f"{condition}_participant_oof.csv",
        )
        participant_results[condition] = participant_predictions
        summaries.append(result["summary"])
        distributions.append(result["distributions"])
        selected_thresholds.append(thresholds)
        print(f"[TFIDF] completed {condition}", flush=True)

    summary_table = pd.DataFrame(summaries)
    distribution_table = pd.concat(distributions, ignore_index=True)
    threshold_table = pd.concat(selected_thresholds, ignore_index=True)
    threshold_stability = summarize_threshold_stability(threshold_table)
    paired_tests = build_primary_paired_tests(
        participant_results,
        n_permutations=n_permutations,
        base_seed=base_seed,
    )

    atomic_write_csv(summary_table, result_root / "metrics" / "primary_metrics.csv")
    calibration_columns = [
        column
        for column in summary_table.columns
        if column == "condition"
        or any(
            token in column
            for token in ("brier", "ece", "calibration", "platt_", "isotonic_")
        )
    ]
    atomic_write_csv(
        summary_table[calibration_columns],
        result_root / "calibration" / "calibration_metrics.csv",
    )
    atomic_write_csv(
        distribution_table,
        result_root / "calibration" / "probability_distributions.csv",
    )
    atomic_write_csv(
        threshold_table,
        result_root / "nested_thresholds" / "selected_thresholds_50folds.csv",
    )
    atomic_write_csv(
        threshold_stability,
        result_root / "nested_thresholds" / "threshold_stability_summary.csv",
    )
    atomic_write_csv(
        paired_tests,
        result_root
        / "paired_tests"
        / "primary_input_source_paired_tests_fdr.csv",
    )

    inventory, inventory_hash = _write_output_inventory(result_root)
    completed = datetime.now(timezone.utc)
    manifest = {
        "status": "complete",
        "started_at_utc": started.isoformat(),
        "completed_at_utc": completed.isoformat(),
        "elapsed_seconds": (completed - started).total_seconds(),
        "analysis": "TF-IDF C1-C5 with nested thresholds and train-only calibration",
        "cohort": {
            "n": int(len(labels)),
            "positive": int(labels.sum()),
            "negative": int((1 - labels).sum()),
        },
        "input_files": {
            condition: {
                "path": str(output_root / "01_inputs" / CONDITION_OUTPUT_NAMES[condition]),
                "sha256": sha256_file(
                    output_root / "01_inputs" / CONDITION_OUTPUT_NAMES[condition]
                ),
            }
            for condition in PRIMARY_CONDITIONS
        },
        "split_file": {
            "path": str(membership_path),
            "sha256": sha256_file(membership_path),
        },
        "cv": {
            "outer_repeats": expected_repeats,
            "outer_folds": int(membership["fold"].nunique()),
            "inner_folds": inner_splits_count,
            "base_seed": base_seed,
            "inner_seed_formula": "base_seed + 1000 + 100*repeat + fold",
        },
        "model": {
            "vectorizer": {
                "lowercase": True,
                "strip_accents": "unicode",
                "ngram_range": [1, 2],
                "min_df": 2,
                "max_features": 50_000,
                "sublinear_tf": True,
            },
            "classifier": {
                "type": "LogisticRegression",
                "C": 1.0,
                "class_weight": "balanced",
                "solver": "liblinear",
                "max_iter": 2000,
            },
        },
        "calibration": {
            "primary": "Platt/sigmoid, outer-train only",
            "supplementary": "isotonic, outer-train only",
        },
        "inference": {
            "level": "participant mean across repeated OOF predictions",
            "permutations": n_permutations,
            "fdr": "Benjamini-Hochberg within primary input-source family",
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "output_file_count": int(len(inventory)),
        "output_inventory_sha256": inventory_hash,
    }
    atomic_write_json(manifest, result_root / "run_manifest.json")
    return manifest
