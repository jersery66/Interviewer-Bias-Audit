from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .control_analysis import (
    _raw_summary,
    build_c5_control_inputs,
    build_c5_family_tests,
    run_raw_feature_condition,
    run_raw_text_condition,
)
from .controls import DOMAIN_ORDER
from .io import atomic_write_csv, atomic_write_json, sha256_file
from .prepare import CONDITION_OUTPUT_NAMES
from .runner import _write_output_inventory


DEFAULT_C5_KEYWORDS = [
    "depressed",
    "depression",
    "sad",
    "down",
    "hopeless",
    "worthless",
    "guilty",
    "guilt",
    "sleep",
    "insomnia",
    "tired",
    "fatigue",
    "energy",
    "appetite",
    "eating",
    "weight",
    "concentrate",
    "concentration",
    "focus",
    "suicide",
    "suicidal",
    "kill",
    "myself",
    "self-harm",
    "interest",
    "pleasure",
]


def _save_raw_result(root: Path, condition: str, result: dict[str, object]) -> None:
    atomic_write_csv(
        result["fold_predictions"], root / "oof_probs" / f"{condition}_fold_oof.csv"
    )
    atomic_write_csv(
        result["participant_predictions"],
        root / "oof_probs" / f"{condition}_participant_oof.csv",
    )


def _random_metric_distribution(random_summaries: list[dict]) -> pd.DataFrame:
    rows = []
    for metric in ("roc_auc", "pr_auc", "macro_f1_at_0_5"):
        values = np.asarray([row[metric] for row in random_summaries], dtype=float)
        rows.append(
            {
                "metric": metric,
                "n_seeds": int(len(values)),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "percentile_2_5": float(np.quantile(values, 0.025)),
                "median": float(np.quantile(values, 0.5)),
                "percentile_97_5": float(np.quantile(values, 0.975)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def run_c5_controls(
    output_root: str | Path,
    *,
    spans_path: str | Path,
    random_master_seeds: Sequence[int] = tuple(range(20260704, 20260714)),
    nonsymptom_master_seed: int = 20260714,
    keywords: Sequence[str] = tuple(DEFAULT_C5_KEYWORDS),
    expected_n: int = 142,
    expected_repeats: int = 10,
    n_permutations: int = 10_000,
    base_seed: int = 20260624,
) -> dict:
    output_root = Path(output_root).resolve()
    root = output_root / "04_c5_controls"
    spans_path = Path(spans_path).resolve()
    started = datetime.now(timezone.utc)
    participant = pd.read_csv(
        output_root / "01_inputs" / CONDITION_OUTPUT_NAMES["participant_speech"],
        keep_default_na=False,
    ).sort_values("participant_id", kind="stable")
    reviewed_input = pd.read_csv(
        output_root
        / "01_inputs"
        / CONDITION_OUTPUT_NAMES["participant_symptom_evidence"],
        keep_default_na=False,
    ).sort_values("participant_id", kind="stable")
    spans = pd.read_csv(spans_path, keep_default_na=False)
    if len(participant) != expected_n or len(reviewed_input) != expected_n:
        raise ValueError("C5 inputs do not match the frozen cohort size")
    participant_ids = participant["participant_id"].to_numpy(dtype=int)
    labels = participant["paper_label_phq8_ge10"].to_numpy(dtype=int)
    membership_path = output_root / "00_splits" / "repeated_5fold_splits_10x5.csv"
    membership = pd.read_csv(membership_path)

    controls = build_c5_control_inputs(
        participant,
        reviewed_input,
        spans,
        random_master_seeds=random_master_seeds,
        nonsymptom_master_seed=nonsymptom_master_seed,
        keywords=keywords,
    )
    atomic_write_csv(
        controls["random_audit"], root / "random_same_word" / "sampling_audit.csv"
    )
    atomic_write_csv(
        controls["nonsymptom"],
        root / "nonsymptom_same_word" / "input.csv",
    )
    atomic_write_csv(
        controls["quote_exclusion_audit"],
        root / "nonsymptom_same_word" / "quote_exclusion_audit.csv",
    )
    atomic_write_csv(
        controls["domain_presence"], root / "domain_presence" / "input.csv"
    )
    atomic_write_csv(controls["domain_count"], root / "domain_count" / "input.csv")
    atomic_write_csv(
        controls["keyword_masked"], root / "keyword_masked" / "input.csv"
    )
    atomic_write_csv(
        controls["keyword_mask_audit"],
        root / "keyword_masked" / "mask_audit.csv",
    )

    summaries = []
    participant_results = {}
    reviewed = run_raw_text_condition(
        participant_ids,
        labels,
        reviewed_input["text"].astype(str).to_numpy(),
        membership,
        condition="c5_reviewed_evidence",
        expected_repeats=expected_repeats,
        base_seed=base_seed,
    )
    _save_raw_result(root, "c5_reviewed_evidence", reviewed)
    summaries.append(reviewed["summary"])
    participant_results["c5_reviewed_evidence"] = reviewed["participant_predictions"]

    random_participants = {}
    random_summaries = []
    for seed_index, (seed_name, table) in enumerate(
        controls["random_tables"].items(), start=1
    ):
        condition = f"c5_random_same_word_seed_{seed_index:02d}"
        seed_root = root / "random_same_word" / condition
        atomic_write_csv(table, seed_root / "input.csv")
        result = run_raw_text_condition(
            participant_ids,
            labels,
            table["text"].astype(str).to_numpy(),
            membership,
            condition=condition,
            expected_repeats=expected_repeats,
            base_seed=base_seed,
        )
        atomic_write_csv(result["fold_predictions"], seed_root / "fold_oof.csv")
        atomic_write_csv(
            result["participant_predictions"], seed_root / "participant_oof.csv"
        )
        summaries.append(result["summary"])
        random_summaries.append(result["summary"])
        random_participants[seed_name] = result["participant_predictions"]

    stacked = []
    for seed_name, table in random_participants.items():
        part = table.copy()
        part["seed_name"] = seed_name
        stacked.append(part)
    ensemble = (
        pd.concat(stacked, ignore_index=True)
        .groupby("participant_id", as_index=False)
        .agg(label=("label", "first"), raw_probability=("raw_probability", "mean"))
        .sort_values("participant_id", kind="stable")
    )
    ensemble["prediction_at_0_5"] = ensemble["raw_probability"].ge(0.5).astype(int)
    ensemble_summary = _raw_summary(ensemble, "random_ensemble")
    atomic_write_csv(
        ensemble, root / "random_same_word" / "random_ensemble_participant_oof.csv"
    )
    atomic_write_csv(
        _random_metric_distribution(random_summaries),
        root / "random_same_word" / "random_seed_metric_distribution.csv",
    )
    summaries.append(ensemble_summary)

    named_results = {}
    for condition, table in (
        ("nonsymptom_same_word", controls["nonsymptom"]),
        ("keyword_masked", controls["keyword_masked"]),
    ):
        result = run_raw_text_condition(
            participant_ids,
            labels,
            table["text"].astype(str).to_numpy(),
            membership,
            condition=condition,
            expected_repeats=expected_repeats,
            base_seed=base_seed,
        )
        _save_raw_result(root, condition, result)
        summaries.append(result["summary"])
        named_results[condition] = result["participant_predictions"]
    for condition, table in (
        ("domain_presence", controls["domain_presence"]),
        ("domain_count", controls["domain_count"]),
    ):
        aligned = table.set_index("participant_id").loc[participant_ids]
        result = run_raw_feature_condition(
            participant_ids,
            labels,
            aligned[DOMAIN_ORDER].to_numpy(dtype=float),
            membership,
            condition=condition,
            expected_repeats=expected_repeats,
            base_seed=base_seed,
        )
        _save_raw_result(root, condition, result)
        summaries.append(result["summary"])
        named_results[condition] = result["participant_predictions"]
    named_results = {"random_ensemble": ensemble, **named_results}

    family = build_c5_family_tests(
        reviewed["participant_predictions"],
        random_participants,
        named_results,
        n_permutations=n_permutations,
        base_seed=base_seed,
    )
    summary_table = pd.DataFrame(summaries)
    atomic_write_csv(summary_table, root / "metrics" / "c5_control_metrics.csv")
    atomic_write_csv(family, root / "paired_tests" / "c5_control_family_fdr.csv")

    tfidf_match = None
    tfidf_path = (
        output_root
        / "02_tfidf_main"
        / "oof_probs"
        / "participant_symptom_evidence_participant_oof.csv"
    )
    if tfidf_path.exists():
        historical = pd.read_csv(tfidf_path).sort_values("participant_id", kind="stable")
        current = reviewed["participant_predictions"].sort_values(
            "participant_id", kind="stable"
        )
        tfidf_match = bool(
            np.array_equal(
                historical["participant_id"].to_numpy(), current["participant_id"].to_numpy()
            )
            and np.allclose(
                historical["raw_probability"].to_numpy(),
                current["raw_probability"].to_numpy(),
                rtol=0,
                atol=1e-12,
            )
        )
        if not tfidf_match:
            raise ValueError("reviewed C5 raw OOF does not reproduce the frozen TF-IDF result")

    inventory, inventory_hash = _write_output_inventory(root)
    completed = datetime.now(timezone.utc)
    manifest = {
        "status": "complete",
        "started_at_utc": started.isoformat(),
        "completed_at_utc": completed.isoformat(),
        "elapsed_seconds": (completed - started).total_seconds(),
        "cohort": {"n": expected_n, "positive": int(labels.sum()), "negative": int((1-labels).sum())},
        "random_master_seeds": list(random_master_seeds),
        "nonsymptom_master_seed": nonsymptom_master_seed,
        "keywords": list(keywords),
        "domain_order": DOMAIN_ORDER,
        "spans_source": {"path": str(spans_path), "sha256": sha256_file(spans_path)},
        "split_sha256": sha256_file(membership_path),
        "n_permutations": n_permutations,
        "family_test_count": int(len(family)),
        "reviewed_raw_matches_tfidf_main": tfidf_match,
        "output_file_count": int(len(inventory)),
        "output_inventory_sha256": inventory_hash,
    }
    atomic_write_json(manifest, root / "run_manifest.json")
    return manifest
