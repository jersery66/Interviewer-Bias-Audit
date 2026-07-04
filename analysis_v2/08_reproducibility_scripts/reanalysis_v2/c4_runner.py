from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .control_analysis import (
    build_c4_control_inputs,
    build_c4_family_tests,
    run_fold_safe_shuffled_text_condition,
    run_raw_feature_condition,
    run_raw_text_condition,
)
from .io import atomic_write_csv, atomic_write_json, sha256_file
from .prepare import CONDITION_OUTPUT_NAMES
from .runner import _write_output_inventory


def _save_condition(root: Path, condition: str, result: dict[str, object]) -> None:
    destination = root / condition
    atomic_write_csv(result["fold_predictions"], destination / "fold_oof.csv")
    atomic_write_csv(
        result["participant_predictions"], destination / "participant_oof.csv"
    )


def run_c4_controls(
    output_root: str | Path,
    *,
    turns_path: str | Path,
    length_master_seed: int = 20260715,
    expected_n: int = 142,
    expected_repeats: int = 10,
    n_permutations: int = 10_000,
    base_seed: int = 20260624,
) -> dict:
    output_root = Path(output_root).resolve()
    root = output_root / "05_c4_controls"
    turns_path = Path(turns_path).resolve()
    started = datetime.now(timezone.utc)
    cohort = pd.read_csv(
        output_root / "01_inputs" / CONDITION_OUTPUT_NAMES["interviewer_speech"],
        keep_default_na=False,
    ).sort_values("participant_id", kind="stable")
    if len(cohort) != expected_n or cohort["participant_id"].nunique() != expected_n:
        raise ValueError("C4 cohort does not match the frozen participant count")
    participant_ids = cohort["participant_id"].to_numpy(dtype=int)
    labels = cohort["paper_label_phq8_ge10"].to_numpy(dtype=int)
    turns = pd.read_csv(turns_path, keep_default_na=False)
    membership_path = output_root / "00_splits" / "repeated_5fold_splits_10x5.csv"
    membership = pd.read_csv(membership_path)

    inputs = build_c4_control_inputs(
        turns,
        participant_ids,
        length_master_seed=length_master_seed,
    )
    participant_texts = inputs["participant_texts"].set_index("participant_id").loc[
        participant_ids
    ].reset_index()
    presence = inputs["template_presence"].set_index("participant_id").loc[
        participant_ids
    ].reset_index()
    atomic_write_csv(participant_texts, root / "input_audit" / "participant_texts.csv")
    atomic_write_csv(presence, root / "template_presence" / "input.csv")
    atomic_write_csv(
        inputs["template_id_map"], root / "template_presence" / "template_id_map.csv"
    )

    summaries = []
    participant_results = {}
    for condition, column in (
        ("template_only", "template_only"),
        ("nontemplate_lengthmatched", "nontemplate_lengthmatched"),
        ("nontemplate_positionmatched", "nontemplate_positionmatched"),
    ):
        result = run_raw_text_condition(
            participant_ids,
            labels,
            participant_texts[column].astype(str).to_numpy(),
            membership,
            condition=condition,
            expected_repeats=expected_repeats,
            base_seed=base_seed,
        )
        _save_condition(root, condition, result)
        summaries.append(result["summary"])
        participant_results[condition] = result["participant_predictions"]

    template_dict = dict(
        zip(participant_texts["participant_id"], participant_texts["template_only"])
    )
    shuffled = run_fold_safe_shuffled_text_condition(
        participant_ids,
        labels,
        template_dict,
        membership,
        condition="template_shuffled",
        expected_repeats=expected_repeats,
        base_seed=base_seed,
    )
    _save_condition(root, "template_shuffled", shuffled)
    atomic_write_csv(
        shuffled["shuffle_mapping"], root / "template_shuffled" / "shuffle_mapping.csv"
    )
    summaries.append(shuffled["summary"])
    participant_results["template_shuffled"] = shuffled["participant_predictions"]

    feature_columns = [column for column in presence.columns if column != "participant_id"]
    presence_result = run_raw_feature_condition(
        participant_ids,
        labels,
        presence[feature_columns].to_numpy(dtype=float),
        membership,
        condition="template_presence",
        expected_repeats=expected_repeats,
        base_seed=base_seed,
    )
    _save_condition(root, "template_presence", presence_result)
    summaries.append(presence_result["summary"])
    participant_results["template_presence"] = presence_result["participant_predictions"]

    family = build_c4_family_tests(
        participant_results["template_only"],
        {
            "nontemplate_lengthmatched": participant_results[
                "nontemplate_lengthmatched"
            ],
            "nontemplate_positionmatched": participant_results[
                "nontemplate_positionmatched"
            ],
            "template_shuffled": participant_results["template_shuffled"],
            "template_presence": participant_results["template_presence"],
        },
        n_permutations=n_permutations,
        base_seed=base_seed,
    )
    atomic_write_csv(pd.DataFrame(summaries), root / "metrics" / "c4_control_metrics.csv")
    atomic_write_csv(family, root / "paired_tests" / "c4_control_family_fdr.csv")

    inventory, inventory_hash = _write_output_inventory(root)
    completed = datetime.now(timezone.utc)
    manifest = {
        "status": "complete",
        "started_at_utc": started.isoformat(),
        "completed_at_utc": completed.isoformat(),
        "elapsed_seconds": (completed - started).total_seconds(),
        "cohort": {"n": expected_n, "positive": int(labels.sum()), "negative": int((1-labels).sum())},
        "template_definition": "canonical interviewer turns with is_explicit_clinical_prompt == 1",
        "nontemplate_definition": "canonical interviewer turns with is_explicit_clinical_prompt == 0",
        "template_id_definition": "sorted unique normalized explicit-clinical-prompt text",
        "length_master_seed": length_master_seed,
        "shuffled_seed_formula": "base_seed + 2000 + 100*repeat + fold; donors permuted separately within outer train and test",
        "turns_source": {"path": str(turns_path), "sha256": sha256_file(turns_path)},
        "split_sha256": sha256_file(membership_path),
        "n_permutations": n_permutations,
        "family_test_count": int(len(family)),
        "output_file_count": int(len(inventory)),
        "output_inventory_sha256": inventory_hash,
        "interpretation_guardrail": "Template, position, topic coverage, and text quantity jointly constitute protocol signal; results do not isolate natural-language template semantics.",
    }
    atomic_write_json(manifest, root / "run_manifest.json")
    return manifest
