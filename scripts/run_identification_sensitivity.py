"""Run the locked post-audit interviewer-signal identification sensitivities.

The script consumes the already frozen input tables and embedding caches.  It
creates a new immutable run directory and never modifies the formal
embedding-incremental outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reanalysis_v2.embedding_incremental import (  # noqa: E402
    BASE_SEED,
    DEFAULT_C_GRID,
    DOMAIN_FEATURES,
    FrozenEmbeddingPair,
    FrozenIncrementalInputs,
    load_frozen_embedding_pair,
    load_frozen_incremental_inputs,
    paired_metric_deltas,
)
from reanalysis_v2.identification_sensitivity import (  # noqa: E402
    D_EXTRA_FEATURES,
    D_PHQ_FEATURES,
    PAIRING_MATCH_COLUMNS,
    make_derangement,
    make_fake_domain_permutation,
    make_structure_matched_permutation,
    partition_domains,
    permute_interviewer_pair,
    prediction_metrics,
    replace_domains,
    run_level_pair_oof,
    run_single_level_oof,
)
from reanalysis_v2.io import sha256_file  # noqa: E402


EXPECTED_DIMENSIONS = {
    "model_1_all_mpnet_base_v2": 768,
    "model_2_bge_large_en_v1_5": 1024,
}
FORMAL_PAIRING_DRAWS = 50
FORMAL_FAKE_D_DRAWS = 50
FORMAL_INNER_FOLDS = 3
FORMAL_REPEAT = 1


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_membership(path: Path) -> pd.DataFrame:
    membership = pd.read_csv(path)
    required = {"repeat", "fold", "role", "participant_id", "label"}
    missing = required - set(membership.columns)
    if missing:
        raise ValueError(f"split membership is missing columns: {sorted(missing)}")
    return membership


def _align_structural(path: Path, participant_ids: np.ndarray) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "participant_id" not in frame.columns:
        raise ValueError("structural input has no participant_id column")
    if frame["participant_id"].duplicated().any():
        raise ValueError("structural input contains duplicate participant IDs")
    indexed = frame.set_index("participant_id")
    missing = sorted(set(participant_ids.tolist()) - set(indexed.index.tolist()))
    if missing:
        raise ValueError(f"structural input is missing participant IDs: {missing[:5]}")
    return indexed.loc[participant_ids].reset_index()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def _pairing_ledger(
    ids: np.ndarray,
    permutation: np.ndarray,
    *,
    condition: str,
    draw_id: int,
    structural: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    values = structural.loc[:, list(columns)].to_numpy(dtype=float)
    scale = values.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    z = (values - values.mean(axis=0)) / scale
    distances = np.square(z - z[permutation]).sum(axis=1)
    return pd.DataFrame(
        {
            "pairing_condition": condition,
            "draw_id": int(draw_id),
            "recipient_participant_id": ids,
            "donor_participant_id": ids[permutation],
            "donor_row_index": permutation,
            "self_match": permutation == np.arange(len(ids)),
            "standardized_structural_distance": distances,
        }
    )


def _prediction_with_metadata(
    prediction: pd.DataFrame,
    *,
    experiment: str,
    condition: str,
    draw_id: int | str,
    domain_variant: str | None = None,
) -> pd.DataFrame:
    result = prediction.copy()
    result.insert(0, "experiment", experiment)
    result.insert(1, "condition", condition)
    result.insert(2, "draw_id", draw_id)
    if domain_variant is not None:
        result.insert(3, "domain_variant", domain_variant)
    return result


def _metric_row(
    prediction: pd.DataFrame,
    *,
    experiment: str,
    condition: str,
    draw_id: int | str,
    model_name: str | None = None,
    domain_variant: str | None = None,
) -> dict[str, object]:
    values = prediction_metrics(prediction)
    row: dict[str, object] = {
        "experiment": experiment,
        "condition": condition,
        "draw_id": draw_id,
        "embedding_model": str(prediction["embedding_model"].iloc[0]),
        "fusion_method": str(prediction["fusion_method"].iloc[0]),
        "model_level": str(prediction["model_level"].iloc[0]),
        "model_name": model_name or str(prediction["model_name"].iloc[0]),
        "n": int(len(prediction)),
        **values,
    }
    if domain_variant is not None:
        row["domain_variant"] = domain_variant
    return row


def _delta_row(
    paired: pd.DataFrame,
    *,
    experiment: str,
    condition: str,
    draw_id: int | str,
    domain_variant: str | None = None,
) -> dict[str, object]:
    values = paired_metric_deltas(
        paired["label"].to_numpy(dtype=int),
        paired["base_probability"].to_numpy(dtype=float),
        paired["augmented_probability"].to_numpy(dtype=float),
    )
    row: dict[str, object] = {
        "experiment": experiment,
        "condition": condition,
        "draw_id": draw_id,
        "embedding_model": str(paired["embedding_model"].iloc[0]),
        "fusion_method": str(paired["fusion_method"].iloc[0]),
        "model_level": str(paired["model_level"].iloc[0]),
        **values,
    }
    if domain_variant is not None:
        row["domain_variant"] = domain_variant
    return row


def _pair_one_condition(
    pair: FrozenEmbeddingPair,
    inputs: FrozenIncrementalInputs,
    membership: pd.DataFrame,
    *,
    condition: str,
    draw_id: int,
    permutation: np.ndarray,
    fusion_method: str,
    base_seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    active_pair = pair if condition == "real" else permute_interviewer_pair(pair, permutation)
    prediction = run_single_level_oof(
        pair=active_pair,
        inputs=inputs,
        membership=membership,
        repeat=FORMAL_REPEAT,
        level="L1",
        augmented=True,
        fusion_method=fusion_method,
        inner_folds=FORMAL_INNER_FOLDS,
        c_grid=DEFAULT_C_GRID,
        base_seed=base_seed,
    )
    prediction = _prediction_with_metadata(
        prediction,
        experiment="pairing",
        condition=condition,
        draw_id=draw_id,
    )
    metric = _metric_row(
        prediction,
        experiment="pairing",
        condition=condition,
        draw_id=draw_id,
    )
    metric["embedding_model"] = pair.slug
    metric["fusion_method"] = fusion_method
    return prediction, metric


def _fake_or_partition_one_condition(
    pair: FrozenEmbeddingPair,
    inputs: FrozenIncrementalInputs,
    membership: pd.DataFrame,
    *,
    experiment: str,
    condition: str,
    draw_id: int | str,
    domain_variant: str,
    fusion_method: str,
    base_seed: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    prediction = run_level_pair_oof(
        pair=pair,
        inputs=inputs,
        membership=membership,
        repeat=FORMAL_REPEAT,
        level="L3",
        fusion_method=fusion_method,
        inner_folds=FORMAL_INNER_FOLDS,
        c_grid=DEFAULT_C_GRID,
        base_seed=base_seed,
    )
    prediction = _prediction_with_metadata(
        prediction,
        experiment=experiment,
        condition=condition,
        draw_id=draw_id,
        domain_variant=domain_variant,
    )
    metrics: list[dict[str, object]] = []
    for model_name, group in prediction.groupby("model_name", sort=True):
        metrics.append(
            _metric_row(
                group,
                experiment=experiment,
                condition=condition,
                draw_id=draw_id,
                model_name=str(model_name),
                domain_variant=domain_variant,
            )
        )
    base = prediction.loc[prediction["model_name"].eq("base")].rename(
        columns={"probability": "base_probability"}
    )
    augmented = prediction.loc[prediction["model_name"].eq("augmented")].rename(
        columns={"probability": "augmented_probability"}
    )
    key = ["participant_id", "label", "embedding_model", "fusion_method", "model_level"]
    paired = base[key + ["base_probability"]].merge(
        augmented[key + ["augmented_probability"]], on=key, validate="one_to_one"
    )
    metrics.append(
        _delta_row(
            paired,
            experiment=experiment,
            condition=condition,
            draw_id=draw_id,
            domain_variant=domain_variant,
        )
    )
    return prediction, metrics


def _summary_by_condition(metrics: pd.DataFrame, *, metric_columns: tuple[str, ...]) -> pd.DataFrame:
    group_columns = [
        "experiment",
        "condition",
        "embedding_model",
        "fusion_method",
        "model_level",
    ]
    if "model_name" in metrics.columns and metrics["model_name"].notna().any():
        group_columns.append("model_name")
    if "domain_variant" in metrics.columns:
        group_columns.append("domain_variant")
    rows: list[dict[str, object]] = []
    for keys, group in metrics.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["n_draws"] = int(group["draw_id"].nunique())
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
            if len(values) == 0:
                continue
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_q025"] = float(np.quantile(values, 0.025))
            row[f"{metric}_q975"] = float(np.quantile(values, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def _run_parallel(tasks, worker, workers: int):
    if int(workers) <= 1:
        return [worker(task) for task in tasks]
    results = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        pending = {executor.submit(worker, task): index for index, task in enumerate(tasks)}
        for future in as_completed(pending):
            results[pending[future]] = future.result()
    return results


def run_analysis(
    *,
    output_root: Path,
    embedding_root: Path,
    output_dir: Path,
    workers: int = 1,
    pairing_draws: int = FORMAL_PAIRING_DRAWS,
    fake_d_draws: int = FORMAL_FAKE_D_DRAWS,
    allow_nonformal_draws: bool = False,
) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"immutable output directory is not empty: {output_dir}")
    if not allow_nonformal_draws and (
        int(pairing_draws) != FORMAL_PAIRING_DRAWS or int(fake_d_draws) != FORMAL_FAKE_D_DRAWS
    ):
        raise ValueError("formal identification analysis requires 50 pairing and 50 Fake-D draws")
    output_dir.mkdir(parents=True, exist_ok=False)
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    inputs = load_frozen_incremental_inputs(output_root)
    membership_path = output_root / "00_splits" / "repeated_5fold_splits_10x5.csv"
    structural_path = output_root / "12_submission_audit" / "structural_baseline_features.csv"
    membership = _load_membership(membership_path)
    structural = _align_structural(structural_path, inputs.participant_ids)
    partition_domains(DOMAIN_FEATURES)
    for column in PAIRING_MATCH_COLUMNS:
        if column not in structural.columns:
            raise ValueError(f"pairing structural column missing: {column}")

    pairs: dict[str, FrozenEmbeddingPair] = {}
    for slug, dimension in EXPECTED_DIMENSIONS.items():
        pairs[slug] = load_frozen_embedding_pair(
            embedding_root,
            slug,
            inputs.participant_ids,
            expected_dim=dimension,
        )

    ids = inputs.participant_ids
    real = np.arange(len(ids), dtype=int)
    pairing_maps: list[tuple[str, int, np.ndarray]] = [("real", 0, real)]
    for draw_id in range(1, int(pairing_draws) + 1):
        pairing_maps.append(
            ("random", draw_id, make_derangement(len(ids), seed=BASE_SEED + 10_000 + draw_id))
        )
        pairing_maps.append(
            (
                "structure_matched_random",
                draw_id,
                make_structure_matched_permutation(
                    structural,
                    columns=PAIRING_MATCH_COLUMNS,
                    seed=BASE_SEED + 20_000 + draw_id,
                ),
            )
        )

    pairing_ledger = pd.concat(
        [
            _pairing_ledger(
                ids,
                permutation,
                condition=condition,
                draw_id=draw_id,
                structural=structural,
                columns=PAIRING_MATCH_COLUMNS,
            )
            for condition, draw_id, permutation in pairing_maps
        ],
        ignore_index=True,
    )
    pairing_tasks = [
        (slug, fusion_method, condition, draw_id, permutation)
        for slug in pairs
        for fusion_method in ("late_fusion", "early_concat")
        for condition, draw_id, permutation in pairing_maps
    ]

    def pair_worker(task):
        slug, fusion_method, condition, draw_id, permutation = task
        return _pair_one_condition(
            pairs[slug],
            inputs,
            membership,
            condition=condition,
            draw_id=draw_id,
            permutation=permutation,
            fusion_method=fusion_method,
            base_seed=BASE_SEED,
        )

    pairing_results = _run_parallel(pairing_tasks, pair_worker, workers)
    pairing_predictions = pd.concat([item[0] for item in pairing_results], ignore_index=True)
    pairing_metrics = pd.DataFrame([item[1] for item in pairing_results])
    pairing_metrics = pairing_metrics.sort_values(
        ["embedding_model", "fusion_method", "condition", "draw_id"], kind="stable"
    ).reset_index(drop=True)
    pairing_summary = _summary_by_condition(
        pairing_metrics,
        metric_columns=("auc", "pr_auc", "brier", "log_loss"),
    )

    fake_maps: list[tuple[str, int, np.ndarray]] = [("real", 0, real)]
    for draw_id in range(1, int(fake_d_draws) + 1):
        fake_maps.append(
            ("fake_d", draw_id, make_fake_domain_permutation(len(ids), seed=BASE_SEED + 30_000 + draw_id))
        )
    fake_ledger = pd.concat(
        [
            _pairing_ledger(
                ids,
                permutation,
                condition=condition,
                draw_id=draw_id,
                structural=structural,
                columns=PAIRING_MATCH_COLUMNS,
            )
            for condition, draw_id, permutation in fake_maps
        ],
        ignore_index=True,
    ).rename(columns={"pairing_condition": "condition"})
    fake_tasks = [
        (slug, fusion_method, condition, draw_id, permutation)
        for slug in pairs
        for fusion_method in ("late_fusion", "early_concat")
        for condition, draw_id, permutation in fake_maps
    ]

    def fake_worker(task):
        slug, fusion_method, condition, draw_id, permutation = task
        active_inputs = inputs
        variant = "D-P"
        if condition == "fake_d":
            active_inputs = replace_domains(
                inputs,
                DOMAIN_FEATURES,
                matrix=inputs.domains[permutation],
            )
            variant = "Fake-D"
        return _fake_or_partition_one_condition(
            pairs[slug],
            active_inputs,
            membership,
            experiment="fake_d",
            condition=condition,
            draw_id=draw_id,
            domain_variant=variant,
            fusion_method=fusion_method,
            base_seed=BASE_SEED,
        )

    fake_results = _run_parallel(fake_tasks, fake_worker, workers)
    fake_predictions = pd.concat([item[0] for item in fake_results], ignore_index=True)
    fake_metrics = pd.DataFrame([row for item in fake_results for row in item[1]])
    fake_metrics = fake_metrics.sort_values(
        ["embedding_model", "fusion_method", "condition", "draw_id", "model_name"],
        kind="stable",
    ).reset_index(drop=True)
    fake_summary = _summary_by_condition(
        fake_metrics.loc[fake_metrics["model_name"].ne("augmented") | fake_metrics["model_name"].isna()],
        metric_columns=("auc", "pr_auc", "brier", "log_loss", "delta_auc", "delta_pr_auc", "delta_brier", "delta_log_loss"),
    )

    partition_tasks = [
        (slug, fusion_method, variant, columns)
        for slug in pairs
        for fusion_method in ("late_fusion", "early_concat")
        for variant, columns in (("D-PHQ", D_PHQ_FEATURES), ("D-Extra", D_EXTRA_FEATURES))
    ]

    def partition_worker(task):
        slug, fusion_method, variant, columns = task
        active_inputs = replace_domains(inputs, columns)
        return _fake_or_partition_one_condition(
            pairs[slug],
            active_inputs,
            membership,
            experiment="phq_partition",
            condition=variant,
            draw_id=0,
            domain_variant=variant,
            fusion_method=fusion_method,
            base_seed=BASE_SEED,
        )

    partition_results = _run_parallel(partition_tasks, partition_worker, workers)
    partition_predictions = pd.concat([item[0] for item in partition_results], ignore_index=True)
    partition_metrics = pd.DataFrame([row for item in partition_results for row in item[1]])
    partition_metrics = partition_metrics.sort_values(
        ["embedding_model", "fusion_method", "condition", "model_name"], kind="stable"
    ).reset_index(drop=True)

    _write_csv(pairing_predictions, output_dir / "pairing_predictions.csv")
    _write_csv(pairing_metrics, output_dir / "pairing_metrics_by_draw.csv")
    _write_csv(pairing_summary, output_dir / "pairing_summary.csv")
    _write_csv(pairing_ledger, output_dir / "pairing_ledger.csv")
    _write_csv(fake_predictions, output_dir / "fake_d_predictions.csv")
    _write_csv(fake_metrics, output_dir / "fake_d_metrics_by_draw.csv")
    _write_csv(fake_summary, output_dir / "fake_d_summary.csv")
    _write_csv(fake_ledger, output_dir / "fake_d_ledger.csv")
    _write_csv(partition_predictions, output_dir / "phq_partition_predictions.csv")
    _write_csv(partition_metrics, output_dir / "phq_partition_metrics.csv")

    output_hashes = {
        path.name: _sha256(path)
        for path in sorted(output_dir.glob("*.csv"))
    }
    manifest = {
        "status": "complete",
        "run_class": "formal" if int(pairing_draws) == FORMAL_PAIRING_DRAWS and int(fake_d_draws) == FORMAL_FAKE_D_DRAWS else "smoke_test",
        "analysis": "post_audit_identification_sensitivity",
        "scope": "D-P; no D-All model",
        "cohort_n": int(len(inputs.participant_ids)),
        "positive_n": int(inputs.labels.sum()),
        "repeat": FORMAL_REPEAT,
        "inner_folds": FORMAL_INNER_FOLDS,
        "c_grid": list(DEFAULT_C_GRID),
        "pairing_draws": int(pairing_draws),
        "fake_d_draws": int(fake_d_draws),
        "base_seed": BASE_SEED,
        "workers": int(workers),
        "inputs": {
            "structural": {"path": str(structural_path), "sha256": _sha256(structural_path)},
            "splits": {"path": str(membership_path), "sha256": _sha256(membership_path)},
        },
        "embedding_root": str(embedding_root),
        "embedding_models": {
            slug: {
                "model_name": pair.model_name,
                "revision": pair.revision,
                "embedding_dim": int(pair.embedding_dim),
                "cache_files": list(pair.cache_files),
                "manifest_hashes": pair.manifest_hashes,
            }
            for slug, pair in pairs.items()
        },
        "pairing_match_columns": list(PAIRING_MATCH_COLUMNS),
        "domain_partition": {"D-PHQ": list(D_PHQ_FEATURES), "D-Extra": list(D_EXTRA_FEATURES)},
        "output_file_hashes": output_hashes,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run post-audit interviewer-signal identification sensitivities")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "analysis_v2")
    parser.add_argument(
        "--embedding-root",
        type=Path,
        default=REPO_ROOT / "analysis_v2" / "03_embedding_robustness",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "analysis_v2" / "10_source_importance" / "08_identification_sensitivity" / "run_1",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--pairing-draws", type=int, default=FORMAL_PAIRING_DRAWS)
    parser.add_argument("--fake-d-draws", type=int, default=FORMAL_FAKE_D_DRAWS)
    parser.add_argument(
        "--allow-nonformal-draws",
        action="store_true",
        help="allow smoke-sized draw counts; output remains explicitly nonformal",
    )
    args = parser.parse_args()
    manifest = run_analysis(
        output_root=args.output_root,
        embedding_root=args.embedding_root,
        output_dir=args.output_dir,
        workers=args.workers,
        pairing_draws=args.pairing_draws,
        fake_d_draws=args.fake_d_draws,
        allow_nonformal_draws=args.allow_nonformal_draws,
    )
    print(json.dumps({"status": manifest["status"], "output_dir": str(args.output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
