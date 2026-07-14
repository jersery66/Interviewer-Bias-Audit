"""Run the locked ten-repeat pairing/Fake-D identification sensitivity analysis.

This runner is deliberately separate from the exploratory repeat-1 runner.  It
uses only the two frozen embeddings and early-concat models, creates the same
label-blind draw maps for both embeddings, and writes a new immutable output
directory.  The 50 draws are averaged within each outer repeat before any
confidence interval, sign-flip test, or BH-FDR calculation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from reanalysis_v2.identification_formal import (  # noqa: E402
    FORMAL_DRAW_COUNT,
    FORMAL_REPEATS,
    build_primary_repeat_contrasts,
    summarize_primary_contrasts,
)
from reanalysis_v2.identification_sensitivity import (  # noqa: E402
    PAIRING_MATCH_COLUMNS,
    make_derangement,
    make_fake_domain_permutation,
    make_structure_matched_permutation,
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
INNER_FOLDS = 3
FUSION_METHOD = "early_concat"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_membership(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"repeat", "fold", "role", "participant_id", "label"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"split membership is missing columns: {missing}")
    frame["repeat"] = pd.to_numeric(frame["repeat"], errors="raise").astype(int)
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    frame["participant_id"] = pd.to_numeric(frame["participant_id"], errors="raise").astype(int)
    repeats = sorted(frame["repeat"].unique().tolist())
    if repeats != list(FORMAL_REPEATS):
        raise ValueError(f"split membership must contain repeats 1..10, got {repeats}")
    for repeat in FORMAL_REPEATS:
        folds = sorted(frame.loc[frame["repeat"].eq(repeat), "fold"].unique().tolist())
        if folds != [1, 2, 3, 4, 5]:
            raise ValueError(f"repeat {repeat} must contain five outer folds")
    return frame


def _align_structural(path: Path, participant_ids: np.ndarray) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "participant_id" not in frame.columns or frame["participant_id"].duplicated().any():
        raise ValueError("structural input must contain unique participant_id values")
    frame["participant_id"] = pd.to_numeric(frame["participant_id"], errors="raise").astype(int)
    indexed = frame.set_index("participant_id")
    ids = [int(value) for value in participant_ids.tolist()]
    missing = sorted(set(ids).difference(indexed.index.tolist()))
    if missing:
        raise ValueError(f"structural input is missing participant IDs: {missing[:5]}")
    result = indexed.loc[ids].reset_index()
    for column in PAIRING_MATCH_COLUMNS:
        if column not in result.columns:
            raise ValueError(f"pairing structural column missing: {column}")
        result[column] = pd.to_numeric(result[column], errors="raise")
    return result


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def _ledger(
    participant_ids: np.ndarray,
    permutation: np.ndarray,
    *,
    repeat: int,
    condition: str,
    draw_id: int,
    seed: int,
    structural: pd.DataFrame,
) -> pd.DataFrame:
    ids = np.asarray(participant_ids, dtype=int)
    permutation = np.asarray(permutation, dtype=int)
    values = structural.loc[:, list(PAIRING_MATCH_COLUMNS)].to_numpy(dtype=float)
    scale = values.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    z = (values - values.mean(axis=0)) / scale
    distance = np.square(z - z[permutation]).sum(axis=1)
    return pd.DataFrame(
        {
            "repeat": int(repeat),
            "condition": condition,
            "draw_id": int(draw_id),
            "seed": int(seed),
            "recipient_participant_id": ids,
            "donor_participant_id": ids[permutation],
            "donor_row_index": permutation,
            "self_match": permutation == np.arange(len(ids)),
            "standardized_structural_distance": distance,
        }
    )


def _pair_task(
    pair: FrozenEmbeddingPair,
    inputs: FrozenIncrementalInputs,
    membership: pd.DataFrame,
    *,
    repeat: int,
    condition: str,
    draw_id: int,
    permutation: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, object]]:
    active_pair = pair if condition == "real" else permute_interviewer_pair(pair, permutation)
    prediction = run_single_level_oof(
        pair=active_pair,
        inputs=inputs,
        membership=membership,
        repeat=int(repeat),
        level="L1",
        augmented=True,
        fusion_method=FUSION_METHOD,
        inner_folds=INNER_FOLDS,
        c_grid=DEFAULT_C_GRID,
        base_seed=BASE_SEED,
    )
    prediction = prediction.copy()
    prediction.insert(0, "condition", condition)
    prediction.insert(1, "draw_id", int(draw_id))
    metrics = prediction_metrics(prediction)
    metric_row: dict[str, object] = {
        "embedding_model": pair.slug,
        "repeat": int(repeat),
        "condition": condition,
        "draw_id": int(draw_id),
        "model_level": "L1",
        "fusion_method": FUSION_METHOD,
        "n": int(len(prediction)),
        **metrics,
    }
    return prediction, metric_row


def _fake_d_task(
    pair: FrozenEmbeddingPair,
    inputs: FrozenIncrementalInputs,
    membership: pd.DataFrame,
    *,
    repeat: int,
    condition: str,
    draw_id: int,
    permutation: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, object]]:
    active_inputs = inputs
    domain_variant = "D-P"
    if condition == "fake_d":
        active_inputs = replace_domains(inputs, DOMAIN_FEATURES, matrix=inputs.domains[permutation])
        domain_variant = "Fake-D"
    prediction = run_level_pair_oof(
        pair=pair,
        inputs=active_inputs,
        membership=membership,
        repeat=int(repeat),
        level="L3",
        fusion_method=FUSION_METHOD,
        inner_folds=INNER_FOLDS,
        c_grid=DEFAULT_C_GRID,
        base_seed=BASE_SEED,
    )
    prediction = prediction.copy()
    prediction.insert(0, "condition", condition)
    prediction.insert(1, "draw_id", int(draw_id))
    prediction.insert(2, "domain_variant", domain_variant)
    base = prediction.loc[prediction["model_name"].eq("base")].copy()
    augmented = prediction.loc[prediction["model_name"].eq("augmented")].copy()
    merged = base[["participant_id", "label", "probability"]].rename(
        columns={"probability": "base_probability"}
    ).merge(
        augmented[["participant_id", "probability"]].rename(
            columns={"probability": "augmented_probability"}
        ),
        on="participant_id",
        validate="one_to_one",
    )
    values = paired_metric_deltas(
        merged["label"].to_numpy(dtype=int),
        merged["base_probability"].to_numpy(dtype=float),
        merged["augmented_probability"].to_numpy(dtype=float),
    )
    metric_row: dict[str, object] = {
        "embedding_model": pair.slug,
        "repeat": int(repeat),
        "condition": condition,
        "draw_id": int(draw_id),
        "domain_variant": domain_variant,
        "model_level": "L3",
        "fusion_method": FUSION_METHOD,
        "n": int(len(merged)),
        **values,
    }
    return prediction, metric_row


def _run_parallel(tasks: list[tuple], worker, workers: int) -> list[tuple]:
    if int(workers) < 1:
        raise ValueError("workers must be at least one")
    results: list[tuple] = []
    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        futures = [executor.submit(worker, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if index % 100 == 0 or index == len(futures):
                print(f"completed {index}/{len(futures)} model fits", flush=True)
    return results


def _draw_maps(
    participant_ids: np.ndarray,
    structural: pd.DataFrame,
) -> tuple[list[dict[str, object]], list[dict[str, object]], pd.DataFrame, pd.DataFrame]:
    n = len(participant_ids)
    identity = np.arange(n, dtype=int)
    pairing_tasks: list[dict[str, object]] = []
    fake_tasks: list[dict[str, object]] = []
    pairing_ledgers: list[pd.DataFrame] = []
    fake_ledgers: list[pd.DataFrame] = []
    for repeat in FORMAL_REPEATS:
        real_seed = BASE_SEED + int(repeat) * 1_000_000
        pairing_tasks.append({"repeat": repeat, "condition": "real", "draw_id": 0, "seed": real_seed, "permutation": identity})
        pairing_ledgers.append(
            _ledger(participant_ids, identity, repeat=repeat, condition="real", draw_id=0, seed=real_seed, structural=structural)
        )
        fake_tasks.append({"repeat": repeat, "condition": "real", "draw_id": 0, "seed": real_seed, "permutation": identity})
        fake_ledgers.append(
            _ledger(participant_ids, identity, repeat=repeat, condition="real", draw_id=0, seed=real_seed, structural=structural)
        )
        for draw_id in range(1, FORMAL_DRAW_COUNT + 1):
            random_seed = BASE_SEED + 10_000_000 + int(repeat) * 100_000 + draw_id
            random_map = make_derangement(n, seed=random_seed)
            pairing_tasks.append({"repeat": repeat, "condition": "random", "draw_id": draw_id, "seed": random_seed, "permutation": random_map})
            pairing_ledgers.append(
                _ledger(participant_ids, random_map, repeat=repeat, condition="random", draw_id=draw_id, seed=random_seed, structural=structural)
            )
            matched_seed = BASE_SEED + 20_000_000 + int(repeat) * 100_000 + draw_id
            matched_map = make_structure_matched_permutation(
                structural, columns=PAIRING_MATCH_COLUMNS, seed=matched_seed
            )
            pairing_tasks.append({"repeat": repeat, "condition": "matched", "draw_id": draw_id, "seed": matched_seed, "permutation": matched_map})
            pairing_ledgers.append(
                _ledger(participant_ids, matched_map, repeat=repeat, condition="matched", draw_id=draw_id, seed=matched_seed, structural=structural)
            )
            fake_seed = BASE_SEED + 30_000_000 + int(repeat) * 100_000 + draw_id
            fake_map = make_fake_domain_permutation(n, seed=fake_seed)
            fake_tasks.append({"repeat": repeat, "condition": "fake_d", "draw_id": draw_id, "seed": fake_seed, "permutation": fake_map})
            fake_ledgers.append(
                _ledger(participant_ids, fake_map, repeat=repeat, condition="fake_d", draw_id=draw_id, seed=fake_seed, structural=structural)
            )
    return pairing_tasks, fake_tasks, pd.concat(pairing_ledgers, ignore_index=True), pd.concat(fake_ledgers, ignore_index=True)


def _git_state() -> tuple[str, str]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True)
    return head, status


def _hash_outputs(output_dir: Path) -> dict[str, str]:
    return {
        path.name: _sha256(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "run_manifest.json"
    }


def _manifest_path(path: Path) -> str:
    """Serialize filesystem paths with forward slashes for valid portable JSON."""

    return Path(path).as_posix()


def run_analysis(
    *,
    output_root: Path,
    embedding_root: Path,
    output_dir: Path,
    workers: int = 4,
    code_commit: str,
) -> dict[str, object]:
    """Execute the locked formal run and return its manifest."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"immutable output directory is not empty: {output_dir}")
    head, status = _git_state()
    if str(code_commit) != head:
        raise ValueError(f"code_commit must equal current HEAD {head}, got {code_commit}")
    if status.strip():
        raise RuntimeError("formal run requires a clean git worktree; run git status --porcelain first")
    output_dir.mkdir(parents=True, exist_ok=False)
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    inputs = load_frozen_incremental_inputs(output_root)
    membership_path = output_root / "00_splits" / "repeated_5fold_splits_10x5.csv"
    structural_path = output_root / "12_submission_audit" / "structural_baseline_features.csv"
    membership = _load_membership(membership_path)
    structural = _align_structural(structural_path, inputs.participant_ids)
    pairs: dict[str, FrozenEmbeddingPair] = {
        slug: load_frozen_embedding_pair(
            embedding_root,
            slug,
            inputs.participant_ids,
            expected_dim=dimension,
        )
        for slug, dimension in EXPECTED_DIMENSIONS.items()
    }
    pairing_tasks, fake_tasks, pairing_ledger, fake_ledger = _draw_maps(
        inputs.participant_ids, structural
    )
    pairing_jobs = [
        (slug, task)
        for slug in EXPECTED_DIMENSIONS
        for task in pairing_tasks
    ]
    fake_jobs = [
        (slug, task)
        for slug in EXPECTED_DIMENSIONS
        for task in fake_tasks
    ]

    def pairing_worker(job):
        slug, task = job
        return _pair_task(
            pairs[slug],
            inputs,
            membership,
            repeat=task["repeat"],
            condition=task["condition"],
            draw_id=task["draw_id"],
            permutation=task["permutation"],
        )

    def fake_worker(job):
        slug, task = job
        return _fake_d_task(
            pairs[slug],
            inputs,
            membership,
            repeat=task["repeat"],
            condition=task["condition"],
            draw_id=task["draw_id"],
            permutation=task["permutation"],
        )

    pairing_results = _run_parallel(pairing_jobs, pairing_worker, workers)
    fake_results = _run_parallel(fake_jobs, fake_worker, workers)
    pairing_predictions = pd.concat([result[0] for result in pairing_results], ignore_index=True)
    pairing_metrics = pd.DataFrame([result[1] for result in pairing_results])
    fake_predictions = pd.concat([result[0] for result in fake_results], ignore_index=True)
    fake_metrics = pd.DataFrame([result[1] for result in fake_results])
    pairing_metrics = pairing_metrics.sort_values(
        ["embedding_model", "repeat", "condition", "draw_id"], kind="stable"
    ).reset_index(drop=True)
    fake_metrics = fake_metrics.sort_values(
        ["embedding_model", "repeat", "condition", "draw_id"], kind="stable"
    ).reset_index(drop=True)
    repeat_contrasts = build_primary_repeat_contrasts(pairing_metrics, fake_metrics)
    primary_contrasts = summarize_primary_contrasts(repeat_contrasts, n_bootstrap=5000, seed=BASE_SEED + 90_000_000)

    _write_csv(pairing_metrics, output_dir / "pairing_draw_metrics.csv")
    _write_csv(pairing_predictions.sort_values(["embedding_model", "repeat", "condition", "draw_id", "participant_id"], kind="stable"), output_dir / "pairing_oof_predictions.csv")
    _write_csv(fake_metrics, output_dir / "fake_d_draw_metrics.csv")
    _write_csv(fake_predictions.sort_values(["embedding_model", "repeat", "condition", "draw_id", "model_name", "participant_id"], kind="stable"), output_dir / "fake_d_oof_predictions.csv")
    _write_csv(pairing_ledger, output_dir / "pairing_ledger.csv")
    _write_csv(fake_ledger, output_dir / "fake_d_ledger.csv")
    _write_csv(repeat_contrasts, output_dir / "repeat_contrasts.csv")
    _write_csv(primary_contrasts, output_dir / "primary_contrasts.csv")

    output_hashes = _hash_outputs(output_dir)
    manifest = {
        "status": "complete",
        "analysis": "post_audit_identification_sensitivity_formal",
        "run_class": "formal",
        "scope": "pairing destruction and Fake-D only; no new models; no D-All",
        "cohort_n": int(len(inputs.participant_ids)),
        "positive_n": int(inputs.labels.sum()),
        "negative_n": int((inputs.labels == 0).sum()),
        "repeats": list(FORMAL_REPEATS),
        "outer_folds": 5,
        "inner_folds": INNER_FOLDS,
        "fusion_method": FUSION_METHOD,
        "embedding_models": list(EXPECTED_DIMENSIONS),
        "pairing_draws_per_repeat": FORMAL_DRAW_COUNT,
        "fake_d_draws_per_repeat": FORMAL_DRAW_COUNT,
        "primary_contrasts": [
            "real_minus_matched",
            "real_minus_random",
            "d_delta_minus_fake",
        ],
        "inference": {
            "repeat_level_unit": "ten outer repeat contrasts",
            "ci": "5000-repeat bootstrap percentile 95% interval",
            "permutation": "exact two-sided sign-flip over ten repeat means",
            "fdr": "BH across six embedding-by-contrast primary rows",
        },
        "base_seed": BASE_SEED,
        "workers": int(workers),
        "code_commit": head,
        "inputs": {
            "structural": {"path": _manifest_path(structural_path), "sha256": _sha256(structural_path)},
            "splits": {"path": _manifest_path(membership_path), "sha256": _sha256(membership_path)},
            "domain_count": {
                "path": _manifest_path(output_root / "04_c5_controls" / "domain_count" / "input.csv"),
                "sha256": _sha256(output_root / "04_c5_controls" / "domain_count" / "input.csv"),
            },
        },
        "embedding_root": _manifest_path(embedding_root),
        "embedding_manifests": {
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
        "output_file_hashes": output_hashes,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run formal ten-repeat pairing/Fake-D identification sensitivity")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "analysis_v2")
    parser.add_argument("--embedding-root", type=Path, default=REPO_ROOT / "analysis_v2" / "03_embedding_robustness")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "analysis_v2" / "10_source_importance" / "08_identification_sensitivity" / "run_formal_10x")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    manifest = run_analysis(
        output_root=args.output_root,
        embedding_root=args.embedding_root,
        output_dir=args.output_dir,
        workers=args.workers,
        code_commit=args.code_commit,
    )
    print(json.dumps({"status": manifest["status"], "output_dir": str(args.output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
