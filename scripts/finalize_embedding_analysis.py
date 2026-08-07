from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from reanalysis_v2.analysis import PRIMARY_CONDITIONS
from reanalysis_v2.embedding_runner import build_representation_family_tests
from reanalysis_v2.io import atomic_write_csv, atomic_write_json, sha256_file
from reanalysis_v2.runner import _write_output_inventory


MODEL_SLUGS = ["model_1_all_mpnet_base_v2", "model_2_bge_large_en_v1_5"]


def _load_tables(root: Path, representation_root: Path) -> dict[str, pd.DataFrame]:
    return {
        condition: pd.read_csv(
            representation_root / "oof_probs" / f"{condition}_participant_oof.csv"
        )
        for condition in PRIMARY_CONDITIONS
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Finalize the joint representation robustness family"
    )
    parser.add_argument("--output", type=Path, default=Path("analysis_v2"))
    parser.add_argument("--permutations", type=int, default=10_000)
    args = parser.parse_args()
    output_root = args.output.resolve()
    representation_root = output_root / "03_embedding_robustness"
    started = datetime.now(timezone.utc)

    tfidf = _load_tables(output_root, output_root / "02_tfidf_main")
    embeddings = {
        slug: _load_tables(output_root, representation_root / slug)
        for slug in MODEL_SLUGS
    }
    mpnet_precomputed = pd.read_csv(
        representation_root
        / MODEL_SLUGS[0]
        / "paired_tests"
        / "input_source_paired_tests_descriptive.csv"
    )
    tests = build_representation_family_tests(
        tfidf,
        embeddings,
        precomputed_within={MODEL_SLUGS[0]: mpnet_precomputed},
        n_permutations=args.permutations,
        base_seed=20260624,
    )
    tests_path = representation_root / "representation_family_paired_tests_fdr.csv"
    atomic_write_csv(tests, tests_path)

    metric_tables = []
    tfidf_metrics = pd.read_csv(
        output_root / "02_tfidf_main" / "metrics" / "primary_metrics.csv"
    )
    tfidf_metrics.insert(0, "representation", "tfidf")
    metric_tables.append(tfidf_metrics)
    for slug in MODEL_SLUGS:
        table = pd.read_csv(
            representation_root / slug / "metrics" / "primary_metrics.csv"
        )
        table.insert(0, "representation", slug)
        metric_tables.append(table)
    combined = pd.concat(metric_tables, ignore_index=True)
    combined_path = representation_root / "combined_representation_metrics.csv"
    atomic_write_csv(combined, combined_path)

    inventory, inventory_hash = _write_output_inventory(representation_root)
    completed = datetime.now(timezone.utc)
    manifest = {
        "status": "complete",
        "started_at_utc": started.isoformat(),
        "completed_at_utc": completed.isoformat(),
        "elapsed_seconds": (completed - started).total_seconds(),
        "models": MODEL_SLUGS,
        "family": "representation_robustness",
        "family_test_count": int(len(tests)),
        "n_permutations": args.permutations,
        "fdr": "Benjamini-Hochberg jointly across all 120 predeclared tests",
        "mpnet_within_tests_reused_from_verified_10000_permutation_file": True,
        "bge_within_and_all_cross_representation_tests_recomputed": True,
        "tests_sha256": sha256_file(tests_path),
        "combined_metrics_sha256": sha256_file(combined_path),
        "output_file_count": int(len(inventory)),
        "output_inventory_sha256": inventory_hash,
    }
    atomic_write_json(manifest, representation_root / "run_manifest.json")
    print(json.dumps({"status": "complete", "tests": len(tests)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
