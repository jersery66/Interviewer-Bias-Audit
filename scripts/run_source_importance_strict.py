from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) in sys.path:
    sys.path.remove(str(PACKAGE_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT))

from reanalysis_v2.source_importance_strict import (  # noqa: E402
    BASE_SEED,
    infer_incremental_comparisons,
    load_strict_source_frame,
    run_domain_leave_one_out,
    run_group_permutation_importance,
    run_repeated_cv,
    summarize_model_metrics,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fold-safe raw-source incremental models and FDR-controlled inference."
    )
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "analysis_v2",
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--importance-shuffles", type=int, default=100)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analysis_root = args.analysis_root.resolve()
    output = analysis_root / "10_source_importance" / "07_strict_joint_models"
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)

    frame = load_strict_source_frame(analysis_root)
    split_path = analysis_root / "00_splits" / "repeated_5fold_splits_10x5.csv"
    splits = pd.read_csv(split_path)
    if splits["repeat"].nunique() != 10 or splits["fold"].nunique() != 5:
        raise ValueError("Expected frozen 10x5 split contract")

    fold_rows, participant_rows = run_repeated_cv(frame, splits, base_seed=args.seed)
    metrics, auc_ci = summarize_model_metrics(
        participant_rows,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    comparisons = infer_incremental_comparisons(
        participant_rows,
        n_bootstrap=args.bootstrap,
        n_permutations=args.permutations,
        seed=args.seed,
    )
    fold_rows.to_csv(output / "model_fold_oof_predictions.csv", index=False, encoding="utf-8")
    participant_rows.to_csv(output / "participant_oof_predictions.csv", index=False, encoding="utf-8")
    metrics.to_csv(output / "model_metrics.csv", index=False, encoding="utf-8")
    auc_ci.to_csv(output / "model_auc_ci.csv", index=False, encoding="utf-8")
    comparisons.to_csv(output / "incremental_comparisons_fdr.csv", index=False, encoding="utf-8")
    print("stage=incremental_models complete", flush=True)
    importance, importance_predictions = run_group_permutation_importance(
        frame,
        splits,
        shuffles_per_fold=args.importance_shuffles,
        n_bootstrap=args.bootstrap,
        n_permutations=args.permutations,
        seed=args.seed,
    )
    importance.to_csv(output / "permutation_importance_fdr.csv", index=False, encoding="utf-8")
    importance_predictions.to_csv(
        output / "permutation_participant_predictions.csv", index=False, encoding="utf-8"
    )
    print("stage=permutation_importance complete", flush=True)
    domain_loo = run_domain_leave_one_out(
        frame,
        splits,
        n_bootstrap=args.bootstrap,
        n_permutations=args.permutations,
        seed=args.seed,
    )
    domain_loo.to_csv(output / "domain_leave_one_out_fdr.csv", index=False, encoding="utf-8")
    print("stage=domain_leave_one_out complete", flush=True)

    outputs = {
        "model_fold_oof_predictions.csv": fold_rows,
        "participant_oof_predictions.csv": participant_rows,
        "model_metrics.csv": metrics,
        "model_auc_ci.csv": auc_ci,
        "incremental_comparisons_fdr.csv": comparisons,
        "permutation_importance_fdr.csv": importance,
        "permutation_participant_predictions.csv": importance_predictions,
        "domain_leave_one_out_fdr.csv": domain_loo,
    }
    for filename, table in outputs.items():
        table.to_csv(output / filename, index=False, encoding="utf-8")

    input_paths = [
        analysis_root / "01_inputs" / "c2_participant_speech.csv",
        analysis_root / "01_inputs" / "c3_interviewer_speech.csv",
        analysis_root / "01_inputs" / "c5_reviewed_symptom_evidence.csv",
        analysis_root / "04_c5_controls" / "domain_presence" / "input.csv",
        analysis_root / "04_c5_controls" / "domain_count" / "input.csv",
        analysis_root / "05_c4_controls" / "template_presence" / "input.csv",
        split_path,
    ]
    manifest = {
        "status": "complete",
        "analysis": "strict raw-source joint models without aggregated-OOF stacking",
        "started_at_utc": started.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": {"n": len(frame), "positive": int(frame["label"].sum()), "negative": int((frame["label"] == 0).sum())},
        "cv": {"repeats": 10, "folds": 5, "split_level": "participant", "preprocessing": "fit within outer training fold only"},
        "inference": {
            "bootstrap_resamples": args.bootstrap,
            "paired_permutations": args.permutations,
            "importance_shuffles_per_fold": args.importance_shuffles,
            "incremental_family_tests": len(comparisons),
            "importance_family_tests": len(importance),
            "domain_loo_family_tests": len(domain_loo),
            "multiplicity": "Benjamini-Hochberg within each prespecified family",
        },
        "seed": args.seed,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "statsmodels": statsmodels.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "inputs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in input_paths
        ],
    }
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    inventory_rows = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "output_manifest_sha256.csv":
            inventory_rows.append(
                {
                    "relative_path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    pd.DataFrame(inventory_rows).to_csv(
        output / "output_manifest_sha256.csv", index=False, encoding="utf-8"
    )

    print(f"strict_source_importance_complete output={output}")
    print(metrics[["model", "roc_auc", "pr_auc", "brier", "log_loss"]].to_string(index=False))
    print(comparisons[["comparison", "delta_auc", "ci_lower", "ci_upper", "p_value", "q_value", "significance"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
