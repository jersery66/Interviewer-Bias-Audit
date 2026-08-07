from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reanalysis_v2.embedding_incremental_reporting import (  # noqa: E402
    make_incremental_main_table,
    add_global_bh,
    write_markdown_table,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _forest(table: pd.DataFrame, output_dir: Path) -> None:
    ordered = table.sort_values(["embedding", "fusion", "level"], kind="stable").reset_index(drop=True)
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(7.2, 5.7))
    colors = {"late_fusion": "#0F4D92", "early_concat": "#B77A3C"}
    for row_index, row in ordered.iterrows():
        value = float(row["delta_auc"])
        lower = value - float(row["auc_ci_low"])
        upper = float(row["auc_ci_high"]) - value
        ax.errorbar(
            value,
            row_index,
            xerr=np.array([[lower], [upper]]),
            fmt="o",
            color=colors[str(row["fusion"])],
            ecolor=colors[str(row["fusion"])],
            capsize=2.5,
            markersize=4.5,
            linewidth=1.0,
        )
    labels = [f"{r.embedding} | {r.fusion.replace('_', ' ')} | {r.level}" for r in ordered.itertuples()]
    ax.set_yticks(y, labels)
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_xlabel("Interviewer increment ΔAUC (augmented − base)")
    ax.set_title("Embedding incremental results: all 16 paired comparisons")
    ax.legend(
        [plt.Line2D([0], [0], marker="o", color=color, linestyle="") for color in colors.values()],
        ["Late fusion", "Early concatenation"],
        frameon=False,
        loc="lower right",
    )
    fig.text(
        0.5,
        0.01,
        "Error bars: 95% participant-bootstrap CI. Global BH q-values over all 16 AUC deltas are a sensitivity family; locked primary/secondary q-values are retained separately.",
        ha="center",
        fontsize=6.5,
    )
    fig.subplots_adjust(left=0.25, right=0.98, bottom=0.14, top=0.92)
    for suffix, kwargs in (("svg", {}), ("pdf", {}), ("png", {"dpi": 240})):
        fig.savefig(output_dir / f"embedding_incremental_delta_forest.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def _trajectory(table: pd.DataFrame, output_dir: Path) -> None:
    metrics = table.melt(
        id_vars=["embedding", "fusion", "level"],
        value_vars=["base_roc_auc", "augmented_roc_auc"],
        var_name="model",
        value_name="auc",
    )
    metrics["model"] = metrics["model"].map(
        {"base_roc_auc": "Base", "augmented_roc_auc": "Augmented"}
    )
    order = {"L1": 1, "L2": 2, "L3": 3, "L4": 4}
    metrics["level_order"] = metrics["level"].map(order)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.1), sharey=True)
    for ax, (embedding, fusion) in zip(axes.flat, [("model_1_all_mpnet_base_v2", "early_concat"), ("model_1_all_mpnet_base_v2", "late_fusion"), ("model_2_bge_large_en_v1_5", "early_concat"), ("model_2_bge_large_en_v1_5", "late_fusion")]):
        subset = metrics.loc[(metrics["embedding"] == embedding) & (metrics["fusion"] == fusion)]
        for model, color, marker in (("Base", "#666666", "o"), ("Augmented", "#0F4D92", "s")):
            line = subset.loc[subset["model"] == model].sort_values("level_order")
            ax.plot(line["level"], line["auc"], marker=marker, color=color, linewidth=1.2, label=model)
        ax.set_title(f"{embedding.replace('model_', '').replace('_all_mpnet_base_v2', ' MPNet').replace('_bge_large_en_v1_5', ' BGE')} | {fusion.replace('_', ' ')}")
        ax.set_ylim(0.55, 0.86)
        ax.set_xlabel("Model level")
        ax.set_ylabel("ROC AUC")
        ax.axhline(0.5, color="#AAAAAA", linestyle="--", linewidth=0.6)
    axes.flat[0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.suptitle("Base and interviewer-augmented ROC AUC across L1–L4", y=0.99, fontsize=10)
    fig.text(0.5, 0.01, "D-P is the frozen participant-language-derived symptom-domain control used at L3/L4.", ha="center", fontsize=6.5)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.88, hspace=0.4, wspace=0.25)
    for suffix, kwargs in (("svg", {}), ("pdf", {}), ("png", {"dpi": 240})):
        fig.savefig(output_dir / f"embedding_incremental_auc_trajectory.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def generate_report(final_dir: Path, output_dir: Path) -> dict[str, object]:
    deltas_path = final_dir / "embedding_incremental_deltas.csv"
    metrics_path = final_dir / "embedding_incremental_model_metrics.csv"
    if not deltas_path.exists() or not metrics_path.exists():
        raise FileNotFoundError("formal embedding incremental deltas and metrics are required")
    deltas = pd.read_csv(deltas_path)
    metrics = pd.read_csv(metrics_path)
    deltas = deltas.rename(columns={"embedding": "embedding", "fusion": "fusion", "level": "level"})
    table = make_incremental_main_table(deltas, metrics)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "embedding_incremental_main_results.csv", index=False)
    write_markdown_table(table, output_dir / "embedding_incremental_main_results.md")
    _configure()
    _forest(table, output_dir)
    _trajectory(table, output_dir)
    manifest = {
        "status": "complete",
        "report": "embedding_incremental_reader_facing_tables_and_figures",
        "n_rows": int(len(table)),
        "global_bh_family": "all 16 embedding x fusion x level AUC deltas",
        "global_bh_q_column": "global_bh_q16",
        "result_scope_counts": table["result_scope"].value_counts().to_dict(),
        "sources": {
            "deltas": {
                "path": "analysis_v2/03_embedding_robustness/embedding_incremental/final/embedding_incremental_deltas.csv",
                "sha256": _sha256(deltas_path),
            },
            "metrics": {
                "path": "analysis_v2/03_embedding_robustness/embedding_incremental/final/embedding_incremental_model_metrics.csv",
                "sha256": _sha256(metrics_path),
            },
        },
        "outputs": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate tables and figures from frozen embedding incremental outputs")
    parser.add_argument("--final-dir", type=Path, default=REPO_ROOT / "analysis_v2" / "03_embedding_robustness" / "embedding_incremental" / "final")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "analysis_v2" / "06_tables_figures" / "embedding_incremental")
    args = parser.parse_args()
    result = generate_report(args.final_dir, args.output_dir)
    print(json.dumps({"status": result["status"], "n_rows": result["n_rows"], "output": str(args.output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
