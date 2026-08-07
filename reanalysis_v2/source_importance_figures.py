from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


PALETTE = {
    "patient": "#7884B4",
    "domain": "#D49A73",
    "protocol": "#6FA89A",
    "evidence": "#B79AC8",
    "interviewer": "#C58B93",
    "primary": "#0F4D92",
    "neutral": "#767676",
    "light": "#E8EAF2",
    "signal": "#2E7D32",
    "negative": "#B64342",
}

MODEL_LABELS = {
    "M0": "M0  Patient speech",
    "M1": "M1  + domain counts",
    "M2": "M2  + protocol presence",
    "M3": "M3  + domain + protocol",
    "M4": "M4  M3 + symptom evidence",
    "M5": "M5  M3 + interviewer speech",
}

SOURCE_LABELS = {
    "c2_text": "Patient speech",
    "domain_count": "Domain counts",
    "template_presence": "Protocol presence",
}

DOMAIN_LABELS = {
    "anhedonia_interest": "Anhedonia / interest",
    "appetite_weight": "Appetite / weight",
    "concentration_psychomotor": "Concentration / psychomotor",
    "depressed_mood": "Depressed mood",
    "functioning_impairment": "Functional impairment",
    "mental_health_history": "Mental-health history",
    "protective_or_absent_symptom": "Protective / absent symptom",
    "self_worth_guilt": "Self-worth / guilt",
    "sleep_fatigue_energy": "Sleep / fatigue / energy",
    "suicide_self_harm": "Suicide / self-harm",
}


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(
        output_dir / f"{stem}.tiff",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(output_dir / f"{stem}.png", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    color: str,
    *,
    fontsize: float = 7,
    bold: bool = False,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor=color,
        edgecolor="#4D4D4D",
        linewidth=0.7,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
    )


def _figure_1(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.text(0.2, 7.62, "a", fontsize=9, fontweight="bold")
    ax.text(0.55, 7.62, "Fold-safe source decomposition", fontsize=9, fontweight="bold")

    inputs = [
        (0.5, "Patient speech\n(C2 text)", PALETTE["patient"]),
        (3.1, "Clinical-domain\ncounts", PALETTE["domain"]),
        (5.7, "Protocol-template\npresence", PALETTE["protocol"]),
        (8.3, "Sensitivity sources\n(C5 / C3 text)", PALETTE["evidence"]),
    ]
    for x, label, color in inputs:
        _box(ax, x, 6.2, 1.55, 0.85, label, color, bold=True)
        ax.annotate(
            "",
            xy=(5.0, 5.4),
            xytext=(x + 0.78, 6.18),
            arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#767676"},
        )

    _box(
        ax,
        2.0,
        4.45,
        6.0,
        0.95,
        "Within each outer fold: fit TF-IDF / scaling on training participants only",
        PALETTE["light"],
        fontsize=7.2,
        bold=True,
    )
    ax.annotate("", xy=(5.0, 3.78), xytext=(5.0, 4.42), arrowprops={"arrowstyle": "->", "lw": 0.9})
    _box(
        ax,
        2.0,
        2.8,
        6.0,
        0.95,
        "10×5 participant-level OOF predictions for M0–M5 joint models",
        "#DCE8F4",
        fontsize=7.2,
        bold=True,
    )
    ax.annotate("", xy=(5.0, 2.13), xytext=(5.0, 2.77), arrowprops={"arrowstyle": "->", "lw": 0.9})
    _box(
        ax,
        1.2,
        1.0,
        7.6,
        1.1,
        "Participant bootstrap CI + paired permutation tests + BH-FDR\n"
        "Only domain-count permutation importance remained significant (q=0.0039)",
        "#EEF4EA",
        fontsize=7.2,
        bold=True,
    )
    ax.text(
        5,
        0.42,
        "No incremental model comparison survived the seven-test FDR family.",
        ha="center",
        fontsize=6.8,
        color=PALETTE["neutral"],
    )
    _save_figure(fig, output_dir, "figure_1_source_decomposition_framework")


def _figure_2(strict_dir: Path, output_dir: Path) -> None:
    metrics = pd.read_csv(strict_dir / "model_metrics.csv")
    auc_ci = pd.read_csv(strict_dir / "model_auc_ci.csv")
    comparisons = pd.read_csv(strict_dir / "incremental_comparisons_fdr.csv")
    model_table = metrics.merge(auc_ci[["model", "ci_lower", "ci_upper"]], on="model")
    model_table["order"] = model_table["model"].str.extract(r"(\d+)").astype(int)
    model_table = model_table.sort_values("order")

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(7.2, 4.35),
        gridspec_kw={"width_ratios": [0.9, 1.2]},
    )
    y = np.arange(len(model_table))[::-1]
    auc = model_table["roc_auc"].to_numpy()
    lower = auc - model_table["ci_lower"].to_numpy()
    upper = model_table["ci_upper"].to_numpy() - auc
    colors = [
        PALETTE["patient"],
        PALETTE["domain"],
        PALETTE["protocol"],
        PALETTE["primary"],
        PALETTE["evidence"],
        PALETTE["interviewer"],
    ]
    for yi, value, lo, hi, color in zip(y, auc, lower, upper, colors):
        ax_a.errorbar(
            value,
            yi,
            xerr=np.array([[lo], [hi]]),
            fmt="o",
            color=color,
            ecolor=color,
            capsize=2.5,
            markersize=5,
            linewidth=1.2,
        )
        ax_a.text(value + 0.008, yi, f"{value:.4f}", va="center", fontsize=6.3)
    ax_a.set_yticks(y, [MODEL_LABELS[m] for m in model_table["model"]])
    ax_a.set_xlabel("ROC AUC")
    ax_a.set_xlim(0.52, 0.94)
    ax_a.axvline(0.5, color="#B0B0B0", linestyle="--", linewidth=0.7)
    ax_a.set_title("Model discrimination")
    ax_a.text(-0.18, 1.04, "a", transform=ax_a.transAxes, fontsize=9, fontweight="bold")

    comparison_labels = comparisons["comparison"].str.replace(" vs ", " − ", regex=False)
    yb = np.arange(len(comparisons))[::-1]
    delta = comparisons["delta_auc"].to_numpy()
    lower_delta = delta - comparisons["ci_lower"].to_numpy()
    upper_delta = comparisons["ci_upper"].to_numpy() - delta
    for yi, value, lo, hi, q_value, marker in zip(
        yb,
        delta,
        lower_delta,
        upper_delta,
        comparisons["q_value"],
        comparisons["significance"],
    ):
        significant = marker != "ns"
        color = PALETTE["signal"] if significant else PALETTE["primary"]
        ax_b.errorbar(
            value,
            yi,
            xerr=np.array([[lo], [hi]]),
            fmt="o",
            color=color,
            ecolor=color,
            capsize=2.5,
            markersize=4.5,
            linewidth=1.1,
        )
        ax_b.text(0.282, yi, f"q={q_value:.3f}  {marker}", va="center", ha="right", fontsize=6.1)
    ax_b.axvline(0, color="#4D4D4D", linewidth=0.8)
    ax_b.set_yticks(yb, comparison_labels)
    ax_b.axvline(0.225, color="#D9D9D9", linewidth=0.6)
    ax_b.set_xlim(-0.09, 0.29)
    ax_b.set_xlabel("Paired ΔAUC (model A − model B)")
    ax_b.set_title("Incremental comparisons")
    ax_b.text(-0.18, 1.04, "b", transform=ax_b.transAxes, fontsize=9, fontweight="bold")

    fig.text(
        0.5,
        0.01,
        "Points: participant-level repeated-OOF estimates; error bars: 95% participant bootstrap CI. "
        "Two-sided paired permutation tests with BH-FDR across seven comparisons; all comparisons ns.",
        ha="center",
        fontsize=5.9,
    )
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.17, top=0.88, wspace=0.52)
    _save_figure(fig, output_dir, "figure_2_incremental_model_auc")


def _figure_3(strict_dir: Path, output_dir: Path) -> None:
    table = pd.read_csv(strict_dir / "permutation_importance_fdr.csv")
    table = table.sort_values("delta_auc", ascending=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    y = np.arange(len(table))
    source_colors = {
        "c2_text": PALETTE["patient"],
        "domain_count": PALETTE["domain"],
        "template_presence": PALETTE["protocol"],
    }
    for yi, row in enumerate(table.itertuples(index=False)):
        value = float(row.delta_auc)
        lo = value - float(row.ci_lower)
        hi = float(row.ci_upper) - value
        color = source_colors[row.source]
        ax.errorbar(
            value,
            yi,
            xerr=np.array([[lo], [hi]]),
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=6,
            linewidth=1.4,
        )
        ax.text(
            0.218,
            yi,
            f"Δ={value:+.4f}   q={float(row.q_value):.4f} {row.significance}",
            ha="right",
            va="center",
            fontsize=6.5,
            fontweight="bold" if row.significance != "ns" else "normal",
        )
    ax.axvline(0, color="#4D4D4D", linewidth=0.8)
    ax.set_yticks(y, [SOURCE_LABELS[source] for source in table["source"]])
    ax.set_xlim(-0.035, 0.23)
    ax.set_xlabel("AUC decrease after group permutation")
    ax.set_title("Conditional source importance in M3")
    ax.text(-0.12, 1.06, "a", transform=ax.transAxes, fontsize=9, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "Error bars: 95% participant bootstrap CI. Two-sided paired permutation tests; "
        "BH-FDR across three source groups. **q<0.01; ns, q≥0.05.",
        ha="center",
        fontsize=6,
    )
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.24, top=0.84)
    _save_figure(fig, output_dir, "figure_3_permutation_importance")


def _figure_4(strict_dir: Path, output_dir: Path) -> None:
    table = pd.read_csv(strict_dir / "domain_leave_one_out_fdr.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 5.25), sharex=True)
    titles = {"domain_count": "Domain counts", "domain_presence": "Domain presence"}
    for panel_index, (ax, representation) in enumerate(zip(axes, ("domain_count", "domain_presence"))):
        subset = table.loc[table["representation"] == representation].sort_values(
            "delta_auc", ascending=True
        )
        y = np.arange(len(subset))
        values = subset["delta_auc"].to_numpy()
        lower = values - subset["ci_lower"].to_numpy()
        upper = subset["ci_upper"].to_numpy() - values
        colors = [PALETTE["negative"] if value < 0 else PALETTE["domain"] for value in values]
        for yi, value, lo, hi, color in zip(y, values, lower, upper, colors):
            ax.errorbar(
                value,
                yi,
                xerr=np.array([[lo], [hi]]),
                fmt="o",
                color=color,
                ecolor=color,
                capsize=2,
                markersize=4,
                linewidth=1,
            )
        ax.axvline(0, color="#4D4D4D", linewidth=0.8)
        ax.set_yticks(y, [DOMAIN_LABELS[name] for name in subset["removed_domain"]])
        ax.set_xlim(-0.04, 0.105)
        ax.set_xlabel("AUC decrease when removed")
        ax.set_title(titles[representation])
        ax.text(-0.22, 1.03, chr(ord("a") + panel_index), transform=ax.transAxes, fontsize=9, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "Error bars: 95% participant bootstrap CI. Two-sided paired permutation tests with "
        "BH-FDR across all 20 domain leave-one-out tests; all q≥0.228 (ns).",
        ha="center",
        fontsize=5.9,
    )
    fig.subplots_adjust(left=0.23, right=0.98, bottom=0.15, top=0.9, wspace=0.48)
    _save_figure(fig, output_dir, "figure_4_domain_leave_one_out")


def build_source_importance_figures(
    analysis_root: str | Path,
    output_dir: str | Path,
) -> dict[str, int | str]:
    _configure_matplotlib()
    root = Path(analysis_root).resolve()
    destination = Path(output_dir).resolve()
    strict_dir = root / "10_source_importance" / "07_strict_joint_models"
    required = (
        "model_metrics.csv",
        "model_auc_ci.csv",
        "incremental_comparisons_fdr.csv",
        "permutation_importance_fdr.csv",
        "domain_leave_one_out_fdr.csv",
    )
    missing = [name for name in required if not (strict_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing strict source data: {missing}")
    _figure_1(destination)
    _figure_2(strict_dir, destination)
    _figure_3(strict_dir, destination)
    _figure_4(strict_dir, destination)
    return {"status": "complete", "figure_count": 4, "output_dir": str(destination)}
