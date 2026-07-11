from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import sklearn
import tifffile


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reanalysis_v2.source_importance_strict import (  # noqa: E402
    BASE_SEED,
    run_repeated_cv,
    summarize_model_metrics,
)
from reanalysis_v2.submission_audit import (  # noqa: E402
    LOCKED_REPEAT,
    TOKEN_MATCH_DRAWS,
    audit_c5_traceability,
    build_core_paired_comparisons,
    build_structural_features,
    calibration_curve_data,
    load_locked_source_predictions,
    run_structural_locked_cv,
    run_token_matched_control,
    select_locked_split,
    summarize_locked_source_predictions,
    summarize_token_matched_draws,
    threshold_curve_data,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_figure_bundle(fig: plt.Figure, output_stem: Path) -> None:
    fig.savefig(output_stem.with_suffix(".png"), dpi=600)
    fig.savefig(output_stem.with_suffix(".svg"))
    fig.savefig(output_stem.with_suffix(".pdf"))
    tiff_path = output_stem.with_suffix(".tiff")
    fig.savefig(tiff_path, dpi=600)
    pixels = tifffile.imread(tiff_path)
    tifffile.imwrite(
        tiff_path,
        pixels,
        photometric="rgb",
        compression=tifffile.COMPRESSION.ADOBE_DEFLATE,
        resolution=(600, 600),
        resolutionunit="INCH",
        metadata=None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the locked-split submission audit and necessary reviewer controls."
    )
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=REPO_ROOT / "analysis_v2",
    )
    parser.add_argument(
        "--c5-source-root",
        type=Path,
        required=True,
        help="Restricted directory containing the four official142 reviewed C5 CSV files.",
    )
    parser.add_argument("--token-draws", type=int, default=TOKEN_MATCH_DRAWS)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    return parser.parse_args()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            encoding="utf-8",
        ).strip()
    except Exception:
        return "unavailable"


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _merge_metrics_ci(metrics: pd.DataFrame, auc_ci: pd.DataFrame) -> pd.DataFrame:
    return metrics.merge(auc_ci, on="model", how="left", validate="one_to_one")


def run_template_controls(
    analysis_root: Path,
    splits: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    template = pd.read_csv(
        analysis_root / "05_c4_controls" / "input_audit" / "participant_texts.csv",
        encoding="utf-8-sig",
    )
    labels = pd.read_csv(
        analysis_root / "01_inputs" / "c2_participant_speech.csv",
        encoding="utf-8-sig",
    )[["participant_id", "paper_label_phq8_ge10"]]
    frame = template.merge(labels, on="participant_id", validate="one_to_one")
    frame = frame.rename(
        columns={
            "paper_label_phq8_ge10": "label",
            "template_only": "template_only_text",
            "nontemplate_lengthmatched": "nontemplate_lengthmatched_text",
        }
    )
    specs: "OrderedDict[str, tuple[str, ...]]" = OrderedDict(
        {
            "T_template": ("template_only_text",),
            "T_nontemplate": ("nontemplate_text",),
            "T_nontemplate_lengthmatched": ("nontemplate_lengthmatched_text",),
        }
    )
    locked = select_locked_split(splits)
    _, participant = run_repeated_cv(
        frame,
        locked,
        model_specs=specs,
        base_seed=seed,
    )
    metrics, auc_ci = summarize_model_metrics(
        participant,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    return participant, metrics, auc_ci


def build_control_figure_data(
    structural_metrics: pd.DataFrame,
    structural_ci: pd.DataFrame,
    template_metrics: pd.DataFrame,
    template_ci: pd.DataFrame,
    token_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    labels = {
        "S_length": "Length only",
        "S_interaction": "Interaction structure",
        "S_protocol": "Protocol structure",
        "S_all": "All structural features",
        "T_template": "Template interviewer text",
        "T_nontemplate": "Non-template interviewer text",
        "T_nontemplate_lengthmatched": "Non-template text matched to template length",
        "participant_matched": "Participant text, source-token matched",
        "interviewer_matched": "Interviewer text, source-token matched",
    }
    for family, metrics, intervals in (
        ("structural", structural_metrics, structural_ci),
        ("template", template_metrics, template_ci),
    ):
        merged = metrics.merge(intervals, on="model", validate="one_to_one")
        for row in merged.itertuples(index=False):
            rows.append(
                {
                    "family": family,
                    "condition": row.model,
                    "label": labels[row.model],
                    "auc": float(row.roc_auc),
                    "ci_lower": float(row.ci_lower),
                    "ci_upper": float(row.ci_upper),
                }
            )
    for row in token_metrics.itertuples(index=False):
        rows.append(
            {
                "family": "token_matched",
                "condition": row.condition,
                "label": labels[row.condition],
                "auc": float(row.mean_auc),
                "ci_lower": float(row.participant_random_draw_ci_low),
                "ci_upper": float(row.participant_random_draw_ci_high),
            }
        )
    return pd.DataFrame(rows)


def render_calibration_threshold_figure(
    calibration: pd.DataFrame,
    thresholds: pd.DataFrame,
    output_stem: Path,
) -> None:
    selected = [
        "full_transcript",
        "participant_speech",
        "interviewer_speech",
        "participant_symptom_evidence",
    ]
    colors = {
        "full_transcript": "#3B6FB6",
        "participant_speech": "#2D8A57",
        "interviewer_speech": "#C04B3F",
        "participant_symptom_evidence": "#7B5AA6",
    }
    labels = {
        "full_transcript": "Full transcript",
        "participant_speech": "Participant speech",
        "interviewer_speech": "Interviewer speech",
        "participant_symptom_evidence": "Guided evidence representation",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), constrained_layout=True)
    calibration_ax = axes[0, 0]
    calibration_ax.plot([0, 1], [0, 1], linestyle="--", color="#666666", linewidth=1)
    for condition in selected:
        data = calibration.loc[calibration["condition"].eq(condition)]
        calibration_ax.plot(
            data["mean_predicted_probability"],
            data["observed_positive_fraction"],
            marker="o",
            linewidth=1.8,
            color=colors[condition],
            label=labels[condition],
        )
    calibration_ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean predicted probability",
        ylabel="Observed positive fraction",
        title="a  Locked-split calibration",
    )
    calibration_ax.legend(frameon=False, fontsize=8)

    for ax, condition, panel in zip(
        (axes[0, 1], axes[1, 0], axes[1, 1]),
        ("participant_speech", "interviewer_speech", "full_transcript"),
        ("b", "c", "d"),
    ):
        data = thresholds.loc[thresholds["condition"].eq(condition)]
        ax.plot(data["threshold"], data["sensitivity"], color="#C04B3F", label="Sensitivity")
        ax.plot(data["threshold"], data["specificity"], color="#3B6FB6", label="Specificity")
        ax.plot(data["threshold"], data["macro_f1"], color="#2D8A57", label="Macro-F1")
        ax.axvline(0.5, linestyle="--", color="#777777", linewidth=1)
        ax.set(
            xlim=(0.05, 0.95),
            ylim=(0, 1),
            xlabel="Probability threshold",
            ylabel="Metric",
            title=f"{panel}  {labels[condition]}",
        )
        ax.legend(frameon=False, fontsize=8)
    save_figure_bundle(fig, output_stem)
    plt.close(fig)


def render_source_signal_figure(output_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.2, 6.8), constrained_layout=True)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    boxes = {
        "state": (0.4, 5.1, 2.2, 1.0, "Participant state\nand context", "#DCE8F5"),
        "protocol": (4.9, 5.1, 2.2, 1.0, "Interview protocol\nand branching", "#E8E2F2"),
        "label": (9.4, 5.1, 2.2, 1.0, "PHQ-8 self-report\nlabel", "#F4E3DF"),
        "participant": (0.4, 2.9, 2.2, 1.0, "Participant speech\n(raw source)", "#DDEEDC"),
        "interviewer": (4.9, 2.9, 2.2, 1.0, "Interviewer speech\n(source + adaptation)", "#F2E4CE"),
        "evidence": (0.4, 0.7, 2.2, 1.0, "Clinically guided\nevidence representation", "#E7DDF1"),
        "structure": (4.9, 0.7, 2.2, 1.0, "Interaction and\nprotocol structure", "#DDE9EA"),
        "prediction": (9.4, 1.8, 2.2, 1.2, "PHQ-8 label\nprediction", "#E6E6E6"),
    }
    for x, y, width, height, label, color in boxes.values():
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=1.2,
            edgecolor="#4A4A4A",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=10)

    def arrow(start: tuple[float, float], end: tuple[float, float], *, dashed: bool = False) -> None:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": "->",
                "linewidth": 1.4,
                "color": "#555555",
                "linestyle": "--" if dashed else "-",
            },
        )

    arrow((1.5, 5.1), (1.5, 3.9))
    arrow((2.6, 5.45), (4.9, 3.55), dashed=True)
    arrow((6.0, 5.1), (6.0, 3.9))
    ax.annotate(
        "",
        xy=(9.4, 5.8),
        xytext=(2.6, 5.8),
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.4,
            "color": "#555555",
            "connectionstyle": "arc3,rad=-0.18",
        },
    )
    arrow((1.5, 2.9), (1.5, 1.7))
    arrow((6.0, 2.9), (6.0, 1.7))
    arrow((2.6, 3.35), (9.4, 2.55))
    arrow((7.1, 3.35), (9.4, 2.55))
    arrow((2.6, 1.2), (9.4, 2.2))
    arrow((7.1, 1.2), (9.4, 2.2))
    arrow((10.5, 5.1), (10.5, 3.0), dashed=True)

    ax.text(
        6.0,
        6.65,
        "Source-dependent predictive signals in a semi-structured interview",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )
    ax.text(
        6.0,
        0.15,
        "Dashed arrows indicate plausible coupling, not identified causal effects. Source performance alone does not establish bias, leakage, or clinical utility.",
        ha="center",
        va="center",
        fontsize=9,
        color="#555555",
    )
    save_figure_bundle(fig, output_stem)
    plt.close(fig)


def render_locked_source_performance_figure(
    source_metrics: pd.DataFrame,
    output_stem: Path,
) -> None:
    selected = [
        "participant_speech",
        "interviewer_speech",
        "full_transcript",
        "participant_symptom_evidence",
    ]
    labels = {
        "participant_speech": "Participant speech",
        "interviewer_speech": "Interviewer speech",
        "full_transcript": "Full transcript",
        "participant_symptom_evidence": "Guided evidence representation",
    }
    data = source_metrics.set_index("condition").loc[selected].reset_index()
    fig, ax = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    y = np.arange(len(data))
    colors = ["#2D8A57", "#C04B3F", "#3B6FB6", "#7B5AA6"]
    for index, row in data.iterrows():
        ax.errorbar(
            row["roc_auc"],
            index,
            xerr=[
                [row["roc_auc"] - row["auc_ci_low"]],
                [row["auc_ci_high"] - row["roc_auc"]],
            ],
            fmt="o",
            color=colors[index],
            capsize=4,
            markersize=7,
        )
        ax.text(row["auc_ci_high"] + 0.008, index, f"{row['roc_auc']:.3f}", va="center", fontsize=9)
    ax.axvline(0.5, linestyle="--", color="#777777", linewidth=1)
    ax.set_yticks(y, [labels[value] for value in selected])
    ax.invert_yaxis()
    ax.set(xlim=(0.48, 0.93), xlabel="ROC AUC (participant-bootstrap 95% CI)")
    ax.set_title("Locked 5-fold cross-fitted source performance (n=142)")
    save_figure_bundle(fig, output_stem)
    plt.close(fig)


def render_structure_control_figure(data: pd.DataFrame, output_stem: Path) -> None:
    ordered = data.reset_index(drop=True)
    y = np.arange(len(ordered))
    colors = {
        "structural": "#3B6FB6",
        "template": "#C04B3F",
        "token_matched": "#2D8A57",
    }
    fig, ax = plt.subplots(figsize=(9.6, 6.0), constrained_layout=True)
    for index, row in ordered.iterrows():
        ax.errorbar(
            row["auc"],
            index,
            xerr=[[row["auc"] - row["ci_lower"]], [row["ci_upper"] - row["auc"]]],
            fmt="o",
            color=colors[row["family"]],
            capsize=3,
        )
    ax.axvline(0.5, linestyle="--", color="#777777", linewidth=1)
    ax.set_yticks(y, ordered["label"])
    ax.invert_yaxis()
    ax.set(xlabel="ROC AUC (95% uncertainty interval)", xlim=(0.45, 0.9))
    ax.set_title("Locked-split structural and text-quantity controls")
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[family],
                   markeredgecolor=colors[family], label=label)
            for family, label in (
                ("structural", "Structural variables"),
                ("template", "Template controls"),
                ("token_matched", "Cross-source token matching"),
            )
        ],
        loc="upper right",
        frameon=False,
    )
    save_figure_bundle(fig, output_stem)
    plt.close(fig)


def c5_summary(
    spans: pd.DataFrame,
    manual_audit: pd.DataFrame,
    traceability: pd.DataFrame,
    condition_rows: pd.DataFrame,
    frozen_c5: pd.DataFrame,
) -> dict[str, object]:
    action_counts = manual_audit["action"].value_counts().to_dict()
    corrected = int(action_counts.get("added_corrected_participant_quote", 0))
    duplicates = int(action_counts.get("kept_excluded_duplicate", 0))
    invalid = int(action_counts.get("excluded_manual_fail", 0))
    def normalize_condition_text(series: pd.Series) -> pd.Series:
        normalized = series.fillna("").astype(str).str.split().str.join(" ")
        return normalized.replace({"NO_EXPLICIT_SYMPTOM_EVIDENCE": ""})

    normalized_source = condition_rows[["participant_id", "text"]].copy()
    normalized_source["normalized"] = normalize_condition_text(normalized_source["text"])
    normalized_frozen = frozen_c5[["participant_id", "text"]].copy()
    normalized_frozen["normalized"] = normalize_condition_text(normalized_frozen["text"])
    aligned = normalized_source[["participant_id", "normalized"]].merge(
        normalized_frozen[["participant_id", "normalized"]],
        on="participant_id",
        suffixes=("_source", "_frozen"),
        validate="one_to_one",
    )
    label_columns = [
        column
        for column in manual_audit.columns
        if "label" in column.lower() or "phq" in column.lower()
    ]
    reviewer_columns = [
        column
        for column in manual_audit.columns
        if "reviewer" in column.lower() or "rater" in column.lower()
    ]
    return {
        "candidate_span_n": int(len(spans) + duplicates + invalid),
        "final_span_n": int(len(spans)),
        "participant_n_with_evidence": int(spans["participant_id"].nunique()),
        "participant_n_total": int(frozen_c5["participant_id"].nunique()),
        "unchanged_source_matched_n": int(len(spans) - corrected),
        "manually_corrected_to_source_n": corrected,
        "excluded_duplicate_n": duplicates,
        "excluded_invalid_n": invalid,
        "manual_audit_row_n": int(len(manual_audit)),
        "final_source_match_n": int(traceability["traceable_to_participant_source"].sum()),
        "unique_location_n": int(traceability["location_status"].eq("unique_exact_match").sum()),
        "multiple_exact_location_n": int(
            traceability["location_status"].eq("multiple_exact_matches").sum()
        ),
        "not_found_after_review_n": int(traceability["location_status"].eq("not_found").sum()),
        "frozen_condition_rows_match_reviewed_source": bool(
            len(aligned) == len(frozen_c5)
            and aligned["normalized_source"].equals(aligned["normalized_frozen"])
        ),
        "manual_review_label_columns_present": label_columns,
        "manual_review_blinding_status": "not_demonstrated_labels_present_in_audit_table",
        "documented_reviewer_identity_columns": reviewer_columns,
        "interrater_reliability_status": "not_available",
    }


def write_c5_summary(summary: dict[str, object], path: Path) -> None:
    candidate = int(summary["candidate_span_n"])
    final = int(summary["final_span_n"])
    unchanged = int(summary["unchanged_source_matched_n"])
    corrected = int(summary["manually_corrected_to_source_n"])
    lines = [
        "# C5 evidence traceability audit",
        "",
        "C5 is a clinically guided, LLM-assisted derived representation. It is not a natural source equivalent to raw participant speech.",
        "",
        f"- Candidates entering review: {candidate}.",
        f"- Final retained spans: {final}.",
        f"- Retained without source correction: {unchanged} ({100 * unchanged / candidate:.1f}% of candidates).",
        f"- Corrected to exact participant-source wording: {corrected} ({100 * corrected / candidate:.1f}% of candidates).",
        f"- Excluded duplicates: {summary['excluded_duplicate_n']}; excluded invalid span: {summary['excluded_invalid_n']}.",
        f"- Final spans traceable to normalized participant source: {summary['final_source_match_n']}/{final}.",
        f"- Unique normalized character location: {summary['unique_location_n']}/{final}; repeated exact phrase: {summary['multiple_exact_location_n']}; not found: {summary['not_found_after_review_n']}.",
        f"- Participants with retained evidence: {summary['participant_n_with_evidence']}/{summary['participant_n_total']}.",
        f"- Frozen C5 participant texts match the reviewed condition rows: {summary['frozen_condition_rows_match_reviewed_source']}.",
        "",
        "## Blinding and review boundary",
        "",
        "The retained manual-review audit table contains PHQ-8 score/label columns. Outcome blinding therefore cannot be demonstrated and must not be claimed. Reviewer identity fields and a second independent C5 rating are not documented, so inter-rater reliability is unavailable. These are limitations of the existing C5 representation, not values to impute retrospectively.",
        "",
        "The public traceability CSV omits quote text. It contains participant ID, domain, polarity, source order, source-text hash, quote hash, normalized character offsets, match multiplicity, and review status.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_submission_summary(
    source_metrics: pd.DataFrame,
    structural_metrics: pd.DataFrame,
    token_metrics: pd.DataFrame,
    token_comparison: pd.DataFrame,
    comparisons: pd.DataFrame,
    c5: dict[str, object],
    path: Path,
) -> None:
    source_lookup = source_metrics.set_index("condition")
    structural_lookup = structural_metrics.set_index("model")
    token_lookup = token_metrics.set_index("condition")
    lines = [
        "# Submission audit: locked main analysis and required controls",
        "",
        "## Locked claim",
        "",
        "This study audits the sources of PHQ-8 label-predictive signals in semi-structured interviews and tests whether full-transcript performance can be attributed directly to participant language alone.",
        "",
        "## Main analysis contract",
        "",
        "- Main estimates use frozen repeat 1 of the prespecified participant-level 5-fold split; every participant contributes one cross-fitted prediction.",
        "- The existing 10 x 5 repeated cross-validation is retained only as a split-stability supplement.",
        "- TF-IDF fitting, scaling, calibration, and threshold selection remain inside training data. The 0.5 threshold is called a prespecified/default operating threshold, not a clinical cutoff.",
        "- Paired uncertainty resamples participants, not folds or repeated OOF rows.",
        "",
        "## Locked-split source results",
        "",
    ]
    for condition in (
        "participant_speech",
        "interviewer_speech",
        "full_transcript",
        "participant_symptom_evidence",
    ):
        row = source_lookup.loc[condition]
        lines.append(
            f"- {condition}: AUC {row.roc_auc:.3f} (95% CI {row.auc_ci_low:.3f}-{row.auc_ci_high:.3f}), Brier {row.raw_brier:.3f}; nested-threshold sensitivity/specificity {row.sensitivity_nested:.3f}/{row.specificity_nested:.3f}."
        )
    lines.extend(
        [
            "",
            "Single-source performance measures predictive sufficiency, not independent contribution or causal bias. Conditional increments are reported separately in the paired-comparison table.",
            "",
            "## Structural and text-quantity controls",
            "",
        ]
    )
    for model in ("S_length", "S_interaction", "S_protocol", "S_all"):
        row = structural_lookup.loc[model]
        lines.append(f"- {model}: AUC {row.roc_auc:.3f}.")
    for model in ("participant_matched", "interviewer_matched"):
        row = token_lookup.loc[model]
        lines.append(
            f"- {model}, mean across 50 deterministic contiguous-window draws: AUC {row.mean_auc:.3f} (participant + random-draw 95% CI {row.participant_random_draw_ci_low:.3f}-{row.participant_random_draw_ci_high:.3f})."
        )
    token_delta = token_comparison.iloc[0]
    lines.append(
        f"- Matched-text mean Delta AUC (interviewer minus participant): {token_delta.mean_delta_auc:.3f} (participant + random-draw 95% CI {token_delta.participant_random_draw_ci_low:.3f}-{token_delta.participant_random_draw_ci_high:.3f})."
    )
    lines.extend(
        [
            "",
            "These controls determine whether process structure and text quantity can account for source-level performance. They do not isolate a single generative mechanism because protocol branching, participant responses, and interviewer adaptation remain coupled.",
            "",
            "## Prespecified paired comparisons",
            "",
            "| Comparison | Scope | Delta AUC | 95% CI | q |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in comparisons.itertuples(index=False):
        lines.append(
            f"| {row.comparison} | {row.interpretation_scope} | {row.delta_auc:.3f} | {row.ci_lower:.3f} to {row.ci_upper:.3f} | {row.q_value:.3f} |"
        )
    lines.extend(
        [
            "",
            "## C5 boundary",
            "",
            f"All {c5['final_source_match_n']}/{c5['final_span_n']} retained C5 spans match participant source text after review, including {c5['manually_corrected_to_source_n']} corrected spans. However, outcome blinding cannot be demonstrated and inter-rater reliability is unavailable. C5 is therefore reported as a clinically guided derived representation, not as an unprocessed natural signal or evidence of pathological understanding.",
            "",
            "## E-DAIC boundary",
            "",
            "Module 14 remains a small descriptive supplement only. It is excluded from the title, abstract, core claims, and main figures; it does not establish generalizability or an error mechanism.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    analysis_root = args.analysis_root.resolve()
    c5_root = args.c5_source_root.resolve()
    output = analysis_root / "12_submission_audit"
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)

    split_path = analysis_root / "00_splits" / "repeated_5fold_splits_10x5.csv"
    splits = pd.read_csv(split_path, encoding="utf-8-sig")
    locked_splits = select_locked_split(splits)
    _write_csv(locked_splits, output / "locked_repeat1_split.csv")

    source_predictions = load_locked_source_predictions(analysis_root)
    source_metrics = summarize_locked_source_predictions(
        source_predictions,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    _write_csv(source_predictions, output / "locked_source_predictions.csv")
    _write_csv(source_metrics, output / "locked_source_metrics.csv")
    render_source_signal_figure(figures / "figure_1_source_signal_paths")
    render_locked_source_performance_figure(
        source_metrics,
        figures / "figure_2_locked_source_performance",
    )

    strict_predictions = pd.read_csv(
        analysis_root
        / "10_source_importance"
        / "07_strict_joint_models"
        / "model_fold_oof_predictions.csv",
        encoding="utf-8-sig",
    )
    strict_locked = strict_predictions.loc[strict_predictions["repeat"].eq(LOCKED_REPEAT)].copy()
    if strict_locked["participant_id"].duplicated().any():
        raise ValueError("Strict locked joint predictions are not one row per participant")
    comparisons = build_core_paired_comparisons(
        source_predictions,
        strict_locked,
        n_bootstrap=args.bootstrap,
        n_permutations=args.permutations,
        seed=args.seed,
    )
    _write_csv(strict_locked, output / "locked_joint_predictions.csv")
    _write_csv(comparisons, output / "locked_core_paired_comparisons.csv")

    turns_path = analysis_root / "01_inputs" / "c4_turns_official142_frozen.csv"
    turns = pd.read_csv(turns_path, encoding="utf-8-sig")
    structural_features = build_structural_features(turns)
    structural_fold, structural_oof, structural_metrics, structural_ci = run_structural_locked_cv(
        structural_features,
        splits,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    _write_csv(structural_features, output / "structural_baseline_features.csv")
    _write_csv(structural_fold, output / "structural_baseline_fold_oof.csv")
    _write_csv(structural_oof, output / "structural_baseline_participant_oof.csv")
    _write_csv(structural_metrics, output / "structural_baseline_metrics.csv")
    _write_csv(structural_ci, output / "structural_baseline_auc_ci.csv")

    c2_path = analysis_root / "01_inputs" / "c2_participant_speech.csv"
    c3_path = analysis_root / "01_inputs" / "c3_interviewer_speech.csv"
    participant_text = pd.read_csv(c2_path, encoding="utf-8-sig")
    interviewer_text = pd.read_csv(c3_path, encoding="utf-8-sig")
    (
        token_predictions,
        token_draw_metrics,
        token_averaged,
        token_ensemble_metrics,
        token_ensemble_comparison,
        token_length_audit,
    ) = run_token_matched_control(
        participant_text,
        interviewer_text,
        splits,
        n_draws=args.token_draws,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    _write_csv(token_predictions, output / "token_matched_draw_predictions.csv")
    _write_csv(token_draw_metrics, output / "token_matched_draw_metrics.csv")
    _write_csv(token_averaged, output / "token_matched_participant_oof.csv")
    token_metrics, token_comparison = summarize_token_matched_draws(
        token_predictions,
        token_draw_metrics,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    _write_csv(token_metrics, output / "token_matched_metrics.csv")
    _write_csv(token_comparison, output / "token_matched_paired_comparison.csv")
    _write_csv(token_ensemble_metrics, output / "token_matched_ensemble_metrics_audit_only.csv")
    _write_csv(
        token_ensemble_comparison,
        output / "token_matched_ensemble_comparison_audit_only.csv",
    )
    _write_csv(token_length_audit, output / "token_matching_length_audit.csv")

    template_oof, template_metrics, template_ci = run_template_controls(
        analysis_root,
        splits,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    _write_csv(template_oof, output / "locked_template_control_oof.csv")
    _write_csv(template_metrics, output / "locked_template_control_metrics.csv")
    _write_csv(template_ci, output / "locked_template_control_auc_ci.csv")

    calibration = calibration_curve_data(source_predictions)
    thresholds = threshold_curve_data(source_predictions)
    _write_csv(calibration, output / "locked_calibration_curve_data.csv")
    _write_csv(thresholds, output / "locked_threshold_curve_data.csv")
    render_calibration_threshold_figure(
        calibration,
        thresholds,
        figures / "figure_3_calibration_threshold_locked",
    )

    control_figure = build_control_figure_data(
        structural_metrics,
        structural_ci,
        template_metrics,
        template_ci,
        token_metrics,
    )
    _write_csv(control_figure, output / "structure_control_figure_data.csv")
    render_structure_control_figure(
        control_figure,
        figures / "figure_4_structure_controls_locked",
    )

    spans_path = c5_root / "c5_quotes_only_v3_gpt55_reviewed_spans_official142.csv"
    manual_path = c5_root / "c5_quotes_only_v3_gpt55_reviewed_manual_review_application_audit_official142.csv"
    condition_path = c5_root / "c5_quotes_only_v3_gpt55_reviewed_condition_rows_official142.csv"
    spans = pd.read_csv(spans_path, encoding="utf-8-sig")
    manual = pd.read_csv(manual_path, encoding="utf-8-sig")
    condition_rows = pd.read_csv(condition_path, encoding="utf-8-sig")
    frozen_c5 = pd.read_csv(
        analysis_root / "01_inputs" / "c5_reviewed_symptom_evidence.csv",
        encoding="utf-8-sig",
    )
    traceability = audit_c5_traceability(spans, participant_text)
    _write_csv(traceability, output / "c5_traceability_audit_no_quotes.csv")
    c5 = c5_summary(spans, manual, traceability, condition_rows, frozen_c5)
    (output / "c5_traceability_summary.json").write_text(
        json.dumps(c5, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_c5_summary(c5, output / "c5_traceability_summary.md")

    structural_public = _merge_metrics_ci(structural_metrics, structural_ci)
    write_submission_summary(
        source_metrics,
        structural_public,
        token_metrics,
        token_comparison,
        comparisons,
        c5,
        output / "submission_audit_summary.md",
    )

    inputs = [
        split_path,
        turns_path,
        c2_path,
        c3_path,
        spans_path,
        manual_path,
        condition_path,
        analysis_root / "01_inputs" / "c5_reviewed_symptom_evidence.csv",
        analysis_root / "10_source_importance" / "07_strict_joint_models" / "model_fold_oof_predictions.csv",
    ]
    manifest = {
        "status": "complete",
        "analysis": "submission audit with locked main split and reviewer-required controls",
        "started_at_utc": started.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "main_cv": {
            "locked_repeat": LOCKED_REPEAT,
            "folds": 5,
            "participant_predictions_per_condition": 142,
            "repeated_10x5_role": "supplementary_split_stability_only",
        },
        "token_matching": {
            "draws": args.token_draws,
            "method": "per-participant contiguous window to min(participant_tokens, interviewer_tokens)",
            "label_used_for_sampling": False,
        },
        "inference": {
            "bootstrap_resamples": args.bootstrap,
            "paired_permutations": args.permutations,
            "resampling_unit": "participant",
            "core_comparison_count": 5,
        },
        "c5": {
            **c5,
            "public_quote_text_included": False,
        },
        "interpretation_guardrails": {
            "source_performance": "predictive_sufficiency_not_independent_contribution",
            "interviewer_signal": "source_dependent_signal_not_causal_bias_or_leakage",
            "c5": "clinically_guided_derived_representation",
            "phq8": "self_report_label_not_clinical_diagnosis",
            "edaic": "descriptive_supplement_not_external_validation",
        },
        "seed": args.seed,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
            "tifffile": tifffile.__version__,
        },
        "inputs": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in inputs
        ],
    }
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    public_files = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "output_manifest_sha256.csv"
    )
    inventory = pd.DataFrame(
        [
            {
                "file": path.relative_to(output).as_posix(),
                "sha256": sha256(path),
            }
            for path in public_files
        ]
    )
    _write_csv(inventory, output / "output_manifest_sha256.csv")
    print(f"submission_audit_complete output={output}")
    print(source_metrics[["condition", "roc_auc", "auc_ci_low", "auc_ci_high", "raw_brier"]].to_string(index=False))
    print(structural_public[["model", "roc_auc", "ci_lower", "ci_upper"]].to_string(index=False))
    print(
        token_metrics[
            [
                "condition",
                "mean_auc",
                "participant_random_draw_ci_low",
                "participant_random_draw_ci_high",
            ]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
