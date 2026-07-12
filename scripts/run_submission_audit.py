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
from reanalysis_v2.post_audit_sensitivity import (  # noqa: E402
    build_cross_fitted_null_predictions,
    build_retained_alignment_frames,
    run_retained_alignment_sensitivity,
    run_symmetric_half_min_control,
    run_symmetric_position_control,
    summarize_brier_skill,
    summarize_token_budget_audit,
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
    symmetric_token_metrics: pd.DataFrame,
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
        "symmetric_participant_matched": "Participant text, symmetric half-min budget",
        "symmetric_interviewer_matched": "Interviewer text, symmetric half-min budget",
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
    for row in symmetric_token_metrics.itertuples(index=False):
        condition = f"symmetric_{row.condition}"
        rows.append(
            {
                "family": "symmetric_half_min",
                "condition": condition,
                "label": labels[condition],
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
        "symmetric_half_min": "#7A5AA6",
    }
    fig, ax = plt.subplots(figsize=(9.6, 6.8), constrained_layout=True)
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
                ("symmetric_half_min", "Symmetric half-min budget"),
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


def write_model_configuration(path: Path) -> None:
    text = """# Model configuration for the analysis-locked audit

| Component | Frozen configuration |
|---|---|
| Split | Repeat 1 of the existing 10 x 5 participant-level split file; five outer folds; one cross-fitted test prediction per participant |
| Fold fitting | Vocabulary, TF-IDF weights, numeric scaling, calibration, and threshold selection fit without outer-test participants |
| TF-IDF | lowercase=True; strip_accents=unicode; word ngrams=(1, 2); min_df=2; max_features=50,000; sublinear_tf=True |
| Classifier | LogisticRegression; C=1.0; class_weight=balanced; solver=liblinear; max_iter=2,000 |
| Numeric features | StandardScaler fit on each outer-training fold |
| Random seed | Base seed 20260705; fold/model offsets recorded in source code and run manifest |
| Calibration | Platt and isotonic calibrators fit from inner cross-fitted training predictions, then applied to the outer test fold |
| Default operation | Probability 0.50 is a default operating point, not a clinical or preregistered cutoff |
| Alternative thresholds | Selected only from inner training predictions for max Macro-F1, max Youden J, and sensitivity subject to specificity >=0.80 |
| Uncertainty | 5,000 participant bootstrap replicates; paired tests use 10,000 within-participant swaps |
| Multiplicity | Original five analysis-locked comparisons retain their own BH family; post-audit positional and C5 sensitivity families are adjusted separately |

The analysis-locked split was fixed before the post-audit sensitivity run to prevent further split selection. This is not formal preregistration.
"""
    path.write_text(text, encoding="utf-8")


def write_submission_summary(
    source_metrics: pd.DataFrame,
    brier_metrics: pd.DataFrame,
    brier_fold_metrics: pd.DataFrame,
    structural_metrics: pd.DataFrame,
    token_metrics: pd.DataFrame,
    token_comparison: pd.DataFrame,
    token_asymmetry: pd.DataFrame,
    symmetric_metrics: pd.DataFrame,
    symmetric_comparison: pd.DataFrame,
    symmetric_budget: pd.DataFrame,
    position_metrics: pd.DataFrame,
    position_comparisons: pd.DataFrame,
    comparisons: pd.DataFrame,
    c5: dict[str, object],
    c5_alignment_audit: pd.DataFrame,
    c5_alignment_metrics: pd.DataFrame,
    c5_alignment_differences: pd.DataFrame,
    c5_probability_change: pd.DataFrame,
    path: Path,
) -> None:
    source_lookup = source_metrics.set_index("condition")
    structural_lookup = structural_metrics.set_index("model")
    token_lookup = token_metrics.set_index("condition")
    symmetric_lookup = symmetric_metrics.set_index("condition")
    brier_lookup = brier_metrics.set_index(["condition", "probability_variant"])
    direct_rate = int(c5["unchanged_source_matched_n"]) / int(c5["candidate_span_n"])
    corrected_participants = int(c5_alignment_audit["corrected_span_count"].gt(0).sum())
    changed_participants = int(c5_alignment_audit["representation_changed"].sum())
    token_delta = int(c5_alignment_audit["token_count_delta"].sum())
    c5_metric_lookup = c5_alignment_metrics.set_index("model")
    probability_change = c5_probability_change.iloc[0]
    lines = [
        "# Submission audit: analysis-locked main analysis and post-audit sensitivity checks",
        "",
        "## Timeline and claim boundary",
        "",
        "- Main estimates use frozen repeat 1 of the existing participant-level five-fold split; every participant contributes one cross-fitted prediction.",
        "- Repeat 1 was locked before the post-audit sensitivity run to prevent further split selection. This is not formal preregistration and is not described as prospectively prespecified.",
        "- The existing 10 x 5 repeated cross-validation remains a split-stability supplement.",
        "- Single-source performance measures predictive sufficiency, not independent contribution, causal bias, or leakage.",
        "",
        "## Analysis-locked source results",
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
            f"- {condition}: AUC {row.roc_auc:.3f} (95% CI {row.auc_ci_low:.3f}-{row.auc_ci_high:.3f}), raw Brier {row.raw_brier:.3f}; nested-threshold sensitivity/specificity {row.sensitivity_nested:.3f}/{row.specificity_nested:.3f}."
        )
    lines.extend(
        [
            "",
            "## Brier no-skill baselines",
            "",
            "The formal no-skill comparator assigns each outer-test participant the positive prevalence from that outer-training fold. The fixed 43/142 cohort prevalence is descriptive only. Raw, Platt, and isotonic values use the same outer cross-fitted predictions; no calibrator is refit on pooled OOF predictions.",
            "",
            "| Condition | Variant | Brier | Train-fold null | BSS | 95% BSS CI | Descriptive cohort-null BSS |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for condition in (
        "participant_speech",
        "interviewer_speech",
        "full_transcript",
        "participant_symptom_evidence",
    ):
        for variant in ("raw", "platt", "isotonic"):
            row = brier_lookup.loc[(condition, variant)]
            lines.append(
                f"| {condition} | {variant} | {row.model_brier:.3f} | {row.trainfold_null_brier:.3f} | {row.bss_trainfold:.3f} | {row.bss_trainfold_ci_low:.3f} to {row.bss_trainfold_ci_high:.3f} | {row.bss_cohort_descriptive:.3f} |"
            )
    isotonic_fold = brier_fold_metrics.loc[
        brier_fold_metrics["probability_variant"].eq("isotonic"), "model_brier"
    ]
    lines.extend(
        [
            "",
            f"Isotonic fold-level Brier values are retained as a stability audit (range {isotonic_fold.min():.3f}-{isotonic_fold.max():.3f}); no calibration method is declared best from one aggregate estimate.",
            "",
            "## Structural and text-quantity controls",
            "",
        ]
    )
    for model in ("S_length", "S_interaction", "S_protocol", "S_all"):
        row = structural_lookup.loc[model]
        lines.append(
            f"- {model}: AUC {row.roc_auc:.3f} (95% CI {row.ci_lower:.3f}-{row.ci_upper:.3f})."
        )
    lines.append(
        "Interaction-only uncertainty includes 0.5; protocol-only and all-structure estimates are reported separately and are not collapsed into a claim that all structure families clearly predict above chance."
    )
    for model in ("participant_matched", "interviewer_matched"):
        row = token_lookup.loc[model]
        lines.append(
            f"- Longer-source-to-shorter-source-length {model}: mean AUC {row.mean_auc:.3f} (participant + random-window 95% CI {row.participant_random_draw_ci_low:.3f}-{row.participant_random_draw_ci_high:.3f})."
        )
    asym_p = token_asymmetry.set_index("source").loc["participant"]
    asym_i = token_asymmetry.set_index("source").loc["interviewer"]
    lines.append(
        f"- Existing control asymmetry: participant/interviewer truncated n={int(asym_p.truncated_n)}/{int(asym_i.truncated_n)}; mean retained fraction among non-empty texts={asym_p.mean_retained_fraction_nonempty:.3f}/{asym_i.mean_retained_fraction_nonempty:.3f}."
    )
    for model in ("participant_matched", "interviewer_matched"):
        row = symmetric_lookup.loc[model]
        lines.append(
            f"- Symmetric half-min {model}: mean AUC {row.mean_auc:.3f} (participant + random-window 95% CI {row.participant_random_draw_ci_low:.3f}-{row.participant_random_draw_ci_high:.3f})."
        )
    symmetric_delta = symmetric_comparison.iloc[0]
    budget_target = symmetric_budget.set_index("source").loc["shared_target"]
    lines.append(
        f"- Symmetric half-min interviewer-minus-participant mean Delta AUC {symmetric_delta.mean_delta_auc:.3f} (joint 95% CI {symmetric_delta.participant_random_draw_ci_low:.3f}-{symmetric_delta.participant_random_draw_ci_high:.3f}); median shared target={budget_target.target_token_median:.0f}, zero targets={int(budget_target.zero_target_n)}, targets below 10={int(budget_target.target_below_10_n)}."
    )
    lines.extend(
        [
            "",
            "Deterministic early, middle, and late windows use the identical half-min budget for both sources:",
            "",
            "| Position | Source | AUC | PR-AUC | Brier |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in position_metrics.itertuples(index=False):
        lines.append(
            f"| {row.position} | {row.model} | {row.roc_auc:.3f} | {row.pr_auc:.3f} | {row.brier:.3f} |"
        )
    position_lines = []
    for row in position_comparisons.itertuples(index=False):
        position_lines.append(
            f"{row.position} {row.delta_auc:.4f} (95% CI {row.ci_lower:.4f}-{row.ci_upper:.4f}; raw p={row.p_value:.4f}, BH q={row.q_value:.4f})"
        )
    lines.extend(
        [
            "",
            "Position-paired interviewer-minus-participant Delta AUCs were "
            + "; ".join(position_lines)
            + ".",
            "",
            "The two token controls answer different questions. The original longer-to-shorter control was strongly asymmetric. Under a common half-min information budget, the source contrast attenuated and its joint interval included zero; its magnitude and direction also varied by window position. The evidence therefore does not rule out text-budget or position effects and does not identify semantics or a causal mechanism.",
            "",
            "## Original analysis-locked paired comparisons",
            "",
            "Bootstrap confidence intervals and paired permutation p-values are different calculations. The five comparisons retain their original BH family.",
            "",
            "| Comparison | Scope | Delta AUC | Participant-bootstrap 95% CI | Raw p | BH q |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons.itertuples(index=False):
        lines.append(
            f"| {row.comparison} | {row.interpretation_scope} | {row.delta_auc:.4f} | {row.ci_lower:.4f} to {row.ci_upper:.4f} | {row.p_value:.4f} | {row.q_value:.4f} |"
        )
    lines.extend(
        [
            "",
            "## C5 retained-span source-alignment sensitivity",
            "",
            f"Among {c5['candidate_span_n']} candidates, {c5['unchanged_source_matched_n']} were direct source matches ({100 * direct_rate:.1f}%), {c5['manually_corrected_to_source_n']} were source-alignment revisions, and {int(c5['excluded_duplicate_n']) + int(c5['excluded_invalid_n'])} were excluded. All {c5['final_source_match_n']}/{c5['final_span_n']} retained spans are traceable after review. The final traceability rate is not extraction accuracy.",
            f"The retained-span sensitivity holds span inclusion, source order, domain, and polarity fixed and changes quote text only. The {c5['manually_corrected_to_source_n']} revisions involve {corrected_participants} participants; {changed_participants} participant representations change and total whitespace-token count changes by {token_delta:+d}. It cannot reconstruct the six excluded candidates or a complete pre-review representation.",
            "",
            "| Representation | AUC | PR-AUC | Brier |",
            "|---|---:|---:|---:|",
        ]
    )
    for model in ("original_retained", "aligned_retained"):
        row = c5_metric_lookup.loc[model]
        lines.append(
            f"| {model} | {row.roc_auc:.3f} | {row.pr_auc:.3f} | {row.brier:.3f} |"
        )
    lines.extend(
        [
            "",
            "| Metric | Aligned minus original | Participant-bootstrap 95% CI | Raw p | BH q |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in c5_alignment_differences.itertuples(index=False):
        lines.append(
            f"| {row.metric} | {row.aligned_minus_original:.4f} | {row.ci_lower:.4f} to {row.ci_upper:.4f} | {row.p_value:.4f} | {row.q_value:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Absolute participant probability change: median {probability_change.absolute_change_median:.4f}, 95th percentile {probability_change.absolute_change_q95:.4f}, maximum {probability_change.absolute_change_max:.4f}; n>=0.01: {int(probability_change.absolute_change_ge_0_01_n)}, n>=0.05: {int(probability_change.absolute_change_ge_0_05_n)}.",
            "",
            "The available review table contains PHQ-8 fields and lacks rater identities or independent double review. Outcome blinding and inter-rater reliability cannot be claimed.",
            "",
            "## E-DAIC boundary",
            "",
            "Module 14 remains a small descriptive supplement only. It does not establish generalizability or an independent error mechanism.",
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
    null_predictions = build_cross_fitted_null_predictions(source_predictions)
    brier_metrics, brier_fold_metrics = summarize_brier_skill(
        source_predictions,
        null_predictions,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    _write_csv(null_predictions, output / "locked_null_brier_predictions.csv")
    _write_csv(brier_metrics, output / "locked_brier_skill_metrics.csv")
    _write_csv(brier_fold_metrics, output / "locked_brier_fold_metrics.csv")
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
    token_asymmetry = summarize_token_budget_audit(
        token_length_audit,
        control="longer_source_to_shorter_source_length",
    )
    _write_csv(token_asymmetry, output / "token_matching_asymmetry_summary.csv")

    (
        symmetric_predictions,
        symmetric_draw_metrics,
        symmetric_metrics,
        symmetric_comparison,
        symmetric_length_audit,
    ) = run_symmetric_half_min_control(
        participant_text,
        interviewer_text,
        splits,
        n_draws=args.token_draws,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    symmetric_budget = summarize_token_budget_audit(
        symmetric_length_audit,
        control="symmetric_half_min",
    )
    _write_csv(symmetric_length_audit, output / "symmetric_half_min_length_audit.csv")
    _write_csv(symmetric_budget, output / "symmetric_half_min_audit_summary.csv")
    _write_csv(symmetric_predictions, output / "symmetric_half_min_draw_predictions.csv")
    _write_csv(symmetric_draw_metrics, output / "symmetric_half_min_draw_metrics.csv")
    _write_csv(symmetric_metrics, output / "symmetric_half_min_metrics.csv")
    _write_csv(symmetric_comparison, output / "symmetric_half_min_paired_comparison.csv")

    (
        position_predictions,
        position_metrics,
        position_comparisons,
        position_audit,
    ) = run_symmetric_position_control(
        participant_text,
        interviewer_text,
        splits,
        n_bootstrap=args.bootstrap,
        n_permutations=args.permutations,
        seed=args.seed,
    )
    if not position_audit.equals(symmetric_length_audit):
        raise AssertionError("Random and positional symmetric token budgets differ")
    _write_csv(position_predictions, output / "symmetric_position_predictions.csv")
    _write_csv(position_metrics, output / "symmetric_position_metrics.csv")
    _write_csv(position_comparisons, output / "symmetric_position_paired_comparisons.csv")

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
        symmetric_metrics,
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

    original_retained, aligned_retained, c5_alignment_audit = build_retained_alignment_frames(
        spans,
        frozen_c5,
    )
    (
        c5_alignment_predictions,
        c5_alignment_metrics,
        c5_alignment_differences,
        c5_probability_change,
    ) = run_retained_alignment_sensitivity(
        original_retained,
        aligned_retained,
        splits,
        n_bootstrap=args.bootstrap,
        n_permutations=args.permutations,
        seed=args.seed,
    )
    _write_csv(c5_alignment_audit, output / "c5_retained_alignment_audit.csv")
    _write_csv(c5_alignment_predictions, output / "c5_retained_alignment_predictions.csv")
    _write_csv(c5_alignment_metrics, output / "c5_retained_alignment_metrics.csv")
    _write_csv(
        c5_alignment_differences,
        output / "c5_retained_alignment_paired_differences.csv",
    )
    _write_csv(
        c5_probability_change,
        output / "c5_retained_alignment_probability_change_summary.csv",
    )

    structural_public = _merge_metrics_ci(structural_metrics, structural_ci)
    write_model_configuration(output / "model_configuration.md")
    write_submission_summary(
        source_metrics,
        brier_metrics,
        brier_fold_metrics,
        structural_public,
        token_metrics,
        token_comparison,
        token_asymmetry,
        symmetric_metrics,
        symmetric_comparison,
        symmetric_budget,
        position_metrics,
        position_comparisons,
        comparisons,
        c5,
        c5_alignment_audit,
        c5_alignment_metrics,
        c5_alignment_differences,
        c5_probability_change,
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
        output / "post_submission_audit_sensitivity_plan.md",
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
            "existing_method": "longer source contiguous window to min(participant_tokens, interviewer_tokens)",
            "symmetric_method": "both sources contiguous windows to floor(0.5 * min(participant_tokens, interviewer_tokens))",
            "symmetric_positions": ["early", "middle", "late"],
            "zero_target_policy": "retain participant with both source texts empty",
            "label_used_for_sampling": False,
        },
        "inference": {
            "bootstrap_resamples": args.bootstrap,
            "paired_permutations": args.permutations,
            "resampling_unit": "participant",
            "core_comparison_count": 5,
            "post_audit_sensitivity_status": "not_preregistered_separate_families",
        },
        "c5": {
            **c5,
            "public_quote_text_included": False,
            "alignment_sensitivity_scope": "same_1137_retained_spans_quote_text_only",
        },
        "interpretation_guardrails": {
            "source_performance": "predictive_sufficiency_not_independent_contribution",
            "interviewer_signal": "source_dependent_signal_not_causal_bias_or_leakage",
            "c5": "clinically_guided_derived_representation",
            "phq8": "self_report_label_not_clinical_diagnosis",
            "edaic": "descriptive_supplement_not_external_validation",
            "ci": "code_tests_and_derived_integrity_not_full_restricted_data_reproduction",
        },
        "post_audit_plan": {
            "file": "post_submission_audit_sensitivity_plan.md",
            "sha256": sha256(output / "post_submission_audit_sensitivity_plan.md"),
            "status": "frozen_before_sensitivity_results",
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
