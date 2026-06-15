from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BASE_DIR = Path("processed_research")
BASELINE_DIR = BASE_DIR / "baseline_results"
DEFAULT_OUTPUT_DIR = BASE_DIR / "paper_outputs"
CONDITIONS_CSV = (
    BASE_DIR / "conditions_long__RECOMMENDED_article-aligned_analysis-ready_C1-C4_no-C5.csv"
)
PARTICIPANT_INDEX_CSV = BASE_DIR / "participant_index.csv"
CV_SUMMARY_CSV = BASELINE_DIR / "baseline_cv_summary__combined_latest.csv"
TRAIN_DEV_METRICS_CSV = BASELINE_DIR / "baseline_metrics__train_dev_combined_latest.csv"
BIAS_METRICS_CSV = BASELINE_DIR / "bias_metrics__f1_macro_mcc_latest.csv"
THRESHOLD_CV_SUMMARY_CSV = (
    BASELINE_DIR / "threshold_cv_summary__threshold_mcc_tfidf_lr_svm_5fold_20260615.csv"
)
MCNEMAR_TESTS_CSV = BASELINE_DIR / "mcnemar_tests__mcnemar_train_dev_20260615.csv"

CONDITION_ORDER = [
    "full_dialogue",
    "participant_only",
    "interviewer_only",
    "interviewer_cleaned",
]
CONDITION_LABELS = {
    "full_dialogue": "C1 Full dialogue",
    "participant_only": "C2 Participant-only",
    "interviewer_only": "C3 Interviewer-only",
    "interviewer_cleaned": "C4 Interviewer-cleaned",
}
MODEL_ORDER = [
    "TF-IDF + LR balanced",
    "TF-IDF + Linear SVM balanced",
    "all-mpnet-base-v2 + LR balanced (CUDA)",
]
MODEL_LABELS = {
    "TF-IDF + LR balanced": "TF-IDF + LR",
    "TF-IDF + Linear SVM balanced": "TF-IDF + SVM",
    "all-mpnet-base-v2 + LR balanced (CUDA)": "MPNet + LR",
}
THRESHOLD_MODEL_ORDER = ["TF-IDF + LR", "TF-IDF + SVM"]
SENSITIVITY_MODEL_LABELS = {
    "TF-IDF + LR balanced": "TF-IDF + LR",
    "TF-IDF + Linear SVM balanced": "TF-IDF + SVM",
    "TF-IDF + LR": "TF-IDF + LR",
    "TF-IDF + SVM": "TF-IDF + SVM",
}
COMPARISONS_FOR_TABLES = [
    ("full_dialogue", "participant_only", "Full vs Participant"),
    ("participant_only", "interviewer_only", "Participant vs Interviewer"),
    ("interviewer_only", "interviewer_cleaned", "Interviewer vs Cleaned"),
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def fmt_number(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def fmt_p_value(value: float | int | None, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return ""
    threshold = 10 ** (-digits)
    if float(value) < threshold:
        return f"<{threshold:.{digits}f}"
    return f"{float(value):.{digits}f}"


def fmt_mean_sd(values: Iterable[float | int], digits: int = 2) -> str:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if series.empty:
        return ""
    return f"{series.mean():.{digits}f} +/- {series.std(ddof=1):.{digits}f}"


def fmt_median_iqr(values: Iterable[float | int], digits: int = 2) -> str:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if series.empty:
        return ""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    return f"{series.median():.{digits}f} [{q1:.{digits}f}, {q3:.{digits}f}]"


def fmt_count_pct(count: int, denom: int, digits: int = 1) -> str:
    pct = 100 * count / denom if denom else math.nan
    return f"{count} ({pct:.{digits}f}%)"


def display_model_label(model_short: str) -> str:
    return MODEL_LABELS.get(model_short, model_short)


def sensitivity_model_label(model_short: str) -> str:
    return SENSITIVITY_MODEL_LABELS.get(model_short, model_short)


def safe_mannwhitney_p(group0: pd.Series, group1: pd.Series) -> str:
    try:
        from scipy.stats import mannwhitneyu

        g0 = pd.to_numeric(group0, errors="coerce").dropna()
        g1 = pd.to_numeric(group1, errors="coerce").dropna()
        if g0.empty or g1.empty:
            return ""
        return fmt_p_value(mannwhitneyu(g0, g1, alternative="two-sided").pvalue, 4)
    except Exception:
        return ""


def safe_chi2_p(labels: pd.Series, categories: pd.Series) -> str:
    try:
        from scipy.stats import chi2_contingency

        table = pd.crosstab(labels, categories)
        if table.empty or table.shape[0] < 2 or table.shape[1] < 2:
            return ""
        return fmt_p_value(chi2_contingency(table).pvalue, 4)
    except Exception:
        return ""


def table1_sample_characteristics(participants: pd.DataFrame) -> pd.DataFrame:
    df = participants.copy()
    df["gender"] = df["gender"].fillna("unknown").astype(str).str.strip().str.lower()
    df["label_group"] = df["label"].map({0: "PHQ-8 < 10", 1: "PHQ-8 >= 10"})
    group0 = df[df["label"] == 0]
    group1 = df[df["label"] == 1]

    def cells_for_numeric(column: str, digits: int = 2) -> dict[str, str]:
        return {
            "Overall": fmt_mean_sd(df[column], digits),
            "PHQ-8 < 10": fmt_mean_sd(group0[column], digits),
            "PHQ-8 >= 10": fmt_mean_sd(group1[column], digits),
            "P value": safe_mannwhitney_p(group0[column], group1[column]),
        }

    def row(characteristic: str, values: dict[str, str]) -> dict[str, str]:
        return {"Characteristic": characteristic, **values}

    rows: list[dict[str, str]] = []
    rows.append(
        row(
            "Participants, n",
            {
                "Overall": str(len(df)),
                "PHQ-8 < 10": str(len(group0)),
                "PHQ-8 >= 10": str(len(group1)),
                "P value": "",
            },
        )
    )
    for split in ["train", "dev"]:
        rows.append(
            row(
                f"Split: {split}, n (%)",
                {
                    "Overall": fmt_count_pct(int((df["split"] == split).sum()), len(df)),
                    "PHQ-8 < 10": fmt_count_pct(
                        int((group0["split"] == split).sum()), len(group0)
                    ),
                    "PHQ-8 >= 10": fmt_count_pct(
                        int((group1["split"] == split).sum()), len(group1)
                    ),
                    "P value": safe_chi2_p(df["label"], df["split"])
                    if split == "train"
                    else "",
                },
            )
        )
    rows.append(row("Age, mean +/- SD", cells_for_numeric("age", 2)))
    gender_values = sorted(df["gender"].dropna().unique())
    for gender in gender_values:
        rows.append(
            row(
                f"Gender: {gender}, n (%)",
                {
                    "Overall": fmt_count_pct(int((df["gender"] == gender).sum()), len(df)),
                    "PHQ-8 < 10": fmt_count_pct(
                        int((group0["gender"] == gender).sum()), len(group0)
                    ),
                    "PHQ-8 >= 10": fmt_count_pct(
                        int((group1["gender"] == gender).sum()), len(group1)
                    ),
                    "P value": safe_chi2_p(df["label"], df["gender"])
                    if gender == gender_values[0]
                    else "",
                },
            )
        )
    rows.append(
        row(
            "PTSD positive, n (%)",
            {
                "Overall": fmt_count_pct(int((df["ptsd_binary"] == 1).sum()), len(df)),
                "PHQ-8 < 10": fmt_count_pct(
                    int((group0["ptsd_binary"] == 1).sum()), len(group0)
                ),
                "PHQ-8 >= 10": fmt_count_pct(
                    int((group1["ptsd_binary"] == 1).sum()), len(group1)
                ),
                "P value": safe_chi2_p(df["label"], df["ptsd_binary"]),
            },
        )
    )

    numeric_rows = [
        ("PHQ-8 score, mean +/- SD", "phq_score"),
        ("Total turns, mean +/- SD", "turns_total"),
        ("Participant turns, mean +/- SD", "participant_turns"),
        ("Interviewer turns, mean +/- SD", "interviewer_turns"),
        ("Diagnostic interviewer turns, mean +/- SD", "diagnostic_interviewer_turns"),
        ("Participant words, mean +/- SD", "participant_words"),
        ("Interviewer words, mean +/- SD", "interviewer_words"),
        ("Cleaned interviewer words, mean +/- SD", "interviewer_cleaned_words"),
    ]
    for label, column in numeric_rows:
        rows.append(row(label, cells_for_numeric(column, 2)))
    return pd.DataFrame(rows)


def table2_input_lengths(conditions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str | int]] = []
    for condition in CONDITION_ORDER:
        g = conditions[conditions["input_condition"] == condition].copy()
        rows.append(
            {
                "Condition": CONDITION_LABELS[condition],
                "Rows": int(len(g)),
                "Non-empty rows": int((g["text"].fillna("").astype(str).str.strip() != "").sum()),
                "Empty rows": int((g["text"].fillna("").astype(str).str.strip() == "").sum()),
                "Words, mean +/- SD": fmt_mean_sd(g["word_count"], 2),
                "Words, median [IQR]": fmt_median_iqr(g["word_count"], 2),
                "Characters, mean +/- SD": fmt_mean_sd(g["char_count"], 2),
                "Source turns, mean +/- SD": fmt_mean_sd(g["source_turn_count"], 2),
                "Diagnostic turns removed, mean +/- SD": fmt_mean_sd(
                    g["diagnostic_removed_turn_count"], 2
                ),
            }
        )
    return pd.DataFrame(rows)


def fmt_mean_std_pair(mean: float, std: float, digits: int = 3) -> str:
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def table3_cv_performance(cv: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    cv = cv.copy()
    cv["model_rank"] = cv["model_short"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    cv["condition_rank"] = cv["input_condition"].map(
        {c: i for i, c in enumerate(CONDITION_ORDER)}
    )
    cv = cv.sort_values(["model_rank", "condition_rank"])
    for _, item in cv.iterrows():
        rows.append(
            {
                "Model": MODEL_LABELS.get(item["model_short"], item["model_short"]),
                "Input condition": CONDITION_LABELS[item["input_condition"]],
                "Accuracy": fmt_mean_std_pair(item["accuracy_mean"], item["accuracy_std"]),
                "F1": fmt_mean_std_pair(item["f1_mean"], item["f1_std"]),
                "Macro-F1": fmt_mean_std_pair(
                    item["macro_f1_mean"], item["macro_f1_std"]
                ),
                "AUC": fmt_mean_std_pair(item["auc_mean"], item["auc_std"]),
                "MCC": fmt_mean_std_pair(item["mcc_mean"], item["mcc_std"]),
            }
        )
    return pd.DataFrame(rows)


def table3_train_dev_performance(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    metrics = metrics.copy()
    metrics["model_rank"] = metrics["model_short"].map(
        {m: i for i, m in enumerate(MODEL_ORDER)}
    )
    metrics["condition_rank"] = metrics["input_condition"].map(
        {c: i for i, c in enumerate(CONDITION_ORDER)}
    )
    metrics = metrics.sort_values(["model_rank", "condition_rank"])
    for _, item in metrics.iterrows():
        rows.append(
            {
                "Model": MODEL_LABELS.get(item["model_short"], item["model_short"]),
                "Input condition": CONDITION_LABELS[item["input_condition"]],
                "Accuracy": fmt_number(item["accuracy"], 3),
                "F1": fmt_number(item["f1"], 3),
                "Macro-F1": fmt_number(item["macro_f1"], 3),
                "AUC": fmt_number(item["auc"], 3),
                "MCC": fmt_number(item["mcc"], 3),
            }
        )
    return pd.DataFrame(rows)


def table4_bias_metrics(bias: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    bias = bias.copy()
    bias["model_rank"] = bias["model_short"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    bias["eval_rank"] = bias["evaluation"].map({"5fold_cv": 0, "train_dev": 1})
    bias = bias.sort_values(["eval_rank", "model_rank"])
    for _, item in bias.iterrows():
        rows.append(
            {
                "Evaluation": "5-fold CV" if item["evaluation"] == "5fold_cv" else "Train/dev",
                "Model": MODEL_LABELS.get(item["model_short"], item["model_short"]),
                "IBPG F1": fmt_number(item["IBPG_f1"], 3),
                "DPD F1": fmt_number(item["DPD_f1"], 3),
                "IBPG Macro-F1": fmt_number(item["IBPG_macro_f1"], 3),
                "DPD Macro-F1": fmt_number(item["DPD_macro_f1"], 3),
                "IBPG MCC": fmt_number(item["IBPG_mcc"], 3),
                "DPD MCC": fmt_number(item["DPD_mcc"], 3),
            }
        )
    return pd.DataFrame(rows)


def threshold_sensitivity_frame(default_cv: pd.DataFrame, threshold_cv: pd.DataFrame) -> pd.DataFrame:
    default = default_cv.copy()
    default["model_family"] = default["model_short"].map(sensitivity_model_label)
    default = default[default["model_family"].isin(THRESHOLD_MODEL_ORDER)].copy()
    default = default[
        ["model_family", "input_condition", "macro_f1_mean", "mcc_mean"]
    ].rename(
        columns={
            "macro_f1_mean": "default_macro_f1",
            "mcc_mean": "default_mcc",
        }
    )

    tuned = threshold_cv.copy()
    tuned["model_family"] = tuned["model_short"].map(sensitivity_model_label)
    tuned = tuned[tuned["model_family"].isin(THRESHOLD_MODEL_ORDER)].copy()
    tuned = tuned[
        [
            "model_family",
            "input_condition",
            "macro_f1_mean",
            "mcc_mean",
            "threshold_mean",
        ]
    ].rename(
        columns={
            "macro_f1_mean": "tuned_macro_f1",
            "mcc_mean": "tuned_mcc",
        }
    )

    merged = default.merge(tuned, on=["model_family", "input_condition"], how="inner")
    merged["delta_macro_f1"] = merged["tuned_macro_f1"] - merged["default_macro_f1"]
    merged["delta_mcc"] = merged["tuned_mcc"] - merged["default_mcc"]
    merged["model_rank"] = merged["model_family"].map(
        {m: i for i, m in enumerate(THRESHOLD_MODEL_ORDER)}
    )
    merged["condition_rank"] = merged["input_condition"].map(
        {c: i for i, c in enumerate(CONDITION_ORDER)}
    )
    return merged.sort_values(["model_rank", "condition_rank"]).drop(
        columns=["model_rank", "condition_rank"]
    )


def table5_threshold_sensitivity(
    default_cv: pd.DataFrame, threshold_cv: pd.DataFrame
) -> pd.DataFrame:
    sensitivity = threshold_sensitivity_frame(default_cv, threshold_cv)
    rows: list[dict[str, str]] = []
    for _, item in sensitivity.iterrows():
        rows.append(
            {
                "Model": item["model_family"],
                "Input condition": CONDITION_LABELS[item["input_condition"]],
                "Default Macro-F1": fmt_number(item["default_macro_f1"], 3),
                "Tuned Macro-F1": fmt_number(item["tuned_macro_f1"], 3),
                "Delta Macro-F1": fmt_number(item["delta_macro_f1"], 3),
                "Default MCC": fmt_number(item["default_mcc"], 3),
                "Tuned MCC": fmt_number(item["tuned_mcc"], 3),
                "Delta MCC": fmt_number(item["delta_mcc"], 3),
                "Mean tuned threshold": fmt_number(item["threshold_mean"], 3),
            }
        )
    return pd.DataFrame(rows)


def table6_mcnemar_tests(mcnemar: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str | int]] = []
    df = mcnemar.copy()
    df["model_display"] = df["model_short"].map(display_model_label)
    df["model_rank"] = df["model_short"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    df["comparison_rank"] = df["comparison"].map(
        {name: i for i, (*_, name) in enumerate(COMPARISONS_FOR_TABLES)}
    )
    df = df.sort_values(["model_rank", "comparison_rank"])
    for _, item in df.iterrows():
        rows.append(
            {
                "Evaluation": "Train/dev" if item["evaluation"] == "train_dev" else item["evaluation"],
                "Model": item["model_display"],
                "Comparison": item["comparison"],
                "Paired n": int(item["paired_n"]),
                "Discordant n": int(item["discordant_n"]),
                "A correct / B wrong": int(item["a_correct_b_wrong"]),
                "A wrong / B correct": int(item["a_wrong_b_correct"]),
                "Exact p": fmt_p_value(item["mcnemar_exact_p"], 4),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    columns = [str(c) for c in df.columns]
    rows = [[str(v) for v in row] for row in df.to_numpy().tolist()]
    widths = []
    for i, col in enumerate(columns):
        values = [col] + [row[i] for row in rows]
        widths.append(max(len(v) for v in values))

    def fmt_row(values: list[str]) -> str:
        return "| " + " | ".join(values[i].ljust(widths[i]) for i in range(len(values))) + " |"

    sep = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([fmt_row(columns), sep] + [fmt_row(row) for row in rows])


def write_table_outputs(tables: dict[str, pd.DataFrame], output_dir: Path) -> Path:
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8")
        (output_dir / f"{name}.md").write_text(markdown_table(table), encoding="utf-8")

    workbook_path = output_dir / "paper_tables_C1_C4.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name[:31], index=False)
            worksheet = writer.sheets[name[:31]]
            for column_cells in worksheet.columns:
                values = [str(cell.value) if cell.value is not None else "" for cell in column_cells]
                width = min(max(len(value) for value in values) + 2, 60)
                worksheet.column_dimensions[column_cells[0].column_letter].width = width
    return workbook_path


def write_combined_markdown(tables: dict[str, pd.DataFrame], output_dir: Path) -> Path:
    sections = [
        "# C1-C4 Paper Tables",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]
    titles = {
        "table1_sample_characteristics": "Table 1. Sample characteristics",
        "table2_input_lengths": "Table 2. Input condition text length",
        "table3_model_performance_5fold_cv": "Table 3. Model x input condition performance (5-fold CV)",
        "table3_model_performance_train_dev": "Supplementary Table. Model x input condition performance (train/dev)",
        "table4_bias_metrics": "Table 4. IBPG / DPD bias metrics",
        "table5_threshold_sensitivity_5fold_cv": "Supplementary Table. Threshold sensitivity (5-fold CV, MCC-tuned)",
        "table6_mcnemar_tests_train_dev": "Supplementary Table. McNemar paired tests (train/dev)",
    }
    for name, table in tables.items():
        sections.extend([f"## {titles.get(name, name)}", "", markdown_table(table), ""])
    path = output_dir / "paper_tables_C1_C4.md"
    path.write_text("\n".join(sections), encoding="utf-8")
    return path


def plot_macro_f1_mcc(cv: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    plot_df = cv.copy()
    plot_df = plot_df[plot_df["input_condition"].isin(CONDITION_ORDER)].copy()
    plot_df["model_short_label"] = plot_df["model_short"].map(MODEL_LABELS)
    plot_df["condition_label"] = plot_df["input_condition"].map(CONDITION_LABELS)
    plot_df["condition_rank"] = plot_df["input_condition"].map(
        {c: i for i, c in enumerate(CONDITION_ORDER)}
    )
    plot_df["model_rank"] = plot_df["model_short"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    plot_df = plot_df.sort_values(["condition_rank", "model_rank"])

    source_path = output_dir / "figure_macro_f1_mcc_source_data.csv"
    plot_df.to_csv(source_path, index=False, encoding="utf-8")

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )

    colors = {
        "TF-IDF + LR": "#4C78A8",
        "TF-IDF + SVM": "#59A14F",
        "MPNet + LR": "#E15759",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharex=True)
    x = np.arange(len(CONDITION_ORDER))
    width = 0.24
    offsets = np.linspace(-width, width, len(MODEL_ORDER))

    panels = [
        ("A", "Macro-F1", "macro_f1_mean", "macro_f1_std"),
        ("B", "MCC", "mcc_mean", "mcc_std"),
    ]
    for ax, (panel, ylabel, mean_col, std_col) in zip(axes, panels):
        for offset, model in zip(offsets, MODEL_ORDER):
            model_label = MODEL_LABELS[model]
            model_df = (
                plot_df[plot_df["model_short"] == model]
                .set_index("input_condition")
                .reindex(CONDITION_ORDER)
            )
            means = model_df[mean_col].to_numpy()
            errors = model_df[std_col].to_numpy()
            ax.bar(
                x + offset,
                means,
                width=width,
                yerr=errors,
                capsize=2.5,
                color=colors[model_label],
                edgecolor="black",
                linewidth=0.4,
                label=model_label,
            )
            zero_mask = np.isclose(means, 0, atol=1e-12) & np.isclose(errors, 0, atol=1e-12)
            if zero_mask.any():
                ax.scatter(
                    (x + offset)[zero_mask],
                    np.full(int(zero_mask.sum()), 0.012),
                    marker="o",
                    s=18,
                    facecolors="white",
                    edgecolors=colors[model_label],
                    linewidths=1.0,
                    zorder=4,
                )
                for xpos in (x + offset)[zero_mask]:
                    ax.text(
                        xpos,
                        0.032,
                        "0",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        color=colors[model_label],
                    )
        ax.set_title(f"{panel}. {ylabel}")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(
            ["Full", "Participant", "Interviewer", "Cleaned"],
            rotation=25,
            ha="right",
        )
        ax.set_ylim(0, max(0.9, ax.get_ylim()[1]))
        ax.grid(axis="y", linewidth=0.4, alpha=0.25)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 1.18), ncols=3)
    fig.tight_layout()

    out_base = output_dir / "figure_macro_f1_mcc_5fold_cv"
    paths = {
        "png": out_base.with_suffix(".png"),
        "svg": out_base.with_suffix(".svg"),
        "pdf": out_base.with_suffix(".pdf"),
        "source": source_path,
    }
    fig.savefig(paths["png"], dpi=600, bbox_inches="tight")
    fig.savefig(paths["svg"], bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_threshold_sensitivity(
    default_cv: pd.DataFrame, threshold_cv: pd.DataFrame, output_dir: Path
) -> dict[str, Path]:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    sensitivity = threshold_sensitivity_frame(default_cv, threshold_cv)
    sensitivity["condition_label"] = sensitivity["input_condition"].map(CONDITION_LABELS)
    source_path = output_dir / "figure_threshold_sensitivity_source_data.csv"
    sensitivity.to_csv(source_path, index=False, encoding="utf-8")

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )

    colors = {
        "TF-IDF + LR": "#4C78A8",
        "TF-IDF + SVM": "#59A14F",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True)
    x = np.arange(len(CONDITION_ORDER))
    width = 0.34
    offsets = [-width / 2, width / 2]
    panels = [
        ("A", "Delta Macro-F1", "delta_macro_f1"),
        ("B", "Delta MCC", "delta_mcc"),
    ]
    for ax, (panel, ylabel, metric_col) in zip(axes, panels):
        max_abs = 0.02
        for offset, model in zip(offsets, THRESHOLD_MODEL_ORDER):
            model_df = (
                sensitivity[sensitivity["model_family"] == model]
                .set_index("input_condition")
                .reindex(CONDITION_ORDER)
            )
            values = model_df[metric_col].to_numpy(dtype=float)
            max_abs = max(max_abs, float(np.nanmax(np.abs(values))))
            ax.bar(
                x + offset,
                values,
                width=width,
                color=colors[model],
                edgecolor="black",
                linewidth=0.4,
                label=model,
            )
            zero_mask = np.isclose(values, 0, atol=1e-12)
            if zero_mask.any():
                ax.scatter(
                    (x + offset)[zero_mask],
                    np.zeros(int(zero_mask.sum())),
                    marker="o",
                    s=18,
                    facecolors="white",
                    edgecolors=colors[model],
                    linewidths=1.0,
                    zorder=4,
                )
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_title(f"{panel}. {ylabel}")
        ax.set_ylabel("Tuned - default")
        ax.set_xticks(x)
        ax.set_xticklabels(
            ["Full", "Participant", "Interviewer", "Cleaned"],
            rotation=25,
            ha="right",
        )
        ax.set_ylim(-max_abs * 1.25, max_abs * 1.25)
        ax.grid(axis="y", linewidth=0.4, alpha=0.25)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 1.18), ncols=2)
    fig.tight_layout()

    out_base = output_dir / "figure_threshold_sensitivity_5fold_cv"
    paths = {
        "png": out_base.with_suffix(".png"),
        "svg": out_base.with_suffix(".svg"),
        "pdf": out_base.with_suffix(".pdf"),
        "source": source_path,
    }
    fig.savefig(paths["png"], dpi=600, bbox_inches="tight")
    fig.savefig(paths["svg"], bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    plt.close(fig)
    return paths


def build_outputs(output_dir: Path) -> dict[str, Path]:
    output_dir = ensure_dir(output_dir)
    participants = pd.read_csv(PARTICIPANT_INDEX_CSV)
    conditions = pd.read_csv(CONDITIONS_CSV)
    cv = pd.read_csv(CV_SUMMARY_CSV)
    train_dev = pd.read_csv(TRAIN_DEV_METRICS_CSV)
    bias = pd.read_csv(BIAS_METRICS_CSV)
    threshold_cv = pd.read_csv(THRESHOLD_CV_SUMMARY_CSV)
    mcnemar = pd.read_csv(MCNEMAR_TESTS_CSV)

    tables = {
        "table1_sample_characteristics": table1_sample_characteristics(participants),
        "table2_input_lengths": table2_input_lengths(conditions),
        "table3_model_performance_5fold_cv": table3_cv_performance(cv),
        "table3_model_performance_train_dev": table3_train_dev_performance(train_dev),
        "table4_bias_metrics": table4_bias_metrics(bias),
        "table5_threshold_sensitivity_5fold_cv": table5_threshold_sensitivity(
            cv, threshold_cv
        ),
        "table6_mcnemar_tests_train_dev": table6_mcnemar_tests(mcnemar),
    }
    workbook_path = write_table_outputs(tables, output_dir)
    markdown_path = write_combined_markdown(tables, output_dir)
    figure_paths = plot_macro_f1_mcc(cv, output_dir)
    threshold_figure_paths = plot_threshold_sensitivity(cv, threshold_cv, output_dir)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "participant_index": str(PARTICIPANT_INDEX_CSV),
            "conditions": str(CONDITIONS_CSV),
            "cv_summary": str(CV_SUMMARY_CSV),
            "train_dev_metrics": str(TRAIN_DEV_METRICS_CSV),
            "bias_metrics": str(BIAS_METRICS_CSV),
            "threshold_cv_summary": str(THRESHOLD_CV_SUMMARY_CSV),
            "mcnemar_tests": str(MCNEMAR_TESTS_CSV),
        },
        "outputs": {
            "workbook": str(workbook_path),
            "markdown": str(markdown_path),
            **{f"macro_f1_mcc_figure_{k}": str(v) for k, v in figure_paths.items()},
            **{
                f"threshold_sensitivity_figure_{k}": str(v)
                for k, v in threshold_figure_paths.items()
            },
        },
    }
    manifest_path = output_dir / "paper_outputs_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "workbook": workbook_path,
        "markdown": markdown_path,
        "manifest": manifest_path,
        **{f"figure_{k}": v for k, v in figure_paths.items()},
        **{f"threshold_figure_{k}": v for k, v in threshold_figure_paths.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate C1-C4 paper tables and figures.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    outputs = build_outputs(args.output_dir)
    print("Generated paper outputs:")
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
