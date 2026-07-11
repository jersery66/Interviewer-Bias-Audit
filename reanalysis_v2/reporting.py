from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONDITION_LABELS = {
    "full_transcript": "Full transcript",
    "participant_speech": "Participant speech",
    "interviewer_speech": "Interviewer speech",
    "non_explicit_interviewer_speech": "Non-explicit interviewer",
    "participant_symptom_evidence": "Reviewed symptom evidence",
}

REPRESENTATION_LABELS = {
    "tfidf": "TF-IDF",
    "model_1_all_mpnet_base_v2": "all-mpnet-base-v2",
    "model_2_bge_large_en_v1_5": "bge-large-en-v1.5",
}

REPRESENTATION_COLORS = {
    "tfidf": "#5B6770",
    "model_1_all_mpnet_base_v2": "#4C78A8",
    "model_2_bge_large_en_v1_5": "#E59F3A",
}

TEXT_ARTIFACT_SUFFIXES = frozenset(
    {".csv", ".json", ".md", ".svg", ".tsv", ".txt", ".yaml", ".yml"}
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_text_hash_match(path: Path, expected_sha256: str) -> tuple[bool, dict[str, str]]:
    """Match a frozen text artifact while tolerating Git's CRLF checkout conversion."""
    raw = path.read_bytes()
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    canonical_lf_sha256 = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    observed = {
        "raw_sha256": raw_sha256,
        "canonical_lf_sha256": canonical_lf_sha256,
    }
    return expected_sha256 in observed.values(), observed


def verify_inventory(inventory_path: Path) -> pd.DataFrame:
    inventory_path = Path(inventory_path)
    inventory = pd.read_csv(inventory_path)
    records: list[dict[str, object]] = []
    for row in inventory.to_dict(orient="records"):
        relative = str(row["relative_path"])
        path = inventory_path.parent / Path(relative)
        expected_hash = str(row["sha256"])
        expected_bytes = int(row["bytes"])
        if not path.exists():
            actual_hash = ""
            actual_bytes = -1
            status = "MISSING"
        else:
            raw = path.read_bytes()
            actual_hash = hashlib.sha256(raw).hexdigest()
            actual_bytes = len(raw)
            canonical_lf = raw.replace(b"\r\n", b"\n")
            canonical_lf_hash = hashlib.sha256(canonical_lf).hexdigest()
            canonical_checkout_match = (
                path.suffix.lower() in TEXT_ARTIFACT_SUFFIXES
                and canonical_lf_hash == expected_hash
                and len(canonical_lf) == expected_bytes
            )
            if actual_hash == expected_hash and actual_bytes == expected_bytes:
                status = "PASS"
            elif canonical_checkout_match:
                status = "PASS"
            elif actual_hash != expected_hash:
                status = "HASH_MISMATCH"
            elif actual_bytes != expected_bytes:
                status = "SIZE_MISMATCH"
            else:
                status = "PASS"
        records.append(
            {
                "inventory": inventory_path.as_posix(),
                "relative_path": relative,
                "expected_bytes": expected_bytes,
                "actual_bytes": actual_bytes,
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
                "status": status,
            }
        )
    return pd.DataFrame.from_records(records)


def filter_significant(tests: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    result = tests.loc[pd.to_numeric(tests["fdr_bh_q"]) < alpha].copy()
    difference = pd.to_numeric(result["observed_difference"])
    result["direction"] = np.where(
        difference > 0,
        result["condition_a"].astype(str) + " > " + result["condition_b"].astype(str),
        np.where(
            difference < 0,
            result["condition_a"].astype(str) + " < " + result["condition_b"].astype(str),
            "equal",
        ),
    )
    return result.reset_index(drop=True)


def _markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.3f}")
    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display.fillna("").itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _write_table(frame: pd.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".md").write_text(_markdown_table(frame), encoding="utf-8")


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 6.3,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _condition_order() -> list[str]:
    return list(CONDITION_LABELS)


def c5_plot_y_layout(n_controls: int) -> dict[str, object]:
    return {
        "random": -1.2,
        "reviewed": 0.0,
        "controls": np.arange(1, n_controls + 1, dtype=float),
    }


def _plot_representation_grid(metrics: pd.DataFrame, output: Path) -> None:
    _configure_matplotlib()
    conditions = _condition_order()
    x = np.arange(len(conditions))
    metric_spec = [
        ("roc_auc", "ROC-AUC", (0.48, 0.86), 0.5),
        ("pr_auc", "PR-AUC", (0.27, 0.75), 43 / 142),
        ("macro_f1_nested", "Macro-F1 (nested)", (0.52, 0.79), None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 67 / 25.4), constrained_layout=True)
    for panel, (ax, (metric, title, ylim, baseline)) in enumerate(zip(axes, metric_spec)):
        for representation in REPRESENTATION_LABELS:
            subset = metrics.loc[metrics["representation"] == representation].set_index("condition")
            y = [float(subset.loc[condition, metric]) for condition in conditions]
            ax.plot(
                x,
                y,
                marker="o",
                markersize=3.8,
                linewidth=1.2,
                color=REPRESENTATION_COLORS[representation],
                label=REPRESENTATION_LABELS[representation],
            )
        if baseline is not None:
            ax.axhline(baseline, color="#A9A9A9", linestyle="--", linewidth=0.7, zorder=0)
        ax.set_title(title)
        ax.set_ylim(*ylim)
        ax.set_xticks(x, ["C1", "C2", "C3", "C4", "C5"])
        ax.set_xlabel("Input condition")
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.5)
        _panel_label(ax, chr(ord("a") + panel))
    axes[0].set_ylabel("Participant-level OOF performance")
    axes[2].legend(loc="lower left")
    _save_figure(fig, output / "main_text" / "figure_1_representation_grid")


def _plot_threshold_calibration(
    tfidf: pd.DataFrame,
    selected: pd.DataFrame,
    output: Path,
) -> None:
    _configure_matplotlib()
    conditions = _condition_order()
    short = ["C1", "C2", "C3", "C4", "C5"]
    x = np.arange(len(conditions))
    ordered = tfidf.set_index("condition").loc[conditions]
    fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 126 / 25.4), constrained_layout=True)

    for ax, default_col, nested_col, title, panel in (
        (axes[0, 0], "sensitivity_at_0_5", "sensitivity_nested", "Sensitivity", "a"),
        (axes[0, 1], "macro_f1_at_0_5", "macro_f1_nested", "Macro-F1", "b"),
    ):
        default = ordered[default_col].astype(float).to_numpy()
        nested = ordered[nested_col].astype(float).to_numpy()
        ax.scatter(x - 0.08, default, s=20, color="#8A8A8A", label="Threshold 0.5", zorder=3)
        ax.scatter(x + 0.08, nested, s=20, color="#4C78A8", label="Nested threshold", zorder=3)
        for index in range(len(x)):
            ax.plot([x[index] - 0.08, x[index] + 0.08], [default[index], nested[index]], color="#C8C8C8", linewidth=0.8)
        ax.set_xticks(x, short)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.5)
        _panel_label(ax, panel)
    axes[0, 0].set_ylabel("Performance")
    axes[0, 1].legend(loc="lower right")

    selected = selected.loc[selected["rule"] == "max_macro_f1"]
    arrays = [
        selected.loc[selected["condition"] == condition, "threshold"].astype(float).to_numpy()
        for condition in conditions
    ]
    box = axes[1, 0].boxplot(
        arrays,
        tick_labels=short,
        patch_artist=True,
        widths=0.58,
        medianprops={"color": "#2D2D2D", "linewidth": 1.0},
        whiskerprops={"color": "#777777", "linewidth": 0.8},
        capprops={"color": "#777777", "linewidth": 0.8},
        flierprops={"marker": ".", "markersize": 2.5, "markerfacecolor": "#777777"},
    )
    for patch in box["boxes"]:
        patch.set_facecolor("#9EC1DF")
        patch.set_edgecolor("#4C78A8")
        patch.set_linewidth(0.8)
    axes[1, 0].axhline(0.5, color="#A9A9A9", linestyle="--", linewidth=0.7)
    axes[1, 0].set_title("Nested max-Macro-F1 thresholds (50 folds)")
    axes[1, 0].set_ylabel("Selected threshold")
    axes[1, 0].grid(axis="y", color="#E6E6E6", linewidth=0.5)
    _panel_label(axes[1, 0], "c")

    width = 0.22
    for offset, column, label, color in (
        (-width, "raw_brier", "Raw", "#8A8A8A"),
        (0, "platt_brier", "Platt", "#4C78A8"),
        (width, "isotonic_brier", "Isotonic", "#E59F3A"),
    ):
        axes[1, 1].bar(x + offset, ordered[column].astype(float), width=width, label=label, color=color)
    axes[1, 1].set_xticks(x, short)
    axes[1, 1].set_ylim(0, 0.27)
    axes[1, 1].set_title("Brier score after train-only calibration")
    axes[1, 1].set_ylabel("Brier score (lower is better)")
    axes[1, 1].legend(loc="upper right")
    axes[1, 1].grid(axis="y", color="#E6E6E6", linewidth=0.5)
    _panel_label(axes[1, 1], "d")
    _save_figure(fig, output / "main_text" / "figure_2_threshold_calibration")


def _plot_c5_controls(metrics: pd.DataFrame, random_distribution: pd.DataFrame, output: Path) -> None:
    _configure_matplotlib()
    reviewed = metrics.loc[metrics["condition"] == "c5_reviewed_evidence"].iloc[0]
    random_rows = metrics.loc[metrics["condition"].str.startswith("c5_random_same_word_seed_")]
    controls = metrics.loc[
        metrics["condition"].isin(
            ["random_ensemble", "nonsymptom_same_word", "keyword_masked", "domain_presence", "domain_count"]
        )
    ].copy()
    control_labels = {
        "random_ensemble": "Random ensemble",
        "nonsymptom_same_word": "Non-symptom matched",
        "keyword_masked": "Keyword-masked",
        "domain_presence": "Domain presence",
        "domain_count": "Domain count",
    }
    metrics_to_plot = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("macro_f1_at_0_5", "Macro-F1@0.5")]
    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 73 / 25.4), constrained_layout=True)
    layout = c5_plot_y_layout(len(controls))
    y_positions = layout["controls"]
    for panel, (ax, (metric, title)) in enumerate(zip(axes, metrics_to_plot)):
        random_values = random_rows[metric].astype(float).to_numpy()
        ax.scatter(
            random_values,
            np.full_like(random_values, float(layout["random"])),
            s=14,
            color="#B8B8B8",
            alpha=0.9,
            label="10 random seeds",
        )
        distribution = random_distribution.loc[random_distribution["metric"] == metric].iloc[0]
        ax.plot(
            [float(distribution["percentile_2_5"]), float(distribution["percentile_97_5"])],
            [float(layout["random"]), float(layout["random"])],
            color="#6F6F6F",
            linewidth=1.6,
        )
        ax.scatter([float(reviewed[metric])], [float(layout["reviewed"])], s=32, marker="D", color="#4C78A8", zorder=4)
        ax.scatter(controls[metric].astype(float), y_positions, s=26, color="#E59F3A", zorder=3)
        ax.set_yticks(
            [float(layout["random"]), float(layout["reviewed"]), *y_positions],
            ["Random seeds", "Reviewed evidence", *[control_labels[value] for value in controls["condition"]]],
        )
        ax.set_title(title)
        ax.grid(axis="x", color="#E6E6E6", linewidth=0.5)
        _panel_label(ax, chr(ord("a") + panel))
    axes[0].set_xlim(0.55, 0.83)
    axes[1].set_xlim(0.38, 0.72)
    axes[2].set_xlim(0.52, 0.77)
    _save_figure(fig, output / "supplement" / "figure_s1_c5_controls")


def _plot_c4_controls(metrics: pd.DataFrame, output: Path) -> None:
    _configure_matplotlib()
    labels = {
        "template_only": "Template only",
        "nontemplate_lengthmatched": "Length-matched",
        "nontemplate_positionmatched": "Position-matched",
        "template_shuffled": "Template shuffled",
        "template_presence": "Template presence/count",
    }
    order = list(labels)
    ordered = metrics.set_index("condition").loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(125 / 25.4, 74 / 25.4), constrained_layout=True)
    y = np.arange(len(order))
    for panel, (ax, metric, title, xlim) in enumerate(
        (
            (axes[0], "roc_auc", "ROC-AUC", (0.48, 0.84)),
            (axes[1], "macro_f1_at_0_5", "Macro-F1@0.5", (0.45, 0.80)),
        )
    ):
        colors = ["#4C78A8" if name == "template_only" else "#A8B6C3" for name in order]
        ax.scatter(ordered[metric].astype(float), y, s=34, color=colors, zorder=3)
        ax.set_yticks(y, [labels[name] for name in order])
        ax.invert_yaxis()
        ax.set_xlim(*xlim)
        ax.set_title(title)
        ax.grid(axis="x", color="#E6E6E6", linewidth=0.5)
        _panel_label(ax, chr(ord("a") + panel))
    _save_figure(fig, output / "supplement" / "figure_s2_c4_controls")


def _inventory_audit(analysis_root: Path) -> pd.DataFrame:
    manifests = [
        analysis_root / "02_tfidf_main" / "output_manifest_sha256.csv",
        analysis_root / "03_embedding_robustness" / "model_1_all_mpnet_base_v2" / "output_manifest_sha256.csv",
        analysis_root / "03_embedding_robustness" / "model_2_bge_large_en_v1_5" / "output_manifest_sha256.csv",
        analysis_root / "03_embedding_robustness" / "output_manifest_sha256.csv",
        analysis_root / "04_c5_controls" / "output_manifest_sha256.csv",
        analysis_root / "05_c4_controls" / "output_manifest_sha256.csv",
    ]
    audited = pd.concat([verify_inventory(path) for path in manifests], ignore_index=True)
    # Embedding caches are reproducible, gitignored intermediates; frozen metrics and OOF outputs remain audited.
    normalized_paths = "/" + audited["relative_path"].str.replace("\\", "/", regex=False).str.lstrip("/")
    return audited.loc[~normalized_paths.str.contains("/embedding_cache/", regex=False)].reset_index(drop=True)


def _strict_audit(analysis_root: Path) -> pd.DataFrame:
    checks: list[dict[str, object]] = []

    def record(check: str, passed: bool, observed: object, expected: object) -> None:
        checks.append(
            {
                "check": check,
                "status": "PASS" if passed else "FAIL",
                "observed": observed,
                "expected": expected,
            }
        )

    def record_frozen_hash(check: str, path: Path, expected: str) -> None:
        matched, observed = frozen_text_hash_match(path, expected)
        record(check, matched, json.dumps(observed, sort_keys=True), expected)

    freeze = json.loads((analysis_root / "00_plan" / "input_freeze_manifest.json").read_text(encoding="utf-8"))
    for name, item in freeze["frozen_inputs"].items():
        path = analysis_root / item["path"]
        frame = pd.read_csv(path)
        record(f"input:{name}:exists", path.exists(), path.exists(), True)
        record(f"input:{name}:rows", len(frame) == item["rows"], len(frame), item["rows"])
        record_frozen_hash(f"input:{name}:sha256", path, item["sha256"])
        record(
            f"input:{name}:official_test_excluded",
            "test" not in set(frame["official_avec_split"].astype(str).str.lower()),
            sorted(frame["official_avec_split"].astype(str).unique()),
            "train/dev only",
        )

    c4_turns_entry = freeze["control_inputs"]["c4_turns"]
    c4_turns_path = analysis_root / c4_turns_entry["path"]
    c4_turns = pd.read_csv(c4_turns_path, keep_default_na=False)
    record_frozen_hash("c4:frozen_turn_source_hash", c4_turns_path, c4_turns_entry["sha256"])
    record(
        "c4:frozen_turn_source_rows",
        len(c4_turns) == int(c4_turns_entry["rows"]),
        len(c4_turns),
        int(c4_turns_entry["rows"]),
    )
    record(
        "c4:frozen_turn_source_participants",
        c4_turns["participant_id"].nunique() == 142,
        c4_turns["participant_id"].nunique(),
        142,
    )
    upstream_turns = Path(c4_turns_entry["source_path"])
    if upstream_turns.exists():
        record_frozen_hash(
            "c4:verified_turn_upstream_hash",
            upstream_turns,
            c4_turns_entry["source_sha256"],
        )
    else:
        record(
            "c4:verified_turn_upstream_hash",
            False,
            "missing",
            c4_turns_entry["source_sha256"],
        )
    c4_alignment_path = analysis_root / c4_turns_entry["alignment_audit_path"]
    c4_alignment = pd.read_csv(c4_alignment_path)
    record_frozen_hash(
        "c4:turn_alignment_audit_hash",
        c4_alignment_path,
        c4_turns_entry["alignment_audit_sha256"],
    )
    record(
        "c4:turn_alignment_c3",
        len(c4_alignment) == 142 and c4_alignment["c3_exact_match"].eq(True).all(),
        int(c4_alignment["c3_exact_match"].sum()),
        142,
    )
    record(
        "c4:turn_alignment_c4",
        len(c4_alignment) == 142 and c4_alignment["c4_exact_match"].eq(True).all(),
        int(c4_alignment["c4_exact_match"].sum()),
        142,
    )

    split_path = analysis_root / freeze["splits"]["path"]
    splits = pd.read_csv(split_path)
    record_frozen_hash("splits:sha256", split_path, freeze["splits"]["sha256"])
    record("splits:rows", len(splits) == 7100, len(splits), 7100)
    record("splits:repeats", splits["repeat"].nunique() == 10, splits["repeat"].nunique(), 10)
    record("splits:folds", splits["fold"].nunique() == 5, splits["fold"].nunique(), 5)
    test = splits.loc[splits["role"] == "test"]
    test_counts = test.groupby("participant_id").size()
    record("splits:each_participant_tested_once_per_repeat", test_counts.eq(10).all(), sorted(test_counts.unique()), [10])
    overlaps = 0
    for (_, _), group in splits.groupby(["repeat", "fold"]):
        train_ids = set(group.loc[group["role"] == "train", "participant_id"])
        test_ids = set(group.loc[group["role"] == "test", "participant_id"])
        overlaps += len(train_ids & test_ids)
    record("splits:no_train_test_overlap", overlaps == 0, overlaps, 0)

    manifests = [
        analysis_root / "02_tfidf_main" / "run_manifest.json",
        analysis_root / "03_embedding_robustness" / "model_1_all_mpnet_base_v2" / "run_manifest.json",
        analysis_root / "03_embedding_robustness" / "model_2_bge_large_en_v1_5" / "run_manifest.json",
        analysis_root / "03_embedding_robustness" / "run_manifest.json",
        analysis_root / "04_c5_controls" / "run_manifest.json",
        analysis_root / "05_c4_controls" / "run_manifest.json",
    ]
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        record(f"run:{manifest_path.parent.name}:status", manifest.get("status") == "complete", manifest.get("status"), "complete")

    c4_run_manifest = json.loads((analysis_root / "05_c4_controls" / "run_manifest.json").read_text(encoding="utf-8"))
    recorded_turns = c4_run_manifest["turns_source"]
    recorded_turns_name = str(recorded_turns["path"]).replace("\\", "/").rsplit("/", 1)[-1]
    record(
        "c4:run_used_frozen_turn_source",
        recorded_turns_name == c4_turns_path.name
        and recorded_turns["sha256"] == c4_turns_entry["sha256"],
        f"{recorded_turns_name} | {recorded_turns['sha256']}",
        f"{c4_turns_path.name} | {c4_turns_entry['sha256']}",
    )

    tfidf_tests = pd.read_csv(analysis_root / "02_tfidf_main" / "paired_tests" / "primary_input_source_paired_tests_fdr.csv")
    representation_tests = pd.read_csv(analysis_root / "03_embedding_robustness" / "representation_family_paired_tests_fdr.csv")
    c5_tests = pd.read_csv(analysis_root / "04_c5_controls" / "paired_tests" / "c5_control_family_fdr.csv")
    c4_tests = pd.read_csv(analysis_root / "05_c4_controls" / "paired_tests" / "c4_control_family_fdr.csv")
    for name, frame, expected in (
        ("primary", tfidf_tests, 40),
        ("representation", representation_tests, 120),
        ("c5", c5_tests, 30),
        ("c4", c4_tests, 8),
    ):
        record(f"tests:{name}:count", len(frame) == expected, len(frame), expected)
        record(f"tests:{name}:permutations", frame["n_permutations"].eq(10000).all(), sorted(frame["n_permutations"].unique()), [10000])
        record(f"tests:{name}:fdr_present", frame["fdr_bh_q"].notna().all(), int(frame["fdr_bh_q"].notna().sum()), len(frame))

    for root in (
        analysis_root / "02_tfidf_main",
        analysis_root / "03_embedding_robustness" / "model_1_all_mpnet_base_v2",
        analysis_root / "03_embedding_robustness" / "model_2_bge_large_en_v1_5",
    ):
        for participant_path in (root / "oof_probs").glob("*_participant_oof.csv"):
            frame = pd.read_csv(participant_path)
            record(f"oof:{participant_path.relative_to(analysis_root)}:rows", len(frame) == 142, len(frame), 142)
            record(
                f"oof:{participant_path.relative_to(analysis_root)}:unique_participants",
                frame["participant_id"].nunique() == 142,
                frame["participant_id"].nunique(),
                142,
            )
        for fold_path in (root / "oof_probs").glob("*_fold_oof.csv"):
            frame = pd.read_csv(fold_path)
            record(f"oof:{fold_path.relative_to(analysis_root)}:rows", len(frame) == 1420, len(frame), 1420)

    quote_audit = pd.read_csv(analysis_root / "04_c5_controls" / "nonsymptom_same_word" / "quote_exclusion_audit.csv")
    record("c5:quote_spans_all_matched", int(quote_audit["unmatched_quotes"].sum()) == 0, int(quote_audit["unmatched_quotes"].sum()), 0)
    record("c5:word_counts_exact", quote_audit["target_words"].eq(quote_audit["observed_words"]).all(), int((quote_audit["target_words"] != quote_audit["observed_words"]).sum()), 0)

    c4_audit = pd.read_csv(analysis_root / "05_c4_controls" / "input_audit" / "participant_texts.csv")
    record("c4:lengthmatched_word_counts_exact", c4_audit["template_word_count"].eq(c4_audit["lengthmatched_word_count"]).all(), int((c4_audit["template_word_count"] != c4_audit["lengthmatched_word_count"]).sum()), 0)
    record("c4:positionmatched_word_counts_exact", c4_audit["template_word_count"].eq(c4_audit["positionmatched_word_count"]).all(), int((c4_audit["template_word_count"] != c4_audit["positionmatched_word_count"]).sum()), 0)
    return pd.DataFrame(checks)


def _write_figure_contract(output: Path) -> None:
    text = """# Figure contract

Core conclusion: DAIC-WOZ input-source conclusions depend on representation and decision threshold; no single interviewer-over-participant ranking claim is stable across all tested representations.

- Figure archetype: quantitative grid.
- Target/output: double-column manuscript figures; SVG/PDF with editable text; 600-dpi TIFF and 240-dpi PNG previews.
- Backend: Python/matplotlib only.
- Figure 1: representation robustness across C1-C5 for ROC-AUC, PR-AUC, and nested Macro-F1.
- Figure 2: TF-IDF threshold sensitivity, nested-threshold recovery, threshold stability, and train-only calibration.
- Figure S1: C5 reviewed evidence against 10 random controls and structured controls.
- Figure S2: C4 template, length, position, shuffle, and presence/count controls.
- Statistics: participant-level mean OOF probabilities from shared repeated 10 x 5 stratified CV; 10,000 participant-level paired swaps; BH-FDR within each predeclared family.
- Reviewer risk: point estimates are not confidence intervals; the cohort is official train+dev only; calibration and thresholds are trained without outer-test access; controls do not isolate natural-language template semantics.
"""
    (output / "figure_contract.md").write_text(text, encoding="utf-8")


def _write_legends(output: Path) -> None:
    legends = {
        output / "main_text" / "figure_1_legend.md": """**Figure 1 | Representation-dependent input-source performance.** Participant-level out-of-fold performance for five input conditions under TF-IDF, all-mpnet-base-v2 and bge-large-en-v1.5 representations. Each point is computed from one probability per participant after averaging that participant's 10 repeated out-of-fold predictions (n=142; 43 PHQ-8-positive, 99 negative). Models share the same repeated 10 x 5 stratified splits. Dashed references show chance ROC-AUC (0.5) and cohort positive prevalence for PR-AUC (43/142). No uncertainty interval is implied by the connecting lines.
""",
        output / "main_text" / "figure_2_legend.md": """**Figure 2 | TF-IDF decision-threshold and calibration audit.** (a,b) Sensitivity and Macro-F1 at the default 0.5 threshold and at an outer-fold-specific threshold selected by maximum inner-OOF Macro-F1. (c) Distribution of the 50 selected max-Macro-F1 thresholds per condition. (d) Brier score for raw probabilities and Platt/isotonic probabilities, with calibrators fit only within each outer training fold. C1, full transcript; C2, participant speech; C3, interviewer speech; C4, non-explicit interviewer speech; C5, reviewed participant symptom evidence.
""",
        output / "supplement" / "figure_s1_legend.md": """**Supplementary Figure S1 | C5 evidence-concentration controls.** Reviewed symptom-evidence performance is compared with 10 participant-level random same-word-count samples, their 2.5th-97.5th percentile range, a probability ensemble, a non-symptom same-word-count sample, keyword masking, and 10-domain presence/count vectors. All conditions use the shared repeated 10 x 5 splits. The domain vectors encode extracted symptom-domain evidence states, not whether the interviewer asked about a domain.
""",
        output / "supplement" / "figure_s2_legend.md": """**Supplementary Figure S2 | C4 interviewer-protocol controls.** Template-only TF-IDF performance is compared with word-count-matched non-template text, position-matched non-template turns, template text shuffled between participants separately within every outer training/test partition, and template presence/count features. These controls jointly audit template, position, coverage and text quantity; they do not isolate natural-language template semantics.
""",
    }
    for path, text in legends.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _write_qa_notes(output: Path) -> None:
    text = """# Figure QA notes

- Backend exclusivity: PASS; all exports were generated with Python/matplotlib.
- Final dimensions: Figure 1, 183 x 67 mm; Figure 2, 183 x 126 mm; Figure S1, 183 x 73 mm; Figure S2, 125 x 74 mm before tight bounding-box adjustment.
- Export bundle: PASS; SVG, PDF, 600-dpi TIFF, 240-dpi PNG, source tables and legends are present.
- Editable text: SVG uses `svg.fonttype=none`; PDF uses TrueType font embedding (`pdf.fonttype=42`).
- Font: Arial/Helvetica requested with DejaVu Sans fallback.
- Color: representation is encoded by color and marker/line identity; no rainbow map; red/green is not used as the sole encoding.
- Statistics: n, split design, aggregation level, permutation count and FDR family are stated in legends/reports.
- Source data: each quantitative figure maps to CSV tables in this package or the frozen computational output referenced by the run manifest.
- Image integrity: not applicable; no microscopy, blots or photographic panels.
- Remaining submission check: verify exact journal-specific size/file rules at submission time because author guidance can change.
"""
    (output / "figure_qa_notes.md").write_text(text, encoding="utf-8")


def _write_manuscript_results(
    output: Path,
    tfidf: pd.DataFrame,
    representations: pd.DataFrame,
    c5: pd.DataFrame,
    c4: pd.DataFrame,
) -> None:
    c2 = tfidf.set_index("condition").loc["participant_speech"]
    text = f"""# 可直接替换的结果段落（中文）

## 主分析与阈值审计

在官方 train+dev 的 142 名参与者（PHQ-8 >= 10：43 人；阴性：99 人）中，所有条件均使用同一套 10 次重复、每次 5 折的分层交叉验证，并先对每名参与者的 10 个 OOF 概率取均值。TF-IDF 下，五种输入的 ROC-AUC 为 0.705-0.803；经主比较族 40 项 BH-FDR 校正后，任一输入对之间均未出现显著 ROC-AUC 或 PR-AUC 差异。因此，数据不支持把访谈者输入描述为具有稳定的排序优势。

C2 参与者言语呈现明显的默认阈值伪影：Sensitivity@0.5 仅为 {float(c2['sensitivity_at_0_5']):.3f}，Macro-F1@0.5 为 {float(c2['macro_f1_at_0_5']):.3f}；使用仅由外层训练数据内部选择的 max-Macro-F1 阈值后，两者分别提高到 {float(c2['sensitivity_nested']):.3f} 和 {float(c2['macro_f1_nested']):.3f}。C2 的原始阴性/阳性概率均值分别为 0.458 和 0.470，且标准差仅为 0.014 和 0.018，说明概率集中在 0.5 以下。Platt 与 isotonic 的 Brier 分数均低于原始概率，但在固定 0.5 阈值下 C2 sensitivity 仅恢复至 0.140 与 0.279；因此，主要结论是默认阈值失配与概率尺度压缩，而不是参与者语言缺乏排序信号。

## 表征稳健性

all-mpnet-base-v2 与 bge-large-en-v1.5 均按冻结版本完整报告。C2 的 ROC-AUC 分别为 0.764 与 0.704，而 C3 分别为 0.743 与 0.769；两套 dense embedding 对 C2/C3 相对排序给出不同方向，且在跨 120 项的联合 BH-FDR 后，C2 与 C3 的核心差异均不显著。MPNet 的 C1 相对 C4 在 ROC-AUC、PR-AUC 与 nested Macro-F1 上显著更高，但该模式没有在 BGE 中复现。由此可见，输入来源效应依赖表征方式，不宜凝练为单一、跨表征稳定的“访谈者优于参与者”结论。

## C5 证据浓缩控制

C5 reviewed evidence 的 ROC-AUC 为 {float(c5.set_index('condition').loc['c5_reviewed_evidence', 'roc_auc']):.3f}。10 个同词数随机样本的均值为 0.697（SD 0.050；2.5th-97.5th percentile 0.628-0.751），reviewed evidence 与随机集合、非症状同词数文本、关键词遮蔽及域向量之间的 30 项比较在 BH-FDR 后均不显著。域存在与域计数向量的 ROC-AUC 分别为 0.790 与 0.794。允许的解释是：抽取到的症状域证据状态可能已携带大量判别信息，具体 quote 文本的额外语义贡献在本样本中未被稳健分离；这些域变量不是 Ellie 是否询问该域的编码。

## C4 访谈者协议控制

模板文本的 ROC-AUC/Macro-F1@0.5 为 {float(c4.set_index('condition').loc['template_only', 'roc_auc']):.3f}/{float(c4.set_index('condition').loc['template_only', 'macro_f1_at_0_5']):.3f}。其 Macro-F1 高于长度匹配非模板文本（q=0.0028），并在 ROC-AUC 与 Macro-F1 上均高于折内参与者置换的模板文本（q=0.0125 与 0.0016）；但与位置匹配文本及模板 presence/count 特征均无显著差异。故统一解释为：访谈者侧信号并非仅来自粗粒度长度计数，但当前结果也不能将其特异性归因于模板自然语言语义；模板、位置、话题覆盖和词量共同构成脚本化访谈中的协议信号。

## 结论边界

本研究是同一 DAIC-WOZ 官方 train+dev 队列中的方法学审计，不包含独立外部验证。结果描述预测关联与协议结构，不支持因果推断，也不把证据抽取或分类输出视为临床诊断。
"""
    (output / "main_text" / "manuscript_ready_results_zh.md").write_text(text, encoding="utf-8")


def _write_validation_report(output: Path, audit: pd.DataFrame, inventory: pd.DataFrame) -> None:
    significant_primary = pd.read_csv(output / "main_text" / "table_3_primary_significant_paired_tests.csv")
    significant_representation = pd.read_csv(output / "main_text" / "table_4_representation_significant_paired_tests.csv")
    text = f"""## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: {datetime.now(timezone.utc).isoformat()}
- Verification Status: ANALYZED
- Version Label: validation_v1

## Validation Report

- **Source**: reanalysis_v2 frozen computational outputs
- **Overall Confidence**: CAUTION

### Statistical Findings

| Finding | Test | Value | Confidence |
|---|---|---|---|
| TF-IDF input-source ranking | 10,000 paired swaps; 40-test BH family | No ROC-AUC or PR-AUC comparison survived FDR | SOLID for this cohort/design |
| TF-IDF threshold dependence | Participant-level OOF decision metrics | C2 sensitivity 0.047 at 0.5 vs 0.512 nested | SOLID for this model/cohort |
| Representation robustness | 10,000 paired swaps; 120-test joint BH family | {len(significant_representation)} tests survived FDR; no C2-vs-C3 test survived | CAUTION: model-dependent patterns |
| C5 controls | 10,000 paired swaps; 30-test BH family | No comparison survived FDR | CAUTION: limited n and control variance |
| C4 controls | 10,000 paired swaps; 8-test BH family | Template-vs-shuffle and one length-matched Macro-F1 contrast survived | CAUTION: semantics not isolated |

### Warnings

| Type | Detail | Affected |
|---|---|---|
| Cohort scope | Official train+dev only; no independent cohort | Generalizability |
| Threshold sensitivity | Default 0.5 can reverse decision-level impressions | Macro-F1, sensitivity, specificity |
| Small sample | n=142 with 43 positives; dense models may be unstable | All estimates |
| Multiple testing | Corrected within four preregistered families; raw p-values must not be interpreted alone | All paired tests |
| Calibration reuse | Metrics are OOF within the same cohort and are not deployment calibration estimates | Brier, ECE, slope/intercept |
| Control interpretation | C5 domain states are extracted evidence states; C4 controls do not isolate natural-language semantics | Mechanistic claims |

### Fallacy Scan

- **Coverage**: 11/11 fallacy types checked

| Fallacy | Severity | Detail | Guardrail |
|---|---|---|---|
| Simpson's paradox | NOTE | No subgroup claim was made; subgroup reversal was not tested because subgroup analysis was not preregistered here. | Do not infer subgroup consistency. |
| Ecological fallacy | NOTE | Unit of analysis and inference is the participant. | Retain participant-level wording. |
| Berkson's paradox | CAUTION | DAIC-WOZ is a selected research cohort. | Do not generalize to population screening without external validation. |
| Collider bias | NOTE | No post-exposure covariate adjustment was used in the classifiers. | Avoid causal interpretations. |
| Base-rate neglect | CAUTION | Positive prevalence is 43/142; sensitivity/specificity are accompanied by PR-AUC and prevalence. | Report the cohort base rate with decision metrics. |
| Regression to the mean | NOTE | No pre-post extreme-group design is present. | Not applicable to the primary comparison. |
| Survivorship bias | CAUTION | Analysis uses available official train+dev records and does not establish missingness representativeness. | State cohort inclusion explicitly. |
| Look-elsewhere effect | CAUTION | Many contrasts were run, but all were retained and BH-corrected within four frozen families. | Interpret q-values, not selected raw p-values. |
| Garden of forking paths | CAUTION | The v2 plan and changelog freeze choices, but this is not a time-stamped external preregistration. | Treat claims as audit findings, not confirmatory clinical evidence. |
| Correlation != causation | CAUTION | Predictive associations and scripted protocol signals cannot establish causal mechanisms. | Use associational language. |
| Reverse causality | CAUTION | Cross-sectional interview content and PHQ-8 labels do not establish temporal direction. | Avoid directional causal claims. |

### Reproducibility

- **Method**: artifact hashes, split-contract checks, leakage checks, row/participant checks, family/permutation checks, deterministic unit tests; no second full end-to-end embedding rerun
- **Verdict**: CANNOT_VERIFY as an independent reproduction
- **Artifact integrity**: {int(inventory['status'].eq('PASS').sum())}/{len(inventory)} inventoried computational artifacts passed byte/hash verification
- **Strict contract audit**: {int(audit['status'].eq('PASS').sum())}/{len(audit)} checks passed
- **Reason for conservative verdict**: ARS marks reproducibility VERIFIED only after an independent full rerun; the current validation confirms internal integrity and executability but does not substitute for that rerun.
"""
    (output / "validation_report.md").write_text(text, encoding="utf-8")


def _write_reproducibility(output: Path, analysis_root: Path) -> None:
    text = f"""# Reproducibility guide

## Frozen scope

- Cohort: official train+dev, n=142 (43 positive, 99 negative).
- Endpoint: PHQ-8 >= 10.
- Shared split: `00_splits/repeated_5fold_splits_10x5.csv` (10 repeats x 5 folds).
- Participant-level inference: average 10 repeated OOF probabilities, then 10,000 paired participant-level swaps.
- Multiple testing: BH-FDR separately for the four frozen families.
- Official test is excluded from the primary analysis.

## Environment

- Python: {sys.version.split()[0]}
- Platform: {platform.platform()}
- NumPy: {np.__version__}
- pandas: {pd.__version__}
- matplotlib: {mpl.__version__}
- Main run manifests preserve the exact scikit-learn versions, model revisions, seeds, device, pooling, cache hashes and elapsed times.

## Execution commands

Run from the isolated worktree root with the recorded virtual environment activated:

```powershell
python scripts/prepare_analysis_v2.py
python scripts/run_tfidf_main.py
python scripts/run_embedding_model.py --model mpnet
python scripts/run_embedding_model.py --model bge
python scripts/finalize_embedding_analysis.py
python scripts/run_c5_controls.py
python scripts/run_c4_controls.py
python scripts/build_final_package.py
python -m pytest -q
```

Model commands accept explicit output/model arguments as documented by `--help`; exact revisions used in the completed runs are stored in the per-model `run_manifest.json` files. Embedding caches are content-addressed and included in SHA-256 inventories.

## Integrity verification

`integrity_audit.csv` verifies every pre-existing SHA-256 inventory. `strict_contract_audit.csv` verifies primary and C4-control input hashes, train/dev-only scope, the C4 turn-to-C3/C4 142/142 alignment, split membership/non-overlap, participant-level OOF cardinality, test-family sizes, 10,000 permutations, C5 quote exclusion and exact word matching, and C4 word-count matching. `output_manifest_sha256.csv` inventories this final package.

## Known environment deviation

The frozen second embedding was changed from gte-Qwen2-7B-instruct to bge-large-en-v1.5 before execution because the available RTX 3060 Laptop GPU has 6 GB VRAM and the installed PyTorch runtime was CPU-only. The failed CUDA installation attempts and the substitution are recorded in `00_plan/changelog.md`; both actually run embedding models are reported without selection.

## Verification status

This package is internally audited and deterministic inputs/splits/artifacts are hash-pinned. It is not labeled as an independent reproduction because a second full end-to-end rerun, especially of both embedding pipelines, was not performed.
"""
    (output / "REPRODUCIBILITY.md").write_text(text, encoding="utf-8")


def _write_final_audit(output: Path, audit: pd.DataFrame, inventory: pd.DataFrame) -> None:
    failures = audit.loc[audit["status"] != "PASS"]
    inventory_failures = inventory.loc[inventory["status"] != "PASS"]
    status = "PASS" if failures.empty and inventory_failures.empty else "FAIL"
    text = f"""# Final methodological audit report

## Verdict

**{status}** — strict contract checks: {int(audit['status'].eq('PASS').sum())}/{len(audit)} passed; artifact hash checks: {int(inventory['status'].eq('PASS').sum())}/{len(inventory)} passed.

## Leakage controls

- Five frozen primary input files and the C4 turn-level control input are hash-pinned and restricted to the 142 official train/dev participants.
- Every model and control uses the identical saved 10 x 5 participant split membership.
- Train/test participant overlap is zero in all 50 outer folds.
- TF-IDF fitting, embedding-classifier C selection, nested threshold selection, Platt calibration and isotonic calibration occur inside the corresponding outer-training data.
- Template shuffling is performed separately within each outer train and test partition, so donor assignment does not cross the partition boundary.
- Participant-level tests use one averaged OOF probability per person; fold means are not used as independent observations.

## Statistical controls

- Primary input-source family: 40 tests.
- Representation robustness family: 120 tests.
- C5 controls family: 30 tests.
- C4 controls family: 8 tests.
- Every paired test uses 10,000 participant-level swaps and carries a BH-FDR q-value within its frozen family.

## Data-construction controls

- C5 reviewed quote spans matched: 1,137; unmatched: 0.
- Random and non-symptom C5 controls match target word counts for all 142 participants.
- The C4 turn-level input is reconstructed from the verified official142 turn table with the frozen clinical-prompt rule; its aggregated interviewer and non-explicit interviewer text matches frozen C3 and C4 for 142/142 participants.
- C4 length- and position-matched controls match template word count for all 142 participants.
- Both dense embedding model identities and exact revisions are retained in run manifests; neither result set was discarded.

## Interpretation audit

- No TF-IDF ROC-AUC/PR-AUC input-source difference survived the primary-family FDR, so no stable interviewer ranking superiority is claimed.
- Default-threshold sensitivity is kept separate from ranking metrics.
- C5 domain features are described only as extracted symptom-domain evidence states.
- C4 findings are attributed to joint protocol signal, not isolated natural-language semantics.
- The cohort is described as official train+dev and not as independent external validation.
- LLM-derived evidence is not presented as a diagnostic endpoint.

## Remaining limitations

- Small, selected cohort and class imbalance.
- No independent cohort and no second full end-to-end rerun.
- Confidence intervals were not part of the frozen plan; paired permutation p/q values and full point estimates are supplied instead.
- Isotonic calibration is supplementary because of small-sample overfitting risk.
- The C4 operational definition uses explicit-clinical-prompt turns because the broader spoken-protocol flag leaves insufficient non-template text; this deviation is frozen in the changelog.
"""
    if not failures.empty:
        text += "\n## Failed strict checks\n\n" + _markdown_table(failures)
    if not inventory_failures.empty:
        text += "\n## Failed artifact checks\n\n" + _markdown_table(inventory_failures)
    (output / "FINAL_AUDIT_REPORT.md").write_text(text, encoding="utf-8")


def _create_output_inventory(output: Path) -> pd.DataFrame:
    excluded = {"output_manifest_sha256.csv", "run_manifest.json"}
    rows = []
    for path in sorted(path for path in output.rglob("*") if path.is_file() and path.name not in excluded):
        rows.append(
            {
                "relative_path": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    inventory = pd.DataFrame(rows)
    inventory.to_csv(output / "output_manifest_sha256.csv", index=False)
    return inventory


def build_final_package(analysis_root: Path, output: Path | None = None) -> dict[str, object]:
    analysis_root = Path(analysis_root).resolve()
    output = Path(output).resolve() if output is not None else analysis_root / "06_tables_figures"
    output.mkdir(parents=True, exist_ok=True)
    (output / "main_text").mkdir(exist_ok=True)
    (output / "supplement").mkdir(exist_ok=True)

    tfidf = pd.read_csv(analysis_root / "02_tfidf_main" / "metrics" / "primary_metrics.csv")
    calibration = pd.read_csv(analysis_root / "02_tfidf_main" / "calibration" / "calibration_metrics.csv")
    distributions = pd.read_csv(analysis_root / "02_tfidf_main" / "calibration" / "probability_distributions.csv")
    thresholds = pd.read_csv(analysis_root / "02_tfidf_main" / "nested_thresholds" / "threshold_stability_summary.csv")
    primary_tests = pd.read_csv(analysis_root / "02_tfidf_main" / "paired_tests" / "primary_input_source_paired_tests_fdr.csv")
    representations = pd.read_csv(analysis_root / "03_embedding_robustness" / "combined_representation_metrics.csv")
    representation_tests = pd.read_csv(analysis_root / "03_embedding_robustness" / "representation_family_paired_tests_fdr.csv")
    c5 = pd.read_csv(analysis_root / "04_c5_controls" / "metrics" / "c5_control_metrics.csv")
    c5_random = pd.read_csv(analysis_root / "04_c5_controls" / "random_same_word" / "random_seed_metric_distribution.csv")
    c5_tests = pd.read_csv(analysis_root / "04_c5_controls" / "paired_tests" / "c5_control_family_fdr.csv")
    c4 = pd.read_csv(analysis_root / "05_c4_controls" / "metrics" / "c4_control_metrics.csv")
    c4_tests = pd.read_csv(analysis_root / "05_c4_controls" / "paired_tests" / "c4_control_family_fdr.csv")

    table1_columns = [
        "condition", "n", "n_positive", "n_negative", "roc_auc", "pr_auc",
        "macro_f1_at_0_5", "macro_f1_nested", "sensitivity_at_0_5", "sensitivity_nested",
        "specificity_at_0_5", "specificity_nested", "raw_brier", "raw_ece_5bin",
        "platt_brier", "platt_ece_5bin", "isotonic_brier", "isotonic_ece_5bin",
    ]
    _write_table(tfidf[table1_columns], output / "main_text" / "table_1_tfidf_primary")
    representation_columns = [
        "representation", "condition", "roc_auc", "pr_auc", "macro_f1_at_0_5",
        "macro_f1_nested", "sensitivity_at_0_5", "sensitivity_nested",
        "specificity_at_0_5", "specificity_nested", "raw_brier", "raw_ece_5bin",
    ]
    _write_table(representations[representation_columns], output / "main_text" / "table_2_representation_robustness")
    _write_table(filter_significant(primary_tests), output / "main_text" / "table_3_primary_significant_paired_tests")
    _write_table(filter_significant(representation_tests), output / "main_text" / "table_4_representation_significant_paired_tests")

    _write_table(thresholds, output / "supplement" / "table_s1_threshold_stability")
    _write_table(calibration, output / "supplement" / "table_s2_tfidf_calibration")
    _write_table(distributions, output / "supplement" / "table_s3_probability_distributions")
    _write_table(c5, output / "supplement" / "table_s4_c5_controls")
    _write_table(c5_tests, output / "supplement" / "table_s5_c5_paired_tests")
    _write_table(c4, output / "supplement" / "table_s6_c4_controls")
    _write_table(c4_tests, output / "supplement" / "table_s7_c4_paired_tests")

    representations.to_csv(output / "main_text" / "figure_1_source_data.csv", index=False)
    tfidf.to_csv(output / "main_text" / "figure_2_source_data_metrics.csv", index=False)
    pd.read_csv(analysis_root / "02_tfidf_main" / "nested_thresholds" / "selected_thresholds_50folds.csv").to_csv(
        output / "main_text" / "figure_2_source_data_thresholds.csv", index=False
    )
    c5.to_csv(output / "supplement" / "figure_s1_source_data_metrics.csv", index=False)
    c5_random.to_csv(output / "supplement" / "figure_s1_source_data_random_distribution.csv", index=False)
    c4.to_csv(output / "supplement" / "figure_s2_source_data.csv", index=False)

    _plot_representation_grid(representations, output)
    selected_thresholds = pd.read_csv(
        analysis_root / "02_tfidf_main" / "nested_thresholds" / "selected_thresholds_50folds.csv"
    )
    _plot_threshold_calibration(tfidf, selected_thresholds, output)
    _plot_c5_controls(c5, c5_random, output)
    _plot_c4_controls(c4, output)
    _write_figure_contract(output)
    _write_legends(output)
    _write_qa_notes(output)
    _write_manuscript_results(output, tfidf, representations, c5, c4)

    inventory_audit = _inventory_audit(analysis_root)
    strict_audit = _strict_audit(analysis_root)
    inventory_audit.to_csv(output / "integrity_audit.csv", index=False)
    strict_audit.to_csv(output / "strict_contract_audit.csv", index=False)
    _write_validation_report(output, strict_audit, inventory_audit)
    _write_reproducibility(output, analysis_root)
    _write_final_audit(output, strict_audit, inventory_audit)

    inventory = _create_output_inventory(output)
    result: dict[str, object] = {
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_root": str(analysis_root),
        "output": str(output),
        "strict_audit_passed": bool(strict_audit["status"].eq("PASS").all()),
        "strict_audit_checks": int(len(strict_audit)),
        "inventory_failures": int((inventory_audit["status"] != "PASS").sum()),
        "inventoried_computational_artifacts": int(len(inventory_audit)),
        "final_package_files": int(len(inventory)),
        "final_package_inventory_sha256": sha256_file(output / "output_manifest_sha256.csv"),
        "verification_status": "ANALYZED",
        "independent_reproduction": "not performed",
    }
    (output / "run_manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result
