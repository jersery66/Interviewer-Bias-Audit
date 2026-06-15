from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_text_baselines import (
    compute_binary_metrics,
    load_conditions,
    prepare_condition_frame,
)
from scripts.threshold_sensitivity_and_tests import fit_score_model


BASE_DIR = Path("processed_research")
BASELINE_DIR = BASE_DIR / "baseline_results"
PAPER_OUTPUT_DIR = BASE_DIR / "paper_outputs"
DEFAULT_OUTPUT_DIR = BASE_DIR / "robustness_outputs"
DEFAULT_CONDITIONS_CSV = (
    BASE_DIR / "conditions_long__RECOMMENDED_article-aligned_analysis-ready_C1-C4_no-C5.csv"
)
DEFAULT_EDAIC_PROCESSED_DIR = (
    Path("F:/") / "\u6570\u636e\u5e93" / "E-daic" / "processed_research"
)
DEFAULT_RUN_ID = "nested_oracle_tfidf_lr_svm_5fold_20260615"

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
MODEL_LABELS = {
    "tfidf_logistic_regression": "TF-IDF + LR",
    "tfidf_linear_svm": "TF-IDF + SVM",
}


def model_short(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    columns = [str(c) for c in df.columns]
    rows = [[str(v) for v in row] for row in df.to_numpy().tolist()]
    widths = [
        max(len(col), *(len(row[i]) for row in rows)) for i, col in enumerate(columns)
    ]

    def fmt_row(values: Sequence[object]) -> str:
        return "| " + " | ".join(str(values[i]).ljust(widths[i]) for i in range(len(values))) + " |"

    sep = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([fmt_row(columns), sep] + [fmt_row(row) for row in rows])


def rule_catalog_frame(rules_json: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for rule in rules_json.get("rules", []):
        if not isinstance(rule, dict):
            continue
        rows.append(
            {
                "rule_id": str(rule.get("rule_id", "")),
                "description": str(rule.get("description", "")),
                "patterns": "; ".join(str(p) for p in rule.get("patterns", [])),
            }
        )
    return pd.DataFrame(rows)


def summarize_cleaning_audit(audit: pd.DataFrame) -> dict[str, float | int]:
    removed_turns = pd.to_numeric(
        audit["diagnostic_removed_turn_count"], errors="coerce"
    ).fillna(0)
    retained_turns = pd.to_numeric(audit["retained_turn_count"], errors="coerce").fillna(0)
    removed_words = pd.to_numeric(
        audit["diagnostic_removed_word_count"], errors="coerce"
    ).fillna(0)
    retained_words = pd.to_numeric(audit["retained_word_count"], errors="coerce").fillna(0)
    total_turns = removed_turns + retained_turns
    total_words = removed_words + retained_words
    return {
        "participants": int(len(audit)),
        "participants_with_removed": int((removed_turns > 0).sum()),
        "removed_turns": int(removed_turns.sum()),
        "retained_turns": int(retained_turns.sum()),
        "removed_words": int(removed_words.sum()),
        "retained_words": int(retained_words.sum()),
        "mean_removed_turns": float(removed_turns.mean()),
        "median_removed_turns": float(removed_turns.median()),
        "mean_turn_removal_ratio": float((removed_turns / total_turns.replace(0, np.nan)).mean()),
        "mean_word_removal_ratio": float((removed_words / total_words.replace(0, np.nan)).mean()),
    }


def build_manual_review_template(
    review: pd.DataFrame,
    *,
    random_state: int = 42,
    max_rows: int = 120,
) -> pd.DataFrame:
    df = review.copy()
    df["expected_rule_decision"] = np.where(
        df["is_removed_diagnostic_prompt"].astype(bool), "remove", "retain"
    )
    if len(df) > max_rows:
        buckets = [
            (df["sample_reason"].eq("focus_pattern") & df["is_removed_diagnostic_prompt"].astype(bool), 50),
            (df["sample_reason"].eq("focus_pattern") & ~df["is_removed_diagnostic_prompt"].astype(bool), 20),
            (df["sample_reason"].eq("random_removed_nonfocus"), 25),
            (df["sample_reason"].eq("random_retained_nonfocus"), 25),
        ]
        sampled: list[pd.DataFrame] = []
        for mask, n in buckets:
            group = df[mask].copy()
            if group.empty:
                continue
            sampled.append(
                group.sample(n=min(n, len(group)), random_state=random_state)
                if len(group) > n
                else group
            )
        df = pd.concat(sampled, ignore_index=True)
        if len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)
    df = df.sort_values(
        ["expected_rule_decision", "sample_reason", "participant_id", "turn_index"]
    ).reset_index(drop=True)
    df["human_review_decision"] = ""
    df["reviewer_notes"] = ""
    keep_cols = [
        "participant_id",
        "split",
        "turn_index",
        "text",
        "expected_rule_decision",
        "is_removed_diagnostic_prompt",
        "matched_rule_ids",
        "review_focus_hits",
        "sample_reason",
        "human_review_decision",
        "reviewer_notes",
    ]
    for col in keep_cols:
        if col not in df.columns:
            df[col] = ""
    return df[keep_cols]


def oracle_best_threshold(
    y_true: Sequence[int],
    scores: Sequence[float],
    *,
    metric: str = "mcc",
) -> tuple[float, dict[str, float | int | None]]:
    y = np.asarray(y_true, dtype=int)
    score_arr = np.asarray(scores, dtype=float)
    finite_scores = np.sort(np.unique(score_arr[np.isfinite(score_arr)]))
    if finite_scores.size == 0:
        raise ValueError("No finite scores available for threshold search.")

    candidates: list[float] = [float(finite_scores[0] - 1e-12)]
    if finite_scores.size > 1:
        candidates.extend(((finite_scores[:-1] + finite_scores[1:]) / 2).astype(float))
    candidates.append(float(finite_scores[-1] + 1e-12))

    best_threshold = candidates[0]
    best_metrics = compute_binary_metrics(y, score_arr >= best_threshold, score_arr)
    best_value = best_metrics.get(metric)
    best_value_float = -math.inf if best_value is None or pd.isna(best_value) else float(best_value)
    for threshold in candidates[1:]:
        pred = (score_arr >= threshold).astype(int)
        metrics = compute_binary_metrics(y, pred, score_arr)
        value = metrics.get(metric)
        value_float = -math.inf if value is None or pd.isna(value) else float(value)
        if value_float > best_value_float:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_value_float = value_float
    return float(best_threshold), best_metrics


def run_nested_oracle_threshold_cv(
    *,
    df: pd.DataFrame,
    models: Sequence[str],
    conditions: Sequence[str],
    outer_folds: int,
    inner_folds: int,
    threshold_metric: str,
    random_state: int,
    max_features: int,
    min_df: int,
    ngram_range: tuple[int, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import StratifiedKFold

    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for condition in conditions:
        cond = prepare_condition_frame(df, condition, drop_empty_text=True)
        y = cond["label"].to_numpy(dtype=int)
        outer = StratifiedKFold(
            n_splits=outer_folds, shuffle=True, random_state=random_state
        )
        for model_name in models:
            for fold, (train_idx, test_idx) in enumerate(outer.split(cond, y), start=1):
                train = cond.iloc[train_idx].reset_index(drop=True)
                test = cond.iloc[test_idx].reset_index(drop=True)
                y_train = train["label"].to_numpy(dtype=int)
                inner = StratifiedKFold(
                    n_splits=inner_folds, shuffle=True, random_state=random_state + fold
                )
                oof_scores = np.zeros(len(train), dtype=float)
                for inner_train_idx, inner_cal_idx in inner.split(train, y_train):
                    inner_train = train.iloc[inner_train_idx].reset_index(drop=True)
                    inner_cal = train.iloc[inner_cal_idx].reset_index(drop=True)
                    _, cal_score = fit_score_model(
                        train_text=inner_train["text"].tolist(),
                        train_label=inner_train["label"].tolist(),
                        test_text=inner_cal["text"].tolist(),
                        model_name=model_name,
                        random_state=random_state,
                        max_features=max_features,
                        min_df=min_df,
                        ngram_range=ngram_range,
                    )
                    oof_scores[inner_cal_idx] = cal_score

                nested_threshold, nested_train_metrics = oracle_best_threshold(
                    y_train, oof_scores, metric=threshold_metric
                )
                _, test_score = fit_score_model(
                    train_text=train["text"].tolist(),
                    train_label=train["label"].tolist(),
                    test_text=test["text"].tolist(),
                    model_name=model_name,
                    random_state=random_state,
                    max_features=max_features,
                    min_df=min_df,
                    ngram_range=ngram_range,
                )
                y_test = test["label"].to_numpy(dtype=int)
                nested_pred = (test_score >= nested_threshold).astype(int)
                nested_metrics = compute_binary_metrics(y_test, nested_pred, test_score)
                oracle_threshold, oracle_metrics = oracle_best_threshold(
                    y_test, test_score, metric=threshold_metric
                )

                for evaluation, threshold, metrics, train_metric in [
                    (
                        "nested_threshold",
                        nested_threshold,
                        nested_metrics,
                        nested_train_metrics[threshold_metric],
                    ),
                    (
                        "oracle_test_threshold",
                        oracle_threshold,
                        oracle_metrics,
                        oracle_metrics[threshold_metric],
                    ),
                ]:
                    row: dict[str, object] = {
                        "evaluation": evaluation,
                        "model": model_name,
                        "model_short": model_short(model_name),
                        "input_condition": condition,
                        "condition_label": CONDITION_LABELS.get(condition, condition),
                        "fold": fold,
                        "outer_folds": outer_folds,
                        "inner_folds": inner_folds if evaluation == "nested_threshold" else "",
                        "threshold_metric": threshold_metric,
                        "threshold": threshold,
                        "train_or_oracle_metric_at_threshold": train_metric,
                        "train_rows": len(train),
                        "test_rows": len(test),
                        "class_weight": "balanced",
                    }
                    row.update(metrics)
                    metric_rows.append(row)

                pred = test[
                    ["participant_id", "split", "label", "phq_score", "input_condition"]
                ].copy()
                pred["model"] = model_name
                pred["model_short"] = model_short(model_name)
                pred["fold"] = fold
                pred["nested_threshold"] = nested_threshold
                pred["oracle_threshold"] = oracle_threshold
                pred["pred_score"] = test_score
                pred["nested_pred_label"] = nested_pred
                pred["oracle_pred_label"] = (test_score >= oracle_threshold).astype(int)
                prediction_rows.append(pred)
    return pd.DataFrame(metric_rows), pd.concat(prediction_rows, ignore_index=True)


def aggregate_threshold_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "macro_f1",
        "auc",
        "mcc",
        "threshold",
        "tp",
        "fp",
        "fn",
        "tn",
    ]
    rows: list[dict[str, object]] = []
    for (evaluation, model, condition), group in metrics.groupby(
        ["evaluation", "model", "input_condition"]
    ):
        row: dict[str, object] = {
            "evaluation": evaluation,
            "model": model,
            "model_short": model_short(model),
            "input_condition": condition,
            "condition_label": CONDITION_LABELS.get(condition, condition),
            "folds": int(group["fold"].nunique()),
            "mean_train_rows": float(group["train_rows"].mean()),
            "mean_test_rows": float(group["test_rows"].mean()),
            "threshold_metric": group["threshold_metric"].iloc[0],
        }
        for col in metric_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=1))
            if col in {"tp", "fp", "fn", "tn"}:
                row[f"{col}_sum"] = int(group[col].sum())
        rows.append(row)
    out = pd.DataFrame(rows)
    out["evaluation_rank"] = out["evaluation"].map(
        {"nested_threshold": 0, "oracle_test_threshold": 1}
    )
    out["model_rank"] = out["model"].map(
        {"tfidf_logistic_regression": 0, "tfidf_linear_svm": 1}
    )
    out["condition_rank"] = out["input_condition"].map(
        {c: i for i, c in enumerate(CONDITION_ORDER)}
    )
    return out.sort_values(["evaluation_rank", "model_rank", "condition_rank"]).drop(
        columns=["evaluation_rank", "model_rank", "condition_rank"]
    )


def write_appendix(
    *,
    output_dir: Path,
    rules_json: dict[str, object],
    rule_catalog: pd.DataFrame,
    audit_summary: dict[str, float | int],
    manual_template_path: Path,
    audit_path: Path,
) -> Path:
    description = str(rules_json.get("description", ""))
    rule_version = str(rules_json.get("rule_version", ""))
    summary_rows = pd.DataFrame(
        [
            {"Item": key, "Value": value}
            for key, value in audit_summary.items()
        ]
    )
    sections = [
        "# Appendix A. C4 interviewer-cleaned cleaning rule and audit trail",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Rule version: `{rule_version}`",
        "",
        description,
        "",
        "## Cleaning summary",
        "",
        markdown_table(summary_rows),
        "",
        "## Rule catalogue",
        "",
        markdown_table(rule_catalog),
        "",
        "## Manual review template",
        "",
        "The review template samples removed and retained interviewer turns. It includes blank `human_review_decision` and `reviewer_notes` columns so a human reviewer can mark correct removals, false positives, false negatives, or unclear cases without altering the modeling files.",
        "",
        f"- Audit source: `{audit_path}`",
        f"- Manual review template: `{manual_template_path}`",
        "",
        "## Interpretation boundary",
        "",
        "The C4 condition removes direct symptom and mental-health history probes from interviewer speech. It does not remove all possible interactional signals. Therefore, C4 should be interpreted as a diagnostic-prompt reduction condition, not as a complete removal of interviewer-side information.",
        "",
    ]
    path = output_dir / "appendix_A_C4_cleaning_rules_and_audit.md"
    path.write_text("\n".join(sections), encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate C4 cleaning appendix and nested/oracle threshold supplement."
    )
    parser.add_argument("--conditions-csv", type=Path, default=DEFAULT_CONDITIONS_CSV)
    parser.add_argument("--edaic-processed-dir", type=Path, default=DEFAULT_EDAIC_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--threshold-metric", default="mcc", choices=["f1", "macro_f1", "mcc"])
    parser.add_argument("--tfidf-max-features", type=int, default=50000)
    parser.add_argument("--tfidf-min-df", type=int, default=2)
    parser.add_argument("--tfidf-ngram-min", type=int, default=1)
    parser.add_argument("--tfidf-ngram-max", type=int, default=2)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rules_path = args.edaic_processed_dir / "interviewer_cleaning_rules_article_aligned.json"
    audit_path = args.edaic_processed_dir / "interviewer_cleaning_audit_article_aligned.csv"
    review_path = args.edaic_processed_dir / "interviewer_cleaning_review_sample_article_aligned.csv"

    rules_json = json.loads(rules_path.read_text(encoding="utf-8"))
    rule_catalog = rule_catalog_frame(rules_json)
    audit = pd.read_csv(audit_path)
    audit_summary = summarize_cleaning_audit(audit)
    review = pd.read_csv(review_path)
    manual_template = build_manual_review_template(review, random_state=args.random_state)

    rule_catalog_path = args.output_dir / "appendix_A_C4_cleaning_rule_catalog.csv"
    audit_copy_path = args.output_dir / "appendix_A_C4_cleaning_audit_summary.csv"
    manual_template_path = args.output_dir / "appendix_A_C4_manual_review_template.csv"
    rule_catalog.to_csv(rule_catalog_path, index=False, encoding="utf-8-sig")
    pd.DataFrame([audit_summary]).to_csv(audit_copy_path, index=False, encoding="utf-8-sig")
    manual_template.to_csv(manual_template_path, index=False, encoding="utf-8-sig")
    appendix_path = write_appendix(
        output_dir=args.output_dir,
        rules_json=rules_json,
        rule_catalog=rule_catalog,
        audit_summary=audit_summary,
        manual_template_path=manual_template_path,
        audit_path=audit_path,
    )

    df = load_conditions(args.conditions_csv)
    metrics, predictions = run_nested_oracle_threshold_cv(
        df=df,
        models=["tfidf_logistic_regression", "tfidf_linear_svm"],
        conditions=CONDITION_ORDER,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        threshold_metric=args.threshold_metric,
        random_state=args.random_state,
        max_features=args.tfidf_max_features,
        min_df=args.tfidf_min_df,
        ngram_range=(args.tfidf_ngram_min, args.tfidf_ngram_max),
    )
    summary = aggregate_threshold_rows(metrics)
    metrics_path = args.output_dir / f"threshold_robustness_metrics__{args.run_id}.csv"
    predictions_path = args.output_dir / f"threshold_robustness_predictions__{args.run_id}.csv"
    summary_path = args.output_dir / f"threshold_robustness_summary__{args.run_id}.csv"
    summary_md_path = args.output_dir / f"threshold_robustness_summary__{args.run_id}.md"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8")
    predictions.to_csv(predictions_path, index=False, encoding="utf-8")
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    summary_md_path.write_text(
        "# Nested and oracle threshold robustness summary\n\n"
        + markdown_table(summary)
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": args.run_id,
        "rule_version": rules_json.get("rule_version"),
        "threshold_metric": args.threshold_metric,
        "outputs": {
            "appendix": str(appendix_path),
            "rule_catalog": str(rule_catalog_path),
            "audit_summary": str(audit_copy_path),
            "manual_review_template": str(manual_template_path),
            "threshold_metrics": str(metrics_path),
            "threshold_predictions": str(predictions_path),
            "threshold_summary": str(summary_path),
            "threshold_summary_md": str(summary_md_path),
        },
    }
    manifest_path = args.output_dir / f"robustness_manifest__{args.run_id}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Generated robustness appendix and threshold outputs:")
    for path in [
        appendix_path,
        rule_catalog_path,
        audit_copy_path,
        manual_template_path,
        metrics_path,
        predictions_path,
        summary_path,
        summary_md_path,
        manifest_path,
    ]:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
