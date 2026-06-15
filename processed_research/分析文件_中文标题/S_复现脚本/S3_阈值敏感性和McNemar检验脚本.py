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
    make_linear_svm,
    make_logistic_regression,
    prepare_condition_frame,
)


BASE_DIR = Path("processed_research")
BASELINE_DIR = BASE_DIR / "baseline_results"
DEFAULT_CONDITIONS_CSV = (
    BASE_DIR / "conditions_long__RECOMMENDED_article-aligned_analysis-ready_C1-C4_no-C5.csv"
)
DEFAULT_OUTPUT_DIR = BASELINE_DIR
DEFAULT_RUN_ID = "threshold_mcc_tfidf_lr_svm_5fold_20260615"
DEFAULT_MCNEMAR_ID = "mcnemar_train_dev_20260615"

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
    "mpnet_logistic_regression:sentence-transformers/all-mpnet-base-v2": "MPNet + LR",
}
COMPARISONS = [
    ("full_dialogue", "participant_only", "Full vs Participant"),
    ("participant_only", "interviewer_only", "Participant vs Interviewer"),
    ("interviewer_only", "interviewer_cleaned", "Interviewer vs Cleaned"),
]


def model_short(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def best_threshold(
    y_true: Sequence[int],
    scores: Sequence[float],
    *,
    metric: str = "mcc",
) -> tuple[float, dict[str, float | int | None]]:
    y_true_arr = np.asarray(y_true, dtype=int)
    score_arr = np.asarray(scores, dtype=float)
    finite_scores = np.sort(np.unique(score_arr[np.isfinite(score_arr)]))
    if finite_scores.size == 0:
        raise ValueError("Cannot tune threshold with no finite scores.")

    candidates: list[float] = [float(finite_scores[0] - 1e-12)]
    if finite_scores.size > 1:
        candidates.extend(((finite_scores[:-1] + finite_scores[1:]) / 2).astype(float))
    candidates.append(float(finite_scores[-1] + 1e-12))

    best: tuple[float, dict[str, float | int | None]] | None = None
    best_value = -math.inf
    for threshold in candidates:
        pred = (score_arr >= threshold).astype(int)
        metrics = compute_binary_metrics(y_true_arr, pred, score_arr)
        value = metrics[metric]
        if value is None or pd.isna(value):
            value = -math.inf
        if float(value) > best_value:
            best = (float(threshold), metrics)
            best_value = float(value)
    if best is None:
        raise ValueError("No threshold candidate could be evaluated.")
    return best


def aggregate_cv(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    metric_cols = [
        "accuracy",
        "precision",
        "recall",
        "sensitivity",
        "specificity",
        "f1",
        "macro_f1",
        "auc",
        "mcc",
        "threshold",
    ]
    records: list[dict[str, object]] = []
    for (model, condition), g in df.groupby(["model", "input_condition"]):
        record: dict[str, object] = {
            "model": model,
            "model_short": model_short(model),
            "input_condition": condition,
            "condition_label": CONDITION_LABELS.get(condition, condition),
            "folds": int(g["fold"].nunique()),
            "mean_train_rows": float(g["train_rows"].mean()),
            "mean_test_rows": float(g["test_rows"].mean()),
            "threshold_metric": g["threshold_metric"].iloc[0],
        }
        for col in metric_cols:
            record[f"{col}_mean"] = float(g[col].mean())
            record[f"{col}_std"] = float(g[col].std(ddof=1))
        records.append(record)
    out = pd.DataFrame(records)
    out["condition_rank"] = out["input_condition"].map(
        {c: i for i, c in enumerate(CONDITION_ORDER)}
    )
    out["model_rank"] = out["model"].map(
        {"tfidf_logistic_regression": 0, "tfidf_linear_svm": 1}
    )
    return out.sort_values(["model_rank", "condition_rank"]).drop(
        columns=["model_rank", "condition_rank"]
    )


def fit_score_model(
    *,
    train_text: Sequence[str],
    train_label: Sequence[int],
    test_text: Sequence[str],
    model_name: str,
    random_state: int,
    max_features: int,
    min_df: int,
    ngram_range: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline

    if model_name == "tfidf_logistic_regression":
        classifier = make_logistic_regression(random_state=random_state)
    elif model_name == "tfidf_linear_svm":
        classifier = make_linear_svm(random_state=random_state)
    else:
        raise ValueError(f"Unsupported model for threshold tuning: {model_name}")

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=ngram_range,
                    min_df=min_df,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", classifier),
        ]
    )
    model.fit(list(train_text), np.asarray(train_label, dtype=int))
    if model_name == "tfidf_logistic_regression":
        train_score = model.predict_proba(list(train_text))[:, 1]
        test_score = model.predict_proba(list(test_text))[:, 1]
    else:
        train_score = model.decision_function(list(train_text))
        test_score = model.decision_function(list(test_text))
    return np.asarray(train_score, dtype=float), np.asarray(test_score, dtype=float)


def run_threshold_cv(
    *,
    df: pd.DataFrame,
    models: Sequence[str],
    conditions: Sequence[str],
    threshold_metric: str,
    cv_folds: int,
    random_state: int,
    max_features: int,
    min_df: int,
    ngram_range: tuple[int, int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import StratifiedKFold

    metric_rows: list[dict[str, object]] = []
    pred_rows: list[pd.DataFrame] = []
    for condition in conditions:
        cond = prepare_condition_frame(df, condition, drop_empty_text=True)
        y = cond["label"].to_numpy(dtype=int)
        splitter = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=random_state
        )
        for model_name in models:
            for fold, (train_idx, test_idx) in enumerate(splitter.split(cond, y), start=1):
                train = cond.iloc[train_idx].reset_index(drop=True)
                test = cond.iloc[test_idx].reset_index(drop=True)
                train_score, test_score = fit_score_model(
                    train_text=train["text"].tolist(),
                    train_label=train["label"].tolist(),
                    test_text=test["text"].tolist(),
                    model_name=model_name,
                    random_state=random_state,
                    max_features=max_features,
                    min_df=min_df,
                    ngram_range=ngram_range,
                )
                threshold, train_metrics = best_threshold(
                    train["label"].to_numpy(dtype=int),
                    train_score,
                    metric=threshold_metric,
                )
                y_pred = (test_score >= threshold).astype(int)
                test_metrics = compute_binary_metrics(
                    test["label"].to_numpy(dtype=int), y_pred, test_score
                )
                row: dict[str, object] = {
                    "model": model_name,
                    "model_short": model_short(model_name),
                    "input_condition": condition,
                    "condition_label": CONDITION_LABELS.get(condition, condition),
                    "evaluation": "5fold_cv_threshold_tuned",
                    "fold": fold,
                    "cv_folds": cv_folds,
                    "threshold_metric": threshold_metric,
                    "threshold": threshold,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "train_metric_at_threshold": train_metrics[threshold_metric],
                    "class_weight": "balanced",
                }
                row.update(test_metrics)
                metric_rows.append(row)

                pred = test[
                    ["participant_id", "split", "label", "phq_score", "input_condition"]
                ].copy()
                pred["model"] = model_name
                pred["model_short"] = model_short(model_name)
                pred["evaluation"] = "5fold_cv_threshold_tuned"
                pred["fold"] = fold
                pred["threshold_metric"] = threshold_metric
                pred["threshold"] = threshold
                pred["pred_score"] = test_score
                pred["pred_label"] = y_pred
                pred["correct"] = (pred["pred_label"] == pred["label"]).astype(int)
                pred_rows.append(pred)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(pred_rows, ignore_index=True)
    summary = aggregate_cv(metric_rows)
    return metrics, predictions, summary


def mcnemar_table(
    y_true: Sequence[int],
    pred_a: Sequence[int],
    pred_b: Sequence[int],
) -> dict[str, int]:
    y = np.asarray(y_true, dtype=int)
    a = np.asarray(pred_a, dtype=int)
    b = np.asarray(pred_b, dtype=int)
    a_correct = a == y
    b_correct = b == y
    return {
        "both_correct": int((a_correct & b_correct).sum()),
        "a_correct_b_wrong": int((a_correct & ~b_correct).sum()),
        "a_wrong_b_correct": int((~a_correct & b_correct).sum()),
        "both_wrong": int((~a_correct & ~b_correct).sum()),
    }


def mcnemar_exact_p(b: int, c: int) -> float:
    if b + c == 0:
        return 1.0
    try:
        from scipy.stats import binomtest

        return float(binomtest(min(b, c), n=b + c, p=0.5, alternative="two-sided").pvalue)
    except Exception:
        # Exact two-sided binomial probability fallback.
        n = b + c
        k = min(b, c)
        prob = sum(math.comb(n, i) * (0.5**n) for i in range(0, k + 1))
        return float(min(1.0, 2 * prob))


def run_mcnemar_tests(predictions_path: Path) -> pd.DataFrame:
    preds = pd.read_csv(predictions_path)
    rows: list[dict[str, object]] = []
    for model_short_name, g_model in preds.groupby("model_short"):
        for a_cond, b_cond, label in COMPARISONS:
            a = g_model[g_model["input_condition"] == a_cond][
                ["participant_id", "label", "pred_label"]
            ].rename(columns={"pred_label": "pred_a"})
            b = g_model[g_model["input_condition"] == b_cond][
                ["participant_id", "label", "pred_label"]
            ].rename(columns={"pred_label": "pred_b"})
            merged = a.merge(b, on=["participant_id", "label"], how="inner")
            if merged.empty:
                continue
            table = mcnemar_table(
                merged["label"].tolist(),
                merged["pred_a"].tolist(),
                merged["pred_b"].tolist(),
            )
            b_count = table["a_correct_b_wrong"]
            c_count = table["a_wrong_b_correct"]
            rows.append(
                {
                    "evaluation": "train_dev",
                    "model_short": model_short_name,
                    "comparison": label,
                    "condition_a": a_cond,
                    "condition_b": b_cond,
                    "paired_n": int(len(merged)),
                    **table,
                    "discordant_n": int(b_count + c_count),
                    "mcnemar_exact_p": mcnemar_exact_p(b_count, c_count),
                }
            )
    out = pd.DataFrame(rows)
    order = {
        "TF-IDF + LR balanced": 0,
        "TF-IDF + Linear SVM balanced": 1,
        "all-mpnet-base-v2 + LR balanced (CUDA)": 2,
    }
    out["model_rank"] = out["model_short"].map(order).fillna(99)
    out["comparison_rank"] = out["comparison"].map(
        {name: i for i, (*_, name) in enumerate(COMPARISONS)}
    )
    return out.sort_values(["model_rank", "comparison_rank"]).drop(
        columns=["model_rank", "comparison_rank"]
    )


def markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    rows = [[str(v) for v in row] for row in df.to_numpy().tolist()]
    widths = [
        max(len(str(col)), *(len(row[i]) for row in rows)) if rows else len(str(col))
        for i, col in enumerate(columns)
    ]

    def fmt_row(values: Sequence[object]) -> str:
        return "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(values)) + " |"

    sep = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([fmt_row(columns), sep] + [fmt_row(row) for row in rows])


def write_markdown_table(path: Path, title: str, df: pd.DataFrame) -> None:
    path.write_text(f"# {title}\n\n{markdown_table(df)}\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run threshold sensitivity analysis and McNemar tests."
    )
    parser.add_argument("--conditions-csv", type=Path, default=DEFAULT_CONDITIONS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--mcnemar-id", default=DEFAULT_MCNEMAR_ID)
    parser.add_argument("--threshold-metric", default="mcc", choices=["f1", "macro_f1", "mcc"])
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--tfidf-max-features", type=int, default=50000)
    parser.add_argument("--tfidf-min-df", type=int, default=2)
    parser.add_argument("--tfidf-ngram-min", type=int, default=1)
    parser.add_argument("--tfidf-ngram-max", type=int, default=2)
    parser.add_argument(
        "--train-dev-predictions",
        type=Path,
        default=BASELINE_DIR / "baseline_metrics__train_dev_combined_latest.csv",
        help="Unused placeholder kept for CLI clarity; train/dev predictions are discovered from individual files.",
    )
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_conditions(args.conditions_csv)
    metrics, predictions, summary = run_threshold_cv(
        df=df,
        models=["tfidf_logistic_regression", "tfidf_linear_svm"],
        conditions=CONDITION_ORDER,
        threshold_metric=args.threshold_metric,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        max_features=args.tfidf_max_features,
        min_df=args.tfidf_min_df,
        ngram_range=(args.tfidf_ngram_min, args.tfidf_ngram_max),
    )
    metrics_path = args.output_dir / f"threshold_metrics__{args.run_id}.csv"
    predictions_path = args.output_dir / f"threshold_predictions__{args.run_id}.csv"
    summary_path = args.output_dir / f"threshold_cv_summary__{args.run_id}.csv"
    summary_md_path = args.output_dir / f"threshold_cv_summary__{args.run_id}.md"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8")
    predictions.to_csv(predictions_path, index=False, encoding="utf-8")
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    write_markdown_table(summary_md_path, "Threshold-tuned CV summary", summary)

    # Build combined train/dev predictions on demand from the known individual prediction files.
    pred_files = [
        BASELINE_DIR / "baseline_predictions__tfidf_lr_balanced_20260615.csv",
        BASELINE_DIR / "baseline_predictions__tfidf_svm_balanced_20260615.csv",
        BASELINE_DIR / "baseline_predictions__mpnet_lr_balanced_cuda_20260615.csv",
    ]
    combined_predictions = pd.concat(
        [pd.read_csv(path) for path in pred_files if path.exists()], ignore_index=True
    )
    if "model_short" not in combined_predictions.columns:
        combined_predictions["model_short"] = combined_predictions["model"].map(
            {
                "tfidf_logistic_regression": "TF-IDF + LR balanced",
                "tfidf_linear_svm": "TF-IDF + Linear SVM balanced",
                "mpnet_logistic_regression:sentence-transformers/all-mpnet-base-v2": "all-mpnet-base-v2 + LR balanced (CUDA)",
            }
        )
    combined_pred_path = args.output_dir / f"mcnemar_input_predictions__{args.mcnemar_id}.csv"
    combined_predictions.to_csv(combined_pred_path, index=False, encoding="utf-8")
    mcnemar = run_mcnemar_tests(combined_pred_path)
    mcnemar_path = args.output_dir / f"mcnemar_tests__{args.mcnemar_id}.csv"
    mcnemar_md_path = args.output_dir / f"mcnemar_tests__{args.mcnemar_id}.md"
    mcnemar.to_csv(mcnemar_path, index=False, encoding="utf-8")
    write_markdown_table(mcnemar_md_path, "McNemar train/dev tests", mcnemar)

    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "threshold_metric": args.threshold_metric,
        "cv_folds": args.cv_folds,
        "class_weight": "balanced",
        "conditions_csv": str(args.conditions_csv),
        "outputs": {
            "threshold_metrics": str(metrics_path),
            "threshold_predictions": str(predictions_path),
            "threshold_summary": str(summary_path),
            "mcnemar_tests": str(mcnemar_path),
        },
    }
    config_path = args.output_dir / f"threshold_mcnemar_config__{args.run_id}.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Generated threshold and McNemar outputs:")
    for path in [
        metrics_path,
        predictions_path,
        summary_path,
        summary_md_path,
        combined_pred_path,
        mcnemar_path,
        mcnemar_md_path,
        config_path,
    ]:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
