from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_CONDITIONS_CSV = (
    Path("processed_research")
    / "conditions_long__RECOMMENDED_article-aligned_analysis-ready_C1-C4_no-C5.csv"
)
DEFAULT_OUTPUT_DIR = Path("processed_research") / "baseline_results"
DEFAULT_MPNET_MODEL = "sentence-transformers/all-mpnet-base-v2"
SUMMARY_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "sensitivity",
    "specificity",
    "f1",
    "macro_f1",
    "auc",
    "mcc",
]


def parse_condition_list(raw: str | None) -> list[str] | None:
    if raw is None or raw.strip().lower() in {"", "all"}:
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def text_sha1(values: Iterable[str]) -> str:
    digest = hashlib.sha1()
    for value in values:
        digest.update(str(value).encode("utf-8", errors="replace"))
        digest.update(b"\0")
    return digest.hexdigest()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def load_conditions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"participant_id", "split", "label", "input_condition", "text"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    out = df.copy()
    out["participant_id"] = out["participant_id"].astype(str)
    out["split"] = out["split"].astype(str).str.lower()
    out["label"] = pd.to_numeric(out["label"], errors="raise").astype(int)
    out["text"] = out["text"].fillna("").astype(str)
    return out


def prepare_condition_frame(
    df: pd.DataFrame,
    input_condition: str,
    *,
    drop_empty_text: bool = True,
) -> pd.DataFrame:
    out = df[df["input_condition"] == input_condition].copy()
    out["text"] = out["text"].fillna("").astype(str)
    if drop_empty_text:
        out = out[out["text"].str.strip() != ""].copy()
    out = out.sort_values(["split", "participant_id"]).reset_index(drop=True)
    return out


def train_dev_split(
    df: pd.DataFrame,
    *,
    train_split: str = "train",
    test_split: str = "dev",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["split"] == train_split].copy()
    test = df[df["split"] == test_split].copy()
    if train.empty:
        raise ValueError(f"No rows found for train split {train_split!r}")
    if test.empty:
        raise ValueError(f"No rows found for test split {test_split!r}")
    if train["label"].nunique() < 2:
        raise ValueError("Train split must contain both classes.")
    return train.reset_index(drop=True), test.reset_index(drop=True)


def make_logistic_regression(random_state: int = 42):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        solver="liblinear",
        random_state=random_state,
    )


def make_linear_svm(random_state: int = 42):
    from sklearn.svm import LinearSVC

    return LinearSVC(
        class_weight="balanced",
        max_iter=10000,
        random_state=random_state,
    )


def compute_binary_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Sequence[float] | None = None,
) -> dict[str, float | int | None]:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else math.nan
    sensitivity = tp / (tp + fn) if (tp + fn) else math.nan
    auc = math.nan
    if y_score is not None and len(np.unique(y_true_arr)) == 2:
        auc = float(roc_auc_score(y_true_arr, np.asarray(y_score, dtype=float)))

    return {
        "n": int(len(y_true_arr)),
        "positive_n": int((y_true_arr == 1).sum()),
        "negative_n": int((y_true_arr == 0).sum()),
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(
            precision_score(y_true_arr, y_pred_arr, zero_division=0)
        ),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        "macro_f1": float(
            f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        ),
        "auc": float(auc) if not math.isnan(auc) else math.nan,
        "mcc": float(matthews_corrcoef(y_true_arr, y_pred_arr)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def aggregate_cv_metric_rows(
    rows: list[dict[str, float | int | str | None]],
) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()
    if "evaluation" in df.columns:
        df = df[df["evaluation"] == "cv"].copy()
    if df.empty:
        return pd.DataFrame()

    group_cols = ["model", "input_condition"]
    if "class_weight" in df.columns:
        group_cols.append("class_weight")
    records: list[dict[str, float | int | str]] = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = {col: key for col, key in zip(group_cols, keys)}
        record["folds"] = int(group["fold"].nunique()) if "fold" in group else len(group)
        if "train_rows" in group.columns:
            record["mean_train_rows"] = float(group["train_rows"].mean())
        if "test_rows" in group.columns:
            record["mean_test_rows"] = float(group["test_rows"].mean())
        for metric in SUMMARY_METRICS:
            if metric in group.columns:
                record[f"{metric}_mean"] = float(group[metric].mean())
                record[f"{metric}_std"] = float(group[metric].std(ddof=1))
        records.append(record)
    return pd.DataFrame(records)


def add_common_metric_fields(
    metrics: dict[str, float | int | None],
    *,
    model_name: str,
    input_condition: str,
    train_rows: int,
    test_rows: int,
    dropped_rows: int,
    notes: str,
) -> dict[str, float | int | str | None]:
    out: dict[str, float | int | str | None] = {
        "model": model_name,
        "input_condition": input_condition,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "dropped_empty_rows": dropped_rows,
        "class_weight": "balanced",
        "notes": notes,
    }
    out.update(metrics)
    return out


def prediction_frame(
    test: pd.DataFrame,
    *,
    model_name: str,
    input_condition: str,
    y_pred: Sequence[int],
    y_score: Sequence[float] | None,
) -> pd.DataFrame:
    out = test[["participant_id", "split", "label", "phq_score", "input_condition"]].copy()
    out["model"] = model_name
    out["input_condition"] = input_condition
    out["pred_label"] = np.asarray(y_pred, dtype=int)
    out["pred_score"] = np.asarray(y_score, dtype=float) if y_score is not None else np.nan
    out["correct"] = (out["label"] == out["pred_label"]).astype(int)
    return out


def run_tfidf_lr(
    df: pd.DataFrame,
    *,
    input_condition: str,
    random_state: int,
    max_features: int,
    min_df: int,
    ngram_range: tuple[int, int],
    train_split: str,
    test_split: str,
    drop_empty_text: bool,
) -> tuple[dict[str, float | int | str | None], pd.DataFrame]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline

    all_condition_rows = df[df["input_condition"] == input_condition]
    cond = prepare_condition_frame(
        df, input_condition, drop_empty_text=drop_empty_text
    )
    dropped_rows = int(len(all_condition_rows) - len(cond))
    train, test = train_dev_split(
        cond, train_split=train_split, test_split=test_split
    )
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
            ("lr", make_logistic_regression(random_state=random_state)),
        ]
    )
    model.fit(train["text"].tolist(), train["label"].to_numpy())
    y_pred = model.predict(test["text"].tolist())
    y_score = model.predict_proba(test["text"].tolist())[:, 1]
    metrics = compute_binary_metrics(test["label"].to_numpy(), y_pred, y_score)
    metrics = add_common_metric_fields(
        metrics,
        model_name="tfidf_logistic_regression",
        input_condition=input_condition,
        train_rows=len(train),
        test_rows=len(test),
        dropped_rows=dropped_rows,
        notes=f"TF-IDF max_features={max_features}, min_df={min_df}, ngram_range={ngram_range}",
    )
    preds = prediction_frame(
        test,
        model_name="tfidf_logistic_regression",
        input_condition=input_condition,
        y_pred=y_pred,
        y_score=y_score,
    )
    return metrics, preds


def run_tfidf_lr_cv(
    df: pd.DataFrame,
    *,
    input_condition: str,
    random_state: int,
    max_features: int,
    min_df: int,
    ngram_range: tuple[int, int],
    cv_folds: int,
    drop_empty_text: bool,
) -> tuple[list[dict[str, float | int | str | None]], pd.DataFrame]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline

    all_condition_rows = df[df["input_condition"] == input_condition]
    cond = prepare_condition_frame(
        df, input_condition, drop_empty_text=drop_empty_text
    )
    dropped_rows = int(len(all_condition_rows) - len(cond))
    splitter = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=random_state
    )
    metrics_rows: list[dict[str, float | int | str | None]] = []
    prediction_frames: list[pd.DataFrame] = []
    y = cond["label"].to_numpy()
    for fold, (train_idx, test_idx) in enumerate(splitter.split(cond, y), start=1):
        train = cond.iloc[train_idx].reset_index(drop=True)
        test = cond.iloc[test_idx].reset_index(drop=True)
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
                ("lr", make_logistic_regression(random_state=random_state)),
            ]
        )
        model.fit(train["text"].tolist(), train["label"].to_numpy())
        y_pred = model.predict(test["text"].tolist())
        y_score = model.predict_proba(test["text"].tolist())[:, 1]
        metrics = compute_binary_metrics(test["label"].to_numpy(), y_pred, y_score)
        metrics = add_common_metric_fields(
            metrics,
            model_name="tfidf_logistic_regression",
            input_condition=input_condition,
            train_rows=len(train),
            test_rows=len(test),
            dropped_rows=dropped_rows,
            notes=f"5-fold CV TF-IDF max_features={max_features}, min_df={min_df}, ngram_range={ngram_range}",
        )
        metrics["evaluation"] = "cv"
        metrics["fold"] = fold
        metrics["cv_folds"] = cv_folds
        metrics_rows.append(metrics)
        preds = prediction_frame(
            test,
            model_name="tfidf_logistic_regression",
            input_condition=input_condition,
            y_pred=y_pred,
            y_score=y_score,
        )
        preds["evaluation"] = "cv"
        preds["fold"] = fold
        prediction_frames.append(preds)
    return metrics_rows, pd.concat(prediction_frames, ignore_index=True)


def run_tfidf_svm(
    df: pd.DataFrame,
    *,
    input_condition: str,
    random_state: int,
    max_features: int,
    min_df: int,
    ngram_range: tuple[int, int],
    train_split: str,
    test_split: str,
    drop_empty_text: bool,
) -> tuple[dict[str, float | int | str | None], pd.DataFrame]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline

    all_condition_rows = df[df["input_condition"] == input_condition]
    cond = prepare_condition_frame(
        df, input_condition, drop_empty_text=drop_empty_text
    )
    dropped_rows = int(len(all_condition_rows) - len(cond))
    train, test = train_dev_split(
        cond, train_split=train_split, test_split=test_split
    )
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
            ("svm", make_linear_svm(random_state=random_state)),
        ]
    )
    model.fit(train["text"].tolist(), train["label"].to_numpy())
    y_pred = model.predict(test["text"].tolist())
    y_score = model.decision_function(test["text"].tolist())
    metrics = compute_binary_metrics(test["label"].to_numpy(), y_pred, y_score)
    metrics = add_common_metric_fields(
        metrics,
        model_name="tfidf_linear_svm",
        input_condition=input_condition,
        train_rows=len(train),
        test_rows=len(test),
        dropped_rows=dropped_rows,
        notes=f"TF-IDF + LinearSVC max_features={max_features}, min_df={min_df}, ngram_range={ngram_range}",
    )
    preds = prediction_frame(
        test,
        model_name="tfidf_linear_svm",
        input_condition=input_condition,
        y_pred=y_pred,
        y_score=y_score,
    )
    return metrics, preds


def run_tfidf_svm_cv(
    df: pd.DataFrame,
    *,
    input_condition: str,
    random_state: int,
    max_features: int,
    min_df: int,
    ngram_range: tuple[int, int],
    cv_folds: int,
    drop_empty_text: bool,
) -> tuple[list[dict[str, float | int | str | None]], pd.DataFrame]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline

    all_condition_rows = df[df["input_condition"] == input_condition]
    cond = prepare_condition_frame(
        df, input_condition, drop_empty_text=drop_empty_text
    )
    dropped_rows = int(len(all_condition_rows) - len(cond))
    splitter = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=random_state
    )
    metrics_rows: list[dict[str, float | int | str | None]] = []
    prediction_frames: list[pd.DataFrame] = []
    y = cond["label"].to_numpy()
    for fold, (train_idx, test_idx) in enumerate(splitter.split(cond, y), start=1):
        train = cond.iloc[train_idx].reset_index(drop=True)
        test = cond.iloc[test_idx].reset_index(drop=True)
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
                ("svm", make_linear_svm(random_state=random_state)),
            ]
        )
        model.fit(train["text"].tolist(), train["label"].to_numpy())
        y_pred = model.predict(test["text"].tolist())
        y_score = model.decision_function(test["text"].tolist())
        metrics = compute_binary_metrics(test["label"].to_numpy(), y_pred, y_score)
        metrics = add_common_metric_fields(
            metrics,
            model_name="tfidf_linear_svm",
            input_condition=input_condition,
            train_rows=len(train),
            test_rows=len(test),
            dropped_rows=dropped_rows,
            notes=f"5-fold CV TF-IDF + LinearSVC max_features={max_features}, min_df={min_df}, ngram_range={ngram_range}",
        )
        metrics["evaluation"] = "cv"
        metrics["fold"] = fold
        metrics["cv_folds"] = cv_folds
        metrics_rows.append(metrics)
        preds = prediction_frame(
            test,
            model_name="tfidf_linear_svm",
            input_condition=input_condition,
            y_pred=y_pred,
            y_score=y_score,
        )
        preds["evaluation"] = "cv"
        preds["fold"] = fold
        prediction_frames.append(preds)
    return metrics_rows, pd.concat(prediction_frames, ignore_index=True)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def encode_texts_with_cache(
    *,
    texts: Sequence[str],
    participant_ids: Sequence[str],
    input_condition: str,
    model_name: str,
    batch_size: int,
    device: str,
    cache_dir: Path,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    cache_dir.mkdir(parents=True, exist_ok=True)
    key = text_sha1(
        [
            model_name,
            input_condition,
            device,
            *participant_ids,
            *texts,
        ]
    )[:16]
    cache_path = cache_dir / f"embeddings__{safe_name(model_name)}__{safe_name(input_condition)}__{key}.npz"
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        return cached["embeddings"]

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    np.savez_compressed(cache_path, embeddings=embeddings.astype(np.float32))
    return embeddings.astype(np.float32)


def run_mpnet_lr(
    df: pd.DataFrame,
    *,
    input_condition: str,
    random_state: int,
    mpnet_model: str,
    batch_size: int,
    device: str,
    cache_dir: Path,
    train_split: str,
    test_split: str,
    drop_empty_text: bool,
) -> tuple[dict[str, float | int | str | None], pd.DataFrame]:
    all_condition_rows = df[df["input_condition"] == input_condition]
    cond = prepare_condition_frame(
        df, input_condition, drop_empty_text=drop_empty_text
    )
    dropped_rows = int(len(all_condition_rows) - len(cond))
    train, test = train_dev_split(
        cond, train_split=train_split, test_split=test_split
    )
    ordered = pd.concat([train, test], ignore_index=True)
    embeddings = encode_texts_with_cache(
        texts=ordered["text"].tolist(),
        participant_ids=ordered["participant_id"].tolist(),
        input_condition=input_condition,
        model_name=mpnet_model,
        batch_size=batch_size,
        device=device,
        cache_dir=cache_dir,
    )
    x_train = embeddings[: len(train)]
    x_test = embeddings[len(train) :]
    model = make_logistic_regression(random_state=random_state)
    model.fit(x_train, train["label"].to_numpy())
    y_pred = model.predict(x_test)
    y_score = model.predict_proba(x_test)[:, 1]
    metrics = compute_binary_metrics(test["label"].to_numpy(), y_pred, y_score)
    metrics = add_common_metric_fields(
        metrics,
        model_name=f"mpnet_logistic_regression:{mpnet_model}",
        input_condition=input_condition,
        train_rows=len(train),
        test_rows=len(test),
        dropped_rows=dropped_rows,
        notes=f"SentenceTransformer embeddings, batch_size={batch_size}, device={device}",
    )
    preds = prediction_frame(
        test,
        model_name=f"mpnet_logistic_regression:{mpnet_model}",
        input_condition=input_condition,
        y_pred=y_pred,
        y_score=y_score,
    )
    return metrics, preds


def run_mpnet_lr_cv(
    df: pd.DataFrame,
    *,
    input_condition: str,
    random_state: int,
    mpnet_model: str,
    batch_size: int,
    device: str,
    cache_dir: Path,
    cv_folds: int,
    drop_empty_text: bool,
) -> tuple[list[dict[str, float | int | str | None]], pd.DataFrame]:
    from sklearn.model_selection import StratifiedKFold

    all_condition_rows = df[df["input_condition"] == input_condition]
    cond = prepare_condition_frame(
        df, input_condition, drop_empty_text=drop_empty_text
    )
    dropped_rows = int(len(all_condition_rows) - len(cond))
    embeddings = encode_texts_with_cache(
        texts=cond["text"].tolist(),
        participant_ids=cond["participant_id"].tolist(),
        input_condition=f"{input_condition}_cv_all",
        model_name=mpnet_model,
        batch_size=batch_size,
        device=device,
        cache_dir=cache_dir,
    )
    splitter = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=random_state
    )
    metrics_rows: list[dict[str, float | int | str | None]] = []
    prediction_frames: list[pd.DataFrame] = []
    y = cond["label"].to_numpy()
    for fold, (train_idx, test_idx) in enumerate(splitter.split(embeddings, y), start=1):
        train = cond.iloc[train_idx].reset_index(drop=True)
        test = cond.iloc[test_idx].reset_index(drop=True)
        x_train = embeddings[train_idx]
        x_test = embeddings[test_idx]
        model = make_logistic_regression(random_state=random_state)
        model.fit(x_train, train["label"].to_numpy())
        y_pred = model.predict(x_test)
        y_score = model.predict_proba(x_test)[:, 1]
        metrics = compute_binary_metrics(test["label"].to_numpy(), y_pred, y_score)
        metrics = add_common_metric_fields(
            metrics,
            model_name=f"mpnet_logistic_regression:{mpnet_model}",
            input_condition=input_condition,
            train_rows=len(train),
            test_rows=len(test),
            dropped_rows=dropped_rows,
            notes=f"5-fold CV SentenceTransformer embeddings, batch_size={batch_size}, device={device}",
        )
        metrics["evaluation"] = "cv"
        metrics["fold"] = fold
        metrics["cv_folds"] = cv_folds
        metrics_rows.append(metrics)
        preds = prediction_frame(
            test,
            model_name=f"mpnet_logistic_regression:{mpnet_model}",
            input_condition=input_condition,
            y_pred=y_pred,
            y_score=y_score,
        )
        preds["evaluation"] = "cv"
        preds["fold"] = fold
        prediction_frames.append(preds)
    return metrics_rows, pd.concat(prediction_frames, ignore_index=True)


def write_outputs(
    *,
    metrics_rows: list[dict[str, float | int | str | None]],
    prediction_frames: list[pd.DataFrame],
    output_dir: Path,
    run_id: str,
    config: dict[str, object],
) -> tuple[Path, Path, Path, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"baseline_metrics__{run_id}.csv"
    predictions_path = output_dir / f"baseline_predictions__{run_id}.csv"
    config_path = output_dir / f"baseline_config__{run_id}.json"
    cv_summary_path = output_dir / f"baseline_cv_summary__{run_id}.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False, encoding="utf-8")
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        predictions_path, index=False, encoding="utf-8"
    )
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    cv_summary = aggregate_cv_metric_rows(metrics_rows)
    if not cv_summary.empty:
        cv_summary.to_csv(cv_summary_path, index=False, encoding="utf-8")
    else:
        cv_summary_path = None
    return metrics_path, predictions_path, config_path, cv_summary_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DAIC-WOZ text baselines for C1-C4 input conditions."
    )
    parser.add_argument("--conditions-csv", type=Path, default=DEFAULT_CONDITIONS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--models",
        default="tfidf_lr",
        help="Comma-separated: tfidf_lr,tfidf_svm,mpnet_lr",
    )
    parser.add_argument(
        "--conditions",
        default="all",
        help="Comma-separated input_condition list, or all.",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--test-split", default="dev")
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="Run stratified cross-validation instead of fixed train/dev when >= 2.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--tfidf-max-features", type=int, default=50000)
    parser.add_argument("--tfidf-min-df", type=int, default=2)
    parser.add_argument("--tfidf-ngram-min", type=int, default=1)
    parser.add_argument("--tfidf-ngram-max", type=int, default=2)
    parser.add_argument("--mpnet-model", default=DEFAULT_MPNET_MODEL)
    parser.add_argument("--mpnet-batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--embedding-cache-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "embedding_cache",
    )
    parser.add_argument(
        "--keep-empty-text",
        action="store_true",
        help="Keep empty texts instead of dropping them. Not recommended for interviewer-only conditions.",
    )
    parser.add_argument("--run-id", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    df = load_conditions(args.conditions_csv)
    requested_conditions = parse_condition_list(args.conditions)
    conditions = requested_conditions or sorted(df["input_condition"].unique().tolist())
    requested_models = {item.strip() for item in args.models.split(",") if item.strip()}
    valid_models = {"tfidf_lr", "tfidf_svm", "mpnet_lr"}
    unknown_models = requested_models.difference(valid_models)
    if unknown_models:
        raise ValueError(f"Unknown model names: {sorted(unknown_models)}")

    device = resolve_device(args.device) if "mpnet_lr" in requested_models else args.device
    metrics_rows: list[dict[str, float | int | str | None]] = []
    prediction_frames: list[pd.DataFrame] = []
    drop_empty_text = not args.keep_empty_text
    ngram_range = (args.tfidf_ngram_min, args.tfidf_ngram_max)
    use_cv = args.cv_folds >= 2

    for input_condition in conditions:
        print(f"\n=== {input_condition} ===", flush=True)
        if "tfidf_lr" in requested_models:
            if use_cv:
                print(
                    f"Running {args.cv_folds}-fold CV TF-IDF + Logistic Regression(class_weight='balanced')",
                    flush=True,
                )
                fold_metrics, preds = run_tfidf_lr_cv(
                    df,
                    input_condition=input_condition,
                    random_state=args.random_state,
                    max_features=args.tfidf_max_features,
                    min_df=args.tfidf_min_df,
                    ngram_range=ngram_range,
                    cv_folds=args.cv_folds,
                    drop_empty_text=drop_empty_text,
                )
                metrics_rows.extend(fold_metrics)
                prediction_frames.append(preds)
                cv_summary = aggregate_cv_metric_rows(fold_metrics)
                summary_row = cv_summary.iloc[0]
                print(
                    f"TF-IDF CV done: macro_f1={summary_row['macro_f1_mean']:.4f}+/-{summary_row['macro_f1_std']:.4f}, mcc={summary_row['mcc_mean']:.4f}",
                    flush=True,
                )
            else:
                print("Running TF-IDF + Logistic Regression(class_weight='balanced')", flush=True)
                metrics, preds = run_tfidf_lr(
                    df,
                    input_condition=input_condition,
                    random_state=args.random_state,
                    max_features=args.tfidf_max_features,
                    min_df=args.tfidf_min_df,
                    ngram_range=ngram_range,
                    train_split=args.train_split,
                    test_split=args.test_split,
                    drop_empty_text=drop_empty_text,
                )
                metrics["evaluation"] = "train_dev"
                metrics_rows.append(metrics)
                prediction_frames.append(preds)
                print(
                    f"TF-IDF done: macro_f1={metrics['macro_f1']:.4f}, mcc={metrics['mcc']:.4f}, auc={metrics['auc']:.4f}",
                    flush=True,
                )
        if "tfidf_svm" in requested_models:
            if use_cv:
                print(
                    f"Running {args.cv_folds}-fold CV TF-IDF + Linear SVM(class_weight='balanced')",
                    flush=True,
                )
                fold_metrics, preds = run_tfidf_svm_cv(
                    df,
                    input_condition=input_condition,
                    random_state=args.random_state,
                    max_features=args.tfidf_max_features,
                    min_df=args.tfidf_min_df,
                    ngram_range=ngram_range,
                    cv_folds=args.cv_folds,
                    drop_empty_text=drop_empty_text,
                )
                metrics_rows.extend(fold_metrics)
                prediction_frames.append(preds)
                cv_summary = aggregate_cv_metric_rows(fold_metrics)
                summary_row = cv_summary.iloc[0]
                print(
                    f"TF-IDF SVM CV done: macro_f1={summary_row['macro_f1_mean']:.4f}+/-{summary_row['macro_f1_std']:.4f}, mcc={summary_row['mcc_mean']:.4f}",
                    flush=True,
                )
            else:
                print("Running TF-IDF + Linear SVM(class_weight='balanced')", flush=True)
                metrics, preds = run_tfidf_svm(
                    df,
                    input_condition=input_condition,
                    random_state=args.random_state,
                    max_features=args.tfidf_max_features,
                    min_df=args.tfidf_min_df,
                    ngram_range=ngram_range,
                    train_split=args.train_split,
                    test_split=args.test_split,
                    drop_empty_text=drop_empty_text,
                )
                metrics["evaluation"] = "train_dev"
                metrics_rows.append(metrics)
                prediction_frames.append(preds)
                print(
                    f"TF-IDF SVM done: macro_f1={metrics['macro_f1']:.4f}, mcc={metrics['mcc']:.4f}, auc={metrics['auc']:.4f}",
                    flush=True,
                )
        if "mpnet_lr" in requested_models:
            if use_cv:
                print(
                    f"Running {args.cv_folds}-fold CV {args.mpnet_model} embeddings + Logistic Regression(class_weight='balanced') on {device}",
                    flush=True,
                )
                fold_metrics, preds = run_mpnet_lr_cv(
                    df,
                    input_condition=input_condition,
                    random_state=args.random_state,
                    mpnet_model=args.mpnet_model,
                    batch_size=args.mpnet_batch_size,
                    device=device,
                    cache_dir=args.embedding_cache_dir,
                    cv_folds=args.cv_folds,
                    drop_empty_text=drop_empty_text,
                )
                metrics_rows.extend(fold_metrics)
                prediction_frames.append(preds)
                cv_summary = aggregate_cv_metric_rows(fold_metrics)
                summary_row = cv_summary.iloc[0]
                print(
                    f"MPNet CV done: macro_f1={summary_row['macro_f1_mean']:.4f}+/-{summary_row['macro_f1_std']:.4f}, mcc={summary_row['mcc_mean']:.4f}",
                    flush=True,
                )
            else:
                print(
                    f"Running {args.mpnet_model} embeddings + Logistic Regression(class_weight='balanced') on {device}",
                    flush=True,
                )
                metrics, preds = run_mpnet_lr(
                    df,
                    input_condition=input_condition,
                    random_state=args.random_state,
                    mpnet_model=args.mpnet_model,
                    batch_size=args.mpnet_batch_size,
                    device=device,
                    cache_dir=args.embedding_cache_dir,
                    train_split=args.train_split,
                    test_split=args.test_split,
                    drop_empty_text=drop_empty_text,
                )
                metrics["evaluation"] = "train_dev"
                metrics_rows.append(metrics)
                prediction_frames.append(preds)
                print(
                    f"MPNet done: macro_f1={metrics['macro_f1']:.4f}, mcc={metrics['mcc']:.4f}, auc={metrics['auc']:.4f}",
                    flush=True,
                )

    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "conditions_csv": str(args.conditions_csv),
        "output_dir": str(args.output_dir),
        "run_id": run_id,
        "models": sorted(requested_models),
        "conditions": conditions,
        "train_split": args.train_split,
        "test_split": args.test_split,
        "cv_folds": args.cv_folds,
        "evaluation": "cv" if use_cv else "train_dev",
        "random_state": args.random_state,
        "class_weight": "balanced",
        "drop_empty_text": drop_empty_text,
        "tfidf": {
            "max_features": args.tfidf_max_features,
            "min_df": args.tfidf_min_df,
            "ngram_range": ngram_range,
        },
        "mpnet": {
            "model": args.mpnet_model,
            "batch_size": args.mpnet_batch_size,
            "device": device,
            "embedding_cache_dir": str(args.embedding_cache_dir),
        },
    }
    metrics_path, predictions_path, config_path, cv_summary_path = write_outputs(
        metrics_rows=metrics_rows,
        prediction_frames=prediction_frames,
        output_dir=args.output_dir,
        run_id=run_id,
        config=config,
    )
    print("\nWrote outputs:", flush=True)
    print(metrics_path, flush=True)
    print(predictions_path, flush=True)
    print(config_path, flush=True)
    if cv_summary_path is not None:
        print(cv_summary_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
