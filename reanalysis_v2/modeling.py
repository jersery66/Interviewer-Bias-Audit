from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_tfidf_classifier(*, c: float = 1.0, seed: int = 20260624) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=50_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=c,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        ]
    )


def make_dense_classifier(*, c: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        ]
    )


def fit_predict_tfidf(
    train_texts: Sequence[str],
    y_train: np.ndarray,
    test_texts: Sequence[str],
    *,
    seed: int,
    c: float = 1.0,
) -> tuple[np.ndarray, Pipeline]:
    estimator = make_tfidf_classifier(c=c, seed=seed)
    estimator.fit(list(train_texts), np.asarray(y_train, dtype=int))
    return estimator.predict_proba(list(test_texts))[:, 1], estimator


def fit_predict_dense(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    c: float,
    seed: int,
) -> tuple[np.ndarray, Pipeline]:
    estimator = make_dense_classifier(c=c, seed=seed)
    estimator.fit(np.asarray(X_train), np.asarray(y_train, dtype=int))
    return estimator.predict_proba(np.asarray(X_test))[:, 1], estimator


def make_inner_splits(
    y_outer_train: np.ndarray, *, n_splits: int = 3, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    y_outer_train = np.asarray(y_outer_train, dtype=int)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [
        (train_idx.astype(int), validation_idx.astype(int))
        for train_idx, validation_idx in splitter.split(
            np.zeros(len(y_outer_train), dtype=np.int8), y_outer_train
        )
    ]


def tune_dense_c(
    X_outer_train: np.ndarray,
    y_outer_train: np.ndarray,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    c_grid: Sequence[float],
    seed: int,
) -> tuple[float, dict[float, float]]:
    X_outer_train = np.asarray(X_outer_train)
    y_outer_train = np.asarray(y_outer_train, dtype=int)
    scores: dict[float, float] = {}
    for c in sorted(float(value) for value in c_grid):
        fold_scores = []
        for inner_fold, (train_idx, validation_idx) in enumerate(inner_splits, start=1):
            probability, _ = fit_predict_dense(
                X_outer_train[train_idx],
                y_outer_train[train_idx],
                X_outer_train[validation_idx],
                c=c,
                seed=seed + inner_fold,
            )
            fold_scores.append(
                roc_auc_score(y_outer_train[validation_idx], probability)
            )
        scores[c] = float(np.mean(fold_scores))
    selected = min(scores, key=lambda c: (-scores[c], c))
    return float(selected), scores
