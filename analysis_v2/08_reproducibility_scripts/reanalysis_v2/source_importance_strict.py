from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


BASE_SEED = 20260705

MODEL_SPECS: "OrderedDict[str, tuple[str, ...]]" = OrderedDict(
    {
        "M0": ("c2_text",),
        "M1": ("c2_text", "domain_count"),
        "M2": ("c2_text", "template_presence"),
        "M3": ("c2_text", "domain_count", "template_presence"),
        "M4": ("c2_text", "domain_count", "template_presence", "c5_text"),
        "M5": ("c2_text", "domain_count", "template_presence", "c3_text"),
    }
)


def bh_fdr(p_values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values in original order."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p_values must be finite and within [0, 1]")
    order = np.argsort(values)
    ranked = values[order]
    n = len(values)
    adjusted_sorted = ranked * n / np.arange(1, n + 1)
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def significance_marker(q_value: float) -> str:
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < 0.05:
        return "*"
    return "ns"


def auc_pairwise(
    label: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> float:
    """AUC via positive-negative pair comparisons, including half credit for ties."""
    y = np.asarray(label, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    positive = probability[y == 1]
    negative = probability[y == 0]
    if positive.size == 0 or negative.size == 0:
        raise ValueError("AUC requires both classes")
    differences = positive[:, None] - negative[None, :]
    return float((np.count_nonzero(differences > 0) + 0.5 * np.count_nonzero(differences == 0)) / differences.size)


def _auc_rows_pairwise(label: np.ndarray, probability_rows: np.ndarray) -> np.ndarray:
    positive = probability_rows[:, label == 1]
    negative = probability_rows[:, label == 0]
    differences = positive[:, :, None] - negative[:, None, :]
    wins = np.count_nonzero(differences > 0, axis=(1, 2))
    ties = np.count_nonzero(differences == 0, axis=(1, 2))
    return (wins + 0.5 * ties) / differences.shape[1] / differences.shape[2]


def _numeric_columns(frame: pd.DataFrame, group: str) -> list[str]:
    prefix = f"{group}__"
    columns = [column for column in frame.columns if column.startswith(prefix)]
    if not columns:
        raise ValueError(f"No numeric columns found for group {group!r}")
    return columns


@dataclass
class FittedSourceModel:
    groups: tuple[str, ...]
    vectorizers: dict[str, TfidfVectorizer]
    scalers: dict[str, StandardScaler]
    numeric_columns: dict[str, list[str]]
    classifier: LogisticRegression

    def transform_parts(self, frame: pd.DataFrame) -> dict[str, sparse.csr_matrix]:
        parts: dict[str, sparse.csr_matrix] = {}
        for group in self.groups:
            if group.endswith("_text"):
                parts[group] = self.vectorizers[group].transform(
                    frame[group].fillna("").astype(str)
                )
            else:
                columns = self.numeric_columns[group]
                parts[group] = sparse.csr_matrix(
                    self.scalers[group].transform(frame[columns].to_numpy(dtype=float))
                )
        return parts

    def predict_from_parts(self, parts: Mapping[str, sparse.spmatrix]) -> np.ndarray:
        matrix = sparse.hstack([parts[group] for group in self.groups], format="csr")
        return self.classifier.predict_proba(matrix)[:, 1]

    def transform(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        parts = self.transform_parts(frame)
        return sparse.hstack([parts[group] for group in self.groups], format="csr")

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.classifier.predict_proba(self.transform(frame))[:, 1]


def _fit_group_matrices(
    train: pd.DataFrame,
    test: pd.DataFrame,
    groups: Sequence[str],
    *,
    min_df: int,
) -> tuple[dict[str, sparse.csr_matrix], dict[str, sparse.csr_matrix]]:
    train_parts: dict[str, sparse.csr_matrix] = {}
    test_parts: dict[str, sparse.csr_matrix] = {}
    for group in groups:
        if group.endswith("_text"):
            vectorizer = TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=min_df,
                max_features=50_000,
                sublinear_tf=True,
            )
            train_parts[group] = vectorizer.fit_transform(
                train[group].fillna("").astype(str)
            ).tocsr()
            test_parts[group] = vectorizer.transform(
                test[group].fillna("").astype(str)
            ).tocsr()
        else:
            columns = _numeric_columns(train, group)
            scaler = StandardScaler()
            train_parts[group] = sparse.csr_matrix(
                scaler.fit_transform(train[columns].to_numpy(dtype=float))
            )
            test_parts[group] = sparse.csr_matrix(
                scaler.transform(test[columns].to_numpy(dtype=float))
            )
    return train_parts, test_parts


def fit_source_model(
    train: pd.DataFrame,
    groups: Sequence[str],
    *,
    seed: int,
    min_df: int = 2,
) -> FittedSourceModel:
    train_parts: list[sparse.spmatrix] = []
    vectorizers: dict[str, TfidfVectorizer] = {}
    scalers: dict[str, StandardScaler] = {}
    numeric_columns: dict[str, list[str]] = {}
    for group in groups:
        if group.endswith("_text"):
            if group not in train.columns:
                raise ValueError(f"Missing text column {group!r}")
            vectorizer = TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=min_df,
                max_features=50_000,
                sublinear_tf=True,
            )
            train_matrix = vectorizer.fit_transform(train[group].fillna("").astype(str))
            vectorizers[group] = vectorizer
        else:
            columns = _numeric_columns(train, group)
            scaler = StandardScaler()
            train_matrix = sparse.csr_matrix(
                scaler.fit_transform(train[columns].to_numpy(dtype=float))
            )
            scalers[group] = scaler
            numeric_columns[group] = columns
        train_parts.append(train_matrix)
    X_train = sparse.hstack(train_parts, format="csr")
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=2000,
        random_state=seed,
    )
    classifier.fit(X_train, train["label"].to_numpy(dtype=int))
    return FittedSourceModel(
        groups=tuple(groups),
        vectorizers=vectorizers,
        scalers=scalers,
        numeric_columns=numeric_columns,
        classifier=classifier,
    )


def fit_predict_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    groups: Sequence[str],
    *,
    seed: int,
    min_df: int = 2,
) -> np.ndarray:
    model = fit_source_model(train, groups, seed=seed, min_df=min_df)
    return model.predict_proba(test)


def run_repeated_cv(
    frame: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    model_specs: Mapping[str, Sequence[str]] = MODEL_SPECS,
    min_df: int = 2,
    base_seed: int = BASE_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"participant_id", "label"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame["participant_id"].duplicated().any():
        raise ValueError("participant_id must be unique")
    rows: list[pd.DataFrame] = []
    repeats = sorted(splits["repeat"].unique())
    for repeat in repeats:
        repeat_splits = splits.loc[splits["repeat"] == repeat]
        for fold in sorted(repeat_splits["fold"].unique()):
            fold_splits = repeat_splits.loc[repeat_splits["fold"] == fold]
            test_ids = set(
                fold_splits.loc[fold_splits["role"] == "test", "participant_id"].astype(int)
            )
            test_mask = frame["participant_id"].astype(int).isin(test_ids)
            train = frame.loc[~test_mask].copy()
            test = frame.loc[test_mask].copy()
            if test.empty:
                raise ValueError(f"No test participants for repeat={repeat}, fold={fold}")
            if train["label"].nunique() < 2 or test["label"].nunique() < 2:
                raise ValueError(f"Both classes required in repeat={repeat}, fold={fold}")
            fold_output = test[["participant_id", "label"]].copy()
            fold_output.insert(1, "repeat", int(repeat))
            fold_output.insert(2, "fold", int(fold))
            unique_groups = tuple(dict.fromkeys(group for groups in model_specs.values() for group in groups))
            train_parts, test_parts = _fit_group_matrices(
                train,
                test,
                unique_groups,
                min_df=min_df,
            )
            for model_index, (model_name, groups) in enumerate(model_specs.items()):
                seed = base_seed + 1000 * int(repeat) + 10 * int(fold) + model_index
                X_train = sparse.hstack([train_parts[group] for group in groups], format="csr")
                X_test = sparse.hstack([test_parts[group] for group in groups], format="csr")
                model = LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=2000,
                    random_state=seed,
                )
                model.fit(X_train, train["label"].to_numpy(dtype=int))
                fold_output[f"{model_name}_prob"] = model.predict_proba(X_test)[:, 1]
            rows.append(fold_output)
    fold_rows = pd.concat(rows, ignore_index=True).sort_values(
        ["repeat", "fold", "participant_id"]
    )
    expected_repeats = len(repeats)
    counts = fold_rows.groupby("participant_id").size()
    if not counts.eq(expected_repeats).all():
        raise AssertionError("Each participant must have one OOF prediction per repeat")
    probability_columns = [column for column in fold_rows.columns if column.endswith("_prob")]
    participant_rows = (
        fold_rows.groupby(["participant_id", "label"], as_index=False)[probability_columns]
        .mean()
        .sort_values("participant_id")
        .reset_index(drop=True)
    )
    return fold_rows.reset_index(drop=True), participant_rows


def paired_auc_inference(
    label: Sequence[int] | np.ndarray,
    probabilities_a: Sequence[float] | np.ndarray,
    probabilities_b: Sequence[float] | np.ndarray,
    *,
    n_bootstrap: int = 5000,
    n_permutations: int = 10_000,
    seed: int = BASE_SEED,
) -> dict[str, float | int]:
    y = np.asarray(label, dtype=int)
    a = np.asarray(probabilities_a, dtype=float)
    b = np.asarray(probabilities_b, dtype=float)
    if not (len(y) == len(a) == len(b)):
        raise ValueError("label and probability arrays must have equal length")
    observed = float(auc_pairwise(y, a) - auc_pairwise(y, b))
    rng = np.random.default_rng(seed)
    bootstrap: list[float] = []
    for _ in range(n_bootstrap):
        index = rng.integers(0, len(y), size=len(y))
        if np.unique(y[index]).size < 2:
            continue
        bootstrap.append(
            float(auc_pairwise(y[index], a[index]) - auc_pairwise(y[index], b[index]))
        )
    if not bootstrap:
        raise ValueError("No valid bootstrap samples contained both classes")
    null_more_extreme = 0
    completed = 0
    batch_size = 512
    while completed < n_permutations:
        batch = min(batch_size, n_permutations - completed)
        swap = rng.random((batch, len(y))) < 0.5
        perm_a = np.where(swap, b[None, :], a[None, :])
        perm_b = np.where(swap, a[None, :], b[None, :])
        null_delta = _auc_rows_pairwise(y, perm_a) - _auc_rows_pairwise(y, perm_b)
        null_more_extreme += int(np.count_nonzero(np.abs(null_delta) >= abs(observed) - 1e-15))
        completed += batch
    return {
        "delta_auc": observed,
        "ci_lower": float(np.percentile(bootstrap, 2.5)),
        "ci_upper": float(np.percentile(bootstrap, 97.5)),
        "p_value": float((null_more_extreme + 1) / (n_permutations + 1)),
        "n_bootstrap_valid": int(len(bootstrap)),
        "n_permutations": int(n_permutations),
    }


def bootstrap_auc_ci(
    label: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    n_bootstrap: int = 5000,
    seed: int = BASE_SEED,
) -> dict[str, float | int]:
    y = np.asarray(label, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_bootstrap):
        index = rng.integers(0, len(y), size=len(y))
        if np.unique(y[index]).size < 2:
            continue
        values.append(float(auc_pairwise(y[index], probability[index])))
    if not values:
        raise ValueError("No valid bootstrap samples contained both classes")
    return {
        "auc": float(roc_auc_score(y, probability)),
        "ci_lower": float(np.percentile(values, 2.5)),
        "ci_upper": float(np.percentile(values, 97.5)),
        "n_bootstrap_valid": int(len(values)),
    }


def summarize_model_metrics(
    participant_rows: pd.DataFrame,
    *,
    n_bootstrap: int = 5000,
    seed: int = BASE_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = participant_rows["label"].to_numpy(dtype=int)
    model_names = [column.removesuffix("_prob") for column in participant_rows if column.endswith("_prob")]
    metric_rows: list[dict[str, float | str]] = []
    ci_rows: list[dict[str, float | int | str]] = []
    for index, model_name in enumerate(model_names):
        probability = participant_rows[f"{model_name}_prob"].to_numpy(dtype=float)
        metric_rows.append(
            {
                "model": model_name,
                "roc_auc": float(roc_auc_score(y, probability)),
                "pr_auc": float(average_precision_score(y, probability)),
                "brier": float(brier_score_loss(y, probability)),
                "log_loss": float(log_loss(y, probability, labels=[0, 1])),
            }
        )
        ci = bootstrap_auc_ci(
            y,
            probability,
            n_bootstrap=n_bootstrap,
            seed=seed + index,
        )
        ci_rows.append({"model": model_name, **ci})
    return pd.DataFrame(metric_rows), pd.DataFrame(ci_rows)


DEFAULT_COMPARISONS: tuple[tuple[str, str], ...] = (
    ("M1", "M0"),
    ("M2", "M0"),
    ("M3", "M0"),
    ("M3", "M1"),
    ("M3", "M2"),
    ("M4", "M3"),
    ("M5", "M3"),
)


def infer_incremental_comparisons(
    participant_rows: pd.DataFrame,
    *,
    comparisons: Sequence[tuple[str, str]] = DEFAULT_COMPARISONS,
    n_bootstrap: int = 5000,
    n_permutations: int = 10_000,
    seed: int = BASE_SEED,
) -> pd.DataFrame:
    y = participant_rows["label"].to_numpy(dtype=int)
    rows: list[dict[str, float | int | str]] = []
    for index, (model_a, model_b) in enumerate(comparisons):
        result = paired_auc_inference(
            y,
            participant_rows[f"{model_a}_prob"].to_numpy(dtype=float),
            participant_rows[f"{model_b}_prob"].to_numpy(dtype=float),
            n_bootstrap=n_bootstrap,
            n_permutations=n_permutations,
            seed=seed + 101 * index,
        )
        rows.append({"comparison": f"{model_a} vs {model_b}", "model_a": model_a, "model_b": model_b, **result})
    result_frame = pd.DataFrame(rows)
    result_frame["q_value"] = bh_fdr(result_frame["p_value"].to_numpy(dtype=float))
    result_frame["significance"] = result_frame["q_value"].map(significance_marker)
    return result_frame


def _permuted_group_frame(
    frame: pd.DataFrame,
    group: str,
    order: np.ndarray,
) -> pd.DataFrame:
    result = frame.copy()
    columns = [group] if group.endswith("_text") else _numeric_columns(frame, group)
    result.loc[:, columns] = frame.iloc[order][columns].to_numpy()
    return result


def run_group_permutation_importance(
    frame: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    groups: Sequence[str] = MODEL_SPECS["M3"],
    min_df: int = 2,
    shuffles_per_fold: int = 100,
    n_bootstrap: int = 5000,
    n_permutations: int = 10_000,
    seed: int = BASE_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows: list[pd.DataFrame] = []
    for repeat in sorted(splits["repeat"].unique()):
        repeat_splits = splits.loc[splits["repeat"] == repeat]
        for fold in sorted(repeat_splits["fold"].unique()):
            fold_splits = repeat_splits.loc[repeat_splits["fold"] == fold]
            test_ids = set(fold_splits.loc[fold_splits["role"] == "test", "participant_id"].astype(int))
            test_mask = frame["participant_id"].astype(int).isin(test_ids)
            train = frame.loc[~test_mask].copy()
            test = frame.loc[test_mask].copy()
            model = fit_source_model(
                train,
                groups,
                seed=seed + 1000 * int(repeat) + 10 * int(fold),
                min_df=min_df,
            )
            fold_output = test[["participant_id", "label"]].copy()
            fold_output["repeat"] = int(repeat)
            fold_output["fold"] = int(fold)
            baseline_parts = model.transform_parts(test)
            fold_output["baseline_prob"] = model.predict_from_parts(baseline_parts)
            for group in groups:
                predictions: list[np.ndarray] = []
                for _ in range(shuffles_per_fold):
                    order = rng.permutation(len(test))
                    permuted_parts = dict(baseline_parts)
                    permuted_parts[group] = baseline_parts[group][order]
                    predictions.append(model.predict_from_parts(permuted_parts))
                fold_output[f"permuted_{group}_prob"] = np.mean(predictions, axis=0)
            rows.append(fold_output)
    fold_rows = pd.concat(rows, ignore_index=True)
    probability_columns = [column for column in fold_rows if column.endswith("_prob")]
    participant = (
        fold_rows.groupby(["participant_id", "label"], as_index=False)[probability_columns]
        .mean()
        .sort_values("participant_id")
        .reset_index(drop=True)
    )
    y = participant["label"].to_numpy(dtype=int)
    importance_rows: list[dict[str, float | int | str]] = []
    for index, group in enumerate(groups):
        inference = paired_auc_inference(
            y,
            participant["baseline_prob"].to_numpy(dtype=float),
            participant[f"permuted_{group}_prob"].to_numpy(dtype=float),
            n_bootstrap=n_bootstrap,
            n_permutations=n_permutations,
            seed=seed + 5000 + index * 101,
        )
        importance_rows.append({"source": group, **inference})
    importance = pd.DataFrame(importance_rows)
    importance["q_value"] = bh_fdr(importance["p_value"].to_numpy(dtype=float))
    importance["significance"] = importance["q_value"].map(significance_marker)
    importance = importance.sort_values("delta_auc", ascending=False).reset_index(drop=True)
    return importance, participant


def _run_numeric_repeated_cv(
    frame: pd.DataFrame,
    splits: pd.DataFrame,
    columns: Sequence[str],
    *,
    seed: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for repeat in sorted(splits["repeat"].unique()):
        repeat_splits = splits.loc[splits["repeat"] == repeat]
        for fold in sorted(repeat_splits["fold"].unique()):
            fold_splits = repeat_splits.loc[repeat_splits["fold"] == fold]
            test_ids = set(fold_splits.loc[fold_splits["role"] == "test", "participant_id"].astype(int))
            test_mask = frame["participant_id"].astype(int).isin(test_ids)
            train = frame.loc[~test_mask]
            test = frame.loc[test_mask]
            scaler = StandardScaler()
            X_train = scaler.fit_transform(train[list(columns)].to_numpy(dtype=float))
            X_test = scaler.transform(test[list(columns)].to_numpy(dtype=float))
            model = LogisticRegression(
                C=1.0,
                class_weight="balanced",
                solver="liblinear",
                max_iter=2000,
                random_state=seed + 1000 * int(repeat) + 10 * int(fold),
            )
            model.fit(X_train, train["label"].to_numpy(dtype=int))
            output = test[["participant_id", "label"]].copy()
            output["probability"] = model.predict_proba(X_test)[:, 1]
            rows.append(output)
    fold_rows = pd.concat(rows, ignore_index=True)
    return (
        fold_rows.groupby(["participant_id", "label"], as_index=False)["probability"]
        .mean()
        .sort_values("participant_id")
        .reset_index(drop=True)
    )


def run_domain_leave_one_out(
    frame: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    representations: Sequence[str] = ("domain_presence", "domain_count"),
    n_bootstrap: int = 5000,
    n_permutations: int = 10_000,
    seed: int = BASE_SEED,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    row_index = 0
    for representation_index, representation in enumerate(representations):
        columns = _numeric_columns(frame, representation)
        representation_seed = seed + representation_index * 100_000
        full = _run_numeric_repeated_cv(frame, splits, columns, seed=representation_seed)
        y = full["label"].to_numpy(dtype=int)
        full_prob = full["probability"].to_numpy(dtype=float)
        full_auc = float(roc_auc_score(y, full_prob))
        for removed_column in columns:
            reduced_columns = [column for column in columns if column != removed_column]
            reduced = _run_numeric_repeated_cv(
                frame,
                splits,
                reduced_columns,
                seed=representation_seed,
            )
            if not np.array_equal(full["participant_id"], reduced["participant_id"]):
                raise AssertionError("Domain LOO participant order mismatch")
            reduced_prob = reduced["probability"].to_numpy(dtype=float)
            inference = paired_auc_inference(
                y,
                full_prob,
                reduced_prob,
                n_bootstrap=n_bootstrap,
                n_permutations=n_permutations,
                seed=seed + 20_000 + row_index * 101,
            )
            rows.append(
                {
                    "representation": representation,
                    "removed_domain": removed_column.removeprefix(f"{representation}__"),
                    "full_auc": full_auc,
                    "reduced_auc": float(roc_auc_score(y, reduced_prob)),
                    **inference,
                }
            )
            row_index += 1
    result = pd.DataFrame(rows)
    result["q_value"] = bh_fdr(result["p_value"].to_numpy(dtype=float))
    result["significance"] = result["q_value"].map(significance_marker)
    return result.sort_values(["representation", "delta_auc"], ascending=[True, False]).reset_index(drop=True)


def load_strict_source_frame(analysis_root: str | Path) -> pd.DataFrame:
    root = Path(analysis_root).resolve()
    text_files = {
        "c2_text": root / "01_inputs" / "c2_participant_speech.csv",
        "c3_text": root / "01_inputs" / "c3_interviewer_speech.csv",
        "c5_text": root / "01_inputs" / "c5_reviewed_symptom_evidence.csv",
    }
    base = pd.read_csv(text_files["c2_text"])[
        ["participant_id", "paper_label_phq8_ge10", "text"]
    ].rename(columns={"paper_label_phq8_ge10": "label", "text": "c2_text"})
    for column in ("c3_text", "c5_text"):
        source = pd.read_csv(text_files[column])[["participant_id", "text"]].rename(
            columns={"text": column}
        )
        base = base.merge(source, on="participant_id", how="inner", validate="one_to_one")
    numeric_files = {
        "domain_presence": root / "04_c5_controls" / "domain_presence" / "input.csv",
        "domain_count": root / "04_c5_controls" / "domain_count" / "input.csv",
        "template_presence": root / "05_c4_controls" / "template_presence" / "input.csv",
    }
    for group, path in numeric_files.items():
        source = pd.read_csv(path)
        source = source.rename(
            columns={column: f"{group}__{column}" for column in source if column != "participant_id"}
        )
        base = base.merge(source, on="participant_id", how="inner", validate="one_to_one")
    base = base.sort_values("participant_id").reset_index(drop=True)
    text_columns = ["c2_text", "c3_text", "c5_text"]
    base[text_columns] = base[text_columns].fillna("")
    if len(base) != 142 or int(base["label"].sum()) != 43:
        raise ValueError(
            f"Frozen cohort mismatch: n={len(base)}, positive={int(base['label'].sum())}"
        )
    if base["participant_id"].duplicated().any() or base.drop(columns=text_columns).isna().any().any():
        raise ValueError("Strict source frame contains duplicate IDs or missing values")
    return base
