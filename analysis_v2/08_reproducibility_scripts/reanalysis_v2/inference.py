from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    if p_values.ndim != 1 or not np.isfinite(p_values).all():
        raise ValueError("p-values must be a finite one-dimensional array")
    if ((p_values < 0) | (p_values > 1)).any():
        raise ValueError("p-values must lie in [0, 1]")
    n = len(p_values)
    order = np.argsort(p_values, kind="stable")
    ranked = p_values[order] * n / np.arange(1, n + 1)
    adjusted_ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


def paired_permutation_test(
    y_true: np.ndarray,
    values_a: np.ndarray,
    values_b: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_permutations: int = 10_000,
    seed: int = 20260624,
) -> dict[str, float | int]:
    y_true = np.asarray(y_true)
    values_a = np.asarray(values_a)
    values_b = np.asarray(values_b)
    if not (len(y_true) == len(values_a) == len(values_b)):
        raise ValueError("labels and paired values must have the same length")
    observed = float(metric(y_true, values_a) - metric(y_true, values_b))
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_permutations):
        swap = rng.integers(0, 2, size=len(y_true), dtype=np.int8).astype(bool)
        perm_a = np.where(swap, values_b, values_a)
        perm_b = np.where(swap, values_a, values_b)
        difference = float(metric(y_true, perm_a) - metric(y_true, perm_b))
        if abs(difference) >= abs(observed) - 1e-15:
            extreme += 1
    return {
        "observed_difference": observed,
        "p_value": float((extreme + 1) / (n_permutations + 1)),
        "n_permutations": int(n_permutations),
        "seed": int(seed),
    }


def apply_bh_by_family(
    table: pd.DataFrame,
    *,
    family_column: str = "family",
    p_column: str = "p_value",
) -> pd.DataFrame:
    if family_column not in table or p_column not in table:
        raise ValueError("family and p-value columns are required")
    result = table.copy()
    result["fdr_bh_q"] = np.nan
    for _, indices in result.groupby(family_column, sort=False).groups.items():
        index = list(indices)
        result.loc[index, "fdr_bh_q"] = benjamini_hochberg(
            result.loc[index, p_column].to_numpy(dtype=float)
        )
    return result
