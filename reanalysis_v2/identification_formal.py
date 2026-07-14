"""Locked inference helpers for the ten-repeat identification sensitivity run."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .inference import benjamini_hochberg


FORMAL_REPEATS = tuple(range(1, 11))
FORMAL_DRAW_COUNT = 50
PRIMARY_CONTRASTS = (
    "real_minus_matched",
    "real_minus_random",
    "d_delta_minus_fake",
)
PRIMARY_EMBEDDINGS = (
    "model_1_all_mpnet_base_v2",
    "model_2_bge_large_en_v1_5",
)


def _validate_repeat_values(values: Sequence[float]) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or len(result) < 2 or not np.isfinite(result).all():
        raise ValueError("repeat-level contrast values must be finite and have at least two entries")
    return result


def bootstrap_mean_ci(
    values: Sequence[float], *, n_bootstrap: int = 5000, seed: int
) -> tuple[float, float]:
    """Participant-repeat bootstrap interval for the mean repeat contrast."""

    values = _validate_repeat_values(values)
    if int(n_bootstrap) < 100:
        raise ValueError("n_bootstrap must be at least 100")
    rng = np.random.default_rng(int(seed))
    sample_index = rng.integers(0, len(values), size=(int(n_bootstrap), len(values)))
    means = values[sample_index].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def exact_sign_flip_p(values: Sequence[float]) -> float:
    """Two-sided exact sign-flip p-value over repeat-level contrasts."""

    values = _validate_repeat_values(values)
    n = len(values)
    if n > 20:
        raise ValueError("exact sign-flip inference is limited to at most 20 repeats")
    observed = abs(float(values.mean()))
    signs = ((np.arange(1 << n, dtype=np.uint32)[:, None] >> np.arange(n)) & 1) * 2 - 1
    null_means = (signs * values[None, :]).mean(axis=1)
    return float(np.mean(np.abs(null_means) >= observed - 1e-15))


def summarize_repeat_values(
    values: Sequence[float], *, n_bootstrap: int = 5000, seed: int
) -> dict[str, float | int]:
    """Summarize ten repeat means without treating 50 draws as independent repeats."""

    values = _validate_repeat_values(values)
    ci_low, ci_high = bootstrap_mean_ci(values, n_bootstrap=n_bootstrap, seed=seed)
    return {
        "n_repeats": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "direction_positive_n": int(np.sum(values > 0)),
        "direction_negative_n": int(np.sum(values < 0)),
        "direction_zero_n": int(np.sum(values == 0)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "permutation_p": exact_sign_flip_p(values),
    }


def apply_primary_fdr(table: pd.DataFrame) -> pd.DataFrame:
    """Apply BH-FDR to the six locked contrast-by-embedding rows."""

    required = {"embedding_model", "contrast", "raw_p"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"primary contrast table is missing columns: {sorted(missing)}")
    result = table.copy()
    if len(result) != len(PRIMARY_CONTRASTS) * len(PRIMARY_EMBEDDINGS):
        raise ValueError("primary identification FDR requires exactly six rows")
    if result[["embedding_model", "contrast"]].duplicated().any():
        raise ValueError("primary identification rows must be unique")
    if set(result["embedding_model"]) != set(PRIMARY_EMBEDDINGS):
        raise ValueError("primary identification rows must contain both locked embeddings")
    if set(result["contrast"]) != set(PRIMARY_CONTRASTS):
        raise ValueError("primary identification rows must contain all three locked contrasts")
    p_values = pd.to_numeric(result["raw_p"], errors="coerce").to_numpy(float)
    if not np.isfinite(p_values).all() or ((p_values < 0) | (p_values > 1)).any():
        raise ValueError("primary raw p-values must be finite and lie in [0, 1]")
    result["fdr_family"] = "primary_identification_six"
    result["bh_q"] = benjamini_hochberg(p_values)
    return result


def _require_columns(frame: pd.DataFrame, required: set[str], *, name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _one_value(frame: pd.DataFrame, *, name: str, label: str) -> float:
    if len(frame) != 1:
        raise ValueError(f"{name} must contain exactly one row for {label}; got {len(frame)}")
    value = float(frame.iloc[0])
    if not np.isfinite(value):
        raise ValueError(f"{name} contains a non-finite value for {label}")
    return value


def _draw_values(
    frame: pd.DataFrame,
    *,
    name: str,
    embedding: str,
    repeat: int,
    condition: str,
    value_column: str,
) -> np.ndarray:
    selected = frame.loc[
        frame["embedding_model"].eq(embedding)
        & frame["repeat"].eq(int(repeat))
        & frame["condition"].eq(condition)
    ].copy()
    expected_draws = list(range(1, FORMAL_DRAW_COUNT + 1))
    observed_draws = sorted(pd.to_numeric(selected["draw_id"], errors="raise").astype(int).tolist())
    if observed_draws != expected_draws:
        raise ValueError(
            f"{name} {embedding}/{repeat}/{condition} must contain draw IDs 1..{FORMAL_DRAW_COUNT}"
        )
    values = pd.to_numeric(selected[value_column], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite {value_column} values")
    return values


def build_primary_repeat_contrasts(
    pairing_metrics: pd.DataFrame,
    fake_d_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Build one paired contrast per repeat from the locked draw tables.

    Draws are averaged within repeat before inference.  This prevents the 50
    random/matched permutations from being treated as 50 independent
    replications of the ten outer repeats.
    """

    _require_columns(
        pairing_metrics,
        {"embedding_model", "repeat", "condition", "draw_id", "auc"},
        name="pairing_metrics",
    )
    _require_columns(
        fake_d_metrics,
        {"embedding_model", "repeat", "condition", "draw_id", "delta_auc"},
        name="fake_d_metrics",
    )
    pairing = pairing_metrics.copy()
    fake_d = fake_d_metrics.copy()
    for frame, name in ((pairing, "pairing_metrics"), (fake_d, "fake_d_metrics")):
        frame["repeat"] = pd.to_numeric(frame["repeat"], errors="raise").astype(int)
        frame["draw_id"] = pd.to_numeric(frame["draw_id"], errors="raise").astype(int)
        frame["embedding_model"] = frame["embedding_model"].astype(str)
        frame["condition"] = frame["condition"].astype(str)
        if frame[["repeat", "draw_id"]].isna().any().any():
            raise ValueError(f"{name} contains missing repeat/draw IDs")

    rows: list[dict[str, object]] = []
    for embedding in PRIMARY_EMBEDDINGS:
        for repeat in FORMAL_REPEATS:
            pairing_real = pairing.loc[
                pairing["embedding_model"].eq(embedding)
                & pairing["repeat"].eq(int(repeat))
                & pairing["condition"].eq("real")
                & pairing["draw_id"].eq(0),
                "auc",
            ]
            real_auc = _one_value(pairing_real, name="pairing_metrics.auc", label=f"{embedding}/{repeat}/real")
            random_auc = _draw_values(
                pairing,
                name="pairing_metrics",
                embedding=embedding,
                repeat=repeat,
                condition="random",
                value_column="auc",
            )
            matched_auc = _draw_values(
                pairing,
                name="pairing_metrics",
                embedding=embedding,
                repeat=repeat,
                condition="matched",
                value_column="auc",
            )
            fake_real = fake_d.loc[
                fake_d["embedding_model"].eq(embedding)
                & fake_d["repeat"].eq(int(repeat))
                & fake_d["condition"].eq("real")
                & fake_d["draw_id"].eq(0),
                "delta_auc",
            ]
            real_delta = _one_value(
                fake_real,
                name="fake_d_metrics.delta_auc",
                label=f"{embedding}/{repeat}/real",
            )
            fake_delta = _draw_values(
                fake_d,
                name="fake_d_metrics",
                embedding=embedding,
                repeat=repeat,
                condition="fake_d",
                value_column="delta_auc",
            )
            for contrast, real_value, null_values in (
                ("real_minus_matched", real_auc, matched_auc),
                ("real_minus_random", real_auc, random_auc),
                ("d_delta_minus_fake", real_delta, fake_delta),
            ):
                rows.append(
                    {
                        "embedding_model": embedding,
                        "repeat": int(repeat),
                        "contrast": contrast,
                        "real_value": float(real_value),
                        "null_mean": float(np.mean(null_values)),
                        "null_sd": float(np.std(null_values, ddof=1)),
                        "repeat_contrast": float(real_value - np.mean(null_values)),
                        "n_draws": int(len(null_values)),
                    }
                )
    result = pd.DataFrame(rows)
    if len(result) != len(PRIMARY_EMBEDDINGS) * len(FORMAL_REPEATS) * len(PRIMARY_CONTRASTS):
        raise ValueError("primary repeat contrast table has an unexpected row count")
    return result.sort_values(["embedding_model", "contrast", "repeat"], kind="stable").reset_index(drop=True)


def summarize_primary_contrasts(
    repeat_contrasts: pd.DataFrame,
    *,
    n_bootstrap: int = 5000,
    seed: int,
) -> pd.DataFrame:
    """Summarize the six locked primary contrast-by-embedding rows."""

    _require_columns(
        repeat_contrasts,
        {"embedding_model", "repeat", "contrast", "repeat_contrast"},
        name="repeat_contrasts",
    )
    rows: list[dict[str, object]] = []
    for embedding in PRIMARY_EMBEDDINGS:
        for contrast in PRIMARY_CONTRASTS:
            selected = repeat_contrasts.loc[
                repeat_contrasts["embedding_model"].eq(embedding)
                & repeat_contrasts["contrast"].eq(contrast)
            ].sort_values("repeat")
            if sorted(selected["repeat"].astype(int).tolist()) != list(FORMAL_REPEATS):
                raise ValueError(f"repeat_contrasts is incomplete for {embedding}/{contrast}")
            values = pd.to_numeric(selected["repeat_contrast"], errors="raise").to_numpy(dtype=float)
            offset = sum(ord(char) for char in f"{embedding}:{contrast}")
            summary = summarize_repeat_values(values, n_bootstrap=n_bootstrap, seed=int(seed) + offset)
            rows.append(
                {
                    "embedding_model": embedding,
                    "contrast": contrast,
                    "n_repeats": int(summary["n_repeats"]),
                    "mean": float(summary["mean"]),
                    "median": float(summary["median"]),
                    "direction_consistent_n": int(summary["direction_positive_n"]),
                    "direction_opposite_n": int(summary["direction_negative_n"]),
                    "direction_zero_n": int(summary["direction_zero_n"]),
                    "ci_low": float(summary["ci_low"]),
                    "ci_high": float(summary["ci_high"]),
                    "raw_p": float(summary["permutation_p"]),
                    "n_bootstrap": int(n_bootstrap),
                    "ci_method": "repeat_level_nonparametric_bootstrap_percentile",
                    "permutation_method": "exact_two_sided_sign_flip_over_repeat_means",
                }
            )
    return apply_primary_fdr(pd.DataFrame(rows)).sort_values(
        ["embedding_model", "contrast"], kind="stable"
    ).reset_index(drop=True)
