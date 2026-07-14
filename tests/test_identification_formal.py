import numpy as np
import pandas as pd

from reanalysis_v2.identification_formal import (
    FORMAL_DRAW_COUNT,
    FORMAL_REPEATS,
    PRIMARY_EMBEDDINGS,
    apply_primary_fdr,
    build_primary_repeat_contrasts,
    summarize_primary_contrasts,
    bootstrap_mean_ci,
    exact_sign_flip_p,
    summarize_repeat_values,
)


def test_formal_contract_locks_ten_repeats_and_fifty_draws():
    assert FORMAL_REPEATS == tuple(range(1, 11))
    assert FORMAL_DRAW_COUNT == 50


def test_exact_sign_flip_p_uses_repeat_level_contrasts():
    values = np.array([1.0, 2.0, 3.0, 4.0])

    assert exact_sign_flip_p(values) == 0.125


def test_bootstrap_mean_ci_is_seeded_and_contains_repeat_mean():
    values = np.array([-0.2, 0.1, 0.3, 0.4, 0.2])
    first = bootstrap_mean_ci(values, n_bootstrap=200, seed=13)
    second = bootstrap_mean_ci(values, n_bootstrap=200, seed=13)

    assert first == second
    assert first[0] <= values.mean() <= first[1]


def test_repeat_summary_reports_mean_direction_and_inference():
    values = np.array([0.1, 0.2, -0.1, 0.3])
    result = summarize_repeat_values(values, n_bootstrap=200, seed=7)

    assert result["n_repeats"] == 4
    assert result["direction_positive_n"] == 3
    assert result["mean"] == 0.125
    assert 0 <= result["permutation_p"] <= 1
    assert result["ci_low"] <= result["mean"] <= result["ci_high"]


def test_primary_fdr_is_bh_over_the_six_locked_rows():
    table = pd.DataFrame(
        {
            "embedding_model": ["model_1_all_mpnet_base_v2", "model_2_bge_large_en_v1_5"] * 3,
            "contrast": ["real_minus_matched", "real_minus_matched", "real_minus_random", "real_minus_random", "d_delta_minus_fake", "d_delta_minus_fake"],
            "raw_p": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        }
    )

    result = apply_primary_fdr(table)

    assert result["bh_q"].notna().all()
    assert result["fdr_family"].eq("primary_identification_six").all()
    assert result.loc[0, "bh_q"] == 0.06


def test_primary_contrasts_use_repeat_means_not_draws_as_replicates():
    pairing_rows = []
    fake_rows = []
    for embedding in PRIMARY_EMBEDDINGS:
        for repeat in FORMAL_REPEATS:
            pairing_rows.append(
                {
                    "embedding_model": embedding,
                    "repeat": repeat,
                    "condition": "real",
                    "draw_id": 0,
                    "auc": 0.80,
                }
            )
            for draw_id in range(1, FORMAL_DRAW_COUNT + 1):
                pairing_rows.extend(
                    [
                        {
                            "embedding_model": embedding,
                            "repeat": repeat,
                            "condition": "random",
                            "draw_id": draw_id,
                            "auc": 0.70,
                        },
                        {
                            "embedding_model": embedding,
                            "repeat": repeat,
                            "condition": "matched",
                            "draw_id": draw_id,
                            "auc": 0.75,
                        },
                    ]
                )
            fake_rows.append(
                {
                    "embedding_model": embedding,
                    "repeat": repeat,
                    "condition": "real",
                    "draw_id": 0,
                    "delta_auc": 0.01,
                }
            )
            for draw_id in range(1, FORMAL_DRAW_COUNT + 1):
                fake_rows.append(
                    {
                        "embedding_model": embedding,
                        "repeat": repeat,
                        "condition": "fake_d",
                        "draw_id": draw_id,
                        "delta_auc": 0.00,
                    }
                )

    repeat_table = build_primary_repeat_contrasts(
        pd.DataFrame(pairing_rows), pd.DataFrame(fake_rows)
    )
    assert len(repeat_table) == len(PRIMARY_EMBEDDINGS) * len(FORMAL_REPEATS) * 3
    expected = repeat_table["contrast"].map(
        {"d_delta_minus_fake": 0.01, "real_minus_matched": 0.05, "real_minus_random": 0.10}
    ).to_numpy()
    assert np.isclose(repeat_table["repeat_contrast"].to_numpy(), expected).all()
    summary = summarize_primary_contrasts(repeat_table, n_bootstrap=200, seed=11)
    assert len(summary) == 6
    assert summary["n_repeats"].eq(10).all()
    assert np.isclose(
        summary.sort_values(["embedding_model", "contrast"])["mean"].to_numpy(),
        np.tile(np.array([0.01, 0.05, 0.10]), 2),
    ).all()
