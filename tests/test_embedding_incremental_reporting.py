from __future__ import annotations

import numpy as np
import pandas as pd

from reanalysis_v2.embedding_incremental_reporting import (
    add_global_bh,
    make_incremental_main_table,
)


def _delta_fixture() -> pd.DataFrame:
    rows = []
    index = 0
    for embedding in ("mpnet", "bge"):
        for fusion in ("early_concat", "late_fusion"):
            for level in ("L1", "L2", "L3", "L4"):
                rows.append(
                    {
                        "embedding": embedding,
                        "fusion": fusion,
                        "level": level,
                        "delta_auc": float(index) / 100,
                        "auc_ci_low": -0.1,
                        "auc_ci_high": 0.1,
                        "raw_p": 0.01 + index / 1000,
                        "delta_pr_auc": 0.0,
                        "delta_brier": 0.0,
                        "delta_log_loss": 0.0,
                    }
                )
                index += 1
    return pd.DataFrame(rows)


def test_global_bh_adjusts_all_sixteen_delta_auc_tests():
    result = add_global_bh(_delta_fixture())

    assert result["global_bh_q16"].notna().all()
    assert len(result) == 16
    assert np.all(result["global_bh_q16"].to_numpy() >= result["raw_p"].to_numpy())


def test_main_table_labels_locked_and_global_fdr_families():
    deltas = add_global_bh(_delta_fixture())
    metrics = pd.DataFrame(
        {
            "embedding": ["mpnet", "mpnet"],
            "fusion": ["late_fusion", "late_fusion"],
            "level": ["L1", "L1"],
            "model_name": ["base", "augmented"],
            "roc_auc": [0.7, 0.71],
            "pr_auc": [0.4, 0.41],
            "brier": [0.2, 0.19],
            "log_loss": [0.6, 0.59],
        }
    )
    table = make_incremental_main_table(deltas, metrics)

    assert len(table) == 16
    assert set(table["result_scope"]) == {"exploratory_early_concat", "late_fusion_sensitivity"}
    assert "base_roc_auc" in table.columns
    assert "augmented_roc_auc" in table.columns
    assert "global_bh_q16" in table.columns
