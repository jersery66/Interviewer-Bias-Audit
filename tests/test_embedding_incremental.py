from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reanalysis_v2.embedding_incremental import (
    DOMAIN_FEATURES,
    FrozenEmbeddingPair,
    FrozenIncrementalInputs,
    PROTOCOL_FEATURES,
    _run_repeat,
    apply_incremental_fdr,
    build_model_specifications,
    paired_metric_deltas,
    validate_incremental_inputs,
)
from reanalysis_v2.splits import make_repeated_split_membership


def _small_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ids = np.arange(10)
    labels = np.array([0] * 5 + [1] * 5)
    participant = pd.DataFrame({"participant_id": ids, "label": labels})
    protocol = pd.DataFrame({"participant_id": ids})
    for index, name in enumerate(PROTOCOL_FEATURES):
        protocol[name] = ids + index
    domains = pd.DataFrame({"participant_id": ids})
    for index, name in enumerate(DOMAIN_FEATURES):
        domains[name] = (ids + index) % 3
    return participant, protocol, domains


def test_model_specifications_define_the_four_paired_levels():
    specs = build_model_specifications()

    assert list(specs) == ["L1", "L2", "L3", "L4"]
    assert specs["L1"] == ("P", "P+I")
    assert specs["L2"] == ("P+R", "P+R+I")
    assert specs["L3"] == ("P+D", "P+D+I")
    assert specs["L4"] == ("P+R+D", "P+R+D+I")


def test_validate_incremental_inputs_requires_same_ids_and_frozen_columns():
    participant, protocol, domains = _small_inputs()

    merged = validate_incremental_inputs(
        participant, protocol, domains, expected_n=10
    )

    assert list(merged.participant_id) == list(range(10))
    assert set(PROTOCOL_FEATURES).issubset(merged.columns)
    assert set(DOMAIN_FEATURES).issubset(merged.columns)

    with pytest.raises(ValueError, match="participant IDs"):
        validate_incremental_inputs(
            participant,
            protocol.assign(participant_id=lambda frame: frame.participant_id + 100),
            domains,
            expected_n=10,
        )


def test_paired_metric_deltas_use_positive_improvement_direction():
    y = np.array([0, 0, 0, 1, 1, 1])
    base = np.array([0.2, 0.4, 0.6, 0.4, 0.6, 0.8])
    augmented = np.array([0.1, 0.3, 0.5, 0.5, 0.7, 0.9])

    result = paired_metric_deltas(y, base, augmented)

    assert result["delta_auc"] > 0
    assert result["delta_pr_auc"] > 0
    assert result["delta_brier"] > 0
    assert result["delta_log_loss"] > 0


def test_incremental_fdr_keeps_primary_and_secondary_families_separate():
    table = pd.DataFrame(
        {
            "fdr_family": ["primary_l4", "primary_l4", "secondary_all_delta_auc"],
            "raw_p": [0.01, 0.04, 0.02],
        }
    )

    result = apply_incremental_fdr(table)

    assert result.loc[0, "bh_q"] == pytest.approx(0.02)
    assert result.loc[1, "bh_q"] == pytest.approx(0.04)
    assert result.loc[2, "bh_q"] == pytest.approx(0.02)


def test_synthetic_repeat_produces_paired_rows_for_both_fusion_methods():
    rng = np.random.default_rng(7)
    participant_ids = np.arange(100, 120)
    labels = np.array([0] * 10 + [1] * 10)
    inputs = FrozenIncrementalInputs(
        participant_ids=participant_ids,
        labels=labels,
        protocol=rng.normal(size=(20, len(PROTOCOL_FEATURES))),
        domains=rng.integers(0, 4, size=(20, len(DOMAIN_FEATURES))).astype(float),
    )
    pair = FrozenEmbeddingPair(
        slug="test_tiny",
        model_name="test/tiny",
        revision="locked",
        embedding_dim=4,
        participant=rng.normal(size=(20, 4)),
        interviewer=rng.normal(size=(20, 4)),
        cache_files=(),
    )
    membership = make_repeated_split_membership(
        participant_ids, labels, n_repeats=1, n_splits=5, base_seed=99
    )

    for fusion_method in ("late_fusion", "early_concat"):
        predictions, _ = _run_repeat(
            pair=pair,
            inputs=inputs,
            membership=membership,
            repeat=1,
            fusion_method=fusion_method,
            inner_folds=2,
            c_grid=(0.1, 1.0),
            base_seed=101,
        )
        assert len(predictions) == 20 * 4 * 2
        assert set(predictions["model_level"]) == {"L1", "L2", "L3", "L4"}
        assert set(predictions["model_name"]) == {"base", "augmented"}
