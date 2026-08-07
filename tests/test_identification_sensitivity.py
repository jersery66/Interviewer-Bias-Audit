import numpy as np
import pandas as pd
import pytest

from reanalysis_v2.embedding_incremental import (
    DOMAIN_FEATURES,
    FrozenEmbeddingPair,
    FrozenIncrementalInputs,
    PROTOCOL_FEATURES,
)
from reanalysis_v2.identification_sensitivity import (
    D_EXTRA_FEATURES,
    D_PHQ_FEATURES,
    make_derangement,
    make_fake_domain_permutation,
    make_structure_matched_permutation,
    partition_domains,
    permute_interviewer_pair,
    run_single_level_oof,
    run_level_pair_oof,
)
from reanalysis_v2.splits import make_repeated_split_membership


def _synthetic_structural(n: int = 12) -> pd.DataFrame:
    ids = np.arange(100, 100 + n)
    frame = pd.DataFrame({"participant_id": ids})
    frame["length_only__interviewer_word_count"] = np.repeat([10.0, 100.0], n // 2)
    for index, name in enumerate(
        (
            "protocol_structure__interviewer_question_count",
            "protocol_structure__clinical_question_count",
            "protocol_structure__unique_protocol_prompt_count",
            "protocol_structure__clinical_prompt_coverage_count",
            "protocol_structure__non_explicit_interviewer_turn_count",
        )
    ):
        frame[name] = frame["length_only__interviewer_word_count"] + index
    return frame


def test_partition_domains_is_locked_to_phq_and_extra_groups():
    phq, extra = partition_domains(DOMAIN_FEATURES)

    assert tuple(phq) == D_PHQ_FEATURES
    assert tuple(extra) == D_EXTRA_FEATURES
    assert set(phq).isdisjoint(extra)
    assert set(phq) | set(extra) == set(DOMAIN_FEATURES)


def test_derangement_is_seeded_and_has_no_self_matches():
    first = make_derangement(12, seed=20260713)
    second = make_derangement(12, seed=20260713)

    assert np.array_equal(first, second)
    assert sorted(first.tolist()) == list(range(12))
    assert not np.any(first == np.arange(12))


def test_fake_domain_permutation_preserves_each_column_distribution():
    domains = np.arange(60, dtype=float).reshape(12, 5) % 4
    permutation = make_fake_domain_permutation(len(domains), seed=17)
    fake = domains[permutation]

    assert not np.array_equal(permutation, np.arange(len(domains)))
    for column in range(domains.shape[1]):
        assert np.array_equal(np.sort(fake[:, column]), np.sort(domains[:, column]))


def test_structure_matching_is_one_to_one_nonself_and_beats_random_cost():
    structural = _synthetic_structural()
    columns = [column for column in structural.columns if column != "participant_id"]
    matched = make_structure_matched_permutation(structural, columns=columns, seed=3)
    random = make_derangement(len(structural), seed=3)

    assert sorted(matched.tolist()) == list(range(len(structural)))
    assert not np.any(matched == np.arange(len(structural)))
    values = structural[columns].to_numpy(dtype=float)
    scale = values.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    z = (values - values.mean(axis=0)) / scale
    matched_cost = np.square(z - z[matched]).sum(axis=1).mean()
    random_cost = np.square(z - z[random]).sum(axis=1).mean()
    assert matched_cost <= random_cost


def test_permute_interviewer_pair_reorders_only_interviewer_rows():
    participant = np.arange(12, dtype=float).reshape(4, 3)
    interviewer = (100 + np.arange(12, dtype=float)).reshape(4, 3)
    pair = FrozenEmbeddingPair(
        slug="tiny",
        model_name="tiny",
        revision="locked",
        embedding_dim=3,
        participant=participant,
        interviewer=interviewer,
        cache_files=(),
    )
    permuted = permute_interviewer_pair(pair, np.array([1, 0, 3, 2]))

    assert np.array_equal(permuted.participant, participant)
    assert np.array_equal(permuted.interviewer, interviewer[[1, 0, 3, 2]])
    assert permuted.slug == pair.slug


def test_single_level_oof_returns_one_prediction_per_participant():
    rng = np.random.default_rng(9)
    n = 20
    ids = np.arange(300, 300 + n)
    labels = np.array([0] * 10 + [1] * 10)
    inputs = FrozenIncrementalInputs(
        participant_ids=ids,
        labels=labels,
        protocol=rng.normal(size=(n, len(PROTOCOL_FEATURES))),
        domains=rng.integers(0, 3, size=(n, len(DOMAIN_FEATURES))).astype(float),
    )
    pair = FrozenEmbeddingPair(
        slug="tiny",
        model_name="tiny",
        revision="locked",
        embedding_dim=4,
        participant=rng.normal(size=(n, 4)),
        interviewer=rng.normal(size=(n, 4)),
        cache_files=(),
    )
    membership = make_repeated_split_membership(
        ids, labels, n_repeats=1, n_splits=5, base_seed=99
    )

    result = run_single_level_oof(
        pair=pair,
        inputs=inputs,
        membership=membership,
        repeat=1,
        level="L3",
        augmented=True,
        fusion_method="late_fusion",
        inner_folds=2,
        c_grid=(0.1, 1.0),
        base_seed=101,
    )

    assert len(result) == n
    assert result["participant_id"].nunique() == n
    assert result["probability"].between(0, 1).all()
    assert set(result["model_level"]) == {"L3"}
    assert set(result["model_name"]) == {"augmented"}


def test_level_pair_oof_shares_a_fold_contract_and_returns_base_and_augmented():
    rng = np.random.default_rng(10)
    n = 20
    ids = np.arange(400, 400 + n)
    labels = np.array([0] * 10 + [1] * 10)
    inputs = FrozenIncrementalInputs(
        participant_ids=ids,
        labels=labels,
        protocol=rng.normal(size=(n, len(PROTOCOL_FEATURES))),
        domains=rng.integers(0, 3, size=(n, len(DOMAIN_FEATURES))).astype(float),
    )
    pair = FrozenEmbeddingPair(
        slug="tiny",
        model_name="tiny",
        revision="locked",
        embedding_dim=3,
        participant=rng.normal(size=(n, 3)),
        interviewer=rng.normal(size=(n, 3)),
        cache_files=(),
    )
    membership = make_repeated_split_membership(
        ids, labels, n_repeats=1, n_splits=5, base_seed=101
    )

    result = run_level_pair_oof(
        pair=pair,
        inputs=inputs,
        membership=membership,
        repeat=1,
        level="L3",
        fusion_method="late_fusion",
        inner_folds=2,
        c_grid=(0.1, 1.0),
        base_seed=102,
    )

    assert len(result) == n * 2
    assert set(result["model_name"]) == {"base", "augmented"}
    assert result.groupby("model_name")["participant_id"].nunique().to_dict() == {
        "base": n,
        "augmented": n,
    }


def test_partition_domains_rejects_unknown_or_duplicate_columns():
    with pytest.raises(ValueError, match="unknown"):
        partition_domains([*D_PHQ_FEATURES, "not_a_domain"])
    with pytest.raises(ValueError, match="duplicate"):
        partition_domains([*D_PHQ_FEATURES, D_PHQ_FEATURES[0]])
