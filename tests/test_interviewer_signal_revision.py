from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reanalysis_v2.interviewer_signal_revision import (
    C5_HUMAN_FIELDS,
    build_block_increment_contrasts,
    build_c5_human_validation_dataset,
    build_feature_block_source_table,
    run_r_absorption_sensitivity,
    select_c5_validation_participants,
)
from reanalysis_v2.interviewer_signal_explanation import (
    D_P_FEATURES,
    ExplanationInputs,
    SourceFoldCache,
)
from reanalysis_v2.splits import make_repeated_split_membership, split_indices


def _synthetic_c5_spans() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "participant_id": [101, 101, 102],
            "exact_quote": ["I cannot sleep", "I feel guilty", "I am okay"],
            "domain": ["sleep_fatigue_energy", "self_worth_guilt", "protective_or_absent_symptom"],
            "polarity": ["present", "present", "absent"],
            "review_status": ["reviewed"] * 3,
            "original_model_quote": ["I cannot sleep", "I feel guilty", "I am okay"],
            "source_text_sha256": ["a", "b", "c"],
            "quote_sha256": ["qa", "qb", "qc"],
            "source_turn_index": [1, 2, 1],
            "source_char_start": [0, 5, 0],
            "source_char_end": [13, 17, 9],
        }
    )


def _synthetic_sampling_inputs(n: int = 36) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = np.arange(1000, 1000 + n)
    labels = np.asarray([0, 1] * (n // 2), dtype=int)
    structural = pd.DataFrame(
        {
            "participant_id": ids,
            "label": labels,
            "length_only__participant_word_count": np.arange(10, 10 + n),
        }
    )
    domains = pd.DataFrame({"participant_id": ids})
    for index, name in enumerate(
        [
            "anhedonia_interest",
            "appetite_weight",
            "concentration_psychomotor",
            "depressed_mood",
            "functioning_impairment",
            "mental_health_history",
            "protective_or_absent_symptom",
            "self_worth_guilt",
            "sleep_fatigue_energy",
            "suicide_self_harm",
        ]
    ):
        domains[name] = ((ids + index) % (index + 2) == 0).astype(float)
    return structural, domains


def test_c5_candidate_contains_model_fields_and_blank_human_fields() -> None:
    roster = pd.DataFrame({"participant_id": [101, 102]})
    candidate = build_c5_human_validation_dataset(
        _synthetic_c5_spans(), roster, source_sha256="source-hash"
    )

    assert set(C5_HUMAN_FIELDS).issubset(candidate.columns)
    assert candidate["model_exact_quote"].notna().all()
    for field in C5_HUMAN_FIELDS:
        assert candidate[field].fillna("").eq("").all(), field
    assert "transcript" not in " ".join(candidate.columns).lower()
    assert candidate["source_sha256"].eq("source-hash").all()
    assert not set(candidate.columns).intersection(
        {"exact_quote", "original_model_quote", "speaker_text", "transcript_text"}
    )


def test_c5_candidate_is_deterministic_and_does_not_infer_human_labels() -> None:
    roster = pd.DataFrame({"participant_id": [101, 102]})
    first = build_c5_human_validation_dataset(
        _synthetic_c5_spans(), roster, source_sha256="source-hash"
    )
    second = build_c5_human_validation_dataset(
        _synthetic_c5_spans(), roster, source_sha256="source-hash"
    )
    pd.testing.assert_frame_equal(first, second)
    assert first["human_domain"].eq("").all()
    assert first["human_polarity"].eq("").all()
    assert first["human_span_valid"].eq("").all()


def test_c5_sampling_is_deterministic_and_label_stratified_without_model_fields() -> None:
    structural, domains = _synthetic_sampling_inputs()
    first = select_c5_validation_participants(structural, domains, n=12)
    second = select_c5_validation_participants(structural, domains, n=12)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 12
    assert set(first["label"]) == {0, 1}
    assert "model_correct" not in first.columns
    assert "source_score" not in first.columns


def test_block_source_table_locks_r_as_protocol_structure() -> None:
    class Inputs:
        block_columns = {
            "P": ("participant_logit",),
            "Q": ("q_length",),
            "R": ("clinical_question_count", "prompt_presence_x"),
            "D": ("sleep_fatigue_energy",),
        }

    table = build_feature_block_source_table(Inputs())
    r = table.loc[table["block"].eq("R")]
    assert set(r["feature"]) == {"clinical_question_count", "prompt_presence_x"}
    assert r["source_side"].eq("interviewer-side protocol structure").all()
    assert not table.astype(str).apply(lambda col: col.str.contains("causal", case=False).any()).any()


def test_block_increment_contrasts_are_derived_from_locked_subset_r2() -> None:
    rows = []
    values = {
        "P": 0.10,
        "Q": 0.20,
        "R": 0.30,
        "D": 0.40,
        "P+R": 0.50,
        "R+D": 0.60,
        "P+R+D": 0.70,
        "P+Q+R+D": 0.80,
    }
    for subset, r2 in values.items():
        rows.append({"representation": "tfidf", "repeat": 1, "subset": subset, "r2": r2})
    result = build_block_increment_contrasts(pd.DataFrame(rows))
    row = result.iloc[0]
    assert row["full_r2"] == pytest.approx(0.80)
    assert row["full_minus_R_r2"] == pytest.approx(0.50)
    assert row["P+R_minus_R_r2"] == pytest.approx(0.20)
    assert row["R+D_minus_R_r2"] == pytest.approx(0.30)
    assert row["P+R+D_minus_R_r2"] == pytest.approx(0.40)
    assert row["P+Q+R+D_minus_R_r2"] == pytest.approx(0.50)


def test_block_increment_requires_all_locked_contrasts() -> None:
    with pytest.raises(ValueError, match="missing locked subset"):
        build_block_increment_contrasts(
            pd.DataFrame(
                {
                    "representation": ["tfidf"],
                    "repeat": [1],
                    "subset": ["P+Q+R+D"],
                    "r2": [0.1],
                }
            )
        )


def test_r_absorption_uses_fold_local_residuals_and_reports_locked_models() -> None:
    rng = np.random.default_rng(19)
    n = 30
    ids = np.arange(2000, 2000 + n)
    labels = np.asarray([0] * 15 + [1] * 15)
    q = rng.normal(size=(n, 2))
    r = rng.normal(size=(n, 2))
    d = rng.normal(size=(n, 3))
    structural = pd.DataFrame({"participant_id": ids, "label": labels, "phq8_score": labels})
    domains = pd.DataFrame({"participant_id": ids})
    for index, name in enumerate(D_P_FEATURES):
        domains[name] = d[:, index % 3]
    inputs = ExplanationInputs(
        participant_ids=ids,
        labels=labels,
        structural=structural,
        domains=domains,
        path_features=pd.DataFrame({"participant_id": ids}),
        block_matrices={"Q": q, "R": r, "D": d},
        block_columns={"P": ("participant_logit",), "Q": ("q0", "q1"), "R": ("r0", "r1"), "D": ("d0", "d1", "d2")},
        structural_audit=pd.DataFrame(),
        structural_audit_summary={},
        path_dictionary=(),
        path_audit=pd.DataFrame(),
        path_audit_summary={},
        domain_audit=pd.DataFrame(),
        domain_audit_summary={},
    )
    membership = make_repeated_split_membership(ids, labels, n_repeats=1, n_splits=5, base_seed=41)
    cache = {}
    for fold in range(1, 6):
        train_idx, test_idx = split_indices(membership, 1, fold, ids)
        p_train = labels[train_idx].astype(float) + rng.normal(scale=0.3, size=len(train_idx))
        p_test = labels[test_idx].astype(float) + rng.normal(scale=0.3, size=len(test_idx))
        i_train = 0.4 * labels[train_idx].astype(float) + rng.normal(scale=0.3, size=len(train_idx))
        i_test = 0.4 * labels[test_idx].astype(float) + rng.normal(scale=0.3, size=len(test_idx))
        cache[("tfidf", 1, fold)] = SourceFoldCache(
            train_idx=train_idx,
            test_idx=test_idx,
            participant_train_logit=p_train,
            participant_test_logit=p_test,
            interviewer_train_logit=i_train,
            interviewer_test_logit=i_test,
            participant_c=1.0,
            interviewer_c=1.0,
            tuning=(),
        )

    oof, metrics, _ = run_r_absorption_sensitivity(
        inputs,
        membership,
        Path("."),
        source_cache=cache,
        alpha_grid=(1.0,),
        label_c_grid=(1.0,),
        base_seed=7,
    )
    assert set(metrics.loc[metrics["metric_scope"].eq("label_prediction"), "model_name"]) == {
        "P+Q+D",
        "P+Q+D+residual_without_R",
        "P+Q+R+D",
        "P+Q+R+D+residual_full",
    }
    assert np.allclose(
        oof["residual_without_R"],
        oof["interviewer_logit"] - oof["predicted_interviewer_logit_without_R"],
    )
    assert np.allclose(
        oof["residual_full"],
        oof["interviewer_logit"] - oof["predicted_interviewer_logit_full"],
    )
