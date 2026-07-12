from __future__ import annotations

import numpy as np
import pandas as pd

from reanalysis_v2.submission_audit import (
    audit_c5_traceability,
    build_structural_features,
    build_token_matched_draw,
    select_locked_split,
    summarize_locked_source_predictions,
    summarize_token_matched_draws,
)


def test_select_locked_split_keeps_one_test_prediction_per_participant() -> None:
    rows = []
    for repeat in (1, 2):
        for fold in (1, 2):
            for participant_id in (10, 11, 12, 13):
                assigned = 1 + ((participant_id + repeat) % 2)
                rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "participant_id": participant_id,
                        "role": "test" if fold == assigned else "train",
                    }
                )
    locked = select_locked_split(pd.DataFrame(rows), repeat=1)

    assert locked["repeat"].eq(1).all()
    test_counts = locked.loc[locked["role"].eq("test")].groupby("participant_id").size()
    assert test_counts.eq(1).all()


def test_build_structural_features_uses_only_process_counts() -> None:
    turns = pd.DataFrame(
        [
            {
                "participant_id": 10,
                "paper_label_phq8_ge10": 0,
                "paper_phq8_score": 4,
                "turn_index": 1,
                "start_time": 0.0,
                "stop_time": 2.0,
                "speaker_norm": "interviewer",
                "word_count": 4,
                "is_prompt_annotated": 1,
                "is_explicit_clinical_prompt": 1,
                "prompt_id": "P1",
                "normalized_spoken_text": "how are you sleeping",
            },
            {
                "participant_id": 10,
                "paper_label_phq8_ge10": 0,
                "paper_phq8_score": 4,
                "turn_index": 2,
                "start_time": 2.0,
                "stop_time": 5.0,
                "speaker_norm": "participant",
                "word_count": 8,
                "is_prompt_annotated": 0,
                "is_explicit_clinical_prompt": 0,
                "prompt_id": np.nan,
                "normalized_spoken_text": "participant words must not enter output",
            },
            {
                "participant_id": 10,
                "paper_label_phq8_ge10": 0,
                "paper_phq8_score": 4,
                "turn_index": 3,
                "start_time": 5.0,
                "stop_time": 6.0,
                "speaker_norm": "interviewer",
                "word_count": 1,
                "is_prompt_annotated": 0,
                "is_explicit_clinical_prompt": 0,
                "prompt_id": np.nan,
                "normalized_spoken_text": "mhm",
            },
        ]
    )
    features = build_structural_features(turns)

    assert features.shape[0] == 1
    row = features.iloc[0]
    assert row["length_only__participant_word_count"] == 8
    assert row["length_only__interviewer_word_count"] == 5
    assert row["interaction_structure__total_turn_count"] == 3
    assert row["protocol_structure__interviewer_question_count"] == 1
    assert row["protocol_structure__clinical_prompt_coverage_count"] == 1
    assert not any("text" in column.lower() for column in features.columns)


def test_token_matched_draw_is_deterministic_and_equal_length() -> None:
    participant = pd.DataFrame(
        {
            "participant_id": [10, 11],
            "paper_label_phq8_ge10": [0, 1],
            "text": ["p0 p1 p2 p3 p4 p5", "a b"],
        }
    )
    interviewer = pd.DataFrame(
        {
            "participant_id": [10, 11],
            "paper_label_phq8_ge10": [0, 1],
            "text": ["i0 i1 i2", "c d e f"],
        }
    )
    first, audit_first = build_token_matched_draw(
        participant, interviewer, draw=3, seed=17
    )
    second, audit_second = build_token_matched_draw(
        participant, interviewer, draw=3, seed=17
    )

    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(audit_first, audit_second)
    for row in first.itertuples(index=False):
        assert len(row.participant_matched_text.split()) == len(
            row.interviewer_matched_text.split()
        )
    assert audit_first["matched_token_count"].tolist() == [3, 2]


def test_traceability_audit_never_emits_quote_text() -> None:
    spans = pd.DataFrame(
        {
            "participant_id": [10, 10, 11],
            "source_order": [1, 2, 1],
            "domain": ["mood", "sleep", "mood"],
            "polarity": ["present", "present", "present"],
            "exact_quote": ["exact source words", "repeat phrase", "missing phrase"],
            "review_status": [
                "base_validated_quote",
                "manual_pass_corrected_quote_added",
                "base_validated_quote",
            ],
            "match_status": [
                "matched_normalized_whitespace",
                "manual_review_corrected_to_participant_source",
                "matched_normalized_whitespace",
            ],
        }
    )
    participant = pd.DataFrame(
        {
            "participant_id": [10, 11],
            "text": [
                "exact source words then repeat phrase and repeat phrase",
                "other source",
            ],
            "text_sha256": ["source-hash-10", "source-hash-11"],
        }
    )
    audit = audit_c5_traceability(spans, participant)

    assert "exact_quote" not in audit.columns
    assert "quote_sha256" in audit.columns
    assert audit["location_status"].tolist() == [
        "unique_exact_match",
        "multiple_exact_matches",
        "not_found",
    ]
    assert audit.loc[0, "normalized_char_start"] == 0
    assert pd.isna(audit.loc[2, "normalized_char_start"])


def test_locked_source_summary_uses_fold_specific_nested_thresholds() -> None:
    frame = pd.DataFrame(
        {
            "condition": ["participant"] * 6,
            "participant_id": np.arange(6),
            "label": [0, 0, 0, 1, 1, 1],
            "raw_probability": [0.1, 0.2, 0.45, 0.4, 0.7, 0.9],
            "platt_probability": [0.1, 0.2, 0.3, 0.6, 0.7, 0.9],
            "threshold_max_macro_f1": [0.35] * 6,
        }
    )
    summary = summarize_locked_source_predictions(frame, n_bootstrap=100, seed=23)

    assert summary.loc[0, "n"] == 6
    assert summary.loc[0, "sensitivity_nested"] == 1.0
    assert summary.loc[0, "specificity_nested"] == 2 / 3
    assert summary.loc[0, "auc_ci_low"] <= summary.loc[0, "roc_auc"]
    assert summary.loc[0, "auc_ci_high"] >= summary.loc[0, "roc_auc"]


def test_token_draw_summary_uses_mean_draw_performance_not_probability_ensemble() -> None:
    rows = []
    metrics = []
    labels = [0, 0, 1, 1]
    probabilities = {
        1: {
            "participant_matched_prob": [0.1, 0.4, 0.6, 0.9],
            "interviewer_matched_prob": [0.1, 0.2, 0.8, 0.9],
        },
        2: {
            "participant_matched_prob": [0.4, 0.1, 0.9, 0.6],
            "interviewer_matched_prob": [0.2, 0.1, 0.9, 0.8],
        },
    }
    for draw, values in probabilities.items():
        for participant_id, label in enumerate(labels):
            rows.append(
                {
                    "draw": draw,
                    "participant_id": participant_id,
                    "label": label,
                    "participant_matched_prob": values["participant_matched_prob"][participant_id],
                    "interviewer_matched_prob": values["interviewer_matched_prob"][participant_id],
                }
            )
        for condition in ("participant_matched", "interviewer_matched"):
            metrics.append(
                {
                    "draw": draw,
                    "condition": condition,
                    "roc_auc": 1.0,
                    "pr_auc": 1.0,
                    "brier": 0.1,
                }
            )
    summary, comparison = summarize_token_matched_draws(
        pd.DataFrame(rows), pd.DataFrame(metrics), n_bootstrap=100, seed=11
    )

    assert summary["mean_auc"].eq(1.0).all()
    assert summary["n_draws"].eq(2).all()
    assert comparison.loc[0, "mean_delta_auc"] == 0.0
    assert "participant_random_draw_ci_low" in comparison.columns


def test_token_draw_summary_reports_participant_paired_permutation_p_and_q() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    rows = []
    metrics = []
    for draw in (1, 2):
        participant_probability = np.array([0.10, 0.55, 0.60, 0.40, 0.45, 0.90])
        interviewer_probability = np.array([0.05, 0.10, 0.20, 0.80, 0.90, 0.95])
        for participant_id, label in enumerate(labels):
            rows.append(
                {
                    "draw": draw,
                    "participant_id": participant_id,
                    "label": label,
                    "participant_matched_prob": participant_probability[participant_id],
                    "interviewer_matched_prob": interviewer_probability[participant_id],
                }
            )
        for condition, auc in (("participant_matched", 5 / 9), ("interviewer_matched", 1.0)):
            metrics.append(
                {
                    "draw": draw,
                    "condition": condition,
                    "roc_auc": auc,
                    "pr_auc": 0.5,
                    "brier": 0.2,
                }
            )

    _, comparison = summarize_token_matched_draws(
        pd.DataFrame(rows),
        pd.DataFrame(metrics),
        n_bootstrap=20,
        n_permutations=199,
        seed=31,
        permutation_seed=47,
    )

    assert comparison.loc[0, "n_permutations"] == 199
    assert 0.0 < comparison.loc[0, "p_value"] <= 1.0
    assert comparison.loc[0, "q_value"] == comparison.loc[0, "p_value"]
    assert comparison.loc[0, "permutation_pairing"] == (
        "one participant swap vector held constant across all draws"
    )
