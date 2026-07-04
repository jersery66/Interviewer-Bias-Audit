from pathlib import Path

import numpy as np
import pandas as pd

from reanalysis_v2.control_analysis import (
    build_c4_control_inputs,
    build_c4_family_tests,
    build_c5_control_inputs,
    build_c5_family_tests,
    run_fold_safe_shuffled_text_condition,
    run_raw_feature_condition,
    run_raw_text_condition,
)
from reanalysis_v2.splits import make_repeated_split_membership


def test_raw_control_runners_use_frozen_splits_for_text_and_structured_features():
    participant_ids = np.arange(100, 120)
    labels = np.array([0, 1] * 10)
    texts = np.array(
        ["common low" if label == 0 else "common high" for label in labels],
        dtype=object,
    )
    features = np.column_stack([labels, 1 - labels, np.arange(20) % 3])
    membership = make_repeated_split_membership(
        participant_ids,
        labels,
        n_repeats=1,
        n_splits=5,
        base_seed=20260624,
    )
    text_result = run_raw_text_condition(
        participant_ids,
        labels,
        texts,
        membership,
        condition="text_control",
        expected_repeats=1,
    )
    feature_result = run_raw_feature_condition(
        participant_ids,
        labels,
        features,
        membership,
        condition="feature_control",
        expected_repeats=1,
    )
    for result in (text_result, feature_result):
        assert len(result["fold_predictions"]) == 20
        assert len(result["participant_predictions"]) == 20
        assert result["fold_predictions"].participant_id.value_counts().eq(1).all()
        assert result["summary"]["condition"].endswith("control")


def test_c5_control_builder_exactly_matches_words_excludes_quotes_and_builds_domains(tmp_path: Path):
    participant = pd.DataFrame(
        {
            "participant_id": [1, 2, 3, 4],
            "official_avec_split": ["train", "train", "dev", "dev"],
            "paper_label_phq8_ge10": [0, 1, 0, 1],
            "paper_phq8_score": [2, 12, 3, 15],
            "condition_id": ["participant_speech"] * 4,
            "text": [
                "alpha depressed beta gamma delta epsilon",
                "one two tired three four five six",
                "red blue green yellow black white",
                "cat dog sleep bird fish horse",
            ],
        }
    )
    reviewed = participant.copy()
    reviewed["condition_id"] = "participant_symptom_evidence"
    reviewed["text"] = ["depressed beta", "tired three", "", "sleep bird"]
    spans = pd.DataFrame(
        {
            "participant_id": [1, 2, 4],
            "source_order": [1, 1, 1],
            "domain": [
                "depressed_mood",
                "sleep_fatigue_energy",
                "sleep_fatigue_energy",
            ],
            "exact_quote": ["depressed beta", "tired three", "sleep bird"],
        }
    )
    result = build_c5_control_inputs(
        participant,
        reviewed,
        spans,
        random_master_seeds=[11, 12],
        nonsymptom_master_seed=13,
        keywords=["depressed", "tired", "sleep"],
    )
    target = reviewed.set_index("participant_id").text.str.split().str.len()
    for table in result["random_tables"].values():
        observed = table.set_index("participant_id").text.str.split().str.len()
        pd.testing.assert_series_equal(observed, target, check_names=False)
    nonsymptom = result["nonsymptom"].set_index("participant_id").text
    assert "depressed" not in nonsymptom.loc[1]
    assert "beta" not in nonsymptom.loc[1]
    assert "tired" not in nonsymptom.loc[2]
    assert result["quote_exclusion_audit"].unmatched_quotes.sum() == 0
    assert result["domain_presence"].shape == (4, 11)
    assert result["domain_count"].shape == (4, 11)
    masked = result["keyword_masked"].set_index("participant_id").text
    assert masked.loc[1].count("[MASKED]") == 1


def test_c5_family_tests_include_each_seed_ensemble_and_four_named_controls():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    def table(offset: float):
        probability = np.clip(
            np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]) + offset,
            0,
            1,
        )
        return pd.DataFrame(
            {
                "participant_id": np.arange(8),
                "label": y,
                "raw_probability": probability,
                "prediction_at_0_5": (probability >= 0.5).astype(int),
            }
        )

    reviewed = table(0)
    random = {"seed_01": table(0.01), "seed_02": table(-0.01)}
    controls = {
        "random_ensemble": table(0),
        "nonsymptom_same_word": table(0.02),
        "domain_presence": table(-0.02),
        "domain_count": table(0.03),
        "keyword_masked": table(-0.03),
    }
    tests = build_c5_family_tests(
        reviewed,
        random,
        controls,
        n_permutations=9,
        base_seed=70,
    )
    assert len(tests) == (2 + 5) * 2
    assert tests.family.eq("c5_evidence_controls").all()
    assert tests.fdr_bh_q.between(0, 1).all()


def test_c4_builder_uses_explicit_prompts_and_exact_length_position_controls():
    participant_ids = [1, 2]
    turns = pd.DataFrame(
        {
            "participant_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "turn_index": [1, 2, 3, 4, 1, 2, 3, 4],
            "speaker_norm": ["interviewer"] * 8,
            "normalized_spoken_text": [
                "hello there",
                "do you feel down",
                "tell me more now",
                "how is work today",
                "welcome friend",
                "how is your sleep",
                "what happened next",
                "thanks for sharing",
            ],
            "is_explicit_clinical_prompt": [0, 1, 0, 0, 0, 1, 0, 0],
            "is_protocol_template_spoken": [1] * 8,
        }
    )
    result = build_c4_control_inputs(
        turns, participant_ids, length_master_seed=10
    )
    base = result["participant_texts"].set_index("participant_id")
    assert base.loc[1, "template_only"] == "do you feel down"
    assert base.loc[2, "template_only"] == "how is your sleep"
    assert base.template_word_count.equals(base.lengthmatched_word_count)
    assert base.template_word_count.equals(base.positionmatched_word_count)
    assert result["template_presence"].shape[0] == 2
    assert result["template_id_map"].template_id.nunique() == 2


def test_fold_safe_shuffled_runner_saves_role_safe_mapping():
    participant_ids = np.arange(100, 120)
    labels = np.array([0, 1] * 10)
    texts = {participant_id: f"template {participant_id}" for participant_id in participant_ids}
    membership = make_repeated_split_membership(
        participant_ids,
        labels,
        n_repeats=1,
        n_splits=5,
        base_seed=20260624,
    )
    result = run_fold_safe_shuffled_text_condition(
        participant_ids,
        labels,
        texts,
        membership,
        condition="template_shuffled",
        expected_repeats=1,
        base_seed=20260624,
    )
    assert len(result["fold_predictions"]) == 20
    assert len(result["participant_predictions"]) == 20
    mapping = result["shuffle_mapping"]
    assert len(mapping) == 5 * 20
    assert (mapping.recipient_role == mapping.donor_role).all()


def test_c4_family_has_four_controls_times_two_metrics():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    def table(offset):
        p = np.clip(
            np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]) + offset,
            0,
            1,
        )
        return pd.DataFrame(
            {
                "participant_id": np.arange(8),
                "label": y,
                "raw_probability": p,
                "prediction_at_0_5": (p >= 0.5).astype(int),
            }
        )

    tests = build_c4_family_tests(
        table(0),
        {
            "nontemplate_lengthmatched": table(0.01),
            "nontemplate_positionmatched": table(-0.01),
            "template_shuffled": table(0.02),
            "template_presence": table(-0.02),
        },
        n_permutations=9,
        base_seed=10,
    )
    assert len(tests) == 8
    assert tests.family.eq("c4_interviewer_controls").all()
