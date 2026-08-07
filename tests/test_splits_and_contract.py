import numpy as np
import pandas as pd
import pytest

from reanalysis_v2.contracts import validate_primary_table
from reanalysis_v2.splits import make_repeated_split_membership, split_indices


CONDITIONS = [
    "full_transcript",
    "participant_speech",
    "interviewer_speech",
    "non_explicit_interviewer_speech",
    "participant_symptom_evidence",
]


def test_repeated_split_membership_is_deterministic_and_complete():
    participant_ids = np.arange(100, 120)
    labels = np.array([0, 1] * 10)

    first = make_repeated_split_membership(
        participant_ids, labels, n_repeats=2, n_splits=5, base_seed=20260624
    )
    second = make_repeated_split_membership(
        participant_ids, labels, n_repeats=2, n_splits=5, base_seed=20260624
    )

    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == [
        "repeat",
        "fold",
        "role",
        "participant_id",
        "label",
        "split_seed",
    ]
    assert len(first) == 2 * 5 * 20
    for repeat in (1, 2):
        test_rows = first[(first.repeat == repeat) & (first.role == "test")]
        assert test_rows.participant_id.value_counts().eq(1).all()
        assert set(test_rows.participant_id) == set(participant_ids)
        for fold in range(1, 6):
            train_idx, test_idx = split_indices(first, repeat, fold, participant_ids)
            assert not set(train_idx) & set(test_idx)
            assert set(train_idx) | set(test_idx) == set(range(20))


def test_primary_contract_rejects_test_members_duplicates_and_label_drift():
    rows = []
    for participant_id in (301, 302):
        for condition in CONDITIONS:
            rows.append(
                {
                    "participant_id": participant_id,
                    "official_avec_split": "train" if participant_id == 301 else "dev",
                    "paper_phq8_score": 12 if participant_id == 301 else 3,
                    "paper_label_phq8_ge10": 1 if participant_id == 301 else 0,
                    "condition_id": condition,
                    "text": f"text {participant_id} {condition}",
                }
            )
    valid = pd.DataFrame(rows)
    summary = validate_primary_table(valid, CONDITIONS, expected_n=2)
    assert summary["n_participants"] == 2
    assert summary["n_rows"] == 10

    with_test = valid.copy()
    with_test.loc[0, "official_avec_split"] = "test"
    with pytest.raises(ValueError, match="official test"):
        validate_primary_table(with_test, CONDITIONS, expected_n=2)

    duplicate = pd.concat([valid, valid.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_primary_table(duplicate, CONDITIONS, expected_n=2)

    drift = valid.copy()
    drift.loc[drift.participant_id == 302, "paper_label_phq8_ge10"] = 1
    with pytest.raises(ValueError, match="PHQ-8"):
        validate_primary_table(drift, CONDITIONS, expected_n=2)

