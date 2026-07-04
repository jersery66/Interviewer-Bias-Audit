from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def make_repeated_split_membership(
    participant_ids: np.ndarray,
    labels: np.ndarray,
    *,
    n_repeats: int = 10,
    n_splits: int = 5,
    base_seed: int = 20260624,
) -> pd.DataFrame:
    participant_ids = np.asarray(participant_ids)
    labels = np.asarray(labels, dtype=int)
    if len(participant_ids) != len(labels):
        raise ValueError("participant_ids and labels must have the same length")
    if len(np.unique(participant_ids)) != len(participant_ids):
        raise ValueError("participant_ids must be unique")

    rows: list[dict[str, int | str]] = []
    dummy = np.zeros(len(labels), dtype=np.int8)
    for repeat in range(1, n_repeats + 1):
        split_seed = base_seed + repeat
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=split_seed
        )
        for fold, (train_idx, test_idx) in enumerate(
            splitter.split(dummy, labels), start=1
        ):
            for role, indices in (("train", train_idx), ("test", test_idx)):
                for index in indices:
                    rows.append(
                        {
                            "repeat": repeat,
                            "fold": fold,
                            "role": role,
                            "participant_id": participant_ids[index],
                            "label": int(labels[index]),
                            "split_seed": split_seed,
                        }
                    )
    return pd.DataFrame(rows).sort_values(
        ["repeat", "fold", "role", "participant_id"], kind="stable"
    ).reset_index(drop=True)


def split_indices(
    membership: pd.DataFrame,
    repeat: int,
    fold: int,
    participant_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    participant_ids = np.asarray(participant_ids)
    lookup = {participant_id: i for i, participant_id in enumerate(participant_ids)}
    selected = membership[
        (membership["repeat"] == repeat) & (membership["fold"] == fold)
    ]
    if selected.empty:
        raise ValueError(f"missing split repeat={repeat}, fold={fold}")

    def indices_for(role: str) -> np.ndarray:
        ids = selected.loc[selected["role"] == role, "participant_id"].tolist()
        unknown = [participant_id for participant_id in ids if participant_id not in lookup]
        if unknown:
            raise ValueError(f"split contains unknown participant IDs: {unknown[:5]}")
        return np.asarray([lookup[participant_id] for participant_id in ids], dtype=int)

    train_idx, test_idx = indices_for("train"), indices_for("test")
    if set(train_idx).intersection(test_idx):
        raise ValueError("train/test overlap detected")
    if set(train_idx).union(test_idx) != set(range(len(participant_ids))):
        raise ValueError("train/test split does not cover the participant list")
    return train_idx, test_idx
