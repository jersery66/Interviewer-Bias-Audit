from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


REQUIRED_PRIMARY_COLUMNS = {
    "participant_id",
    "official_avec_split",
    "paper_phq8_score",
    "paper_label_phq8_ge10",
    "condition_id",
    "text",
}


def validate_primary_table(
    table: pd.DataFrame,
    expected_conditions: Sequence[str],
    *,
    expected_n: int = 142,
) -> dict[str, int]:
    """Validate the frozen participant-condition table before any modeling."""
    missing_columns = REQUIRED_PRIMARY_COLUMNS.difference(table.columns)
    if missing_columns:
        raise ValueError(f"missing required columns: {sorted(missing_columns)}")
    if table.empty:
        raise ValueError("primary table is empty")

    split = table["official_avec_split"].astype(str).str.lower()
    if split.eq("test").any():
        raise ValueError("official test participants are prohibited in primary analysis")
    if not split.isin({"train", "dev"}).all():
        invalid = sorted(split[~split.isin({"train", "dev"})].unique())
        raise ValueError(f"invalid primary splits: {invalid}")

    key = ["participant_id", "condition_id"]
    if table.duplicated(key).any():
        raise ValueError("duplicate participant-condition rows detected")

    expected = set(expected_conditions)
    observed = set(table["condition_id"].astype(str).unique())
    if observed != expected:
        raise ValueError(
            f"condition set mismatch: missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )

    n_participants = int(table["participant_id"].nunique())
    if n_participants != expected_n:
        raise ValueError(
            f"participant count mismatch: expected {expected_n}, observed {n_participants}"
        )
    per_participant = table.groupby("participant_id")["condition_id"].nunique()
    if not per_participant.eq(len(expected)).all():
        raise ValueError("not every participant has exactly one row per condition")

    score = pd.to_numeric(table["paper_phq8_score"], errors="raise")
    label = pd.to_numeric(table["paper_label_phq8_ge10"], errors="raise").astype(int)
    recomputed = score.ge(10).astype(int)
    if not label.equals(recomputed):
        raise ValueError("PHQ-8 label drift: label must equal int(score >= 10)")
    if not label.isin({0, 1}).all():
        raise ValueError("labels must be binary")

    participant_consistency = table.assign(_score=score, _label=label).groupby(
        "participant_id"
    )[["_score", "_label", "official_avec_split"]].nunique()
    if participant_consistency.gt(1).any().any():
        raise ValueError("participant labels, scores, or splits vary across conditions")

    participant_rows = table.drop_duplicates("participant_id")
    return {
        "n_rows": int(len(table)),
        "n_participants": n_participants,
        "n_positive": int(
            pd.to_numeric(
                participant_rows["paper_label_phq8_ge10"], errors="raise"
            ).sum()
        ),
        "n_negative": int(
            n_participants
            - pd.to_numeric(
                participant_rows["paper_label_phq8_ge10"], errors="raise"
            ).sum()
        ),
        "n_conditions": len(expected),
    }
