"""Preparation utilities for a blinded D-P content-validity audit.

The functions in this module only prepare the blinded packet and an owner-only
scoring key.  They do not fill reviewer judgments and never infer inter-rater
agreement when a second reviewer file is absent.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .embedding_incremental import DOMAIN_FEATURES


LENGTH_STRATA = ("short", "middle", "long")
FORBIDDEN_BLIND_CASE_COLUMNS = {
    "participant_id",
    "paper_label_phq8_ge10",
    "label",
    "paper_phq8_score",
    "phq8_score",
    "word_count",
    "computed_word_count",
    *DOMAIN_FEATURES,
}


def _require_columns(frame: pd.DataFrame, required: Sequence[str], *, name: str) -> None:
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def make_stratified_sample(
    participants: pd.DataFrame, *, n_per_cell: int = 5, seed: int
) -> pd.DataFrame:
    """Sample a fixed number from label-by-word-count-tertile cells.

    Labels and word counts are used only to construct the owner-only key.  The
    resulting blind cases intentionally omit both fields.
    """

    _require_columns(
        participants,
        ("participant_id", "paper_label_phq8_ge10", "word_count", "text"),
        name="participants",
    )
    if int(n_per_cell) < 1:
        raise ValueError("n_per_cell must be positive")
    frame = participants.copy()
    if frame["participant_id"].duplicated().any():
        raise ValueError("participants contain duplicate participant IDs")
    frame["word_count"] = pd.to_numeric(frame["word_count"], errors="coerce")
    if frame["word_count"].isna().any() or (frame["word_count"] < 0).any():
        raise ValueError("word_count must be finite and non-negative")
    frame["_length_rank"] = frame["word_count"].rank(method="first")
    frame["length_stratum"] = pd.qcut(
        frame["_length_rank"], q=3, labels=LENGTH_STRATA
    ).astype(str)
    labels = sorted(pd.to_numeric(frame["paper_label_phq8_ge10"]).astype(int).unique())
    if set(labels) != {0, 1}:
        raise ValueError("the sample frame must contain both label strata")
    selected: list[pd.DataFrame] = []
    for label in labels:
        for stratum_index, stratum in enumerate(LENGTH_STRATA):
            cell = frame.loc[
                (frame["paper_label_phq8_ge10"].astype(int) == label)
                & frame["length_stratum"].eq(stratum)
            ].copy()
            if len(cell) < int(n_per_cell):
                raise ValueError(
                    f"label={label}, length_stratum={stratum} has only {len(cell)} rows"
                )
            selected.append(
                cell.sample(
                    n=int(n_per_cell),
                    random_state=int(seed) + label * 100 + stratum_index,
                )
            )
    result = pd.concat(selected, ignore_index=True)
    result = result.sort_values(
        ["paper_label_phq8_ge10", "length_stratum", "participant_id"], kind="stable"
    ).reset_index(drop=True)
    result.insert(0, "audit_case_id", [f"DP-AUDIT-{i:03d}" for i in range(1, len(result) + 1)])
    result.insert(1, "sample_order", np.arange(1, len(result) + 1, dtype=int))
    columns = [
        "audit_case_id",
        "sample_order",
        "participant_id",
        "paper_label_phq8_ge10",
        "length_stratum",
        "word_count",
    ]
    if "text_sha256" in result.columns:
        columns.append("text_sha256")
    return result[columns]


def build_blind_cases(participants: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    """Build the reviewer-facing case file with participant language only."""

    _require_columns(participants, ("participant_id", "text"), name="participants")
    _require_columns(sample, ("audit_case_id", "participant_id"), name="sample")
    joined = sample[["audit_case_id", "participant_id"]].merge(
        participants[["participant_id", "text"]],
        on="participant_id",
        how="left",
        validate="one_to_one",
    )
    if joined["text"].isna().any():
        raise ValueError("blind case text is missing")
    cases = joined[["audit_case_id", "text"]].rename(columns={"text": "participant_text"})
    return cases.sort_values("audit_case_id", kind="stable").reset_index(drop=True)


def build_review_form(sample: pd.DataFrame, *, reviewer_id: str) -> pd.DataFrame:
    """Create blank binary/evidence rows for one independent reviewer."""

    _require_columns(sample, ("audit_case_id",), name="sample")
    rows = [
        {
            "reviewer_id": str(reviewer_id),
            "audit_case_id": case_id,
            "domain": domain,
            "present_0_1": pd.NA,
            "evidence_quote": pd.NA,
            "uncertainty_note": pd.NA,
        }
        for case_id in sample["audit_case_id"].tolist()
        for domain in DOMAIN_FEATURES
    ]
    return pd.DataFrame(rows)


def build_scoring_reference(sample: pd.DataFrame, domain_counts: pd.DataFrame) -> pd.DataFrame:
    """Build the owner-only mapping from blinded case IDs to D-P counts."""

    _require_columns(sample, ("audit_case_id", "participant_id"), name="sample")
    _require_columns(domain_counts, ("participant_id", *DOMAIN_FEATURES), name="domain_counts")
    merged = sample[["audit_case_id", "participant_id"]].merge(
        domain_counts[["participant_id", *DOMAIN_FEATURES]],
        on="participant_id",
        how="left",
        validate="one_to_one",
    )
    if merged[list(DOMAIN_FEATURES)].isna().any().any():
        raise ValueError("scoring reference is missing D-P domain counts")
    return merged.sort_values("audit_case_id", kind="stable").reset_index(drop=True)


def validate_blind_packet(cases: pd.DataFrame, form: pd.DataFrame) -> None:
    """Reject reviewer packets that accidentally expose labels or D-P values."""

    if list(cases.columns) != ["audit_case_id", "participant_text"]:
        raise ValueError("blind cases must contain only audit_case_id and participant_text")
    forbidden = FORBIDDEN_BLIND_CASE_COLUMNS.intersection(cases.columns)
    if forbidden:
        raise ValueError(f"blind cases expose forbidden fields: {sorted(forbidden)}")
    _require_columns(
        form,
        ("reviewer_id", "audit_case_id", "domain", "present_0_1", "evidence_quote", "uncertainty_note"),
        name="review form",
    )
    if form[["audit_case_id", "domain"]].duplicated().any():
        raise ValueError("review form contains duplicate case/domain rows")
    if set(form["domain"]) != set(DOMAIN_FEATURES):
        raise ValueError("review form does not contain all ten D-P domains")
    if not set(form["audit_case_id"]).issubset(set(cases["audit_case_id"])):
        raise ValueError("review form contains unknown audit cases")
