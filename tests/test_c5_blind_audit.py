import numpy as np
import pandas as pd

from reanalysis_v2.embedding_incremental import DOMAIN_FEATURES
from reanalysis_v2.c5_blind_audit import (
    build_blind_cases,
    build_review_form,
    build_scoring_reference,
    make_stratified_sample,
    validate_blind_packet,
)


def _small_participants() -> pd.DataFrame:
    rows = []
    for label in (0, 1):
        for length_index, word_count in enumerate((10, 20, 30)):
            rows.append(
                {
                    "participant_id": label * 10 + length_index,
                    "paper_label_phq8_ge10": label,
                    "word_count": word_count,
                    "text": f"participant {label} length {length_index}",
                    "text_sha256": f"hash-{label}-{length_index}",
                }
            )
    return pd.DataFrame(rows)


def test_stratified_sample_is_seeded_and_balances_label_by_length_cells():
    participants = _small_participants()

    first = make_stratified_sample(participants, n_per_cell=1, seed=20260714)
    second = make_stratified_sample(participants, n_per_cell=1, seed=20260714)

    assert first.equals(second)
    assert len(first) == 6
    assert first.groupby(["paper_label_phq8_ge10", "length_stratum"]).size().eq(1).all()
    assert first["audit_case_id"].is_unique


def test_blind_packet_excludes_labels_ids_word_counts_and_existing_domain_counts():
    participants = _small_participants()
    sample = make_stratified_sample(participants, n_per_cell=1, seed=3)

    cases = build_blind_cases(participants, sample)
    form = build_review_form(sample, reviewer_id="reviewer_a")

    assert list(cases.columns) == ["audit_case_id", "participant_text"]
    assert len(form) == len(sample) * len(DOMAIN_FEATURES)
    assert set(form["domain"]) == set(DOMAIN_FEATURES)
    assert form["present_0_1"].isna().all()
    validate_blind_packet(cases, form)


def test_scoring_reference_restores_ids_only_for_the_analysis_owner():
    participants = _small_participants()
    sample = make_stratified_sample(participants, n_per_cell=1, seed=4)
    domains = pd.DataFrame({"participant_id": participants["participant_id"]})
    for index, domain in enumerate(DOMAIN_FEATURES):
        domains[domain] = (domains["participant_id"] + index) % 2

    reference = build_scoring_reference(sample, domains)

    assert len(reference) == len(sample)
    assert {"audit_case_id", "participant_id", *DOMAIN_FEATURES} <= set(reference.columns)
    assert reference["participant_id"].isin(participants["participant_id"]).all()
