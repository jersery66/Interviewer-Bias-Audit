from pathlib import Path

import pandas as pd

from scripts.prepare_c5_blind_audit_v3 import prepare


REPO_ROOT = Path(__file__).resolve().parents[1]


def _restricted_sentinel_participant_id() -> int:
    key = pd.read_csv(
        REPO_ROOT
        / "analysis_v2"
        / "04_c5_controls"
        / "blind_quality_audit"
        / "private"
        / "packet_3"
        / "blind_id_key.csv"
    )
    return int(key.loc[key["sample_type"].eq("sentinel"), "participant_id"].item())


def test_prepare_v3_keeps_main_sample_and_separates_enrichment_and_sentinel(tmp_path):
    public = tmp_path / "packet_3"
    private = tmp_path / "private" / "packet_3"
    sentinel_participant_id = _restricted_sentinel_participant_id()
    manifest = prepare(
        sentinel_participant_id=sentinel_participant_id,
        public_packet=public,
        private_packet=private,
    )

    old_main = pd.read_csv(
        REPO_ROOT
        / "analysis_v2"
        / "04_c5_controls"
        / "blind_quality_audit"
        / "private"
        / "packet_2"
        / "blind_id_key.csv"
    )
    new_key = pd.read_csv(private / "blind_id_key.csv")
    assert set(new_key.loc[new_key["sample_type"] == "main", "participant_id"]) == set(
        old_main["participant_id"]
    )
    assert 10 <= (new_key["sample_type"] == "enriched").sum() <= 15
    assert new_key.loc[
        new_key["sample_type"] == "sentinel", "participant_id"
    ].tolist() == [sentinel_participant_id]
    assert manifest["main_sample_n"] == 30
    assert manifest["sentinel_sample_n"] == 1
    assert manifest["stage2_status"] == "locked_until_human_consensus_freeze"

    reviewer = pd.read_csv(public / "reviewer_A_stage1_blank.csv")
    cases = pd.read_csv(public / "blind_cases.csv")
    assert len(reviewer) == manifest["stage1_cells_per_reviewer"]
    assert list(cases.columns) == ["blind_id", "participant_text"]
    forbidden = {
        "participant_id",
        "sample_type",
        "paper_label_phq8_ge10",
        "paper_phq8_score",
        "existing_dp",
        "c5_evidence",
    }
    assert not forbidden.intersection(reviewer.columns)
    assert not forbidden.intersection(cases.columns)

    enrichment = pd.read_csv(private / "enrichment_selection_log.csv")
    for target in (
        "appetite_weight_positive",
        "suicide_self_harm_positive",
        "anhedonia_interest_positive",
        "sleep_fatigue_energy_zero",
        "protective_or_absent_symptom_zero",
    ):
        assert enrichment[target].sum() >= 4

    evidence_schema = pd.read_csv(public / "stage2_evidence_review_schema.csv")
    omission_schema = pd.read_csv(public / "stage2_omission_review_schema.csv")
    assert evidence_schema.empty
    assert omission_schema.empty


def test_prepare_v3_does_not_modify_packet_2(tmp_path):
    packet_2 = (
        REPO_ROOT
        / "analysis_v2"
        / "04_c5_controls"
        / "blind_quality_audit"
        / "packet_2"
    )
    before = {path.name: path.read_bytes() for path in packet_2.iterdir() if path.is_file()}
    prepare(
        sentinel_participant_id=_restricted_sentinel_participant_id(),
        public_packet=tmp_path / "packet_3",
        private_packet=tmp_path / "private" / "packet_3",
    )
    after = {path.name: path.read_bytes() for path in packet_2.iterdir() if path.is_file()}
    assert before == after
