from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from reanalysis_v2.source_definition_audit import audit_domain_sources, write_domain_source_audit


def _write_fixture_tables(tmp_path):
    participant = pd.DataFrame(
        {
            "participant_id": [1, 2],
            "text": ["I feel sad and tired", "I sleep well"],
        }
    )
    interviewer = pd.DataFrame(
        {
            "participant_id": [1, 2],
            "text": ["How have you been?", "Do you feel sad?"],
        }
    )
    full = pd.DataFrame(
        {
            "participant_id": [1, 2],
            "text": [
                "Interviewer: How have you been? Participant: I feel sad and tired",
                "Interviewer: Do you feel sad? Participant: I sleep well",
            ],
        }
    )
    spans = pd.DataFrame(
        {
            "participant_id": [1, 1],
            "domain": ["depressed_mood", "sleep_fatigue_energy"],
            "exact_quote": ["I feel sad", "I feel sad and tired"],
            "match_status": ["matched_normalized_whitespace", "matched_normalized_whitespace"],
        }
    )
    domains = pd.DataFrame(
        {
            "participant_id": [1, 2],
            "depressed_mood": [1, 0],
            "sleep_fatigue_energy": [1, 0],
        }
    )
    paths = {}
    for name, table in {
        "participant": participant,
        "interviewer": interviewer,
        "full": full,
        "spans": spans,
        "domains": domains,
    }.items():
        path = tmp_path / f"{name}.csv"
        table.to_csv(path, index=False)
        paths[name] = path
    return paths


def test_audit_confirms_participant_derived_domains_and_blocks_missing_d_all(tmp_path):
    paths = _write_fixture_tables(tmp_path)

    result = audit_domain_sources(
        participant_text_path=paths["participant"],
        interviewer_text_path=paths["interviewer"],
        full_text_path=paths["full"],
        reviewed_spans_path=paths["spans"],
        domain_count_path=paths["domains"],
        d_all_path=tmp_path / "not_available.csv",
    )

    assert result["d_p"]["status"] == "confirmed"
    assert result["d_p"]["matching_quote_n"] == 2
    assert result["d_p"]["interviewer_only_quote_n"] == 0
    assert result["d_p"]["count_reconciliation"] == "matched"
    assert result["d_all"]["status"] == "missing"
    assert result["d_all"]["ready_for_modeling"] is False


def test_audit_rejects_d_all_without_full_transcript_scope(tmp_path):
    paths = _write_fixture_tables(tmp_path)
    d_all = pd.DataFrame(
        {
            "participant_id": [1, 2],
            "depressed_mood": [1, 0],
            "sleep_fatigue_energy": [1, 0],
        }
    )
    d_all_path = tmp_path / "d_all.csv"
    d_all.to_csv(d_all_path, index=False)
    provenance_path = tmp_path / "d_all_manifest.json"
    provenance_path.write_text(
        '{"source_scope": "symptom_evidence_only"}', encoding="utf-8"
    )

    result = audit_domain_sources(
        participant_text_path=paths["participant"],
        interviewer_text_path=paths["interviewer"],
        full_text_path=paths["full"],
        reviewed_spans_path=paths["spans"],
        domain_count_path=paths["domains"],
        d_all_path=d_all_path,
        d_all_provenance_path=provenance_path,
    )

    assert result["d_all"]["status"] == "rejected"
    assert result["d_all"]["ready_for_modeling"] is False
    assert "full_transcript" in result["d_all"]["reason"]


def test_audit_accepts_d_all_only_with_full_transcript_provenance(tmp_path):
    paths = _write_fixture_tables(tmp_path)
    d_all = pd.DataFrame(
        {
            "participant_id": [1, 2],
            "depressed_mood": [1, 0],
            "sleep_fatigue_energy": [1, 0],
        }
    )
    d_all_path = tmp_path / "d_all.csv"
    d_all.to_csv(d_all_path, index=False)
    provenance_path = tmp_path / "d_all_manifest.json"
    provenance_path.write_text(
        '{"source_scope": "full_transcript"}', encoding="utf-8"
    )

    result = audit_domain_sources(
        participant_text_path=paths["participant"],
        interviewer_text_path=paths["interviewer"],
        full_text_path=paths["full"],
        reviewed_spans_path=paths["spans"],
        domain_count_path=paths["domains"],
        d_all_path=d_all_path,
        d_all_provenance_path=provenance_path,
    )

    assert result["d_all"]["status"] == "available"
    assert result["d_all"]["ready_for_modeling"] is True


def test_public_audit_writer_serializes_aggregate_fields_only(tmp_path):
    output = tmp_path / "audit.json"
    write_domain_source_audit({"d_p": {"status": "confirmed"}, "d_all": {"status": "missing"}}, output)

    text = output.read_text(encoding="utf-8")
    assert "confirmed" in text
    assert "exact_quote" not in text


def test_committed_domain_audit_keeps_d_all_blocked():
    path = (
        Path(__file__).resolve().parents[1]
        / "analysis_v2"
        / "10_source_importance"
        / "07_strict_joint_models"
        / "domain_source_audit.json"
    )
    record = json.loads(path.read_text(encoding="utf-8"))

    assert record["d_p"]["status"] == "confirmed"
    assert record["d_p"]["matching_quote_n"] == 1137
    assert record["d_all"]["status"] == "missing"
