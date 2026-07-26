from pathlib import Path

import scripts.verify_submission_artifacts as verifier
from scripts.verify_submission_artifacts import verify_blind_audit_packet


PACKET_DIR = (
    Path(__file__).resolve().parents[1]
    / "analysis_v2"
    / "04_c5_controls"
    / "blind_quality_audit"
    / "packet_2"
)
PACKET3_DIR = PACKET_DIR.parent / "packet_3"


def test_prepared_blind_audit_packet_passes_public_integrity_gate() -> None:
    assert verify_blind_audit_packet(PACKET_DIR) == []


def test_prepared_packet3_passes_public_integrity_gate() -> None:
    assert verify_blind_audit_packet(PACKET3_DIR) == []


def test_full_integrity_verifier_checks_packet3(monkeypatch) -> None:
    checked = []

    def record_packet(path):
        checked.append(path)
        return []

    monkeypatch.setattr(verifier, "verify_blind_audit_packet", record_packet)
    verifier.verify()
    assert PACKET3_DIR in checked

