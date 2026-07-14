from pathlib import Path

from scripts.verify_submission_artifacts import verify_blind_audit_packet


PACKET_DIR = (
    Path(__file__).resolve().parents[1]
    / "analysis_v2"
    / "04_c5_controls"
    / "blind_quality_audit"
    / "packet_2"
)


def test_prepared_blind_audit_packet_passes_public_integrity_gate() -> None:
    assert verify_blind_audit_packet(PACKET_DIR) == []

