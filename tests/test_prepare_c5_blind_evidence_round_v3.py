import hashlib
from pathlib import Path

import pandas as pd
import pytest

from reanalysis_v2.c5_blind_workflow_v3 import freeze_consensus
from scripts.prepare_c5_blind_evidence_round_v3 import prepare_stage2


def test_stage2_requires_frozen_consensus_and_expands_actual_spans(tmp_path):
    consensus = pd.DataFrame(
        {
            "blind_id": ["V001"],
            "domain": ["depressed_mood"],
            "consensus_evidence_quote": ["I feel sad"],
            "consensus_line_number": ["2"],
        }
    )
    key = pd.DataFrame(
        {
            "blind_id": ["V001"],
            "participant_id": [101],
            "sample_type": ["main"],
        }
    )
    spans = pd.DataFrame(
        {
            "participant_id": [101, 101, 999],
            "source_order": [1, 2, 1],
            "domain": ["depressed_mood", "depressed_mood", "depressed_mood"],
            "polarity": ["present", "present", "present"],
            "exact_quote": ["I feel sad", "Nothing is fun", "not selected"],
        }
    )
    lines = pd.DataFrame(
        {
            "blind_id": ["V001", "V001"],
            "line_number": [1, 2],
            "participant_text_line": ["Nothing is fun", "I feel sad"],
        }
    )
    consensus_path = tmp_path / "consensus.csv"
    key_path = tmp_path / "key.csv"
    spans_path = tmp_path / "spans.csv"
    lines_path = tmp_path / "lines.csv"
    freeze_path = tmp_path / "freeze.json"
    output = tmp_path / "stage2"
    consensus.to_csv(consensus_path, index=False)
    key.to_csv(key_path, index=False)
    spans.to_csv(spans_path, index=False)
    lines.to_csv(lines_path, index=False)

    with pytest.raises(FileNotFoundError):
        prepare_stage2(
            consensus_path=consensus_path,
            freeze_record_path=freeze_path,
            blind_key_path=key_path,
            reviewed_spans_path=spans_path,
            line_index_path=lines_path,
            output_dir=output,
        )

    freeze_consensus(consensus_path, freeze_path)
    manifest = prepare_stage2(
        consensus_path=consensus_path,
        freeze_record_path=freeze_path,
        blind_key_path=key_path,
        reviewed_spans_path=spans_path,
        line_index_path=lines_path,
        output_dir=output,
    )
    assert manifest["actual_c5_span_row_n"] == 2
    assert manifest["human_reference_evidence_row_n"] == 1
    evidence_a = pd.read_csv(output / "reviewer_A_stage2_evidence_blank.csv")
    evidence_b = pd.read_csv(output / "reviewer_B_stage2_evidence_blank.csv")
    omissions = pd.read_csv(output / "reviewer_A_stage2_omission_blank.csv")
    assert len(evidence_a) == len(evidence_b) == 2
    assert len(omissions) == 1
    assert not (evidence_a["dp_quote"] == "not selected").any()
    assert manifest["count_values_used_to_infer_span_rows"] is False

    rerun_manifest = prepare_stage2(
        consensus_path=consensus_path,
        freeze_record_path=freeze_path,
        blind_key_path=key_path,
        reviewed_spans_path=spans_path,
        line_index_path=lines_path,
        output_dir=output,
        force=True,
    )
    assert "stage2_manifest.json" not in rerun_manifest["output_hashes"]
    for name, expected_hash in rerun_manifest["output_hashes"].items():
        actual_hash = hashlib.sha256((output / name).read_bytes()).hexdigest()
        assert actual_hash == expected_hash
