from pathlib import Path
import subprocess
import sys

import pandas as pd

from reanalysis_v2.c4_runner import run_c4_controls
from reanalysis_v2.prepare import CONDITION_OUTPUT_NAMES, prepare_primary_inputs
from scripts.run_c4_controls import DEFAULT_TURNS


def test_c4_script_is_directly_executable():
    repository_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/run_c4_controls.py", "--help"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Run C4 interviewer mechanism controls" in result.stdout
    assert DEFAULT_TURNS.as_posix().endswith("analysis_v2/01_inputs/c4_turns_official142_frozen.csv")


def test_c4_runner_writes_exact_controls_shuffle_mapping_and_family(tmp_path: Path):
    rows = []
    turns = []
    for participant_id in range(100, 120):
        score = 12 if participant_id % 2 else 3
        label = int(score >= 10)
        for condition in CONDITION_OUTPUT_NAMES:
            text = f"stable text participant {participant_id}"
            rows.append(
                {
                    "participant_id": participant_id,
                    "official_avec_split": "train" if participant_id < 115 else "dev",
                    "paper_label_phq8_ge10": label,
                    "paper_phq8_score": score,
                    "condition_id": condition,
                    "canonical_source_condition": condition,
                    "transcript_format": "plain",
                    "text": text,
                    "word_count": len(text.split()),
                    "computed_word_count": len(text.split()),
                    "is_empty": False,
                    "text_sha256": "placeholder",
                }
            )
        utterances = [
            (1, "hello there friend", 0),
            (2, "do you feel high" if label else "do you feel low", 1),
            (3, "tell me more about that", 0),
            (4, "thank you for sharing today", 0),
        ]
        for turn_index, text, explicit in utterances:
            turns.append(
                {
                    "participant_id": participant_id,
                    "turn_index": turn_index,
                    "speaker_norm": "interviewer",
                    "normalized_spoken_text": text,
                    "is_explicit_clinical_prompt": explicit,
                }
            )
    source = tmp_path / "source.csv"
    turns_path = tmp_path / "turns.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    pd.DataFrame(turns).to_csv(turns_path, index=False)
    output_root = tmp_path / "analysis_v2"
    prepare_primary_inputs(
        source,
        output_root,
        expected_n=20,
        n_repeats=1,
        n_splits=5,
        base_seed=20260624,
    )
    manifest = run_c4_controls(
        output_root,
        turns_path=turns_path,
        length_master_seed=10,
        expected_n=20,
        expected_repeats=1,
        n_permutations=9,
        base_seed=20260624,
    )
    root = output_root / "05_c4_controls"
    summary = pd.read_csv(root / "metrics" / "c4_control_metrics.csv")
    paired = pd.read_csv(root / "paired_tests" / "c4_control_family_fdr.csv")
    mapping = pd.read_csv(root / "template_shuffled" / "shuffle_mapping.csv")
    assert len(summary) == 5
    assert len(paired) == 8
    assert len(mapping) == 5 * 20
    assert (mapping.recipient_role == mapping.donor_role).all()
    assert manifest["status"] == "complete"
