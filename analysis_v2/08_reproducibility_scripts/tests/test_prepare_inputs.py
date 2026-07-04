import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from reanalysis_v2.io import sha256_file
from reanalysis_v2.prepare import (
    CONDITION_OUTPUT_NAMES,
    freeze_c4_turns_input,
    prepare_primary_inputs,
)


def test_prepare_script_is_directly_executable_from_repository_root():
    repository_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/prepare_analysis_v2.py", "--help"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Freeze DAIC-WOZ analysis v2 inputs" in result.stdout
    assert "--turns-source" in result.stdout


def test_prepare_primary_inputs_freezes_condition_files_splits_and_hashes(tmp_path: Path):
    rows = []
    conditions = list(CONDITION_OUTPUT_NAMES)
    for participant_id in range(100, 120):
        score = 12 if participant_id % 2 else 3
        for condition in conditions:
            text = "" if (participant_id == 100 and condition.endswith("evidence")) else f"{condition} {participant_id}"
            rows.append(
                {
                    "participant_id": participant_id,
                    "official_avec_split": "train" if participant_id < 115 else "dev",
                    "paper_label_phq8_ge10": int(score >= 10),
                    "paper_phq8_score": score,
                    "condition_id": condition,
                    "canonical_source_condition": f"source_{condition}",
                    "transcript_format": "plain",
                    "text": text,
                    "word_count": len(text.split()),
                    "computed_word_count": len(text.split()),
                    "is_empty": text == "",
                    "text_sha256": "placeholder",
                }
            )
    source = tmp_path / "source.csv"
    pd.DataFrame(rows).to_csv(source, index=False)

    out = tmp_path / "analysis_v2"
    manifest = prepare_primary_inputs(
        source,
        out,
        expected_n=20,
        n_repeats=2,
        n_splits=5,
        base_seed=20260624,
    )

    assert manifest["cohort"] == {"n_participants": 20, "n_positive": 10, "n_negative": 10}
    assert manifest["source"]["sha256"] == sha256_file(source)
    for condition, filename in CONDITION_OUTPUT_NAMES.items():
        path = out / "01_inputs" / filename
        assert path.exists()
        frozen = pd.read_csv(path, keep_default_na=False)
        assert frozen.condition_id.eq(condition).all()
        assert len(frozen) == 20
        assert manifest["frozen_inputs"][condition]["sha256"] == sha256_file(path)

    membership = pd.read_csv(out / "00_splits" / "repeated_5fold_splits_10x5.csv")
    assert len(membership) == 2 * 5 * 20
    assert not membership.participant_id.isna().any()

    saved = json.loads((out / "00_plan" / "input_freeze_manifest.json").read_text(encoding="utf-8"))
    assert saved == manifest


def test_freeze_c4_turns_derives_annotations_from_verified_official_turns(tmp_path: Path):
    rows = []
    turns = []
    for participant_id in range(100, 120):
        score = 12 if participant_id % 2 else 3
        texts = {
            "full_transcript": "full",
            "participant_speech": "participant response",
            "interviewer_speech": "hello there do you feel depressed",
            "non_explicit_interviewer_speech": "hello there",
            "participant_symptom_evidence": "response",
        }
        for condition, text in texts.items():
            rows.append(
                {
                    "participant_id": participant_id,
                    "official_avec_split": "train" if participant_id < 115 else "dev",
                    "paper_label_phq8_ge10": int(score >= 10),
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
        turns.extend(
            [
                {
                    "participant_id": participant_id,
                    "turn_index": 1,
                    "speaker_norm": "interviewer",
                    "text": "hello there",
                    "official_avec_split": "train" if participant_id < 115 else "dev",
                    "start_time": 0.0,
                    "stop_time": 1.0,
                },
                {
                    "participant_id": participant_id,
                    "turn_index": 2,
                    "speaker_norm": "interviewer",
                    "text": "D1 (do you feel depressed)",
                    "official_avec_split": "train" if participant_id < 115 else "dev",
                    "start_time": 1.0,
                    "stop_time": 2.0,
                },
                {
                    "participant_id": participant_id,
                    "turn_index": 3,
                    "speaker_norm": "participant",
                    "text": "participant response",
                    "official_avec_split": "train" if participant_id < 115 else "dev",
                    "start_time": 2.0,
                    "stop_time": 3.0,
                },
            ]
        )
    source = tmp_path / "source.csv"
    turns_source = tmp_path / "turns_official.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    pd.DataFrame(turns).to_csv(turns_source, index=False)
    out = tmp_path / "analysis_v2"
    prepare_primary_inputs(
        source,
        out,
        expected_n=20,
        n_repeats=1,
        n_splits=5,
    )

    result = freeze_c4_turns_input(turns_source, out, expected_n=20)

    frozen = pd.read_csv(out / "01_inputs" / "c4_turns_official142_frozen.csv")
    audit = pd.read_csv(out / "00_plan" / "c4_turn_source_alignment_audit.csv")
    manifest = json.loads((out / "00_plan" / "input_freeze_manifest.json").read_text(encoding="utf-8"))
    assert len(frozen) == 60
    assert frozen["is_explicit_clinical_prompt"].sum() == 20
    assert frozen.loc[frozen["is_prompt_annotated"].eq(1), "prompt_id"].eq("D1").all()
    assert audit["c3_exact_match"].all()
    assert audit["c4_exact_match"].all()
    assert result["sha256"] == sha256_file(out / "01_inputs" / "c4_turns_official142_frozen.csv")
    assert manifest["control_inputs"]["c4_turns"]["source_sha256"] == sha256_file(turns_source)
