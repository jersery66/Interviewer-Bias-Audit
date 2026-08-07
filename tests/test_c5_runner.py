from pathlib import Path
import subprocess
import sys

import pandas as pd

from reanalysis_v2.c5_runner import run_c5_controls
from reanalysis_v2.prepare import CONDITION_OUTPUT_NAMES, prepare_primary_inputs


def test_c5_script_is_directly_executable():
    repository_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/run_c5_controls.py", "--help"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Run C5 evidence concentration controls" in result.stdout


def test_c5_runner_writes_seed_distribution_controls_and_family(tmp_path: Path):
    rows = []
    spans = []
    for participant_id in range(100, 120):
        score = 12 if participant_id % 2 else 3
        signal = "depressed" if score >= 10 else "calm"
        speech = f"alpha {signal} beta gamma delta epsilon zeta eta"
        evidence = f"{signal} beta"
        for condition in CONDITION_OUTPUT_NAMES:
            text = speech if condition == "participant_speech" else evidence
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
        spans.append(
            {
                "participant_id": participant_id,
                "source_order": 1,
                "domain": "depressed_mood",
                "exact_quote": evidence,
            }
        )
    source = tmp_path / "source.csv"
    spans_path = tmp_path / "spans.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    pd.DataFrame(spans).to_csv(spans_path, index=False)
    output_root = tmp_path / "analysis_v2"
    prepare_primary_inputs(
        source,
        output_root,
        expected_n=20,
        n_repeats=1,
        n_splits=5,
        base_seed=20260624,
    )
    manifest = run_c5_controls(
        output_root,
        spans_path=spans_path,
        random_master_seeds=[11, 12],
        nonsymptom_master_seed=13,
        keywords=["depressed"],
        expected_n=20,
        expected_repeats=1,
        n_permutations=9,
        base_seed=20260624,
    )
    root = output_root / "04_c5_controls"
    summary = pd.read_csv(root / "metrics" / "c5_control_metrics.csv")
    paired = pd.read_csv(root / "paired_tests" / "c5_control_family_fdr.csv")
    distribution = pd.read_csv(
        root / "random_same_word" / "random_seed_metric_distribution.csv"
    )
    assert len(summary) == 1 + 2 + 5
    assert len(paired) == (2 + 5) * 2
    assert set(distribution.metric) == {"roc_auc", "pr_auc", "macro_f1_at_0_5"}
    assert manifest["status"] == "complete"
