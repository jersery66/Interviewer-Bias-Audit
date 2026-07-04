from pathlib import Path

import pandas as pd

from reanalysis_v2.prepare import CONDITION_OUTPUT_NAMES, prepare_primary_inputs
from reanalysis_v2.runner import run_tfidf_analysis


def test_tfidf_runner_writes_complete_tiny_end_to_end_artifacts(tmp_path: Path):
    rows = []
    for participant_id in range(100, 120):
        score = 12 if participant_id % 2 else 3
        signal = "highrisk" if score >= 10 else "lowrisk"
        for condition in CONDITION_OUTPUT_NAMES:
            text = f"stable common {signal} {condition} participant{participant_id}"
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
    source = tmp_path / "source.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    output_root = tmp_path / "analysis_v2"
    prepare_primary_inputs(
        source,
        output_root,
        expected_n=20,
        n_repeats=1,
        n_splits=5,
        base_seed=20260624,
    )

    manifest = run_tfidf_analysis(
        output_root,
        expected_n=20,
        expected_repeats=1,
        inner_splits_count=2,
        n_permutations=19,
        base_seed=20260624,
    )

    result_root = output_root / "02_tfidf_main"
    summary = pd.read_csv(result_root / "metrics" / "primary_metrics.csv")
    paired = pd.read_csv(
        result_root / "paired_tests" / "primary_input_source_paired_tests_fdr.csv"
    )
    thresholds = pd.read_csv(
        result_root / "nested_thresholds" / "selected_thresholds_50folds.csv"
    )
    assert len(summary) == 5
    assert len(paired) == 40
    assert len(thresholds) == 5 * 5 * 3
    assert manifest["status"] == "complete"
    assert manifest["output_inventory_sha256"]
