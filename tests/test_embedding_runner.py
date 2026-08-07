from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from reanalysis_v2.embedding_runner import (
    EmbeddingModelConfig,
    build_representation_family_tests,
    run_embedding_model_analysis,
)
from reanalysis_v2.analysis import PRIMARY_CONDITIONS
from reanalysis_v2.prepare import CONDITION_OUTPUT_NAMES, prepare_primary_inputs


def test_embedding_script_is_directly_executable_from_repository_root():
    repository_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/run_embedding_model.py", "--help"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Run one frozen dense embedding model" in result.stdout


def test_finalize_embedding_script_is_directly_executable():
    repository_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/finalize_embedding_analysis.py", "--help"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Finalize the joint representation robustness family" in result.stdout


class TinyEncoder:
    def encode(self, texts, **kwargs):
        rows = []
        for text in texts:
            words = text.split()
            rows.append(
                [
                    float(len(words)),
                    float(sum(word == "highrisk" for word in words)),
                    float(sum(word == "lowrisk" for word in words)),
                ]
            )
        rows = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        return rows / np.where(norms == 0, 1, norms)


def test_embedding_runner_writes_frozen_cache_oof_tuning_and_metrics(tmp_path: Path):
    rows = []
    for participant_id in range(100, 120):
        score = 12 if participant_id % 2 else 3
        signal = "highrisk" if score >= 10 else "lowrisk"
        for condition in CONDITION_OUTPUT_NAMES:
            text = f"stable {signal} {condition} participant{participant_id}"
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
    config = EmbeddingModelConfig(
        name="test/tiny",
        revision="0123456789abcdef",
        slug="model_test_tiny",
        chunk_words=3,
        overlap=1,
        batch_size=8,
    )
    manifest = run_embedding_model_analysis(
        output_root,
        config,
        encoder=TinyEncoder(),
        embedding_dim=3,
        expected_n=20,
        expected_repeats=1,
        inner_splits_count=2,
        c_grid=[0.01, 0.1],
        n_permutations=19,
        base_seed=20260624,
    )
    model_root = output_root / "03_embedding_robustness" / "model_test_tiny"
    summary = pd.read_csv(model_root / "metrics" / "primary_metrics.csv")
    tuning = pd.read_csv(model_root / "tuning" / "c_selection.csv")
    assert len(summary) == 5
    assert len(tuning) == 5 * 5 * 2
    assert len(list((model_root / "embedding_cache").glob("*.npz"))) == 5
    assert manifest["status"] == "complete"
    assert manifest["model"]["revision"] == "0123456789abcdef"


def test_representation_family_combines_within_embedding_and_cross_representation_tests():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    def tables(offset: float):
        result = {}
        for condition_index, condition in enumerate(PRIMARY_CONDITIONS):
            probability = np.clip(
                np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
                + offset
                + condition_index * 0.001,
                0,
                1,
            )
            result[condition] = pd.DataFrame(
                {
                    "participant_id": np.arange(8),
                    "label": y,
                    "raw_probability": probability,
                    "prediction_at_0_5": (probability >= 0.5).astype(int),
                    "nested_prediction_max_macro_f1": (probability >= 0.45).astype(int),
                }
            )
        return result

    result = build_representation_family_tests(
        tables(0.0),
        {"model_a": tables(0.01), "model_b": tables(-0.01)},
        n_permutations=9,
        base_seed=42,
    )
    assert len(result) == 120
    assert result.family.eq("representation_robustness").all()
    assert set(result.comparison_type) == {
        "within_embedding_input_source",
        "same_condition_cross_representation",
    }
    assert result.fdr_bh_q.between(0, 1).all()
