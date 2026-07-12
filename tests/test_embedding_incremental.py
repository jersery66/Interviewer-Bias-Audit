import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reanalysis_v2.embedding_incremental import (
    DOMAIN_FEATURES,
    FrozenEmbeddingPair,
    FrozenIncrementalInputs,
    LOCKED_INPUT_HASHES,
    PROTOCOL_FEATURES,
    _run_repeat,
    apply_incremental_fdr,
    build_model_specifications,
    make_early_concat_classifier,
    make_late_fusion_classifier,
    make_source_classifier,
    paired_metric_deltas,
    _subset_smoke_inputs,
    _delta_table,
    validate_locked_file,
    validate_locked_inputs,
    validate_run_configuration,
    validate_incremental_inputs,
    verify_embedding_incremental_reruns,
)
from reanalysis_v2.splits import make_repeated_split_membership


def _small_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ids = np.arange(10)
    labels = np.array([0] * 5 + [1] * 5)
    participant = pd.DataFrame({"participant_id": ids, "label": labels})
    protocol = pd.DataFrame({"participant_id": ids})
    for index, name in enumerate(PROTOCOL_FEATURES):
        protocol[name] = ids + index
    domains = pd.DataFrame({"participant_id": ids})
    for index, name in enumerate(DOMAIN_FEATURES):
        domains[name] = (ids + index) % 3
    return participant, protocol, domains


def test_model_specifications_define_the_four_paired_levels():
    specs = build_model_specifications()

    assert list(specs) == ["L1", "L2", "L3", "L4"]
    assert specs["L1"] == ("P", "P+I")
    assert specs["L2"] == ("P+R", "P+R+I")
    assert specs["L3"] == ("P+D", "P+D+I")
    assert specs["L4"] == ("P+R+D", "P+R+D+I")


def test_validate_incremental_inputs_requires_same_ids_and_frozen_columns():
    participant, protocol, domains = _small_inputs()

    merged = validate_incremental_inputs(
        participant, protocol, domains, expected_n=10
    )

    assert list(merged.participant_id) == list(range(10))
    assert set(PROTOCOL_FEATURES).issubset(merged.columns)
    assert set(DOMAIN_FEATURES).issubset(merged.columns)

    with pytest.raises(ValueError, match="participant IDs"):
        validate_incremental_inputs(
            participant,
            protocol.assign(participant_id=lambda frame: frame.participant_id + 100),
            domains,
            expected_n=10,
        )


def test_paired_metric_deltas_use_positive_improvement_direction():
    y = np.array([0, 0, 0, 1, 1, 1])
    base = np.array([0.2, 0.4, 0.6, 0.4, 0.6, 0.8])
    augmented = np.array([0.1, 0.3, 0.5, 0.5, 0.7, 0.9])

    result = paired_metric_deltas(y, base, augmented)

    assert result["delta_auc"] > 0
    assert result["delta_pr_auc"] > 0
    assert result["delta_brier"] > 0
    assert result["delta_log_loss"] > 0


def test_incremental_fdr_keeps_primary_and_secondary_families_separate():
    table = pd.DataFrame(
        {
            "fdr_family": ["primary_l4", "primary_l4", "secondary_all_delta_auc"],
            "raw_p": [0.01, 0.04, 0.02],
        }
    )

    result = apply_incremental_fdr(table)

    assert result.loc[0, "bh_q"] == pytest.approx(0.02)
    assert result.loc[1, "bh_q"] == pytest.approx(0.04)
    assert result.loc[2, "bh_q"] == pytest.approx(0.02)


def test_synthetic_repeat_produces_paired_rows_for_both_fusion_methods():
    rng = np.random.default_rng(7)
    participant_ids = np.arange(100, 120)
    labels = np.array([0] * 10 + [1] * 10)
    inputs = FrozenIncrementalInputs(
        participant_ids=participant_ids,
        labels=labels,
        protocol=rng.normal(size=(20, len(PROTOCOL_FEATURES))),
        domains=rng.integers(0, 4, size=(20, len(DOMAIN_FEATURES))).astype(float),
    )
    pair = FrozenEmbeddingPair(
        slug="test_tiny",
        model_name="test/tiny",
        revision="locked",
        embedding_dim=4,
        participant=rng.normal(size=(20, 4)),
        interviewer=rng.normal(size=(20, 4)),
        cache_files=(),
    )
    membership = make_repeated_split_membership(
        participant_ids, labels, n_repeats=1, n_splits=5, base_seed=99
    )

    for fusion_method in ("late_fusion", "early_concat"):
        predictions, _ = _run_repeat(
            pair=pair,
            inputs=inputs,
            membership=membership,
            repeat=1,
            fusion_method=fusion_method,
            inner_folds=2,
            c_grid=(0.1, 1.0),
            base_seed=101,
        )
        assert len(predictions) == 20 * 4 * 2
        assert set(predictions["model_level"]) == {"L1", "L2", "L3", "L4"}
        assert set(predictions["model_name"]) == {"base", "augmented"}


def test_source_classifier_does_not_standardize_frozen_embeddings():
    classifier = make_source_classifier(c=1.0, seed=7)

    assert "scale" not in classifier.named_steps
    assert classifier.named_steps["classifier"].class_weight == "balanced"
    assert classifier.named_steps["classifier"].solver == "liblinear"


def test_late_fusion_classifier_standardizes_only_inside_second_layer_pipeline():
    classifier = make_late_fusion_classifier(c=1.0, seed=7)

    assert list(classifier.named_steps) == ["scale", "classifier"]
    assert classifier.named_steps["scale"].__class__.__name__ == "StandardScaler"


def test_early_concat_classifier_passthrough_embeddings_and_scales_numeric_columns():
    classifier = make_early_concat_classifier(
        embedding_columns=[0, 1, 2, 3], numeric_columns=[4, 5], c=1.0, seed=7
    )
    transformer = classifier.named_steps["features"]
    transformer_names = {name for name, *_ in transformer.transformers}

    assert transformer_names == {"embedding_passthrough", "numeric_standardizer"}
    numeric = next(
        transformer for name, transformer, _ in transformer.transformers
        if name == "numeric_standardizer"
    )
    assert numeric.__class__.__name__ == "StandardScaler"


def test_locked_file_hash_rejects_a_single_byte_change(tmp_path: Path):
    original = tmp_path / "locked.csv"
    original.write_bytes(b"locked\n")
    expected = {
        "raw": hashlib.sha256(original.read_bytes()).hexdigest(),
        "lf": hashlib.sha256(original.read_bytes()).hexdigest(),
    }
    validate_locked_file(original, expected, label="fixture")
    original.write_bytes(b"changed\n")

    with pytest.raises(ValueError, match="fixture"):
        validate_locked_file(original, expected, label="fixture")


def test_current_frozen_input_registry_matches_the_locked_worktree():
    audit = validate_locked_inputs("analysis_v2")

    assert audit["hashes"]["structural"]["raw"] == LOCKED_INPUT_HASHES["structural"]["raw"]
    assert audit["hashes"]["plan"]["lf"] == LOCKED_INPUT_HASHES["plan"]["lf"]


def test_formal_configuration_cannot_be_reduced_to_smoke_settings():
    with pytest.raises(ValueError, match="formal"):
        validate_run_configuration(
            mode="formal",
            expected_n=142,
            expected_repeats=10,
            main_repeat=1,
            inner_folds=3,
            stability_repeats=1,
            n_bootstrap=100,
            n_permutations=100,
            c_grid=(0.01, 0.1, 1.0, 10.0),
        )
    smoke = validate_run_configuration(
        mode="smoke",
        expected_n=20,
        expected_repeats=1,
        main_repeat=1,
        inner_folds=2,
        stability_repeats=1,
        n_bootstrap=100,
        n_permutations=100,
        c_grid=(0.1, 1.0),
    )
    assert smoke["status"] == "smoke_test"


def test_smoke_subset_is_deterministic_and_class_balanced():
    full = FrozenIncrementalInputs(
        participant_ids=np.arange(20),
        labels=np.array([0] * 14 + [1] * 6),
        protocol=np.arange(100).reshape(20, 5),
        domains=np.arange(200).reshape(20, 10),
    )

    first = _subset_smoke_inputs(full, expected_n=10)
    second = _subset_smoke_inputs(full, expected_n=10)

    assert len(first.labels) == 10
    assert int(first.labels.sum()) == 3
    assert np.array_equal(first.participant_ids, second.participant_ids)


def test_rerun_verification_reports_hash_match_and_mismatch(tmp_path: Path):
    base = tmp_path / "embedding_incremental"
    for run_name in ("run_1", "run_2"):
        run = base / run_name
        run.mkdir(parents=True)
        output = run / "result.csv"
        output.write_text("a,b\n1,2\n", encoding="utf-8")
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        (run / "run_manifest.json").write_text(
            json.dumps({"output_file_hashes": {"result.csv": digest}}),
            encoding="utf-8",
        )

    verified = verify_embedding_incremental_reruns(base)
    assert verified["status"] == "verified"
    (base / "run_2" / "result.csv").write_text("a,b\n9,9\n", encoding="utf-8")
    failed = verify_embedding_incremental_reruns(base)
    assert failed["status"] == "failed"


def test_delta_table_has_only_locked_late_fdr_families():
    rows = []
    rng = np.random.default_rng(12)
    ids = np.arange(20)
    labels = np.array([0] * 10 + [1] * 10)
    for embedding in ("mpnet", "bge"):
        for fusion in ("late_fusion", "early_concat"):
            for level in ("L1", "L2", "L3", "L4"):
                base = rng.uniform(0.05, 0.95, size=20)
                augmented = np.clip(base + rng.normal(0, 0.03, size=20), 0.01, 0.99)
                for participant_id, label, base_probability, augmented_probability in zip(
                    ids, labels, base, augmented
                ):
                    rows.append(
                        {
                            "embedding_model": embedding,
                            "fusion_method": fusion,
                            "model_level": level,
                            "participant_id": participant_id,
                            "label": label,
                            "base_probability": base_probability,
                            "augmented_probability": augmented_probability,
                        }
                    )
    result = _delta_table(
        pd.DataFrame(rows), n_bootstrap=100, n_permutations=9, base_seed=12
    )
    assert result["bh_q_primary_l4"].notna().sum() == 2
    assert result["bh_q_all_late"].notna().sum() == 8
    assert result.loc[result["fusion"].eq("early_concat"), "bh_q_all_late"].isna().all()
