from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from reanalysis_v2.source_importance_strict import (
    MODEL_SPECS,
    auc_pairwise,
    bh_fdr,
    infer_incremental_comparisons,
    load_strict_source_frame,
    paired_auc_inference,
    run_domain_leave_one_out,
    run_group_permutation_importance,
    run_repeated_cv,
    significance_marker,
    summarize_model_metrics,
)


def _synthetic_frame(n: int = 20) -> pd.DataFrame:
    label = np.array(([0, 1] * (n // 2)), dtype=int)
    return pd.DataFrame(
        {
            "participant_id": np.arange(100, 100 + n),
            "label": label,
            "c2_text": [f"patient token_{y} shared" for y in label],
            "c3_text": [f"interviewer path_{y}" for y in label],
            "c5_text": [f"evidence symptom_{y}" for y in label],
            "domain_count__mood": label + 1,
            "domain_count__history": np.roll(label, 1) + 1,
            "domain_presence__mood": label,
            "domain_presence__history": np.roll(label, 1),
            "template_presence__T001": label,
            "template_presence__T002": np.roll(label, 1),
        }
    )


def _two_repeat_two_fold_splits(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, int | str]] = []
    for repeat in (1, 2):
        fold_assignment = (np.arange(len(frame)) // 2 + repeat - 1) % 2 + 1
        for fold in (1, 2):
            for participant_id, assigned_fold in zip(frame["participant_id"], fold_assignment):
                rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "participant_id": int(participant_id),
                        "role": "test" if assigned_fold == fold else "train",
                    }
                )
    return pd.DataFrame(rows)


def test_model_specs_use_raw_source_groups_not_aggregated_oof_probabilities() -> None:
    assert MODEL_SPECS["M0"] == ("c2_text",)
    assert MODEL_SPECS["M3"] == (
        "c2_text",
        "domain_count",
        "template_presence",
    )
    assert MODEL_SPECS["M5"][-1] == "c3_text"
    assert all("prob" not in group.lower() for groups in MODEL_SPECS.values() for group in groups)


def test_bh_fdr_is_monotone_and_bounded() -> None:
    p = np.array([0.04, 0.001, 0.02, 0.8])
    q = bh_fdr(p)
    ordered = q[np.argsort(p)]
    assert np.all(np.diff(ordered) >= -1e-12)
    assert np.all((0 <= q) & (q <= 1))


def test_significance_marker_uses_q_value_only() -> None:
    assert significance_marker(0.0009) == "***"
    assert significance_marker(0.009) == "**"
    assert significance_marker(0.049) == "*"
    assert significance_marker(0.05) == "ns"


def test_repeated_cv_outputs_one_prediction_per_repeat_per_participant() -> None:
    frame = _synthetic_frame()
    splits = _two_repeat_two_fold_splits(frame)

    fold_rows, participant_rows = run_repeated_cv(
        frame,
        splits,
        model_specs={"M0": MODEL_SPECS["M0"], "M3": MODEL_SPECS["M3"]},
        min_df=1,
    )

    assert len(fold_rows) == len(frame) * 2
    assert fold_rows.groupby("participant_id").size().eq(2).all()
    assert participant_rows["participant_id"].nunique() == len(frame)
    assert participant_rows[["M0_prob", "M3_prob"]].notna().all().all()


def test_paired_auc_inference_is_deterministic() -> None:
    label = np.array([0, 0, 0, 1, 1, 1])
    baseline = np.array([0.10, 0.50, 0.40, 0.30, 0.60, 0.90])
    improved = np.array([0.10, 0.20, 0.30, 0.70, 0.80, 0.90])

    first = paired_auc_inference(
        label,
        improved,
        baseline,
        n_bootstrap=200,
        n_permutations=400,
        seed=17,
    )
    second = paired_auc_inference(
        label,
        improved,
        baseline,
        n_bootstrap=200,
        n_permutations=400,
        seed=17,
    )

    assert first == second
    assert first["delta_auc"] > 0
    assert 0 <= first["p_value"] <= 1


def test_pairwise_auc_matches_sklearn_and_handles_ties() -> None:
    from sklearn.metrics import roc_auc_score

    label = np.array([0, 0, 1, 1, 0, 1])
    probability = np.array([0.1, 0.5, 0.5, 0.9, 0.2, 0.7])

    assert auc_pairwise(label, probability) == roc_auc_score(label, probability)


def test_model_summary_and_incremental_inference_have_auditable_fields() -> None:
    frame = _synthetic_frame()
    splits = _two_repeat_two_fold_splits(frame)
    _, participant_rows = run_repeated_cv(
        frame,
        splits,
        model_specs={"M0": MODEL_SPECS["M0"], "M3": MODEL_SPECS["M3"]},
        min_df=1,
    )

    metrics, auc_ci = summarize_model_metrics(
        participant_rows,
        n_bootstrap=100,
        seed=19,
    )
    comparisons = infer_incremental_comparisons(
        participant_rows,
        comparisons=(("M3", "M0"),),
        n_bootstrap=100,
        n_permutations=200,
        seed=19,
    )

    assert metrics["model"].tolist() == ["M0", "M3"]
    assert auc_ci["model"].tolist() == ["M0", "M3"]
    assert {"p_value", "q_value", "significance"}.issubset(comparisons.columns)
    assert comparisons.loc[0, "comparison"] == "M3 vs M0"


def test_group_permutation_and_domain_loo_return_participant_level_inference() -> None:
    frame = _synthetic_frame()
    splits = _two_repeat_two_fold_splits(frame)

    importance, importance_predictions = run_group_permutation_importance(
        frame,
        splits,
        groups=MODEL_SPECS["M3"],
        min_df=1,
        shuffles_per_fold=3,
        n_bootstrap=50,
        n_permutations=100,
        seed=23,
    )
    domain = run_domain_leave_one_out(
        frame,
        splits,
        representations=("domain_presence", "domain_count"),
        n_bootstrap=50,
        n_permutations=100,
        seed=23,
    )

    assert set(importance["source"]) == set(MODEL_SPECS["M3"])
    assert importance_predictions["participant_id"].nunique() == len(frame)
    assert {"ci_lower", "ci_upper", "q_value", "significance"}.issubset(importance.columns)
    assert len(domain) == 4
    assert set(domain["representation"]) == {"domain_presence", "domain_count"}
    assert {"removed_domain", "q_value", "significance"}.issubset(domain.columns)


def test_load_strict_source_frame_normalizes_declared_empty_texts() -> None:
    analysis_root = Path(__file__).resolve().parents[1] / "analysis_v2"
    frame = load_strict_source_frame(analysis_root)

    assert len(frame) == 142
    assert int(frame["label"].sum()) == 43
    assert frame[["c2_text", "c3_text", "c5_text"]].isna().sum().sum() == 0
    assert frame.loc[frame["participant_id"] == 451, "c3_text"].item() == ""
