from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reanalysis_v2.post_audit_sensitivity import (
    build_cross_fitted_null_predictions,
    build_retained_alignment_frames,
    build_symmetric_half_min_draw,
    build_symmetric_position_dataset,
    paired_metric_sensitivity,
    run_retained_alignment_sensitivity,
    run_symmetric_half_min_control,
    run_symmetric_position_control,
    summarize_brier_skill,
    summarize_token_budget_audit,
)


def _source_predictions() -> pd.DataFrame:
    labels = [1, 1, 0, 0, 1, 0, 0, 0]
    folds = [1, 1, 1, 1, 2, 2, 2, 2]
    rows = []
    for condition, offset in (("participant_speech", 0.0), ("interviewer_speech", 0.04)):
        for participant_id, (label, fold) in enumerate(zip(labels, folds), start=1):
            raw = 0.25 + 0.35 * label + 0.01 * participant_id + offset
            rows.append(
                {
                    "condition": condition,
                    "repeat": 1,
                    "fold": fold,
                    "participant_id": participant_id,
                    "label": label,
                    "raw_probability": raw,
                    "platt_probability": min(raw + 0.02, 0.99),
                    "isotonic_probability": min(raw + 0.04, 0.99),
                }
            )
    return pd.DataFrame(rows)


def test_cross_fitted_null_uses_only_other_outer_folds() -> None:
    predictions = _source_predictions()

    result = build_cross_fitted_null_predictions(predictions)

    assert len(result) == 8
    assert result["participant_id"].is_unique
    assert result.loc[result["fold"].eq(1), "null_probability"].eq(0.25).all()
    assert result.loc[result["fold"].eq(2), "null_probability"].eq(0.50).all()
    assert result["null_source"].eq("outer_training_fold_prevalence").all()


def test_brier_skill_uses_same_cross_fitted_rows_for_all_probability_variants() -> None:
    predictions = _source_predictions()
    null_predictions = build_cross_fitted_null_predictions(predictions)

    metrics, fold_metrics = summarize_brier_skill(
        predictions,
        null_predictions,
        n_bootstrap=50,
        seed=91,
    )

    assert set(metrics["probability_variant"]) == {"raw", "platt", "isotonic"}
    assert len(metrics) == 6
    assert len(fold_metrics) == 12
    row = metrics.loc[
        metrics["condition"].eq("participant_speech")
        & metrics["probability_variant"].eq("raw")
    ].iloc[0]
    expected = 1.0 - row["model_brier"] / row["trainfold_null_brier"]
    assert row["bss_trainfold"] == pytest.approx(expected)
    assert row["cohort_prevalence"] == pytest.approx(3 / 8)
    assert row["bss_trainfold_ci_low"] <= row["bss_trainfold"] <= row["bss_trainfold_ci_high"]


def _token_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    participant = pd.DataFrame(
        [
            {
                "participant_id": 1,
                "paper_label_phq8_ge10": 0,
                "text": "p0 p1 p2 p3 p4 p5 p6 p7 p8 p9",
            },
            {
                "participant_id": 2,
                "paper_label_phq8_ge10": 1,
                "text": "only",
            },
        ]
    )
    interviewer = pd.DataFrame(
        [
            {
                "participant_id": 1,
                "paper_label_phq8_ge10": 0,
                "text": "i0 i1 i2 i3 i4 i5",
            },
            {
                "participant_id": 2,
                "paper_label_phq8_ge10": 1,
                "text": "short",
            },
        ]
    )
    return participant, interviewer


def test_symmetric_half_min_draw_applies_identical_budget_to_both_sources() -> None:
    participant, interviewer = _token_frames()

    matched, audit = build_symmetric_half_min_draw(
        participant,
        interviewer,
        draw=3,
        seed=20260705,
    )
    repeated, repeated_audit = build_symmetric_half_min_draw(
        participant,
        interviewer,
        draw=3,
        seed=20260705,
    )

    first = matched.set_index("participant_id").loc[1]
    assert len(first["participant_matched_text"].split()) == 3
    assert len(first["interviewer_matched_text"].split()) == 3
    second = matched.set_index("participant_id").loc[2]
    assert second["participant_matched_text"] == ""
    assert second["interviewer_matched_text"] == ""
    assert audit.set_index("participant_id").loc[2, "target_token_count"] == 0
    pd.testing.assert_frame_equal(matched, repeated)
    pd.testing.assert_frame_equal(audit, repeated_audit)


@pytest.mark.parametrize(
    ("position", "expected_participant", "expected_interviewer"),
    [
        ("early", "p0 p1 p2", "i0 i1 i2"),
        ("middle", "p3 p4 p5", "i1 i2 i3"),
        ("late", "p7 p8 p9", "i3 i4 i5"),
    ],
)
def test_symmetric_position_windows_use_same_budget_without_source_selection(
    position: str,
    expected_participant: str,
    expected_interviewer: str,
) -> None:
    participant, interviewer = _token_frames()

    frame, audit = build_symmetric_position_dataset(
        participant,
        interviewer,
        position=position,
    )

    row = frame.set_index("participant_id").loc[1]
    assert row["participant_matched_text"] == expected_participant
    assert row["interviewer_matched_text"] == expected_interviewer
    assert audit["position"].eq(position).all()


def test_token_budget_summary_reports_truncation_retention_and_short_targets() -> None:
    participant, interviewer = _token_frames()
    _, audit = build_symmetric_half_min_draw(participant, interviewer, draw=1)

    summary = summarize_token_budget_audit(audit, control="symmetric_half_min")

    assert set(summary["source"]) == {"participant", "interviewer", "shared_target"}
    participant_row = summary.set_index("source").loc["participant"]
    interviewer_row = summary.set_index("source").loc["interviewer"]
    target_row = summary.set_index("source").loc["shared_target"]
    assert participant_row["truncated_n"] == 2
    assert interviewer_row["truncated_n"] == 2
    assert participant_row["original_token_median"] == pytest.approx(5.5)
    assert target_row["zero_target_n"] == 1
    assert target_row["target_below_10_n"] == 2


def test_retained_alignment_reconstruction_changes_only_quote_text() -> None:
    spans = pd.DataFrame(
        [
            {
                "participant_id": 1,
                "source_order": 1,
                "domain": "sleep",
                "polarity": "present",
                "original_model_quote": "sleep paraphrase",
                "exact_quote": "exact sleep words",
                "review_status": "manual_pass_corrected_quote_added",
            },
            {
                "participant_id": 1,
                "source_order": 2,
                "domain": "mood",
                "polarity": "present",
                "original_model_quote": "same mood words",
                "exact_quote": "same mood words",
                "review_status": "base_validated_quote",
            },
            {
                "participant_id": 2,
                "source_order": 1,
                "domain": "energy",
                "polarity": "absent",
                "original_model_quote": "same energy words",
                "exact_quote": "same energy words",
                "review_status": "base_validated_quote",
            },
        ]
    )
    frozen = pd.DataFrame(
        [
            {
                "participant_id": 1,
                "paper_label_phq8_ge10": 1,
                "text": "exact sleep words\nsame mood words",
            },
            {
                "participant_id": 2,
                "paper_label_phq8_ge10": 0,
                "text": "same energy words",
            },
            {
                "participant_id": 3,
                "paper_label_phq8_ge10": 0,
                "text": "NO_EXPLICIT_SYMPTOM_EVIDENCE",
            },
        ]
    )

    original, aligned, audit = build_retained_alignment_frames(spans, frozen)

    assert original.set_index("participant_id").loc[1, "retained_text"] == (
        "sleep paraphrase\nsame mood words"
    )
    assert aligned.set_index("participant_id").loc[1, "retained_text"] == (
        "exact sleep words\nsame mood words"
    )
    assert aligned.set_index("participant_id").loc[3, "retained_text"] == ""
    assert audit["representation_changed"].sum() == 1
    assert audit["corrected_span_count"].sum() == 1
    assert {"domain_held_fixed", "polarity_held_fixed", "retained_status_held_fixed"}.issubset(
        audit.columns
    )
    assert not any("quote" in column for column in audit.columns)


def test_paired_metric_sensitivity_reports_three_metrics_and_bh_q_values() -> None:
    labels = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    original = np.array([0.65, 0.55, 0.52, 0.48, 0.42, 0.35, 0.30, 0.20])
    aligned = np.array([0.70, 0.60, 0.58, 0.44, 0.38, 0.30, 0.25, 0.15])

    result = paired_metric_sensitivity(
        labels,
        aligned,
        original,
        n_bootstrap=100,
        n_permutations=200,
        seed=13,
    )

    assert set(result["metric"]) == {"roc_auc", "pr_auc", "brier"}
    assert result["q_value"].between(0, 1).all()
    assert result["n_bootstrap_valid"].gt(0).all()
    brier = result.set_index("metric").loc["brier"]
    assert brier["difference_direction"] == "aligned_minus_original; negative_is_better"


def _modeling_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    participant_rows = []
    interviewer_rows = []
    split_rows = []
    for participant_id in range(1, 21):
        label = participant_id % 2
        participant_rows.append(
            {
                "participant_id": participant_id,
                "paper_label_phq8_ge10": label,
                "text": ("participant positive " if label else "participant negative ")
                + " ".join(f"p{index}" for index in range(20)),
            }
        )
        interviewer_rows.append(
            {
                "participant_id": participant_id,
                "paper_label_phq8_ge10": label,
                "text": ("interviewer positive " if label else "interviewer negative ")
                + " ".join(f"i{index}" for index in range(12)),
            }
        )
        assigned_fold = 1 if participant_id <= 10 else 2
        for fold in (1, 2):
            split_rows.append(
                {
                    "repeat": 1,
                    "fold": fold,
                    "participant_id": participant_id,
                    "role": "test" if fold == assigned_fold else "train",
                }
            )
    return (
        pd.DataFrame(participant_rows),
        pd.DataFrame(interviewer_rows),
        pd.DataFrame(split_rows),
    )


def test_symmetric_half_min_runner_pairs_every_participant_within_each_draw() -> None:
    participant, interviewer, splits = _modeling_fixture()

    predictions, draw_metrics, metrics, comparison, audit = run_symmetric_half_min_control(
        participant,
        interviewer,
        splits,
        n_draws=2,
        n_bootstrap=20,
        n_permutations=19,
        seed=71,
    )

    assert len(predictions) == 40
    assert predictions.groupby("draw")["participant_id"].nunique().eq(20).all()
    assert len(draw_metrics) == 4
    assert set(metrics["condition"]) == {"participant_matched", "interviewer_matched"}
    assert comparison.loc[0, "bootstrap_resamples"] == 20
    assert comparison.loc[0, "n_permutations"] == 19
    assert audit["target_token_count"].eq(7).all()


def test_symmetric_position_runner_uses_one_three_position_comparison_family() -> None:
    participant, interviewer, splits = _modeling_fixture()

    predictions, metrics, comparisons, audit = run_symmetric_position_control(
        participant,
        interviewer,
        splits,
        n_bootstrap=20,
        n_permutations=100,
        seed=73,
    )

    assert len(predictions) == 60
    assert set(predictions["position"]) == {"early", "middle", "late"}
    assert len(metrics) == 6
    assert len(comparisons) == 3
    assert comparisons["q_value"].between(0, 1).all()
    assert comparisons["comparison_family"].eq("symmetric_position_source_auc").all()
    assert audit["target_token_count"].eq(7).all()


def test_retained_alignment_runner_reports_metrics_and_probability_changes() -> None:
    participant, _, splits = _modeling_fixture()
    original = participant[["participant_id", "paper_label_phq8_ge10"]].rename(
        columns={"paper_label_phq8_ge10": "label"}
    )
    aligned = original.copy()
    original["retained_text"] = [
        ("low mood raw " if label else "stable mood raw ") + f"case {participant_id}"
        for participant_id, label in zip(original["participant_id"], original["label"])
    ]
    aligned["retained_text"] = [
        ("low mood exact " if label else "stable mood exact ") + f"case {participant_id}"
        for participant_id, label in zip(aligned["participant_id"], aligned["label"])
    ]

    predictions, metrics, differences, change = run_retained_alignment_sensitivity(
        original,
        aligned,
        splits,
        n_bootstrap=20,
        n_permutations=100,
        seed=79,
    )

    assert len(predictions) == 20
    assert set(metrics["model"]) == {"original_retained", "aligned_retained"}
    assert set(differences["metric"]) == {"roc_auc", "pr_auc", "brier"}
    assert change.loc[0, "n"] == 20
    assert change.loc[0, "absolute_change_max"] >= change.loc[0, "absolute_change_median"]
