from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from .analysis import _participant_ids_sha256
from .controls import (
    DOMAIN_ORDER,
    build_domain_features,
    exclude_exact_quote_tokens,
    fold_safe_shuffle,
    keyword_mask,
    length_matched_sample,
    position_matched_turns,
)
from .inference import apply_bh_by_family, paired_permutation_test
from .modeling import fit_predict_dense, fit_predict_tfidf
from .splits import split_indices


def _raw_summary(
    participant_predictions: pd.DataFrame, condition: str
) -> dict[str, float | int | str]:
    y = participant_predictions["label"].to_numpy(dtype=int)
    p = participant_predictions["raw_probability"].to_numpy(dtype=float)
    pred = participant_predictions["prediction_at_0_5"].to_numpy(dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "condition": condition,
        "n": int(len(y)),
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "macro_f1_at_0_5": float(
            f1_score(y, pred, average="macro", zero_division=0)
        ),
        "sensitivity_at_0_5": float(tp / (tp + fn)),
        "specificity_at_0_5": float(tn / (tn + fp)),
    }


def _aggregate_raw(
    fold_predictions: pd.DataFrame, expected_repeats: int
) -> pd.DataFrame:
    counts = fold_predictions.groupby("participant_id")["repeat"].nunique()
    if not counts.eq(expected_repeats).all():
        raise ValueError("raw control OOF repeats are incomplete")
    if fold_predictions.duplicated(["participant_id", "repeat"]).any():
        raise ValueError("duplicate participant-repeat raw control predictions")
    participant = (
        fold_predictions.groupby("participant_id", as_index=False)
        .agg(label=("label", "first"), raw_probability=("raw_probability", "mean"))
        .sort_values("participant_id", kind="stable")
        .reset_index(drop=True)
    )
    participant["prediction_at_0_5"] = participant["raw_probability"].ge(0.5).astype(int)
    return participant


def run_raw_text_condition(
    participant_ids: np.ndarray,
    labels: np.ndarray,
    texts: np.ndarray,
    membership: pd.DataFrame,
    *,
    condition: str,
    expected_repeats: int = 10,
    base_seed: int = 20260624,
) -> dict[str, object]:
    participant_ids = np.asarray(participant_ids)
    labels = np.asarray(labels, dtype=int)
    texts = np.asarray(texts, dtype=object)
    if not (len(participant_ids) == len(labels) == len(texts)):
        raise ValueError("participant IDs, labels, and texts must align")
    rows = []
    repeats = sorted(pd.to_numeric(membership["repeat"]).astype(int).unique())
    if repeats != list(range(1, expected_repeats + 1)):
        raise ValueError("unexpected control split repeats")
    for repeat in repeats:
        folds = sorted(
            pd.to_numeric(
                membership.loc[membership["repeat"].astype(int).eq(repeat), "fold"]
            ).astype(int).unique()
        )
        for fold in folds:
            train_idx, test_idx = split_indices(membership, repeat, fold, participant_ids)
            probability, _ = fit_predict_tfidf(
                texts[train_idx],
                labels[train_idx],
                texts[test_idx],
                seed=base_seed + repeat,
            )
            train_hash = _participant_ids_sha256(participant_ids[train_idx])
            test_hash = _participant_ids_sha256(participant_ids[test_idx])
            for local_index, global_index in enumerate(test_idx):
                rows.append(
                    {
                        "condition": condition,
                        "repeat": repeat,
                        "fold": fold,
                        "participant_id": participant_ids[global_index],
                        "label": int(labels[global_index]),
                        "raw_probability": float(probability[local_index]),
                        "model_seed": base_seed + repeat,
                        "train_ids_sha256": train_hash,
                        "test_ids_sha256": test_hash,
                    }
                )
    fold_predictions = pd.DataFrame(rows).sort_values(
        ["repeat", "fold", "participant_id"], kind="stable"
    ).reset_index(drop=True)
    participant = _aggregate_raw(fold_predictions, expected_repeats)
    return {
        "fold_predictions": fold_predictions,
        "participant_predictions": participant,
        "summary": _raw_summary(participant, condition),
    }


def run_raw_feature_condition(
    participant_ids: np.ndarray,
    labels: np.ndarray,
    features: np.ndarray,
    membership: pd.DataFrame,
    *,
    condition: str,
    expected_repeats: int = 10,
    base_seed: int = 20260624,
) -> dict[str, object]:
    participant_ids = np.asarray(participant_ids)
    labels = np.asarray(labels, dtype=int)
    features = np.asarray(features, dtype=float)
    if not (len(participant_ids) == len(labels) == len(features)):
        raise ValueError("participant IDs, labels, and features must align")
    rows = []
    repeats = sorted(pd.to_numeric(membership["repeat"]).astype(int).unique())
    if repeats != list(range(1, expected_repeats + 1)):
        raise ValueError("unexpected control split repeats")
    for repeat in repeats:
        folds = sorted(
            pd.to_numeric(
                membership.loc[membership["repeat"].astype(int).eq(repeat), "fold"]
            ).astype(int).unique()
        )
        for fold in folds:
            train_idx, test_idx = split_indices(membership, repeat, fold, participant_ids)
            probability, _ = fit_predict_dense(
                features[train_idx],
                labels[train_idx],
                features[test_idx],
                c=1.0,
                seed=base_seed + repeat,
            )
            train_hash = _participant_ids_sha256(participant_ids[train_idx])
            test_hash = _participant_ids_sha256(participant_ids[test_idx])
            for local_index, global_index in enumerate(test_idx):
                rows.append(
                    {
                        "condition": condition,
                        "repeat": repeat,
                        "fold": fold,
                        "participant_id": participant_ids[global_index],
                        "label": int(labels[global_index]),
                        "raw_probability": float(probability[local_index]),
                        "model_seed": base_seed + repeat,
                        "train_ids_sha256": train_hash,
                        "test_ids_sha256": test_hash,
                    }
                )
    fold_predictions = pd.DataFrame(rows).sort_values(
        ["repeat", "fold", "participant_id"], kind="stable"
    ).reset_index(drop=True)
    participant = _aggregate_raw(fold_predictions, expected_repeats)
    return {
        "fold_predictions": fold_predictions,
        "participant_predictions": participant,
        "summary": _raw_summary(participant, condition),
    }


def _derived_seed(master_seed: int, participant_id: int | str) -> int:
    digest = hashlib.sha256(f"{master_seed}:{participant_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def build_c5_control_inputs(
    participant_speech: pd.DataFrame,
    reviewed_evidence: pd.DataFrame,
    reviewed_spans: pd.DataFrame,
    *,
    random_master_seeds: Sequence[int],
    nonsymptom_master_seed: int,
    keywords: Sequence[str],
) -> dict[str, object]:
    participant_speech = participant_speech.sort_values("participant_id", kind="stable").copy()
    reviewed_evidence = reviewed_evidence.sort_values("participant_id", kind="stable").copy()
    if not np.array_equal(
        participant_speech["participant_id"].to_numpy(),
        reviewed_evidence["participant_id"].to_numpy(),
    ):
        raise ValueError("participant speech and reviewed evidence IDs do not align")
    participant_ids = participant_speech["participant_id"].tolist()
    target_words = {
        row.participant_id: len(str(row.text or "").split())
        for row in reviewed_evidence.itertuples()
    }

    random_tables = {}
    random_audit_rows = []
    for seed_index, master_seed in enumerate(random_master_seeds, start=1):
        table = participant_speech.copy()
        sampled = []
        for row in participant_speech.itertuples():
            derived = _derived_seed(master_seed, row.participant_id)
            text = length_matched_sample(
                row.text,
                target_words=target_words[row.participant_id],
                seed=derived,
            )
            sampled.append(text)
            random_audit_rows.append(
                {
                    "seed_index": seed_index,
                    "master_seed": master_seed,
                    "participant_id": row.participant_id,
                    "derived_seed": derived,
                    "target_words": target_words[row.participant_id],
                    "observed_words": len(text.split()),
                }
            )
        table["condition_id"] = f"c5_random_same_word_seed_{seed_index:02d}"
        table["text"] = sampled
        random_tables[f"seed_{seed_index:02d}"] = table

    nonsymptom = participant_speech.copy()
    nonsymptom_texts = []
    exclusion_rows = []
    for row in participant_speech.itertuples():
        quotes = reviewed_spans.loc[
            reviewed_spans["participant_id"].eq(row.participant_id)
        ].sort_values("source_order", kind="stable")["exact_quote"].tolist()
        remaining, audit = exclude_exact_quote_tokens(row.text, quotes)
        target = target_words[row.participant_id]
        derived = _derived_seed(nonsymptom_master_seed, row.participant_id)
        text = length_matched_sample(remaining, target_words=target, seed=derived)
        nonsymptom_texts.append(text)
        exclusion_rows.append(
            {
                "participant_id": row.participant_id,
                "master_seed": nonsymptom_master_seed,
                "derived_seed": derived,
                "target_words": target,
                "observed_words": len(text.split()),
                **audit,
            }
        )
    nonsymptom["condition_id"] = "c5_nonsymptom_same_word"
    nonsymptom["text"] = nonsymptom_texts

    presence, counts = build_domain_features(reviewed_spans, participant_ids)
    keyword_table = reviewed_evidence.copy()
    masked_texts = []
    mask_rows = []
    for row in reviewed_evidence.itertuples():
        masked, count = keyword_mask(row.text, keywords)
        masked_texts.append(masked)
        mask_rows.append(
            {
                "participant_id": row.participant_id,
                "masked_token_count": count,
                "word_count_before": len(str(row.text or "").split()),
                "word_count_after": len(masked.split()),
            }
        )
    keyword_table["condition_id"] = "c5_keyword_masked"
    keyword_table["text"] = masked_texts
    return {
        "random_tables": random_tables,
        "random_audit": pd.DataFrame(random_audit_rows),
        "nonsymptom": nonsymptom,
        "quote_exclusion_audit": pd.DataFrame(exclusion_rows),
        "domain_presence": presence,
        "domain_count": counts,
        "keyword_masked": keyword_table,
        "keyword_mask_audit": pd.DataFrame(mask_rows),
    }


def build_c5_family_tests(
    reviewed: pd.DataFrame,
    random_tables: dict[str, pd.DataFrame],
    named_controls: dict[str, pd.DataFrame],
    *,
    n_permutations: int = 10_000,
    base_seed: int = 20260624,
) -> pd.DataFrame:
    reviewed = reviewed.sort_values("participant_id", kind="stable")
    ids = reviewed["participant_id"].to_numpy()
    labels = reviewed["label"].to_numpy(dtype=int)
    metrics = {
        "roc_auc": ("raw_probability", roc_auc_score),
        "macro_f1_at_0_5": (
            "prediction_at_0_5",
            lambda y, pred: f1_score(y, pred, average="macro", zero_division=0),
        ),
    }
    comparisons = {**random_tables, **named_controls}
    rows = []
    test_index = 0
    for control_name, control in comparisons.items():
        control = control.sort_values("participant_id", kind="stable")
        if not np.array_equal(ids, control["participant_id"].to_numpy()) or not np.array_equal(
            labels, control["label"].to_numpy(dtype=int)
        ):
            raise ValueError(f"C5 control {control_name} is not participant-aligned")
        for metric_name, (column, metric) in metrics.items():
            test_index += 1
            result = paired_permutation_test(
                labels,
                reviewed[column].to_numpy(),
                control[column].to_numpy(),
                metric,
                n_permutations=n_permutations,
                seed=base_seed + test_index,
            )
            rows.append(
                {
                    "family": "c5_evidence_controls",
                    "condition_a": "c5_reviewed_evidence",
                    "condition_b": control_name,
                    "metric": metric_name,
                    "n": int(len(labels)),
                    **result,
                }
            )
    return apply_bh_by_family(pd.DataFrame(rows))


def build_c4_control_inputs(
    turns: pd.DataFrame,
    participant_ids: Sequence[int | str],
    *,
    length_master_seed: int = 20260715,
) -> dict[str, pd.DataFrame]:
    required = {
        "participant_id",
        "turn_index",
        "speaker_norm",
        "normalized_spoken_text",
        "is_explicit_clinical_prompt",
    }
    if not required.issubset(turns.columns):
        raise ValueError(f"C4 turn table missing columns: {sorted(required-set(turns.columns))}")
    participant_ids = list(participant_ids)
    turns = turns[
        turns["participant_id"].isin(participant_ids)
        & turns["speaker_norm"].astype(str).eq("interviewer")
    ].copy()
    turns["is_explicit_clinical_prompt"] = pd.to_numeric(
        turns["is_explicit_clinical_prompt"], errors="raise"
    ).astype(int)
    turns["normalized_spoken_text"] = turns["normalized_spoken_text"].fillna("").astype(str)

    rows = []
    for participant_id in participant_ids:
        group = turns[turns["participant_id"].eq(participant_id)].sort_values(
            "turn_index", kind="stable"
        )
        template_parts = group.loc[
            group["is_explicit_clinical_prompt"].eq(1), "normalized_spoken_text"
        ].tolist()
        non_parts = group.loc[
            group["is_explicit_clinical_prompt"].eq(0), "normalized_spoken_text"
        ].tolist()
        template_text = " ".join(part for part in template_parts if part.strip())
        non_text = " ".join(part for part in non_parts if part.strip())
        target = len(template_text.split())
        derived = _derived_seed(length_master_seed, participant_id)
        lengthmatched = length_matched_sample(
            non_text, target_words=target, seed=derived
        )
        position_input = group.copy()
        position_input["is_protocol_template_spoken"] = position_input[
            "is_explicit_clinical_prompt"
        ]
        positionmatched = position_matched_turns(
            position_input, target_words=target
        )
        rows.append(
            {
                "participant_id": participant_id,
                "template_only": template_text,
                "nontemplate_text": non_text,
                "nontemplate_lengthmatched": lengthmatched,
                "nontemplate_positionmatched": positionmatched,
                "template_word_count": target,
                "nontemplate_word_count": len(non_text.split()),
                "lengthmatched_word_count": len(lengthmatched.split()),
                "positionmatched_word_count": len(positionmatched.split()),
                "length_master_seed": length_master_seed,
                "length_derived_seed": derived,
                "template_turn_count": int(
                    group["is_explicit_clinical_prompt"].sum()
                ),
            }
        )
    participant_texts = pd.DataFrame(rows)
    if not participant_texts["template_word_count"].equals(
        participant_texts["lengthmatched_word_count"]
    ) or not participant_texts["template_word_count"].equals(
        participant_texts["positionmatched_word_count"]
    ):
        raise ValueError("C4 matched controls do not exactly match template word counts")

    explicit = turns[
        turns["is_explicit_clinical_prompt"].eq(1)
        & turns["normalized_spoken_text"].str.strip().ne("")
    ].copy()
    vocabulary = sorted(explicit["normalized_spoken_text"].unique())
    template_id_map = pd.DataFrame(
        {
            "template_id": [f"T{i:03d}" for i in range(1, len(vocabulary) + 1)],
            "normalized_template_text": vocabulary,
        }
    )
    text_to_id = dict(
        zip(template_id_map["normalized_template_text"], template_id_map["template_id"])
    )
    presence = pd.DataFrame({"participant_id": participant_ids})
    for template_id in template_id_map["template_id"]:
        presence[template_id] = 0
    if vocabulary:
        explicit["template_id"] = explicit["normalized_spoken_text"].map(text_to_id)
        counts = (
            explicit.groupby(["participant_id", "template_id"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=template_id_map["template_id"].tolist(), fill_value=0)
        )
        counts.index.name = "participant_id"
        presence = presence[["participant_id"]].merge(
            counts.reset_index(), on="participant_id", how="left"
        )
        presence[template_id_map["template_id"].tolist()] = presence[
            template_id_map["template_id"].tolist()
        ].fillna(0).astype(int)
    return {
        "participant_texts": participant_texts,
        "template_presence": presence,
        "template_id_map": template_id_map,
    }


def run_fold_safe_shuffled_text_condition(
    participant_ids: np.ndarray,
    labels: np.ndarray,
    texts: dict[int | str, str],
    membership: pd.DataFrame,
    *,
    condition: str,
    expected_repeats: int = 10,
    base_seed: int = 20260624,
) -> dict[str, object]:
    participant_ids = np.asarray(participant_ids)
    labels = np.asarray(labels, dtype=int)
    rows = []
    mapping_rows = []
    repeats = sorted(pd.to_numeric(membership["repeat"]).astype(int).unique())
    if repeats != list(range(1, expected_repeats + 1)):
        raise ValueError("unexpected shuffled-control split repeats")
    for repeat in repeats:
        folds = sorted(
            pd.to_numeric(
                membership.loc[membership["repeat"].astype(int).eq(repeat), "fold"]
            ).astype(int).unique()
        )
        for fold in folds:
            train_idx, test_idx = split_indices(membership, repeat, fold, participant_ids)
            seed = base_seed + 2000 + 100 * repeat + fold
            shuffled, mapping = fold_safe_shuffle(
                participant_ids,
                texts,
                train_ids=participant_ids[train_idx],
                test_ids=participant_ids[test_idx],
                seed=seed,
            )
            train_text = np.asarray([shuffled[pid] for pid in participant_ids[train_idx]], dtype=object)
            test_text = np.asarray([shuffled[pid] for pid in participant_ids[test_idx]], dtype=object)
            probability, _ = fit_predict_tfidf(
                train_text,
                labels[train_idx],
                test_text,
                seed=base_seed + repeat,
            )
            train_hash = _participant_ids_sha256(participant_ids[train_idx])
            test_hash = _participant_ids_sha256(participant_ids[test_idx])
            for local_index, global_index in enumerate(test_idx):
                rows.append(
                    {
                        "condition": condition,
                        "repeat": repeat,
                        "fold": fold,
                        "participant_id": participant_ids[global_index],
                        "label": int(labels[global_index]),
                        "raw_probability": float(probability[local_index]),
                        "shuffle_seed": seed,
                        "train_ids_sha256": train_hash,
                        "test_ids_sha256": test_hash,
                    }
                )
            mapping = mapping.copy()
            mapping["repeat"] = repeat
            mapping["fold"] = fold
            mapping["recipient_role"] = mapping["role"]
            mapping["donor_role"] = mapping["role"]
            mapping_rows.append(mapping)
    fold_predictions = pd.DataFrame(rows).sort_values(
        ["repeat", "fold", "participant_id"], kind="stable"
    ).reset_index(drop=True)
    participant = _aggregate_raw(fold_predictions, expected_repeats)
    return {
        "fold_predictions": fold_predictions,
        "participant_predictions": participant,
        "shuffle_mapping": pd.concat(mapping_rows, ignore_index=True),
        "summary": _raw_summary(participant, condition),
    }


def build_c4_family_tests(
    template_only: pd.DataFrame,
    controls: dict[str, pd.DataFrame],
    *,
    n_permutations: int = 10_000,
    base_seed: int = 20260624,
) -> pd.DataFrame:
    required_controls = {
        "nontemplate_lengthmatched",
        "nontemplate_positionmatched",
        "template_shuffled",
        "template_presence",
    }
    if set(controls) != required_controls:
        raise ValueError("C4 family requires exactly four predeclared controls")
    template_only = template_only.sort_values("participant_id", kind="stable")
    ids = template_only["participant_id"].to_numpy()
    labels = template_only["label"].to_numpy(dtype=int)
    metrics = {
        "roc_auc": ("raw_probability", roc_auc_score),
        "macro_f1_at_0_5": (
            "prediction_at_0_5",
            lambda y, pred: f1_score(y, pred, average="macro", zero_division=0),
        ),
    }
    rows = []
    test_index = 0
    for name, control in controls.items():
        control = control.sort_values("participant_id", kind="stable")
        if not np.array_equal(ids, control["participant_id"].to_numpy()) or not np.array_equal(
            labels, control["label"].to_numpy(dtype=int)
        ):
            raise ValueError(f"C4 control {name} is not participant-aligned")
        for metric_name, (column, metric) in metrics.items():
            test_index += 1
            result = paired_permutation_test(
                labels,
                template_only[column].to_numpy(),
                control[column].to_numpy(),
                metric,
                n_permutations=n_permutations,
                seed=base_seed + test_index,
            )
            rows.append(
                {
                    "family": "c4_interviewer_controls",
                    "condition_a": "template_only",
                    "condition_b": name,
                    "metric": metric_name,
                    "n": int(len(labels)),
                    **result,
                }
            )
    return apply_bh_by_family(pd.DataFrame(rows))
