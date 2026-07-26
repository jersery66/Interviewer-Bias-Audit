"""Version-3 blinded content-validity workflow for the participant-derived D-P proxy.

The workflow keeps the original 30-person probability sample intact, adds a
separately flagged rare-domain enrichment sample, and treats participant 385 as
an owner-side sentinel.  Reviewer-facing stage-1 files never contain D-P
values, sample type, participant identifiers, PHQ labels, or C5 evidence.
Stage 2 can be constructed only after the human consensus file is frozen.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .embedding_incremental import DOMAIN_FEATURES


COMMON_SYMPTOM_DOMAINS = tuple(
    domain
    for domain in DOMAIN_FEATURES
    if domain not in {"mental_health_history", "protective_or_absent_symptom"}
)
COMMON_SYMPTOM_STATUSES = (
    "current_present",
    "explicitly_absent",
    "past_only",
    "conflicting",
    "indeterminate",
)
MENTAL_HEALTH_HISTORY_STATUSES = (
    "history_present",
    "explicitly_no_history",
    "indeterminate",
)
PROTECTIVE_STATUSES = (
    "protective_denial_evidence_present",
    "no_protective_denial_evidence",
    "indeterminate",
)
INDETERMINATE_REASONS = (
    "not_mentioned",
    "not_covered",
    "answer_too_short",
    "insufficient_context",
    "ambiguous_expression",
    "other",
)
COUNT_BINS = ("0", "1", "2", "3+")
TEMPORALITY_VALUES = ("current", "past", "mixed", "unclear", "not_applicable")
NEGATION_VALUES = ("affirmed", "explicitly_negated", "mixed", "unclear", "not_applicable")
SUBJECT_VALUES = ("participant", "other_person", "mixed", "unclear", "not_applicable")

RARE_TARGETS = {
    "appetite_weight_positive": ("appetite_weight", "positive"),
    "suicide_self_harm_positive": ("suicide_self_harm", "positive"),
    "anhedonia_interest_positive": ("anhedonia_interest", "positive"),
    "sleep_fatigue_energy_zero": ("sleep_fatigue_energy", "zero"),
    "protective_or_absent_symptom_zero": ("protective_or_absent_symptom", "zero"),
}

STAGE1_REVIEW_COLUMNS = (
    "reviewer_id",
    "blind_id",
    "domain",
    "manual_status",
    "indeterminate_reason",
    "count_bin",
    "evidence_quote",
    "line_number",
    "temporality",
    "negation",
    "subject",
    "uncertainty_note",
)
FORBIDDEN_STAGE1_COLUMNS = {
    "participant_id",
    "sample_type",
    "paper_label_phq8_ge10",
    "paper_phq8_score",
    "phq8_score",
    "interviewer_text",
    "existing_dp",
    "c5_evidence",
    *DOMAIN_FEATURES,
}
_LINE_NUMBER_RE = re.compile(r"^\s*\d+(?:\s*[,;]\s*\d+)*\s*$")


def _require_columns(frame: pd.DataFrame, required: Sequence[str], *, name: str) -> None:
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def _line_number_text(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def _target_flags(domain_counts: pd.DataFrame) -> pd.DataFrame:
    _require_columns(domain_counts, ("participant_id", *DOMAIN_FEATURES), name="domain counts")
    out = domain_counts[["participant_id"]].copy()
    for target, (domain, state) in RARE_TARGETS.items():
        numeric = pd.to_numeric(domain_counts[domain], errors="raise")
        out[target] = numeric.gt(0) if state == "positive" else numeric.eq(0)
    return out


def select_enrichment_sample(
    domain_counts: pd.DataFrame,
    *,
    excluded_participant_ids: set[int] | set[str],
    target_quota: int = 4,
    min_n: int = 10,
    max_n: int = 15,
) -> pd.DataFrame:
    """Select a deterministic minimum-coverage rare-domain enrichment sample."""

    if not (1 <= int(target_quota)):
        raise ValueError("target_quota must be positive")
    if not (1 <= int(min_n) <= int(max_n)):
        raise ValueError("sample bounds must satisfy 1 <= min_n <= max_n")
    flags = _target_flags(domain_counts)
    excluded = {int(value) for value in excluded_participant_ids}
    candidates = flags.loc[~flags["participant_id"].astype(int).isin(excluded)].copy()
    candidates["participant_id"] = candidates["participant_id"].astype(int)
    available = {target: int(candidates[target].sum()) for target in RARE_TARGETS}
    required = {target: min(int(target_quota), available[target]) for target in RARE_TARGETS}
    if any(value == 0 for value in required.values()):
        absent = [target for target, value in required.items() if value == 0]
        raise ValueError(f"no eligible participant covers rare target(s): {absent}")

    remaining = required.copy()
    selected_ids: list[int] = []
    while any(value > 0 for value in remaining.values()):
        best_id: int | None = None
        best_key: tuple[float, float, int] | None = None
        for row in candidates.itertuples(index=False):
            participant_id = int(row.participant_id)
            if participant_id in selected_ids:
                continue
            active_targets = [
                target
                for target in RARE_TARGETS
                if remaining[target] > 0 and bool(getattr(row, target))
            ]
            coverage = float(len(active_targets))
            scarcity = float(sum(1.0 / available[target] for target in active_targets))
            key = (coverage, scarcity, -participant_id)
            if best_key is None or key > best_key:
                best_key = key
                best_id = participant_id
        if best_id is None or best_key is None or best_key[0] == 0:
            raise ValueError(f"rare-target quotas are infeasible; remaining={remaining}")
        selected_ids.append(best_id)
        chosen = candidates.loc[candidates["participant_id"].eq(best_id)].iloc[0]
        for target in RARE_TARGETS:
            if remaining[target] > 0 and bool(chosen[target]):
                remaining[target] -= 1
        if len(selected_ids) > int(max_n):
            raise ValueError(
                f"rare-target quotas require more than max_n={max_n} participants"
            )

    if len(selected_ids) < int(min_n):
        candidates["total_target_coverage"] = candidates[list(RARE_TARGETS)].sum(axis=1)
        fillers = candidates.loc[~candidates["participant_id"].isin(selected_ids)].sort_values(
            ["total_target_coverage", "participant_id"],
            ascending=[False, True],
            kind="stable",
        )
        selected_ids.extend(
            fillers["participant_id"].head(int(min_n) - len(selected_ids)).astype(int).tolist()
        )
    if len(selected_ids) > int(max_n):
        raise ValueError("enrichment sample exceeds max_n")

    selected = candidates.loc[candidates["participant_id"].isin(selected_ids)].copy()
    selected["selection_order"] = selected["participant_id"].map(
        {participant_id: index + 1 for index, participant_id in enumerate(selected_ids)}
    )
    selected["selection_reasons"] = selected.apply(
        lambda row: ";".join(target for target in RARE_TARGETS if bool(row[target])),
        axis=1,
    )
    return selected.sort_values("selection_order", kind="stable").reset_index(drop=True)


def build_validation_sample(
    main_sample: pd.DataFrame,
    enrichment_sample: pd.DataFrame,
    *,
    sentinel_participant_id: int,
) -> pd.DataFrame:
    """Combine samples owner-side while retaining their analysis strata."""

    _require_columns(main_sample, ("participant_id",), name="main sample")
    _require_columns(enrichment_sample, ("participant_id",), name="enrichment sample")
    main = main_sample.copy()
    main["sample_type"] = "main"
    enriched = enrichment_sample.copy()
    enriched["sample_type"] = "enriched"
    sentinel = pd.DataFrame(
        {"participant_id": [int(sentinel_participant_id)], "sample_type": ["sentinel"]}
    )
    all_columns = list(dict.fromkeys([*main.columns, *enriched.columns, *sentinel.columns]))
    combined = pd.concat(
        [main.reindex(columns=all_columns), enriched.reindex(columns=all_columns), sentinel.reindex(columns=all_columns)],
        ignore_index=True,
    )
    combined["participant_id"] = combined["participant_id"].astype(int)
    if combined["participant_id"].duplicated().any():
        raise ValueError("main, enrichment, and sentinel samples must be disjoint")
    return combined


def make_opaque_blind_key(sample: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """Assign opaque IDs across sample types so type is not visible to reviewers."""

    _require_columns(sample, ("participant_id", "sample_type"), name="validation sample")
    rng = np.random.default_rng(int(seed))
    source = sample.copy()
    if "blind_id" in source.columns:
        source = source.rename(columns={"blind_id": "packet2_blind_id"})
    out = source.iloc[rng.permutation(len(source))].reset_index(drop=True).copy()
    out.insert(0, "blind_id", [f"V{index:03d}" for index in range(1, len(out) + 1)])
    return out


def build_stage1_review_form(
    blind_ids: Iterable[str], *, reviewer_id: str, order_seed: int
) -> pd.DataFrame:
    ids = [str(value) for value in blind_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("blind IDs must be unique")
    rng = np.random.default_rng(int(order_seed))
    ordered_ids = [ids[index] for index in rng.permutation(len(ids))]
    rows = []
    for blind_id in ordered_ids:
        for domain in DOMAIN_FEATURES:
            rows.append(
                {
                    "reviewer_id": str(reviewer_id),
                    "blind_id": blind_id,
                    "domain": domain,
                    "manual_status": pd.NA,
                    "indeterminate_reason": pd.NA,
                    "count_bin": pd.NA,
                    "evidence_quote": pd.NA,
                    "line_number": pd.NA,
                    "temporality": pd.NA,
                    "negation": pd.NA,
                    "subject": pd.NA,
                    "uncertainty_note": pd.NA,
                }
            )
    return pd.DataFrame(rows, columns=list(STAGE1_REVIEW_COLUMNS))


def _allowed_statuses(domain: str) -> tuple[str, ...]:
    if domain == "mental_health_history":
        return MENTAL_HEALTH_HISTORY_STATUSES
    if domain == "protective_or_absent_symptom":
        return PROTECTIVE_STATUSES
    if domain in COMMON_SYMPTOM_DOMAINS:
        return COMMON_SYMPTOM_STATUSES
    raise ValueError(f"unknown D-P domain: {domain}")


def validate_stage1_reviewer_form(
    form: pd.DataFrame,
    *,
    expected_blind_ids: Sequence[str],
    require_completed: bool,
) -> None:
    _require_columns(form, STAGE1_REVIEW_COLUMNS, name="stage-1 reviewer form")
    forbidden = FORBIDDEN_STAGE1_COLUMNS.intersection(form.columns)
    if forbidden:
        raise ValueError(f"stage-1 form exposes forbidden fields: {sorted(forbidden)}")
    if form[["blind_id", "domain"]].duplicated().any():
        raise ValueError("stage-1 form contains duplicate blind_id/domain rows")
    if set(form["blind_id"].astype(str)) != set(map(str, expected_blind_ids)):
        raise ValueError("stage-1 form does not contain exactly the expected blind IDs")
    if set(form["domain"].astype(str)) != set(DOMAIN_FEATURES):
        raise ValueError("stage-1 form does not contain all ten D-P domains")
    if len(form) != len(expected_blind_ids) * len(DOMAIN_FEATURES):
        raise ValueError("stage-1 form has an unexpected row count")
    if not require_completed:
        return

    for row in form.itertuples(index=False):
        status = str(row.manual_status).strip()
        if status not in _allowed_statuses(str(row.domain)):
            raise ValueError(f"invalid manual_status={status!r} for domain={row.domain}")
        reason = "" if pd.isna(row.indeterminate_reason) else str(row.indeterminate_reason).strip()
        if status == "indeterminate" and reason not in INDETERMINATE_REASONS:
            raise ValueError("indeterminate status requires a prespecified reason")
        if status != "indeterminate" and reason:
            raise ValueError("indeterminate_reason must be blank outside indeterminate")
        count_bin = "" if pd.isna(row.count_bin) else str(row.count_bin).strip()
        if count_bin not in COUNT_BINS:
            raise ValueError("completed stage-1 rows require count_bin 0/1/2/3+")
        temporality = "" if pd.isna(row.temporality) else str(row.temporality).strip()
        negation = "" if pd.isna(row.negation) else str(row.negation).strip()
        subject = "" if pd.isna(row.subject) else str(row.subject).strip()
        if temporality not in TEMPORALITY_VALUES:
            raise ValueError("invalid temporality")
        if negation not in NEGATION_VALUES:
            raise ValueError("invalid negation")
        if subject not in SUBJECT_VALUES:
            raise ValueError("invalid subject")
        evidence_required = status not in {
            "indeterminate",
            "no_protective_denial_evidence",
        }
        quote = "" if pd.isna(row.evidence_quote) else str(row.evidence_quote).strip()
        line_number = _line_number_text(row.line_number)
        if evidence_required and not quote:
            raise ValueError(f"manual_status={status} requires exact evidence")
        if evidence_required and not _LINE_NUMBER_RE.match(line_number):
            raise ValueError(f"manual_status={status} requires numeric line numbers")


def manual_status_to_binary(
    domain: str,
    status: str,
    *,
    conflict_policy: str = "exclude",
) -> int | None:
    if conflict_policy not in {"exclude", "as_0", "as_1"}:
        raise ValueError(f"invalid conflict_policy={conflict_policy!r}")
    status = str(status).strip()
    if status not in _allowed_statuses(domain):
        raise ValueError(f"invalid status={status!r} for domain={domain}")
    if domain == "mental_health_history":
        return int(status == "history_present")
    if domain == "protective_or_absent_symptom":
        return int(status == "protective_denial_evidence_present")
    if status == "conflicting":
        return {"exclude": None, "as_0": 0, "as_1": 1}[conflict_policy]
    return int(status == "current_present")


def _merge_reviewers(reviewer_a: pd.DataFrame, reviewer_b: pd.DataFrame) -> pd.DataFrame:
    _require_columns(reviewer_a, STAGE1_REVIEW_COLUMNS, name="reviewer A")
    _require_columns(reviewer_b, STAGE1_REVIEW_COLUMNS, name="reviewer B")
    key = ["blind_id", "domain"]
    left = reviewer_a.drop(columns=["reviewer_id"]).rename(
        columns={column: f"reviewer_A_{column}" for column in STAGE1_REVIEW_COLUMNS if column != "reviewer_id"}
    )
    right = reviewer_b.drop(columns=["reviewer_id"]).rename(
        columns={column: f"reviewer_B_{column}" for column in STAGE1_REVIEW_COLUMNS if column != "reviewer_id"}
    )
    merged = left.merge(
        right,
        left_on=[f"reviewer_A_{column}" for column in key],
        right_on=[f"reviewer_B_{column}" for column in key],
        how="outer",
        validate="one_to_one",
    )
    if merged[["reviewer_A_blind_id", "reviewer_B_blind_id"]].isna().any().any():
        raise ValueError("reviewer forms do not cover the same cells")
    return merged


def build_stage1_consensus_template(
    reviewer_a: pd.DataFrame, reviewer_b: pd.DataFrame
) -> pd.DataFrame:
    merged = _merge_reviewers(reviewer_a, reviewer_b)
    out = pd.DataFrame(
        {
            "blind_id": merged["reviewer_A_blind_id"],
            "domain": merged["reviewer_A_domain"],
        }
    )
    for reviewer in ("A", "B"):
        for column in STAGE1_REVIEW_COLUMNS:
            if column in {"reviewer_id", "blind_id", "domain", "uncertainty_note"}:
                continue
            out[f"reviewer_{reviewer}_{column}"] = merged[f"reviewer_{reviewer}_{column}"]
    for column in (
        "status",
        "indeterminate_reason",
        "count_bin",
        "evidence_quote",
        "line_number",
        "temporality",
        "negation",
        "subject",
    ):
        out[f"consensus_{column}"] = pd.NA
    out["adjudication_reason"] = pd.NA
    return out.sort_values(["blind_id", "domain"], kind="stable").reset_index(drop=True)


def validate_stage1_consensus(
    consensus: pd.DataFrame,
    reviewer_a: pd.DataFrame,
    reviewer_b: pd.DataFrame,
) -> None:
    expected = build_stage1_consensus_template(reviewer_a, reviewer_b)
    raw_columns = [
        column
        for column in expected.columns
        if column.startswith("reviewer_") or column in {"blind_id", "domain"}
    ]
    _require_columns(consensus, expected.columns, name="stage-1 consensus")
    left = expected[raw_columns].sort_values(["blind_id", "domain"]).reset_index(drop=True)
    right = consensus[raw_columns].sort_values(["blind_id", "domain"]).reset_index(drop=True)
    if not left.fillna("").equals(right.fillna("")):
        raise ValueError("consensus does not preserve raw reviewer judgments")
    for row in consensus.itertuples(index=False):
        status = str(row.consensus_status).strip()
        if status not in _allowed_statuses(str(row.domain)):
            raise ValueError("invalid consensus status")
        reason = "" if pd.isna(row.adjudication_reason) else str(row.adjudication_reason).strip()
        if not reason:
            raise ValueError("every consensus cell requires an adjudication reason")
        indeterminate_reason = (
            "" if pd.isna(row.consensus_indeterminate_reason) else str(row.consensus_indeterminate_reason).strip()
        )
        if status == "indeterminate" and indeterminate_reason not in INDETERMINATE_REASONS:
            raise ValueError("indeterminate consensus requires a prespecified reason")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_consensus(consensus_path: Path, freeze_path: Path) -> dict[str, object]:
    if not consensus_path.exists():
        raise FileNotFoundError(consensus_path)
    record = {
        "status": "human_consensus_frozen_stage2_may_open",
        "consensus_path": str(consensus_path),
        "consensus_sha256": _sha256(consensus_path),
    }
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def verify_consensus_freeze(consensus_path: Path, freeze_path: Path) -> dict[str, object]:
    if not freeze_path.exists():
        raise FileNotFoundError(freeze_path)
    record = json.loads(freeze_path.read_text(encoding="utf-8"))
    if record.get("status") != "human_consensus_frozen_stage2_may_open":
        raise ValueError("consensus freeze record has an invalid status")
    if record.get("consensus_sha256") != _sha256(consensus_path):
        raise ValueError("consensus file changed after freeze")
    return record


def _normalize_space(value: object) -> str:
    return " ".join(str(value).split()).strip().lower()


def _quote_line_numbers(blind_id: str, quote: str, lines: pd.DataFrame) -> str:
    subset = lines.loc[lines["blind_id"].astype(str).eq(str(blind_id))]
    normalized_quote = _normalize_space(quote)
    matches = [
        int(row.line_number)
        for row in subset.itertuples(index=False)
        if normalized_quote and normalized_quote in _normalize_space(row.participant_text_line)
    ]
    return ";".join(str(value) for value in matches)


def _split_events(value: object) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    return [part.strip() for part in str(value).split(" || ") if part.strip()]


def build_stage2_evidence_forms(
    consensus: pd.DataFrame,
    blind_key: pd.DataFrame,
    spans: pd.DataFrame,
    line_index: pd.DataFrame,
    *,
    reviewer_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expand actual C5 span rows and human-reference events after consensus freeze."""

    _require_columns(blind_key, ("blind_id", "participant_id", "sample_type"), name="blind key")
    _require_columns(
        spans,
        ("participant_id", "source_order", "domain", "polarity", "exact_quote"),
        name="reviewed C5 spans",
    )
    _require_columns(line_index, ("blind_id", "line_number", "participant_text_line"), name="line index")
    joined = blind_key[["blind_id", "participant_id"]].merge(
        spans[["participant_id", "source_order", "domain", "polarity", "exact_quote"]],
        on="participant_id",
        how="inner",
        validate="one_to_many",
    )
    evidence_rows: list[dict[str, object]] = []
    for row in joined.sort_values(["blind_id", "source_order"], kind="stable").itertuples(index=False):
        evidence_rows.append(
            {
                "reviewer_id": str(reviewer_id),
                "blind_id": str(row.blind_id),
                "domain": str(row.domain),
                "dp_evidence_id": f"{row.blind_id}-DP-{int(row.source_order):03d}",
                "dp_quote": str(row.exact_quote),
                "dp_line_number": _quote_line_numbers(str(row.blind_id), str(row.exact_quote), line_index),
                "dp_polarity": str(row.polarity),
                "source_supported": pd.NA,
                "speaker_correct": pd.NA,
                "domain_correct": pd.NA,
                "negation_preserved": pd.NA,
                "temporality_correct": pd.NA,
                "subject_correct": pd.NA,
                "context_preserved": pd.NA,
                "duplicate_of_evidence_id": pd.NA,
                "human_evidence_match": pd.NA,
                "error_notes": pd.NA,
            }
        )
    evidence = pd.DataFrame(evidence_rows)

    evidence_ids_by_cell: dict[tuple[str, str], str] = {}
    if not evidence.empty:
        evidence_ids_by_cell = (
            evidence.groupby(["blind_id", "domain"], sort=False)["dp_evidence_id"]
            .agg(lambda values: ";".join(values.astype(str)))
            .to_dict()
        )
    omission_rows: list[dict[str, object]] = []
    for row in consensus.itertuples(index=False):
        quotes = _split_events(getattr(row, "consensus_evidence_quote", pd.NA))
        line_numbers = _split_events(getattr(row, "consensus_line_number", pd.NA))
        for index, quote in enumerate(quotes, start=1):
            omission_rows.append(
                {
                    "reviewer_id": str(reviewer_id),
                    "blind_id": str(row.blind_id),
                    "domain": str(row.domain),
                    "human_evidence_id": f"{row.blind_id}-H-{str(row.domain)[:4]}-{index:02d}",
                    "human_evidence_quote": quote,
                    "human_line_number": line_numbers[index - 1] if index <= len(line_numbers) else pd.NA,
                    "same_cell_dp_evidence_ids": evidence_ids_by_cell.get((str(row.blind_id), str(row.domain)), ""),
                    "covered_by_dp": pd.NA,
                    "matched_dp_evidence_ids": pd.NA,
                    "important_omission": pd.NA,
                    "omission_reason": pd.NA,
                }
            )
    omissions = pd.DataFrame(omission_rows)
    return evidence, omissions


def _safe_kappa(y_true: pd.Series, y_pred: pd.Series, *, weights: str | None = None) -> float:
    values = set(y_true.tolist()) | set(y_pred.tolist())
    if len(values) < 2:
        return float("nan")
    return float(cohen_kappa_score(y_true, y_pred, weights=weights))


def _binary_metrics(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "positive_f1": float("nan"),
            "macro_f1": float("nan"),
            "cohen_kappa": float("nan"),
        }
    y_true = frame["human_binary"].astype(int)
    y_pred = frame["dp_binary"].astype(int)
    table = pd.crosstab(y_true, y_pred).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    tn, fp = int(table.loc[0, 0]), int(table.loc[0, 1])
    fn, tp = int(table.loc[1, 0]), int(table.loc[1, 1])
    positive_f1 = float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else float("nan")
    negative_f1 = float(2 * tn / (2 * tn + fp + fn)) if 2 * tn + fp + fn else float("nan")
    return {
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "positive_f1": positive_f1,
        "macro_f1": float(np.nanmean([positive_f1, negative_f1])),
        "cohen_kappa": _safe_kappa(y_true, y_pred),
    }


def _long_dp_reference(scoring_reference: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        scoring_reference,
        ("blind_id", "participant_id", "sample_type", *DOMAIN_FEATURES),
        name="scoring reference",
    )
    rows = []
    for row in scoring_reference.itertuples(index=False):
        for domain in DOMAIN_FEATURES:
            value = float(getattr(row, domain))
            rows.append(
                {
                    "blind_id": str(row.blind_id),
                    "participant_id": int(row.participant_id),
                    "sample_type": str(row.sample_type),
                    "domain": domain,
                    "dp_count": value,
                    "dp_binary": int(value > 0),
                }
            )
    return pd.DataFrame(rows)


def _validity_frame(consensus: pd.DataFrame, scoring_reference: pd.DataFrame) -> pd.DataFrame:
    _require_columns(consensus, ("blind_id", "domain", "consensus_status"), name="consensus")
    dp = _long_dp_reference(scoring_reference)
    frame = dp.merge(
        consensus[["blind_id", "domain", "consensus_status", "consensus_count_bin"]],
        on=["blind_id", "domain"],
        how="inner",
        validate="one_to_one",
    )
    frame["human_binary"] = frame.apply(
        lambda row: manual_status_to_binary(str(row["domain"]), str(row["consensus_status"])),
        axis=1,
    )
    return frame


def _cluster_bootstrap(
    frame: pd.DataFrame, *, bootstrap_n: int, seed: int
) -> dict[str, tuple[float, float]]:
    metric_names = ("sensitivity", "specificity", "positive_f1", "macro_f1", "cohen_kappa")
    if bootstrap_n <= 0 or frame.empty:
        return {name: (float("nan"), float("nan")) for name in metric_names}
    rng = np.random.default_rng(int(seed))
    ids = frame["blind_id"].drop_duplicates().astype(str).to_numpy()
    draws = {name: [] for name in metric_names}
    grouped = {blind_id: frame.loc[frame["blind_id"].astype(str).eq(blind_id)] for blind_id in ids}
    for _ in range(int(bootstrap_n)):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        boot = pd.concat([grouped[blind_id] for blind_id in sampled], ignore_index=True)
        metrics = _binary_metrics(boot)
        for name in metric_names:
            if np.isfinite(metrics[name]):
                draws[name].append(metrics[name])
    return {
        name: (
            float(np.quantile(values, 0.025)) if values else float("nan"),
            float(np.quantile(values, 0.975)) if values else float("nan"),
        )
        for name, values in draws.items()
    }


def compute_dp_validity(
    consensus: pd.DataFrame,
    scoring_reference: pd.DataFrame,
    *,
    bootstrap_n: int = 5000,
    seed: int = 20260721,
) -> pd.DataFrame:
    """Report main and enriched samples separately; sentinel is never pooled."""

    base = _validity_frame(consensus, scoring_reference)
    rows = []
    for policy_index, conflict_policy in enumerate(("exclude", "as_0", "as_1")):
        frame = base.copy()
        frame["human_binary"] = frame.apply(
            lambda row: manual_status_to_binary(
                str(row["domain"]),
                str(row["consensus_status"]),
                conflict_policy=conflict_policy,
            ),
            axis=1,
        )
        frame = frame.loc[frame["human_binary"].notna()].copy()
        for sample_type in ("main", "enriched"):
            sample = frame.loc[frame["sample_type"].eq(sample_type)]
            if sample.empty:
                continue
            scopes: list[tuple[str, pd.DataFrame]] = [("overall", sample)]
            scopes.extend((domain, part) for domain, part in sample.groupby("domain", sort=False))
            for scope_index, (scope, part) in enumerate(scopes):
                metrics = _binary_metrics(part)
                ci = _cluster_bootstrap(
                    part,
                    bootstrap_n=bootstrap_n,
                    seed=int(seed) + policy_index * 1000 + scope_index,
                )
                row: dict[str, object] = {
                    "sample_type": sample_type,
                    "conflict_policy": conflict_policy,
                    "analysis_role": "primary" if conflict_policy == "exclude" else "sensitivity",
                    "scope": str(scope),
                    "n_cells": int(len(part)),
                    "n_participants": int(part["blind_id"].nunique()),
                    "human_positive_n": int(part["human_binary"].astype(int).sum()),
                    "dp_positive_n": int(part["dp_binary"].astype(int).sum()),
                }
                row.update(metrics)
                for name, (low, high) in ci.items():
                    row[f"{name}_ci_low"] = low
                    row[f"{name}_ci_high"] = high
                rows.append(row)
    return pd.DataFrame(rows)


def _count_bin_numeric(value: object) -> float:
    text = str(value).strip()
    mapping = {"0": 0.0, "1": 1.0, "2": 2.0, "3+": 3.0}
    return mapping.get(text, float("nan"))


def compute_count_validity(
    consensus: pd.DataFrame, scoring_reference: pd.DataFrame
) -> pd.DataFrame:
    frame = _validity_frame(consensus, scoring_reference)
    frame = frame.loc[frame["sample_type"].isin(["main", "enriched"])].copy()
    frame["human_count"] = frame["consensus_count_bin"].map(_count_bin_numeric)
    frame["dp_count_capped"] = frame["dp_count"].clip(upper=3)
    frame = frame.loc[frame["human_count"].notna()].copy()
    rows = []
    for sample_type, sample in frame.groupby("sample_type", sort=False):
        scopes: list[tuple[str, pd.DataFrame]] = [("overall", sample)]
        scopes.extend((domain, part) for domain, part in sample.groupby("domain", sort=False))
        for scope, part in scopes:
            human = part["human_count"].astype(float)
            dp = part["dp_count_capped"].astype(float)
            spearman = (
                float(human.corr(dp, method="spearman"))
                if len(part) > 1 and human.nunique() > 1 and dp.nunique() > 1
                else float("nan")
            )
            rows.append(
                {
                    "sample_type": str(sample_type),
                    "scope": str(scope),
                    "n_cells": int(len(part)),
                    "exact_agreement": float((human == dp).mean()),
                    "weighted_kappa": _safe_kappa(human.astype(int), dp.astype(int), weights="linear"),
                    "mean_absolute_difference": float(np.abs(human - dp).mean()),
                    "spearman_rho": spearman,
                    "three_plus_is_capped_lower_bound": True,
                }
            )
    return pd.DataFrame(rows)


def compute_stage1_interrater(
    reviewer_a: pd.DataFrame, reviewer_b: pd.DataFrame
) -> pd.DataFrame:
    """Compute pre-consensus status, derived-binary, and count agreement."""

    merged = _merge_reviewers(reviewer_a, reviewer_b)
    rows: list[dict[str, object]] = []
    scopes: list[tuple[str, pd.DataFrame]] = [("overall", merged)]
    scopes.extend(
        (str(domain), part)
        for domain, part in merged.groupby("reviewer_A_domain", sort=False)
    )
    count_map = {"0": 0, "1": 1, "2": 2, "3+": 3}
    for scope, part in scopes:
        status_a = part["reviewer_A_manual_status"].fillna("").astype(str).str.strip()
        status_b = part["reviewer_B_manual_status"].fillna("").astype(str).str.strip()
        valid_status = status_a.ne("") & status_b.ne("")
        aa = status_a[valid_status]
        bb = status_b[valid_status]
        status_kappa = _safe_kappa(aa, bb) if len(aa) else float("nan")
        status_agreement = float((aa == bb).mean()) if len(aa) else float("nan")

        binary_a: list[float] = []
        binary_b: list[float] = []
        for row in part.itertuples(index=False):
            value_a = manual_status_to_binary(
                str(row.reviewer_A_domain), str(row.reviewer_A_manual_status)
            )
            value_b = manual_status_to_binary(
                str(row.reviewer_B_domain), str(row.reviewer_B_manual_status)
            )
            if value_a is not None and value_b is not None:
                binary_a.append(float(value_a))
                binary_b.append(float(value_b))
        binary_a_series = pd.Series(binary_a, dtype=float)
        binary_b_series = pd.Series(binary_b, dtype=float)
        binary_kappa = (
            _safe_kappa(binary_a_series, binary_b_series)
            if len(binary_a_series)
            else float("nan")
        )

        count_a = (
            part["reviewer_A_count_bin"].fillna("").astype(str).str.strip().map(count_map)
        )
        count_b = (
            part["reviewer_B_count_bin"].fillna("").astype(str).str.strip().map(count_map)
        )
        valid_count = count_a.notna() & count_b.notna()
        ca = count_a[valid_count].astype(int)
        cb = count_b[valid_count].astype(int)
        rows.append(
            {
                "scope": scope,
                "n_cells": int(len(part)),
                "status_complete_n": int(valid_status.sum()),
                "status_exact_agreement": status_agreement,
                "status_cohen_kappa": status_kappa,
                "binary_complete_n": int(len(binary_a_series)),
                "binary_exact_agreement": (
                    float((binary_a_series == binary_b_series).mean())
                    if len(binary_a_series)
                    else float("nan")
                ),
                "binary_cohen_kappa": binary_kappa,
                "count_complete_n": int(valid_count.sum()),
                "count_exact_agreement": (
                    float((ca == cb).mean()) if len(ca) else float("nan")
                ),
                "count_weighted_kappa": (
                    _safe_kappa(ca, cb, weights="linear") if len(ca) else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def _yes_no_error_rate(series: pd.Series) -> float:
    values = series.fillna("").astype(str).str.strip().str.lower()
    valid = values.isin(["yes", "no"])
    if not valid.any():
        return float("nan")
    return float(values[valid].eq("no").mean())


def compute_stage2_quality(
    evidence_review: pd.DataFrame,
    omission_review: pd.DataFrame,
    blind_key: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise evidence errors separately for main and enriched samples."""

    _require_columns(blind_key, ("blind_id", "sample_type"), name="blind key")
    _require_columns(
        evidence_review,
        (
            "blind_id",
            "source_supported",
            "speaker_correct",
            "domain_correct",
            "negation_preserved",
            "temporality_correct",
            "subject_correct",
            "context_preserved",
            "duplicate_of_evidence_id",
        ),
        name="evidence review",
    )
    _require_columns(
        omission_review,
        ("blind_id", "covered_by_dp", "important_omission"),
        name="omission review",
    )
    key = blind_key[["blind_id", "sample_type"]].copy()
    evidence = evidence_review.merge(key, on="blind_id", how="left", validate="many_to_one")
    omissions = omission_review.merge(key, on="blind_id", how="left", validate="many_to_one")
    rows: list[dict[str, object]] = []
    for sample_type in ("main", "enriched"):
        sample = evidence.loc[evidence["sample_type"].eq(sample_type)].copy()
        sample_omissions = omissions.loc[omissions["sample_type"].eq(sample_type)].copy()
        if sample.empty and sample_omissions.empty:
            continue
        duplicate = sample["duplicate_of_evidence_id"].fillna("").astype(str).str.strip().ne("")
        component_columns = [
            "source_supported",
            "speaker_correct",
            "domain_correct",
            "negation_preserved",
            "temporality_correct",
            "subject_correct",
            "context_preserved",
        ]
        valid_evidence = pd.Series(True, index=sample.index)
        for column in component_columns:
            valid_evidence &= sample[column].fillna("").astype(str).str.strip().str.lower().eq("yes")
        valid_evidence &= ~duplicate
        important = (
            sample_omissions["important_omission"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        important_valid = important.isin(["yes", "no"])
        rows.append(
            {
                "sample_type": sample_type,
                "evidence_row_n": int(len(sample)),
                "valid_evidence_rate": (
                    float(valid_evidence.mean()) if len(sample) else float("nan")
                ),
                "unsupported_evidence_rate": _yes_no_error_rate(sample["source_supported"]),
                "speaker_mismatch_rate": _yes_no_error_rate(sample["speaker_correct"]),
                "domain_mismatch_rate": _yes_no_error_rate(sample["domain_correct"]),
                "negation_error_rate": _yes_no_error_rate(sample["negation_preserved"]),
                "current_past_error_rate": _yes_no_error_rate(sample["temporality_correct"]),
                "subject_error_rate": _yes_no_error_rate(sample["subject_correct"]),
                "context_truncation_rate": _yes_no_error_rate(sample["context_preserved"]),
                "duplicate_rate": float(duplicate.mean()) if len(sample) else float("nan"),
                "human_evidence_event_n": int(len(sample_omissions)),
                "important_omission_rate": (
                    float(important[important_valid].eq("yes").mean())
                    if important_valid.any()
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)
