"""Locked workflow utilities for the D-P blinded manual audit.

This module separates the blinded first-pass review from owner-side scoring.  It
never joins the participant identifier, label, or frozen D-P values into a
reviewer-facing table.  The scoring functions require completed independent
files and keep the two raw reviewer columns intact when a consensus file is
created.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .embedding_incremental import DOMAIN_FEATURES


DOMAIN_LABELS = ("0", "1", "NS")
COUNT_BINS = ("0", "1", "2", "3+")
POLARITIES = ("present", "denied/protective", "ambiguous")
EVIDENCE_VALIDITY = ("valid_correct_domain", "valid_wrong_domain", "unsupported", "duplicate", "ambiguous")

REVIEW_COLUMNS = (
    "reviewer_id",
    "blind_id",
    "domain",
    "domain_label",
    "count_bin",
    "evidence_quote",
    "line_number",
    "polarity",
    "uncertainty_reason",
)
FORBIDDEN_REVIEW_COLUMNS = {
    "participant_id",
    "paper_label_phq8_ge10",
    "paper_phq8_score",
    "label",
    "phq8_score",
    "word_count",
    "computed_word_count",
    *DOMAIN_FEATURES,
    "existing_dp",
    "model_probability",
    "interviewer_text",
}
_LINE_NUMBER_RE = re.compile(r"^\s*\d+(?:\s*[,;]\s*\d+)*\s*$")


def _require_columns(frame: pd.DataFrame, required: Sequence[str], *, name: str) -> None:
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def make_blind_id_key(sample: pd.DataFrame) -> pd.DataFrame:
    """Create the owner-only mapping from blind IDs to participant IDs."""

    _require_columns(sample, ("audit_case_id", "participant_id"), name="sample")
    out = sample.copy().reset_index(drop=True)
    out.insert(0, "blind_id", [f"B{i:03d}" for i in range(1, len(out) + 1)])
    return out


def build_blind_cases(participants: pd.DataFrame, blind_key: pd.DataFrame) -> pd.DataFrame:
    """Return only the blind ID and participant language for reviewers."""

    _require_columns(participants, ("participant_id", "text"), name="participants")
    _require_columns(blind_key, ("blind_id", "participant_id"), name="blind_key")
    joined = blind_key[["blind_id", "participant_id"]].merge(
        participants[["participant_id", "text"]],
        on="participant_id",
        how="left",
        validate="one_to_one",
    )
    if joined["text"].isna().any():
        raise ValueError("blind case text is missing")
    return (
        joined[["blind_id", "text"]]
        .rename(columns={"text": "participant_text"})
        .sort_values("blind_id", kind="stable")
        .reset_index(drop=True)
    )


def build_line_index(cases: pd.DataFrame) -> pd.DataFrame:
    """Expand case text into stable, 1-based line numbers for quoting."""

    _require_columns(cases, ("blind_id", "participant_text"), name="cases")
    rows: list[dict[str, object]] = []
    for row in cases.itertuples(index=False):
        text = str(row.participant_text)
        lines = text.splitlines() or [""]
        for line_number, line_text in enumerate(lines, start=1):
            rows.append(
                {
                    "blind_id": str(row.blind_id),
                    "line_number": int(line_number),
                    "participant_text_line": line_text,
                }
            )
    return pd.DataFrame(rows)


def build_review_form(
    blind_ids: Iterable[str], *, reviewer_id: str, order_seed: int
) -> pd.DataFrame:
    """Build a blank 10-domain form with reviewer-specific case order."""

    ids = [str(value) for value in blind_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("blind IDs must be unique")
    rng = np.random.default_rng(int(order_seed))
    ordered_ids = [ids[index] for index in rng.permutation(len(ids))]
    rows: list[dict[str, object]] = []
    for blind_id in ordered_ids:
        for domain in DOMAIN_FEATURES:
            rows.append(
                {
                    "reviewer_id": str(reviewer_id),
                    "blind_id": blind_id,
                    "domain": domain,
                    "domain_label": pd.NA,
                    "count_bin": pd.NA,
                    "evidence_quote": pd.NA,
                    "line_number": pd.NA,
                    "polarity": pd.NA,
                    "uncertainty_reason": pd.NA,
                }
            )
    return pd.DataFrame(rows, columns=list(REVIEW_COLUMNS))


def validate_reviewer_form(
    form: pd.DataFrame,
    *,
    expected_blind_ids: Sequence[str],
    require_completed: bool,
) -> None:
    """Validate a blank or completed reviewer form without using owner data."""

    _require_columns(form, REVIEW_COLUMNS, name="reviewer form")
    forbidden = FORBIDDEN_REVIEW_COLUMNS.intersection(form.columns)
    if forbidden:
        raise ValueError(f"reviewer form exposes forbidden fields: {sorted(forbidden)}")
    if form[["blind_id", "domain"]].duplicated().any():
        raise ValueError("reviewer form contains duplicate blind_id/domain rows")
    if set(form["blind_id"].astype(str)) != set(map(str, expected_blind_ids)):
        raise ValueError("reviewer form does not contain exactly the expected blind IDs")
    if set(form["domain"].astype(str)) != set(DOMAIN_FEATURES):
        raise ValueError("reviewer form does not contain all ten frozen domains")
    if len(form) != len(expected_blind_ids) * len(DOMAIN_FEATURES):
        raise ValueError("reviewer form has an unexpected row count")
    if not require_completed:
        return

    labels = form["domain_label"].fillna("").astype(str).str.strip()
    if (~labels.isin(DOMAIN_LABELS)).any():
        raise ValueError("completed form contains a domain_label outside 0/1/NS")
    counts = form["count_bin"].fillna("").astype(str).str.strip()
    if (~counts.isin(COUNT_BINS) & (labels != "NS")).any():
        raise ValueError("0/1 labels require count_bin 0/1/2/3+")
    polarity = form["polarity"].fillna("").astype(str).str.strip()
    if (polarity.eq("") | ~polarity.isin(POLARITIES)).any():
        raise ValueError("completed form contains a missing or invalid polarity")
    quotes = form["evidence_quote"].fillna("").astype(str).str.strip()
    reasons = form["uncertainty_reason"].fillna("").astype(str).str.strip()
    line_numbers = form["line_number"].fillna("").astype(str).str.strip()
    if ((labels == "1") & quotes.eq("")).any():
        raise ValueError("domain_label=1 requires an exact evidence quote")
    if ((labels == "1") & ~line_numbers.map(lambda value: bool(_LINE_NUMBER_RE.match(value)))).any():
        raise ValueError("domain_label=1 requires one or more numeric line numbers")
    if ((labels == "NS") & reasons.eq("")).any():
        raise ValueError("domain_label=NS requires uncertainty_reason")


def _merged_reviewers(reviewer_a: pd.DataFrame, reviewer_b: pd.DataFrame) -> pd.DataFrame:
    _require_columns(reviewer_a, REVIEW_COLUMNS, name="reviewer A")
    _require_columns(reviewer_b, REVIEW_COLUMNS, name="reviewer B")
    left = reviewer_a.rename(columns={column: f"reviewer_A_{column}" for column in REVIEW_COLUMNS})
    right = reviewer_b.rename(columns={column: f"reviewer_B_{column}" for column in REVIEW_COLUMNS})
    merged = left.merge(
        right,
        left_on=["reviewer_A_blind_id", "reviewer_A_domain"],
        right_on=["reviewer_B_blind_id", "reviewer_B_domain"],
        how="outer",
        validate="one_to_one",
    )
    if merged[["reviewer_A_blind_id", "reviewer_B_blind_id"]].isna().any().any():
        raise ValueError("reviewer forms do not cover the same cells")
    return merged


def build_disagreement_sheet(
    reviewer_a: pd.DataFrame,
    reviewer_b: pd.DataFrame,
    cases: pd.DataFrame,
) -> pd.DataFrame:
    """Return only cells with different first-pass labels, before consensus."""

    merged = _merged_reviewers(reviewer_a, reviewer_b)
    merged["blind_id"] = merged["reviewer_A_blind_id"].astype(str)
    merged["domain"] = merged["reviewer_A_domain"].astype(str)
    labels_a = merged["reviewer_A_domain_label"].fillna("").astype(str)
    labels_b = merged["reviewer_B_domain_label"].fillna("").astype(str)
    disagreements = merged.loc[labels_a.ne(labels_b)].copy()
    case_text = cases[["blind_id", "participant_text"]].copy()
    disagreements = disagreements.merge(case_text, on="blind_id", how="left", validate="many_to_one")
    columns = [
        "blind_id",
        "domain",
        "participant_text",
        "reviewer_A_domain_label",
        "reviewer_A_count_bin",
        "reviewer_A_evidence_quote",
        "reviewer_A_line_number",
        "reviewer_A_polarity",
        "reviewer_A_uncertainty_reason",
        "reviewer_B_domain_label",
        "reviewer_B_count_bin",
        "reviewer_B_evidence_quote",
        "reviewer_B_line_number",
        "reviewer_B_polarity",
        "reviewer_B_uncertainty_reason",
    ]
    return disagreements[columns].sort_values(["blind_id", "domain"], kind="stable").reset_index(drop=True)


def _binary_agreement(a: pd.Series, b: pd.Series) -> tuple[float, float, float, int]:
    valid = a.isin(("0", "1")) & b.isin(("0", "1"))
    if not valid.any():
        return float("nan"), float("nan"), float("nan"), 0
    aa = a[valid]
    bb = b[valid]
    exact = float((aa == bb).mean())
    kappa = _safe_kappa(aa, bb, labels=["0", "1"])
    table = pd.crosstab(aa, bb).reindex(index=["0", "1"], columns=["0", "1"], fill_value=0)
    tn, fp = int(table.loc["0", "0"]), int(table.loc["0", "1"])
    fn, tp = int(table.loc["1", "0"]), int(table.loc["1", "1"])
    positive_den = 2 * tp + fn + fp
    negative_den = 2 * tn + fn + fp
    positive = float(2 * tp / positive_den) if positive_den else float("nan")
    negative = float(2 * tn / negative_den) if negative_den else float("nan")
    return kappa, positive, negative, int(valid.sum())


def _count_metrics(a: pd.Series, b: pd.Series) -> tuple[float, float, float, int]:
    valid = a.isin(COUNT_BINS) & b.isin(COUNT_BINS)
    if not valid.any():
        return float("nan"), float("nan"), float("nan"), 0
    aa = a[valid].map({value: index for index, value in enumerate(COUNT_BINS)})
    bb = b[valid].map({value: index for index, value in enumerate(COUNT_BINS)})
    weighted = _safe_kappa(aa, bb, labels=[0, 1, 2, 3], weights="linear")
    exact = float((aa == bb).mean())
    within_one = float((np.abs(aa.to_numpy() - bb.to_numpy()) <= 1).mean())
    return weighted, exact, within_one, int(valid.sum())


def _safe_kappa(a: pd.Series, b: pd.Series, *, labels: Sequence[object], weights: str | None = None) -> float:
    """Return NaN for a mathematically undefined one-category table."""

    if len(set(a.tolist()) | set(b.tolist())) < 2:
        return float("nan")
    return float(cohen_kappa_score(a, b, labels=list(labels), weights=weights))


def compute_interrater_metrics(reviewer_a: pd.DataFrame, reviewer_b: pd.DataFrame) -> pd.DataFrame:
    """Compute pre-consensus agreement while preserving NS as a third class."""

    merged = _merged_reviewers(reviewer_a, reviewer_b)
    rows: list[dict[str, object]] = []
    scopes: list[tuple[str, pd.DataFrame]] = [("overall", merged)]
    for domain, frame in merged.groupby("reviewer_A_domain", sort=False):
        scopes.append((str(domain), frame))
    for scope, frame in scopes:
        a = frame["reviewer_A_domain_label"].fillna("").astype(str).str.strip()
        b = frame["reviewer_B_domain_label"].fillna("").astype(str).str.strip()
        valid_three = a.isin(DOMAIN_LABELS) & b.isin(DOMAIN_LABELS)
        if valid_three.any():
            aa, bb = a[valid_three], b[valid_three]
            agreement = float((aa == bb).mean())
            kappa_three = _safe_kappa(aa, bb, labels=list(DOMAIN_LABELS))
            n_three = int(valid_three.sum())
        else:
            agreement, kappa_three, n_three = float("nan"), float("nan"), 0
        binary_kappa, positive, negative, n_binary = _binary_agreement(a, b)
        count_a = frame["reviewer_A_count_bin"].fillna("").astype(str).str.strip()
        count_b = frame["reviewer_B_count_bin"].fillna("").astype(str).str.strip()
        weighted_count, exact_count, within_one, n_count = _count_metrics(count_a, count_b)
        rows.append(
            {
                "scope": scope,
                "n_units": int(len(frame)),
                "three_category_n": n_three,
                "three_category_agreement": agreement,
                "three_category_cohen_kappa": kappa_three,
                "reviewer_A_NS_rate": float((a == "NS").mean()),
                "reviewer_B_NS_rate": float((b == "NS").mean()),
                "binary_complete_n": n_binary,
                "binary_cohen_kappa": binary_kappa,
                "positive_agreement": positive,
                "negative_agreement": negative,
                "count_complete_n": n_count,
                "count_weighted_cohen_kappa": weighted_count,
                "count_exact_agreement": exact_count,
                "count_within_one_bin": within_one,
            }
        )
    return pd.DataFrame(rows)


def build_domain_confusion_tables(reviewer_a: pd.DataFrame, reviewer_b: pd.DataFrame) -> pd.DataFrame:
    """Return raw three-category and count-bin contingency cells by domain."""

    merged = _merged_reviewers(reviewer_a, reviewer_b)
    rows: list[dict[str, object]] = []
    for domain, frame in merged.groupby("reviewer_A_domain", sort=False):
        a = frame["reviewer_A_domain_label"].fillna("").astype(str).str.strip()
        b = frame["reviewer_B_domain_label"].fillna("").astype(str).str.strip()
        for label_a in DOMAIN_LABELS:
            for label_b in DOMAIN_LABELS:
                rows.append(
                    {
                        "domain": str(domain),
                        "measure": "domain_label",
                        "reviewer_A_value": label_a,
                        "reviewer_B_value": label_b,
                        "n": int(((a == label_a) & (b == label_b)).sum()),
                    }
                )
        count_a = frame["reviewer_A_count_bin"].fillna("").astype(str).str.strip()
        count_b = frame["reviewer_B_count_bin"].fillna("").astype(str).str.strip()
        for value_a in COUNT_BINS:
            for value_b in COUNT_BINS:
                rows.append(
                    {
                        "domain": str(domain),
                        "measure": "count_bin",
                        "reviewer_A_value": value_a,
                        "reviewer_B_value": value_b,
                        "n": int(((count_a == value_a) & (count_b == value_b)).sum()),
                    }
                )
    return pd.DataFrame(rows)


def validate_consensus_labels(
    consensus: pd.DataFrame,
    reviewer_a: pd.DataFrame,
    reviewer_b: pd.DataFrame,
) -> None:
    """Ensure consensus retains both raw judgments and has an explicit reason."""

    required = {
        "blind_id",
        "domain",
        "reviewer_A_label",
        "reviewer_B_label",
        "consensus_label",
        "adjudication_reason",
    }
    _require_columns(consensus, sorted(required), name="consensus labels")
    expected = _merged_reviewers(reviewer_a, reviewer_b)
    expected = expected[["reviewer_A_blind_id", "reviewer_A_domain", "reviewer_A_domain_label", "reviewer_B_domain_label"]].rename(
        columns={
            "reviewer_A_blind_id": "blind_id",
            "reviewer_A_domain": "domain",
            "reviewer_A_domain_label": "reviewer_A_label",
            "reviewer_B_domain_label": "reviewer_B_label",
        }
    )
    left = expected.sort_values(["blind_id", "domain"]).reset_index(drop=True)
    right = consensus[list(required)].sort_values(["blind_id", "domain"]).reset_index(drop=True)
    if not left[["blind_id", "domain", "reviewer_A_label", "reviewer_B_label"]].equals(
        right[["blind_id", "domain", "reviewer_A_label", "reviewer_B_label"]]
    ):
        raise ValueError("consensus labels do not preserve the two raw reviewer judgments")
    labels = right["consensus_label"].fillna("").astype(str).str.strip()
    if (~labels.isin(DOMAIN_LABELS)).any():
        raise ValueError("consensus_label must be 0, 1, or NS")
    reasons = right["adjudication_reason"].fillna("").astype(str).str.strip()
    if reasons.eq("").any():
        raise ValueError("every consensus cell requires an adjudication_reason")


def build_consensus_template(reviewer_a: pd.DataFrame, reviewer_b: pd.DataFrame) -> pd.DataFrame:
    """Build an unfilled consensus table that retains both raw labels."""

    merged = _merged_reviewers(reviewer_a, reviewer_b)
    return (
        merged[
            [
                "reviewer_A_blind_id",
                "reviewer_A_domain",
                "reviewer_A_domain_label",
                "reviewer_B_domain_label",
            ]
        ]
        .rename(
            columns={
                "reviewer_A_blind_id": "blind_id",
                "reviewer_A_domain": "domain",
                "reviewer_A_domain_label": "reviewer_A_label",
                "reviewer_B_domain_label": "reviewer_B_label",
            }
        )
        .assign(consensus_label=pd.NA, adjudication_reason=pd.NA)
        [["blind_id", "domain", "reviewer_A_label", "reviewer_B_label", "consensus_label", "adjudication_reason"]]
        .sort_values(["blind_id", "domain"], kind="stable")
        .reset_index(drop=True)
    )


def _classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    if len(y_true) == 0:
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan"), "specificity": float("nan"), "accuracy": float("nan")}
    table = pd.crosstab(y_true, y_pred).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    tn, fp = int(table.loc[0, 0]), int(table.loc[0, 1])
    fn, tp = int(table.loc[1, 0]), int(table.loc[1, 1])
    return {
        "precision": float(tp / (tp + fp)) if tp + fp else float("nan"),
        "recall": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "accuracy": float((tp + tn) / len(y_true)),
    }


def compute_dp_vs_consensus(
    consensus: pd.DataFrame,
    scoring_reference: pd.DataFrame,
    *,
    ns_policy: str = "exclude",
) -> pd.DataFrame:
    """Compare frozen D-P presence with consensus, keeping NS policy explicit."""

    if ns_policy not in {"exclude", "as_0", "as_1"}:
        raise ValueError("ns_policy must be exclude, as_0, or as_1")
    _require_columns(scoring_reference, ("blind_id", *DOMAIN_FEATURES), name="scoring reference")
    _require_columns(consensus, ("blind_id", "domain", "consensus_label"), name="consensus labels")
    long_rows: list[dict[str, object]] = []
    for row in scoring_reference.itertuples(index=False):
        for domain in DOMAIN_FEATURES:
            long_rows.append({"blind_id": str(row.blind_id), "domain": domain, "dp_binary": int(float(getattr(row, domain)) > 0)})
    dp = pd.DataFrame(long_rows)
    frame = dp.merge(consensus[["blind_id", "domain", "consensus_label"]], on=["blind_id", "domain"], how="left", validate="one_to_one")
    labels = frame["consensus_label"].fillna("").astype(str).str.strip()
    frame["ns"] = labels.eq("NS")
    frame["human_binary"] = labels.map({"0": 0, "1": 1})
    if ns_policy == "as_0":
        frame.loc[frame["ns"], "human_binary"] = 0
    elif ns_policy == "as_1":
        frame.loc[frame["ns"], "human_binary"] = 1
    rows: list[dict[str, object]] = []
    for scope, subset in [("overall", frame)] + [(str(domain), frame.loc[frame["domain"].eq(domain)]) for domain in DOMAIN_FEATURES]:
        valid = subset["human_binary"].notna()
        y_true = subset.loc[valid, "human_binary"].astype(int)
        y_pred = subset.loc[valid, "dp_binary"].astype(int)
        metrics = _classification_metrics(y_true, y_pred)
        rows.append({"scope": scope, "ns_policy": ns_policy, "n_total": int(len(subset)), "n_valid": int(valid.sum()), "ns_n": int(subset["ns"].sum()), **metrics})
    domain_rows = pd.DataFrame([row for row in rows if row["scope"] != "overall"])
    if not domain_rows.empty:
        macro = {metric: float(domain_rows[metric].mean()) for metric in ("precision", "recall", "f1", "specificity", "accuracy")}
        rows.append({"scope": "macro", "ns_policy": ns_policy, "n_total": int(len(frame)), "n_valid": int(frame["human_binary"].notna().sum()), "ns_n": int(frame["ns"].sum()), **macro})
    return pd.DataFrame(rows)


def compute_evidence_validity(review: pd.DataFrame) -> pd.DataFrame:
    """Summarize the second-round evidence-level review labels."""

    _require_columns(review, ("blind_id", "domain", "evidence_validity"), name="evidence review")
    values = review["evidence_validity"].fillna("").astype(str).str.strip()
    if (~values.isin(EVIDENCE_VALIDITY)).any():
        raise ValueError("evidence_validity contains an unknown category")
    counts = values.value_counts().reindex(EVIDENCE_VALIDITY, fill_value=0)
    total = int(len(values))
    rows = [{"scope": "overall", "evidence_validity": value, "n": int(count), "proportion": float(count / total) if total else float("nan")} for value, count in counts.items()]
    for domain, frame in review.assign(evidence_validity=values).groupby("domain", sort=False):
        domain_counts = frame["evidence_validity"].value_counts().reindex(EVIDENCE_VALIDITY, fill_value=0)
        n = int(len(frame))
        rows.extend({"scope": str(domain), "evidence_validity": value, "n": int(count), "proportion": float(count / n) if n else float("nan")} for value, count in domain_counts.items())
    return pd.DataFrame(rows)
