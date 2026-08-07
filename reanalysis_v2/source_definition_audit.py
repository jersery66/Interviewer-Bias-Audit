"""Audit the provenance of symptom-domain controls before conditional modeling.

The current frozen domain-count table is derived from reviewed C5 spans.  This
module makes that boundary explicit and refuses to treat an unproven table as
the full-interview variant (D-All).  It deliberately writes only aggregate
counts and hashes; quote text is never included in the audit record.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_WHITESPACE = re.compile(r"\s+")


def _hash_variants(path: str | Path) -> dict[str, str]:
    raw = Path(path).read_bytes()
    return {
        "raw": hashlib.sha256(raw).hexdigest(),
        "lf": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),
    }


def _normalize(text: object) -> str:
    return _WHITESPACE.sub(" ", str(text or "")).strip().casefold()


def _read_text_table(path: str | Path, name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"participant_id", "text"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
    if frame["participant_id"].duplicated().any():
        raise ValueError(f"{name} participant_id must be unique")
    result = frame[["participant_id", "text"]].copy()
    result["participant_id"] = pd.to_numeric(result["participant_id"], errors="raise").astype(int)
    result["text"] = result["text"].fillna("").astype(str)
    return result


def _check_domain_counts(
    spans: pd.DataFrame,
    domain_counts: pd.DataFrame,
) -> tuple[str, int, list[str]]:
    if "participant_id" not in domain_counts.columns:
        raise ValueError("domain-count input is missing participant_id")
    if domain_counts["participant_id"].duplicated().any():
        raise ValueError("domain-count participant_id must be unique")
    domain_columns = [column for column in domain_counts.columns if column != "participant_id"]
    if not domain_columns:
        raise ValueError("domain-count input has no domain columns")
    if "domain" not in spans.columns:
        raise ValueError("reviewed spans are missing domain")

    ids = set(pd.to_numeric(domain_counts["participant_id"], errors="raise").astype(int))
    span_ids = set(pd.to_numeric(spans["participant_id"], errors="raise").astype(int))
    if not span_ids.issubset(ids):
        raise ValueError("reviewed spans contain participant IDs absent from domain-count input")

    observed = (
        spans.assign(participant_id=pd.to_numeric(spans["participant_id"], errors="raise").astype(int))
        .groupby(["participant_id", "domain"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=sorted(ids), columns=domain_columns, fill_value=0)
        .fillna(0)
        .astype(int)
    )
    counts = domain_counts.copy()
    counts["participant_id"] = pd.to_numeric(counts["participant_id"], errors="raise").astype(int)
    counts = counts.set_index("participant_id").reindex(sorted(ids))
    numeric = counts[domain_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("domain-count input contains missing or non-finite values")
    numeric = numeric.astype(int)
    if not np.array_equal(observed.to_numpy(), numeric.to_numpy()):
        return "mismatch", int(np.count_nonzero(observed.to_numpy() != numeric.to_numpy())), domain_columns
    return "matched", 0, domain_columns


def _audit_d_all(
    d_all_path: str | Path | None,
    provenance_path: str | Path | None,
    participant_ids: set[int],
    domain_columns: list[str],
) -> dict[str, Any]:
    if d_all_path is None or not Path(d_all_path).exists():
        return {
            "status": "missing",
            "ready_for_modeling": False,
            "reason": "no frozen full_transcript D-All input was supplied",
        }
    if provenance_path is None or not Path(provenance_path).exists():
        return {
            "status": "rejected",
            "ready_for_modeling": False,
            "reason": "D-All requires a provenance manifest declaring full_transcript scope",
            "path_sha256": _hash_variants(d_all_path),
        }
    provenance = json.loads(Path(provenance_path).read_text(encoding="utf-8"))
    scope = str(provenance.get("source_scope", "")).strip().casefold()
    if scope not in {"full_transcript", "full_interview", "all_interview"}:
        return {
            "status": "rejected",
            "ready_for_modeling": False,
            "reason": f"source_scope={scope or '[missing]'} is not full_transcript",
            "path_sha256": _hash_variants(d_all_path),
            "provenance_sha256": _hash_variants(provenance_path),
        }
    frame = pd.read_csv(d_all_path)
    required = {"participant_id", *domain_columns}
    missing = required.difference(frame.columns)
    if missing:
        return {
            "status": "rejected",
            "ready_for_modeling": False,
            "reason": f"D-All schema missing columns: {sorted(missing)}",
            "path_sha256": _hash_variants(d_all_path),
            "provenance_sha256": _hash_variants(provenance_path),
        }
    ids = set(pd.to_numeric(frame["participant_id"], errors="raise").astype(int))
    if ids != participant_ids or frame["participant_id"].duplicated().any():
        return {
            "status": "rejected",
            "ready_for_modeling": False,
            "reason": "D-All participant IDs do not exactly match the frozen cohort",
            "path_sha256": _hash_variants(d_all_path),
            "provenance_sha256": _hash_variants(provenance_path),
        }
    numeric = frame[domain_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        return {
            "status": "rejected",
            "ready_for_modeling": False,
            "reason": "D-All contains missing or non-finite domain values",
            "path_sha256": _hash_variants(d_all_path),
            "provenance_sha256": _hash_variants(provenance_path),
        }
    return {
        "status": "available",
        "ready_for_modeling": True,
        "source_scope": scope,
        "path_sha256": _hash_variants(d_all_path),
        "provenance_sha256": _hash_variants(provenance_path),
        "n": int(len(frame)),
        "domain_columns": domain_columns,
    }


def audit_domain_sources(
    *,
    participant_text_path: str | Path,
    interviewer_text_path: str | Path,
    full_text_path: str | Path,
    reviewed_spans_path: str | Path,
    domain_count_path: str | Path,
    d_all_path: str | Path | None = None,
    d_all_provenance_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return a provenance audit for D-P and a gated status for D-All."""

    participant = _read_text_table(participant_text_path, "participant text")
    interviewer = _read_text_table(interviewer_text_path, "interviewer text")
    full = _read_text_table(full_text_path, "full text")
    if set(participant.participant_id) != set(interviewer.participant_id) or set(participant.participant_id) != set(full.participant_id):
        raise ValueError("text tables do not contain the same participant IDs")

    spans = pd.read_csv(reviewed_spans_path)
    required_span_columns = {"participant_id", "domain", "exact_quote"}
    missing = required_span_columns.difference(spans.columns)
    if missing:
        raise ValueError(f"reviewed spans are missing required columns: {sorted(missing)}")
    domain_counts = pd.read_csv(domain_count_path)
    reconciliation, mismatch_n, domain_columns = _check_domain_counts(spans, domain_counts)

    participant_text = dict(zip(participant.participant_id, participant.text.map(_normalize)))
    interviewer_text = dict(zip(interviewer.participant_id, interviewer.text.map(_normalize)))
    full_text = dict(zip(full.participant_id, full.text.map(_normalize)))
    participant_matches = interviewer_matches = full_matches = 0
    interviewer_only = full_only = unmatched = 0
    for row in spans.itertuples(index=False):
        participant_id = int(row.participant_id)
        quote = _normalize(row.exact_quote)
        p_match = bool(quote) and quote in participant_text[participant_id]
        i_match = bool(quote) and quote in interviewer_text[participant_id]
        f_match = bool(quote) and quote in full_text[participant_id]
        participant_matches += int(p_match)
        interviewer_matches += int(i_match)
        full_matches += int(f_match)
        interviewer_only += int(i_match and not p_match)
        full_only += int(f_match and not p_match and not i_match)
        unmatched += int(not p_match and not i_match and not f_match)

    span_n = int(len(spans))
    d_p_status = "confirmed" if (
        span_n > 0
        and participant_matches == span_n
        and interviewer_only == 0
        and reconciliation == "matched"
    ) else "not_confirmed"
    participant_ids = set(participant.participant_id.astype(int))
    return {
        "d_p": {
            "status": d_p_status,
            "ready_for_modeling": d_p_status == "confirmed",
            "definition": "domain counts derived from reviewed C5 spans matched to participant language",
            "n_participants": int(len(participant_ids)),
            "span_n": span_n,
            "matching_quote_n": int(participant_matches),
            "interviewer_match_n": int(interviewer_matches),
            "interviewer_only_quote_n": int(interviewer_only),
            "full_match_n": int(full_matches),
            "full_only_quote_n": int(full_only),
            "unmatched_quote_n": int(unmatched),
            "count_reconciliation": reconciliation,
            "count_mismatch_cell_n": mismatch_n,
            "domain_columns": domain_columns,
            "spans_sha256": _hash_variants(reviewed_spans_path),
            "domain_count_sha256": _hash_variants(domain_count_path),
            "participant_text_sha256": _hash_variants(participant_text_path),
            "interviewer_text_sha256": _hash_variants(interviewer_text_path),
            "full_text_sha256": _hash_variants(full_text_path),
        },
        "d_all": _audit_d_all(
            d_all_path,
            d_all_provenance_path,
            participant_ids,
            domain_columns,
        ),
    }


def write_domain_source_audit(result: dict[str, Any], output_path: str | Path) -> None:
    """Write an aggregate JSON audit without quote-bearing content."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
