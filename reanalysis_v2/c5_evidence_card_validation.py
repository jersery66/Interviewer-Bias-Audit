"""Validate frozen C5 evidence-card inputs and build safe quote context."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


C5_CANONICAL_RELATIVE = Path(
    "processed_research/latest_usable_data_official142_20260702/"
    "04_c5_evidence_sources/"
    "c5_quotes_only_v3_gpt55_reviewed_spans_official142.csv"
)
C5_CANONICAL_SHA256 = (
    "794d7b43ebd95d57ac4a90a4217953a9bea27e98ff91b13c31b0a06f9aa48c7e"
)
PARTICIPANT_TEXT_RELATIVE = Path(
    "processed_research/participant_for_evidence_extraction.jsonl"
)
PARTICIPANT_TEXT_SHA256 = (
    "f64eefd1b9560a845a4e838b5260016fbad840a2c957ef048c5feea965e4b549"
)

DOMAIN_NAMES = (
    "anhedonia_interest",
    "appetite_weight",
    "concentration_psychomotor",
    "depressed_mood",
    "functioning_impairment",
    "mental_health_history",
    "protective_or_absent_symptom",
    "self_worth_guilt",
    "sleep_fatigue_energy",
    "suicide_self_harm",
)
REQUIRED_SPAN_COLUMNS = (
    "participant_id",
    "domain",
    "polarity",
    "exact_quote",
    "source_order",
)

_HASH_CHUNK_SIZE = 1024 * 1024
_EMAIL_RE = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE
)
_PHONE_RE = re.compile(r"(?<![\w@])\+?\d(?:[\s().-]*\d){6,}(?!\w)")
_LONG_NUMBER_RE = re.compile(r"(?<!\d)\d{3,}(?!\d)")
_NAME_WORD = r"[^\W\d_]+(?:['’\-][^\W\d_]+)*"
_NAME_STOP_WORDS = (
    r"and|but|or|because|so|then|who|which|that|when|where|while|"
    r"although|though|if|from|with|at|in|on|for|to|of|i|we|you|he|she|"
    r"they|it|live|lives|work|works|am|is|are|was|were"
)
_NAME_TOKEN = rf"(?!(?:{_NAME_STOP_WORDS})\b){_NAME_WORD}"
_SELF_INTRODUCED_NAME_RE = re.compile(
    rf"(?P<prefix>\b(?:my\s+name\s+is|i\s+am|i['’]m))"
    rf"(?P<gap>\s+)"
    rf"(?P<name>{_NAME_TOKEN}(?:\s+{_NAME_TOKEN}){{0,3}})",
    re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of the physical bytes at *path*."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(path: Path, expected_sha256: str) -> None:
    actual_sha256 = sha256_file(path)
    if actual_sha256.casefold() != expected_sha256.strip().casefold():
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {expected_sha256}, "
            f"found {actual_sha256}"
        )


def normalize_space(value: Any) -> str:
    """Convert a scalar to text and collapse all runs of whitespace."""

    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return re.sub(r"\s+", " ", str(value)).strip()


def load_and_validate_spans(
    source: Path | pd.DataFrame, *, expected_sha256: str | None
) -> pd.DataFrame:
    """Load a frozen span table or validate a copied in-memory table."""

    if isinstance(source, Path):
        if expected_sha256 is None:
            raise ValueError("expected_sha256 is required for a Path source")
        _validate_sha256(source, expected_sha256)
        spans = pd.read_csv(source)
    elif isinstance(source, pd.DataFrame):
        if expected_sha256 is not None:
            raise ValueError(
                "SHA-256 cannot be checked for a DataFrame; expected_sha256 must be None"
            )
        spans = source.copy(deep=True)
    else:
        raise TypeError("source must be a pathlib.Path or pandas.DataFrame")

    missing_columns = [
        column for column in REQUIRED_SPAN_COLUMNS if column not in spans.columns
    ]
    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing_columns)
        )

    domain_values = spans["domain"]
    unknown_mask = ~domain_values.isin(DOMAIN_NAMES)
    if unknown_mask.any():
        unknown_domains = sorted(
            {
                normalize_space(value) or "<blank>"
                for value in domain_values.loc[unknown_mask]
            }
        )
        raise ValueError(
            "Found unknown domain values: " + ", ".join(unknown_domains)
        )

    for column in ("participant_id", "source_order"):
        raw_values = spans[column]
        contains_bool = raw_values.map(
            lambda value: isinstance(value, (bool, np.bool_))
        ).any()
        try:
            numeric_values = pd.to_numeric(raw_values, errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{column} must contain only finite integer values"
            ) from exc

        numeric_array = numeric_values.to_numpy()
        finite = np.isfinite(numeric_array).all()
        integer_valued = finite and np.equal(
            np.remainder(numeric_array, 1), 0
        ).all()
        if contains_bool or not integer_valued:
            raise ValueError(f"{column} must contain only finite integer values")

        integer_values = [int(value) for value in numeric_array]
        int64_bounds = np.iinfo(np.int64)
        if any(
            value < int64_bounds.min or value > int64_bounds.max
            for value in integer_values
        ):
            raise ValueError(f"{column} values must fit in int64")
        spans[column] = pd.Series(
            integer_values, index=spans.index, dtype="int64"
        )

    blank_quote_mask = spans["exact_quote"].map(normalize_space).eq("")
    if blank_quote_mask.any():
        indices = ", ".join(str(index) for index in spans.index[blank_quote_mask])
        raise ValueError(f"exact_quote contains blank values at row indices: {indices}")

    return spans


def _parse_participant_id(value: Any, *, line_number: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"participant_id on JSONL line {line_number} is not an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise ValueError(f"participant_id on JSONL line {line_number} is not an integer")
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value)
    raise ValueError(f"participant_id on JSONL line {line_number} is not an integer")


def load_participant_texts(
    path: Path, *, expected_sha256: str = PARTICIPANT_TEXT_SHA256
) -> dict[int, str]:
    """Load the hash-locked UTF-8 participant JSONL keyed by integer ID."""

    _validate_sha256(path, expected_sha256)
    participant_texts: dict[int, str] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on JSONL line {line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL line {line_number} must contain an object")

            missing_fields = [
                field
                for field in ("participant_id", "text")
                if field not in record or record[field] is None
            ]
            if missing_fields:
                raise ValueError(
                    f"JSONL line {line_number} is missing "
                    + ", ".join(missing_fields)
                )

            participant_id = _parse_participant_id(
                record["participant_id"], line_number=line_number
            )
            text = record["text"]
            if not isinstance(text, str) or not normalize_space(text):
                raise ValueError(f"text on JSONL line {line_number} must be nonblank text")
            if participant_id in participant_texts:
                raise ValueError(f"duplicate participant_id: {participant_id}")
            participant_texts[participant_id] = text

    return participant_texts


def _redaction_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    patterns = (
        (_EMAIL_RE, "[REDACTED_EMAIL]", 0),
        (_PHONE_RE, "[REDACTED_PHONE]", 0),
        (_SELF_INTRODUCED_NAME_RE, "[REDACTED_NAME]", "name"),
        (_LONG_NUMBER_RE, "[REDACTED_NUMBER]", 0),
    )

    for pattern, replacement, group in patterns:
        for match in pattern.finditer(text):
            start, end = match.span(group)
            if any(
                start < prior_end and end > prior_start
                for prior_start, prior_end, _ in spans
            ):
                continue
            spans.append((start, end, replacement))

    return sorted(spans)


def _apply_redaction_spans(text: str, spans: list[tuple[int, int, str]]) -> str:
    pieces: list[str] = []
    cursor = 0
    for start, end, replacement in spans:
        pieces.extend((text[cursor:start], replacement))
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def deidentify_text(value: Any) -> str:
    """Deterministically redact common direct identifiers from text."""

    text = normalize_space(value)
    return normalize_space(_apply_redaction_spans(text, _redaction_spans(text)))


def _deidentify_around_quote(
    text: str, quote_start: int, quote_end: int
) -> tuple[str, str, str]:
    segment_bounds = (
        (0, quote_start),
        (quote_start, quote_end),
        (quote_end, len(text)),
    )
    pieces: tuple[list[str], list[str], list[str]] = ([], [], [])

    def append_plain(start: int, end: int) -> None:
        for index, (segment_start, segment_end) in enumerate(segment_bounds):
            overlap_start = max(start, segment_start)
            overlap_end = min(end, segment_end)
            if overlap_start < overlap_end:
                pieces[index].append(text[overlap_start:overlap_end])

    cursor = 0
    for start, end, replacement in _redaction_spans(text):
        append_plain(cursor, start)
        if start < quote_end and end > quote_start:
            segment_index = 1
        elif end <= quote_start:
            segment_index = 0
        else:
            segment_index = 2
        pieces[segment_index].append(replacement)
        cursor = end
    append_plain(cursor, len(text))

    return (
        normalize_space("".join(pieces[0])),
        normalize_space("".join(pieces[1])),
        normalize_space("".join(pieces[2])),
    )


def _casefold_boundaries(value: str) -> tuple[str, dict[int, int]]:
    folded_parts: list[str] = []
    boundaries = {0: 0}
    folded_offset = 0
    for original_offset, character in enumerate(value, start=1):
        folded_character = character.casefold()
        folded_parts.append(folded_character)
        folded_offset += len(folded_character)
        boundaries[folded_offset] = original_offset
    return "".join(folded_parts), boundaries


def locate_quote_context(
    text: Any, quote: Any, *, flank_chars: int = 180
) -> tuple[str, str, str]:
    """Locate a normalized quote and return deidentified surrounding context."""

    if not isinstance(flank_chars, int) or isinstance(flank_chars, bool):
        raise TypeError("flank_chars must be an integer")
    if flank_chars < 0:
        raise ValueError("flank_chars must be nonnegative")

    normalized_text = normalize_space(text)
    normalized_quote = normalize_space(quote)
    if not normalized_quote:
        raise ValueError("quote is blank after whitespace normalization")

    folded_text, boundaries = _casefold_boundaries(normalized_text)
    folded_quote = normalized_quote.casefold()
    valid_matches: list[tuple[int, int]] = []
    search_start = 0
    while True:
        folded_start = folded_text.find(folded_quote, search_start)
        if folded_start < 0:
            break
        folded_end = folded_start + len(folded_quote)
        if folded_start in boundaries and folded_end in boundaries:
            valid_matches.append(
                (boundaries[folded_start], boundaries[folded_end])
            )
        search_start = folded_start + 1

    if not valid_matches:
        raise ValueError("quote not found after whitespace normalization and casefold")
    if len(valid_matches) > 1:
        raise ValueError(
            "ambiguous quote: multiple occurrences found after whitespace "
            "normalization and casefold"
        )

    start, end = valid_matches[0]
    before_full, quote_text, after_full = _deidentify_around_quote(
        normalized_text, start, end
    )

    if flank_chars == 0:
        return "", quote_text, ""
    return before_full[-flank_chars:], quote_text, after_full[:flank_chars]
