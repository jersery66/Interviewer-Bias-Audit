"""Validate frozen C5 evidence-card inputs and build safe quote context."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

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
_NAME_STOP_WORDS = r"and|but|from|with|who|because|so|then|when|where"
_SELF_INTRODUCED_NAME_RE = re.compile(
    rf"(?P<prefix>\b(?:my\s+name\s+is|i\s+am|i['’]m))"
    rf"(?P<gap>\s+)"
    rf"(?P<name>{_NAME_WORD}"
    rf"(?:\s+(?!(?:{_NAME_STOP_WORDS})\b){_NAME_WORD}){{0,2}})",
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
        try:
            numeric_values = pd.to_numeric(spans[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{column} must contain only numeric values") from exc
        if numeric_values.isna().any():
            raise ValueError(f"{column} must contain only numeric values")
        spans[column] = numeric_values

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


def _replace_self_introduced_name(match: re.Match[str]) -> str:
    prefix = match.group("prefix")
    name = match.group("name")
    normalized_prefix = normalize_space(prefix).replace("’", "'").casefold()
    if normalized_prefix != "my name is" and not name[0].isupper():
        return match.group(0)
    return f"{prefix} [REDACTED_NAME]"


def deidentify_text(value: Any) -> str:
    """Deterministically redact common direct identifiers from text."""

    text = normalize_space(value)
    text = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = _PHONE_RE.sub("[REDACTED_PHONE]", text)
    text = _SELF_INTRODUCED_NAME_RE.sub(_replace_self_introduced_name, text)
    text = _LONG_NUMBER_RE.sub("[REDACTED_NUMBER]", text)
    return normalize_space(text)


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
    folded_start = folded_text.find(folded_quote)
    while folded_start >= 0:
        folded_end = folded_start + len(folded_quote)
        if folded_start in boundaries and folded_end in boundaries:
            break
        folded_start = folded_text.find(folded_quote, folded_start + 1)
    else:
        raise ValueError("quote not found after whitespace normalization and casefold")

    start = boundaries[folded_start]
    end = boundaries[folded_start + len(folded_quote)]
    before_full = deidentify_text(normalized_text[:start])
    quote_text = deidentify_text(normalized_text[start:end])
    after_full = deidentify_text(normalized_text[end:])

    if flank_chars == 0:
        return "", quote_text, ""
    return before_full[-flank_chars:], quote_text, after_full[:flank_chars]
