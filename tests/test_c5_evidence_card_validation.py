from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pandas as pd
import pytest
from pandas.api.types import is_numeric_dtype


MODULE_NAME = "reanalysis_v2.c5_evidence_card_validation"

EXPECTED_DOMAINS = (
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

VALID_SPAN = {
    "participant_id": "101",
    "domain": "anhedonia_interest",
    "polarity": "present",
    "exact_quote": "I enjoy garden walks.",
    "source_order": "2",
}


def _module():
    return importlib.import_module(MODULE_NAME)


def _span_frame(**overrides: object) -> pd.DataFrame:
    row = {**VALID_SPAN, **overrides}
    return pd.DataFrame([row])


def _physical_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> str:
    payload = "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in records
    ).encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_locked_constants_match_the_frozen_inputs() -> None:
    c5 = _module()

    assert c5.C5_CANONICAL_RELATIVE == Path(
        "processed_research/latest_usable_data_official142_20260702/"
        "04_c5_evidence_sources/"
        "c5_quotes_only_v3_gpt55_reviewed_spans_official142.csv"
    )
    assert (
        c5.C5_CANONICAL_SHA256
        == "794d7b43ebd95d57ac4a90a4217953a9bea27e98ff91b13c31b0a06f9aa48c7e"
    )
    assert c5.PARTICIPANT_TEXT_RELATIVE == Path(
        "processed_research/participant_for_evidence_extraction.jsonl"
    )
    assert (
        c5.PARTICIPANT_TEXT_SHA256
        == "f64eefd1b9560a845a4e838b5260016fbad840a2c957ef048c5feea965e4b549"
    )
    assert tuple(c5.DOMAIN_NAMES) == EXPECTED_DOMAINS
    assert tuple(c5.REQUIRED_SPAN_COLUMNS) == (
        "participant_id",
        "domain",
        "polarity",
        "exact_quote",
        "source_order",
    )


def test_sha256_file_hashes_physical_bytes(tmp_path: Path) -> None:
    c5 = _module()
    source = tmp_path / "physical.bin"
    payload = b"first\r\nsecond\n\x00"
    source.write_bytes(payload)

    assert c5.sha256_file(source) == hashlib.sha256(payload).hexdigest()


def test_span_path_wrong_hash_fails_before_csv_validation(tmp_path: Path) -> None:
    c5 = _module()
    source = tmp_path / "spans.csv"
    source.write_text("not_the_required_columns\nvalue\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        c5.load_and_validate_spans(source, expected_sha256="0" * 64)


def test_span_validation_rejects_missing_required_columns() -> None:
    c5 = _module()
    frame = _span_frame().drop(columns=["polarity"])

    with pytest.raises(ValueError, match=r"required columns.*polarity"):
        c5.load_and_validate_spans(frame, expected_sha256=None)


def test_span_validation_rejects_unknown_domain() -> None:
    c5 = _module()

    with pytest.raises(ValueError, match=r"unknown domain.*not_a_locked_domain"):
        c5.load_and_validate_spans(
            _span_frame(domain="not_a_locked_domain"), expected_sha256=None
        )


def test_span_validation_converts_numeric_columns_and_copies_input() -> None:
    c5 = _module()
    frame = _span_frame(source_order=2.0)

    result = c5.load_and_validate_spans(frame, expected_sha256=None)

    assert result is not frame
    assert frame.loc[0, "participant_id"] == "101"
    assert frame.loc[0, "source_order"] == 2.0
    assert result.loc[0, "participant_id"] == 101
    assert result.loc[0, "source_order"] == 2
    assert is_numeric_dtype(result["participant_id"])
    assert is_numeric_dtype(result["source_order"])
    assert str(result["participant_id"].dtype) == "int64"
    assert str(result["source_order"].dtype) == "int64"


@pytest.mark.parametrize("column", ["participant_id", "source_order"])
def test_span_validation_rejects_non_numeric_values(column: str) -> None:
    c5 = _module()

    with pytest.raises(ValueError, match=column):
        c5.load_and_validate_spans(
            _span_frame(**{column: "not-numeric"}), expected_sha256=None
        )


@pytest.mark.parametrize("column", ["participant_id", "source_order"])
@pytest.mark.parametrize(
    "value",
    [101.5, float("inf"), float("nan"), pd.NA, True],
    ids=["fractional", "infinite", "nan", "pandas-na", "boolean"],
)
def test_span_validation_rejects_non_integer_numeric_values(
    column: str, value: object
) -> None:
    c5 = _module()

    with pytest.raises(ValueError, match=column):
        c5.load_and_validate_spans(
            _span_frame(**{column: value}), expected_sha256=None
        )


@pytest.mark.parametrize("quote", ["", " \t\n", None])
def test_span_validation_rejects_blank_quotes(quote: object) -> None:
    c5 = _module()

    with pytest.raises(ValueError, match="exact_quote"):
        c5.load_and_validate_spans(
            _span_frame(exact_quote=quote), expected_sha256=None
        )


def test_span_path_with_matching_hash_loads_and_validates(tmp_path: Path) -> None:
    c5 = _module()
    source = tmp_path / "spans.csv"
    _span_frame().to_csv(source, index=False)

    result = c5.load_and_validate_spans(
        source, expected_sha256=_physical_sha256(source)
    )

    assert result.loc[0, "participant_id"] == 101
    assert result.loc[0, "exact_quote"] == "I enjoy garden walks."


def test_participant_text_wrong_hash_fails_before_jsonl_parsing(
    tmp_path: Path,
) -> None:
    c5 = _module()
    source = tmp_path / "participants.jsonl"
    source.write_text("this is not JSON\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        c5.load_participant_texts(source, expected_sha256="0" * 64)


def test_participant_text_loader_rejects_duplicate_ids(tmp_path: Path) -> None:
    c5 = _module()
    source = tmp_path / "participants.jsonl"
    expected_hash = _write_jsonl(
        source,
        [
            {"participant_id": 101, "text": "First synthetic interview."},
            {"participant_id": 101, "text": "Second synthetic interview."},
        ],
    )

    with pytest.raises(ValueError, match=r"duplicate participant_id.*101"):
        c5.load_participant_texts(source, expected_sha256=expected_hash)


@pytest.mark.parametrize("missing_key", ["participant_id", "text"])
def test_participant_text_loader_rejects_missing_fields(
    tmp_path: Path, missing_key: str
) -> None:
    c5 = _module()
    source = tmp_path / "participants.jsonl"
    record: dict[str, object] = {
        "participant_id": 101,
        "text": "Synthetic interview text.",
    }
    del record[missing_key]
    expected_hash = _write_jsonl(source, [record])

    with pytest.raises(ValueError, match=missing_key):
        c5.load_participant_texts(source, expected_sha256=expected_hash)


def test_participant_text_loader_returns_integer_keyed_texts(tmp_path: Path) -> None:
    c5 = _module()
    source = tmp_path / "participants.jsonl"
    expected_hash = _write_jsonl(
        source,
        [
            {"participant_id": 101, "text": "Synthetic interview one."},
            {"participant_id": "102", "text": "Synthetic interview two."},
        ],
    )

    assert c5.load_participant_texts(source, expected_sha256=expected_hash) == {
        101: "Synthetic interview one.",
        102: "Synthetic interview two.",
    }


def test_normalize_space_collapses_all_whitespace() -> None:
    c5 = _module()

    assert c5.normalize_space("  alpha\n\tbeta   gamma  ") == "alpha beta gamma"
    assert c5.normalize_space(None) == ""


def test_quote_context_matches_after_whitespace_normalization_and_casefold() -> None:
    c5 = _module()
    text = "Opening words.\nI   ENJOY\tgarden walks every day. Closing words."

    assert c5.locate_quote_context(text, "i enjoy GARDEN walks") == (
        "Opening words.",
        "I ENJOY garden walks",
        "every day. Closing words.",
    )


def test_quote_context_accepts_one_unique_casefold_occurrence() -> None:
    c5 = _module()

    assert c5.locate_quote_context("Before TARGET after.", "target") == (
        "Before",
        "TARGET",
        "after.",
    )


def test_quote_context_rejects_multiple_casefold_occurrences() -> None:
    c5 = _module()

    with pytest.raises(ValueError, match=r"ambiguous|multiple occurrence"):
        c5.locate_quote_context("First TARGET then target.", "target")


def test_quote_context_honors_flank_character_limit() -> None:
    c5 = _module()
    text = "abcdefghij TARGET klmnopqrst"

    assert c5.locate_quote_context(text, "target", flank_chars=4) == (
        "ghij",
        "TARGET",
        "klmn",
    )


def test_quote_context_rejects_absent_quote() -> None:
    c5 = _module()

    with pytest.raises(ValueError, match="not found"):
        c5.locate_quote_context("Synthetic source text.", "missing phrase")


@pytest.mark.parametrize("quote", ["", " \n\t", None])
def test_quote_context_rejects_blank_quote(quote: object) -> None:
    c5 = _module()

    with pytest.raises(ValueError, match="blank"):
        c5.locate_quote_context("Synthetic source text.", quote)


def test_deidentify_text_is_deterministic() -> None:
    c5 = _module()
    source = (
        "  My name is Alice Smith. Email ALICE@example.com, "
        "call +1 (555) 123-4567. Case 12345.  "
    )
    expected = (
        "My name is [REDACTED_NAME]. Email [REDACTED_EMAIL], "
        "call [REDACTED_PHONE]. Case [REDACTED_NUMBER]."
    )

    assert c5.deidentify_text(source) == expected
    assert c5.deidentify_text(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "Before; i am alice smith. After.",
            "Before; i am [REDACTED_NAME]. After.",
        ),
        ("Before; i'm alice. After.", "Before; i'm [REDACTED_NAME]. After."),
        (
            "Before; my name is alice. After.",
            "Before; my name is [REDACTED_NAME]. After.",
        ),
    ],
)
def test_deidentify_text_redacts_lowercase_self_introduced_names(
    source: str, expected: str
) -> None:
    c5 = _module()

    assert c5.deidentify_text(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "my name is Mary Jane Watson.",
            "my name is [REDACTED_NAME].",
        ),
        (
            "my name is mary jane watson.",
            "my name is [REDACTED_NAME].",
        ),
        (
            "my name is mary-jane o'connor lee watson.",
            "my name is [REDACTED_NAME].",
        ),
        (
            "my name is mary jane watson and I live here.",
            "my name is [REDACTED_NAME] and I live here.",
        ),
        (
            "my name is alice from the synthetic town.",
            "my name is [REDACTED_NAME] from the synthetic town.",
        ),
    ],
)
def test_deidentify_text_redacts_bounded_names_without_swallowing_clause(
    source: str, expected: str
) -> None:
    c5 = _module()

    assert c5.deidentify_text(source) == expected


def test_quote_context_redacts_email_crossing_quote_boundaries() -> None:
    c5 = _module()
    text = "Reach alice@example.com for synthetic details."

    assert c5.locate_quote_context(text, "example") == (
        "Reach",
        "[REDACTED_EMAIL]",
        "for synthetic details.",
    )
