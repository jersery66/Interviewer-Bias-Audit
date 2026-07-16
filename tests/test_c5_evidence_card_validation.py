from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import FrozenInstanceError
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


def _balanced_spans(
    candidates_per_domain: int = 16, *, include_nuisance_fields: bool = False
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for domain_index, domain in enumerate(EXPECTED_DOMAINS):
        for item_index in range(candidates_per_domain):
            row: dict[str, object] = {
                "participant_id": 10_000 + domain_index * 100 + item_index,
                "domain": domain,
                "polarity": "present" if item_index % 2 == 0 else "absent",
                "exact_quote": (
                    f"Synthetic evidence domain {domain_index} item {item_index}."
                ),
                "source_order": item_index + 1,
            }
            if include_nuisance_fields:
                row.update(
                    {
                        "label": item_index % 2,
                        "paper_label_phq8_ge10": (item_index + domain_index) % 2,
                        "paper_phq8_score": item_index + domain_index,
                        "model_prediction": f"prediction-{item_index % 3}",
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _sample_rows(sample: object) -> pd.DataFrame:
    return pd.concat([sample.training, sample.formal], ignore_index=True)


def _participant_texts(selected: pd.DataFrame) -> dict[int, str]:
    return {
        int(row.participant_id): (
            f"Synthetic context before. {row.exact_quote} Synthetic context after."
        )
        for row in selected.itertuples(index=False)
    }


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


def test_evidence_card_sample_is_frozen() -> None:
    c5 = _module()
    sample = c5.EvidenceCardSample(
        training=pd.DataFrame(), formal=pd.DataFrame(), audit=pd.DataFrame()
    )

    with pytest.raises(FrozenInstanceError):
        sample.training = pd.DataFrame({"replacement": [1]})


def test_balanced_sampling_returns_exact_domain_quotas_and_audit() -> None:
    c5 = _module()

    sample = c5.select_evidence_cards(_balanced_spans(), seed=20260716)

    assert len(sample.training) == 10
    assert len(sample.formal) == 120
    assert sample.training.groupby("domain").size().to_dict() == {
        domain: 1 for domain in EXPECTED_DOMAINS
    }
    assert sample.formal.groupby("domain").size().to_dict() == {
        domain: 12 for domain in EXPECTED_DOMAINS
    }
    assert {
        "domain",
        "eligible_n",
        "training_n",
        "formal_n",
        "seed",
        "successful_attempt",
    }.issubset(sample.audit.columns)
    assert sample.audit["domain"].tolist() == list(EXPECTED_DOMAINS)
    assert sample.audit["eligible_n"].tolist() == [16] * 10
    assert sample.audit["training_n"].tolist() == [1] * 10
    assert sample.audit["formal_n"].tolist() == [12] * 10
    assert sample.audit["seed"].tolist() == [20260716] * 10
    assert sample.audit["successful_attempt"].tolist() == [0] * 10


def test_balanced_sampling_is_deterministic_and_stably_ordered() -> None:
    c5 = _module()

    first = c5.select_evidence_cards(_balanced_spans(), seed=20260716)
    second = c5.select_evidence_cards(_balanced_spans(), seed=20260716)

    pd.testing.assert_frame_equal(first.training, second.training)
    pd.testing.assert_frame_equal(first.formal, second.formal)
    pd.testing.assert_frame_equal(first.audit, second.audit)
    for selected in (first.training, first.formal):
        expected = selected.sort_values(
            ["domain", "participant_id", "source_order", "candidate_key"],
            kind="stable",
        ).reset_index(drop=True)
        pd.testing.assert_frame_equal(selected, expected)


def test_training_and_formal_are_disjoint_and_formal_reuse_is_capped() -> None:
    c5 = _module()

    sample = c5.select_evidence_cards(_balanced_spans(), seed=20260716)

    assert set(sample.training["candidate_key"]).isdisjoint(
        sample.formal["candidate_key"]
    )
    assert sample.formal.groupby("participant_id").size().max() <= 2


def test_candidate_preparation_deduplicates_normalized_quote_and_keeps_first() -> None:
    c5 = _module()
    spans = _balanced_spans(candidates_per_domain=13)
    original = spans.iloc[0].to_dict()
    duplicate = {
        **original,
        "exact_quote": "  SYNTHETIC\n evidence domain 0 item 0.  ",
        "source_order": 0,
        "polarity": "changed-but-not-used-for-the-key",
    }
    spans = pd.concat([spans, pd.DataFrame([duplicate])], ignore_index=True)

    sample = c5.select_evidence_cards(spans, seed=20260716)
    selected = _sample_rows(sample)
    expected_payload = (
        f"{int(original['participant_id'])}|{original['domain']}|"
        "synthetic evidence domain 0 item 0."
    )
    expected_key = hashlib.sha256(expected_payload.encode("utf-8")).hexdigest()
    retained = selected.loc[selected["candidate_key"].eq(expected_key)]

    assert sample.audit.set_index("domain").loc[EXPECTED_DOMAINS[0], "eligible_n"] == 13
    assert len(retained) == 1
    assert retained.iloc[0]["source_order"] == 0
    assert retained.iloc[0]["exact_quote"] == duplicate["exact_quote"]
    assert retained.iloc[0]["normalized_quote"] == (
        "synthetic evidence domain 0 item 0."
    )
    assert selected["candidate_key"].is_unique


def test_duplicate_candidate_keys_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    c5 = _module()
    monkeypatch.setattr(c5, "_candidate_key", lambda *_args: "0" * 64)

    with pytest.raises(ValueError, match=r"duplicate candidate_key"):
        c5.select_evidence_cards(_balanced_spans(candidates_per_domain=13), seed=7)


def test_domain_with_fewer_than_thirteen_candidates_reports_domain_counts() -> None:
    c5 = _module()
    spans = _balanced_spans(candidates_per_domain=13)
    scarce_domain = EXPECTED_DOMAINS[-1]
    spans = spans.loc[
        ~(
            spans["domain"].eq(scarce_domain)
            & spans["source_order"].eq(13)
        )
    ]

    with pytest.raises(ValueError) as exc_info:
        c5.select_evidence_cards(spans, seed=20260716)

    message = str(exc_info.value)
    assert "13" in message
    assert f"{scarce_domain}=12" in message
    assert f"{EXPECTED_DOMAINS[0]}=13" in message


def test_infeasible_global_participant_cap_fails_hard() -> None:
    c5 = _module()
    spans = _balanced_spans(candidates_per_domain=13)
    spans["participant_id"] = [
        500 + (item_index % 5)
        for _domain_index in range(len(EXPECTED_DOMAINS))
        for item_index in range(13)
    ]

    with pytest.raises(RuntimeError, match=r"feasible|participant cap"):
        c5.select_evidence_cards(spans, seed=20260716)


def test_phq_labels_and_model_predictions_do_not_influence_sampling() -> None:
    c5 = _module()
    original = _balanced_spans(include_nuisance_fields=True)
    changed = original.copy(deep=True)
    changed["label"] = 99
    changed["paper_label_phq8_ge10"] = 1 - changed["paper_label_phq8_ge10"]
    changed["paper_phq8_score"] = changed["paper_phq8_score"] + 1000
    changed["model_prediction"] = "completely-changed"

    first = c5.select_evidence_cards(original, seed=20260716)
    second = c5.select_evidence_cards(changed, seed=20260716)

    assert first.training["candidate_key"].tolist() == second.training[
        "candidate_key"
    ].tolist()
    assert first.formal["candidate_key"].tolist() == second.formal[
        "candidate_key"
    ].tolist()


@pytest.mark.parametrize("seed", [True, False, 1.5, "20260716", None])
def test_sampling_rejects_bool_and_non_integer_seeds(seed: object) -> None:
    c5 = _module()

    with pytest.raises(TypeError, match="seed"):
        c5.select_evidence_cards(_balanced_spans(candidates_per_domain=13), seed=seed)


def test_sampling_validates_required_columns_defensively() -> None:
    c5 = _module()

    with pytest.raises(ValueError, match=r"required columns.*polarity"):
        c5.select_evidence_cards(
            _balanced_spans(candidates_per_domain=13).drop(columns="polarity"),
            seed=20260716,
        )


def test_review_tables_have_exact_blank_reviewer_columns_and_owner_linkage() -> None:
    c5 = _module()
    spans = _balanced_spans(include_nuisance_fields=True)
    sample = c5.select_evidence_cards(spans, seed=20260716)
    participant_texts = _participant_texts(sample.formal)

    owner, reviewer_a, reviewer_b = c5.build_review_tables(
        sample.formal, participant_texts, seed=20260716
    )

    expected_reviewer_columns = [
        "review_case_id",
        "evidence_quote",
        "context_before",
        "context_after",
        "human_span_valid",
        "human_domain",
        "human_polarity",
        "notes",
    ]
    expected_owner_columns = [
        "review_case_id",
        "participant_id",
        "source_order",
        "candidate_key",
        "model_domain",
        "model_polarity",
        "model_exact_quote",
        "evidence_quote",
        "context_before",
        "context_after",
        "sample_scope",
    ]
    answer_fields = [
        "human_span_valid",
        "human_domain",
        "human_polarity",
        "notes",
    ]
    assert list(c5.REVIEWER_COLUMNS) == expected_reviewer_columns
    assert list(reviewer_a.columns) == expected_reviewer_columns
    assert list(reviewer_b.columns) == expected_reviewer_columns
    assert list(owner.columns) == expected_owner_columns
    assert reviewer_a[answer_fields].eq("").all().all()
    assert reviewer_b[answer_fields].eq("").all().all()
    assert owner["review_case_id"].is_unique
    assert owner["sample_scope"].eq("formal").all()
    assert owner["candidate_key"].tolist() == sample.formal["candidate_key"].tolist()
    assert owner["participant_id"].tolist() == sample.formal["participant_id"].tolist()
    assert owner["source_order"].tolist() == sample.formal["source_order"].tolist()
    assert owner["model_domain"].tolist() == sample.formal["domain"].tolist()
    assert owner["model_polarity"].tolist() == sample.formal["polarity"].tolist()
    assert owner["model_exact_quote"].tolist() == sample.formal["exact_quote"].tolist()

    explicitly_prohibited = {
        "participant_id",
        "label",
        "paper_label_phq8_ge10",
        "paper_phq8_score",
        "model_domain",
        "model_polarity",
        "domain",
        "polarity",
        "source_order",
        "candidate_key",
    }
    assert explicitly_prohibited.issubset(c5.PROHIBITED_REVIEWER_COLUMNS)
    assert c5.PROHIBITED_REVIEWER_COLUMNS.isdisjoint(reviewer_a.columns)
    assert c5.PROHIBITED_REVIEWER_COLUMNS.isdisjoint(reviewer_b.columns)


def test_reviewer_orders_are_different_and_deterministic_with_the_same_id_set() -> None:
    c5 = _module()
    sample = c5.select_evidence_cards(_balanced_spans(), seed=20260716)
    participant_texts = _participant_texts(sample.formal)

    first = c5.build_review_tables(sample.formal, participant_texts, seed=20260716)
    second = c5.build_review_tables(sample.formal, participant_texts, seed=20260716)
    _, reviewer_a, reviewer_b = first

    for first_table, second_table in zip(first, second):
        pd.testing.assert_frame_equal(first_table, second_table)
    assert set(reviewer_a["review_case_id"]) == set(reviewer_b["review_case_id"])
    assert reviewer_a["review_case_id"].is_unique
    assert reviewer_b["review_case_id"].is_unique
    assert reviewer_a["review_case_id"].tolist() != reviewer_b[
        "review_case_id"
    ].tolist()


def test_review_case_ids_are_opaque_sha256_values_and_scope_specific() -> None:
    c5 = _module()
    participant_id = 987_654_321
    candidate_key = hashlib.sha256(b"synthetic candidate").hexdigest()
    expected_digest = hashlib.sha256(
        (
            f"{c5.C5_CANONICAL_SHA256}|formal|{candidate_key}"
        ).encode("utf-8")
    ).hexdigest()

    formal_id = c5._review_case_id(candidate_key, "formal")
    training_id = c5._review_case_id(candidate_key, "training")

    assert formal_id == f"C5-{expected_digest}"
    assert str(participant_id) not in formal_id
    assert formal_id != training_id


def test_review_tables_reject_missing_participant_text() -> None:
    c5 = _module()
    sample = c5.select_evidence_cards(_balanced_spans(), seed=20260716)
    selected = sample.formal.iloc[:2].copy()
    missing_id = int(selected.iloc[0]["participant_id"])
    participant_texts = _participant_texts(selected.iloc[1:])

    with pytest.raises(ValueError, match=rf"participant text.*{missing_id}"):
        c5.build_review_tables(selected, participant_texts, seed=20260716)


def test_review_tables_propagate_ambiguous_repeated_quote_failure() -> None:
    c5 = _module()
    sample = c5.select_evidence_cards(_balanced_spans(), seed=20260716)
    selected = sample.formal.iloc[:2].copy()
    participant_texts = _participant_texts(selected)
    first = selected.iloc[0]
    participant_texts[int(first["participant_id"])] = (
        f"{first['exact_quote']} Between. {first['exact_quote']}"
    )

    with pytest.raises(ValueError, match=r"ambiguous|multiple occurrence"):
        c5.build_review_tables(selected, participant_texts, seed=20260716)


def test_review_tables_reject_duplicate_reviewer_ids() -> None:
    c5 = _module()
    sample = c5.select_evidence_cards(_balanced_spans(), seed=20260716)
    selected = pd.concat(
        [sample.formal.iloc[[0]], sample.formal.iloc[[0]]], ignore_index=True
    )
    participant_texts = _participant_texts(selected)

    with pytest.raises(ValueError, match=r"duplicate review_case_id"):
        c5.build_review_tables(selected, participant_texts, seed=20260716)


@pytest.mark.parametrize("seed", [True, False, 1.5, "20260716", None])
def test_review_tables_reject_bool_and_non_integer_seeds(seed: object) -> None:
    c5 = _module()
    sample = c5.select_evidence_cards(
        _balanced_spans(candidates_per_domain=13), seed=20260716
    )
    selected = sample.formal.iloc[:2].copy()

    with pytest.raises(TypeError, match="seed"):
        c5.build_review_tables(
            selected, _participant_texts(selected), seed=seed
        )


@pytest.mark.parametrize(
    "scope", ["", "validation", "FORMAL", None, True, ["unhashable"]]
)
def test_review_tables_reject_invalid_scopes(scope: object) -> None:
    c5 = _module()
    sample = c5.select_evidence_cards(
        _balanced_spans(candidates_per_domain=13), seed=20260716
    )
    selected = sample.formal.iloc[:2].copy()

    with pytest.raises(ValueError, match="scope"):
        c5.build_review_tables(
            selected,
            _participant_texts(selected),
            seed=20260716,
            scope=scope,
        )


@pytest.mark.parametrize(
    "missing_column",
    [
        "candidate_key",
        "participant_id",
        "source_order",
        "domain",
        "polarity",
        "exact_quote",
    ],
)
def test_review_tables_reject_missing_required_columns(missing_column: str) -> None:
    c5 = _module()
    sample = c5.select_evidence_cards(
        _balanced_spans(candidates_per_domain=13), seed=20260716
    )
    selected = sample.formal.iloc[:2].drop(columns=missing_column)

    with pytest.raises(ValueError, match=rf"required columns.*{missing_column}"):
        c5.build_review_tables(selected, {}, seed=20260716)
