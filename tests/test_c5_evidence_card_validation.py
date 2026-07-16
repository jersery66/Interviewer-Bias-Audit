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


def _cross_domain_reuse_spans() -> pd.DataFrame:
    spans = _balanced_spans()
    spans["participant_id"] = [
        20_000 + (row_index % 100) for row_index in range(len(spans))
    ]
    return spans


def _sample_rows(sample: object) -> pd.DataFrame:
    return pd.concat([sample.training, sample.formal], ignore_index=True)


def _participant_texts(selected: pd.DataFrame) -> dict[int, str]:
    return {
        int(row.participant_id): (
            f"Synthetic context before. {row.exact_quote} Synthetic context after."
        )
        for row in selected.itertuples(index=False)
    }


def _write_csv(path: Path, frame: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False, lineterminator="\n")
    return _physical_sha256(path)


def _write_package_inputs(
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    spans: pd.DataFrame | None = None,
    seed: int = 20260716,
    ambiguous_selected_quote: bool = False,
) -> tuple[Path, Path, pd.DataFrame, object]:
    c5 = _module()
    source = (
        _balanced_spans() if spans is None else spans.copy(deep=True)
    )
    canonical_path = project_root / c5.C5_CANONICAL_RELATIVE
    canonical_sha256 = _write_csv(canonical_path, source)

    sample = c5.select_evidence_cards(source, seed=seed)
    selected = _sample_rows(sample)
    records: list[dict[str, object]] = []
    for participant_id, participant_rows in selected.groupby(
        "participant_id", sort=True
    ):
        quotes = participant_rows["exact_quote"].astype(str).tolist()
        text = (
            "Synthetic context before. "
            + " Synthetic separator. ".join(quotes)
            + " Synthetic context after."
        )
        records.append(
            {"participant_id": int(participant_id), "text": text}
        )

    if ambiguous_selected_quote:
        first = selected.iloc[0]
        first_id = int(first["participant_id"])
        for record in records:
            if record["participant_id"] == first_id:
                record["text"] = (
                    f"{record['text']} Repeated once: {first['exact_quote']}"
                )
                break

    participant_text_path = project_root / c5.PARTICIPANT_TEXT_RELATIVE
    participant_text_path.parent.mkdir(parents=True, exist_ok=True)
    participant_sha256 = _write_jsonl(participant_text_path, records)

    monkeypatch.setattr(c5, "C5_CANONICAL_SHA256", canonical_sha256)
    monkeypatch.setattr(
        c5, "PARTICIPANT_TEXT_SHA256", participant_sha256
    )
    monkeypatch.setattr(c5, "C5_CANONICAL_ROW_COUNT", len(source))
    return canonical_path, participant_text_path, source, sample


def _staging_path(output_root: Path) -> Path:
    return output_root.parent / f".{output_root.name}.staging"


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


EXPECTED_PACKAGE_FILES = {
    "评分者A_培训/README_评分者A.md",
    "评分者A_培训/reviewer_A_training.json",
    "评分者B_培训/README_评分者B.md",
    "评分者B_培训/reviewer_B_training.json",
    "OWNER_ONLY/README_OWNER_ONLY.md",
    "OWNER_ONLY/evidence_card_owner_crosswalk.csv",
    "OWNER_ONLY/evidence_card_sampling_audit.csv",
    "OWNER_ONLY/evidence_card_manifest.json",
    "OWNER_ONLY/待编码手册冻结后发放/reviewer_A_formal.json",
    "OWNER_ONLY/待编码手册冻结后发放/reviewer_B_formal.json",
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
    assert c5.C5_CANONICAL_ROW_COUNT == 1137
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
    assert list(sample.audit.columns) == [
        "domain",
        "eligible_n",
        "training_n",
        "formal_n",
        "seed",
        "attempt",
    ]
    assert "successful_attempt" not in sample.audit.columns
    assert sample.audit["domain"].tolist() == list(EXPECTED_DOMAINS)
    assert sample.audit["eligible_n"].tolist() == [16] * 10
    assert sample.audit["training_n"].tolist() == [1] * 10
    assert sample.audit["formal_n"].tolist() == [12] * 10
    assert sample.audit["seed"].tolist() == [20260716] * 10
    assert sample.audit["attempt"].tolist() == [0] * 10


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


def test_feasible_cross_domain_participant_reuse_respects_formal_cap() -> None:
    c5 = _module()
    spans = _cross_domain_reuse_spans()
    candidate_domain_counts = spans.groupby("participant_id")["domain"].nunique()
    assert candidate_domain_counts.eq(2).sum() == 60

    sample = c5.select_evidence_cards(spans, seed=20260716)
    formal_reuse = sample.formal.groupby("participant_id").size()

    assert (formal_reuse == 2).any()
    assert formal_reuse.max() == 2
    assert set(sample.training["candidate_key"]).isdisjoint(
        sample.formal["candidate_key"]
    )
    assert set(sample.training["participant_id"]).intersection(
        sample.formal["participant_id"]
    )


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


def test_review_tables_reject_duplicate_candidate_keys() -> None:
    c5 = _module()
    sample = c5.select_evidence_cards(_balanced_spans(), seed=20260716)
    selected = pd.concat(
        [sample.formal.iloc[[0]], sample.formal.iloc[[0]]], ignore_index=True
    )
    participant_texts = _participant_texts(selected)

    with pytest.raises(ValueError, match=r"duplicate candidate_key"):
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


def test_evidence_card_package_paths_are_frozen_and_isolated(
    tmp_path: Path,
) -> None:
    c5 = _module()
    root = tmp_path / "restricted-package"

    paths = c5.EvidenceCardPackagePaths(root)

    assert paths.root == root
    assert paths.reviewer_a_training_json == (
        root / "评分者A_培训" / "reviewer_A_training.json"
    )
    assert paths.reviewer_a_training_readme == (
        root / "评分者A_培训" / "README_评分者A.md"
    )
    assert paths.reviewer_b_training_json == (
        root / "评分者B_培训" / "reviewer_B_training.json"
    )
    assert paths.reviewer_b_training_readme == (
        root / "评分者B_培训" / "README_评分者B.md"
    )
    assert paths.owner_crosswalk_csv == (
        root / "OWNER_ONLY" / "evidence_card_owner_crosswalk.csv"
    )
    assert paths.sampling_audit_csv == (
        root / "OWNER_ONLY" / "evidence_card_sampling_audit.csv"
    )
    assert paths.manifest_json == (
        root / "OWNER_ONLY" / "evidence_card_manifest.json"
    )
    assert paths.owner_readme == (
        root / "OWNER_ONLY" / "README_OWNER_ONLY.md"
    )
    assert paths.reviewer_a_formal_json == (
        root
        / "OWNER_ONLY"
        / "待编码手册冻结后发放"
        / "reviewer_A_formal.json"
    )
    assert paths.reviewer_b_formal_json == (
        root
        / "OWNER_ONLY"
        / "待编码手册冻结后发放"
        / "reviewer_B_formal.json"
    )
    with pytest.raises(FrozenInstanceError):
        paths.root = tmp_path / "replacement"


def test_package_build_has_exact_layout_and_keeps_formal_files_owner_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c5 = _module()
    canonical, context, _spans, _sample = _write_package_inputs(
        tmp_path / "project", monkeypatch
    )
    output_root = tmp_path / "restricted-package"

    result = c5.build_evidence_card_package(
        canonical, context, output_root, seed=20260716
    )

    assert result == c5.EvidenceCardPackagePaths(output_root.resolve())
    assert set(_tree_bytes(output_root)) == EXPECTED_PACKAGE_FILES
    assert not _staging_path(output_root).exists()
    assert not list(output_root.rglob("*.xlsx"))
    for reviewer_dir in ("评分者A_培训", "评分者B_培训"):
        reviewer_files = {
            path.name for path in (output_root / reviewer_dir).iterdir()
        }
        assert not any("formal" in name for name in reviewer_files)


def test_package_reviewer_and_owner_schemas_are_blinded_and_blank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c5 = _module()
    canonical, context, _spans, _sample = _write_package_inputs(
        tmp_path / "project", monkeypatch
    )
    output_root = tmp_path / "restricted-package"
    paths = c5.build_evidence_card_package(
        canonical, context, output_root, seed=20260716
    )

    reviewer_paths = (
        paths.reviewer_a_training_json,
        paths.reviewer_b_training_json,
        paths.reviewer_a_formal_json,
        paths.reviewer_b_formal_json,
    )
    answer_fields = (
        "human_span_valid",
        "human_domain",
        "human_polarity",
        "notes",
    )
    reviewer_records: list[list[dict[str, object]]] = []
    for reviewer_path in reviewer_paths:
        records = json.loads(reviewer_path.read_text(encoding="utf-8"))
        assert isinstance(records, list)
        assert records
        assert all(list(record) == list(c5.REVIEWER_COLUMNS) for record in records)
        assert all(
            record[field] == ""
            for record in records
            for field in answer_fields
        )
        assert all(
            not c5._prohibited_reviewer_columns(pd.Index(record))
            for record in records
        )
        reviewer_records.append(records)

    assert len(reviewer_records[0]) == len(reviewer_records[1]) == 10
    assert len(reviewer_records[2]) == len(reviewer_records[3]) == 120
    for left, right in (
        (reviewer_records[0], reviewer_records[1]),
        (reviewer_records[2], reviewer_records[3]),
    ):
        left_ids = [str(record["review_case_id"]) for record in left]
        right_ids = [str(record["review_case_id"]) for record in right]
        assert set(left_ids) == set(right_ids)
        assert left_ids != right_ids

    owner = pd.read_csv(paths.owner_crosswalk_csv, keep_default_na=False)
    assert list(owner.columns) == list(c5._OWNER_COLUMNS)
    assert len(owner) == 130
    assert owner["sample_scope"].tolist() == ["training"] * 10 + [
        "formal"
    ] * 120
    assert {"participant_id", "model_domain", "model_polarity"}.issubset(
        owner.columns
    )
    audit = pd.read_csv(paths.sampling_audit_csv)
    assert list(audit.columns) == [
        "domain",
        "eligible_n",
        "training_n",
        "formal_n",
        "seed",
        "attempt",
    ]
    assert audit["domain"].tolist() == list(EXPECTED_DOMAINS)


def test_package_outputs_and_manifest_are_deterministic_and_hash_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c5 = _module()
    canonical, context, spans, sample = _write_package_inputs(
        tmp_path / "project", monkeypatch
    )
    first_root = tmp_path / "package-one"
    second_root = tmp_path / "package-two"

    first_paths = c5.build_evidence_card_package(
        canonical, context, first_root, seed=20260716
    )
    c5.build_evidence_card_package(
        canonical, context, second_root, seed=20260716
    )

    assert _tree_bytes(first_root) == _tree_bytes(second_root)
    manifest = json.loads(first_paths.manifest_json.read_text(encoding="utf-8"))
    assert manifest["status"] == "prepared_not_scored"
    assert manifest["design"] == "balanced_evidence_cards"
    assert manifest["canonical_path"] == str(canonical.resolve())
    assert manifest["participant_text_path"] == str(context.resolve())
    assert manifest["canonical_sha256"] == c5.C5_CANONICAL_SHA256
    assert manifest["participant_text_sha256"] == c5.PARTICIPANT_TEXT_SHA256
    assert manifest["seed"] == 20260716
    assert manifest["input_row_count"] == len(spans)
    assert manifest["training_total"] == 10
    assert manifest["formal_total"] == 120
    assert manifest["training_per_domain"] == {
        domain: 1 for domain in EXPECTED_DOMAINS
    }
    assert manifest["formal_per_domain"] == {
        domain: 12 for domain in EXPECTED_DOMAINS
    }
    assert manifest["formal_max_participant_reuse"] == int(
        sample.formal.groupby("participant_id").size().max()
    )
    assert manifest["training_formal_key_overlap_count"] == 0
    assert manifest["reviewer_id_checks"] == {
        "training": {"id_sets_equal": True, "orders_differ": True},
        "formal": {"id_sets_equal": True, "orders_differ": True},
    }
    assert manifest["reviewer_prohibited_field_check"] == {
        "passed": True,
        "prohibited_fields_found": [],
    }
    assert manifest["all_answer_fields_blank"] is True
    assert manifest["formal_materials_owner_only_until_codebook_freeze"] is True
    assert "工作簿" in manifest["formal_release_policy_zh"]
    assert "JSON" in manifest["formal_release_policy_zh"]
    assert "OWNER_ONLY" in manifest["formal_release_policy_zh"]
    assert "编码手册冻结" in manifest["formal_release_policy_zh"]
    assert manifest["human_review_completed"] is False
    assert manifest["agreement_completed"] is False
    assert manifest["workbooks_generated"] is False

    manifest_relative = "OWNER_ONLY/evidence_card_manifest.json"
    non_manifest_files = EXPECTED_PACKAGE_FILES - {manifest_relative}
    expected_hashes = {
        relative: _physical_sha256(first_root / Path(relative))
        for relative in sorted(non_manifest_files)
    }
    assert manifest["file_hashes"] == expected_hashes
    assert manifest["file_hashes_excludes"] == [manifest_relative]
    assert "自身" in manifest["file_hashes_exclusion_reason_zh"]


@pytest.mark.parametrize(
    "failure_case",
    [
        "canonical_wrong_hash",
        "context_wrong_hash",
        "canonical_missing",
        "context_missing",
        "row_count",
        "domain_shortage",
        "ambiguous_selected_quote",
    ],
)
def test_invalid_preflight_leaves_no_output_or_staging(
    failure_case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c5 = _module()
    canonical, context, spans, _sample = _write_package_inputs(
        tmp_path / "project",
        monkeypatch,
        ambiguous_selected_quote=failure_case == "ambiguous_selected_quote",
    )
    if failure_case == "canonical_wrong_hash":
        monkeypatch.setattr(c5, "C5_CANONICAL_SHA256", "0" * 64)
    elif failure_case == "context_wrong_hash":
        monkeypatch.setattr(c5, "PARTICIPANT_TEXT_SHA256", "0" * 64)
    elif failure_case == "canonical_missing":
        canonical.unlink()
    elif failure_case == "context_missing":
        context.unlink()
    elif failure_case == "row_count":
        monkeypatch.setattr(c5, "C5_CANONICAL_ROW_COUNT", len(spans) + 1)
    elif failure_case == "domain_shortage":
        scarce_domain = EXPECTED_DOMAINS[-1]
        scarce = spans.loc[
            ~(
                spans["domain"].eq(scarce_domain)
                & spans["source_order"].isin([13, 14, 15, 16])
            )
        ].copy()
        extras: list[dict[str, object]] = []
        for extra_index in range(4):
            extra = spans.iloc[0].to_dict()
            extra.update(
                {
                    "participant_id": 900_000 + extra_index,
                    "source_order": 900 + extra_index,
                    "exact_quote": f"Extra replacement candidate {extra_index}.",
                }
            )
            extras.append(extra)
        scarce = pd.concat([scarce, pd.DataFrame(extras)], ignore_index=True)
        assert len(scarce) == len(spans)
        monkeypatch.setattr(
            c5, "C5_CANONICAL_SHA256", _write_csv(canonical, scarce)
        )

    output_root = tmp_path / "restricted-package"
    with pytest.raises((FileNotFoundError, ValueError)):
        c5.build_evidence_card_package(
            canonical, context, output_root, seed=20260716
        )

    assert not output_root.exists()
    assert not _staging_path(output_root).exists()


@pytest.mark.parametrize("existing_target", ["output", "staging"])
def test_existing_output_or_staging_is_refused_without_modification(
    existing_target: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c5 = _module()
    canonical, context, _spans, _sample = _write_package_inputs(
        tmp_path / "project", monkeypatch
    )
    output_root = tmp_path / "restricted-package"
    staging = _staging_path(output_root)
    protected = output_root if existing_target == "output" else staging
    protected.mkdir(parents=True)
    (protected / "sentinel.bin").write_bytes(b"must remain unchanged\x00")
    before = _tree_bytes(protected)

    with pytest.raises(FileExistsError):
        c5.build_evidence_card_package(
            canonical, context, output_root, seed=20260716
        )

    assert _tree_bytes(protected) == before
    if existing_target == "output":
        assert not staging.exists()
    else:
        assert not output_root.exists()


def test_mid_write_failure_removes_only_the_expected_staging_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c5 = _module()
    canonical, context, _spans, _sample = _write_package_inputs(
        tmp_path / "project", monkeypatch
    )
    output_root = tmp_path / "restricted-package"
    unrelated = tmp_path / ".unrelated.staging"
    unrelated.mkdir()
    sentinel = unrelated / "sentinel.bin"
    sentinel.write_bytes(b"unrelated path must survive")
    original_writer = c5._write_json_records
    call_count = 0

    def fail_on_second_json(frame: pd.DataFrame, path: Path) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise OSError("synthetic mid-write failure")
        original_writer(frame, path)

    monkeypatch.setattr(c5, "_write_json_records", fail_on_second_json)

    with pytest.raises(OSError, match="synthetic mid-write failure"):
        c5.build_evidence_card_package(
            canonical, context, output_root, seed=20260716
        )

    assert call_count == 2
    assert not output_root.exists()
    assert not _staging_path(output_root).exists()
    assert sentinel.read_bytes() == b"unrelated path must survive"


def test_json_records_writer_round_trips_quotes_and_newlines(
    tmp_path: Path,
) -> None:
    c5 = _module()
    frame = pd.DataFrame(
        [
            {
                "review_case_id": "C5-" + "a" * 64,
                "evidence_quote": 'Line one\n"Line two"',
                "context_before": "前文",
                "context_after": "后文",
                "human_span_valid": "",
                "human_domain": "",
                "human_polarity": "",
                "notes": "",
            }
        ],
        columns=list(c5.REVIEWER_COLUMNS),
    )
    target = tmp_path / "records.json"

    c5._write_json_records(frame, target)

    assert json.loads(target.read_text(encoding="utf-8")) == frame.to_dict(
        orient="records"
    )
    raw = target.read_bytes()
    assert b"\\n" in raw
    assert b'\\"Line two\\"' in raw
    assert "前文" in raw.decode("utf-8")


def test_generated_and_public_markdown_is_chinese_and_contains_no_quotes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c5 = _module()
    canonical, context, spans, _sample = _write_package_inputs(
        tmp_path / "project", monkeypatch
    )
    paths = c5.build_evidence_card_package(
        canonical,
        context,
        tmp_path / "restricted-package",
        seed=20260716,
    )
    generated_markdown = (
        paths.reviewer_a_training_readme,
        paths.reviewer_b_training_readme,
        paths.owner_readme,
    )
    source_quotes = spans["exact_quote"].astype(str).tolist()
    for markdown_path in generated_markdown:
        text = markdown_path.read_text(encoding="utf-8")
        assert sum("\u4e00" <= char <= "\u9fff" for char in text) >= 20
        assert not any(quote in text for quote in source_quotes)

    for reviewer_readme in (
        paths.reviewer_a_training_readme,
        paths.reviewer_b_training_readme,
    ):
        text = reviewer_readme.read_text(encoding="utf-8")
        assert "三项" in text
        assert "有效性" in text
        assert "领域" in text
        assert "极性" in text
        assert "XLSX" in text
        assert "不要返回 JSON" in text

    owner_text = paths.owner_readme.read_text(encoding="utf-8")
    assert "先完成培训" in owner_text
    assert "编码手册冻结" in owner_text
    assert "正式" in owner_text
    assert "OWNER_ONLY" in owner_text

    public_readme = (
        Path(__file__).resolve().parents[1]
        / "analysis_v2"
        / "04_c5_controls"
        / "blind_quality_audit"
        / "evidence_card_v1"
        / "README.md"
    )
    public_text = public_readme.read_text(encoding="utf-8")
    assert sum("\u4e00" <= char <= "\u9fff" for char in public_text) >= 50
    assert "旧有的全转录十领域重新编码流程" in public_text
    assert "每位评分者 120 张正式证据卡" in public_text
    assert "有效性" in public_text
    assert "领域" in public_text
    assert "极性" in public_text
    assert "不评估缺失证据" in public_text
    assert "不计算召回率" in public_text
    assert "受限评分材料不纳入 Git" in public_text
    assert "材料准备" in public_text
    assert "尚未完成人工验证" in public_text


def test_cli_success_uses_locked_paths_and_prints_only_safe_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    c5 = _module()
    project_root = tmp_path / "project"
    _write_package_inputs(project_root, monkeypatch)
    output_root = tmp_path / "restricted-package"
    cli = importlib.import_module("scripts.build_c5_evidence_card_validation")

    exit_code = cli.main(
        [
            "--project-root",
            str(project_root),
            "--output-root",
            str(output_root),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    summary = json.loads(captured.out)
    assert summary["status"] == "prepared_not_scored"
    assert summary["output_root"] == str(output_root.resolve())
    assert summary["manifest_path"] == str(
        (
            output_root
            / "OWNER_ONLY"
            / "evidence_card_manifest.json"
        ).resolve()
    )
    assert summary["training_total"] == 10
    assert summary["formal_total"] == 120
    assert summary["canonical_sha256"] == c5.C5_CANONICAL_SHA256
    assert summary["participant_text_sha256"] == c5.PARTICIPANT_TEXT_SHA256
    assert set(summary["file_hashes"]) == (
        EXPECTED_PACKAGE_FILES
        - {"OWNER_ONLY/evidence_card_manifest.json"}
    )
    assert "evidence_quote" not in captured.out
    assert "review_case_id" not in captured.out
    assert "Synthetic evidence" not in captured.out


def test_cli_failure_exits_nonzero_without_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    c5 = _module()
    project_root = tmp_path / "project"
    _write_package_inputs(project_root, monkeypatch)
    monkeypatch.setattr(c5, "PARTICIPANT_TEXT_SHA256", "0" * 64)
    output_root = tmp_path / "restricted-package"
    cli = importlib.import_module("scripts.build_c5_evidence_card_validation")

    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "--project-root",
                str(project_root),
                "--output-root",
                str(output_root),
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert captured.out == ""
    assert "error:" in captured.err
    assert not output_root.exists()
    assert not _staging_path(output_root).exists()
