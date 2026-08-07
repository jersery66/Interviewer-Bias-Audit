"""Validate frozen C5 evidence-card inputs and build safe quote context."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
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
C5_CANONICAL_ROW_COUNT = 1137
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
REVIEWER_COLUMNS = (
    "review_case_id",
    "evidence_quote",
    "context_before",
    "context_after",
    "human_span_valid",
    "human_domain",
    "human_polarity",
    "notes",
)
PROHIBITED_REVIEWER_COLUMNS = frozenset(
    {
        "participant_id",
        "label",
        "phq",
        "phq8",
        "phq8_binary",
        "phq8_label",
        "phq8_score",
        "paper_label_phq8_ge10",
        "paper_phq8_score",
        "model_prediction",
        "model_domain",
        "model_polarity",
        "domain",
        "polarity",
        "source_order",
        "candidate_key",
    }
)

_REQUIRED_REVIEW_COLUMNS = (
    "candidate_key",
    "participant_id",
    "source_order",
    "domain",
    "polarity",
    "exact_quote",
)
_OWNER_COLUMNS = (
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
)
_VALID_SAMPLE_SCOPES = frozenset({"training", "formal"})
_TRAINING_PER_DOMAIN = 1
_FORMAL_PER_DOMAIN = 12
_MIN_ELIGIBLE_PER_DOMAIN = _TRAINING_PER_DOMAIN + _FORMAL_PER_DOMAIN
_MAX_FORMAL_PARTICIPANT_REUSE = 2
_MAX_SELECTION_ATTEMPTS = 2000

OWNER_PRIVACY_APPROVAL_SCHEMA = "c5_owner_privacy_approval_v1"
OWNER_PRIVACY_RELEASE_BLOCKED = (
    "blocked_pending_owner_manual_privacy_audit"
)
OWNER_PRIVACY_RELEASE_APPROVED = (
    "approved_after_owner_manual_privacy_audit"
)
_PRIVACY_PAYLOAD_COLUMNS = (
    "review_case_id",
    "evidence_quote",
    "context_before",
    "context_after",
)
_OWNER_PRIVACY_CASE_FIELDS = frozenset(
    {
        "review_case_id",
        "reviewer_payload_sha256",
        "owner_privacy_approved",
    }
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


@dataclass(frozen=True)
class EvidenceCardSample:
    """Balanced training/formal selections plus their sampling audit."""

    training: pd.DataFrame
    formal: pd.DataFrame
    audit: pd.DataFrame


@dataclass(frozen=True)
class EvidenceCardPackagePaths:
    """Restricted evidence-card package paths rooted at one directory."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    @property
    def reviewer_a_training_dir(self) -> Path:
        return self.root / "评分者A_培训"

    @property
    def reviewer_a_training_json(self) -> Path:
        return self.reviewer_a_training_dir / "reviewer_A_training.json"

    @property
    def reviewer_a_training_readme(self) -> Path:
        return self.reviewer_a_training_dir / "README_评分者A.md"

    @property
    def reviewer_b_training_dir(self) -> Path:
        return self.root / "评分者B_培训"

    @property
    def reviewer_b_training_json(self) -> Path:
        return self.reviewer_b_training_dir / "reviewer_B_training.json"

    @property
    def reviewer_b_training_readme(self) -> Path:
        return self.reviewer_b_training_dir / "README_评分者B.md"

    @property
    def owner_dir(self) -> Path:
        return self.root / "OWNER_ONLY"

    @property
    def owner_crosswalk_csv(self) -> Path:
        return self.owner_dir / "evidence_card_owner_crosswalk.csv"

    @property
    def sampling_audit_csv(self) -> Path:
        return self.owner_dir / "evidence_card_sampling_audit.csv"

    @property
    def manifest_json(self) -> Path:
        return self.owner_dir / "evidence_card_manifest.json"

    @property
    def owner_readme(self) -> Path:
        return self.owner_dir / "README_OWNER_ONLY.md"

    @property
    def owner_privacy_approval_template_json(self) -> Path:
        return (
            self.owner_dir
            / "evidence_card_owner_privacy_approval_template.json"
        )

    @property
    def formal_hold_dir(self) -> Path:
        return self.owner_dir / "待编码手册冻结后发放"

    @property
    def reviewer_a_formal_json(self) -> Path:
        return self.formal_hold_dir / "reviewer_A_formal.json"

    @property
    def reviewer_b_formal_json(self) -> Path:
        return self.formal_hold_dir / "reviewer_B_formal.json"


@dataclass(frozen=True)
class _EvidenceCardPackageData:
    sample: EvidenceCardSample
    training_owner: pd.DataFrame
    training_reviewer_a: pd.DataFrame
    training_reviewer_b: pd.DataFrame
    formal_owner: pd.DataFrame
    formal_reviewer_a: pd.DataFrame
    formal_reviewer_b: pd.DataFrame


@dataclass(frozen=True)
class OwnerPrivacyReleaseStatus:
    """Machine-checkable status for the owner privacy release gate."""

    status: str
    release_allowed: bool
    expected_case_count: int
    approved_case_count: int
    automatic_redaction_sufficient_for_release: bool = False


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


def _validate_seed(seed: Any) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise TypeError("seed must be an integer and must not be boolean")
    return int(seed)


def _numpy_seed(seed: int, offset: int = 0) -> int:
    value = seed + offset
    if 0 <= value < 2**64:
        return value
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _candidate_key(row: pd.Series) -> str:
    payload = (
        f"{int(row['participant_id'])}|{row['domain']}|"
        f"{row['normalized_quote']}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _domain_count_text(counts: pd.Series) -> str:
    return ", ".join(
        f"{domain}={int(counts.loc[domain])}" for domain in DOMAIN_NAMES
    )


def _prepare_candidates(spans: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    frame = load_and_validate_spans(spans, expected_sha256=None)
    frame["normalized_quote"] = frame["exact_quote"].map(
        lambda value: normalize_space(value).casefold()
    )
    frame = frame.sort_values(
        ["domain", "participant_id", "source_order"], kind="stable"
    )
    frame = frame.drop_duplicates(
        ["participant_id", "domain", "normalized_quote"], keep="first"
    ).copy()
    frame["candidate_key"] = frame.apply(_candidate_key, axis=1)

    duplicate_key_mask = frame["candidate_key"].duplicated(keep=False)
    if duplicate_key_mask.any():
        duplicate_keys = sorted(
            frame.loc[duplicate_key_mask, "candidate_key"].astype(str).unique()
        )
        raise ValueError(
            "duplicate candidate_key values after candidate preparation: "
            + ", ".join(duplicate_keys)
        )

    frame = frame.reset_index(drop=True)
    counts = (
        frame.groupby("domain", sort=False)
        .size()
        .reindex(DOMAIN_NAMES, fill_value=0)
        .astype("int64")
    )
    if counts.lt(_MIN_ELIGIBLE_PER_DOMAIN).any():
        raise ValueError(
            f"Each locked domain requires at least {_MIN_ELIGIBLE_PER_DOMAIN} "
            "eligible unique candidates; domain counts: "
            + _domain_count_text(counts)
        )
    return frame, counts


def _stable_sample_order(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        ["domain", "participant_id", "source_order", "candidate_key"],
        kind="stable",
    ).reset_index(drop=True)


def select_evidence_cards(
    spans: pd.DataFrame, *, seed: int
) -> EvidenceCardSample:
    """Select balanced, disjoint training and formal evidence cards."""

    validated_seed = _validate_seed(seed)
    candidates, counts = _prepare_candidates(spans)

    participant_capacity = int(
        candidates.groupby("participant_id")
        .size()
        .clip(upper=_MAX_FORMAL_PARTICIPANT_REUSE)
        .sum()
    )
    formal_total = len(DOMAIN_NAMES) * _FORMAL_PER_DOMAIN
    if participant_capacity < formal_total:
        raise RuntimeError(
            "No feasible formal sample under the global participant cap: "
            f"capacity={participant_capacity}, required={formal_total}"
        )

    domain_order = sorted(
        DOMAIN_NAMES, key=lambda domain: (int(counts.loc[domain]), domain)
    )
    for attempt in range(_MAX_SELECTION_ATTEMPTS):
        rng = np.random.default_rng(_numpy_seed(validated_seed, attempt))
        participant_uses: dict[int, int] = {}
        training_parts: list[pd.DataFrame] = []
        formal_parts: list[pd.DataFrame] = []
        feasible = True

        for domain in domain_order:
            domain_candidates = candidates.loc[
                candidates["domain"].eq(domain)
            ].copy()
            domain_candidates["_seeded_order"] = rng.random(
                len(domain_candidates)
            )
            domain_candidates = domain_candidates.sort_values(
                [
                    "_seeded_order",
                    "participant_id",
                    "source_order",
                    "candidate_key",
                ],
                kind="stable",
            )

            training = domain_candidates.iloc[[_TRAINING_PER_DOMAIN - 1]].copy()
            remaining = domain_candidates.drop(index=training.index)
            formal_indices: list[int] = []
            for index, row in remaining.iterrows():
                participant_id = int(row["participant_id"])
                prior_uses = participant_uses.get(participant_id, 0)
                if prior_uses >= _MAX_FORMAL_PARTICIPANT_REUSE:
                    continue
                formal_indices.append(index)
                participant_uses[participant_id] = prior_uses + 1
                if len(formal_indices) == _FORMAL_PER_DOMAIN:
                    break

            if len(formal_indices) != _FORMAL_PER_DOMAIN:
                feasible = False
                break
            training_parts.append(training)
            formal_parts.append(remaining.loc[formal_indices].copy())

        if not feasible:
            continue

        training = pd.concat(training_parts, ignore_index=True).drop(
            columns="_seeded_order"
        )
        formal = pd.concat(formal_parts, ignore_index=True).drop(
            columns="_seeded_order"
        )
        training = _stable_sample_order(training)
        formal = _stable_sample_order(formal)
        if not set(training["candidate_key"]).isdisjoint(
            formal["candidate_key"]
        ):
            raise RuntimeError("training and formal candidate_key values overlap")
        if (
            formal.groupby("participant_id").size().max()
            > _MAX_FORMAL_PARTICIPANT_REUSE
        ):
            raise RuntimeError("formal sample exceeds the global participant cap")

        audit = pd.DataFrame(
            {
                "domain": list(DOMAIN_NAMES),
                "eligible_n": [int(counts.loc[domain]) for domain in DOMAIN_NAMES],
                "training_n": [_TRAINING_PER_DOMAIN] * len(DOMAIN_NAMES),
                "formal_n": [_FORMAL_PER_DOMAIN] * len(DOMAIN_NAMES),
                "seed": [validated_seed] * len(DOMAIN_NAMES),
                "attempt": [attempt] * len(DOMAIN_NAMES),
            }
        )
        return EvidenceCardSample(training=training, formal=formal, audit=audit)

    raise RuntimeError(
        "No feasible formal sample under the global participant cap after "
        f"{_MAX_SELECTION_ATTEMPTS} deterministic seeded attempts"
    )


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


def _validate_scope(scope: Any) -> str:
    if not isinstance(scope, str) or scope not in _VALID_SAMPLE_SCOPES:
        raise ValueError("scope must be exactly 'training' or 'formal'")
    return scope


def _review_case_id(candidate_key: str, scope: str) -> str:
    """Return an opaque scope-specific ID derived from the frozen source hash."""

    validated_scope = _validate_scope(scope)
    if not isinstance(candidate_key, str) or not candidate_key.strip():
        raise ValueError("candidate_key must be nonblank text")
    payload = (
        f"{C5_CANONICAL_SHA256}|{validated_scope}|{candidate_key}"
    ).encode("utf-8")
    return "C5-" + hashlib.sha256(payload).hexdigest()


def _prohibited_reviewer_columns(columns: pd.Index) -> set[str]:
    prohibited: set[str] = set()
    for column in columns:
        name = str(column)
        folded = name.casefold()
        if (
            name in PROHIBITED_REVIEWER_COLUMNS
            or "phq" in folded
            or folded == "label"
            or folded.endswith("_label")
            or folded.startswith("label_")
        ):
            prohibited.add(name)
    return prohibited


def build_review_tables(
    selected: pd.DataFrame,
    participant_texts: dict[int, str],
    *,
    seed: int,
    scope: str = "formal",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build owner linkage and independently ordered blinded reviewer tables."""

    validated_seed = _validate_seed(seed)
    validated_scope = _validate_scope(scope)
    if not isinstance(selected, pd.DataFrame):
        raise TypeError("selected must be a pandas.DataFrame")
    missing_columns = [
        column for column in _REQUIRED_REVIEW_COLUMNS if column not in selected.columns
    ]
    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing_columns)
        )

    frame = load_and_validate_spans(selected, expected_sha256=None)
    frame["candidate_key"] = frame["candidate_key"].map(normalize_space)
    blank_candidate_mask = frame["candidate_key"].eq("")
    if blank_candidate_mask.any():
        raise ValueError("candidate_key contains blank values")
    if frame["candidate_key"].duplicated().any():
        raise ValueError("duplicate candidate_key values are not allowed")
    if len(frame) < 2:
        raise ValueError(
            "selected must contain at least two rows for different A/B orders"
        )

    owner_rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        participant_id = int(row.participant_id)
        if participant_id not in participant_texts:
            raise ValueError(
                f"Missing participant text for participant_id {participant_id}"
            )
        context_before, evidence_quote, context_after = locate_quote_context(
            participant_texts[participant_id], row.exact_quote
        )
        owner_rows.append(
            {
                "review_case_id": _review_case_id(
                    str(row.candidate_key), validated_scope
                ),
                "participant_id": participant_id,
                "source_order": int(row.source_order),
                "candidate_key": str(row.candidate_key),
                "model_domain": row.domain,
                "model_polarity": row.polarity,
                "model_exact_quote": row.exact_quote,
                "evidence_quote": evidence_quote,
                "context_before": context_before,
                "context_after": context_after,
                "sample_scope": validated_scope,
            }
        )

    owner = pd.DataFrame(owner_rows, columns=list(_OWNER_COLUMNS))
    if owner["review_case_id"].duplicated().any():
        raise ValueError("duplicate review_case_id values generated")

    reviewer = owner.loc[
        :, ["review_case_id", "evidence_quote", "context_before", "context_after"]
    ].copy()
    for answer_column in (
        "human_span_valid",
        "human_domain",
        "human_polarity",
        "notes",
    ):
        reviewer[answer_column] = ""
    reviewer = reviewer.loc[:, list(REVIEWER_COLUMNS)]

    prohibited = _prohibited_reviewer_columns(reviewer.columns)
    if prohibited:
        raise ValueError(
            "reviewer table contains prohibited columns: "
            + ", ".join(sorted(prohibited))
        )

    reviewer_a_order = np.random.default_rng(
        _numpy_seed(validated_seed)
    ).permutation(len(reviewer))
    reviewer_b_order = np.random.default_rng(
        _numpy_seed(validated_seed, 1)
    ).permutation(len(reviewer))
    if np.array_equal(reviewer_a_order, reviewer_b_order):
        reviewer_b_order = np.roll(reviewer_a_order, 1)

    reviewer_a = reviewer.iloc[reviewer_a_order].reset_index(drop=True)
    reviewer_b = reviewer.iloc[reviewer_b_order].reset_index(drop=True)
    if set(reviewer_a["review_case_id"]) != set(reviewer_b["review_case_id"]):
        raise RuntimeError("reviewer A/B review_case_id sets differ")
    if reviewer_a["review_case_id"].tolist() == reviewer_b[
        "review_case_id"
    ].tolist():
        raise RuntimeError("reviewer A/B row orders must differ")
    return owner, reviewer_a, reviewer_b


def _answer_fields_blank(frame: pd.DataFrame) -> bool:
    answer_fields = [
        "human_span_valid",
        "human_domain",
        "human_polarity",
        "notes",
    ]
    return set(answer_fields).issubset(frame.columns) and bool(
        frame.loc[:, answer_fields].eq("").all().all()
    )


def _reviewer_scope_checks(
    owner: pd.DataFrame,
    reviewer_a: pd.DataFrame,
    reviewer_b: pd.DataFrame,
    *,
    scope: str,
    expected_total: int,
    expected_per_domain: int,
) -> dict[str, bool]:
    for reviewer_name, reviewer in (
        ("A", reviewer_a),
        ("B", reviewer_b),
    ):
        if list(reviewer.columns) != list(REVIEWER_COLUMNS):
            raise RuntimeError(
                f"reviewer {reviewer_name} {scope} schema changed"
            )
        if len(reviewer) != expected_total:
            raise RuntimeError(
                f"reviewer {reviewer_name} {scope} row count changed"
            )
        if reviewer["review_case_id"].duplicated().any():
            raise RuntimeError(
                f"reviewer {reviewer_name} {scope} IDs are not unique"
            )
        prohibited = _prohibited_reviewer_columns(reviewer.columns)
        if prohibited:
            raise RuntimeError(
                f"reviewer {reviewer_name} {scope} has prohibited fields: "
                + ", ".join(sorted(prohibited))
            )
        if not _answer_fields_blank(reviewer):
            raise RuntimeError(
                f"reviewer {reviewer_name} {scope} answer fields are not blank"
            )

    if list(owner.columns) != list(_OWNER_COLUMNS):
        raise RuntimeError(f"owner {scope} schema changed")
    if len(owner) != expected_total:
        raise RuntimeError(f"owner {scope} row count changed")
    if not owner["sample_scope"].eq(scope).all():
        raise RuntimeError(f"owner {scope} scope labels changed")
    counts = (
        owner.groupby("model_domain", sort=False)
        .size()
        .reindex(DOMAIN_NAMES, fill_value=0)
    )
    if not counts.eq(expected_per_domain).all():
        raise RuntimeError(
            f"owner {scope} domain quotas changed: "
            + _domain_count_text(counts)
        )

    owner_ids = set(owner["review_case_id"])
    reviewer_a_ids = set(reviewer_a["review_case_id"])
    reviewer_b_ids = set(reviewer_b["review_case_id"])
    id_sets_equal = owner_ids == reviewer_a_ids == reviewer_b_ids
    orders_differ = reviewer_a["review_case_id"].tolist() != reviewer_b[
        "review_case_id"
    ].tolist()
    if not id_sets_equal:
        raise RuntimeError(f"reviewer A/B {scope} ID sets differ")
    if not orders_differ:
        raise RuntimeError(f"reviewer A/B {scope} orders do not differ")
    return {
        "id_sets_equal": id_sets_equal,
        "orders_differ": orders_differ,
    }


def _validate_package_data(
    data: _EvidenceCardPackageData,
) -> dict[str, dict[str, bool]]:
    reviewer_checks = {
        "training": _reviewer_scope_checks(
            data.training_owner,
            data.training_reviewer_a,
            data.training_reviewer_b,
            scope="training",
            expected_total=len(DOMAIN_NAMES) * _TRAINING_PER_DOMAIN,
            expected_per_domain=_TRAINING_PER_DOMAIN,
        ),
        "formal": _reviewer_scope_checks(
            data.formal_owner,
            data.formal_reviewer_a,
            data.formal_reviewer_b,
            scope="formal",
            expected_total=len(DOMAIN_NAMES) * _FORMAL_PER_DOMAIN,
            expected_per_domain=_FORMAL_PER_DOMAIN,
        ),
    }
    overlap = set(data.training_owner["candidate_key"]).intersection(
        data.formal_owner["candidate_key"]
    )
    if overlap:
        raise RuntimeError("training and formal candidate_key values overlap")
    formal_reuse = data.formal_owner.groupby("participant_id").size()
    if formal_reuse.empty or int(formal_reuse.max()) > _MAX_FORMAL_PARTICIPANT_REUSE:
        raise RuntimeError("formal sample exceeds the participant reuse cap")
    if len(data.training_owner) + len(data.formal_owner) != 130:
        raise RuntimeError("selected evidence-card total is not 130")
    return reviewer_checks


def _preflight_evidence_card_package(
    canonical_path: Path,
    participant_text_path: Path,
    *,
    seed: int,
    expected_canonical_sha256: str,
    expected_participant_text_sha256: str,
    expected_row_count: int,
) -> _EvidenceCardPackageData:
    validated_seed = _validate_seed(seed)
    _validate_sha256(canonical_path, expected_canonical_sha256)
    _validate_sha256(participant_text_path, expected_participant_text_sha256)

    spans = pd.read_csv(canonical_path)
    spans = load_and_validate_spans(spans, expected_sha256=None)
    if len(spans) != expected_row_count:
        raise ValueError(
            "canonical row count mismatch: "
            f"expected {expected_row_count}, found {len(spans)}"
        )
    sample = select_evidence_cards(spans, seed=validated_seed)
    participant_texts = load_participant_texts(
        participant_text_path,
        expected_sha256=expected_participant_text_sha256,
    )
    training_owner, training_reviewer_a, training_reviewer_b = (
        build_review_tables(
            sample.training,
            participant_texts,
            seed=validated_seed,
            scope="training",
        )
    )
    formal_owner, formal_reviewer_a, formal_reviewer_b = build_review_tables(
        sample.formal,
        participant_texts,
        seed=validated_seed,
        scope="formal",
    )
    data = _EvidenceCardPackageData(
        sample=sample,
        training_owner=training_owner,
        training_reviewer_a=training_reviewer_a,
        training_reviewer_b=training_reviewer_b,
        formal_owner=formal_owner,
        formal_reviewer_a=formal_reviewer_a,
        formal_reviewer_b=formal_reviewer_b,
    )
    _validate_package_data(data)
    return data


def _reviewer_readme_text(reviewer: str) -> str:
    return f"""# C5 证据卡评分者 {reviewer} 培训说明

## 当前阶段

本目录仅用于培训。请先独立完成培训材料，再参加编码手册讨论；正式材料在编码手册冻结后另行发放。

## 隐私放行门槛

引文与上下文仅经过启发式自动去标识，不能据此认定隐私处理完整。负责人逐卡人工审查全部培训与正式证据卡并明确审批；审批通过前不得发放任何评分者材料。该门槛由负责人完成，不增加评分者的判断任务。

## 三项必填判断

每张证据卡只需作出以下三项必填判断：

1. `human_span_valid`（有效性）：判断引文是否为上下文支持的有效证据。
2. `human_domain`（领域）：判断证据所属领域。
3. `human_polarity`（极性）：判断证据极性。

`notes` 仅用于必要说明，不属于第四项必填判断。不得推断参与者标签、量表分数或模型结果。

## 返回方式

后续请在项目负责人发放的 XLSX 工作簿中填写并返回。当前 JSON 仅是程序中间表，请不要编辑，也不要返回 JSON。
"""


def _owner_readme_text() -> str:
    return """# C5 证据卡 OWNER_ONLY 管理说明

## 发放顺序

自动去标识不足以放行。负责人必须先对全部 130 张入选证据卡的引文与上下文逐卡人工审查，并为每个 `review_case_id` 明确记录审批值；任何一张未审批时，培训与正式评分者材料均不得发放。

隐私审批通过后，才可先发放培训材料；评分者应先完成培训，再讨论边界并冻结编码手册，之后才能发放正式材料。培训答案不计入正式一致性或效度统计。

## 审批文件

`evidence_card_owner_privacy_approval_template.json` 仅存放于 OWNER_ONLY。负责人应复制模板，在逐卡审查后把对应的 `owner_privacy_approved` 明确改为 `true`，再运行程序化放行检查。模板中的默认 `false` 表示继续阻断发放，不代表已完成审查。

## 正式材料暂缓发放

正式 JSON 与后续生成的正式 XLSX 工作簿在负责人隐私审批通过且编码手册冻结前必须保留在 OWNER_ONLY，不得复制到评分者培训目录，也不得提前发放。

## 当前状态

本目录处于材料准备阶段，尚未完成人工评分、复核、一致性计算或效度结论。负责人应保留清单与文件哈希，并按冻结后的流程发放正式工作簿。
"""


def _write_utf8_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = text.rstrip() + "\n"
    path.write_bytes(payload.encode("utf-8"))


def _write_json_records(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        frame.to_dict(orient="records"),
        ensure_ascii=False,
        indent=2,
    )
    path.write_bytes((payload + "\n").encode("utf-8"))


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False, lineterminator="\n")


def _write_json_object(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    path.write_bytes((payload + "\n").encode("utf-8"))


def _reviewer_payload_sha256(record: dict[str, Any]) -> str:
    payload: dict[str, str] = {}
    for field in _PRIVACY_PAYLOAD_COLUMNS:
        value = record.get(field)
        if not isinstance(value, str):
            raise ValueError(
                f"reviewer privacy payload field must be text: {field}"
            )
        payload[field] = value
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _scope_privacy_payload_hashes(
    reviewer_a: pd.DataFrame,
    reviewer_b: pd.DataFrame,
    *,
    scope: str,
    expected_total: int,
) -> dict[str, str]:
    records_by_reviewer: list[dict[str, dict[str, Any]]] = []
    for reviewer_name, frame in (("A", reviewer_a), ("B", reviewer_b)):
        if list(frame.columns) != list(REVIEWER_COLUMNS):
            raise ValueError(
                f"reviewer {reviewer_name} {scope} schema contains prohibited fields"
            )
        if len(frame) != expected_total:
            raise ValueError(f"reviewer {reviewer_name} {scope} row count changed")
        if frame["review_case_id"].duplicated().any():
            raise ValueError(
                f"reviewer {reviewer_name} {scope} contains duplicate IDs"
            )
        if _prohibited_reviewer_columns(frame.columns):
            raise ValueError(
                f"reviewer {reviewer_name} {scope} contains prohibited fields"
            )
        if not _answer_fields_blank(frame):
            raise ValueError(
                f"reviewer {reviewer_name} {scope} answer fields are not blank"
            )
        records_by_reviewer.append(
            {
                str(record["review_case_id"]): record
                for record in frame.to_dict(orient="records")
            }
        )

    reviewer_a_records, reviewer_b_records = records_by_reviewer
    if set(reviewer_a_records) != set(reviewer_b_records):
        raise ValueError(f"reviewer A/B {scope} ID sets differ")
    payload_hashes: dict[str, str] = {}
    for review_case_id in sorted(reviewer_a_records):
        reviewer_a_hash = _reviewer_payload_sha256(
            reviewer_a_records[review_case_id]
        )
        reviewer_b_hash = _reviewer_payload_sha256(
            reviewer_b_records[review_case_id]
        )
        if reviewer_a_hash != reviewer_b_hash:
            raise ValueError(
                f"reviewer A/B {scope} reviewed text payloads differ"
            )
        payload_hashes[review_case_id] = reviewer_a_hash
    return payload_hashes


def _privacy_payload_hashes(
    training_reviewer_a: pd.DataFrame,
    training_reviewer_b: pd.DataFrame,
    formal_reviewer_a: pd.DataFrame,
    formal_reviewer_b: pd.DataFrame,
) -> dict[str, str]:
    training_hashes = _scope_privacy_payload_hashes(
        training_reviewer_a,
        training_reviewer_b,
        scope="training",
        expected_total=len(DOMAIN_NAMES) * _TRAINING_PER_DOMAIN,
    )
    formal_hashes = _scope_privacy_payload_hashes(
        formal_reviewer_a,
        formal_reviewer_b,
        scope="formal",
        expected_total=len(DOMAIN_NAMES) * _FORMAL_PER_DOMAIN,
    )
    overlap = set(training_hashes).intersection(formal_hashes)
    if overlap:
        raise ValueError("training and formal privacy approval IDs overlap")
    payload_hashes = {**training_hashes, **formal_hashes}
    if len(payload_hashes) != 130:
        raise ValueError("privacy approval must cover exactly 130 reviewer IDs")
    return payload_hashes


def _owner_privacy_approval_template(
    data: _EvidenceCardPackageData,
) -> dict[str, Any]:
    payload_hashes = _privacy_payload_hashes(
        data.training_reviewer_a,
        data.training_reviewer_b,
        data.formal_reviewer_a,
        data.formal_reviewer_b,
    )
    case_order = data.training_owner["review_case_id"].tolist() + data.formal_owner[
        "review_case_id"
    ].tolist()
    if len(case_order) != 130 or len(set(case_order)) != 130:
        raise ValueError("owner privacy approval case order must contain 130 unique IDs")
    if set(case_order) != set(payload_hashes):
        raise ValueError("owner and reviewer privacy approval ID sets differ")
    return {
        "schema": OWNER_PRIVACY_APPROVAL_SCHEMA,
        "automatic_redaction_sufficient_for_release": False,
        "manual_owner_audit_required": True,
        "required_case_count": 130,
        "cases": [
            {
                "review_case_id": review_case_id,
                "reviewer_payload_sha256": payload_hashes[review_case_id],
                "owner_privacy_approved": False,
            }
            for review_case_id in case_order
        ],
    }


def _relative_to_package(paths: EvidenceCardPackagePaths, path: Path) -> str:
    return path.relative_to(paths.root).as_posix()


def _non_manifest_file_map(
    paths: EvidenceCardPackagePaths,
) -> dict[str, Path]:
    files = (
        paths.reviewer_a_training_readme,
        paths.reviewer_a_training_json,
        paths.reviewer_b_training_readme,
        paths.reviewer_b_training_json,
        paths.owner_readme,
        paths.owner_crosswalk_csv,
        paths.sampling_audit_csv,
        paths.owner_privacy_approval_template_json,
        paths.reviewer_a_formal_json,
        paths.reviewer_b_formal_json,
    )
    return {
        _relative_to_package(paths, path): path
        for path in files
    }


def _file_hash_inventory(paths: EvidenceCardPackagePaths) -> dict[str, str]:
    inventory = _non_manifest_file_map(paths)
    missing = [relative for relative, path in inventory.items() if not path.is_file()]
    if missing:
        raise RuntimeError(
            "package files missing before hashing: " + ", ".join(sorted(missing))
        )
    return {
        relative: sha256_file(inventory[relative])
        for relative in sorted(inventory)
    }


def _manifest_for_package(
    data: _EvidenceCardPackageData,
    canonical_path: Path,
    participant_text_path: Path,
    *,
    seed: int,
    canonical_sha256: str,
    participant_text_sha256: str,
    input_row_count: int,
    file_hashes: dict[str, str],
    manifest_relative_path: str,
    privacy_approval_template_relative_path: str,
) -> dict[str, Any]:
    reviewer_checks = _validate_package_data(data)
    training_counts = (
        data.training_owner.groupby("model_domain", sort=False)
        .size()
        .reindex(DOMAIN_NAMES, fill_value=0)
    )
    formal_counts = (
        data.formal_owner.groupby("model_domain", sort=False)
        .size()
        .reindex(DOMAIN_NAMES, fill_value=0)
    )
    formal_reuse = data.formal_owner.groupby("participant_id").size()
    prohibited_found = sorted(
        set().union(
            *(
                _prohibited_reviewer_columns(frame.columns)
                for frame in (
                    data.training_reviewer_a,
                    data.training_reviewer_b,
                    data.formal_reviewer_a,
                    data.formal_reviewer_b,
                )
            )
        )
    )
    all_answers_blank = all(
        _answer_fields_blank(frame)
        for frame in (
            data.training_reviewer_a,
            data.training_reviewer_b,
            data.formal_reviewer_a,
            data.formal_reviewer_b,
        )
    )
    return {
        "status": "prepared_not_scored",
        "design": "balanced_evidence_cards",
        "canonical_path": str(canonical_path),
        "participant_text_path": str(participant_text_path),
        "canonical_sha256": canonical_sha256,
        "participant_text_sha256": participant_text_sha256,
        "seed": seed,
        "input_row_count": input_row_count,
        "training_total": int(len(data.training_owner)),
        "formal_total": int(len(data.formal_owner)),
        "training_per_domain": {
            domain: int(training_counts.loc[domain]) for domain in DOMAIN_NAMES
        },
        "formal_per_domain": {
            domain: int(formal_counts.loc[domain]) for domain in DOMAIN_NAMES
        },
        "formal_max_participant_reuse": int(formal_reuse.max()),
        "training_formal_key_overlap_count": len(
            set(data.training_owner["candidate_key"]).intersection(
                data.formal_owner["candidate_key"]
            )
        ),
        "reviewer_id_checks": reviewer_checks,
        "reviewer_prohibited_field_check": {
            "passed": not prohibited_found,
            "prohibited_fields_found": prohibited_found,
        },
        "all_answer_fields_blank": all_answers_blank,
        "reviewer_material_release_status": OWNER_PRIVACY_RELEASE_BLOCKED,
        "automatic_redaction_sufficient_for_release": False,
        "owner_manual_privacy_audit_required": True,
        "owner_privacy_required_case_count": 130,
        "owner_privacy_approval_template": (
            privacy_approval_template_relative_path
        ),
        "privacy_release_policy_zh": (
            "自动去标识仅为启发式处理，不能证明隐私处理完整。负责人必须对"
            "全部 130 张入选证据卡的引文与上下文逐卡人工审查，并为每个"
            " review_case_id 明确审批；全部通过前不得发放任何评分者材料。"
        ),
        "formal_materials_owner_only_until_codebook_freeze": True,
        "training_release_policy_zh": (
            "培训材料仅可在全部 130 张证据卡通过负责人逐卡隐私审批后发放。"
        ),
        "formal_release_policy_zh": (
            "正式工作簿与正式 JSON 在负责人逐卡隐私审批全部通过且编码手册"
            "冻结前仅保存在 OWNER_ONLY；两个门槛均满足后方可发放。"
        ),
        "human_review_completed": False,
        "agreement_completed": False,
        "workbooks_generated": False,
        "file_hash_algorithm": "sha256",
        "file_hashes": file_hashes,
        "file_hashes_excludes": [manifest_relative_path],
        "file_hashes_exclusion_reason_zh": (
            "清单包含其他文件的哈希，无法稳定包含自身哈希，因此明确排除清单自身。"
        ),
    }


def _load_reviewer_json(path: Path) -> pd.DataFrame:
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid reviewer JSON: {path}") from exc
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"reviewer JSON must be a nonempty record list: {path}")
    if any(
        not isinstance(record, dict)
        or set(record) != set(REVIEWER_COLUMNS)
        for record in records
    ):
        raise RuntimeError(f"reviewer JSON schema changed: {path}")
    return pd.DataFrame(records, columns=list(REVIEWER_COLUMNS))


def _selected_owner_case_ids(path: Path) -> set[str]:
    try:
        owner = pd.read_csv(path, keep_default_na=False)
    except (OSError, pd.errors.ParserError) as exc:
        raise ValueError("owner crosswalk is unreadable") from exc
    expected_scope_order = (
        ["training"] * (len(DOMAIN_NAMES) * _TRAINING_PER_DOMAIN)
        + ["formal"] * (len(DOMAIN_NAMES) * _FORMAL_PER_DOMAIN)
    )
    if (
        list(owner.columns) != list(_OWNER_COLUMNS)
        or len(owner) != len(expected_scope_order)
        or owner["sample_scope"].tolist() != expected_scope_order
    ):
        raise ValueError("owner crosswalk schema, count, or scope order is invalid")
    review_case_ids = owner["review_case_id"].tolist()
    if len(set(review_case_ids)) != len(review_case_ids):
        raise ValueError("owner crosswalk contains duplicate selected IDs")
    if any(
        not isinstance(review_case_id, str)
        or not re.fullmatch(r"C5-[0-9a-f]{64}", review_case_id)
        for review_case_id in review_case_ids
    ):
        raise ValueError("owner crosswalk contains an invalid selected ID")
    return set(review_case_ids)


def _coerce_package_paths(
    package: Path | EvidenceCardPackagePaths,
) -> EvidenceCardPackagePaths:
    if isinstance(package, EvidenceCardPackagePaths):
        return package
    return EvidenceCardPackagePaths(Path(package).expanduser().resolve())


def check_owner_privacy_release(
    package: Path | EvidenceCardPackagePaths,
    *,
    approval_path: Path | None = None,
) -> OwnerPrivacyReleaseStatus:
    """Fail closed unless the owner approved every exact reviewer payload.

    Automatic redaction is deliberately never treated as sufficient.  The
    approval JSON must remain under ``OWNER_ONLY`` and contain one explicit
    boolean decision for each of the 130 opaque reviewer IDs.  Per-case payload
    hashes bind approval to the exact quote and context inspected by the owner.
    """

    paths = _coerce_package_paths(package)
    approval = Path(
        approval_path
        if approval_path is not None
        else paths.owner_privacy_approval_template_json
    ).expanduser().resolve()
    owner_dir = paths.owner_dir.resolve()
    try:
        approval.relative_to(owner_dir)
    except ValueError as exc:
        raise ValueError(
            "owner privacy approval artifact must remain under OWNER_ONLY"
        ) from exc

    try:
        raw_approval = json.loads(approval.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("owner privacy approval artifact is unreadable") from exc
    required_top_level = {
        "schema",
        "automatic_redaction_sufficient_for_release",
        "manual_owner_audit_required",
        "required_case_count",
        "cases",
    }
    if not isinstance(raw_approval, dict) or set(raw_approval) != required_top_level:
        raise ValueError("owner privacy approval schema is invalid")
    if raw_approval["schema"] != OWNER_PRIVACY_APPROVAL_SCHEMA:
        raise ValueError("owner privacy approval schema version is invalid")
    if raw_approval["automatic_redaction_sufficient_for_release"] is not False:
        raise ValueError("automatic redaction cannot authorize privacy release")
    if raw_approval["manual_owner_audit_required"] is not True:
        raise ValueError("manual owner privacy audit must remain required")

    training_a = _load_reviewer_json(paths.reviewer_a_training_json)
    training_b = _load_reviewer_json(paths.reviewer_b_training_json)
    formal_a = _load_reviewer_json(paths.reviewer_a_formal_json)
    formal_b = _load_reviewer_json(paths.reviewer_b_formal_json)
    expected_hashes = _privacy_payload_hashes(
        training_a,
        training_b,
        formal_a,
        formal_b,
    )
    expected_ids = set(expected_hashes)
    expected_count = len(expected_ids)
    selected_owner_ids = _selected_owner_case_ids(paths.owner_crosswalk_csv)
    if selected_owner_ids != expected_ids:
        raise ValueError(
            "owner crosswalk selected IDs differ from reviewer material IDs"
        )
    if raw_approval["required_case_count"] != expected_count:
        raise ValueError("owner privacy approval required case count is invalid")
    cases = raw_approval["cases"]
    if not isinstance(cases, list):
        raise ValueError("owner privacy approval cases must be a list")

    approvals_by_id: dict[str, bool] = {}
    payload_hashes_by_id: dict[str, str] = {}
    for case in cases:
        if not isinstance(case, dict) or set(case) != _OWNER_PRIVACY_CASE_FIELDS:
            raise ValueError("owner privacy approval case schema is invalid")
        review_case_id = case["review_case_id"]
        payload_sha256 = case["reviewer_payload_sha256"]
        approved = case["owner_privacy_approved"]
        if not isinstance(review_case_id, str) or not re.fullmatch(
            r"C5-[0-9a-f]{64}", review_case_id
        ):
            raise ValueError("owner privacy approval case ID is invalid")
        if review_case_id in approvals_by_id:
            raise ValueError("owner privacy approval contains duplicate case IDs")
        if not isinstance(payload_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", payload_sha256
        ):
            raise ValueError("owner privacy approval payload hash is invalid")
        if not isinstance(approved, bool):
            raise ValueError(
                "owner privacy approval requires an explicit boolean per case"
            )
        approvals_by_id[review_case_id] = approved
        payload_hashes_by_id[review_case_id] = payload_sha256

    approval_ids = set(approvals_by_id)
    if approval_ids != expected_ids:
        raise ValueError(
            "owner privacy approval case coverage has missing or extra IDs"
        )
    mismatched_payloads = [
        review_case_id
        for review_case_id in expected_ids
        if payload_hashes_by_id[review_case_id]
        != expected_hashes[review_case_id]
    ]
    if mismatched_payloads:
        raise ValueError(
            "owner privacy approval payload hash does not match reviewed text"
        )

    approved_count = sum(approvals_by_id.values())
    release_allowed = approved_count == expected_count
    return OwnerPrivacyReleaseStatus(
        status=(
            OWNER_PRIVACY_RELEASE_APPROVED
            if release_allowed
            else OWNER_PRIVACY_RELEASE_BLOCKED
        ),
        release_allowed=release_allowed,
        expected_case_count=expected_count,
        approved_case_count=approved_count,
        automatic_redaction_sufficient_for_release=False,
    )


def _verify_written_package(
    paths: EvidenceCardPackagePaths,
    expected_manifest: dict[str, Any],
) -> None:
    manifest_relative = _relative_to_package(paths, paths.manifest_json)
    expected_files = set(_non_manifest_file_map(paths)) | {manifest_relative}
    actual_files = {
        path.relative_to(paths.root).as_posix()
        for path in paths.root.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise RuntimeError(
            "package file inventory mismatch: expected "
            f"{sorted(expected_files)}, found {sorted(actual_files)}"
        )

    actual_manifest = json.loads(paths.manifest_json.read_text(encoding="utf-8"))
    if actual_manifest != expected_manifest:
        raise RuntimeError("written package manifest does not match memory")
    actual_hashes = _file_hash_inventory(paths)
    if actual_hashes != expected_manifest["file_hashes"]:
        raise RuntimeError("written package file hash verification failed")

    training_a = _load_reviewer_json(paths.reviewer_a_training_json)
    training_b = _load_reviewer_json(paths.reviewer_b_training_json)
    formal_a = _load_reviewer_json(paths.reviewer_a_formal_json)
    formal_b = _load_reviewer_json(paths.reviewer_b_formal_json)
    for scope, reviewer_a, reviewer_b, expected_total in (
        ("training", training_a, training_b, 10),
        ("formal", formal_a, formal_b, 120),
    ):
        if len(reviewer_a) != expected_total or len(reviewer_b) != expected_total:
            raise RuntimeError(f"written {scope} reviewer row count changed")
        if set(reviewer_a["review_case_id"]) != set(
            reviewer_b["review_case_id"]
        ):
            raise RuntimeError(f"written {scope} reviewer ID sets differ")
        if reviewer_a["review_case_id"].tolist() == reviewer_b[
            "review_case_id"
        ].tolist():
            raise RuntimeError(f"written {scope} reviewer orders do not differ")
        if not _answer_fields_blank(reviewer_a) or not _answer_fields_blank(
            reviewer_b
        ):
            raise RuntimeError(f"written {scope} reviewer answers are not blank")

    owner = pd.read_csv(paths.owner_crosswalk_csv, keep_default_na=False)
    if list(owner.columns) != list(_OWNER_COLUMNS) or len(owner) != 130:
        raise RuntimeError("written owner crosswalk schema or row count changed")
    if owner["sample_scope"].tolist() != ["training"] * 10 + ["formal"] * 120:
        raise RuntimeError("written owner crosswalk scope order changed")
    audit = pd.read_csv(paths.sampling_audit_csv)
    if list(audit.columns) != [
        "domain",
        "eligible_n",
        "training_n",
        "formal_n",
        "seed",
        "attempt",
    ] or audit["domain"].tolist() != list(DOMAIN_NAMES):
        raise RuntimeError("written sampling audit schema or order changed")

    privacy_status = check_owner_privacy_release(paths)
    if (
        privacy_status.status != OWNER_PRIVACY_RELEASE_BLOCKED
        or privacy_status.release_allowed
        or privacy_status.expected_case_count != 130
        or privacy_status.approved_case_count != 0
        or privacy_status.automatic_redaction_sufficient_for_release
    ):
        raise RuntimeError("new package must fail closed pending owner privacy audit")


def _staging_path(output_root: Path) -> Path:
    if not output_root.name:
        raise ValueError("output_root must name a directory")
    return output_root.with_name(f".{output_root.name}.staging")


def _path_entry_exists(path: Path) -> bool:
    """Return true for any directory entry, including a broken symlink."""

    return path.exists() or path.is_symlink()


def _promote_staging_no_overwrite(staging: Path, output_root: Path) -> None:
    """Atomically rename on Windows without replacing an existing destination.

    ``os.rename`` uses the Windows wide-character filesystem API and fails when
    the destination already exists.  The precheck gives a clear early error;
    the rename itself is the race-safe, fail-closed operation.
    """

    if _path_entry_exists(output_root):
        raise FileExistsError(
            f"output_root appeared during build: {output_root}"
        )
    os.rename(os.fspath(staging), os.fspath(output_root))


def _remove_expected_staging(staging: Path, output_root: Path) -> None:
    expected = _staging_path(output_root)
    if staging != expected or staging.parent != output_root.parent:
        raise RuntimeError("refusing to remove an unexpected staging path")
    if staging == output_root:
        raise RuntimeError("staging path must differ from output_root")
    if staging.is_symlink():
        staging.unlink()
    elif staging.exists():
        shutil.rmtree(staging)


def build_evidence_card_package(
    canonical_path: Path,
    participant_text_path: Path,
    output_root: Path,
    seed: int,
) -> EvidenceCardPackagePaths:
    """Atomically prepare the restricted, unscored evidence-card package."""

    canonical = Path(canonical_path).expanduser().resolve()
    participant_text = Path(participant_text_path).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    staging = _staging_path(output)
    if _path_entry_exists(output):
        raise FileExistsError(f"output_root already exists: {output}")
    if _path_entry_exists(staging):
        raise FileExistsError(f"staging directory already exists: {staging}")

    validated_seed = _validate_seed(seed)
    canonical_sha256 = C5_CANONICAL_SHA256
    participant_text_sha256 = PARTICIPANT_TEXT_SHA256
    expected_row_count = C5_CANONICAL_ROW_COUNT
    data = _preflight_evidence_card_package(
        canonical,
        participant_text,
        seed=validated_seed,
        expected_canonical_sha256=canonical_sha256,
        expected_participant_text_sha256=participant_text_sha256,
        expected_row_count=expected_row_count,
    )

    staging_created = False
    staging_paths = EvidenceCardPackagePaths(staging)
    try:
        staging.mkdir(parents=True, exist_ok=False)
        staging_created = True
        _write_utf8_text(
            staging_paths.reviewer_a_training_readme,
            _reviewer_readme_text("A"),
        )
        _write_utf8_text(
            staging_paths.reviewer_b_training_readme,
            _reviewer_readme_text("B"),
        )
        _write_utf8_text(staging_paths.owner_readme, _owner_readme_text())
        _write_json_records(
            data.training_reviewer_a,
            staging_paths.reviewer_a_training_json,
        )
        _write_json_records(
            data.training_reviewer_b,
            staging_paths.reviewer_b_training_json,
        )
        _write_json_records(
            data.formal_reviewer_a,
            staging_paths.reviewer_a_formal_json,
        )
        _write_json_records(
            data.formal_reviewer_b,
            staging_paths.reviewer_b_formal_json,
        )
        privacy_approval_template = _owner_privacy_approval_template(data)
        _write_json_object(
            privacy_approval_template,
            staging_paths.owner_privacy_approval_template_json,
        )
        owner_crosswalk = pd.concat(
            [data.training_owner, data.formal_owner], ignore_index=True
        ).loc[:, list(_OWNER_COLUMNS)]
        _write_csv(owner_crosswalk, staging_paths.owner_crosswalk_csv)
        _write_csv(data.sample.audit, staging_paths.sampling_audit_csv)

        file_hashes = _file_hash_inventory(staging_paths)
        manifest_relative = _relative_to_package(
            staging_paths, staging_paths.manifest_json
        )
        privacy_approval_template_relative = _relative_to_package(
            staging_paths,
            staging_paths.owner_privacy_approval_template_json,
        )
        manifest = _manifest_for_package(
            data,
            canonical,
            participant_text,
            seed=validated_seed,
            canonical_sha256=canonical_sha256,
            participant_text_sha256=participant_text_sha256,
            input_row_count=expected_row_count,
            file_hashes=file_hashes,
            manifest_relative_path=manifest_relative,
            privacy_approval_template_relative_path=(
                privacy_approval_template_relative
            ),
        )
        _write_json_object(manifest, staging_paths.manifest_json)
        _verify_written_package(staging_paths, manifest)
        _promote_staging_no_overwrite(staging, output)
        staging_created = False
    except BaseException:
        if staging_created:
            _remove_expected_staging(staging, output)
        raise

    return EvidenceCardPackagePaths(output)
