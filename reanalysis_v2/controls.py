from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd


DOMAIN_ORDER = [
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
]


def length_matched_sample(text: str, *, target_words: int, seed: int) -> str:
    tokens = str(text or "").split()
    if target_words < 0:
        raise ValueError("target_words must be non-negative")
    if target_words > len(tokens):
        raise ValueError(
            f"cannot sample {target_words} words from a {len(tokens)}-word source"
        )
    if target_words == 0:
        return ""
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(tokens), size=target_words, replace=False))
    return " ".join(tokens[index] for index in selected)


def exclude_exact_quote_tokens(
    source_text: str, exact_quotes: Iterable[str]
) -> tuple[str, dict[str, int]]:
    tokens = str(source_text or "").split()
    excluded = np.zeros(len(tokens), dtype=bool)
    matched_quotes = 0
    unmatched_quotes = 0
    for quote in exact_quotes:
        quote_tokens = str(quote or "").split()
        if not quote_tokens:
            continue
        match_start = None
        for start in range(0, len(tokens) - len(quote_tokens) + 1):
            stop = start + len(quote_tokens)
            if tokens[start:stop] == quote_tokens:
                match_start = start
                break
        if match_start is None:
            unmatched_quotes += 1
            continue
        excluded[match_start : match_start + len(quote_tokens)] = True
        matched_quotes += 1
    remaining = " ".join(token for token, remove in zip(tokens, excluded) if not remove)
    return remaining, {
        "matched_quotes": matched_quotes,
        "unmatched_quotes": unmatched_quotes,
        "excluded_word_count": int(excluded.sum()),
        "remaining_word_count": int((~excluded).sum()),
    }


def build_domain_features(
    spans: pd.DataFrame, participant_ids: Sequence[int | str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"participant_id", "domain"}
    if not required.issubset(spans.columns):
        raise ValueError("spans require participant_id and domain")
    unknown = set(spans["domain"].dropna().astype(str)).difference(DOMAIN_ORDER)
    if unknown:
        raise ValueError(f"unknown C5 domains: {sorted(unknown)}")
    base = pd.DataFrame({"participant_id": list(participant_ids)})
    counts = (
        spans.groupby(["participant_id", "domain"]).size().unstack(fill_value=0)
        if not spans.empty
        else pd.DataFrame()
    )
    counts = counts.reindex(columns=DOMAIN_ORDER, fill_value=0)
    counts.index.name = "participant_id"
    counts = base.merge(counts.reset_index(), on="participant_id", how="left")
    counts[DOMAIN_ORDER] = counts[DOMAIN_ORDER].fillna(0).astype(int)
    presence = counts.copy()
    presence[DOMAIN_ORDER] = presence[DOMAIN_ORDER].gt(0).astype(int)
    return presence, counts


def keyword_mask(text: str, keywords: Sequence[str]) -> tuple[str, int]:
    escaped = [re.escape(keyword) for keyword in keywords if keyword]
    if not escaped:
        return str(text), 0
    pattern = re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)
    return pattern.subn("[MASKED]", str(text or ""))


def position_matched_turns(turns: pd.DataFrame, *, target_words: int) -> str:
    required = {
        "turn_index",
        "normalized_spoken_text",
        "is_protocol_template_spoken",
    }
    if not required.issubset(turns.columns):
        raise ValueError(f"turn table missing columns: {sorted(required - set(turns.columns))}")
    if target_words < 0:
        raise ValueError("target_words must be non-negative")
    if target_words == 0:
        return ""
    ordered = turns.sort_values("turn_index", kind="stable").copy()
    template_positions = ordered.loc[
        ordered["is_protocol_template_spoken"].astype(int).eq(1), "turn_index"
    ].to_numpy(dtype=float)
    candidates = ordered.loc[
        ordered["is_protocol_template_spoken"].astype(int).eq(0)
        & ordered["normalized_spoken_text"].fillna("").astype(str).str.strip().ne("")
    ].copy()
    if not len(template_positions):
        template_positions = np.array([float(ordered["turn_index"].median())])
    candidates["distance"] = candidates["turn_index"].apply(
        lambda position: float(np.min(np.abs(template_positions - float(position))))
    )
    candidates = candidates.sort_values(["distance", "turn_index"], kind="stable")
    selected: list[str] = []
    for text in candidates["normalized_spoken_text"].astype(str):
        for token in text.split():
            if len(selected) == target_words:
                break
            selected.append(token)
        if len(selected) == target_words:
            break
    if len(selected) < target_words:
        raise ValueError(
            f"position-matched pool shortage: requested {target_words}, found {len(selected)}"
        )
    return " ".join(selected)


def fold_safe_shuffle(
    participant_ids: np.ndarray,
    texts: dict[int | str, str],
    *,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    seed: int,
) -> tuple[dict[int | str, str], pd.DataFrame]:
    participant_ids = np.asarray(participant_ids)
    train_ids = np.asarray(train_ids)
    test_ids = np.asarray(test_ids)
    if set(train_ids).intersection(test_ids):
        raise ValueError("train/test participant IDs overlap")
    if set(train_ids).union(test_ids) != set(participant_ids):
        raise ValueError("train/test IDs must cover all participants")
    if set(texts) != set(participant_ids):
        raise ValueError("texts must contain exactly the participant IDs")
    rng = np.random.default_rng(seed)
    shuffled: dict[int | str, str] = {}
    rows: list[dict[str, int | str]] = []
    for role, recipients in (("train", train_ids), ("test", test_ids)):
        recipients = np.asarray(recipients)
        donors = rng.permutation(recipients)
        if len(recipients) > 1:
            for _ in range(len(recipients) + 1):
                if not np.any(donors == recipients):
                    break
                donors = np.roll(donors, 1)
        for recipient, donor in zip(recipients, donors):
            shuffled[recipient] = texts[donor]
            rows.append(
                {
                    "role": role,
                    "recipient_id": recipient,
                    "donor_id": donor,
                    "seed": int(seed),
                }
            )
    return shuffled, pd.DataFrame(rows)
