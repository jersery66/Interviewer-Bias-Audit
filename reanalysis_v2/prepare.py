from __future__ import annotations

import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from .contracts import validate_primary_table
from .io import atomic_write_csv, atomic_write_json, sha256_file
from .splits import make_repeated_split_membership


CONDITION_OUTPUT_NAMES = {
    "full_transcript": "c1_full_transcript.csv",
    "participant_speech": "c2_participant_speech.csv",
    "interviewer_speech": "c3_interviewer_speech.csv",
    "non_explicit_interviewer_speech": "c4_non_explicit_interviewer.csv",
    "participant_symptom_evidence": "c5_reviewed_symptom_evidence.csv",
}


PROMPT_ANNOTATION_RE = re.compile(
    r"^(?P<prompt_id>[A-Za-z][A-Za-z0-9_]*)\s+\((?P<spoken>.*)\)$",
    flags=re.DOTALL,
)

CLINICAL_PATTERNS = [
    r"\bdepress(?:ed|ion)?\b",
    r"\bsad(?:ness)?\b",
    r"\bfeel(?:ing)?\s+down\b",
    r"\bhopeless(?:ness)?\b",
    r"\bmood\b",
    r"\bsleep(?:ing)?\b",
    r"\binsomnia\b",
    r"\btired(?:ness)?\b",
    r"\bfatigue\b",
    r"\benergy\b",
    r"\bappetite\b",
    r"\bweight\b",
    r"\bsuicid(?:e|al)\b",
    r"\bkill yourself\b",
    r"\bhurt yourself\b",
    r"\bself[- ]harm\b",
    r"\bmental health\b",
    r"\btherap(?:y|ist)\b",
    r"\bcounsel(?:or|lor|ing)\b",
    r"\btreatment\b",
    r"\bmedication\b",
    r"\bantidepressant\b",
    r"\btrauma(?:tic)?\b",
    r"\babuse(?:d)?\b",
    r"\bp_?t_?s_?d\b",
    r"\bdiagnos(?:e|ed|is)\b",
    r"\banxiety\b",
]
CLINICAL_RE = re.compile("|".join(CLINICAL_PATTERNS), flags=re.IGNORECASE)


def _parse_interviewer_value(value: str) -> tuple[str, str, int]:
    raw = re.sub(r"\s+", " ", str(value)).strip()
    match = PROMPT_ANNOTATION_RE.fullmatch(raw)
    if not match:
        return raw, "", 0
    return match.group("spoken").strip(), match.group("prompt_id"), 1


def _normalize_spoken_text(value: str) -> str:
    text = str(value).lower().replace("_", " ")
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _join_spoken(values: pd.Series) -> str:
    return " ".join(" ".join(values.fillna("").astype(str)).split())


def freeze_c4_turns_input(
    turns_source_path: str | Path,
    output_root: str | Path,
    *,
    expected_n: int = 142,
) -> dict[str, str | int]:
    turns_source_path = Path(turns_source_path).resolve()
    output_root = Path(output_root).resolve()
    c3 = pd.read_csv(
        output_root / "01_inputs" / CONDITION_OUTPUT_NAMES["interviewer_speech"],
        keep_default_na=False,
    ).sort_values("participant_id", kind="stable")
    c4 = pd.read_csv(
        output_root / "01_inputs" / CONDITION_OUTPUT_NAMES["non_explicit_interviewer_speech"],
        keep_default_na=False,
    ).sort_values("participant_id", kind="stable")
    participant_ids = c3["participant_id"].astype(int).tolist()
    if len(participant_ids) != expected_n or set(participant_ids) != set(c4["participant_id"].astype(int)):
        raise ValueError("C4 turn freeze cohort does not match the frozen primary cohort")

    source = pd.read_csv(turns_source_path, keep_default_na=False)
    required = {
        "participant_id",
        "turn_index",
        "speaker_norm",
        "text",
        "official_avec_split",
        "start_time",
        "stop_time",
    }
    if not required.issubset(source.columns):
        raise ValueError(f"verified turn source missing columns: {sorted(required-set(source.columns))}")
    source["participant_id"] = pd.to_numeric(source["participant_id"], errors="raise").astype(int)
    source["turn_index"] = pd.to_numeric(source["turn_index"], errors="raise").astype(int)
    source = source.loc[source["participant_id"].isin(participant_ids)].copy()
    if set(source["participant_id"].unique()) != set(participant_ids):
        raise ValueError("verified turn source does not cover every frozen participant")
    source = source.sort_values(["participant_id", "turn_index"], kind="stable").reset_index(drop=True)
    source["speaker_norm"] = source["speaker_norm"].astype(str).str.lower()

    parsed = source.apply(
        lambda row: _parse_interviewer_value(row["text"])
        if row["speaker_norm"] == "interviewer"
        else (re.sub(r"\s+", " ", str(row["text"])).strip(), "", 0),
        axis=1,
        result_type="expand",
    )
    parsed.columns = ["spoken_text", "prompt_id", "is_prompt_annotated"]
    source = pd.concat([source, parsed], axis=1)
    source["normalized_spoken_text"] = source["spoken_text"].map(_normalize_spoken_text)
    source["is_explicit_clinical_prompt"] = (
        source["speaker_norm"].eq("interviewer")
        & source["spoken_text"].map(lambda value: bool(CLINICAL_RE.search(value)))
    ).astype(int)
    source = source.rename(columns={"text": "verified_text"})

    c3_lookup = c3.set_index("participant_id")["text"].to_dict()
    c4_lookup = c4.set_index("participant_id")["text"].to_dict()
    audit_rows = []
    for participant_id in participant_ids:
        interviewer = source.loc[
            source["participant_id"].eq(participant_id)
            & source["speaker_norm"].eq("interviewer")
        ].sort_values("turn_index", kind="stable")
        all_text = _join_spoken(interviewer["spoken_text"])
        non_explicit = _join_spoken(
            interviewer.loc[interviewer["is_explicit_clinical_prompt"].eq(0), "spoken_text"]
        )
        expected_c3 = " ".join(str(c3_lookup[participant_id]).split())
        expected_c4 = " ".join(str(c4_lookup[participant_id]).split())
        audit_rows.append(
            {
                "participant_id": participant_id,
                "interviewer_turns": int(len(interviewer)),
                "explicit_clinical_turns": int(interviewer["is_explicit_clinical_prompt"].sum()),
                "c3_exact_match": all_text == expected_c3,
                "c4_exact_match": non_explicit == expected_c4,
                "derived_c3_words": len(all_text.split()),
                "frozen_c3_words": len(expected_c3.split()),
                "derived_c4_words": len(non_explicit.split()),
                "frozen_c4_words": len(expected_c4.split()),
            }
        )
    alignment = pd.DataFrame(audit_rows)
    if not alignment["c3_exact_match"].all() or not alignment["c4_exact_match"].all():
        mismatch = alignment.loc[
            ~alignment["c3_exact_match"] | ~alignment["c4_exact_match"], "participant_id"
        ].tolist()
        raise ValueError(f"verified turn reconstruction does not match frozen C3/C4: {mismatch}")

    frozen_path = output_root / "01_inputs" / "c4_turns_official142_frozen.csv"
    audit_path = output_root / "00_plan" / "c4_turn_source_alignment_audit.csv"
    atomic_write_csv(source, frozen_path)
    atomic_write_csv(alignment, audit_path)
    entry: dict[str, str | int] = {
        "path": frozen_path.relative_to(output_root).as_posix(),
        "rows": int(len(source)),
        "participants": int(source["participant_id"].nunique()),
        "sha256": sha256_file(frozen_path),
        "source_path": str(turns_source_path),
        "source_sha256": sha256_file(turns_source_path),
        "alignment_audit_path": audit_path.relative_to(output_root).as_posix(),
        "alignment_audit_sha256": sha256_file(audit_path),
        "c3_exact_matches": int(alignment["c3_exact_match"].sum()),
        "c4_exact_matches": int(alignment["c4_exact_match"].sum()),
        "explicit_clinical_turns": int(source["is_explicit_clinical_prompt"].sum()),
        "annotation_rule": "frozen clinical regex v1, reproduced from canonical source build",
    }
    manifest_path = output_root / "00_plan" / "input_freeze_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("control_inputs", {})["c4_turns"] = entry
    atomic_write_json(manifest, manifest_path)
    return entry


def prepare_primary_inputs(
    source_path: str | Path,
    output_root: str | Path,
    *,
    expected_n: int = 142,
    n_repeats: int = 10,
    n_splits: int = 5,
    base_seed: int = 20260624,
) -> dict:
    source_path = Path(source_path).resolve()
    output_root = Path(output_root).resolve()
    table = pd.read_csv(source_path, keep_default_na=False)
    summary = validate_primary_table(
        table, list(CONDITION_OUTPUT_NAMES), expected_n=expected_n
    )
    table["participant_id"] = pd.to_numeric(table["participant_id"], errors="raise").astype(int)
    table["paper_label_phq8_ge10"] = pd.to_numeric(
        table["paper_label_phq8_ge10"], errors="raise"
    ).astype(int)

    frozen_inputs: dict[str, dict[str, str | int]] = {}
    for condition, filename in CONDITION_OUTPUT_NAMES.items():
        condition_table = (
            table.loc[table["condition_id"].eq(condition)]
            .sort_values("participant_id", kind="stable")
            .reset_index(drop=True)
        )
        destination = output_root / "01_inputs" / filename
        atomic_write_csv(condition_table, destination)
        frozen_inputs[condition] = {
            "path": destination.relative_to(output_root).as_posix(),
            "rows": int(len(condition_table)),
            "sha256": sha256_file(destination),
        }

    participant_table = (
        table.sort_values(["participant_id", "condition_id"], kind="stable")
        .drop_duplicates("participant_id")
        .sort_values("participant_id", kind="stable")
    )
    participant_ids = participant_table["participant_id"].to_numpy(dtype=int)
    labels = participant_table["paper_label_phq8_ge10"].to_numpy(dtype=int)
    membership = make_repeated_split_membership(
        participant_ids,
        labels,
        n_repeats=n_repeats,
        n_splits=n_splits,
        base_seed=base_seed,
    )
    split_path = output_root / "00_splits" / "repeated_5fold_splits_10x5.csv"
    atomic_write_csv(membership, split_path)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
        },
        "cohort": {
            "n_participants": summary["n_participants"],
            "n_positive": summary["n_positive"],
            "n_negative": summary["n_negative"],
        },
        "endpoint": "paper_label_phq8_ge10 = int(paper_phq8_score >= 10)",
        "official_test_in_primary_analysis": False,
        "frozen_inputs": frozen_inputs,
        "splits": {
            "path": split_path.relative_to(output_root).as_posix(),
            "sha256": sha256_file(split_path),
            "n_repeats": n_repeats,
            "n_splits": n_splits,
            "base_seed": base_seed,
            "repeat_seed_formula": "base_seed + repeat, repeat starts at 1",
            "membership_rows": int(len(membership)),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    atomic_write_json(manifest, output_root / "00_plan" / "input_freeze_manifest.json")
    return manifest
