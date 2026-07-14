"""Prepare the locked two-reviewer D-P blind-audit packet.

The public packet contains only blind IDs, participant language, line numbers,
the frozen codebook, and blank reviewer forms.  The blind-ID key and D-P
scoring reference are written under the ignored ``private`` directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reanalysis_v2.c5_blind_audit import make_stratified_sample
from reanalysis_v2.c5_blind_workflow import (
    DOMAIN_FEATURES,
    build_blind_cases,
    build_line_index,
    build_review_form,
    make_blind_id_key,
    validate_reviewer_form,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = REPO_ROOT / "analysis_v2"
PUBLIC_PACKET = ANALYSIS_ROOT / "04_c5_controls" / "blind_quality_audit" / "packet_2"
PRIVATE_PACKET = ANALYSIS_ROOT / "04_c5_controls" / "blind_quality_audit" / "private" / "packet_2"
SAMPLE_SEED = 20260714
TRAINING_SEED = 20260715
REVIEWER_A_ORDER_SEED = 202607141
REVIEWER_B_ORDER_SEED = 202607142


DOMAIN_DEFINITIONS = {
    "anhedonia_interest": "兴趣、愉快感或动机下降；必须有被试语言的明确支持。",
    "appetite_weight": "食欲或体重的改变。",
    "concentration_psychomotor": "注意力、集中、决策困难，或精神运动性迟缓/激越。",
    "depressed_mood": "悲伤、低落、情绪低沉或类似的抑郁心境表达。",
    "functioning_impairment": "工作、学习、社交或日常生活功能受影响。",
    "mental_health_history": "既往心理健康诊断、治疗、咨询、住院或相关病史。",
    "protective_or_absent_symptom": "明确的保护因素，或明确否认/不存在某症状；不能自动转译为另一症状域阳性。",
    "self_worth_guilt": "无价值感、自责、过度内疚或强烈自我贬低。",
    "sleep_fatigue_energy": "睡眠异常、疲倦或精力下降。",
    "suicide_self_harm": "自杀想法、死亡愿望、自伤行为或自伤想法。",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def _protocol_text() -> str:
    return """# D-P blinded manual quality audit protocol

## Status and scope

This is a post-audit, non-preregistered content-validity check of the frozen
participant-derived D-P domain counts.  The first pass is independent: each of
two reviewers completes all 30 cases × 10 domains (300 cells).  The raw first
pass is frozen before any discussion or exposure to the current D-P.

## Blinding contract

Reviewers may see only `blind_id`, participant language, line numbers, and the
frozen codebook.  They must not see participant IDs, PHQ-8 scores or item
scores, labels, interviewer language, current D-P counts or evidence, model
probabilities, prediction correctness, or Real/Random/Matched/Fake-D conditions.
The owner-only key is kept outside the public packet.  The two reviewers work
from separate files and do not discuss specific cells during the first pass.

## Training gate

Five cases outside the formal 30-case sample are provided as training cases.
Reviewers first score them independently, then discuss examples and clarify
the codebook.  Any codebook change must be completed before the formal 300
cells begin; the formal codebook is then frozen.  Training cases are excluded
from all agreement and D-P validity statistics.

## First-pass fields

Each formal cell records `domain_label` (0/1/NS), `count_bin` (0/1/2/3+), an
exact evidence quote and line number when label=1, `polarity`, and an
`uncertainty_reason` when label=NS.  A zero means only that no supporting
evidence was found in the visible participant text; it does not mean clinical
absence.  NS is used when wording, context, or domain boundaries are not
reliably determined.  Original reviewer values are never overwritten.

## Locked order of operations

1. Complete training independently, then freeze the codebook.
2. Complete all 300 formal cells independently; save hashes of both files.
3. Compute three-category and complete-case binary/count agreement before any
   consensus discussion.
4. Generate a disagreement sheet containing only discrepant cells and the
   participant text; adjudicate using the frozen codebook without opening D-P.
5. Save a consensus table that retains both raw labels and the reason for each
   consensus decision.  Unresolved cells are NS or go to a third adjudicator.
6. Only after consensus is frozen, unblind the owner key and compute D-P versus
   consensus precision, recall, F1, specificity, accuracy, and NS counts.
7. Review the frozen D-P evidence in a separate second round and classify each
   evidence item as valid/correct-domain, valid/wrong-domain, unsupported,
   duplicate, or ambiguous.

## Reporting rules

Primary D-P comparisons exclude human-consensus NS cells and report NS counts
and the number of valid cells.  Sensitivity analyses code NS as 0 and as 1;
neither is selected post hoc.  Report overall micro-F1, domain macro-F1, and
each domain's full metric row.  If reviewer B is unavailable, report only a
single-reviewer blinded content-validity sample; do not report kappa,
inter-rater agreement, or a fabricated consensus.
"""


def _codebook_text() -> str:
    domain_rows = "\n".join(f"- `{domain}`：{definition}" for domain, definition in DOMAIN_DEFINITIONS.items())
    return f"""# D-P blinded audit codebook (formal version)

## Domain definitions

{domain_rows}

## Domain label

- `1`: the visible participant text contains sufficiently explicit support for
  the domain and at least one exact quote plus line number is recorded.
- `0`: no supporting evidence is found in the visible participant text.  This
  is not a clinical claim that the symptom is absent.
- `NS`: meaning, context, or domain assignment cannot be determined reliably.
  Short answers such as `yes` or `sometimes` without the interviewer question,
  indirect evidence, and unresolved domain boundaries are NS.

## Count bin

Count independent evidence events or distinct symptom manifestations, not
repeated wording of the same manifestation: `0`, `1`, `2`, or `3+`.  A compound
sentence may count as two only when it contains two distinct manifestations
under this rule.  Identical repetition counts once.  For NS, count may be left
blank; the cell is excluded from the primary binary comparison.

## Evidence, line number, and polarity

Copy the participant's exact words; do not replace them with a summary.  Line
numbers are 1-based and may be comma- or semicolon-separated.  Use
`present` for positive symptom evidence, `denied/protective` for an explicit
denial or protective statement, and `ambiguous` when the text cannot support a
reliable polarity.  `protective_or_absent_symptom` can be positive for a
protective or explicit absence statement, but that statement must not also be
automatically coded as `suicide_self_harm=1`.

## Freeze rule

Reviewers may discuss the five training cases only.  Once formal scoring starts,
this codebook cannot be changed based on formal results.  Disagreements are
resolved after the independent metrics are calculated.  Raw reviewer files
remain immutable.
"""


def prepare(*, force: bool = False) -> dict[str, object]:
    if PUBLIC_PACKET.exists() and any(PUBLIC_PACKET.iterdir()) and not force:
        raise FileExistsError(f"public packet is not empty: {PUBLIC_PACKET}")
    if PRIVATE_PACKET.exists() and any(PRIVATE_PACKET.iterdir()) and not force:
        raise FileExistsError(f"private packet is not empty: {PRIVATE_PACKET}")
    PUBLIC_PACKET.mkdir(parents=True, exist_ok=True)
    PRIVATE_PACKET.mkdir(parents=True, exist_ok=True)

    participants_path = ANALYSIS_ROOT / "01_inputs" / "c2_participant_speech.csv"
    domain_path = ANALYSIS_ROOT / "04_c5_controls" / "domain_count" / "input.csv"
    participants = pd.read_csv(participants_path)
    domain_counts = pd.read_csv(domain_path)

    formal_sample = make_stratified_sample(participants, n_per_cell=5, seed=SAMPLE_SEED)
    formal_key = make_blind_id_key(formal_sample)
    formal_cases = build_blind_cases(participants, formal_key)
    formal_lines = build_line_index(formal_cases)
    reviewer_a = build_review_form(formal_cases["blind_id"], reviewer_id="A", order_seed=REVIEWER_A_ORDER_SEED)
    reviewer_b = build_review_form(formal_cases["blind_id"], reviewer_id="B", order_seed=REVIEWER_B_ORDER_SEED)
    validate_reviewer_form(reviewer_a, expected_blind_ids=formal_cases["blind_id"], require_completed=False)
    validate_reviewer_form(reviewer_b, expected_blind_ids=formal_cases["blind_id"], require_completed=False)

    excluded_ids = set(formal_sample["participant_id"].astype(int))
    remaining = participants.loc[~participants["participant_id"].astype(int).isin(excluded_ids)].copy()
    training = remaining.sample(n=5, random_state=TRAINING_SEED).sort_values("participant_id", kind="stable").reset_index(drop=True)
    training.insert(0, "audit_case_id", [f"TRAIN-{i:03d}" for i in range(1, len(training) + 1)])
    training_key = make_blind_id_key(training[["audit_case_id", "participant_id"]])
    training_key["blind_id"] = [f"T{i:03d}" for i in range(1, len(training_key) + 1)]
    training_cases = build_blind_cases(participants, training_key)
    training_lines = build_line_index(training_cases)
    training_a = build_review_form(training_cases["blind_id"], reviewer_id="A", order_seed=REVIEWER_A_ORDER_SEED + 1)
    training_b = build_review_form(training_cases["blind_id"], reviewer_id="B", order_seed=REVIEWER_B_ORDER_SEED + 1)

    # Public reviewer-facing files.
    _write_csv(formal_cases, PUBLIC_PACKET / "blind_cases.csv")
    _write_csv(formal_lines, PUBLIC_PACKET / "blind_case_lines.csv")
    _write_csv(reviewer_a, PUBLIC_PACKET / "reviewer_A_blank.csv")
    _write_csv(reviewer_b, PUBLIC_PACKET / "reviewer_B_blank.csv")
    _write_csv(training_cases, PUBLIC_PACKET / "training_cases.csv")
    _write_csv(training_lines, PUBLIC_PACKET / "training_case_lines.csv")
    _write_csv(training_a, PUBLIC_PACKET / "training_reviewer_A_blank.csv")
    _write_csv(training_b, PUBLIC_PACKET / "training_reviewer_B_blank.csv")
    _write_csv(
        pd.DataFrame(
            columns=[
                "reviewer_id",
                "blind_id",
                "domain",
                "dp_evidence_id",
                "dp_evidence_quote",
                "dp_line_number",
                "evidence_validity",
                "adjudication_reason",
            ]
        ),
        PUBLIC_PACKET / "evidence_review_template.csv",
    )
    (PUBLIC_PACKET / "blind_audit_protocol.md").write_text(_protocol_text(), encoding="utf-8")
    (PUBLIC_PACKET / "blind_codebook.md").write_text(_codebook_text(), encoding="utf-8")
    (PUBLIC_PACKET / "README.md").write_text(
        """# D-P blinded manual audit packet (locked preparation)

Give each reviewer `blind_cases.csv`, `blind_case_lines.csv`,
`blind_codebook.md`, and exactly one reviewer blank form. Use the five
`training_*` files first; training is not part of the formal statistics. The
formal sample has 30 cases and 300 cells. Returned files should be saved as
`reviewer_A_frozen.csv` and `reviewer_B_frozen.csv` under the ignored private
packet directory.

The owner-only key and D-P scoring reference are not in this public packet.
Do not calculate agreement, consensus, or D-P validity until both independent
formal files are complete and their hashes are saved.
""",
        encoding="utf-8",
    )

    # Owner-only files: these paths are ignored by git.
    formal_key.to_csv(PRIVATE_PACKET / "blind_id_key.csv", index=False, lineterminator="\n")
    training_key.to_csv(PRIVATE_PACKET / "training_id_key.csv", index=False, lineterminator="\n")
    scoring_reference = formal_key[["blind_id", "participant_id"]].merge(
        domain_counts[["participant_id", *DOMAIN_FEATURES]],
        on="participant_id",
        how="left",
        validate="one_to_one",
    )
    if scoring_reference[list(DOMAIN_FEATURES)].isna().any().any():
        raise ValueError("owner scoring reference is missing a D-P domain count")
    scoring_reference.to_csv(PRIVATE_PACKET / "scoring_reference_owner_only.csv", index=False, lineterminator="\n")

    public_files = [
        path
        for path in sorted(PUBLIC_PACKET.iterdir())
        if path.is_file() and path.name not in {"packet_manifest.json", "blind_audit_manifest.json"}
    ]
    manifest = {
        "status": "pending_reviewer_input",
        "analysis": "d_p_blinded_manual_quality_audit_v2",
        "formal_sample_n": int(len(formal_cases)),
        "formal_cells": int(len(reviewer_a)),
        "training_sample_n": int(len(training_cases)),
        "training_excluded_from_statistics": True,
        "reviewers_complete_case_requirement": True,
        "sampling_seed": SAMPLE_SEED,
        "training_seed": TRAINING_SEED,
        "reviewer_order_seeds": {"A": REVIEWER_A_ORDER_SEED, "B": REVIEWER_B_ORDER_SEED},
        "domains": list(DOMAIN_FEATURES),
        "inputs": {
            "participant_speech": {"path": str(participants_path.relative_to(REPO_ROOT)), "sha256": _sha256(participants_path)},
            "domain_count": {"path": str(domain_path.relative_to(REPO_ROOT)), "sha256": _sha256(domain_path)},
        },
        "private_owner_files": {
            "blind_id_key.csv": _sha256(PRIVATE_PACKET / "blind_id_key.csv"),
            "training_id_key.csv": _sha256(PRIVATE_PACKET / "training_id_key.csv"),
            "scoring_reference_owner_only.csv": _sha256(PRIVATE_PACKET / "scoring_reference_owner_only.csv"),
        },
        "public_output_hashes": {path.name: _sha256(path) for path in public_files},
        "code_commit": _code_commit(),
        "fallback_if_reviewer_b_missing": "single-reviewer blinded content-validity sample only; no kappa, inter-rater agreement, or consensus metrics",
    }
    manifest_text = json.dumps(manifest, indent=2, ensure_ascii=False)
    (PUBLIC_PACKET / "packet_manifest.json").write_text(manifest_text, encoding="utf-8")
    (PUBLIC_PACKET / "blind_audit_manifest.json").write_text(manifest_text, encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the two-reviewer D-P blinded audit packet")
    parser.add_argument("--force", action="store_true", help="replace packet_2 and its ignored owner directory")
    args = parser.parse_args()
    manifest = prepare(force=args.force)
    print(json.dumps({"status": manifest["status"], "formal_cells": manifest["formal_cells"], "packet": str(PUBLIC_PACKET)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
