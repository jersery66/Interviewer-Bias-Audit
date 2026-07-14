# C5 agreement material inventory

This inventory separates C5 model extraction, post-hoc source correction,
LLM repeatability, and the separate two-rater error-attribution study. Hashes
are SHA-256 and paths are relative to `F:/数据库/DAIC-WOZ` or its named
worktree.

## Canonical C5 material

The official-142 reviewed span file contains 1,137 model evidence spans from
141 participants and ten model-assigned symptom domains. Its exact quote
field is traceable to participant source text. The current audit separately
records source and quote hashes, location status, and manual correction status.

This is suitable as the model side of a future human validation dataset. It is
not sufficient for human–machine agreement because no independent human
validity, domain, or polarity labels are present.

## What is not C5 human–machine agreement

The `c5_reviewed_symptom_evidence.csv` and D-P files are frozen model-derived
participant-level inputs. The manual correction audit records whether a model
quote was replaced by a source quote, but it does not contain independent
blind human judgments.

The 24-participant `participant_level_agreement.csv` compares two LLM runs and
must be called **LLM extraction repeatability**. The 14-case
`coding_sheet_two_human_raters.csv` contains two human coders, but they coded
R1–R6 causes of model-error cases. It is valid for error-attribution
interrater agreement only, not for C5 span/domain agreement.

The `Desktop/人机一致性` folder is excluded because it is a C4
template/artifact rule-validation study with 200 items, not C5 symptom
evidence.

## Recommended next use

Use the canonical official-142 reviewed spans to create a blank human
validation dataset. Human reviewers must independently judge whether each
model quote is a valid participant span, its symptom domain and polarity, and
whether evidence is missing or a false positive. Reviewer identity, rater ID,
independent raw labels, and the sampling roster must be retained.
