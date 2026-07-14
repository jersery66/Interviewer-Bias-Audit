# D-P blinded manual quality audit protocol

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
