# C5/D-P human-validation status

## Finding

The previous human–machine agreement data gap has been closed at the
participant-by-domain level. Packet 3 now contains two independent Stage-1
ratings, a fully adjudicated 440-cell consensus table, and the owner-only D-P
count reference. The main 30-participant sample therefore supports formal
domain-presence and count validation.

For the main sample, human status agreement was 0.870 with Cohen's kappa 0.810
(participant-cluster bootstrap 95% CI 0.739–0.872). Against the frozen human
consensus, D-P domain presence had accuracy 0.799, sensitivity 0.792,
specificity 0.805, positive-class F1 0.774, macro-F1 0.797, and Cohen's kappa
0.594. The primary kappa interval was 0.509–0.673.

## Remaining evidence-level boundary

Stage 2 contains independent ratings of selected model evidence spans and
human-reference evidence events. Its structural checks passed, but final
metric adjudication was not completed. Consequently, final evidence-span
validity, exact evidence-event linkage, and important-omission rates are not
promoted to manuscript claims.

This remaining gap does not erase the completed domain-level validation. It
limits a stronger claim that the retained C5 spans form a complete and
error-free evidence set.

## Terminology boundary

- Stage-1 results are **blinded domain-level D-P human validation**.
- Stage-2 unadjudicated results are **preliminary evidence-level quality
  audit**, not final consensus validity estimates.
- The 24-participant model-run comparison remains **LLM extraction
  repeatability** and must not be relabeled as human validity.
- The older post-hoc correction log remains a source-traceability artifact,
  not an independent human gold standard.

The current paper may describe D-P as a participant-derived proxy with
moderate domain-level human validity. It may not describe D-P as a clinical
gold standard, a complete symptom inventory, or proof that interviewer-signal
attenuation is uniquely caused by true symptom absorption.

## Stage-2 dual-review audit (2026-08-04)

Both independent Stage-2 reviewer files were structurally complete, their
record keys matched, and the fixed blind IDs, domains, source quotes, and line
numbers were unchanged between reviewers. The evidence-span review contained
392 rows. Domain-correctness agreement was 0.980 (Cohen's kappa 0.823), and
context-preservation agreement was 0.972 (kappa 0.409). Twenty rows differed
on at least one core evidence field; three additional rows differed only on
the duplicate-evidence identifier, yielding 23 evidence rows requiring
adjudication before final evidence-error rates can be reported.

The omission review contained 850 human-reference events. Agreement on whether
an event was covered by D-P was 0.927 (kappa 0.855), but agreement on whether
an uncovered event was an important omission was only 0.674 (kappa 0.369).
Reviewer-specific important-omission rates were 0.528 and 0.202, and 280 rows
differed on at least one core omission field. This is a systematic criterion
difference, not a minor residual disagreement.

Therefore, the paper continues without a final Stage-2 omission-rate claim.
The optional evidence-span analysis can be completed by adjudicating the 23
evidence rows. The 280 omission disagreements are deferred unless a target
journal explicitly requires a final evidence-level omission estimate. Formal
row-level outputs and hashes are stored under
`analysis_v2/04_c5_controls/blind_quality_audit/private/packet_3/stage2/interrater_validation/`.
