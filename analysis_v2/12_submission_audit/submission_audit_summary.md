# Submission audit: locked main analysis and required controls

## Locked claim

This study audits the sources of PHQ-8 label-predictive signals in semi-structured interviews and tests whether full-transcript performance can be attributed directly to participant language alone.

## Main analysis contract

- Main estimates use frozen repeat 1 of the prespecified participant-level 5-fold split; every participant contributes one cross-fitted prediction.
- The existing 10 x 5 repeated cross-validation is retained only as a split-stability supplement.
- TF-IDF fitting, scaling, calibration, and threshold selection remain inside training data. The 0.5 threshold is called a prespecified/default operating threshold, not a clinical cutoff.
- Paired uncertainty resamples participants, not folds or repeated OOF rows.

## Locked-split source results

- participant_speech: AUC 0.717 (95% CI 0.614-0.811), Brier 0.231; nested-threshold sensitivity/specificity 0.488/0.869.
- interviewer_speech: AUC 0.808 (95% CI 0.729-0.879), Brier 0.196; nested-threshold sensitivity/specificity 0.674/0.778.
- full_transcript: AUC 0.760 (95% CI 0.670-0.844), Brier 0.217; nested-threshold sensitivity/specificity 0.581/0.727.
- participant_symptom_evidence: AUC 0.735 (95% CI 0.640-0.819), Brier 0.224; nested-threshold sensitivity/specificity 0.535/0.818.

Single-source performance measures predictive sufficiency, not independent contribution or causal bias. Conditional increments are reported separately in the paired-comparison table.

## Structural and text-quantity controls

- S_length: AUC 0.569.
- S_interaction: AUC 0.573.
- S_protocol: AUC 0.675.
- S_all: AUC 0.729.
- participant_matched, mean across 50 deterministic contiguous-window draws: AUC 0.651 (participant + random-draw 95% CI 0.514-0.780).
- interviewer_matched, mean across 50 deterministic contiguous-window draws: AUC 0.808 (participant + random-draw 95% CI 0.726-0.882).
- Matched-text mean Delta AUC (interviewer minus participant): 0.158 (participant + random-draw 95% CI 0.023-0.294).

These controls determine whether process structure and text quantity can account for source-level performance. They do not isolate a single generative mechanism because protocol branching, participant responses, and interviewer adaptation remain coupled.

## Prespecified paired comparisons

| Comparison | Scope | Delta AUC | 95% CI | q |
|---|---|---:|---:|---:|
| full_transcript vs participant_speech | source_sufficiency_not_independent_contribution | 0.043 | -0.039 to 0.130 | 0.473 |
| interviewer_speech vs participant_speech | source_sufficiency_not_independent_contribution | 0.091 | -0.000 to 0.190 | 0.438 |
| participant_symptom_evidence vs participant_speech | derived_representation_vs_raw_source | 0.018 | -0.079 to 0.107 | 0.727 |
| M5 vs M3 | conditional_interviewer_increment | 0.003 | -0.004 to 0.010 | 0.473 |
| M4 vs M3 | conditional_evidence_representation_increment | 0.004 | -0.004 to 0.011 | 0.473 |

## C5 boundary

All 1137/1137 retained C5 spans match participant source text after review, including 35 corrected spans. However, outcome blinding cannot be demonstrated and inter-rater reliability is unavailable. C5 is therefore reported as a clinically guided derived representation, not as an unprocessed natural signal or evidence of pathological understanding.

## E-DAIC boundary

Module 14 remains a small descriptive supplement only. It is excluded from the title, abstract, core claims, and main figures; it does not establish generalizability or an error mechanism.
