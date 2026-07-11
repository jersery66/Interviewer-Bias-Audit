# Post-submission-audit sensitivity analysis plan

## Status and timing

- Contract frozen at: `2026-07-11T21:13:28+08:00`.
- Repository base before this plan-only commit: `c262bca90143404e444b8dffba2457601828c5f5`.
- These analyses were proposed after review of the first submission-audit draft. They are not preregistered, are not part of the original analysis plan, and must be described as post-audit sensitivity analyses.
- The analysis-locked main split is frozen repeat 1 of the existing 10 x 5 participant-level split file. Locking repeat 1 before this sensitivity run prevents further result-dependent split selection, but does not constitute prospective preregistration.
- Definitions, seeds, output names, and interpretation rules below will not be changed after inspecting results. Any unavoidable implementation correction must be documented in the revision log and committed before rerunning affected analyses.

## Evidence hierarchy

1. The original five analysis-locked comparisons remain the main inferential family. Their existing BH-adjusted q-values are not recomputed with the post-audit analyses.
2. Brier baselines, symmetric token controls, positional windows, and retained-span source-alignment checks are post-audit sensitivity analyses.
3. The existing 10 x 5 analyses remain split-stability supplements only.
4. E-DAIC remains a small descriptive supplement and is excluded from the title, abstract core claims, and main figures.

## 1. Timeline terminology

All manuscript and audit text will use `analysis-locked split` or the Chinese equivalent `在最终投稿审计开始前锁定的第 1 组五折划分`. The following claims are prohibited unless supported by an earlier timestamped record:

- preregistered or formally preregistered;
- prespecified split or prospectively prespecified split;
- selected before any result was seen.

The default probability operating point may be called `default 0.50 operating point`, but not a clinical cutoff or preregistered threshold.

## 2. Brier no-skill baselines and skill scores

### 2.1 Formal no-leakage baseline

For each outer fold, the no-skill probability assigned to every test participant is the PHQ-8-positive prevalence calculated only from that fold's outer-training participants. These five test-fold predictions are concatenated into one participant-level cross-fitted null vector.

For each source condition and each probability variant (`raw`, `Platt`, `isotonic`), use the already frozen outer-test predictions in `locked_source_predictions.csv`. Calibrators must not be refit on pooled OOF predictions.

The primary skill score is:

`BSS_trainfold = 1 - Brier_model / Brier_trainfold_null`.

Uncertainty is estimated by 5,000 stratified participant bootstrap samples. In every sample, model and null squared errors are resampled as paired participant-level observations. Report the 2.5th and 97.5th percentiles for Brier and BSS.

### 2.2 Descriptive cohort-prevalence benchmark

The fixed probability `43/142` is used only as an intuitive full-cohort benchmark. Its Brier score and corresponding descriptive BSS are reported separately and never called the formal no-leakage baseline.

### 2.3 Isotonic interpretation

Report raw, Platt, and isotonic results on the same outer cross-fitted participants. Add fold-level Brier values and their dispersion. No calibration method will be called best solely because one aggregate Brier estimate is lowest; isotonic estimates will be described as potentially unstable in this sample of 142 participants.

### 2.4 Outputs

- `locked_brier_skill_metrics.csv`
- `locked_brier_fold_metrics.csv`
- `locked_null_brier_predictions.csv`

## 3. Existing asymmetric token control audit

The existing control is retained and renamed `longer-source-to-shorter-source-length control`. For participant `i`, its target is `min(L_participant,i, L_interviewer,i)`. The shorter source remains intact and only the longer source is truncated.

The audit will report, by source:

- original token median and quartiles;
- number with target length below original length;
- mean and quartiles of retained fraction among non-empty source texts;
- shared target-token minimum, quartiles, median, and maximum;
- zero-length source count, zero target count, and target count below 10 tokens.

This analysis asks what happens when the longer source is limited to the shorter source's length. It is not a symmetric information-budget intervention.

### Output

- `token_matching_asymmetry_summary.csv`

## 4. Symmetric half-min token control

### 4.1 Budget

For participant `i`:

`L_i = floor(0.5 * min(L_participant,i, L_interviewer,i))`.

Both participant and interviewer text are truncated to exactly `L_i` whitespace-delimited tokens. The same 142 participants, labels, draw IDs, and frozen five folds are used for both sources.

If `L_i = 0`, both source texts are represented as empty strings for that participant and retained in every draw. Targets from 1 to 9 tokens are retained and reported as very short; no participant is excluded based on target length.

### 4.2 Random windows

- Draws: 50, numbered 1 through 50.
- Base seed: `20260705`.
- Participant-specific draw seed: `20260705 + 2,000,003 * draw + 97 * participant_id`.
- One RNG initialized from that seed samples the participant start first and interviewer start second. This fixes a paired dataset for each participant and draw without selecting source-specific favorable windows.
- Each draw completes a full five-fold model run. The model seed is `20260705 + 20,000 * draw`, with the existing fold/model offsets applied by `run_repeated_cv`.
- Each draw contains both source probabilities for the same participant. Source differences are paired within participant and draw.

The primary point estimate is the mean of the 50 independently calculated AUCs, not the AUC of probabilities averaged across draws. The joint 95% interval uses 5,000 replicates; each replicate selects one complete draw, then resamples positive and negative participants with replacement and computes the paired source AUC difference. This jointly represents participant and window uncertainty.

### 4.3 Deterministic positional windows

The same `L_i` budget is used for both sources:

- early: start index `0`;
- middle: start index `floor((L_source,i - L_i) / 2)`;
- late: start index `L_source,i - L_i`.

If `L_i = 0`, the position-specific text is empty. Each position is evaluated on the same participants and frozen folds. No source-specific position selection is permitted. Report source AUC, PR-AUC, and Brier at each position. The three participant-versus-interviewer AUC contrasts form one post-audit positional family and receive BH correction across the three positions.

### 4.4 Outputs

- `symmetric_half_min_length_audit.csv`
- `symmetric_half_min_audit_summary.csv`
- `symmetric_half_min_draw_predictions.csv`
- `symmetric_half_min_draw_metrics.csv`
- `symmetric_half_min_metrics.csv`
- `symmetric_half_min_paired_comparison.csv`
- `symmetric_position_predictions.csv`
- `symmetric_position_metrics.csv`
- `symmetric_position_paired_comparisons.csv`

### 4.5 Interpretation rule

The strongest permitted wording is: `Text quantity differences do not fully account for the observed source pattern under the tested whitespace-token budgets.` Neither token control identifies semantics, interviewer bias, leakage, or a causal mechanism.

## 5. Retained-span source-alignment sensitivity analysis

### 5.1 Scope

This analysis is named `retained-span source-alignment sensitivity analysis` or `最终保留证据的来源对齐修订敏感性分析`.

It compares two representations of the same 1,137 retained span records:

1. `original retained quotes`: `original_model_quote` ordered by frozen participant and source order;
2. `source-aligned retained quotes`: the reviewed exact-source quote used by the frozen C5 condition.

The same span inclusion, participant set, source order, symptom domain, and polarity are held fixed by construction. Only quote text is replaced. The analysis cannot reconstruct the five excluded duplicates, the one excluded invalid candidate, or any content not retained in the reviewed span table. It must not be called pre-review versus post-review C5, original versus corrected C5, or a complete manual-review comparison.

Before modeling, reconstruction of the aligned retained representation must match the frozen reviewed C5 participant text after the existing normalization rule. Failure to match aborts this sensitivity analysis.

### 5.2 Descriptive audit

Report:

- 1,143 total candidates;
- 1,102 direct source matches among all candidates (`96.4%` after exact calculation from frozen counts);
- 35 source-alignment revisions and their number of unique participants;
- 5 duplicate and 1 invalid exclusions;
- 1,137 retained spans traceable after review;
- participants whose retained-span text representation changes;
- total and participant-level token-count changes;
- explicit confirmation that domain, polarity, source order, and retained status were held fixed in this sensitivity construction;
- the available audit record documents quote-alignment actions but does not establish outcome blinding or independent double review.

The reviewed C5 input was committed before this post-audit sensitivity run. This establishes ordering relative to the present run only, not prospective blinding or timing relative to every historical analysis.

### 5.3 Predictive sensitivity

Both representations use the frozen analysis-locked five folds and the existing TF-IDF logistic-regression configuration. Report AUC, PR-AUC, Brier, and participant-level prediction changes.

For each metric, report aligned-minus-original difference, 5,000 participant-bootstrap 95% CI, raw paired permutation p-value from 10,000 within-participant swaps, and BH q-value across the three C5 sensitivity metrics. For Brier, a negative aligned-minus-original difference indicates improvement.

Prediction-change summaries include mean, median, interquartile range, 95th percentile, maximum absolute probability change, and counts above absolute changes of 0.01 and 0.05.

### 5.4 Outputs

- `c5_retained_alignment_audit.csv`
- `c5_retained_alignment_predictions.csv`
- `c5_retained_alignment_metrics.csv`
- `c5_retained_alignment_paired_differences.csv`
- `c5_retained_alignment_probability_change_summary.csv`

No quote text is written to public outputs.

## 6. Statistical presentation and model configuration

- AUC differences are printed to four decimals. The interviewer-minus-participant lower bound is reported as the actual negative value, not rounded to `-0.000`.
- Every inferential comparison table distinguishes 5,000-participant-bootstrap confidence intervals from 10,000-swap raw permutation p-values and BH-adjusted q-values.
- Add a complete model configuration table covering TF-IDF lowercasing, Unicode accent stripping, 1-2 grams, `min_df=2`, `max_features=50,000`, sublinear TF, logistic `C=1.0`, balanced class weights, liblinear solver, 2,000 iterations, fold-specific seeds, training-fold-only fitting, numeric standardization, frozen fold membership, and calibration implementation.

### Output

- `model_configuration.md`

## 7. Manuscript revision rules

The manuscript is revised only after all post-audit outputs are frozen.

- Interaction-only AUC is described as uncertain because its interval includes 0.5. Protocol-only and all-structure estimates are reported separately.
- Original and symmetric token controls are described as answering different questions.
- C5 direct-match rate and retained-span post-review traceability are always reported separately.
- Novelty language is limited to the representative studies included in the documented targeted search. The manuscript will not claim first, field-wide rarity, or systematic-review completeness.
- New sensitivity results may weaken existing claims but cannot be omitted because of direction or significance.

Files synchronized after result freeze:

- `analysis_v2/06_tables_figures/manuscript_submission_zh.md`
- `analysis_v2/12_submission_audit/claim_evidence_matrix.md`
- `analysis_v2/12_submission_audit/submission_audit_summary.md`
- `analysis_v2/12_submission_audit/reviewer_requirement_matrix.md`
- `analysis_v2/06_tables_figures/manuscript_review_and_revision_log.md`

## 8. Targeted literature search

The literature step is a documented targeted search, not a systematic review. Record database/source, exact query, search date, returned records inspected, inclusion/exclusion rule, deduplication, and reason each study enters the representative comparison table. Use primary paper or publisher records for bibliographic verification.

### Output

- `targeted_literature_search_log.md`

## 9. Reproducibility and CI boundary

GitHub Actions may verify only:

1. code/test reproducibility from the public repository;
2. integrity and cross-file consistency of committed derived results;
3. schema, path portability, synthetic fixtures, and figure-generation executability.

Full data-to-result reproduction requires restricted DAIC-WOZ data and the private reviewed C5 source directory. A passing CI job must not be described as raw-data reproduction.

The README and CI workflow will state three levels explicitly:

- `code/test reproducibility`;
- `result integrity verification`;
- `full data-to-result reproduction with restricted inputs`.

## 10. Commit and execution order

1. `docs: lock post-audit sensitivity analysis plan` -- this file only.
2. `analysis: add post-audit sensitivity checks` -- implementation and tests, before generating final outputs.
3. `data: publish post-audit sensitivity outputs` -- regenerated derived outputs whose run manifest records the analysis-code commit.
4. `manuscript: integrate post-audit sensitivity results and CI documentation` -- manuscript, evidence matrices, targeted search log, README, and GitHub Actions.

After implementation, run the full test suite, verify output SHA-256 hashes, perform two independent recomputations of core numerical outputs, verify a clean worktree, push the branch, and confirm that local and remote branch heads match.
