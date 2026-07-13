# Interviewer-side predictive signal explanation analysis plan

**Lock date:** 2026-07-13
**Status:** Stage A plan/code contract; no formal results are included
**Repository:** `jersery66/Interviewer-Bias-Audit`
**Branch:** `codex/submission-audit`
**Stage A.3 parent commit:** `81a934837d25d90715769eacac2a57fafb97919c`
**Current checkout inspected before this revision:** `81a934837d25d90715769eacac2a57fafb97919c`
**Known upstream baseline:** `bbf72b4ae7ccf25b9262a1dd94c3a9da21a9431e`

## 1. Scope and research question

This module is independent of the frozen analyses in `analysis_v2/00` through
`analysis_v2/12`. It does not overwrite or regenerate any existing result.
Formal outputs will be written only below:

```text
analysis_v2/15_interviewer_signal_explanation/
```

The module-level question is:

> Which observable information blocks can statistically reconstruct the
> interviewer-side PHQ-8 predictive signal, and does a residual signal remain
> after that reconstruction?

The module estimates statistical explanation, score recoverability, conditional
attenuation, overlapping predictive information, residual signal, pairing
dependence, and representation stability. It does not identify causal
mediation, a causal interviewer effect, a signal composition percentage, an
independent mechanism, or a complete explanation.

This module is a second-layer constraint on the existing source and conditional
increment analyses. It is not assumed to supersede the original incremental
research question or to become the primary manuscript storyline. Its target is
a PHQ-label-trained interviewer model score, not an interviewer mental process,
objective language composition, individual interaction mechanism, or causal
source of interviewer-side predictive signal. Results must therefore be framed
as statistical recoverability, conditional attenuation, overlapping predictive
information, residual signal, and pairing dependence.

The module will not use `D-All`. The symptom block is always the frozen
participant-derived `D-P` ten-domain count vector.

## 2. Stage and execution contract

Stage A.3 contains only this revised plan, implementation code, tests, and
non-result input-contract documentation. It is a method revision of the Stage
A.2 parent commit above; the future formal `code_commit` must be the new Stage
A.3 commit, never the parent commit. It must be committed before any formal
command is run. The Stage A.3 commit SHA is the `code_commit` recorded by
future formal runs.

Stage B is deliberately not run in the current task. A future Stage B run must:

1. run `run_1` and `run_2` with identical code, inputs, settings, and seeds;
2. compare SHA-256 hashes for every core output;
3. create `final/` only after exact equality;
4. write `rerun_verification.json`;
5. update manuscript and claim matrices only after verification.

The current Stage A commit must not contain formal result CSVs, result figures,
`final/`, manuscript revisions, or a result-bearing `run_manifest.json`.
The same restriction applies to Stage A.3. This revision is not a formal run
and does not create `run_1`, `run_2`, `final/`, or `rerun_verification.json`.

### Stage A.1 inherited contracts

This revision replaces the original pairing implementation's precomputed-logit
reassignment with fold-contained source refitting, removes optimum jitter, and
replaces the marginal-SMD gate with pairwise distance and paired-difference
criteria. It also replaces the recoverability bootstrap-tail p-value with the
locked paired-prediction-swap permutation p-value and adds raw versus
outer-training-standardized score-scale audits. The four revised contracts are
covered by tests in the parent commit.

### Stage A.2 inherited method revision

This revision changes the primary pairing comparison to the complete-control
comparison `P+Q+R+D+I-real` versus
`P+Q+R+D+I-optimal_matched`. The weak `P+I` comparison remains explicitly
secondary and its p-value cannot enter the four-test primary TF-IDF FDR family.

Stage A.2 adds two label-alignment benchmarks. The label-only benchmark predicts
the interviewer logit from outer-training class means, with outer-training rows
predicted by inner cross-fitting that excludes the recipient. The
label-conditioned benchmark subtracts those same class means from both source
scores and predicts `I_within` from `P_within`, Q, R and D. Outer-test class
means are always estimated from outer-training source scores only. These are
recoverability sensitivity analyses, not deployment or causal analyses.

For every optimal inner-training assignment, the runner records 100 random
non-self derangements and the locked distance/paired-feature quality fields.
All inner assignments must be no-self and one-to-one, and at least 80% of
optimal inner assignments must pass the existing matching gate. Otherwise the
formal manifest is marked `training_match_quality_limited`; pairing dependence
must not be interpreted as reliable under that status.

The source-score scale-sensitive rule is substantive and fixed at 0.05:
`source_score_scale_sensitive=True` iff the absolute raw-versus-standardized
R2 difference is at least 0.05, the absolute Spearman difference is at least
0.05, or raw and standardized R2 have opposite signs.

The Stage A.2 parent commit contains only the inherited plan, implementation
code, tests, and non-result input-contract documentation. Stage A.3 supersedes
its `code_commit` for any future formal run.

### Stage A.3 locked engineering revision

Stage A.3 does not add an analysis, change a research question, change an
estimator, change a threshold, or change a primary test family. It only removes
ambiguous primary-pairing aliases and reduces repeated fitting under the same
fold-containment contract.

For each inner-validation partition, one source model is fit on inner-training
rows and predicts the complete inner-validation partition once. For outer test,
one source model is fit on outer-training rows and predicts the complete
outer-test partition once. Optimal and eligible random donor assignments then
reorder these held-out logits within the same partition. This is not a
cross-person permutation of heterogeneous OOF scores: every reordered value in
a partition comes from one model that did not see that partition.

Random mismatch prediction draws and complete Fake-D draws are restricted to
the primary representation `tfidf` and main repeat `1`. For all representations
and repeats, the deterministic optimal pairing comparison remains available;
100 random mismatch assignments are still generated as distance references for
the matching gate. Repeats 2--10 and non-primary representations therefore do
not repeat the 100 random label-model controls. Complete Fake-D draws are not
repeated outside the primary representation/main repeat; deterministic real-D
baselines remain available for repeat stability.

The formal manifest records the requested configuration, the estimated
held-out source prediction invocations, the legacy repeated-fit estimate, the
random-prediction draw scope, the Fake-D draw scope, and the resulting fitting
configuration. These are workload controls only and do not alter scientific
interpretation.

## 3. Frozen inputs and hashes

The following files were present at inspection and are the only authoritative
inputs used by the implementation. Hashes are recorded here to make later
input drift explicit. The runner records fresh hashes in each formal run
manifest and fails on a locked mismatch where a frozen hash is required.

| Input | Path | SHA-256 at plan lock |
|---|---|---|
| repeated participant-level splits | `analysis_v2/00_splits/repeated_5fold_splits_10x5.csv` | `ceaddff172bf9582987cf76ff0522dcfdd0fe90c5b3014565bcfaacf003703ae` |
| structural baseline features | `analysis_v2/12_submission_audit/structural_baseline_features.csv` | `70c439c4a78ab7bea2b1106f59d06214ae0fa9bf09540bd24c859350cc2fce4a` |
| structural feature dictionary | `analysis_v2/12_submission_audit/structural_feature_dictionary.md` | `fe3c56b0ebd51118bffee4ef2091c5088fad0d60a3537ac67c3242f52f97f821` |
| frozen D-P counts | `analysis_v2/04_c5_controls/domain_count/input.csv` | `7687c70d3770b7e8902900ec3c117d8a679fc6d718abf1c2a7f8d1361b07ae33` |
| D-P/D-All source audit | `analysis_v2/10_source_importance/07_strict_joint_models/domain_source_audit.json` | `8a9b2694bfbaa17d67e75e60545b87acda21e37da871bf3bac7dd3a180d5a864` |
| participant speech text table | `analysis_v2/01_inputs/c2_participant_speech.csv` | `c520c713922e671431b40382238d64d0b07779ad5e77d905c9224a6f66b4ec9b` |
| interviewer speech text table | `analysis_v2/01_inputs/c3_interviewer_speech.csv` | `cf1278335dd4e7ee27dbff13bb792a7c1875341dd9412900c0f72a69f06c5e48` |
| frozen turn/prompt table | `analysis_v2/01_inputs/c4_turns_official142_frozen.csv` | `763cd3857a430f7a8dc6ee997d2820b07787f5e30ddceaee8bf83329fedc1a9a` |
| MPNet cache manifest | `analysis_v2/03_embedding_robustness/model_1_all_mpnet_base_v2/run_manifest.json` | `0c5c025e9f3bc926646dea389edbd217b44e3e082310ec1d0f06cc30b43cdedd` |
| BGE cache manifest | `analysis_v2/03_embedding_robustness/model_2_bge_large_en_v1_5/run_manifest.json` | `7a98e4941323fe14503fa7b194d456d746414883446204e8daad0908d50b729c` |

The cohort contract is 142 official train+dev participants, 43 positive and
99 negative, with participant-level 10-repeat x 5-fold membership. The split
file label column must agree with the frozen participant labels. Official test
participants are not admitted.

If a restricted or frozen input is unavailable, the runner writes
`input_availability.json`, records the missing item and reason, skips only the
affected representation/derivation, and labels that output
`not_run_data_unavailable`. It never downloads a model, calls an external API,
creates substitute data, or writes interview source text to public outputs.

## 4. Feature blocks

All feature blocks are numeric and are fit/standardized only inside the training
fold of the model that consumes them.

### P: participant source score

`P` is the participant-only source model decision logit (`participant_logit`),
not a high-dimensional embedding in the primary explanation model.

### Q: quantity and interaction distribution

Q is fixed to the following existing structural columns:

```text
length_only__participant_word_count
length_only__interviewer_word_count
length_only__total_word_count
length_only__interview_duration_sec
interaction_structure__participant_turn_count
interaction_structure__interviewer_turn_count
interaction_structure__total_turn_count
interaction_structure__participant_mean_words_per_turn
interaction_structure__interviewer_mean_words_per_turn
```

### R: protocol path and coverage

R starts with every non-constant, non-near-zero-variance member of the frozen
protocol block:

```text
protocol_structure__interviewer_question_count
protocol_structure__clinical_question_count
protocol_structure__unique_protocol_prompt_count
protocol_structure__clinical_prompt_coverage_count
protocol_structure__non_explicit_interviewer_turn_count
```

When `c4_turns_official142_frozen.csv` is available, the runner derives the
following without reading labels or re-encoding transcript content:

```text
first_clinical_prompt_position_normalized
last_clinical_prompt_position_normalized
clinical_prompt_density
clinical_prompt_count_early_third
clinical_prompt_count_middle_third
clinical_prompt_count_late_third
clinical_nonclinical_transition_count
max_consecutive_clinical_prompt_run
protocol_path_depth
```

For every non-empty frozen `prompt_id` observed on an annotated interviewer
turn, it also derives both a count and a presence variable. IDs are sanitized
for column names and a dictionary maps each derived name to the frozen ID.
These variables are selected by the frozen input schema, not by outcome
performance.

Definitions are locked as follows:

- normalized positions use the ordered turn index within a participant, with
  the first turn at 0 and the last turn at 1; a one-turn interview is 0;
- clinical prompt density is clinical interviewer prompt count divided by
  interviewer turn count, with denominator one when no interviewer turn exists;
- early/middle/late counts use normalized positions in `[0, 1/3)`, `[1/3, 2/3)`,
  and `[2/3, 1]`;
- transitions count adjacent interviewer-turn clinical/non-clinical state
  changes in either direction;
- maximum clinical run is the longest consecutive run of clinical interviewer
  prompts in interviewer-turn order;
- protocol path depth is the number of distinct non-empty frozen prompt IDs
  present for that participant.

The path feature dictionary and audit are written before modeling. If a derived
feature is constant or near-zero variance, it remains in the audit but is
excluded from R. The original frozen structural file is never modified.

### D: participant-derived symptom domains

D is exactly the following ten columns from the frozen D-P input, in this order:

```text
anhedonia_interest
appetite_weight
concentration_psychomotor
depressed_mood
functioning_impairment
mental_health_history
protective_or_absent_symptom
self_worth_guilt
sleep_fatigue_energy
suicide_self_harm
```

The implementation checks the D-P source audit, input hash, participant IDs,
finite numeric values, and column order. It does not rename D-P to D-All or
infer an independent clinical measurement.

## 5. Strict cross-fitted source scores

For every representation, repeat, and outer fold:

1. split participants using the frozen membership table;
2. generate outer-training P and I logits by inner cross-fitting, so each
   outer-training row's source score comes from a model that did not train on
   that row;
3. tune dense source-model C only inside the outer-training data when the
   existing frozen dense-source implementation requires it;
4. fit the selected source model on all outer-training rows;
5. generate outer-test logits without seeing outer-test labels;
6. save only outer-test rows in `source_score_crossfit.csv` with
   `source_score_scope=outer_test`.

The TF-IDF source uses the existing tokenizer/vectorizer and balanced
liblinear logistic model with fixed `C=1.0`. MPNet and BGE use the existing
byte-verified caches and dense source classifier with the locked C grid
`[0.01, 0.1, 1.0, 10.0]`. No embedding is re-encoded.

The source-score output includes at least:

```text
participant_id,label,repeat,outer_fold,representation,
participant_logit,interviewer_logit,
participant_probability,interviewer_probability,
source_model_C,source_score_scope
```

## 6. Direct recoverability analysis

The explanation target is the interviewer source logit `Z_I`. All 16 subsets
are run for each available representation and requested repeat:

```text
null
P
Q
R
D
P+Q
P+R
P+D
Q+R
Q+D
R+D
P+Q+R
P+Q+D
P+R+D
Q+R+D
P+Q+R+D
```

The model is ridge regression. Numeric scaling is a training-fold pipeline;
the alpha grid is `[0.01, 0.1, 1.0, 10.0, 100.0]`; inner folds are three;
alpha is selected by inner mean squared error; ties select the larger alpha.
The empty/null prediction is the outer-training mean target, not the global
cohort mean.

The OOF metrics are R2 relative to the paired outer-training-mean null MSE,
MAE, RMSE, Spearman rho, and MSE improvement. `explanation_oof_predictions`
retains the observed target, null prediction, model prediction, and fold
metadata for every subset.

For scale sensitivity, every explanation prediction also retains the
outer-training interviewer-score mean/SD and participant-score mean/SD. The
fold metrics report raw and outer-training-standardized target/prediction
metrics. The scale-sensitivity table compares pooled raw metrics with the
corresponding fold-standardized metrics; no test-fold mean or SD is used to
standardize a score.

Recoverability inference keeps a participant bootstrap confidence interval for
delta MSE, but its p-value is from 10,000 paired within-participant swaps of
the null and full predictions. The global FDR table uses this permutation
p-value, not a bootstrap tail proportion.

The scale audit uses the substantive 0.05 rule locked in Section 2. A result is
`source_score_scale_sensitive` only when at least one of the following holds:
`abs(raw R2 - standardized R2) >= 0.05`,
`abs(raw Spearman - standardized Spearman) >= 0.05`, or raw and standardized
R2 have opposite signs. A `1e-10` numerical-difference rule is not used.

## 7. Label-only recoverability benchmark

For every representation, repeat and outer fold, the label-only benchmark uses
the interviewer source logits already generated by strict source cross-fitting.
For outer-training rows, the predicted value is the mean interviewer logit in
the recipient's class computed from the other inner-training rows. For an
outer-test participant, the predicted value is the outer-training class mean
for that participant's observed label. Neither the outer-test interviewer
logit nor the recipient's own source score enters a class mean.

The benchmark reports R2, MSE improvement relative to the outer-training
overall-mean null, MAE, RMSE and Spearman rho. It is a common-label alignment
reference only and does not support a deployment or individual-level
recoverability claim. Outputs are `label_only_recoverability.csv` and
`label_only_recoverability_metrics.csv`.

## 8. Label-conditioned recoverability

Within each outer-training fold, interviewer and participant logits are
residualized against the recipient label using inner cross-fitted class means:

```text
I_within_train = I_train - class_mean_I_excluding_validation
P_within_train = P_train - class_mean_P_excluding_validation
```

For outer-test rows, class means use the complete outer-training source scores
only:

```text
I_within_test = I_test - outer_train_class_mean_I[label_test]
P_within_test = P_test - outer_train_class_mean_P[label_test]
```

Ridge models predict `I_within` from the locked subsets `P_within`, Q, R, D,
and `P_within+Q+R+D`, with training-fold scaling and the inherited alpha grid
and inner-MSE tuning. The full subset reports R2, MAE, RMSE, Spearman rho and
delta MSE relative to the label-conditioned null. This analysis is a key
interpretation sensitivity and is outside the four-test primary FDR family.
Outputs are `label_conditioned_recoverability_oof.csv` and
`label_conditioned_recoverability_metrics.csv`.

For a deterministic descriptive flag, "high" is locked as total full-model
R2 `>= 0.50` and "near zero" as absolute full label-conditioned
`P_within+Q+R+D` R2 `<= 0.05`. The same per-representation/repeat flag is
attached to the subset rows for auditability.
If total recoverability is high while label-conditioned recoverability is near
zero, the result is marked
`recoverability_largely_shared_outcome_alignment`. If label-conditioned
R2 is `> 0.05`, the permitted interpretation is that
the observable blocks reconstructed interviewer-score variation beyond
binary-label alignment; repeat stability is still required before calling this
pattern stable.

## 9. Block Shapley summary

For the main repeat, the four block contributions P/Q/R/D are computed from
the jointly aligned OOF predictions of all 16 subsets. All 24 block-entry
orders are averaged. The score is the same null-relative R2 used in Section 6,
so the efficiency check is:

```text
sum(block Shapley deltas) = R2(P+Q+R+D) - R2(null)
```

Here `R2(null)=0` by definition. Five thousand participant bootstrap
resamples use one common sampled participant index vector for every subset in a
resample. Negative contributions are retained, no contribution is normalized
to a percentage, and the output scope explicitly says statistical explanation
rather than causal contribution.

## 10. Residual interviewer signal

For each outer fold, a full P+Q+R+D ridge model is fit on outer-training
participants and predicts the outer test interviewer logit. Predictions for
outer-training participants are themselves produced by a fresh inner
cross-fitting pass. The residual is:

```text
interviewer_residual = interviewer_logit - predicted_interviewer_logit
```

The label models compare:

```text
base:     P + Q + R + D
enhanced: P + Q + R + D + interviewer_residual
```

They use L2 logistic regression, balanced class weights, liblinear, intercept,
2000 iterations, and C grid `[0.01, 0.1, 1.0, 10.0]`; scaling and tuning are
training-fold only. Main inference uses 5,000 participant-paired bootstrap
resamples and 10,000 paired participant swaps. Primary delta AUC is
enhanced-minus-base. Delta PR-AUC uses the same direction; delta Brier and
delta log loss use base-minus-enhanced improvement direction.

## 11. Fake-D negative control

One hundred deterministic Fake-D draws are generated per outer fold only for
TF-IDF/repeat 1. The ten D-P columns move as an intact row. The train and test
partitions are permuted independently, each assignment is a non-identity
derangement, and no train donor ID can occur in a test assignment. Labels are
never read to make a draw. Recipient and donor IDs are written to
`fake_domain_assignments.csv`.

Each draw runs the full P+Q+R+D-fake explanation and its corresponding residual
label increment in that primary configuration. Deterministic real-D baselines
remain available for every requested representation/repeat. The result reports
the real-D value, fake-D draw mean/SD, 2.5th/97.5th percentiles, and
participant/draw uncertainty. The primary negative-control contrasts are real-D
R2 minus fake-D mean R2 and real-D residual delta AUC minus fake-D mean residual
delta AUC.

## 12. Pairing dependence

Pairing is a statistical pairing-dependence analysis, not an identified human
interaction effect. Matching variables are locked to interviewer word count,
interviewer turn count, clinical question count, clinical prompt coverage,
non-explicit interviewer turn count, D-P domain breadth, and D-P total evidence
count. No label, PHQ score, source probability, correctness, or post-result
variable is used. PCA is disabled (`0` components) in this contract, so no
data-dependent dimensionality choice is introduced.

For each outer training and outer test partition independently, Hungarian
assignment minimizes standardized Euclidean distance with the recipient/donor
diagonal forbidden. Standardization is fit on outer training and reused for
outer test. Draw 0 is one deterministic `optimal_matched` assignment. One
hundred random non-self derangements are generated for the distance reference
in every configuration, but random mismatch prediction draws 1--100 are
executed only for TF-IDF/repeat 1. No jitter is added to the optimum and random
draws are not described as near-optimal.

Source prediction is cached at the held-out partition level before donor
reordering. For each outer training fold, an inner-validation source model is
fit only on inner-training rows and predicts every inner-validation source
record once; optimal or random donor assignments then reorder those logits
within that same inner-validation partition. For outer test, a source model is
fit only on outer-training rows and predicts every outer-test source record
once; donor assignments reorder those logits within the same outer-test
partition. A precomputed outer OOF logit is never permuted or reassigned to a
recipient, and the cached inner-validation split seed is held fixed across
assignment draws.

Balance is assessed pairwise rather than by post-match marginal SMD, because a
one-to-one permutation of the same partition would make its marginal SMD zero
by construction. The audit reports per-feature mean/median/q95 absolute
standardized paired differences, per-pair standardized Euclidean distances,
total assignment cost, and the distance distribution of 100 random non-self
assignments. The primary gate requires no self-match, one-to-one donors, an
optimal mean distance below the random-reference 5th percentile, and at least
5 of 7 features with mean absolute standardized paired difference below 0.50.
If this gate fails, the run is marked
`matching_not_adequate_for_primary_interpretation` and no pairing advantage is
claimed.

For each outer-training inner-validation partition, the runner additionally
writes `pairing_inner_training_quality.csv` with assignment type, inner fold,
recipient count, mean/median/q95 distance, 100-draw random-reference q05,
optimal distance percentile, all seven mean absolute standardized paired
differences, feature pass count, no-self, one-to-one and matching adequacy.
Every optimal inner assignment must be no-self and one-to-one, and at least 80%
must pass the distance and 5-of-7 feature gate. Failure is recorded as
`training_match_quality_limited` and prevents a reliable pairing-dependence
interpretation.

The late-fusion logit-level label models compare:

```text
weak:     P, P + I-real, P + I-matched
complete: P+Q+R+D, P+Q+R+D+I-real, P+Q+R+D+I-matched
```

The primary estimates are the complete-control real-minus-optimal delta AUC,
complete-control optimal-minus-no-I delta AUC, complete-control delta PR-AUC,
and absolute probability change. The primary randomization test swaps the
`full+I-real` and `full+I-optimal_matched` probabilities within participant.
The weak `P+I-real` versus `P+I-optimal_matched` result remains available with
scope `secondary_weak_control_pairing` and is not the primary FDR p-value.
Both participant and matching-draw uncertainty are retained. Draw 0 is the
optimal matched comparison; random draws are a reference distribution and are
reported separately. The generic fields `p_value`,
`real_minus_matched_delta_auc_observed`, `ci_low`, `ci_high`, and
`optimal_minus_no_i_ci_low/high` refer to the complete-control primary result;
pairing delta/stability fields without a `weak_` prefix likewise refer to the
complete-control result; weak-control results retain only their `weak_`-
prefixed names.

## 13. Statistical families and repeat stability

TF-IDF is the primary representation. The four primary test families are
fixed before results:

1. full P+Q+R+D recoverability MSE improvement versus null;
2. real-D versus Fake-D explanation performance;
3. residual delta AUC after P+Q+R+D;
4. real interviewer versus matched interviewer pairing delta AUC.

For item 4, the TF-IDF p-value is specifically
`full_real_minus_optimal_p_value` from the complete-control pairing
randomization test. The weak-control p-value is never substituted. BH-FDR is
applied to those four TF-IDF family p-values. MPNet and BGE are
representation-stability results and are not added to the primary family;
the runner also writes a global all-representation BH sensitivity table.

Repeat 1 is the main estimate. Repeats 1 through 10 are summarized as
stability rows for full recoverability R2, P/Q/R/D Shapley deltas, residual
delta AUC, optimal-pairing delta AUC, and deterministic real-D baselines. The
real-D minus Fake-D contrast is complete only for TF-IDF/repeat 1 under the
locked draw scope; it is not fabricated for other representations or repeats.
Repeats are not treated as independent samples and no repeat-level t test is
run.

## 14. Formal output contract

Each `run_1` or `run_2` contains:

```text
input_availability.json
structural_feature_audit.csv
structural_feature_audit.json
interviewer_path_coverage_features.csv
interviewer_path_feature_dictionary.md
interviewer_path_feature_audit.json
source_score_crossfit.csv
explanation_oof_predictions.csv
explanation_subset_metrics.csv
explanation_tuning.csv
recoverability_fold_metrics.csv
recoverability_scale_sensitivity.csv
label_only_recoverability.csv
label_only_recoverability_metrics.csv
label_conditioned_recoverability_oof.csv
label_conditioned_recoverability_metrics.csv
explanation_block_shapley.csv
residual_signal_oof_predictions.csv
residual_signal_metrics.csv
residual_signal_deltas.csv
fake_domain_assignments.csv
fake_domain_explanation_results.csv
fake_domain_residual_results.csv
fake_domain_explanation_oof_predictions.csv
fake_domain_residual_oof_predictions.csv
fake_domain_summary.csv
pairing_assignments.csv
pairing_balance.csv
pairing_pairwise_balance.csv
pairing_random_reference.csv
pairing_oof_predictions.csv
pairing_metrics.csv
pairing_deltas.csv
pairing_draw_stability.csv
pairing_inner_training_quality.csv
explanation_repeat_stability.csv
global_fdr_sensitivity.csv
run_manifest.json
figures/interviewer_score_recoverability_forest.{png,svg,pdf}
figures/explanation_block_shapley_forest.{png,svg,pdf}
figures/residual_signal_forest.{png,svg,pdf}
figures/pairing_dependence_plot.{png,svg,pdf}
figures/fake_domain_negative_control_plot.{png,svg,pdf}
```

`run_manifest.json` records code commit, all input hashes, cohort and split
hash, feature blocks, dropped constant/near-zero features, model and grid
settings, label-only and label-conditioned class-mean scopes, the 0.05 scale
sensitivity threshold, bootstrap/permutation counts, matching and Fake-D
settings, the random-distance and random-prediction draw scopes, estimated
fitting invocations, held-out source prediction reuse, the 80% inner-quality
gate, training-match-quality status, seeds, output hashes, and completion
status. `verify_reruns` compares the manifest output hash maps and the locked
plan/code fields; only a verified match may be promoted to `final/`.

No output file contains `text`, `spoken_text`, `verified_text`, an exact quote,
or an embedding cache payload. The source-score and feature outputs are
participant-level derived signals only.

## 15. CLI contract

The command is:

```powershell
python scripts/run_interviewer_signal_explanation.py `
  --output-root analysis_v2 `
  --mode formal `
  --run-id 1 `
  --representations tfidf mpnet bge
```

`--mode smoke` uses a deterministic small synthetic-compatible cohort only for
code-path testing and never writes `final/`. `--verify-reruns` compares
`run_1` and `run_2`; `--promote-final` is accepted only when verification is
successful. The current task runs neither formal mode nor verification mode.

## 16. Allowed and prohibited interpretation

Allowed after verified results:

- observable blocks statistically reconstruct much or little of the
  interviewer model output under the locked operation;
- conditional attenuation or residual label increment under the specified
  models;
- real D outperforming Fake D as evidence that D explanation depends on true
  participant-domain correspondence;
- real interviewer outperforming matched interviewer as pairing dependence;
- representation-sensitive or representation-stable interpretation.

Prohibited:

- causal mediation or causal interviewer bias effect;
- signal composition percentage;
- complete explanation;
- independent mechanism;
- label leakage as a conclusion from this module;
- individual interaction effect unless explicitly qualified as a descriptive,
  non-identified pairing comparison;
- D-P described as an independent clinical measurement or D-All.

## 17. Stage A implementation and test checklist

The implementation files are fixed to:

```text
reanalysis_v2/interviewer_signal_explanation.py
scripts/run_interviewer_signal_explanation.py
tests/test_interviewer_signal_explanation.py
```

The test suite must cover source outer-test isolation, inner cross-fitted
training logits, all 16 block subsets, fold-contained scaling, zero-variance
handling, D-P hash/column contract, intact-row Fake-D draws and no-self
assignments, label-blind/non-cross-fold matching, train/test donor isolation,
cross-fitted residual training predictions, Shapley symmetry and efficiency,
joint participant bootstrap indexing, formal/smoke output separation, rerun
hash verification, and absence of interview source text in public outputs. The
Stage A.2 tests additionally cover complete-control pairing FDR selection over
weak-control direction reversals, outer-training-only label means, exclusion
of the label-only recipient, outer-test label-conditioned residuals, shared
label-only versus within-label simulated recoverability, inner matching-quality
output and the 80% gate, the substantive scale threshold and R2 direction
flip.

Before the Stage A commit:

1. run the new tests through a red-green cycle;
2. run all existing tests plus the new test file;
3. inspect `git diff --check` and the exact staged file list;
4. commit plan/code/tests only;
5. push `codex/submission-audit` and report the Stage A SHA.
