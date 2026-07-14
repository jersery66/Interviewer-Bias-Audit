# Interviewer-signal revision: locked material and sensitivity plan

## Scope

This independent module starts from formal result commit
`ed66d68aa20e789fbd4829b44c49362b784a61e2`. It does not modify or overwrite
`analysis_v2/15_interviewer_signal_explanation/final/`.

The first task is to organize the materials needed for a future C5
human–machine validation study. The current files contain model-generated C5
evidence, source-traceability checks, post-hoc quote corrections, model
repeatability, and a separate two-rater error-attribution study. They do not
contain independent human span/domain annotations for C5. No human answer is
to be inferred or filled in.

## Locked C5 organization

The canonical official-142 C5 span source is:

`F:/数据库/DAIC-WOZ/processed_research/latest_usable_data_official142_20260702/04_c5_evidence_sources/c5_quotes_only_v3_gpt55_reviewed_spans_official142.csv`

Its SHA-256 is locked as
`794d7b43ebd95d57ac4a90a4217953a9bea27e98ff91b13c31b0a06f9aa48c7e`.
The same bytes are present in the 20260626 and 20260630 dated packages.
The current formal traceability audit contains 1,137 rows for 141
participants, ten model-assigned domains, polarity, source/quote hashes, and
post-hoc correction status.

The restricted owner-only crosswalk will contain model span fields, linkage,
and blank human fields. Separate blinded reviewer packets will contain only an
opaque review case ID, deidentified quote, opaque local-context reference, and
blank human coding fields. The deterministic 24-participant sampling roster is
stratified by frozen binary label, participant speech length tertile, D-P
domain breadth, and C5 evidence-count tertile. Selection cannot use model
correctness, source score, or any result. Participant transcript text remains
an external restricted review reference and is not copied into the public
output.

## R block source and descriptive increments

The block-source table is descriptive and locked as follows:

| Block | Source side | Operational content |
| --- | --- | --- |
| P | participant-side | participant source logit |
| D | participant-derived | ten D-P domain counts |
| Q | mixed participant/interviewer/process | frozen length and interaction quantities |
| R | interviewer-side protocol structure | frozen protocol counts plus audited path/prompt coverage features |

`block_increment_contrasts.csv` will be calculated from the frozen
`analysis_v2/15_interviewer_signal_explanation/final/explanation_subset_metrics.csv`
without refitting models or changing the 16 locked subsets. It will report
R, P, D, Q alone; full R2; full minus R; P+R minus R; R+D minus R;
P+R+D minus R; and P+Q+R+D minus R. These are descriptive score
recoverability contrasts, not new significance tests and not causal effects.

## R absorption sensitivity

The primary R absorption sensitivity is TF-IDF, repeat 1. It uses strict
outer-fold isolation and the existing locked source-model, ridge, and label
model settings. Within every outer fold it constructs:

```text
residual_without_R = I - predicted_I(P+Q+D)
residual_full      = I - predicted_I(P+Q+R+D)
```

It compares:

```text
P+Q+D
P+Q+D+residual_without_R
P+Q+R+D
P+Q+R+D+residual_full
```

The new result is exploratory and outside the frozen four-test FDR family.
It reports OOF AUC, PR-AUC, Brier, log loss, and descriptive residual
recoverability. It must not be called causal mediation or interpreted as a
causal effect of R.

The existing MPNet/BGE block-increment results remain available as descriptive
representation context. No new dense source model is refit for this optional
sensitivity unless a later plan revision explicitly expands its scope.

## Inputs and hashes

The base formal result inputs remain frozen. Key hashes are recorded here so
the new outputs can be audited:

| Input | SHA-256 |
| --- | --- |
| formal explanation subset metrics | `a550ba89d14e0f470d87327002bc7e5cf2c3f8cb9c5f5e691de3fefe72c1b726` |
| formal explanation block Shapley | `6791aaa4a2309afede7fe3871eb8851c8668a28053a79d37f8a7d28b907818f3` |
| formal explanation repeat stability | `e8de05fdbd14ff1b8fa70027435dce88af4db08a0e706e2d784c5df2b3885bfe` |
| formal source scores | `01ba43c969ec0e2cbbec8214b8dc90e6869cd1118339a237f4914a79e9d3dd80` |
| formal run manifest | `e665d0b003bd4ed22e1880901720d02ea7d58c198c3b3879bdb6c807f0533394` |
| official-142 reviewed C5 spans | `794d7b43ebd95d57ac4a90a4217953a9bea27e98ff91b13c31b0a06f9aa48c7e` |
| current C5 traceability audit | `2a11e33d8996c98b6ba3f40ca406e67188ed77b8f09117d1bde4490e3649b186` |
| current C5 traceability summary | `256fb807c8bd8ab497adc63fed22b280bccca9bb10cad56f27d1331b7400e47b` |

## Outputs

Material organization:

- `c5_agreement_material_inventory.csv`
- `c5_agreement_material_inventory.md`
- `c5_agreement_data_gap.md`
- `c5_human_validation_sampling_roster.csv`
- `c5_human_validation_field_dictionary.md`
- `c5_human_validation_source_hash.json`
- no C5 quotes, reviewer packets, or owner linkage in the public result tree

Only when two explicitly separate restricted directories are supplied:

- reviewer directory: `c5_reviewer_a_blank.csv/.xlsx`,
  `c5_reviewer_b_blank.csv/.xlsx`, and
  `c5_missing_evidence_audit_blank.csv`
- owner directory: `c5_owner_crosswalk.csv/.xlsx`

R analyses:

- `feature_block_source_table.csv`
- `block_increment_contrasts.csv`
- `source_model_performance_by_repeat.csv`
- `source_model_performance_summary.csv`
- `r_absorption_sensitivity_oof.csv`
- `r_absorption_sensitivity_metrics.csv`
- `stratified_fake_domain_assignments.csv`
- `stratified_fake_domain_explanation_results.csv`
- `stratified_fake_domain_residual_results.csv`
- `stratified_fake_domain_summary.csv`
- `run_manifest.json`

Formal execution uses separate `run_1/` and `run_2/` directories. A
`final/` directory is created only after every core output hash is identical
between the two runs.

## Required source benchmark and targeted Fake-D sensitivity

Before a formal run, the frozen
`analysis_v2/15_interviewer_signal_explanation/final/source_score_crossfit.csv`
is summarized directly; participant-only (`P`) and interviewer-only (`I`)
probabilities are not refit. Repeat 1 receives a 5,000-draw label-stratified
participant bootstrap for descriptive 95% intervals. Repeats 1--10 are
reported with mean, SD, median, minimum, and maximum. Repeats are not treated
as independent observations and no repeat-level significance test is run.

The original non-stratified Fake-D output remains unchanged. A separate
`targeted_post_result_sensitivity` uses 100 TF-IDF/repeat-1 draws. In each
outer training and outer test partition, complete D-P rows are deranged within
label 0 and within label 1. A label stratum with fewer than two participants
raises an availability error; the implementation never falls back to the
non-stratified control. It retains effect estimates, the 100-draw Fake-D
reference distribution, and participant-plus-draw bootstrap 95% confidence
intervals only. It does not calculate, store, or interpret `p_value`,
`q_value`, or BH adjustment, and it enters no FDR family.

## C5 human-validation boundary

The 24-person roster covers the frozen binary label, participant speech-length
tertile, D-P domain-breadth tertile, and model-evidence-count tertile. It is
not selected using model correctness, source scores, residuals, or mismatch
status. The public result tree contains no exact quote. In an explicitly
supplied restricted review directory, each rater receives a blinded packet
with a deidentified review case ID, quote, local review reference, and blank
human fields only. A separate owner-only crosswalk retains participant IDs,
labels, model fields, traceability, and the review-case mapping. The agreement
script refuses to calculate candidate-span agreement when either rater file
has incomplete validity labels.

## Stage A.5 locked implementation correction

This Stage A.5 revision changes only revision-module implementation, tests,
and this plan. It does not modify `analysis_v2/15`, any frozen input, the
feature blocks, CV partitions, models, hyperparameters, seeds, draw counts,
or the Stage B analysis design. It does not create `run_1/`, `run_2/`, or
`final/`.

### Frozen-input gate before any result-directory creation

Before a future `run_1` directory can be created, the runner must verify the
following exact SHA-256 values. Any missing file or mismatch raises an error
before `mkdir`, so no partial formal output is created.

| Locked input | SHA-256 |
| --- | --- |
| Stage 15 explanation subset metrics | `a550ba89d14e0f470d87327002bc7e5cf2c3f8cb9c5f5e691de3fefe72c1b726` |
| Stage 15 source score crossfit | `01ba43c969ec0e2cbbec8214b8dc90e6869cd1118339a237f4914a79e9d3dd80` |
| Stage 15 formal manifest | `e665d0b003bd4ed22e1880901720d02ea7d58c198c3b3879bdb6c807f0533394` |
| Stage 15 original Fake-D explanation results | `682da41d7c873d3b3fef75e764961cc351e769edc9a9a0b264c451d552c50df0` |
| Stage 15 original Fake-D residual results | `6ac82cb7eb5af954d8cda70e082f2aca8edc8e93d30b3f0fa7648691990950f3` |
| C5 traceability audit | `2a11e33d8996c98b6ba3f40ca406e67188ed77b8f09117d1bde4490e3649b186` |
| Canonical official-142 C5 spans | `794d7b43ebd95d57ac4a90a4217953a9bea27e98ff91b13c31b0a06f9aa48c7e` |

The manifest will record the verified input-hash map. This gate is an input
integrity check, not a new analytical decision. When the linked worktree does
not contain the untracked canonical C5 source, the runner resolves the nearest
ancestor repository root that contains the locked canonical path, then applies
the same SHA-256 gate before creating any result directory.

### Targeted label-stratified Fake-D negative control

The existing real-D result and 100 within-label, within-partition Fake-D
draws remain unchanged. The result retains effect estimates, the 100-draw
Fake-D reference distribution, and 5,000 participant-plus-draw bootstrap
95% confidence intervals. It does not calculate, store, or interpret
`p_value`, `q_value`, or BH adjustment. The revision manifest must state:

```text
targeted_negative_control_sensitivity = true
modifies_frozen_primary_FDR_family = false
new_targeted_sensitivity_FDR_family = false
inference = effect estimates and participant-and-draw bootstrap intervals only
```

This is a targeted post-result negative-control sensitivity, not a new FDR
family. The roster `selection_rule` explicitly includes the C5 evidence-count
tertile as well as label, speech-length tertile, and D-P-breadth tertile.

### C5 blinded review packet and owner-only linkage

Candidate spans are first assembled into an owner-only crosswalk. It retains
the frozen participant identifier, label, model evidence fields, traceability
fields, and deterministic `review_case_id`. It is written only under an
explicit restricted owner directory and is never part of public revision
output.

Each reviewer receives a separate blind file with only:

```text
review_case_id
deidentified_quote
local_context_reference
human_span_valid
human_domain
human_polarity
rater_id
notes
```

The reviewer packet must not expose participant ID, label, model domain,
model polarity, model review/correction status, model correctness, or an
identifying transcript reference. A separate participant-level full-text
missing-evidence audit form is used for `human_missing_evidence`; it is not
mixed into candidate-span agreement.

### C5 scoring contract after independent coding

The agreement function accepts two blinded rater sheets plus the owner-only
crosswalk for model comparisons. It calculates A-versus-B validity Cohen's
kappa across all candidate spans. Domain and polarity agreement are calculated
only on spans marked valid by both raters. Model-versus-rater metrics are
reported separately for rater A and rater B. Model-versus-consensus metrics
are absent unless a distinct consensus table is explicitly supplied.

Invalid candidate spans may leave `human_domain` and `human_polarity` blank.
Valid candidate spans must supply both values. Candidate-span validity is not
a full-transcript recall estimate; missing-evidence review remains a separate
participant-level full-text check.

### Stage A.5 verification contract

Tests must cover the frozen-input gate before result creation; the absence of
Fake-D p/q/FDR fields while retaining bootstrap intervals; blind reviewer
column exclusion and owner linkage retention; the updated agreement scopes;
the evidence-count selection rule; and all existing Stage 16 safety tests.
Before a code-only commit, run the targeted revision tests, the full test
suite, `python -m py_compile reanalysis_v2/interviewer_signal_revision.py`,
and `git diff --check`. Confirm that no Stage 16 formal result directory was
created.

## Permitted conclusions

The results may describe statistical score recoverability, descriptive block
increments, and whether adding residuals after R changes predictive
performance under the locked operationalization. They may not claim that R
causes interviewer judgments, that R is an independent mechanism, that the
Shapley value is a signal-composition percentage, or that C5 has human–machine
validity before independent human annotations are collected.
