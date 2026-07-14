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

The organized validation dataset will contain the model span fields and blank
human fields only. It will include a deterministic 24-participant sampling
roster stratified by frozen binary label, participant speech length tertile,
and D-P domain breadth. Selection cannot use model correctness, source score,
or any result. Participant transcript text remains an external review
reference and is not copied into the public output.

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
- `c5_human_validation_candidate_dataset.csv`
- `c5_human_validation_template.xlsx`

R analyses:

- `feature_block_source_table.csv`
- `block_increment_contrasts.csv`
- `r_absorption_sensitivity_oof.csv`
- `r_absorption_sensitivity_metrics.csv`
- `run_manifest.json`

Formal execution uses separate `run_1/` and `run_2/` directories. A
`final/` directory is created only after every core output hash is identical
between the two runs.

## Permitted conclusions

The results may describe statistical score recoverability, descriptive block
increments, and whether adding residuals after R changes predictive
performance under the locked operationalization. They may not claim that R
causes interviewer judgments, that R is an independent mechanism, that the
Shapley value is a signal-composition percentage, or that C5 has human–machine
validity before independent human annotations are collected.
