## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-04T00:16:45.216513+00:00
- Verification Status: ANALYZED
- Version Label: validation_v1

## Validation Report

- **Source**: reanalysis_v2 frozen computational outputs
- **Overall Confidence**: CAUTION

### Statistical Findings

| Finding | Test | Value | Confidence |
|---|---|---|---|
| TF-IDF input-source ranking | 10,000 paired swaps; 40-test BH family | No ROC-AUC or PR-AUC comparison survived FDR | SOLID for this cohort/design |
| TF-IDF threshold dependence | Participant-level OOF decision metrics | C2 sensitivity 0.047 at 0.5 vs 0.512 nested | SOLID for this model/cohort |
| Representation robustness | 10,000 paired swaps; 120-test joint BH family | 5 tests survived FDR; no C2-vs-C3 test survived | CAUTION: model-dependent patterns |
| C5 controls | 10,000 paired swaps; 30-test BH family | No comparison survived FDR | CAUTION: limited n and control variance |
| C4 controls | 10,000 paired swaps; 8-test BH family | Template-vs-shuffle and one length-matched Macro-F1 contrast survived | CAUTION: semantics not isolated |

### Warnings

| Type | Detail | Affected |
|---|---|---|
| Cohort scope | Official train+dev only; no independent cohort | Generalizability |
| Threshold sensitivity | Default 0.5 can reverse decision-level impressions | Macro-F1, sensitivity, specificity |
| Small sample | n=142 with 43 positives; dense models may be unstable | All estimates |
| Multiple testing | Corrected within four preregistered families; raw p-values must not be interpreted alone | All paired tests |
| Calibration reuse | Metrics are OOF within the same cohort and are not deployment calibration estimates | Brier, ECE, slope/intercept |
| Control interpretation | C5 domain states are extracted evidence states; C4 controls do not isolate natural-language semantics | Mechanistic claims |

### Fallacy Scan

- **Coverage**: 11/11 fallacy types checked

| Fallacy | Severity | Detail | Guardrail |
|---|---|---|---|
| Simpson's paradox | NOTE | No subgroup claim was made; subgroup reversal was not tested because subgroup analysis was not preregistered here. | Do not infer subgroup consistency. |
| Ecological fallacy | NOTE | Unit of analysis and inference is the participant. | Retain participant-level wording. |
| Berkson's paradox | CAUTION | DAIC-WOZ is a selected research cohort. | Do not generalize to population screening without external validation. |
| Collider bias | NOTE | No post-exposure covariate adjustment was used in the classifiers. | Avoid causal interpretations. |
| Base-rate neglect | CAUTION | Positive prevalence is 43/142; sensitivity/specificity are accompanied by PR-AUC and prevalence. | Report the cohort base rate with decision metrics. |
| Regression to the mean | NOTE | No pre-post extreme-group design is present. | Not applicable to the primary comparison. |
| Survivorship bias | CAUTION | Analysis uses available official train+dev records and does not establish missingness representativeness. | State cohort inclusion explicitly. |
| Look-elsewhere effect | CAUTION | Many contrasts were run, but all were retained and BH-corrected within four frozen families. | Interpret q-values, not selected raw p-values. |
| Garden of forking paths | CAUTION | The v2 plan and changelog freeze choices, but this is not a time-stamped external preregistration. | Treat claims as audit findings, not confirmatory clinical evidence. |
| Correlation != causation | CAUTION | Predictive associations and scripted protocol signals cannot establish causal mechanisms. | Use associational language. |
| Reverse causality | CAUTION | Cross-sectional interview content and PHQ-8 labels do not establish temporal direction. | Avoid directional causal claims. |

### Reproducibility

- **Method**: artifact hashes, split-contract checks, leakage checks, row/participant checks, family/permutation checks, deterministic unit tests; no second full end-to-end embedding rerun
- **Verdict**: CANNOT_VERIFY as an independent reproduction
- **Artifact integrity**: 177/177 inventoried computational artifacts passed byte/hash verification
- **Strict contract audit**: 101/101 checks passed
- **Reason for conservative verdict**: ARS marks reproducibility VERIFIED only after an independent full rerun; the current validation confirms internal integrity and executability but does not substitute for that rerun.
