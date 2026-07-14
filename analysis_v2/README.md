# DAIC-WOZ source-signal audit: authoritative entry point

## Locked claim

This study audits the sources of PHQ-8 label-predictive signals in semi-structured interviews and tests whether full-transcript performance can be attributed directly to participant language alone.

The authoritative Chinese manuscript is [`06_tables_figures/manuscript_submission_zh.md`](06_tables_figures/manuscript_submission_zh.md). The companion [`06_tables_figures/main_text/manuscript_restructured_zh.md`](06_tables_figures/main_text/manuscript_restructured_zh.md) is its four-group structure index; older drafts are audit history only.

The data-only snapshot is [`12_submission_audit/frozen_data_summary.md`](12_submission_audit/frozen_data_summary.md).

## Reader-facing narrative

The manuscript uses four result groups: single-source predictability; conditional increments; interviewer-score recoverability and residual label increment; and formal pairing destruction/Fake-D identification sensitivity. Real pairing retains extra ranking information, while the negative Fake-D contrast prevents a simple claim that D-P symptom content uniquely absorbs interviewer information.

## Evidence hierarchy

- **Main analysis:** `12_submission_audit/`, repeat 1 of the existing five-fold participant split fixed as the analysis-locked split before the final post-audit sensitivity run, one cross-fitted prediction per participant. This is not formal preregistration.
- **Split-stability supplement:** existing 10 x 5 outputs in `02_tfidf_main/` and `10_source_importance/07_strict_joint_models/`.
- **C5 boundary:** C5 is an LLM-assisted, clinically guided derived representation. It is not a natural source equivalent to raw participant speech.
- **Domain-control boundary:** the frozen ten-domain count table is D-P, a participant-language-derived control built from reviewed C5 spans. A qualifying full-interview D-All table is not currently available and is not inferred from D-P, protocol variables, or symptom-evidence-only files.
- **E-DAIC boundary:** module 14 is a small descriptive supplement and is excluded from the title, abstract, core claims, and main figures.
- **Formal identification boundary:** `10_source_importance/08_identification_sensitivity/run_formal_10x/` contains the frozen ten-repeat MPNet/BGE early-concat pairing and Fake-D outputs; late fusion and PHQ partition remain supplemental.
- **D-P manual boundary:** `04_c5_controls/blind_quality_audit/packet_1/` contains a 30-person blinded packet. Reviewer files are blank, so no human agreement or consensus metrics are claimed.

## Terminology contract

Allowed: source-dependent predictive signal, protocol-related signal, interaction-structure signal, conditional incremental information, PHQ-8 label prediction.

Not supported by this design: interviewer bias as an identified causal effect, label leakage, contamination, inflated performance, clinical diagnosis, reliable screening, biomarker, or clinical deployment.

Single-source performance measures predictive sufficiency. It does not measure independent contribution. Conditional increments require joint comparisons such as M5 versus M3 or M4 versus M3.

Existing L3/L4 results that use `04_c5_controls/domain_count/input.csv` must be reported as conditional results after D-P, not as results after a full-interview symptom-domain control. The source-variant gate and audit record are in [`10_source_importance/07_strict_joint_models/domain_source_variants_plan.md`](10_source_importance/07_strict_joint_models/domain_source_variants_plan.md) and [`10_source_importance/07_strict_joint_models/domain_source_audit.json`](10_source_importance/07_strict_joint_models/domain_source_audit.json).

## Reproducibility levels

1. **Code/test reproducibility:** the public repository runs unit tests and synthetic fixtures without restricted interview data.
2. **Result integrity verification:** the public repository verifies committed derived outputs, schemas, interpretation guardrails, and SHA-256 hashes with `python scripts/verify_submission_artifacts.py`.
3. **Full data-to-result reproduction:** requires restricted DAIC-WOZ source data and the private quote-bearing C5 review source; public CI does not claim this level.

## Restricted rerun

The submission audit uses no external model API. It requires the restricted reviewed C5 source directory for hash-and-offset verification:

```powershell
python scripts\run_submission_audit.py `
  --c5-source-root <restricted_reviewed_c5_official142_directory>
```

Public outputs contain no interview quote text. See `12_submission_audit/run_manifest.json` and `output_manifest_sha256.csv` for the frozen contract.
