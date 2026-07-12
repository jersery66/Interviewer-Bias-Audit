# DAIC-WOZ source-signal audit: authoritative entry point

## Locked claim

This study audits the sources of PHQ-8 label-predictive signals in semi-structured interviews and tests whether full-transcript performance can be attributed directly to participant language alone.

The Chinese submission draft is [`06_tables_figures/manuscript_submission_zh.md`](06_tables_figures/manuscript_submission_zh.md). Other manuscript files are retained for audit history and are not submission-authoritative.

## Three research questions

1. Does raw participant speech contain PHQ-8 label-predictive information, and how do ranking, calibration, and operating-threshold behavior differ?
2. Do interviewer language and protocol/interaction structure also carry predictive information after explicit structure and text-quantity controls, and do they add detectable conditional information to joint models?
3. How does a clinically guided symptom-evidence representation change performance and interpretation, and what do source traceability and review limitations permit us to claim?

## Evidence hierarchy

- **Main analysis:** `12_submission_audit/`, repeat 1 of the existing five-fold participant split fixed as the analysis-locked split before the final post-audit sensitivity run, one cross-fitted prediction per participant. This is not formal preregistration.
- **Split-stability supplement:** existing 10 x 5 outputs in `02_tfidf_main/` and `10_source_importance/07_strict_joint_models/`.
- **C5 boundary:** C5 is an LLM-assisted, clinically guided derived representation. It is not a natural source equivalent to raw participant speech.
- **E-DAIC boundary:** module 14 is a small descriptive supplement and is excluded from the title, abstract, core claims, and main figures.

## Terminology contract

Allowed: source-dependent predictive signal, protocol-related signal, interaction-structure signal, conditional incremental information, PHQ-8 label prediction.

Not supported by this design: interviewer bias as an identified causal effect, label leakage, contamination, inflated performance, clinical diagnosis, reliable screening, biomarker, or clinical deployment.

Single-source performance measures predictive sufficiency. It does not measure independent contribution. Conditional increments require joint comparisons such as M5 versus M3 or M4 versus M3.

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
