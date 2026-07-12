# DAIC-WOZ source-signal audit

This repository contains the analysis code, tests, frozen derived outputs, and Chinese submission draft for a source-signal audit of PHQ-8 prediction in semi-structured interviews.

The authoritative manuscript is [`analysis_v2/06_tables_figures/manuscript_submission_zh.md`](analysis_v2/06_tables_figures/manuscript_submission_zh.md). The submission audit, claim boundaries, post-audit sensitivity plan, frozen results, and hashes are in [`analysis_v2/12_submission_audit/`](analysis_v2/12_submission_audit/).

## Reproducibility boundary

1. **Code/test reproducibility:** public CI runs unit tests and synthetic fixtures.
2. **Result integrity verification:** public CI verifies committed derived outputs, schemas, guardrails, and SHA-256 hashes.
3. **Full data-to-result reproduction:** requires restricted DAIC-WOZ data and the private quote-bearing C5 review source. A passing CI run does not claim clean-clone reproduction from raw data.

```powershell
python -m pytest -q
python scripts/verify_submission_artifacts.py
```

No external model API is required for the public verification steps. Public C5 audit artifacts contain hashes and offsets, not interview quotes.
