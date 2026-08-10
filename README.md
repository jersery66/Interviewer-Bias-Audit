# DAIC-WOZ source-signal audit

This public repository contains analysis code, synthetic tests, aggregate result tables, publication figures and Chinese manuscript drafts for a source-signal audit of PHQ-8 prediction in semi-structured interviews.

The current manuscript is [`analysis_v2/06_tables_figures/manuscript_submission_zh.md`](analysis_v2/06_tables_figures/manuscript_submission_zh.md). A shorter five-section Chinese draft is available at [`analysis_v2/06_tables_figures/manuscript_five_sections_zh.md`](analysis_v2/06_tables_figures/manuscript_five_sections_zh.md). Claim boundaries and aggregate submission-audit results are under [`analysis_v2/12_submission_audit/`](analysis_v2/12_submission_audit/).

## Public-data boundary

DAIC-WOZ access is restricted by the dataset provider. This repository therefore excludes:

- transcripts, speaker turns, question-answer pairs and text-bearing model inputs;
- symptom-evidence quotes and blinded-review case text;
- participant identifiers, participant-level labels, fold assignments and out-of-fold predictions;
- reviewer returns, owner keys, consensus sheets and other private audit files;
- large embedding caches, duplicate run directories and temporary publication artifacts.

Only aggregate statistics, non-text manifests, figures, code and synthetic tests are intended for public release. See [`PUBLIC_RELEASE_POLICY.md`](PUBLIC_RELEASE_POLICY.md) for the inclusion and exclusion rules.

## Reproducibility boundary

Public materials support inspection of the analysis logic and aggregate reported results. Full data-to-result reproduction requires separately authorized DAIC-WOZ inputs and private quote-bearing audit sources. The public repository must not be interpreted as redistributing the underlying dataset.

```powershell
python -m pip install -r requirements-ci.txt
python -m pytest -q -c pytest-public.ini
```

The public test configuration deselects integration checks that require restricted source data, private review packets or removed row-level outputs. No external model API is required for the remaining tests.

## Participant-ID release gate

Before each public release, run the roster-aware scanner locally with the official participant roster stored outside this repository:

```powershell
python scripts/check_public_participant_ids.py --repo . --roster "ABSOLUTE_RESTRICTED_ROSTER_PATH"
```

The scanner examines Git-tracked searchable text and fails on roster values used in participant-identifier contexts or identifier fields. The roster and scan output remain local. GitHub Actions cannot access the restricted roster, so public CI tests the scanner with synthetic fixtures and separately enforces the public file-boundary rules.
