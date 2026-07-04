# Bridge Interpretation Analyses - Summary

This document summarizes the supplementary analyses created for the DAIC-WOZ reanalysis.

## Completed Analyses

### 01. Signal Decomposition Table ✅

**Location:** `09_bridge_interpretation/01_signal_decomposition/`

**Files:**
- `signal_source_decomposition.csv` - Structured CSV table
- `signal_source_decomposition.md` - English Markdown version
- `signal_source_decomposition_zh.md` - Chinese Markdown version

**Content:**
Six signal layers identified:
1. Participant language (C2) - contains signal but threshold-sensitive
2. Symptom evidence text (C5) - usable but domain coverage may capture much of the signal
3. Symptom-domain coverage (domain_presence/count) - strong predictor, but features are derived from C5 extraction pipeline
4. Interviewer protocol structure (C4) - protocol structure carries signal
5. Representation family - input effects depend on representation
6. Threshold decision rule - default threshold creates artifacts

**Updates (2026-07-04):**
- Modified interpretation to be more cautious: domain_presence/count "may capture much of the discriminative information" rather than "drives much of the signal"
- Added note that domain features are derived from C5 extraction pipeline, not fully independent sources

**Status:** COMPLETE

### 02. Domain-Level Description Table ✅

**Location:** `09_bridge_interpretation/02_domain_level_description/`

**Files:**
- `domain_presence_group_comparison.csv` - Presence analysis results
- `domain_count_group_comparison.csv` - Count analysis results
- `domain_summary_table.csv` - Combined summary table
- `domain_summary_table.md` - Markdown summary with cautious interpretation
- `figure_domain_group_difference.png/svg` - NOT CREATED (needs matplotlib)

**Key Findings:**
Significant domains (FDR < 0.05) in presence analysis:
- depressed_mood: OR=3.16, p=0.005, FDR=0.012
- appetite_weight: OR=5.67, p=0.009, FDR=0.018
- suicide_self_harm: OR=10.74, p<0.001, FDR=0.001
- functioning_impairment: OR=8.00, p<0.001, FDR<0.001
- mental_health_history: OR=7.20, p<0.001, FDR<0.001

**Important note:** The discriminative performance is contributed by both PHQ-core symptoms AND clinical-context domains (functioning_impairment, mental_health_history), not by PHQ-core symptoms alone.

**Updates (2026-07-04):**
- Added cautious interpretation in `domain_summary_table.md` explaining that domain_presence/count performance is contributed by multiple extracted domains (symptom domains + functional impairment + mental health history), not by PHQ-core symptoms alone

**Status:** COMPLETE (except figure, which needs matplotlib)

### 03. Case-Based Error Analysis ⚠️ (MOSTLY COMPLETE)

**Location:** `09_bridge_interpretation/03_case_based_error_analysis/`

**Files:**
- `case_analysis_table.md` - Case analysis table (Markdown) ✅ UPDATED with C1 fields
- `selected_cases_for_manuscript.csv` - Selected cases (CSV) ✅ UPDATED with C1 fields
- `case_selection_log.md` - Selection criteria and counts
- `candidate_cases.csv` - ✅ GENERATED (142 rows, eligibility flags for 4 case types)
- `generate_candidate_cases.py` - Script to generate candidate_cases.csv
- `run_case_analysis.py` - Selection script (blocked by sandbox)
- `inspect_cases_readonly.py` - Read-only inspection script

**Selected Cases (UPDATED 2026-07-04):**
- Case A (414): C2 threshold artifact demonstration ✅ Includes C1 prob/pred
- Case B (356): Protocol signal vs participant signal ✅ Includes C1 prob/pred
- Case C (343): C5 close to domain features ✅ Includes C1 prob/pred
- Case D (305): Full transcript vs single source conflict ✅ NOW INCLUDES C1 prob/pred to support interpretation

**Updates (2026-07-04):**
- Added C1 prob and C1 default prediction fields to `selected_cases_for_manuscript.csv` and `case_analysis_table.md`
- Case D now shows: C1 pred=1 (false positive), C2 pred=0 (correct), demonstrating source conflict

**Status:** COMPLETE (including candidate_cases.csv)

### 04. Bootstrap CI for Primary Input-Source Tables ⏸️ (OPTIONAL)

**Status:** NOT STARTED (optional). The primary input-source tables (C1-C5, representation, C4/C5 controls) do not yet have bootstrap CIs.

**Note:** Bootstrap CIs have been completed for source-level incremental models in `10_source_importance/` (5000x paired participant bootstrap for M0-M5 comparisons). These provide CI coverage for the main incremental contribution findings. All-primary-table CI remains optional and is not required for the current manuscript.

## Technical Issues Encountered

### 1. F: Drive 100% Full
- The main git repository is on F: drive
- Git operations (commit, gc) fail with "No space left on device"
- **Solution needed:** Free up space on F: drive, or move .git directory to E: drive

### 2. Missing Python Packages
- matplotlib not available (figure generation skipped)
- tabulate not available (Markdown generation done manually)
- **Solution:** Install packages in managed environment, or create figures separately

## Next Steps

1. **Create domain figure** (install matplotlib or create manually)
2. **Optional: Bootstrap CI for primary tables** (if time permits for English paper; source-level model CI already done in `10_source_importance/`)

## File Inventory

```
09_bridge_interpretation/
├── 01_signal_decomposition/
│   ├── signal_source_decomposition.csv ✅
│   ├── signal_source_decomposition.md ✅ (updated 2026-07-04)
│   └── signal_source_decomposition_zh.md ✅ (updated 2026-07-04)
├── 02_domain_level_description/
│   ├── domain_presence_group_comparison.csv ✅
│   ├── domain_count_group_comparison.csv ✅
│   ├── domain_summary_table.csv ✅
│   ├── domain_summary_table.md ✅ (updated 2026-07-04)
│   ├── figure_domain_group_difference.png ❌ (needs matplotlib)
│   ├── figure_domain_group_difference.svg ❌ (needs matplotlib)
│   └── run_domain_analysis.py ✅
├── 03_case_based_error_analysis/
│   ├── case_analysis_table.md ✅ (updated 2026-07-04, added C1 fields)
│   ├── selected_cases_for_manuscript.csv ✅ (updated 2026-07-04, added C1 fields)
│   ├── candidate_cases.csv ✅ (142 rows, eligibility flags for A/B/C/D)
│   ├── case_selection_log.md ✅
│   ├── generate_candidate_cases.py ✅
│   ├── run_case_analysis.py ✅
│   ├── inspect_cases_readonly.py ✅
│   └── check_case_d.py ✅
└── 04_bootstrap_ci_optional/
    └── (empty — source-level model CI completed in 10_source_importance/)
```

## Reproducibility

All analyses were performed using the correct data source:
- 142 participants (99 negative, 43 positive) for training/validation
- 47 participants for test set (separate)
- Data verified via `input_freeze_manifest.json` and split files

## Cross-Reference

- **Bootstrap CI**: Completed for source-level incremental models (M0-M5) in `10_source_importance/` (5000x paired participant bootstrap)
- **Candidate cases**: Full candidate pool with eligibility flags available in `09_bridge_interpretation/03_case_based_error_analysis/candidate_cases.csv` (142 rows)

