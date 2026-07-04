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
- `candidate_cases.csv` - ❌ NOT YET GENERATED (see Issue A below)
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

**Status:** MOSTLY COMPLETE (pending candidate_cases.csv generation)

### Issue A: candidate_cases.csv Not Yet Generated

**Problem:** The analysis plan requires `candidate_cases.csv` to show the full candidate pool for each case type, but this file has not been generated yet.

**Why:** Python scripts that write files are blocked by sandbox permissions.

**Solution:** Run `generate_candidate_cases.py` locally to generate `candidate_cases.csv`.

**Instructions for user:**
```bash
cd E:/CodexWorktrees/DAIC-WOZ/reanalysis-v2/analysis_v2/09_bridge_interpretation/03_case_based_error_analysis
python generate_candidate_cases.py
```

This will generate `candidate_cases.csv` with:
- All 142 participants
- Probabilities and predictions from all conditions (C1-C5, domain_presence, domain_count, template_only, template_presence)
- Eligibility flags for each case type (eligible_A, eligible_B, eligible_C, eligible_D)

**Candidate counts (from script output):**
- Case Type A: 20 eligible
- Case Type B: 32 eligible
- Case Type C: 118 eligible
- Case Type D: 51 eligible

### 04. Bootstrap CI (Optional) ⏸️

**Status:** NOT STARTED (optional, but recommended for English paper)

**Note:** The final audit report acknowledges lack of CI as a remaining limitation. Adding bootstrap CI would increase conformity with English paper norms, but is not fatal.

## Technical Issues Encountered

### 1. F: Drive 100% Full
- The main git repository is on F: drive
- Git operations (commit, gc) fail with "No space left on device"
- **Solution needed:** Free up space on F: drive, or move .git directory to E: drive

### 2. Sandbox Permission Issues
- Python scripts that write files are blocked by sandbox
- **Workaround:** Used Write tool to create files manually, ran read-only scripts to generate content
- **Impact:** `candidate_cases.csv` not yet generated (requires local execution)

### 3. Missing Python Packages
- matplotlib not available (figure generation skipped)
- tabulate not available (Markdown generation done manually)
- **Solution:** Install packages in managed environment, or create figures separately

## Next Steps

1. **Generate candidate_cases.csv** - Run `generate_candidate_cases.py` locally
2. **Create domain figure** (install matplotlib or create manually)
3. **Optional: Run bootstrap CI analysis** (if time permits)
4. **Commit all changes** to git
5. **Push to GitHub**

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
│   ├── domain_summary_table.md ✅ (updated 2026-07-04, added cautious interpretation)
│   ├── figure_domain_group_difference.png ❌ (needs matplotlib)
│   ├── figure_domain_group_difference.svg ❌ (needs matplotlib)
│   └── run_domain_analysis.py ✅
├── 03_case_based_error_analysis/
│   ├── case_analysis_table.md ✅ (updated 2026-07-04, added C1 fields)
│   ├── selected_cases_for_manuscript.csv ✅ (updated 2026-07-04, added C1 fields)
│   ├── candidate_cases.csv ❌ (PENDING: run generate_candidate_cases.py locally)
│   ├── case_selection_log.md ✅
│   ├── generate_candidate_cases.py ✅ (creates candidate_cases.csv)
│   ├── run_case_analysis.py ✅
│   ├── inspect_cases_readonly.py ✅
│   └── check_case_d.py ✅
└── 04_bootstrap_ci_optional/
    └── (empty, not started)
```

## Reproducibility

All analyses were performed using the correct data source:
- 142 participants (99 negative, 43 positive) for training/validation
- 47 participants for test set (separate)
- Data verified via `input_freeze_manifest.json` and split files

The analysis scripts are saved in the respective directories for reproducibility.

**Important:** `candidate_cases.csv` must be generated locally before submitting the manuscript, to demonstrate that case selection was done from a documented candidate pool.

