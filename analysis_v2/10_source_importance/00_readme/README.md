# Source-Level Incremental Contribution and Importance Analysis

This module quantifies the incremental contribution and relative importance of different signal sources in DAIC-WOZ PHQ-8 classification.

## Data Source Verification

**CRITICAL**: This analysis MUST use the correct data source:
- 142 participants (99 negative, 43 positive) for training/validation
- 47 participants for test set (separate, NOT included in main analysis)
- Verified via `input_freeze_manifest.json`: `official_test_in_primary_analysis = false`

**DO NOT USE**:
- 189-participant old version
- Official test set in primary analysis

## Directory Structure

```
10_source_importance/
├── 00_readme/
│   └── README.md (this file)
│
├── 01_source_level_table/
│   ├── source_level_features.csv
│   ├── source_level_feature_manifest.md
│   ├── source_feature_correlation_matrix.csv
│   └── source_feature_correlation_summary.md
│
├── 02_incremental_models/
│   ├── incremental_model_results.csv
│   ├── incremental_model_results.md
│   ├── incremental_model_oof_predictions.csv
│   └── incremental_model_interpretation.md
│
├── 03_source_importance/
│   ├── permutation_importance_auc.csv
│   ├── permutation_importance_logloss.csv
│   ├── source_importance_rank_summary.md
│   └── source_importance_interpretation.md
│
├── 04_domain_multivariable/
│   ├── domain_multivariable_results.csv
│   ├── domain_leave_one_out_results.csv
│   ├── domain_permutation_importance.csv
│   └── domain_importance_summary.md
│
├── 05_sensitivity_checks/
│   ├── sensitivity_source_set_results.csv
│   ├── sensitivity_no_protocol_results.csv
│   ├── sensitivity_no_domain_results.csv
│   └── sensitivity_summary.md
│
└── 06_manuscript_ready/
    ├── table_source_incremental_models.md
    ├── table_source_importance.md
    ├── table_domain_importance.md
    ├── source_importance_results_zh.md
    └── source_importance_results_en.md
```

## Analysis Plan

### Step 1: Build Source-Level Feature Table
- Merge participant-level OOF probabilities from all conditions
- Add raw domain presence/count features
- Verify data integrity (142 participants, correct labels)
- Output: `source_level_features.csv`

### Step 2: Source Feature Correlation Analysis
- Compute Spearman and Pearson correlations between source probabilities
- Focus on: c2 vs c5, c5 vs domain_count, template_only vs template_presence
- Output: `source_feature_correlation_matrix.csv`

### Step 3: Incremental Models (M0-M4)
- Train source-level logistic regression with repeated 5-fold CV
- Prespecified models:
  - M0: c2_participant_prob only
  - M1: c2 + domain_count
  - M2: c2 + template_presence
  - M3: c2 + domain_count + template_presence
  - M4: c2 + domain_count + template_presence + c5_evidence
- Output: `incremental_model_results.csv`

### Step 4: Source Importance Analysis
- Permutation importance for M3
- Optional: subset-based dominance analysis
- Output: `permutation_importance_auc.csv`

### Step 5: Domain Multivariable Analysis
- 5-domain multivariable model (FDR-significant domains only)
- Leave-one-domain-out analysis
- Output: `domain_multivariable_results.csv`

### Step 6: Sensitivity Checks
- domain_presence vs domain_count
- template_presence vs template_only
- With/without C5 quote
- Output: `sensitivity_source_set_results.csv`

## Interpretation Guidelines

**DO NOT interpret as:**
- Causal source attribution
- Evidence that model "understands" depression
- Proof of specific quote semantic contribution

**DO interpret as:**
- Descriptive source-importance audit
- Relative contribution under current experimental setup
- Signal overlap and redundancy analysis

## Reproducibility

All analyses use:
- Same 142 participants (verified)
- Same repeated 5-fold splits (`repeated_5fold_splits_10x5.csv`)
- Same OOF probabilities from primary and control analyses
- Prespecified model sequence (no post hoc model selection)
- **Fixed CV**: OOF predictions averaged across all 10 repeats (not overwritten)
- **Permutation importance**: 100x repeated shuffles per fold per feature with fixed seed (20260704)
- **Bootstrap CI**: 5000x paired participant bootstrap for incremental model comparisons

## Status

- [x] Step 1: Source-level feature table
- [x] Step 2: Correlation analysis
- [x] Step 3: Incremental models (M0-M5) with fixed CV averaging
- [x] Step 4: Source importance (permutation, 100x repeated, correct ranking)
- [x] Step 5: Domain multivariable with leave-one-out
- [x] Step 6: Sensitivity checks
- [x] Bootstrap CI for M0-M5 comparisons
- [ ] Subset-based dominance analysis (optional, not done)
- [ ] Domain figure (needs matplotlib)
- [x] Manuscript-ready tables

## Key Results (from fixed analysis)

### Incremental Models
- M0 (C2 only): AUC baseline
- M1 (+domain_count): significant improvement
- M2 (+template_presence): improvement
- M3 (+both): best performance
- **M4 (+C5): NO improvement over M3** → C5 quote adds zero incremental value

### Source Importance (Permutation, M3)
- Most important: domain_count_prob
- Second: template_presence_prob
- Not important: c2_prob (near-zero or negative ΔAUC)

### Domain Importance (Leave-one-out)
- Top domains: functioning_impairment, mental_health_history
- Clinical-context domains dominate over PHQ-core symptoms

### Sensitivity
- Results robust to domain_count vs domain_presence encoding
- Results robust to template_presence vs template_only encoding
- C5 quote consistently adds zero incremental value

## Last Updated

2026-07-04 (fixed CV averaging, permutation repeats, bootstrap CI, correct ranking)
