# Manuscript Statistics and Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the leakage-prone source-probability stacking analysis with fold-safe joint source models, recompute multiplicity-controlled inference, regenerate publication figures, and typeset the Chinese Markdown manuscript as a complete paper.

**Architecture:** Fit each incremental model directly from raw source groups inside the frozen 10×5 participant splits, so vectorizers and scalers are learned from the training fold only. Save participant-level repeated-OOF predictions, derive paired participant bootstrap intervals and two-sided paired permutation p-values, apply BH-FDR within prespecified families, then drive every table, figure annotation, and manuscript claim from those audited CSV outputs.

**Tech Stack:** Python 3.12, pandas, NumPy, SciPy, scikit-learn, statsmodels, matplotlib, unittest, Markdown.

---

### Task 1: Lock the strict statistical contract with failing tests

**Files:**
- Create: `tests/test_source_importance_strict.py`
- Create: `reanalysis_v2/source_importance_strict.py`

- [ ] **Step 1: Write failing tests** for fold-local feature fitting, 10 OOF predictions per participant, deterministic paired permutation tests, BH-FDR monotonicity, and absence of participant-aggregated source probabilities from model inputs.

```python
class StrictSourceImportanceTests(unittest.TestCase):
    def test_model_specs_use_raw_source_groups(self):
        self.assertEqual(MODEL_SPECS["M3"], ("c2_text", "domain_count", "template_presence"))

    def test_each_participant_has_one_prediction_per_repeat(self):
        fold_rows, participant_rows = run_incremental_models(self.frame, self.splits, n_repeats=2)
        self.assertTrue((fold_rows.groupby("participant_id").size() == 2).all())

    def test_bh_fdr_is_monotone_in_sorted_p_order(self):
        q = bh_fdr(np.array([0.04, 0.001, 0.02]))
        ordered = q[np.argsort([0.04, 0.001, 0.02])]
        self.assertTrue(np.all(np.diff(ordered) >= 0))
```

- [ ] **Step 2: Run the tests and verify RED.**

Run: `E:\Anaconda\python.exe -m pytest tests/test_source_importance_strict.py -q`

Expected: import failure because `source_importance_strict.py` does not yet exist.

### Task 2: Implement fold-safe incremental modeling and inference

**Files:**
- Modify: `reanalysis_v2/source_importance_strict.py`
- Create: `scripts/run_source_importance_strict.py`
- Test: `tests/test_source_importance_strict.py`

- [ ] **Step 1: Implement raw source loading** for C2/C3/C5 text, domain counts, domain presence, and template presence with participant-ID joins and frozen-cohort assertions (`n=142`, `43/99`).
- [ ] **Step 2: Implement fold-local preprocessing** with separate TF-IDF vectorizers per text group, train-only numeric scaling, sparse concatenation, and balanced logistic regression.
- [ ] **Step 3: Implement repeated 10×5 OOF prediction** and save both fold-level and participant-averaged outputs.
- [ ] **Step 4: Implement paired participant bootstrap CI, two-sided paired permutation p-values, and BH-FDR** for the seven incremental AUC comparisons.
- [ ] **Step 5: Implement group permutation importance** using raw test-fold group shuffling and participant-level aggregated perturbed probabilities; infer three source effects with BH-FDR.
- [ ] **Step 6: Implement domain leave-one-out** for presence and count representations with participant-level paired inference and one 20-test exploratory BH family.
- [ ] **Step 7: Run tests and verify GREEN.**

Run: `E:\Anaconda\python.exe -m pytest tests/test_source_importance_strict.py -q`

Expected: all strict source-importance tests pass.

### Task 3: Run the strict analysis and audit outputs

**Files:**
- Create: `analysis_v2/10_source_importance/07_strict_joint_models/model_metrics.csv`
- Create: `analysis_v2/10_source_importance/07_strict_joint_models/model_auc_ci.csv`
- Create: `analysis_v2/10_source_importance/07_strict_joint_models/incremental_comparisons_fdr.csv`
- Create: `analysis_v2/10_source_importance/07_strict_joint_models/permutation_importance_fdr.csv`
- Create: `analysis_v2/10_source_importance/07_strict_joint_models/domain_leave_one_out_fdr.csv`
- Create: `analysis_v2/10_source_importance/07_strict_joint_models/participant_oof_predictions.csv`
- Create: `analysis_v2/10_source_importance/07_strict_joint_models/run_manifest.json`
- Create: `analysis_v2/10_source_importance/07_strict_joint_models/output_manifest_sha256.csv`

- [ ] **Step 1: Run the strict analysis.**

Run: `E:\python3.12.8\python.exe scripts/run_source_importance_strict.py`

Expected: successful completion with 142 participants, 50 outer folds, fixed seeds, and all listed outputs.

- [ ] **Step 2: Verify output contracts**: no missing OOF probabilities, exactly 10 predictions per participant before averaging, q-values in `[0,1]`, and all SHA-256 entries match.

### Task 4: Regenerate figures with defensible significance labels

**Files:**
- Modify: `analysis_v2/10_source_importance/generate_figures.py`
- Modify: `analysis_v2/08_reproducibility_scripts/tests/test_reporting.py`
- Replace: `analysis_v2/06_tables_figures/figures/figure_1_source_decomposition_framework.{svg,pdf,tiff}`
- Replace: `analysis_v2/06_tables_figures/figures/figure_2_incremental_model_auc.{svg,pdf,tiff}`
- Replace: `analysis_v2/06_tables_figures/figures/figure_3_permutation_importance.{svg,pdf,tiff}`
- Replace: `analysis_v2/06_tables_figures/figures/figure_4_domain_leave_one_out.{svg,pdf,tiff}`

- [ ] **Step 1: Add a failing reporting test** asserting that stars are assigned only from `q_value < 0.05`, legends define the BH-FDR family, and descriptive effects without valid inference receive no star.
- [ ] **Step 2: Verify RED.**
- [ ] **Step 3: Redesign Figure 2** as model AUC estimates plus a paired ΔAUC forest panel; show 95% CI and q-value, with `*`, `**`, `***` based only on BH-adjusted q.
- [ ] **Step 4: Redesign Figure 3** as a participant-level group permutation importance forest plot with CI, q-value, and directional zero line.
- [ ] **Step 5: Redesign Figure 4** as domain leave-one-out forest panels with CI and only corrected significance marks.
- [ ] **Step 6: Render Python-only SVG/PDF/600-dpi TIFF exports and verify the reporting tests pass.**

### Task 5: Rewrite and typeset the Chinese manuscript

**Files:**
- Modify: `analysis_v2/06_tables_figures/manuscript_draft_zh.md`
- Create: `analysis_v2/06_tables_figures/manuscript_review_and_revision_log.md`

- [ ] **Step 1: Replace every result value from the strict audited CSVs**, never from hard-coded legacy text.
- [ ] **Step 2: Rebuild Methods** with cohort, raw source definitions, fold-local preprocessing, 10×5 repeated CV, conditional scope of OOF inference, paired permutation tests, bootstrap intervals, BH-FDR families, seeds, and leakage controls.
- [ ] **Step 3: Rebuild Results** in model order M0–M5 and distinguish point estimates, nominal CI exclusion, and FDR-controlled significance.
- [ ] **Step 4: Calibrate Discussion and limitations** so predictive association is not written as mechanism or causality; remove unsupported claims about patient-language independence.
- [ ] **Step 5: Insert all figures and captions** immediately after first citation, format tables consistently, add abbreviations/statistical-note blocks, data/code availability, and a reference section.
- [ ] **Step 6: Record every substantive change and unresolved limitation** in the revision log.

### Task 6: Final reproducibility and manuscript QA

**Files:**
- Modify: `analysis_v2/06_tables_figures/figure_contract.md`
- Modify: `analysis_v2/06_tables_figures/figure_qa_notes.md`
- Modify: `analysis_v2/06_tables_figures/validation_report.md`
- Modify: `analysis_v2/06_tables_figures/output_manifest_sha256.csv`

- [ ] **Step 1: Run all reproducibility tests.**
- [ ] **Step 2: Re-run the strict analysis and compare SHA-256 outputs for determinism.**
- [ ] **Step 3: Audit manuscript numbers against CSVs, figure labels against q-values, Markdown image paths, and all figure file formats.**
- [ ] **Step 4: Inspect rendered figures at publication size for clipping, overlap, font embedding, and grayscale legibility.**
- [ ] **Step 5: Review `git diff --check` and the complete scoped diff before handoff.**
