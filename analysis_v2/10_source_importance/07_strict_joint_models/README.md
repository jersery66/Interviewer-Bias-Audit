# Strict raw-source joint analysis

This directory is the authoritative source for the revised manuscript.

## Design

- Frozen cohort: official train+dev, `n = 142` (43 positive, 99 negative).
- Shared split: 10 repetitions × 5 participant-level folds.
- Models: M0–M5 use raw text/numeric source groups directly.
- Leakage control: TF–IDF fitting, standardization and logistic regression occur inside each outer training fold.
- Inference: 5000 participant bootstrap samples and 10,000 paired permutations.
- Multiplicity: BH-FDR within the 7 incremental, 3 permutation-importance and 20 domain-LOO families.

## Files

- `model_fold_oof_predictions.csv`: 50-fold OOF rows before participant aggregation.
- `participant_oof_predictions.csv`: one averaged OOF prediction per participant and model.
- `model_metrics.csv`, `model_auc_ci.csv`: performance and participant-bootstrap intervals.
- `incremental_comparisons_fdr.csv`: seven paired model comparisons.
- `permutation_importance_fdr.csv`: three M3 group-permutation results.
- `domain_leave_one_out_fdr.csv`: 20 structured-domain ablations.
- `run_manifest.json`: environment, seeds, input hashes and design contract.
- `output_manifest_sha256.csv`: output integrity inventory.

The revised manuscript and Figures 1–4 consume only these strict files.
