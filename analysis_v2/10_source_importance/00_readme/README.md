# Source-importance analysis: authoritative status

## Use this result for the manuscript

The authoritative source-importance analysis is:

`07_strict_joint_models/`

It uses the frozen official train+dev cohort (`n = 142`, 43 positive), the shared 10 × 5 participant-level splits and direct raw-source joint models. Text vectorizers, numeric scalers and classifiers are fitted only on each outer training fold. Inference uses one averaged OOF prediction per participant, 5000 participant bootstrap samples, 10,000 paired permutations and BH-FDR within three prespecified families.

Key current results:

- M0 AUC: 0.7052.
- M3 AUC: 0.8137.
- M3 − M0: ΔAUC 0.1085, `q = 0.2450`, `ns`.
- All seven incremental comparisons: `ns` after BH-FDR.
- M3 domain-count group permutation: ΔAUC 0.1142, `q = 0.0039`, `**`.
- Template presence and participant text group permutation: `ns`.
- All 20 domain leave-one-out tests: `ns` after BH-FDR.

## Historical outputs

Directories `01_source_level_table/` through `06_manuscript_ready/` are retained only for audit history. They were produced by the superseded source-probability stacking workflow and must not be cited, copied into the manuscript or used for confirmatory conclusions. In particular, claims that M3 significantly improves on M0, that template presence is an established second source, or that individual clinical domains are confirmed are no longer valid.

## Reproduction

From the repository root:

```powershell
E:\python3.12.8\python.exe scripts\run_source_importance_strict.py
E:\python3.12.8\python.exe analysis_v2\10_source_importance\generate_figures.py
E:\Anaconda\python.exe -m pytest tests\test_source_importance_strict.py tests\test_source_importance_figures.py tests\test_manuscript_strict_consistency.py -q
```

See `07_strict_joint_models/run_manifest.json` and `output_manifest_sha256.csv` for the exact environment, seeds, input hashes and output hashes.
