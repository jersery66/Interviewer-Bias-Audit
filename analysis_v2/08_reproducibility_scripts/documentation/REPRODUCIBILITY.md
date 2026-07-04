# Reproducibility guide

## Frozen scope

- Cohort: official train+dev, n=142 (43 positive, 99 negative).
- Endpoint: PHQ-8 >= 10.
- Shared split: `00_splits/repeated_5fold_splits_10x5.csv` (10 repeats x 5 folds).
- Participant-level inference: average 10 repeated OOF probabilities, then 10,000 paired participant-level swaps.
- Multiple testing: BH-FDR separately for the four frozen families.
- Official test is excluded from the primary analysis.

## Environment

- Python: 3.12.8
- Platform: Windows-11-10.0.26200-SP0
- NumPy: 2.3.3
- pandas: 2.3.3
- matplotlib: 3.10.7
- Main run manifests preserve the exact scikit-learn versions, model revisions, seeds, device, pooling, cache hashes and elapsed times.

## Execution commands

Run from the isolated worktree root with the recorded virtual environment activated:

```powershell
python scripts/prepare_analysis_v2.py
python scripts/run_tfidf_main.py
python scripts/run_embedding_model.py --model mpnet
python scripts/run_embedding_model.py --model bge
python scripts/finalize_embedding_analysis.py
python scripts/run_c5_controls.py
python scripts/run_c4_controls.py
python scripts/build_final_package.py
python -m pytest -q
```

Model commands accept explicit output/model arguments as documented by `--help`; exact revisions used in the completed runs are stored in the per-model `run_manifest.json` files. Embedding caches are content-addressed and included in SHA-256 inventories.

## Integrity verification

`integrity_audit.csv` verifies every pre-existing SHA-256 inventory. `strict_contract_audit.csv` verifies primary and C4-control input hashes, train/dev-only scope, the C4 turn-to-C3/C4 142/142 alignment, split membership/non-overlap, participant-level OOF cardinality, test-family sizes, 10,000 permutations, C5 quote exclusion and exact word matching, and C4 word-count matching. `output_manifest_sha256.csv` inventories this final package.

## Known environment deviation

The frozen second embedding was changed from gte-Qwen2-7B-instruct to bge-large-en-v1.5 before execution because the available RTX 3060 Laptop GPU has 6 GB VRAM and the installed PyTorch runtime was CPU-only. The failed CUDA installation attempts and the substitution are recorded in `00_plan/changelog.md`; both actually run embedding models are reported without selection.

## Verification status

This package is internally audited and deterministic inputs/splits/artifacts are hash-pinned. It is not labeled as an independent reproduction because a second full end-to-end rerun, especially of both embedding pipelines, was not performed.
