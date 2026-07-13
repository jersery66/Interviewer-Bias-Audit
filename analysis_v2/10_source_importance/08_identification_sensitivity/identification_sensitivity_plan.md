# Identification sensitivity analysis contract

**Lock date:** 2026-07-13  
**Status:** post-audit, non-preregistered identification analysis  
**Priority:** this contract supersedes D-All as the next identification analysis; no D-All model is run under this contract.

## Question

The existing D-P conditional analysis shows that the interviewer increment is smaller after participant-derived symptom-domain counts are added. That attenuation has at least four compatible explanations: participant-related interaction information, feature-engineering advantage of D-P, label-isomorphic symptom coding, or generic complexity/noise. The analyses below are fixed before inspecting their results and are intended to distinguish those explanations.

## Frozen inputs

- Cohort: the same 142 official train+dev participants and labels used by the embedding-incremental analysis.
- Participant and interviewer representations: the two existing, byte-hashed embedding caches (`all-mpnet-base-v2` and `bge-large-en-v1.5`). No re-encoding is allowed.
- Splits: the committed repeated five-fold participant-level split membership.
- Classifier: the existing fold-contained logistic-regression pipeline, C grid `(0.01, 0.1, 1.0, 10.0)`, inner three-fold tuning, and the existing base seed.
- D-P: the ten columns in `04_c5_controls/domain_count/input.csv`; its source audit remains authoritative.
- Protocol matching variables: interviewer word count, interviewer question count, clinical question count, unique protocol prompt count, clinical prompt coverage count, and non-explicit interviewer-turn count. Labels are never used to construct pairings or fake controls.

## Experiment 1: paired interviewer-source destruction

Participant embeddings remain in their original row. Interviewer embeddings are supplied as:

1. **Real:** identity mapping `P_i + I_i`.
2. **Random:** a seeded, label-blind derangement of interviewer rows.
3. **Structure-matched random:** a seeded one-to-one assignment that minimizes standardized Euclidean distance on the six frozen interviewer/protocol length and question variables, with the diagonal forbidden. The assignment uses only the frozen structural table and does not use labels or outcomes.

The same draw IDs and seeds are used across both embedding models and both fusion methods. Each condition uses the same outer folds and inner tuning. The primary descriptive comparison is AUC; PR-AUC, Brier and log loss are also retained. Random and matched-random results are sensitivity distributions, not a new confirmatory FDR family.

Locked draw count: 50 random draws and 50 structure-matched draws, plus the real pairing. The pairing utility must record every donor ID, self-match count, one-to-one status, and structural matching cost.

## Experiment 2: Fake-D negative control

The D-P matrix is row-permuted across participants with a seeded non-identity permutation. Row permutation preserves the exact ten-dimensional vector distribution, each column's mean/variance/nonzero count, and the observed domain-count schema while destroying participant-domain correspondence. Fifty permutations are generated with fixed draw IDs and seeds. For each permutation, the same folds and tuning are used for `P`, `P + Fake-D`, and `P + Fake-D + I`; the real D-P comparison is rerun in the same execution for paired bookkeeping.

The intended contrast is whether a structurally similar but participant-unmatched control attenuates the interviewer increment. A Fake-D attenuation is evidence against a symptom-content-specific interpretation; it is not evidence that the interviewer signal is causal.

## Experiment 3: PHQ-alignment partition

The ten D-P columns are partitioned before results into:

- **D-PHQ:** `anhedonia_interest`, `appetite_weight`, `concentration_psychomotor`, `depressed_mood`, `self_worth_guilt`, `sleep_fatigue_energy`, `suicide_self_harm`.
- **D-Extra:** `functioning_impairment`, `mental_health_history`, `protective_or_absent_symptom`.

The bundled columns are operational domain groups, not claims of one-to-one PHQ item identity. The analysis compares `P + D-PHQ` versus `P + D-PHQ + I`, and `P + D-Extra` versus `P + D-Extra + I`, under the same two embeddings, two fusion methods, folds and tuning. It is a descriptive decomposition of label alignment versus broader symptom/context structure.

## Missing/short data rules

No new text is created. Empty or zero embedding rows remain the frozen all-zero cache vectors. A missing or non-finite structural/domain value fails the run rather than being imputed. A pairing donor is never the recipient itself. Fake-D permutations are rejected if they are the identity mapping.

## Outputs and interpretation boundary

Outputs are written under `analysis_v2/10_source_importance/08_identification_sensitivity/` in a new immutable run directory and include configuration, input hashes, donor/permutation ledgers, participant-level OOF predictions, metric summaries, and an output SHA-256 inventory. These are post-audit sensitivity results, not preregistered confirmatory findings. They may support statements about compatibility with participant-related information, structure-only controls, and PHQ-aligned versus extra-domain attenuation; they may not establish causality, prove that interviewer bias is absent, or upgrade D-P to D-All.
