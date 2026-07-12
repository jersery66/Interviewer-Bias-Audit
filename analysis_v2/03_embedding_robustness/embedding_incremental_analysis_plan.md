# Embedding incremental analysis plan

**Status:** post-submission locked supplementary conditional-increment analysis
**Lock date:** 2026-07-12
**Preregistration status:** this plan was added after the existing analyses and is **not** a preregistration or an original pre-specified analysis.  The definitions below are frozen before running this supplementary analysis.  Results must not be used to change the cohort, feature list, model levels, embedding representation, split file, or hypothesis families.

## 1. Aim and interpretation boundary

The analysis asks how the interviewer-language increment changes after adding participant language, frozen protocol-structure variables, and ten frozen symptom-domain counts.  It estimates conditional/residual increments and attenuation; it does not identify causal effects, mediation, confounding, or interviewer behaviour as a causal exposure.

The reportable terms are **conditional increment**, **residual increment**, **information overlap**, **attenuation**, and **representation stability**.  The analysis must not be described as showing that interviewer language has no value, that interviewer bias does not exist, or that prior studies are wrong.

## 2. Frozen cohort and input boundary

- Cohort: 142 participants; PHQ-8 >= 10: 43; PHQ-8 < 10: 99.
- No participant is added or removed.  Participant IDs, binary labels, and outer-fold membership come from the committed frozen files.
- Participant table: `analysis_v2/12_submission_audit/structural_baseline_features.csv`.
- Protocol table: the five `S_protocol` columns in the same frozen structural file.
- Symptom table: `analysis_v2/04_c5_controls/domain_count/input.csv`.
- Missing/non-finite protocol or domain values are an input error; no post-result imputation is permitted.
- The domain-presence table is not included in the primary model.  It remains a separately lockable sensitivity analysis and is not silently substituted for the ten count variables.

Frozen input SHA-256 values at lock time:

| Input | SHA-256 |
|---|---|
| structural baseline features | `70c439c4a78ab7bea2b1106f59d06214ae0fa9bf09540bd24c859350cc2fce4a` |
| ten-domain count input | `7687c70d3770b7e8902900ec3c117d8a679fc6d718abf1c2a7f8d1361b07ae33` |
| structural feature dictionary | `fe3c56b0ebd51118bffee4ef2091c5088fad0d60a3537ac67c3242f52f97f821` |
| repeated 5-fold membership | `ceaddff172bf9582987cf76ff0522dcfdd0fe90c5b3014565bcfaacf003703ae` |

The split hash above is the byte hash of the current Windows checkout (CRLF).  Its LF-normalized Git-content hash is `e4202fcd4d492a37fa68c579767ea940ca0629b4230374b06ddc8778e32045b3`; both hashes identify the same committed membership rows.  The runner uses the current membership rows and records the local byte hash in its manifest.

## 3. Frozen embeddings

The runner consumes existing `.npz` cache files and fails if a required cache is unavailable or its manifest hash, participant IDs, dimension, or L2 normalization check fails.  It never downloads a model or re-encodes text.

| Embedding | Dimension | Revision | Existing cache manifest |
|---|---:|---|---|
| all-mpnet-base-v2 | 768 | `e8c3b32edf5434bc2275fc9bab85f82640a19130` | `analysis_v2/03_embedding_robustness/model_1_all_mpnet_base_v2/run_manifest.json` |
| bge-large-en-v1.5 | 1024 | `d4aa6901d3a41ba39fb536a557fa166f842b0e09` | `analysis_v2/03_embedding_robustness/model_2_bge_large_en_v1_5/run_manifest.json` |

Both representations use the previously frozen 200-word chunks, 50-word overlap, normalized chunk mean, and document L2 normalization.  Only participant-speech and interviewer-speech embeddings enter the primary incremental models.  Full-transcript and symptom-evidence embeddings are not part of this core question.

The two existing embedding manifests have SHA-256 values `0c5c025e9f3bc926646dea389edbd217b44e3e082310ec1d0f06cc30b43cdedd` and `7a98e4941323fe14503fa7b194d456d746414883446204e8daad0908d50b729c`, respectively.  The cache SHA-256 values are read and re-checked from those manifests at runtime; cache files are restricted/local inputs and are not required to be committed to the public repository.

## 4. Frozen feature inventory

### Protocol structure (`R`, primary)

| Column | Definition | Type | Missing handling | Standardization |
|---|---|---|---|---|
| `protocol_structure__interviewer_question_count` | interviewer turns with a frozen prompt annotation | numeric count | reject missing/non-finite | fit `StandardScaler` on each outer training fold |
| `protocol_structure__clinical_question_count` | turns meeting the frozen explicit-clinical-prompt rule | numeric count | reject missing/non-finite | outer-training fit only |
| `protocol_structure__unique_protocol_prompt_count` | distinct nonmissing frozen prompt IDs | numeric count | reject missing/non-finite | outer-training fit only |
| `protocol_structure__clinical_prompt_coverage_count` | distinct normalized explicit-clinical-prompt strings | numeric count | reject missing/non-finite | outer-training fit only |
| `protocol_structure__non_explicit_interviewer_turn_count` | interviewer turns not meeting the explicit-clinical-prompt rule | numeric count | reject missing/non-finite | outer-training fit only |

### Symptom domains (`D`, primary)

`anhedonia_interest`, `appetite_weight`, `concentration_psychomotor`, `depressed_mood`, `functioning_impairment`, `mental_health_history`, `protective_or_absent_symptom`, `self_worth_guilt`, `sleep_fatigue_energy`, and `suicide_self_harm` are the ten frozen domain counts.  They are numeric counts, missing/non-finite values are rejected, and each outer training fold supplies the `StandardScaler` parameters used for its corresponding test fold.

## 5. Model matrix

For each embedding representation, the main analysis runs these four paired levels:

| Level | Base model | Augmented model | Question |
|---|---|---|---|
| L1 | `P` | `P+I` | overall interviewer-language increment beyond participant language |
| L2 | `P+R` | `P+R+I` | residual increment after protocol structure |
| L3 | `P+D` | `P+D+I` | residual increment after symptom-domain information |
| L4 | `P+R+D` | `P+R+D+I` | final residual increment after both controls |

Here `P` and `I` are participant/interviewer source representations.  In the primary late-fusion analysis they are first-layer outer-training cross-fitted logits; in the early-concatenation sensitivity analysis they are the frozen normalized embedding vectors themselves.  `R` and `D` are always standardized inside the outer training fold.  All eight models use the same participant-level outer split.

## 6. Cross-fitting and tuning

- Main estimate: repeat 1 of the committed participant-level five-fold split; each participant appears in one outer test fold.
- Stability: the same implementation may run repeats 1--10 of the committed 10x5 membership.  These repeats summarize stability and are not treated as independent samples or t-test units.
- Inner selection: three stratified folds and the frozen C grid `[0.01, 0.1, 1.0, 10.0]`; highest mean inner ROC AUC wins, ties choose the smaller C.
- Late fusion first layer: for every outer fold and source, the outer-training rows are split into inner folds.  Each inner-validation logit is produced by a source model fitted on the complementary meta-training rows; that source model's C is selected only within a further three-fold split of its meta-training rows.  A source model selected on all outer-training rows then produces outer-test logits.
- Late fusion second layer: base and augmented logistic models are fitted on the source cross-fitted logits plus the relevant standardized `R`/`D` columns.  C is selected by three-fold validation within outer training rows only; the fitted outer-training model predicts the outer test rows.
- Early concatenation: participant/interviewer embeddings are concatenated with the relevant outer-training-standardized `R`/`D` columns and fitted with the same nested outer/inner boundaries and L2 logistic regression.  Embedding vectors are not re-standardized.
- The outer test fold is invisible to feature transformation, source-model tuning, second-layer tuning, calibration, and threshold selection.

## 7. Metrics and effect directions

Participant-level outer-fold probabilities are saved for every base/augmented pair.  Primary metric: ROC AUC.  Support metrics: PR-AUC, Brier, log loss, sensitivity at 0.50, and specificity at 0.50.  Fixed-threshold sensitivity/specificity are descriptive and are not the primary increment conclusion.

Positive deltas mean improvement from adding interviewer language:

- `delta_auc = AUC(augmented) - AUC(base)`;
- `delta_pr_auc = PR-AUC(augmented) - PR-AUC(base)`;
- `delta_brier = Brier(base) - Brier(augmented)`;
- `delta_log_loss = LogLoss(base) - LogLoss(augmented)`.

## 8. Confidence intervals, permutation tests, and FDR

- Every paired interval resamples participants jointly for the base and augmented predictions (5,000 resamples; percentile 95% CI).
- AUC deltas use 10,000 within-participant swap permutations for a two-sided raw p-value.
- Primary FDR family: the two late-fusion L4 AUC deltas (MPNet and BGE).
- Secondary family: all late-fusion L1--L4 AUC deltas (eight tests across the two embeddings), with `bh_q_all_late` retained in the output.  Non-primary early-concatenation AUC deltas are labelled exploratory and are not silently added to the core family.
- Support metrics are reported with paired point estimates and CIs; they are not merged into the AUC FDR family.

## 9. Attenuation

For each embedding and fusion method, attenuation is calculated as `delta_L1 - delta_L2` (protocol), `delta_L1 - delta_L3` (domain), and `delta_L1 - delta_L4` (protocol plus domain), for AUC and the three support metrics.  These are descriptive overlap/attenuation quantities, not mediation or causal proportions.  An attenuation proportion is not reported by default; if later requested it is interpretable only when the L1 AUC increment is positive and not near zero.

## 10. Locked output files

All files are written under `analysis_v2/03_embedding_robustness/embedding_incremental/`:

- `embedding_incremental_oof_predictions.csv`: participant ID, label, outer fold, embedding, fusion, level, paired base/augmented probabilities.
- `embedding_incremental_model_metrics.csv`: model-level AUC, PR-AUC, Brier, log loss, sensitivity, specificity.
- `embedding_incremental_deltas.csv`: paired deltas, AUC/support CIs, raw p, `bh_q`, `bh_q_all_late`, and FDR-family labels.
- `embedding_incremental_attenuation.csv`: attenuation type, metric, estimate, and participant-bootstrap CI.
- `embedding_incremental_repeat_stability.csv`: one row per embedding/fusion/repeat/level with base, augmented, and delta metrics; no repeat-level significance tests.
- `embedding_incremental_tuning.csv`: auditable inner-fold C choices.
- `run_manifest.json`: input paths/hashes, embedding cache hashes, split hash, seeds, grids, model matrix, and output inventory.

## 11. Completion and rerun requirements

The supplementary analysis is complete only when both embeddings, both fusion methods, L1--L4, strict cross-fitting, the locked input hashes, paired CIs, independent FDR, and stability output are present.  A second full run must produce identical result-file hashes.  Any missing restricted cache is a data-availability boundary, not permission to re-encode text or change the representation.

At the time this plan is committed, the implementation is code-only and the expensive embedding analysis has not been run.  Results must be generated only after code review of this plan and the runner.
