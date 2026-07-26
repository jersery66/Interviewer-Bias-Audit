# Manuscript review and revision log

## 2026-07-14 formal identification closeout and full rewrite

The manuscript was rebuilt around four result groups: single-source predictability; conditional increments; interviewer-score recoverability and residual label increment; and formal pairing destruction/Fake-D identification sensitivity. The new formal output uses MPNet and BGE early concat, ten existing repeats, and 50 Random, 50 Matched and 50 Fake-D draws per repeat. The six primary repeat-level contrasts are recorded in `10_source_importance/08_identification_sensitivity/run_formal_10x/primary_contrasts.csv` with exact sign-flip p-values, bootstrap intervals and BH q-values.

The formal Fake-D contrast is negative for both embeddings, so the manuscript no longer states that adding D-P removes interviewer information because it is symptom content. The claim is limited to representation- and operationalization-dependent conditional attenuation. Module 15 is integrated as statistical score recoverability and retains its `training_match_quality_limited` status; Shapley values are not presented as signal percentages or causal effects.

The D-P blind quality packet contains a deterministic 30-person label-by-length sample, two blank independent reviewer forms and owner-only linkage. No human agreement or consensus metric is reported before reviewer input exists. The canonical manuscript, claim matrix and submission-audit summary were synchronized after the formal output hash check.

Before formal review dispatch, packet 3 superseded packet 2 as the active D-P validity protocol while preserving the original 30-person main sample. It adds a separately reported 13-person rare-domain enrichment sample and ID 385 as a descriptive sentinel, replaces the former 0/1/NS coding with domain-specific human states, and prevents actual C5 span review from opening until the stage-1 human consensus is hash-frozen. No validation result is implied by packet preparation.

## 2026-07-13 domain-source audit amendment

The post-review D-variable audit confirms that the frozen ten-domain count control is D-P, not D-All: all 1,137 reviewed spans match participant language, no span is interviewer-only, and the count table reconciles exactly. A search of the available C5 artifacts found no qualifying full-transcript D-All input; symptom-evidence-only files are not relabelled. The clean manuscript, audit summary and claim matrix now state this boundary. A gated D-P/D-All contract records the required full-transcript provenance and prevents a D-All comparison from being generated until that input is frozen.

The formal embedding-incremental outputs were not changed. A reader-facing report now derives a 16-row paired table with the locked FDR columns plus a separate all-16 global-BH sensitivity column, and generates a delta forest plot and L1–L4 trajectory plot. The table and figures are derived from the immutable `final/` outputs and do not add a new model or alter the primary family.

## 2026-07-12 reader-facing manuscript restructure

The technical audit draft is now preserved as `manuscript_submission_zh.md`, while the submission-oriented text is a separate clean manuscript at `main_text/manuscript_restructured_zh.md`. The new document is organized as a standard paper rather than an analysis checklist: one title and abstract, four natural Introduction paragraphs, four Method subsections, three reader-facing Results subsections, and one five-paragraph Discussion. Internal labels, audit timeline language, and exhaustive output details remain in the audit底稿 or supplement.

| Structural decision | Implementation |
|---|---|
| Single narrative | Source decomposition is the main line; calibration, structure, text budget and retained-span alignment support that line |
| Reader terminology | Internal model/source IDs are replaced by descriptive names such as 基础联合模型、文本长度结构模型 and 临床导向症状证据表示 |
| Claim downgrade | The symmetric half-min interval crosses zero and position results vary; the text-quantity explanation is therefore not ruled out |
| Causal boundary | Source-related prediction is not described as interviewer bias, leakage, clinical diagnosis or deployment evidence |
| Evidence placement | Main text keeps headline AUCs, conditional increments, the joint interval and the C5 alignment delta; draw-level, hash, CI and traceability details move to the supplement |

## 2026-07-12 post-audit sensitivity integration

The analysis rules were frozen in `post_submission_audit_sensitivity_plan.md` before results were generated. This revision does not claim preregistration and does not replace the original five-comparison family.

| Reviewer risk | Resolution |
|---|---|
| “Prespecified” could overstate the timeline | Replaced with analysis-locked terminology and stated that the lock prevents further split selection but is not formal preregistration |
| Brier scores lacked a no-skill comparator | Added outer-training-fold prevalence as the formal null, descriptive cohort prevalence, and BSS for raw/Platt/isotonic predictions |
| Equal-token control was asymmetric | Quantified truncation (132 participant versus 10 interviewer), retained the original question, and added 50 paired half-min draws plus early/middle/late windows |
| Text quantity was written as excluded | Downgraded the claim: under a symmetric budget the source contrast attenuated and its joint interval crossed zero; position results varied |
| Interaction structure was overgeneralized | Separated interaction-only uncertainty from protocol and all-structure results |
| C5 1137/1137 could be mistaken for extraction accuracy | Reported 1102/1143 direct matches, 35 alignment revisions, six exclusions, and 1137/1137 final post-review traceability |
| C5 “before/after” wording could overstate recoverable scope | Named the analysis retained-span source-alignment sensitivity and fixed inclusion, order, domain, and polarity |
| Novelty claim rested on a short selective table | Added a dated targeted-search log and limited the claim to representative retained studies; no “first” or field-wide rarity claim |
| Passing CI could be mistaken for raw-data reproduction | Separated code/test reproducibility, frozen-result integrity, and restricted full data-to-result reproduction |

## 2026-07-12 closeout amendment

The completion audit found that the symmetric half-min output had a window-inclusive joint interval but no raw p/q field. Before the closeout rerun, a post hoc amendment fixed the reporting contract without changing the half-min budget, draws, model, or joint interval: one participant-level source swap vector is held constant across all 50 draws, with 10,000 permutations and a separate one-comparison BH family. The manuscript reports both quantities and gives priority to the interval that includes window randomness. The same rerun also records newline-normalized public hashes and will be accompanied by an explicit two-run recomputation comparison.

## 2026-07-11 submission-audit revision

The previous strict-joint-model draft fixed leakage and participant-level inference but still treated repeated 10 x 5 estimates as the main narrative, omitted explicit structure and cross-source text-quantity controls, and did not integrate calibration, nested thresholds, or C5 review limitations into the paper. It has been superseded by `manuscript_submission_zh.md`.

| Priority | Reviewer risk | Resolution |
|---|---|---|
| P0 | Source performance could be miswritten as interviewer bias or leakage | Locked the paper to source-dependent predictive signals; causal bias and leakage claims are prohibited |
| P0 | Single-source AUC could be misread as independent contribution | Separated predictive sufficiency from M5-M3 and M4-M3 conditional increments |
| P0 | Repeated CV could invite pseudo-replication | Main analysis now uses frozen repeat 1 only; every participant has one cross-fitted prediction |
| P0 | Structural explanations were not directly controlled | Added length, interaction, protocol, and combined structural baselines |
| P0 | Source text quantity differed | Added 50 per-participant matched-whitespace-token draws with participant + random-draw uncertainty |
| P0 | Fixed 0.50 sensitivity was overinterpretable | Integrated Brier, calibration intercept/slope, calibration curves, and nested training-fold thresholds |
| P0 | C5 source integrity and blinding were underreported | Verified 1137/1137 final spans; disclosed 35 corrections, visible PHQ columns, and unavailable inter-rater reliability |
| P1 | Too many analyses competed for the main story | Locked three RQs and four main figures; domain LOO, repeated CV, mismatch errors, and E-DAIC moved to supplements |
| P1 | E-DAIC could be mistaken for external validation | Removed from title, abstract, core results, and main figures |
| P1 | Clinical wording exceeded the target | Standardized on PHQ-8 label prediction and prohibited diagnosis/deployment claims |

## Overall verdict

The previous draft was not publication-safe. Its headline AUC values came from the superseded two-level OOF-probability stacking analysis, several conclusions relied on unadjusted or fold-level uncertainty, figures were not embedded into the article, and the reference/availability/ethics sections were incomplete. The manuscript has been rebuilt around the strict raw-source joint analysis.

## Major issues and resolutions

| Priority | Previous issue | Risk | Resolution in current draft |
|---|---|---|---|
| P0 | Aggregated base-model OOF probabilities were reused as second-level CV features | Cross-level leakage and overly optimistic source attribution | Replaced with direct raw-source models; every vectorizer, scaler and classifier is fitted inside each outer training fold |
| P0 | Repeated folds were treated as independent evidence for uncertainty | Pseudo-replication | All CIs and paired tests now resample or swap at the 142-participant level |
| P0 | Old values (M0 0.6956, M3 0.8252, ΔAUC 0.1294) were presented as current | Numerically stale manuscript | Replaced throughout with strict results: M0 0.7052, M3 0.8137, ΔAUC 0.1085 |
| P0 | M3 improvement and protocol/domain conclusions were called significant without family-wise multiplicity handling | False-positive/overclaim risk | Seven model comparisons, three source-importance tests and 20 domain LOO tests are separately BH-FDR controlled |
| P1 | Bootstrap CI exclusion was treated as the sole significance rule | Conflicting inferential language | Primary decisions now use paired permutation `q`; CIs are explicitly descriptive |
| P1 | Template presence and single-domain rankings were written as established drivers | Unsupported mechanistic interpretation | Only domain-count group importance is starred; template and all 20 single-domain results are `ns` |
| P1 | Non-significant results were sometimes described as proof of no contribution | Equivalence fallacy | Rewritten as “not detected under the current model/power,” with redundancy and collinearity caveats |
| P1 | Figures and captions were separate from the text | Not a paper-formatted deliverable | Four figures are embedded at the point of first discussion with complete captions and cross-referenced tables |
| P1 | References, data/code availability and ethics were absent/incomplete | Submission incompleteness | Added five primary references and dedicated availability and ethics sections |

## Final interpretation boundary

The current manuscript supports a moderate participant-text baseline and significant conditional importance of domain counts in M3 (`q = 0.0039`, `**`). It does not support a multiplicity-controlled claim that any incremental model outperforms its comparator, that template presence is independently important, or that any individual clinical domain is confirmed.
