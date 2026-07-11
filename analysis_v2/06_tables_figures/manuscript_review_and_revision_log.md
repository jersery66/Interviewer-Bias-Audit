# Manuscript review and revision log

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
