# Reviewer requirement matrix

| Requirement | Status | Evidence or action |
|---|---|---|
| Replace bias framing with source-signal audit | Complete | New title, locked claim, three RQs, terminology contract |
| Separate source sufficiency from independent contribution | Complete | Single-source results and M5-M3/M4-M3 conditional increments reported separately |
| Add pure structural baseline | Complete | Length, interaction, protocol, and combined structural models in locked five-fold CV |
| Audit the original equal-text-quantity control | Complete with corrected boundary | Original control truncated participant/interviewer text for 132/10 participants; it is retained as a longer-to-shorter-source analysis, not a symmetric budget test |
| Add a symmetric text-budget control | Complete with dual uncertainty reporting | Both sources use `floor(0.5 x min(lengths))`; 50 paired draws plus deterministic early/middle/late windows; participant/window joint interval and fixed-draw paired-swap p/q are reported separately |
| Separate template and non-template interviewer content | Complete with boundary | Template, non-template, and template-length-matched non-template conditions; non-template is not called purely adaptive |
| Report calibration against a no-skill baseline | Complete | Outer-training-fold prevalence is the formal null; cohort prevalence is descriptive; BSS reported for raw, Platt, and isotonic on the same cross-fitted participants |
| Select thresholds inside training folds | Complete | Existing nested thresholds reused; outer-test predictions are untouched during selection |
| Use one analysis-locked outer split for main inference | Complete with timeline boundary | Frozen repeat 1, five folds, one cross-fitted prediction per participant; locked before the post-audit run but not claimed as preregistered |
| Treat repeated CV as stability only | Complete | 10 x 5 results moved to supplement and removed from main claim contract |
| Use participant-level paired uncertainty | Complete | 5000 participant bootstrap samples; folds/repeats are not inferential units |
| Limit primary comparison family | Complete | Five analysis-locked paired comparisons with BH-FDR; post-audit position and C5 tests are separate families |
| Audit C5 source integrity | Complete | 1102/1143 candidate spans direct-match; 35 alignment revisions and 6 exclusions; 1137/1137 is final post-review traceability, not extraction accuracy |
| Test retained-span source alignment | Complete with scope boundary | Same 1137 retained spans only; domain, polarity, inclusion, and order fixed; cannot recover the six excluded candidates or a complete pre-review representation |
| Report C5 correction, invalidity, abstention, and agreement | Partially complete | Corrections and exclusions quantified; one participant has no retained evidence. Blinding and inter-rater agreement are unavailable and disclosed |
| Demote E-DAIC | Complete | Excluded from title, abstract, core results, and main figures |
| Add claim-evidence matrix | Complete | `claim_evidence_matrix.md` |
| Lower clinical language | Complete | PHQ-8 label terminology; no diagnostic, screening, biomarker, or deployment claim |
| Add prior-work audit table and search record | Complete with non-systematic boundary | Manuscript Table 1 and `targeted_literature_search_log.md`; novelty claim is restricted to representative studies retained by the targeted search |
| Add public CI | Complete with data boundary | Unit tests and frozen-output integrity are public; full data-to-result reproduction still requires restricted DAIC-WOZ and private quote-bearing C5 sources |
| Reduce main figures to four | Complete | Signal paths; locked source AUC; calibration/threshold; structural/text-quantity controls |
| Add more LLMs, embeddings, SHAP, or E-DAIC models | Intentionally not done | Explicitly outside the locked research questions |

## Remaining non-computable limitations

The current artifacts cannot establish blinded C5 review or inter-rater reliability because PHQ fields were present and rater identities were not recorded. These limitations are disclosed rather than reconstructed retrospectively. Independent external generalizability also remains untested.
