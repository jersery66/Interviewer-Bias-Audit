# Reviewer requirement matrix

| Requirement | Status | Evidence or action |
|---|---|---|
| Replace bias framing with source-signal audit | Complete | New title, locked claim, three RQs, terminology contract |
| Separate source sufficiency from independent contribution | Complete | Single-source results and M5-M3/M4-M3 conditional increments reported separately |
| Add pure structural baseline | Complete | Length, interaction, protocol, and combined structural models in locked five-fold CV |
| Add equal-text-quantity control | Complete | 50 deterministic per-participant contiguous-window draws at matched whitespace-token counts |
| Separate template and non-template interviewer content | Complete with boundary | Template, non-template, and template-length-matched non-template conditions; non-template is not called purely adaptive |
| Report calibration | Complete | Brier, five-bin ECE, calibration intercept/slope, calibration curves |
| Select thresholds inside training folds | Complete | Existing nested thresholds reused; outer-test predictions are untouched during selection |
| Use one locked outer split for main inference | Complete | Frozen repeat 1, five folds, one cross-fitted prediction per participant |
| Treat repeated CV as stability only | Complete | 10 x 5 results moved to supplement and removed from main claim contract |
| Use participant-level paired uncertainty | Complete | 5000 participant bootstrap samples; folds/repeats are not inferential units |
| Limit primary comparison family | Complete | Five prespecified paired comparisons with BH-FDR |
| Audit C5 source integrity | Complete | 1137/1137 final spans match participant source; no quotes in public audit |
| Report C5 correction, invalidity, abstention, and agreement | Partially complete | Corrections and exclusions quantified; one participant has no retained evidence. Blinding and inter-rater agreement are unavailable and disclosed |
| Demote E-DAIC | Complete | Excluded from title, abstract, core results, and main figures |
| Add claim-evidence matrix | Complete | `claim_evidence_matrix.md` |
| Lower clinical language | Complete | PHQ-8 label terminology; no diagnostic, screening, biomarker, or deployment claim |
| Add prior-work audit table | Complete | Manuscript Table 1 compares source, protocol, length, calibration, and evidence-audit elements |
| Reduce main figures to four | Complete | Signal paths; locked source AUC; calibration/threshold; structural/text-quantity controls |
| Add more LLMs, embeddings, SHAP, or E-DAIC models | Intentionally not done | Explicitly outside the locked research questions |

## Remaining non-computable limitations

The current artifacts cannot establish blinded C5 review or inter-rater reliability because PHQ fields were present and rater identities were not recorded. These limitations are disclosed rather than reconstructed retrospectively. Independent external generalizability also remains untested.
