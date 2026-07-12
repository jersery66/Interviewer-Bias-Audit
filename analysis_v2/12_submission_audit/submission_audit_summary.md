# Submission audit: analysis-locked main analysis and post-audit sensitivity checks

## Timeline and claim boundary

- Main estimates use frozen repeat 1 of the existing participant-level five-fold split; every participant contributes one cross-fitted prediction.
- Repeat 1 was locked before the post-audit sensitivity run to prevent further split selection. This is not formal preregistration and is not described as prospectively prespecified.
- The existing 10 x 5 repeated cross-validation remains a split-stability supplement.
- Single-source performance measures predictive sufficiency, not independent contribution, causal bias, or leakage.

## Analysis-locked source results

- participant_speech: AUC 0.717 (95% CI 0.614-0.811), raw Brier 0.231; nested-threshold sensitivity/specificity 0.488/0.869.
- interviewer_speech: AUC 0.808 (95% CI 0.729-0.879), raw Brier 0.196; nested-threshold sensitivity/specificity 0.674/0.778.
- full_transcript: AUC 0.760 (95% CI 0.670-0.844), raw Brier 0.217; nested-threshold sensitivity/specificity 0.581/0.727.
- participant_symptom_evidence: AUC 0.735 (95% CI 0.640-0.819), raw Brier 0.224; nested-threshold sensitivity/specificity 0.535/0.818.

## Brier no-skill baselines

The formal no-skill comparator assigns each outer-test participant the positive prevalence from that outer-training fold. The fixed 43/142 cohort prevalence is descriptive only. Raw, Platt, and isotonic values use the same outer cross-fitted predictions; no calibrator is refit on pooled OOF predictions.

| Condition | Variant | Brier | Train-fold null | BSS | 95% BSS CI | Descriptive cohort-null BSS |
|---|---|---:|---:|---:|---:|---:|
| participant_speech | raw | 0.231 | 0.211 | -0.092 | -0.105 to -0.080 | -0.093 |
| participant_speech | platt | 0.179 | 0.211 | 0.152 | 0.047 to 0.249 | 0.151 |
| participant_speech | isotonic | 0.176 | 0.211 | 0.165 | 0.026 to 0.301 | 0.164 |
| interviewer_speech | raw | 0.196 | 0.211 | 0.073 | -0.001 to 0.145 | 0.072 |
| interviewer_speech | platt | 0.159 | 0.211 | 0.246 | 0.093 to 0.386 | 0.245 |
| interviewer_speech | isotonic | 0.172 | 0.211 | 0.187 | 0.010 to 0.348 | 0.186 |
| full_transcript | raw | 0.217 | 0.211 | -0.029 | -0.065 to 0.007 | -0.030 |
| full_transcript | platt | 0.175 | 0.211 | 0.171 | 0.049 to 0.287 | 0.170 |
| full_transcript | isotonic | 0.184 | 0.211 | 0.131 | -0.041 to 0.289 | 0.130 |
| participant_symptom_evidence | raw | 0.224 | 0.211 | -0.063 | -0.093 to -0.031 | -0.063 |
| participant_symptom_evidence | platt | 0.186 | 0.211 | 0.117 | -0.007 to 0.232 | 0.117 |
| participant_symptom_evidence | isotonic | 0.196 | 0.211 | 0.071 | -0.095 to 0.229 | 0.071 |

Isotonic fold-level Brier values are retained as a stability audit (range 0.108-0.275); no calibration method is declared best from one aggregate estimate.

## Structural and text-quantity controls

- S_length: AUC 0.569 (95% CI 0.468-0.669).
- S_interaction: AUC 0.573 (95% CI 0.469-0.675).
- S_protocol: AUC 0.675 (95% CI 0.573-0.773).
- S_all: AUC 0.729 (95% CI 0.634-0.821).
Interaction-only uncertainty includes 0.5; protocol-only and all-structure estimates are reported separately and are not collapsed into a claim that all structure families clearly predict above chance.
- Longer-source-to-shorter-source-length participant_matched: mean AUC 0.651 (participant + random-window 95% CI 0.514-0.780).
- Longer-source-to-shorter-source-length interviewer_matched: mean AUC 0.808 (participant + random-window 95% CI 0.726-0.882).
- Existing control asymmetry: participant/interviewer truncated n=132/10; mean retained fraction among non-empty texts=0.459/0.987.
- Symmetric half-min participant_matched: mean AUC 0.588 (participant + random-window 95% CI 0.440-0.737).
- Symmetric half-min interviewer_matched: mean AUC 0.706 (participant + random-window 95% CI 0.563-0.824).
- Symmetric half-min interviewer-minus-participant mean Delta AUC 0.1174 (joint 95% CI -0.0691-0.2953; raw p=0.0039, BH q=0.0039); median shared target=250, zero targets=2, targets below 10=2.

Deterministic early, middle, and late windows use the identical half-min budget for both sources:

| Position | Source | AUC | PR-AUC | Brier |
|---|---|---:|---:|---:|
| early | participant_matched | 0.435 | 0.295 | 0.240 |
| early | interviewer_matched | 0.698 | 0.471 | 0.222 |
| middle | participant_matched | 0.763 | 0.591 | 0.223 |
| middle | interviewer_matched | 0.738 | 0.585 | 0.205 |
| late | participant_matched | 0.726 | 0.572 | 0.228 |
| late | interviewer_matched | 0.818 | 0.649 | 0.187 |

Position-paired interviewer-minus-participant Delta AUCs were early 0.2638 (95% CI 0.1440-0.3794; raw p=0.0001, BH q=0.0003); middle -0.0244 (95% CI -0.1259-0.0756; raw p=0.6693, BH q=0.6693); late 0.0928 (95% CI -0.0246-0.2100; raw p=0.1196, BH q=0.1794).

The two token controls answer different questions. The original longer-to-shorter control was strongly asymmetric. Under a common half-min information budget, the source contrast attenuated and its window-inclusive joint interval included zero. The fixed-draw paired-swap p/q is a separate post-audit test and does not replace that interval; the magnitude and direction also varied by window position. The evidence therefore does not rule out text-budget or position effects and does not identify semantics or a causal mechanism.

## Original analysis-locked paired comparisons

Bootstrap confidence intervals and paired permutation p-values are different calculations. The five comparisons retain their original BH family.

| Comparison | Scope | Delta AUC | Participant-bootstrap 95% CI | Raw p | BH q |
|---|---|---:|---:|---:|---:|
| full_transcript vs participant_speech | source_sufficiency_not_independent_contribution | 0.0430 | -0.0395 to 0.1301 | 0.3611 | 0.4731 |
| interviewer_speech vs participant_speech | source_sufficiency_not_independent_contribution | 0.0909 | -0.0004 to 0.1898 | 0.0876 | 0.4380 |
| participant_symptom_evidence vs participant_speech | derived_representation_vs_raw_source | 0.0181 | -0.0790 to 0.1069 | 0.7273 | 0.7273 |
| M5 vs M3 | conditional_interviewer_increment | 0.0028 | -0.0040 to 0.0101 | 0.3785 | 0.4731 |
| M4 vs M3 | conditional_evidence_representation_increment | 0.0035 | -0.0038 to 0.0114 | 0.2499 | 0.4731 |

## C5 retained-span source-alignment sensitivity

Among 1143 candidates, 1102 were direct source matches (96.4%), 35 were source-alignment revisions, and 6 were excluded. All 1137/1137 retained spans are traceable after review. The final traceability rate is not extraction accuracy.
The retained-span sensitivity holds span inclusion, source order, domain, and polarity fixed and changes quote text only. The 35 revisions involve 24 participants; 24 participant representations change and total whitespace-token count changes by +103. It cannot reconstruct the six excluded candidates or a complete pre-review representation.

| Representation | AUC | PR-AUC | Brier |
|---|---:|---:|---:|
| original_retained | 0.739 | 0.557 | 0.224 |
| aligned_retained | 0.735 | 0.558 | 0.224 |

| Metric | Aligned minus original | Participant-bootstrap 95% CI | Raw p | BH q |
|---|---:|---:|---:|---:|
| roc_auc | -0.0042 | -0.0148 to 0.0049 | 0.3160 | 0.4740 |
| pr_auc | 0.0011 | -0.0147 to 0.0173 | 0.8573 | 0.8573 |
| brier | 0.0003 | -0.0002 to 0.0008 | 0.2714 | 0.4740 |

Absolute participant probability change: median 0.0012, 95th percentile 0.0048, maximum 0.0158; n>=0.01: 3, n>=0.05: 0.

The available review table contains PHQ-8 fields and lacks rater identities or independent double review. Outcome blinding and inter-rater reliability cannot be claimed.

## E-DAIC boundary

Module 14 remains a small descriptive supplement only. It does not establish generalizability or an independent error mechanism.
