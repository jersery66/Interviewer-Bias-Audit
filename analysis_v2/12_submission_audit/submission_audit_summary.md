# Submission audit closeout summary（2026-07-14）

## Status

The post-audit formal outputs are frozen. The closeout now has three separate evidence layers:

1. the analysis-locked source and conditional results;
2. the module-15 interviewer-score explanation result, whose manifest status is `training_match_quality_limited`;
3. the new ten-repeat early-concat pairing/Fake-D sensitivity result.

The D-P manual quality question is not silently treated as solved. A 30-person blinded packet is prepared, but reviewer A/B sheets are blank and no agreement, kappa or consensus F1 is reported.

## Analysis-locked source results

The locked five-fold outer-test estimates are:

| Source | AUC | 95% CI | Raw Brier |
|---|---:|---:|---:|
| participant speech | 0.717 | 0.614–0.811 | 0.231 |
| interviewer speech | 0.808 | 0.729–0.879 | 0.196 |
| full transcript | 0.760 | 0.670–0.844 | 0.217 |
| participant symptom evidence | 0.735 | 0.640–0.819 | 0.224 |

The paired interviewer-minus-participant source difference crosses zero. Adding interviewer text to the P+Q+R+D-P strong baseline gives ΔAUC=0.0028 (95% CI −0.0040–0.0101, q=0.473). This is a conditional residual statement, not an absolute statement about interviewer information.

## Formal ten-repeat pairing/Fake-D result

Formal output directory:

`analysis_v2/10_source_importance/08_identification_sensitivity/run_formal_10x/`

The run uses the two frozen embeddings, early concat only, repeats 1–10, 50 Random draws, 50 Matched draws and 50 Fake-D draws per repeat. Each draw completes the full five-fold model. Repeat means, not 50 draw rows, are the inferential units. Six primary rows receive BH correction.

| Embedding | Contrast | Mean | 95% CI | Direction | p | q |
|---|---|---:|---:|---:|---:|---:|
| MPNet | Real−Matched | 0.0401 | 0.0287–0.0509 | 10/10 positive | 0.001953 | 0.001953 |
| MPNet | Real−Random | 0.0555 | 0.0430–0.0679 | 10/10 positive | 0.001953 | 0.001953 |
| MPNet | Real D-P ΔAUC−Fake-D ΔAUC | −0.0373 | −0.0514–−0.0237 | 10/10 negative | 0.001953 | 0.001953 |
| BGE | Real−Matched | 0.0592 | 0.0509–0.0687 | 10/10 positive | 0.001953 | 0.001953 |
| BGE | Real−Random | 0.0826 | 0.0722–0.0936 | 10/10 positive | 0.001953 | 0.001953 |
| BGE | Real D-P ΔAUC−Fake-D ΔAUC | −0.0729 | −0.0871–−0.0589 | 10/10 negative | 0.001953 | 0.001953 |

The positive Real−Matched and Real−Random contrasts indicate additional ranking difference in real pairing under this operationalization. The negative Real-D-P-versus-Fake-D contrast does not support a simple symptom-specific absorption explanation; it is compatible with feature engineering, label alignment, high-dimensional competition or representation/model interaction.

## Module 15 interviewer-score explanation

Formal `final/` exists and its output hashes match its manifest. The manifest status is `training_match_quality_limited`; matching-based results are therefore not promoted to primary causal or interaction evidence.

Repeat-1 score recoverability R² for the full P+Q+R+D-P block is 0.780 (TF-IDF), 0.297 (MPNet) and 0.424 (BGE). The R block has the largest descriptive block-Shapley estimate in all three representations: 0.491, 0.144 and 0.289 respectively. These are statistical recoverability summaries, not signal percentages.

Repeat-1 residual label increments after the full block are TF-IDF −0.0242 (CI −0.0517–−0.0005, p=0.0460), MPNet −0.0216 (CI −0.0524–0.0059, p=0.1400), and BGE +0.0139 (CI −0.0099–0.0390, p=0.3032). The mixed directions are reported as representation sensitivity; they are not a universal null claim.

## C5 and D-P audit boundary

C5 alignment is named `retained-span source-alignment sensitivity analysis`. It compares only the 1,137 retained evidence spans. Thirty-five quote alignments were revised and 24 participants' retained-span representations changed; the aligned-minus-original AUC difference was −0.0042 (95% CI −0.0148–0.0049). This cannot be described as a full pre-review versus post-review comparison.

The D-P blinded quality packet is at:

`analysis_v2/04_c5_controls/blind_quality_audit/packet_1/`

It contains 30 blinded cases, two blank reviewer forms, an owner-only sample key and a scoring reference. No reviewer result is present. If only one reviewer returns a form, the output must be labelled a single-reviewer content-validity sample; no kappa or consensus metric may be inferred.

## Reproducibility boundary

Public CI may verify unit tests, synthetic fixtures, parameter/schema checks, path independence, figure executability and hashes of committed derived results. It does not claim full data-to-result reproduction. The latter requires restricted DAIC-WOZ inputs and the private C5 source audit.

## Authoritative files

- Main manuscript: `analysis_v2/06_tables_figures/manuscript_submission_zh.md`
- Claim gate: `analysis_v2/12_submission_audit/claim_evidence_matrix.md`
- Formal pairing/Fake-D: `analysis_v2/10_source_importance/08_identification_sensitivity/run_formal_10x/`
- Module-15 final: `analysis_v2/15_interviewer_signal_explanation/final/`
- D-P blind packet: `analysis_v2/04_c5_controls/blind_quality_audit/packet_1/`
- Formal plan and timing contract: `analysis_v2/12_submission_audit/post_submission_audit_sensitivity_plan.md`
