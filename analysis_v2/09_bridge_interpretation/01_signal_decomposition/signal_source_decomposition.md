# Table X. Decomposition of predictive signal sources in DAIC-WOZ PHQ-8 classification

This table decomposes the predictive signals in DAIC-WOZ PHQ-8 classification into six interpretable layers. It is a descriptive integration table, not a statistical test.

## Signal Source Decomposition Table

| Signal Layer | Operationalization | Corresponding Analysis | Representative Result | Supports | Does Not Support |
|-------------|-------------------|----------------------|---------------------|----------|-----------------|
| Participant language | Use of participant speech only (C2 participant_speech) | Primary TF-IDF analysis | AUC=0.705; Sensitivity@0.5=0.047; nested Sensitivity=0.512 | Participant speech contains risk-ranking signal | Cannot claim default 0.5 threshold enables direct screening |
| Symptom evidence text | Manually reviewed symptom quotes (C5 reviewed evidence) | C5 control analysis | AUC=0.754; Macro-F1@0.5=0.682 | Symptom evidence condensation yields usable classification performance | Cannot claim specific quote semantics significantly outperform controls |
| Symptom-domain coverage | Symptom domain presence/count vectors (domain_presence, domain_count) | C5 control analysis (domain features) | domain_presence AUC=0.790; domain_count AUC=0.794 | Symptom-domain coverage status carries strong discriminative information | Cannot claim model truly understands pathological mechanisms |
| Interviewer protocol structure | Interviewer text, template text, template presence (C3, C4, template controls) | C4 control analysis | template_only AUC=0.754; template_presence AUC=0.759 | Interview protocol and structure contain exploitable signal | Cannot claim interviewer natural language semantics alone cause prediction |
| Representation family | TF-IDF, MPNet, BGE (three representation methods) | Representation robustness analysis | MPNet: C2 > C3; BGE: C3 > C2 | Input-source effects depend on representation method | Cannot claim one input source is universally strongest across models |
| Threshold decision rule | Default threshold (0.5) vs nested threshold (fold-specific) | Threshold analysis | C2 Sensitivity: 0.047 (default) → 0.512 (nested); Macro-F1: 0.459 → 0.674 | Default threshold creates decision artifacts; nested threshold recovers signal | Cannot equate low sensitivity with absence of signal |

## Interpretation Sentences (for manuscript)

1. **Participant language**: "C2 participant_speech shows ROC-AUC 0.705 and nested Sensitivity 0.512, but default threshold yields near-zero Sensitivity (0.047), indicating threshold artifact rather than absence of signal."

2. **Symptom evidence text**: "C5 reviewed evidence achieves AUC 0.754, while domain_presence (AUC 0.790) and domain_count (AUC 0.794) perform similarly, suggesting domain_presence/count may capture much of the discriminative information in symptom-domain coverage status."

3. **Symptom-domain coverage**: "Domain_presence and domain_count achieve AUC 0.790-0.794, suggesting symptom-domain coverage status carries discriminative information, though these features are derived from the C5 extraction pipeline rather than being fully independent signal sources."

4. **Interviewer protocol structure**: "C4 template_only (AUC 0.754) and template_presence (AUC 0.759) show protocol structure alone carries predictive signal, independent of participant symptom language."

5. **Representation family**: "Under MPNet, participant_speech (C2) AUC=0.764 > interviewer_speech (C3) AUC=0.743; under BGE, C3 AUC=0.769 > C2 AUC=0.704."

6. **Threshold decision rule**: "C2 participant_speech shows near-zero Sensitivity at default 0.5 threshold, but nested threshold recovers to 0.512, demonstrating threshold-dependent classification behavior."

## Evidence Tables

- Primary results: `table_1_tfidf_primary.csv`
- Representation robustness: `table_2_representation_robustness.csv`
- C5 controls: `table_s4_c5_controls.csv`
- C4 controls: `table_s6_c4_controls.csv`

## Methods Note

This decomposition table was created after all primary and control analyses were completed. It does not introduce additional post hoc model selection. It is used as an interpretive scaffold to organize the results for readers.

## How to Cite This Table

**Table X. Decomposition of predictive signal sources in DAIC-WOZ PHQ-8 classification.** Six signal layers are identified: participant language, symptom evidence text, symptom-domain coverage, interviewer protocol structure, representation family, and threshold decision rule. Representative results and interpretive boundaries are provided for each layer.
