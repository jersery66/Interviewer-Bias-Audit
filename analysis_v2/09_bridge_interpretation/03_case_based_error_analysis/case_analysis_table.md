# Case Analysis Table

This table presents selected participant-level cases to illustrate key findings.

## Selected Cases

| case_id | participant_id | label | C2 prob | C3/template prob | C5 prob | domain prob | pattern | interpretation |
|---------|----------------|-------|---------|-------------------|---------|-------------|---------|----------------|
| A | 414 | 1 | 0.494 | 0.742 | 0.489 | 0.526 | C2 default false negative, nested positive | Threshold artifact: C2 has signal but default 0.5 threshold creates false negative |
| B | 356 | 1 | 0.481 | 0.848 (template) | 0.523 | 0.537 | Protocol high, participant low | Protocol structure carries signal independent of participant symptom language |
| C | 343 | 0 | 0.476 | 0.567 | 0.482 | 0.479 | C5 close to domain | Symptom evidence and domain coverage give similar predictions |
| D | 305 | 0 | 0.481 (pred=0) | 0.642 | 0.501 | 0.463 | Full transcript/source conflict | C1 (full transcript) includes both participant and interviewer, cannot be directly interpreted as participant language |

## Case Details

### Case A (Participant 414)
- Label: PHQ-8 positive (1)
- C2 participant_speech probability: 0.494 (just below 0.5)
- C2 default threshold prediction: 0 (false negative)
- C2 nested threshold prediction: 1 (corrected)
- Interpretation: Demonstrates that C2 contains signal, but default 0.5 threshold creates false negatives for borderline cases

### Case B (Participant 356)
- Label: PHQ-8 positive (1)
- C2 participant_speech probability: 0.481 (below 0.5)
- Template_only probability: 0.848 (well above 0.5)
- Interpretation: Model gives high risk probability under template condition but low under participant-only condition, suggesting protocol structure carries signal

### Case C (Participant 343)
- Label: PHQ-8 negative (0)
- C5 symptom evidence probability: 0.482
- Domain_presence probability: 0.479
- Difference: 0.003 (very small)
- Interpretation: Specific symptom quotes and symptom-domain presence give nearly identical predictions, suggesting domain coverage drives much of C5 signal

### Case D (Participant 305)
- Label: PHQ-8 negative (0)
- C1 full_transcript prediction: 1 (false positive)
- C2 participant_speech prediction: 0 (correct)
- Interpretation: C1 incorrectly predicts positive, likely due to interviewer speech influence. C1 cannot be directly interpreted as participant language contribution.

## Note on Case Selection

Cases were selected using prespecified rules:
- Case A: label=1, C2 default prediction=0, C2 nested prediction=1, sorted by C2 probability descending
- Case B: label=1, C2 prob<0.5, template_only prob>=0.5, sorted by probability difference descending
- Case C: abs(C5 prob - domain_presence prob) < 0.15, sorted by absolute difference ascending
- Case D: C1 prediction != C2 prediction, demonstrating source conflict

These cases are illustrative examples, not statistical evidence. They are selected to help readers understand the mechanisms behind the quantitative results.
