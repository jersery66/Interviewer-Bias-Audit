# Case Selection Log

This file documents the case selection process for the case-based error analysis.

## Case Type A: C2 Default Threshold Failure but Nested Threshold Recovery

### Selection Rules

- label == 1
- c2_default_pred == 0
- c2_nested_pred == 1
- Sort by c2_prob descending (closest to 0.5 but below)

### Interpretation

These positive samples have low raw probability under default 0.5 threshold, but are correctly classified under nested threshold. This demonstrates that C2 participant_speech contains signal, but the default threshold creates false negatives.

### Candidate Count

20 participants met the selection criteria.

### Selected Cases

- Participant 414: label=1, c2_prob=0.494, c2_default_pred=0, c2_nested_pred=1
  - Interpretation: Threshold artifact - C2 has signal but default threshold creates false negative
- (Additional cases can be included in supplementary materials)

## Case Type B: Interview Protocol Signal High, Participant Speech Signal Low

### Selection Rules

- label == 1
- c2_prob < 0.5
- template_only_prob >= 0.5 OR c3_prob >= 0.5
- Sort by abs(template_only_prob - c2_prob) descending

### Interpretation

These samples show that model gives high risk probability under interviewer/template conditions but low probability under participant-only condition. This supports that protocol structure carries label-related signal, but cannot be directly interpreted as patient symptom language.

### Candidate Count

31 participants met the selection criteria.

### Selected Cases

- Participant 356: label=1, c2_prob=0.481, template_only_prob=0.848, prob_diff=0.367
  - Interpretation: Protocol structure carries signal independent of participant symptom language
- (Additional cases can be included in supplementary materials)

## Case Type C: C5 and Domain Features Give Similar Predictions

### Selection Rules

- abs(c5_prob - domain_presence_prob) is small (< 0.15)
- Sort by abs(c5_prob - domain_presence_prob) ascending

### Interpretation

These cases show that specific symptom quotes (C5) and symptom-domain presence (domain_presence) give similar probabilities, suggesting that the discriminative information in symptom evidence is largely captured by symptom-domain coverage status.

### Candidate Count

36 participants met the selection criteria (abs(c5_prob - domain_presence_prob) < 0.15).

### Selected Cases

- Participant 343: label=0, c5_prob=0.482, domain_presence_prob=0.479, diff=0.003
  - Interpretation: Symptom evidence and domain coverage give nearly identical predictions, suggesting domain coverage drives much of C5 signal
- (Additional cases can be included in supplementary materials)

## Case Type D: Full Transcript vs Single Source Conflict

### Selection Rules

- C1 prediction != C2 prediction (disagreement between full transcript and participant speech)
- OR C1 prediction != C3 prediction (disagreement between full transcript and interviewer speech)

### Interpretation

Full transcript includes both participant and interviewer information, so C1 prediction cannot be directly attributed to participant symptom expression. These cases demonstrate the importance of separating information sources.

### Candidate Count

51 participants met the selection criteria (C1 and C2 probability difference > 0.3, or prediction disagreement).

### Selected Cases

- Participant 305: label=0, c1_pred=1 (false positive), c2_pred=0 (correct)
  - Interpretation: C1 incorrectly predicts positive, likely due to interviewer speech influence
- (Additional cases can be included in supplementary materials)

## Selection Bias Mitigation

To avoid "cherry-picking" favorable cases, selection was done using prespecified rules implemented in a Python script. The script and selection criteria are documented in:

- `run_case_analysis.py` (selection script)
- This log file (selection criteria and counts)

The selected cases are illustrative examples, not statistical evidence. They are selected to help readers understand the mechanisms behind the quantitative results.
