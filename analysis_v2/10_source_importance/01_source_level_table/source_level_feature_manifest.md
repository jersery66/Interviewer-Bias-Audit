# Source-Level Feature Table Manifest

## Data Source Verification

- Total participants: 142
- Positive (label=1): 43
- Negative (label=0): 99
- Expected: 142 participants (99 negative, 43 positive)

## Missing Values

| Column | Missing Count |
|--------|---------------|

## Source Files

- c1: `full_transcript_participant_oof.csv`
- c2: `participant_speech_participant_oof.csv`
- c3: `interviewer_speech_participant_oof.csv`
- c4: `non_explicit_interviewer_speech_participant_oof.csv`
- c5: `participant_symptom_evidence_participant_oof.csv`
- domain_presence: `domain_presence_participant_oof.csv`
- domain_count: `domain_count_participant_oof.csv`
- template_only: `participant_oof.csv`
- template_presence: `participant_oof.csv`

## Verification Checklist

✓ Sample size = 142
✓ Label distribution = 99/43
✓ No missing values
✓ Participant IDs unique
✓ All source files from official142 frozen cohort
✓ Official test NOT in primary analysis

## Column Description

### Source Probabilities

- c1_full_prob: Full transcript OOF probability
- c2_participant_prob: Participant speech OOF probability
- c3_interviewer_prob: Interviewer speech OOF probability
- c4_nonexplicit_interviewer_prob: Non-explicit interviewer speech OOF probability
- c5_evidence_prob: Symptom evidence OOF probability
- domain_presence_prob: Domain presence OOF probability
- domain_count_prob: Domain count OOF probability
- template_only_prob: Template only OOF probability
- template_presence_prob: Template presence OOF probability

### Raw Domain Features

- [domain]_presence: Binary presence of symptom domain
- [domain]_count: Count of symptom evidence for domain

### Predictions

- [source]_default_pred: Prediction at default 0.5 threshold
- [source]_nested_pred: Prediction at nested threshold
