# Table SX. Prespecified Case-Pattern Categories for Error Analysis

Case selection was performed using prespecified rules applied to all 142 participants. Cases are illustrative examples, not statistical evidence.

| Type | Selection Rule | N Eligible | Representative | Demonstrates | Interpretation |
|------|---------------|-----------|---------------|-------------|----------------|
| A: Threshold artifact | label=1, c2_default_pred=0, c2_nested_pred=1; sort by c2_prob descending | 20 | Participant 414 (c2_prob=0.494) | C2 participant speech contains signal, but default 0.5 threshold creates false negatives for borderline cases | Default threshold decision rule creates artifacts; nested threshold confirms C2 signal is present |
| B: Protocol signal dominance | label=1, c2_prob<0.5, template_only_prob≥0.5; sort by |template_only−c2| descending | 32 | Participant 356 (c2_prob=0.481, template_only=0.848) | Model assigns high risk under template condition but low under participant-only condition | Protocol structure carries label-related signal independent of participant symptom language |
| C: Evidence-domain convergence | abs(c5_prob − domain_presence_prob) < 0.15; sort by absolute difference ascending | 36 | Participant 343 (c5_prob=0.482, domain_presence=0.479, diff=0.003) | C5 symptom evidence and domain-presence features give nearly identical predictions | Symptom evidence discrimination is largely captured by domain-coverage state |
| D: Source conflict | c1_pred ≠ c2_pred (full transcript vs participant speech disagreement) | 51 | Participant 305 (c1_prob=0.519 pred=1, c2_prob=0.481 pred=0) | Full transcript incorrectly predicts positive while participant-only speech correctly predicts negative | C1 includes both participant and interviewer information; cannot be directly interpreted as patient symptom language |

## Selection Methodology

All selection rules were specified before inspecting individual participant data. The candidate pool (`candidate_cases.csv`) contains all 142 participants with eligibility flags. Cases were selected programmatically from the top-ranked candidates in each category.

## Manuscript Use

This table supports the case-based error analysis in Supplementary Materials. It demonstrates that case selection was systematic, not cherry-picked. No de-identified transcripts or quotes are included in the main text.
