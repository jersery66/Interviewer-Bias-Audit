# Appendix A. C4 interviewer-cleaned cleaning rule and audit trail

Generated at: 2026-06-15T18:47:01

Rule version: `diagnostic_prompt_cleaning_article_aligned_v1_2026-06-12`

Cleaning aligned with Burdisso et al. 2024: remove direct depression/PHQ-style symptom probes and personal mental-health diagnosis/treatment-history prompts, while retaining broader background or routing prompts such as military-service questions.

## Cleaning summary

| Item                      | Value               |
| ------------------------- | ------------------- |
| participants              | 189.0               |
| participants_with_removed | 186.0               |
| removed_turns             | 1384.0              |
| retained_turns            | 13615.0             |
| removed_words             | 20107.0             |
| retained_words            | 82875.0             |
| mean_removed_turns        | 7.322751322751323   |
| median_removed_turns      | 7.0                 |
| mean_turn_removal_ratio   | 0.09378130838054329 |
| mean_word_removal_ratio   | 0.1942077846805265  |

## Rule catalogue

| rule_id                   | description                                                                             | patterns                                                                                                                                                                                                                                                                                                                                            |
| ------------------------- | --------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| mood_depression           | Direct depressed mood, sadness, hopelessness, or negative mood probes.                  | \bdepress(?:ed|ion)?\b; \bsad(?:ness)?\b; \bdown\b; \bhopeless(?:ness)?\b; \bunhappy\b; \bfeel(?:ing)?\s+(?:really\s+)?down\b                                                                                                                                                                                                                       |
| anhedonia_interest        | Direct interest, pleasure, enjoyment, or anhedonia-related symptom probes.              | \binterest(?:ed)?\b; \bpleasure\b; \benjoy(?:ed|ing)?\b                                                                                                                                                                                                                                                                                             |
| sleep_fatigue_energy      | Direct sleep, fatigue, tiredness, or low-energy symptom probes.                         | \bsleep(?:ing)?\b; \bgood\s+night'?s\s+sleep\b; \bsleep\s+well\b; \binsomnia\b; \btired(?:ness)?\b; \bfatigue\b; \benergy\b                                                                                                                                                                                                                         |
| appetite_weight           | Direct appetite, eating, or weight-change symptom probes.                               | \bappetite\b; \beat(?:ing)?\b; \bweight\b                                                                                                                                                                                                                                                                                                           |
| self_worth_guilt          | Direct self-worth, guilt, blame, failure, or worthlessness symptom probes.              | \bworthless(?:ness)?\b; \bself[-\s]?worth\b; \bblame\s+yourself\b; \bfailure\b; \bfeel\s+bad\s+about\s+yourself\b                                                                                                                                                                                                                                   |
| concentration_psychomotor | Direct concentration or psychomotor-change symptom probes.                              | \bconcentrat(?:e|ion|ing)\b; \brestless\b; \bslow(?:ed)?\s+down\b; \bmoving\s+or\s+speaking\s+so\s+slowly\b                                                                                                                                                                                                                                         |
| suicide_self_harm         | Suicidal ideation, self-harm, or disturbing-thought probes.                             | \bsuicid(?:e|al)\b; \bkill\s+yourself\b; \bhurt\s+yourself\b; \bself[-\s]?harm\b; \bbetter\s+off\s+dead\b; \bdisturbing\s+thoughts?\b; \bdidn'?t\s+wanna\s+go\s+anymore\b                                                                                                                                                                           |
| mental_health_history     | Diagnosis, therapy, counseling, help-seeking, treatment, or medication history prompts. | \bmental\s+health\b; \btherap(?:y|ist)\b; \bcounsel(?:or|lor|ing)\b; \btreatment\b; \bmedication\b; \bantidepressant\b; \bdiagnos(?:ed|is)\b; \banxiety\b; \bp[\W_]*t[\W_]*s[\W_]*d\b; \bseek\s+help\b; \bseeing\s+a\s+therapist\b; \bstill\s+go\s+to\s+therapy\b; \bstop(?:ped)?\s+(?:going\s+to\s+)?therapy\b; \bpsychologist\b; \bpsychiatrist\b |

## Manual review template

The review template samples removed and retained interviewer turns. It includes blank `human_review_decision` and `reviewer_notes` columns so a human reviewer can mark correct removals, false positives, false negatives, or unclear cases without altering the modeling files.

- Audit source: `F:\数据库\E-daic\processed_research\interviewer_cleaning_audit_article_aligned.csv`
- Manual review template: `processed_research\robustness_outputs\appendix_A_C4_manual_review_template.csv`

## Interpretation boundary

The C4 condition removes direct symptom and mental-health history probes from interviewer speech. It does not remove all possible interactional signals. Therefore, C4 should be interpreted as a diagnostic-prompt reduction condition, not as a complete removal of interviewer-side information.
