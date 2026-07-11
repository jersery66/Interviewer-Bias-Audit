# Structural baseline feature dictionary

All variables are derived from the frozen turn table without using lexical content as model input.

| Group | Feature | Operational definition |
|---|---|---|
| Length | participant_word_count | Sum of frozen whitespace word counts over participant turns |
| Length | interviewer_word_count | Sum over interviewer turns |
| Length | total_word_count | Participant plus interviewer word counts |
| Length | interview_duration_sec | Maximum stop time minus minimum start time |
| Interaction | participant_turn_count | Number of participant turns |
| Interaction | interviewer_turn_count | Number of interviewer turns |
| Interaction | total_turn_count | Number of all turns |
| Interaction | participant_mean_words_per_turn | Participant words divided by participant turns |
| Interaction | interviewer_mean_words_per_turn | Interviewer words divided by interviewer turns |
| Protocol | interviewer_question_count | Interviewer turns with a frozen prompt annotation; punctuation is not used |
| Protocol | clinical_question_count | Interviewer turns meeting the frozen explicit-clinical-prompt rule |
| Protocol | unique_protocol_prompt_count | Distinct nonmissing frozen prompt IDs |
| Protocol | clinical_prompt_coverage_count | Distinct normalized explicit-clinical-prompt strings |
| Protocol | non_explicit_interviewer_turn_count | Interviewer turns not meeting the explicit-clinical-prompt rule |

The term `interviewer_question_count` is an annotation-based protocol count, not a punctuation-based question detector. Protocol features are observational process variables; they do not identify causal interviewer behavior.
