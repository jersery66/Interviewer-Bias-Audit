# conditions_long version guide

## Recommended file

Use this file for the current paper analysis:

`conditions_long__RECOMMENDED_article-aligned_analysis-ready_C1-C4_no-C5.csv`

Why:

- It is based on the article-aligned C1-C4 construction.
- It keeps the paper-aligned interviewer-cleaning rule.
- It adds `label = phq_binary` and `condition_status = ready`, so it is easier to use with the current analysis scripts.
- It contains only C1-C4. C5 `symptom_evidence_only` is still pending LLM evidence extraction.

## Version differences

| File | Location | Main use | Notes |
|---|---|---|---|
| `conditions_long__RECOMMENDED_article-aligned_analysis-ready_C1-C4_no-C5.csv` | `F:\数据库\DAIC-WOZ\processed_research` | Recommended analysis-ready file | Article-aligned cleaning + modeling-friendly `label` and `condition_status` fields. |
| `conditions_long.csv` | `F:\数据库\DAIC-WOZ\processed_research` | Original DAIC-WOZ main file | Current main dataset generated from original speaker-separated DAIC-WOZ transcripts. Uses initial cleaning. C1-C4 only. |
| `conditions_long__DAICWOZ-main_initial-cleaning_C1-C4_no-C5.csv` | `F:\数据库\DAIC-WOZ\processed_research` | Annotated copy of DAIC-WOZ main file | Same content as DAIC-WOZ `conditions_long.csv`, with a clearer filename. |
| `conditions_long.csv` | `F:\数据库\E-daic\processed_research` | Older E-daic C1-C4 file | Uses a stricter v1 interviewer-cleaning rule. Not recommended as the main paper version. |
| `conditions_long__EDAIC-v1_stricter-cleaning_C1-C4_no-C5.csv` | `F:\数据库\E-daic\processed_research` | Annotated copy of older E-daic file | Same content as E-daic `conditions_long.csv`, with a clearer filename. |
| `conditions_long_article_aligned.csv` | `F:\数据库\E-daic\processed_research` | Source article-aligned file | Article-aligned C1-C4 source file; useful for audit trail. |
| `conditions_long__EDAIC-article-aligned_source_C1-C4_no-C5.csv` | `F:\数据库\E-daic\processed_research` | Annotated copy of article-aligned source | Same content as `conditions_long_article_aligned.csv`, with a clearer filename. |

## Shared facts

- All three main condition files have 189 participants and 756 rows.
- All currently contain C1-C4 only: `full_dialogue`, `participant_only`, `interviewer_only`, and `interviewer_cleaned`.
- None contains C5 `symptom_evidence_only` yet.
- Participants 451, 458, and 480 have no interviewer turns, so they should be excluded or reported separately in interviewer-only analyses.

## Key cleaning difference

- DAIC-WOZ main initial cleaning: `interviewer_cleaned` mean length is about 499.57 words.
- E-daic v1 stricter cleaning: removes 1,653 interviewer turns; `interviewer_cleaned` mean length is about 425.26 words.
- Article-aligned cleaning: removes 1,384 interviewer turns; `interviewer_cleaned` mean length is about 438.49 words. This is the preferred version because the rule is explicitly aligned with the paper argument and avoids over-removing broader background or routing prompts.
