# Bias Metrics: IBPG and DPD

Definitions:

- IBPG = interviewer_only - participant_only
- DPD = interviewer_only - interviewer_cleaned
- Metrics are calculated separately for F1, Macro-F1, and MCC.

## Summary

| evaluation | model_short                            | IBPG_f1 | DPD_f1 | IBPG_macro_f1 | DPD_macro_f1 | IBPG_mcc | DPD_mcc |
| ---------- | -------------------------------------- | ------- | ------ | ------------- | ------------ | -------- | ------- |
| train_dev  | TF-IDF + LR balanced                   | 0.364   | 0.016  | 0.137         | 0.017        | 0.155    | 0.031   |
| train_dev  | TF-IDF + Linear SVM balanced           | 0.316   | -0.037 | 0.125         | -0.035       | 0.128    | -0.081  |
| train_dev  | all-mpnet-base-v2 + LR balanced (CUDA) | -0.246  | 0.012  | -0.144        | 0.015        | -0.328   | 0.028   |
| 5fold_cv   | TF-IDF + LR balanced                   | 0.633   | 0.134  | 0.308         | 0.096        | 0.515    | 0.186   |
| 5fold_cv   | TF-IDF + Linear SVM balanced           | 0.640   | -0.014 | 0.328         | -0.011       | 0.534    | -0.028  |
| 5fold_cv   | all-mpnet-base-v2 + LR balanced (CUDA) | -0.043  | -0.039 | -0.023        | -0.016       | -0.058   | -0.049  |

## Interpretation Notes

- Positive IBPG means interviewer-only performs better than participant-only, suggesting interviewer speech carries predictive signal.
- Positive DPD means removing diagnostic interviewer prompts reduces performance, suggesting dependency on diagnostic prompts.
- Negative DPD means cleaned interviewer text performs similarly or better than raw interviewer-only text under that metric/model.