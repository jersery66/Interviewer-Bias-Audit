# C1-C4 Paper Tables

Generated at: 2026-06-15T16:33:23

## Table 1. Sample characteristics

| Characteristic                            | Overall            | PHQ-8 < 10         | PHQ-8 >= 10        | P value |
| ----------------------------------------- | ------------------ | ------------------ | ------------------ | ------- |
| Participants, n                           | 189                | 147                | 42                 |         |
| Split: train, n (%)                       | 143 (75.7%)        | 111 (75.5%)        | 32 (76.2%)         | 1.0000  |
| Split: dev, n (%)                         | 46 (24.3%)         | 36 (24.5%)         | 10 (23.8%)         |         |
| Age, mean +/- SD                          | 38.99 +/- 12.44    | 38.74 +/- 12.72    | 39.86 +/- 11.51    | 0.5274  |
| Gender: female, n (%)                     | 87 (46.0%)         | 64 (43.5%)         | 23 (54.8%)         | 0.2663  |
| Gender: male, n (%)                       | 102 (54.0%)        | 83 (56.5%)         | 19 (45.2%)         |         |
| PTSD positive, n (%)                      | 56 (29.6%)         | 20 (13.6%)         | 36 (85.7%)         | <0.0001 |
| PHQ-8 score, mean +/- SD                  | 6.75 +/- 5.92      | 4.26 +/- 3.68      | 15.45 +/- 3.58     | <0.0001 |
| Total turns, mean +/- SD                  | 250.79 +/- 77.47   | 249.33 +/- 75.97   | 255.93 +/- 83.27   | 0.9630  |
| Participant turns, mean +/- SD            | 171.43 +/- 75.07   | 171.10 +/- 73.08   | 172.60 +/- 82.59   | 0.7919  |
| Interviewer turns, mean +/- SD            | 79.36 +/- 17.63    | 78.22 +/- 18.51    | 83.33 +/- 13.57    | 0.0716  |
| Diagnostic interviewer turns, mean +/- SD | 5.30 +/- 1.81      | 5.14 +/- 1.77      | 5.88 +/- 1.84      | 0.0281  |
| Participant words, mean +/- SD            | 1471.19 +/- 811.39 | 1496.97 +/- 806.31 | 1380.95 +/- 832.47 | 0.2456  |
| Interviewer words, mean +/- SD            | 544.88 +/- 113.27  | 543.02 +/- 113.94  | 551.38 +/- 112.01  | 0.4927  |
| Cleaned interviewer words, mean +/- SD    | 499.57 +/- 105.85  | 498.99 +/- 106.64  | 501.60 +/- 104.29  | 0.3756  |

## Table 2. Input condition text length

| Condition              | Rows | Non-empty rows | Empty rows | Words, mean +/- SD | Words, median [IQR]        | Characters, mean +/- SD | Source turns, mean +/- SD | Diagnostic turns removed, mean +/- SD |
| ---------------------- | ---- | -------------- | ---------- | ------------------ | -------------------------- | ----------------------- | ------------------------- | ------------------------------------- |
| C1 Full dialogue       | 189  | 189            | 0          | 2266.71 +/- 866.61 | 2104.00 [1678.00, 2665.00] | 13563.01 +/- 4891.26    | 250.79 +/- 77.47          | 0.00 +/- 0.00                         |
| C2 Participant-only    | 189  | 189            | 0          | 1471.19 +/- 811.39 | 1293.00 [889.00, 1854.00]  | 7291.78 +/- 3956.52     | 171.43 +/- 75.07          | 0.00 +/- 0.00                         |
| C3 Interviewer-only    | 189  | 186            | 3          | 544.88 +/- 113.27  | 550.00 [491.00, 610.00]    | 3237.41 +/- 837.74      | 79.36 +/- 17.63           | 0.00 +/- 0.00                         |
| C4 Interviewer-cleaned | 189  | 186            | 3          | 438.49 +/- 91.20   | 444.00 [394.00, 485.00]    | 2626.28 +/- 666.23      | 72.04 +/- 16.54           | 7.32 +/- 2.50                         |

## Table 3. Model x input condition performance (5-fold CV)

| Model        | Input condition        | Accuracy        | F1              | Macro-F1        | AUC             | MCC             |
| ------------ | ---------------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| TF-IDF + LR  | C1 Full dialogue       | 0.815 +/- 0.019 | 0.483 +/- 0.103 | 0.685 +/- 0.055 | 0.803 +/- 0.065 | 0.394 +/- 0.106 |
| TF-IDF + LR  | C2 Participant-only    | 0.778 +/- 0.014 | 0.000 +/- 0.000 | 0.437 +/- 0.004 | 0.748 +/- 0.121 | 0.000 +/- 0.000 |
| TF-IDF + LR  | C3 Interviewer-only    | 0.796 +/- 0.092 | 0.633 +/- 0.125 | 0.746 +/- 0.097 | 0.851 +/- 0.067 | 0.515 +/- 0.176 |
| TF-IDF + LR  | C4 Interviewer-cleaned | 0.716 +/- 0.073 | 0.500 +/- 0.077 | 0.650 +/- 0.066 | 0.815 +/- 0.074 | 0.329 +/- 0.117 |
| TF-IDF + SVM | C1 Full dialogue       | 0.783 +/- 0.011 | 0.192 +/- 0.193 | 0.533 +/- 0.096 | 0.831 +/- 0.070 | 0.142 +/- 0.182 |
| TF-IDF + SVM | C2 Participant-only    | 0.778 +/- 0.014 | 0.000 +/- 0.000 | 0.437 +/- 0.004 | 0.755 +/- 0.119 | 0.000 +/- 0.000 |
| TF-IDF + SVM | C3 Interviewer-only    | 0.833 +/- 0.039 | 0.640 +/- 0.099 | 0.766 +/- 0.062 | 0.879 +/- 0.060 | 0.534 +/- 0.124 |
| TF-IDF + SVM | C4 Interviewer-cleaned | 0.844 +/- 0.074 | 0.654 +/- 0.170 | 0.777 +/- 0.108 | 0.859 +/- 0.087 | 0.562 +/- 0.221 |
| MPNet + LR   | C1 Full dialogue       | 0.683 +/- 0.049 | 0.408 +/- 0.102 | 0.595 +/- 0.066 | 0.624 +/- 0.120 | 0.208 +/- 0.136 |
| MPNet + LR   | C2 Participant-only    | 0.698 +/- 0.041 | 0.478 +/- 0.029 | 0.633 +/- 0.031 | 0.749 +/- 0.027 | 0.297 +/- 0.052 |
| MPNet + LR   | C3 Interviewer-only    | 0.689 +/- 0.080 | 0.436 +/- 0.099 | 0.609 +/- 0.078 | 0.699 +/- 0.014 | 0.239 +/- 0.143 |
| MPNet + LR   | C4 Interviewer-cleaned | 0.688 +/- 0.046 | 0.475 +/- 0.028 | 0.626 +/- 0.034 | 0.677 +/- 0.035 | 0.288 +/- 0.045 |

## Supplementary Table. Model x input condition performance (train/dev)

| Model        | Input condition        | Accuracy | F1    | Macro-F1 | AUC   | MCC   |
| ------------ | ---------------------- | -------- | ----- | -------- | ----- | ----- |
| TF-IDF + LR  | C1 Full dialogue       | 0.761    | 0.353 | 0.603    | 0.669 | 0.217 |
| TF-IDF + LR  | C2 Participant-only    | 0.783    | 0.000 | 0.439    | 0.672 | 0.000 |
| TF-IDF + LR  | C3 Interviewer-only    | 0.682    | 0.364 | 0.576    | 0.688 | 0.155 |
| TF-IDF + LR  | C4 Interviewer-cleaned | 0.659    | 0.348 | 0.559    | 0.685 | 0.124 |
| TF-IDF + SVM | C1 Full dialogue       | 0.783    | 0.286 | 0.579    | 0.706 | 0.211 |
| TF-IDF + SVM | C2 Participant-only    | 0.783    | 0.000 | 0.439    | 0.667 | 0.000 |
| TF-IDF + SVM | C3 Interviewer-only    | 0.705    | 0.316 | 0.564    | 0.738 | 0.128 |
| TF-IDF + SVM | C4 Interviewer-cleaned | 0.750    | 0.353 | 0.599    | 0.712 | 0.209 |
| MPNet + LR   | C1 Full dialogue       | 0.652    | 0.273 | 0.522    | 0.525 | 0.047 |
| MPNet + LR   | C2 Participant-only    | 0.717    | 0.519 | 0.659    | 0.706 | 0.361 |
| MPNet + LR   | C3 Interviewer-only    | 0.636    | 0.273 | 0.515    | 0.650 | 0.033 |
| MPNet + LR   | C4 Interviewer-cleaned | 0.614    | 0.261 | 0.500    | 0.659 | 0.005 |

## Table 4. IBPG / DPD bias metrics

| Evaluation | Model        | IBPG F1 | DPD F1 | IBPG Macro-F1 | DPD Macro-F1 | IBPG MCC | DPD MCC |
| ---------- | ------------ | ------- | ------ | ------------- | ------------ | -------- | ------- |
| 5-fold CV  | TF-IDF + LR  | 0.633   | 0.134  | 0.308         | 0.096        | 0.515    | 0.186   |
| 5-fold CV  | TF-IDF + SVM | 0.640   | -0.014 | 0.328         | -0.011       | 0.534    | -0.028  |
| 5-fold CV  | MPNet + LR   | -0.043  | -0.039 | -0.023        | -0.016       | -0.058   | -0.049  |
| Train/dev  | TF-IDF + LR  | 0.364   | 0.016  | 0.137         | 0.017        | 0.155    | 0.031   |
| Train/dev  | TF-IDF + SVM | 0.316   | -0.037 | 0.125         | -0.035       | 0.128    | -0.081  |
| Train/dev  | MPNet + LR   | -0.246  | 0.012  | -0.144        | 0.015        | -0.328   | 0.028   |

## Supplementary Table. Threshold sensitivity (5-fold CV, MCC-tuned)

| Model        | Input condition        | Default Macro-F1 | Tuned Macro-F1 | Delta Macro-F1 | Default MCC | Tuned MCC | Delta MCC | Mean tuned threshold |
| ------------ | ---------------------- | ---------------- | -------------- | -------------- | ----------- | --------- | --------- | -------------------- |
| TF-IDF + LR  | C1 Full dialogue       | 0.685            | 0.504          | -0.181         | 0.394       | 0.088     | -0.306    | 0.538                |
| TF-IDF + LR  | C2 Participant-only    | 0.437            | 0.437          | 0.000          | 0.000       | 0.000     | 0.000     | 0.509                |
| TF-IDF + LR  | C3 Interviewer-only    | 0.746            | 0.712          | -0.034         | 0.515       | 0.430     | -0.085    | 0.586                |
| TF-IDF + LR  | C4 Interviewer-cleaned | 0.650            | 0.702          | 0.053          | 0.329       | 0.418     | 0.090     | 0.556                |
| TF-IDF + SVM | C1 Full dialogue       | 0.533            | 0.454          | -0.080         | 0.142       | 0.011     | -0.131    | 0.139                |
| TF-IDF + SVM | C2 Participant-only    | 0.437            | 0.437          | 0.000          | 0.000       | 0.000     | 0.000     | 0.043                |
| TF-IDF + SVM | C3 Interviewer-only    | 0.766            | 0.724          | -0.042         | 0.534       | 0.475     | -0.059    | 0.190                |
| TF-IDF + SVM | C4 Interviewer-cleaned | 0.777            | 0.718          | -0.058         | 0.562       | 0.498     | -0.063    | 0.194                |

## Supplementary Table. McNemar paired tests (train/dev)

| Evaluation | Model        | Comparison                 | Paired n | Discordant n | A correct / B wrong | A wrong / B correct | Exact p |
| ---------- | ------------ | -------------------------- | -------- | ------------ | ------------------- | ------------------- | ------- |
| Train/dev  | TF-IDF + LR  | Full vs Participant        | 46       | 7            | 3                   | 4                   | 1.0000  |
| Train/dev  | TF-IDF + LR  | Participant vs Interviewer | 44       | 12           | 8                   | 4                   | 0.3877  |
| Train/dev  | TF-IDF + LR  | Interviewer vs Cleaned     | 44       | 5            | 3                   | 2                   | 1.0000  |
| Train/dev  | TF-IDF + SVM | Full vs Participant        | 46       | 4            | 2                   | 2                   | 1.0000  |
| Train/dev  | TF-IDF + SVM | Participant vs Interviewer | 44       | 9            | 6                   | 3                   | 0.5078  |
| Train/dev  | TF-IDF + SVM | Interviewer vs Cleaned     | 44       | 4            | 1                   | 3                   | 0.6250  |
| Train/dev  | MPNet + LR   | Full vs Participant        | 46       | 13           | 5                   | 8                   | 0.5811  |
| Train/dev  | MPNet + LR   | Participant vs Interviewer | 44       | 11           | 7                   | 4                   | 0.5488  |
| Train/dev  | MPNet + LR   | Interviewer vs Cleaned     | 44       | 1            | 1                   | 0                   | 1.0000  |
