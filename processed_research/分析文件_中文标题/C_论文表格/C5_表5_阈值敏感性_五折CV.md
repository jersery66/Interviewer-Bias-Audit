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