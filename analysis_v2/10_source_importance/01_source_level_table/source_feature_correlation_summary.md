# Source Feature Correlation Analysis

## Key Findings

| Feature A | Feature B | Spearman r | Interpretation | Research Question |
|-----------|-----------|------------|----------------|-------------------|
| C2 Patient Speech | C5 Symptom Evidence | 0.592 | Moderate overlap: interpret incremental contribution cautiously | Is C5 just a compressed version of patient language? |
| C5 Symptom Evidence | Domain Count | 0.588 | Moderate overlap: interpret incremental contribution cautiously | Is C5 signal largely captured by domain count? |
| C5 Symptom Evidence | Domain Presence | 0.612 | High overlap: incremental contribution interpretation requires caution | Is C5 signal captured by domain presence? |
| Template Only | Template Presence | 0.799 | High overlap: incremental contribution interpretation requires caution | Are template text and template pattern highly overlapping? |
| C3 Interviewer | Template Only | 0.854 | High overlap: incremental contribution interpretation requires caution | Is interviewer text mainly template structure? |
| C2 Patient Speech | Template Presence | 0.442 | Moderate overlap: interpret incremental contribution cautiously | Is patient language coupled with protocol path? |
| Domain Count | Template Presence | 0.390 | Moderate overlap: interpret incremental contribution cautiously | Is symptom-domain status coupled with interview flow? |
| C2 Patient Speech | Domain Count | 0.487 | Moderate overlap: interpret incremental contribution cautiously | Are patient language and domain count overlapping? |
| C2 Patient Speech | C3 Interviewer | 0.573 | Moderate overlap: interpret incremental contribution cautiously | Are patient and interviewer speech correlated? |

## Interpretation Standards

| Correlation | Interpretation |
|-------------|----------------|
| abs(r) < 0.30 | Low overlap |
| 0.30 - 0.60 | Moderate overlap: interpret incremental contribution cautiously |
| > 0.60 | High overlap: incremental contribution interpretation requires caution |

## All Correlations (sorted by |Spearman r|)

| Feature A | Feature B | Spearman r | Pearson r | Interpretation |
|-----------|-----------|------------|-----------|----------------|
| C3 Interviewer | C4 Non-explicit | 0.949 | 0.935 | High overlap: incremental contribution interpretation requires caution |
| C4 Non-explicit | C1 Full Transcript | 0.905 | 0.910 | High overlap: incremental contribution interpretation requires caution |
| C3 Interviewer | C1 Full Transcript | 0.895 | 0.896 | High overlap: incremental contribution interpretation requires caution |
| Domain Count | Domain Presence | 0.859 | 0.846 | High overlap: incremental contribution interpretation requires caution |
| Template Only | C3 Interviewer | 0.854 | 0.895 | High overlap: incremental contribution interpretation requires caution |
| Template Presence | Template Only | 0.799 | 0.845 | High overlap: incremental contribution interpretation requires caution |
| Template Only | C1 Full Transcript | 0.797 | 0.753 | High overlap: incremental contribution interpretation requires caution |
| Template Only | C4 Non-explicit | 0.727 | 0.711 | High overlap: incremental contribution interpretation requires caution |
| C2 Patient Speech | C1 Full Transcript | 0.679 | 0.672 | High overlap: incremental contribution interpretation requires caution |
| Template Presence | C3 Interviewer | 0.615 | 0.725 | High overlap: incremental contribution interpretation requires caution |
| C5 Symptom Evidence | Domain Presence | 0.612 | 0.603 | High overlap: incremental contribution interpretation requires caution |
| C2 Patient Speech | C5 Symptom Evidence | 0.592 | 0.639 | Moderate overlap: interpret incremental contribution cautiously |
| C5 Symptom Evidence | Domain Count | 0.588 | 0.571 | Moderate overlap: interpret incremental contribution cautiously |
| C2 Patient Speech | C3 Interviewer | 0.573 | 0.564 | Moderate overlap: interpret incremental contribution cautiously |
| C2 Patient Speech | Template Only | 0.573 | 0.542 | Moderate overlap: interpret incremental contribution cautiously |
| Template Presence | C1 Full Transcript | 0.523 | 0.595 | Moderate overlap: interpret incremental contribution cautiously |
| C2 Patient Speech | C4 Non-explicit | 0.508 | 0.496 | Moderate overlap: interpret incremental contribution cautiously |
| Domain Presence | C3 Interviewer | 0.507 | 0.528 | Moderate overlap: interpret incremental contribution cautiously |
| Domain Presence | Template Only | 0.500 | 0.560 | Moderate overlap: interpret incremental contribution cautiously |
| Template Presence | C4 Non-explicit | 0.493 | 0.589 | Moderate overlap: interpret incremental contribution cautiously |
| C2 Patient Speech | Domain Presence | 0.493 | 0.509 | Moderate overlap: interpret incremental contribution cautiously |
| C5 Symptom Evidence | Template Presence | 0.488 | 0.498 | Moderate overlap: interpret incremental contribution cautiously |
| C2 Patient Speech | Domain Count | 0.487 | 0.504 | Moderate overlap: interpret incremental contribution cautiously |
| C5 Symptom Evidence | Template Only | 0.485 | 0.540 | Moderate overlap: interpret incremental contribution cautiously |
| C5 Symptom Evidence | C3 Interviewer | 0.463 | 0.474 | Moderate overlap: interpret incremental contribution cautiously |
| Domain Count | C3 Interviewer | 0.445 | 0.449 | Moderate overlap: interpret incremental contribution cautiously |
| Domain Presence | Template Presence | 0.444 | 0.476 | Moderate overlap: interpret incremental contribution cautiously |
| C2 Patient Speech | Template Presence | 0.442 | 0.487 | Moderate overlap: interpret incremental contribution cautiously |
| Domain Presence | C1 Full Transcript | 0.440 | 0.458 | Moderate overlap: interpret incremental contribution cautiously |
| Domain Count | Template Only | 0.440 | 0.476 | Moderate overlap: interpret incremental contribution cautiously |
| C5 Symptom Evidence | C1 Full Transcript | 0.418 | 0.426 | Moderate overlap: interpret incremental contribution cautiously |
| Domain Presence | C4 Non-explicit | 0.405 | 0.415 | Moderate overlap: interpret incremental contribution cautiously |
| Domain Count | Template Presence | 0.390 | 0.374 | Moderate overlap: interpret incremental contribution cautiously |
| Domain Count | C1 Full Transcript | 0.390 | 0.398 | Moderate overlap: interpret incremental contribution cautiously |
| C5 Symptom Evidence | C4 Non-explicit | 0.352 | 0.352 | Moderate overlap: interpret incremental contribution cautiously |
| Domain Count | C4 Non-explicit | 0.351 | 0.349 | Moderate overlap: interpret incremental contribution cautiously |
