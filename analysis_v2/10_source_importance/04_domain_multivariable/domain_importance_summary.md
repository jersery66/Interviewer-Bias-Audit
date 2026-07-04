# Domain-Level Multivariable Importance

## Method

Five FDR-significant domains in multivariable logistic regression with repeated 5-fold CV and proper OOF averaging (10 repeats averaged).

## Presence Domain Results

| Removed Domain | Full AUC | Reduced AUC | Δ AUC | Rank |
|---------------|----------|-------------|-------|------|
| mental_health_history_presence | 0.8086 | 0.7491 | +0.0594 | 1 |
| functioning_impairment_presence | 0.8086 | 0.7494 | +0.0592 | 2 |
| depressed_mood_presence | 0.8086 | 0.7677 | +0.0409 | 3 |
| suicide_self_harm_presence | 0.8086 | 0.7891 | +0.0195 | 4 |
| appetite_weight_presence | 0.8086 | 0.7891 | +0.0195 | 5 |

## Count Domain Results

| Removed Domain | Full AUC | Reduced AUC | Δ AUC | Rank |
|---------------|----------|-------------|-------|------|
| functioning_impairment_count | 0.8090 | 0.7677 | +0.0413 | 1 |
| mental_health_history_count | 0.8090 | 0.7961 | +0.0129 | 2 |
| appetite_weight_count | 0.8090 | 0.7980 | +0.0110 | 3 |
| depressed_mood_count | 0.8090 | 0.7982 | +0.0108 | 4 |
| suicide_self_harm_count | 0.8090 | 0.7999 | +0.0092 | 5 |
