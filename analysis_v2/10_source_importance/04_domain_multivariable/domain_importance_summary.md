# Domain-Level Multivariable Importance

## Method

Five FDR-significant domains were used in multivariable logistic regression with repeated 5-fold CV. Leave-one-domain-out analysis quantifies each domain's contribution.

## Presence Domain Results

| Removed Domain | Full AUC | Reduced AUC | Δ AUC | Importance Rank |
|---------------|----------|-------------|-------|----------------|
| depressed_mood_presence | 0.8270 | 0.8122 | +0.0148 | 4 |
| appetite_weight_presence | 0.8270 | 0.7955 | +0.0315 | 3 |
| suicide_self_harm_presence | 0.8270 | 0.8123 | +0.0147 | 5 |
| functioning_impairment_presence | 0.8270 | 0.7689 | +0.0581 | 1 |
| mental_health_history_presence | 0.8270 | 0.7838 | +0.0432 | 2 |

## Count Domain Results

| Removed Domain | Full AUC | Reduced AUC | Δ AUC | Importance Rank |
|---------------|----------|-------------|-------|----------------|
| depressed_mood_count | 0.8161 | 0.8056 | +0.0105 | 4 |
| appetite_weight_count | 0.8161 | 0.8051 | +0.0109 | 3 |
| suicide_self_harm_count | 0.8161 | 0.8071 | +0.0089 | 5 |
| functioning_impairment_count | 0.8161 | 0.7873 | +0.0288 | 1 |
| mental_health_history_count | 0.8161 | 0.8014 | +0.0147 | 2 |

## Interpretation

- Larger Δ AUC → domain contributes more to model performance
- If Δ AUC is near zero or negative → domain is redundant
- Domains with high importance should be discussed in manuscript

**Caution:** These are descriptive importance estimates under the current feature set. Sample size (n=142) limits the reliability of individual domain importance estimates.
