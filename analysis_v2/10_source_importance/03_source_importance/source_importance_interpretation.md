# Source Importance Interpretation

## Method

Permutation importance was computed under the M3 model (c2_prob + domain_count_prob + template_presence_prob) using repeated 5-fold CV. For each fold, test-participant features were shuffled 100x with fixed RNG seed=20260704. Results are averaged across 50 folds (10 repeats × 5 folds).

## Results

The most important source by AUC degradation was **domain_count_prob** (ΔAUC=0.1810 ± 0.0542, 95% CI [0.1663, 0.1959]).

- domain_count_prob: ΔAUC = 0.1810 ± 0.0542 (significant)
- template_presence_prob: ΔAUC = 0.1272 ± 0.0341 (significant)
- c2_prob: ΔAUC = -0.0054 ± 0.0140 (significant)

## Interpretation Guide

- Larger AUC decrease → source is more important for ranking
- If CI crosses zero → importance is not statistically significant
- c2_prob showing negative or near-zero ΔAUC means it does NOT provide unique predictive value in M3
