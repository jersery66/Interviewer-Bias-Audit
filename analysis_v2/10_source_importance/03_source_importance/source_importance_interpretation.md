# Source Importance Interpretation

## Method

Permutation importance was computed under the M3 model (c2_participant_prob + domain_count_prob + template_presence_prob) using repeated 5-fold CV. For each fold, test-participant features were shuffled independently and the resulting performance degradation was recorded.

## Results

The most important source by AUC degradation was c2_prob (ΔAUC = -0.0062 ± 0.0192).

## Interpretation Guide

- Larger AUC decrease → source is more important for ranking
- Larger log-loss increase → source is more important for probability prediction
- If standard deviations are large relative to means, importance is unstable
- If multiple sources are close, signal overlap should be acknowledged

**Note:** These are descriptive importance estimates, not causal attributions.
