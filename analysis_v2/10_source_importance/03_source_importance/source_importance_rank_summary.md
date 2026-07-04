# Source Importance Ranking (Permutation Importance)

Based on M3 model (C2 + domain_count + template_presence).
Each feature was permuted 100x per fold, averaged across 10 repeats x 5 folds = 50 folds, with bootstrap CI over fold-level estimates.

| Source | Δ AUC (mean ± SD) | 95% CI | Δ Log-Loss | N Folds | Rank |
|--------|-------------------|--------|------------|---------|------|
| domain_count_prob | 0.1810 ± 0.0542 | [0.1663, 0.1959] | 0.1814 | 50 | 1 |
| template_presence_prob | 0.1272 ± 0.0341 | [0.1177, 0.1367] | 0.1270 | 50 | 2 |
| c2_prob | -0.0054 ± 0.0140 | [-0.0094, -0.0016] | -0.0050 | 50 | 3 |
