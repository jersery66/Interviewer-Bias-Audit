# Source Importance Analysis Results

## 1. Incremental Model Performance

The best-performing model (M5: Patient + Domain + Protocol + C3 (sensitivity)) achieved AUC=0.8356.

### Key Incremental Comparisons (5000x bootstrap CI)

- M1 vs M0 (AUC): +0.0741 [-0.0026, +0.1545] — does NOT significantly improve
- M1 vs M0 (Brier): -0.0354 [-0.0632, -0.0050] — significantly improves
- M1 vs M0 (LogLoss): -0.0768 [-0.1410, -0.0064] — significantly improves
- M3 vs M0 (AUC): +0.1294 [+0.0460, +0.2138] — significantly improves
- M3 vs M0 (Brier): -0.0517 [-0.0809, -0.0198] — significantly improves
- M3 vs M0 (LogLoss): -0.1291 [-0.2012, -0.0495] — significantly improves
- M4 vs M3 (AUC): -0.0037 [-0.0203, +0.0131] — does NOT significantly improve
- M4 vs M3 (Brier): +0.0025 [-0.0041, +0.0093] — does NOT significantly improve
- M4 vs M3 (LogLoss): +0.0049 [-0.0107, +0.0215] — does NOT significantly improve

## 2. Source Importance Ranking

The most important source was **domain_count_prob** (ΔAUC=0.1810 ± 0.0542), followed by **template_presence_prob** (ΔAUC=0.1272 ± 0.0341). **c2_prob** showed no positive incremental contribution (ΔAUC=-0.0054 ± 0.0140).

## 3. Domain-Level Importance

The most important domain was mental_health_history_presence (ΔAUC=+0.0594 when removed).

## 4. Key Conclusions

1. Domain coverage (domain_count_prob) is the dominant signal source in combined models
2. Patient speech (c2_prob) provides near-zero unique contribution when domain and protocol are accounted for
3. C5 quote probability adds no incremental value beyond domain_count (M4-M3 AUC Δ ≈ 0)
4. Clinical-context domains (functioning_impairment, mental_health_history) are key contributors
