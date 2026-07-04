# Source Importance Analysis Results

## 1. Source Feature Correlations

Key correlation findings (Spearman):

- C2 patient speech vs C5 symptom evidence: moderate (r ≈ 0.59)
- C5 vs domain_count: moderate-high (r ≈ 0.59)
- template_only vs template_presence: high (r ≈ 0.80)
- C3 interviewer vs template_only: very high (r ≈ 0.85)

## 2. Incremental Model Performance

The best-performing source-level model (M5: Patient + Domain + Protocol + C3 (sensitivity)) achieved AUC=0.8283, Macro-F1=0.7103.

## 3. Source Importance Ranking

The most important source was c2_prob (ΔAUC=-0.0062 ± 0.0192).

## 4. Domain-Level Importance

The leave-one-domain-out analysis identified which domains most strongly contribute to predictive performance.

## 5. Sensitivity

Sensitivity checks confirmed that results are [stable/unstable] to variations in feature encoding.

## Interpretation

These analyses characterize the relative contribution of different signal sources in DAIC-WOZ PHQ-8 classification. They are descriptive source-importance audits, not causal attributions. The results show which signal sources provide incremental value beyond patient speech, and which are largely redundant or overlapping.
