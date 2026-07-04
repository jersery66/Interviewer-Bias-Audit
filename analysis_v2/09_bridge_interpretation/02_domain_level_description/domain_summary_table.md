# Domain-Level Group Comparison Results

## Presence Analysis

| domain | domain_group | positive_presence_rate | negative_presence_rate | risk_difference | odds_ratio | fisher_p | fdr_bh_q |
|--------|-------------|------------------------|------------------------|-----------------|------------|----------|----------|
| anhedonia_interest | PHQ-core | 0.163 | 0.152 | 0.011 | 1.120 | 1.0000 | 1.0000 |
| depressed_mood | PHQ-core | 0.791 | 0.535 | 0.255 | 3.156 | 0.0048 | 0.0120 |
| sleep_fatigue_energy | PHQ-core | 0.953 | 0.838 | 0.115 | 3.280 | 0.0964 | 0.1606 |
| appetite_weight | PHQ-core | 0.163 | 0.030 | 0.132 | 5.665 | 0.0088 | 0.0175 |
| self_worth_guilt | PHQ-core | 0.419 | 0.404 | 0.015 | 1.066 | 1.0000 | 1.0000 |
| concentration_psychomotor | PHQ-core | 0.256 | 0.182 | 0.074 | 1.559 | 0.3665 | 0.5236 |
| suicide_self_harm | PHQ-core | 0.209 | 0.020 | 0.189 | 10.739 | 0.0004 | 0.0012 |
| functioning_impairment | clinical-context | 0.767 | 0.283 | 0.485 | 8.004 | 0.0000 | 0.0000 |
| mental_health_history | clinical-context | 0.860 | 0.444 | 0.416 | 7.195 | 0.0000 | 0.0000 |
| protective_or_absent_symptom | protective/denial | 0.953 | 0.929 | 0.024 | 1.346 | 0.7231 | 0.9039 |

## Count Analysis

| domain | domain_group | positive_mean_count | negative_mean_count | mean_difference | mannwhitney_p | fdr_bh_q |
|--------|-------------|---------------------|---------------------|-----------------|---------------|----------|
| anhedonia_interest | PHQ-core | 0.21 | 0.18 | 0.03 | 0.8432 | 0.8470 |
| depressed_mood | PHQ-core | 1.56 | 0.74 | 0.82 | 0.0004 | 0.0009 |
| sleep_fatigue_energy | PHQ-core | 1.88 | 1.37 | 0.51 | 0.0354 | 0.0590 |
| appetite_weight | PHQ-core | 0.23 | 0.04 | 0.19 | 0.0048 | 0.0096 |
| self_worth_guilt | PHQ-core | 0.53 | 0.52 | 0.02 | 0.8470 | 0.8470 |
| concentration_psychomotor | PHQ-core | 0.35 | 0.19 | 0.16 | 0.2565 | 0.3665 |
| suicide_self_harm | PHQ-core | 0.26 | 0.02 | 0.24 | 0.0001 | 0.0004 |
| functioning_impairment | clinical-context | 1.60 | 0.48 | 1.12 | 0.0000 | 0.0000 |
| mental_health_history | clinical-context | 2.16 | 0.85 | 1.31 | 0.0000 | 0.0000 |
| protective_or_absent_symptom | protective/denial | 2.26 | 2.29 | -0.04 | 0.7720 | 0.8470 |

## Summary Table

                      domain      domain_group  positive_presence_rate  negative_presence_rate  risk_difference  odds_ratio     fisher_p  fdr_bh_q  positive_mean_count  negative_mean_count  mean_difference  mannwhitney_p  count_fdr_bh_q                         interpretation
          anhedonia_interest          PHQ-core                0.162791                0.151515         0.011276    1.120194 1.000000e+00  1.000000             0.209302             0.181818         0.027484   8.431527e-01    8.469721e-01              Not significant after FDR
              depressed_mood          PHQ-core                0.790698                0.535354         0.255344    3.156419 4.787483e-03  0.011969             1.558140             0.737374         0.820766   3.697239e-04    9.243097e-04 Higher in positive group (significant)
        sleep_fatigue_energy          PHQ-core                0.953488                0.838384         0.115105    3.280240 9.636285e-02  0.160605             1.883721             1.373737         0.509984   3.538766e-02    5.897944e-02              Not significant after FDR
             appetite_weight          PHQ-core                0.162791                0.030303         0.132488    5.665362 8.755303e-03  0.017511             0.232558             0.040404         0.192154   4.823785e-03    9.647570e-03 Higher in positive group (significant)
            self_worth_guilt          PHQ-core                0.418605                0.404040         0.014564    1.065844 1.000000e+00  1.000000             0.534884             0.515152         0.019732   8.469721e-01    8.469721e-01              Not significant after FDR
   concentration_psychomotor          PHQ-core                0.255814                0.181818         0.073996    1.558836 3.665081e-01  0.523583             0.348837             0.191919         0.156918   2.565374e-01    3.664819e-01              Not significant after FDR
           suicide_self_harm          PHQ-core                0.209302                0.020202         0.189100   10.739130 3.677404e-04  0.001226             0.255814             0.020202         0.235612   1.079786e-04    3.599287e-04 Higher in positive group (significant)
      functioning_impairment  clinical-context                0.767442                0.282828         0.484614    8.004177 1.086042e-07  0.000001             1.604651             0.484848         1.119803   1.731885e-08    1.731885e-07 Higher in positive group (significant)
       mental_health_history  clinical-context                0.860465                0.444444         0.416021    7.195333 2.730594e-06  0.000014             2.162791             0.848485         1.314306   3.743670e-07    1.871835e-06 Higher in positive group (significant)
protective_or_absent_symptom protective/denial                0.953488                0.929293         0.024195    1.345946 7.231088e-01  0.903886             2.255814             2.292929        -0.037115   7.719754e-01    8.469721e-01              Not significant after FDR

## Interpretation

### English

The significant domains (FDR q < 0.05) in the presence analysis include both PHQ-core symptoms (depressed_mood, appetite_weight, suicide_self_harm) and clinical-context domains (functioning_impairment, mental_health_history). This suggests that the discriminative performance of domain_presence/count is not driven solely by PHQ-core symptoms, but by a combination of extracted symptom domains, functional impairment domains, and mental health history domains.

**Key finding**: The high AUC of domain_presence/count (0.790-0.794) is contributed by multiple extracted domains together, not by PHQ-core symptoms alone. Specifically, functioning_impairment and mental_health_history show large effect sizes (OR > 7), suggesting these clinical-context domains strongly contribute to the discriminative performance.

**Interpretation for manuscript**: "The discriminative performance of domain_presence/count features is contributed by a combination of extracted symptom domains (e.g., depressed_mood, suicide_self_harm), functional impairment domains (functioning_impairment), and mental health history domains (mental_health_history), rather than by PHQ-core symptoms alone."

**Caution**: These features encode extracted symptom-domain states, not clinical diagnosis or causal symptom mechanisms. The domain features are derived from the C5 extraction pipeline, so they are not fully independent signal sources.

### 中文

Presence分析中显著的domain（FDR q < 0.05）既包括PHQ核心症状（depressed_mood、appetite_weight、suicide_self_harm），也包括临床背景domain（functioning_impairment、mental_health_history）。这说明domain_presence/count的判别性能并非仅由PHQ核心症状驱动，而是由多个被抽取的症状域、功能损害域和心理健康史域共同构成。

**关键发现**：domain_presence/count的高AUC（0.790-0.794）由多个被抽取域共同贡献，而非仅由PHQ核心症状贡献。具体来说，functioning_impairment和mental_health_history显示大效应量（OR > 7），提示这些临床背景domain强烈贡献了判别性能。

**论文解释句**："domain_presence/count特征的判别性能由多个被抽取域共同构成，包括症状域（如depressed_mood、suicide_self_harm）、功能损害域（functioning_impairment）和心理健康史域（mental_health_history），而非仅由PHQ核心症状贡献。"

**注意事项**：这些变量编码的是被抽取到的症状域状态，而不是临床诊断或症状病理机制。Domain特征是从C5抽取流程派生的，因此它们不是完全独立的信号来源。
