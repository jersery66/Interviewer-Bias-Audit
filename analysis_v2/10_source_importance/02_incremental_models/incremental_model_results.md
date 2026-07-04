# Incremental Model Results (CORRECTED: 10-repeat CV averaging)

## Model Specifications

| Model | Description | Predictors | N Features |
|-------|------------|------------|-----------|
| M0 | Patient speech only | c2_prob | 1 |
| M1 | Patient + Domain | c2_prob + domain_count_prob | 2 |
| M2 | Patient + Protocol | c2_prob + template_presence_prob | 2 |
| M3 | Patient + Domain + Protocol | c2_prob + domain_count_prob + template_presence_prob | 3 |
| M4 | Patient + Domain + Protocol + C5 | c2_prob + domain_count_prob + template_presence_prob + c5_prob | 4 |
| M5 | Patient + Domain + Protocol + C3 (sensitivity) | c2_prob + domain_count_prob + template_presence_prob + c3_prob | 4 |

## Performance Metrics

| Model | ROC-AUC | PR-AUC | Macro-F1 | Sensitivity | Specificity | Brier | Log-Loss |
|-------|---------|--------|----------|-------------|-------------|-------|----------|
| M0 | 0.6956 | 0.5765 | 0.6420 | 0.6512 | 0.6768 | 0.2201 | 0.6340 |
| M1 | 0.7710 | 0.6590 | 0.7202 | 0.6744 | 0.7879 | 0.1844 | 0.5566 |
| M2 | 0.7634 | 0.5847 | 0.7267 | 0.6744 | 0.7980 | 0.1894 | 0.5723 |
| M3 | 0.8252 | 0.7012 | 0.6884 | 0.6744 | 0.7374 | 0.1685 | 0.5051 |
| M4 | 0.8217 | 0.6932 | 0.7067 | 0.7209 | 0.7374 | 0.1710 | 0.5098 |
| M5 | 0.8356 | 0.7309 | 0.7131 | 0.7209 | 0.7475 | 0.1644 | 0.4955 |

## Δ vs M0

| Model | Δ AUC vs M0 | Δ Log-Loss vs M0 |
|-------|------------|-----------------|
| M0 | +0.0000 | +0.0000 |
| M1 | +0.0754 | -0.0775 |
| M2 | +0.0679 | -0.0617 |
| M3 | +0.1297 | -0.1290 |
| M4 | +0.1261 | -0.1242 |
| M5 | +0.1400 | -0.1385 |

## Bootstrap CI (5000x paired participant bootstrap)

| Comparison | Metric | Mean Δ | 95% CI | Significant |
|-----------|--------|--------|--------|-------------|
| M1 vs M0 | AUC | +0.0741 | [-0.0026, +0.1545] | no |
| M1 vs M0 | Brier | -0.0354 | [-0.0632, -0.0050] | YES |
| M1 vs M0 | LogLoss | -0.0768 | [-0.1410, -0.0064] | YES |
| M2 vs M0 | AUC | +0.0681 | [-0.0218, +0.1599] | no |
| M2 vs M0 | Brier | -0.0308 | [-0.0567, -0.0037] | YES |
| M2 vs M0 | LogLoss | -0.0620 | [-0.1242, +0.0017] | no |
| M3 vs M0 | AUC | +0.1294 | [+0.0460, +0.2138] | YES |
| M3 vs M0 | Brier | -0.0517 | [-0.0809, -0.0198] | YES |
| M3 vs M0 | LogLoss | -0.1291 | [-0.2012, -0.0495] | YES |
| M3 vs M1 | AUC | +0.0537 | [-0.0066, +0.1207] | no |
| M3 vs M1 | Brier | -0.0157 | [-0.0378, +0.0074] | no |
| M3 vs M1 | LogLoss | -0.0509 | [-0.1050, +0.0051] | no |
| M3 vs M2 | AUC | +0.0620 | [+0.0038, +0.1239] | YES |
| M3 vs M2 | Brier | -0.0208 | [-0.0462, +0.0060] | no |
| M3 vs M2 | LogLoss | -0.0670 | [-0.1282, -0.0025] | YES |
| M4 vs M3 | AUC | -0.0037 | [-0.0203, +0.0131] | no |
| M4 vs M3 | Brier | +0.0025 | [-0.0041, +0.0093] | no |
| M4 vs M3 | LogLoss | +0.0049 | [-0.0107, +0.0215] | no |
| M5 vs M3 | AUC | +0.0104 | [-0.0119, +0.0328] | no |
| M5 vs M3 | Brier | -0.0043 | [-0.0148, +0.0065] | no |
| M5 vs M3 | LogLoss | -0.0097 | [-0.0371, +0.0173] | no |

## Key Conclusions

- **M4 vs M3: No improvement** (AUC Δ=-0.004, CI crosses zero) → C5 quote adds zero incremental value
- **M3 vs M0: Significant improvement** (AUC Δ=+0.129 [0.046, 0.214])
- **M1 vs M0: Not significant by AUC** (CI crosses zero: [-0.003, +0.155])
- Evidence of signal overlap between domain_count, template_presence, and C5
