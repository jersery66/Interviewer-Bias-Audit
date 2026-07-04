# Table SX. Source-Level Incremental Model Performance

| Model | Predictors | ROC-AUC | Δ AUC vs M0 | Brier | Log-Loss |
|-------|-----------|---------|-------------|-------|----------|
| M0 | c2_prob | 0.6956 | +0.0000 | 0.2201 | 0.6340 |
| M1 | c2_prob + domain_count_prob | 0.7710 | +0.0754 | 0.1844 | 0.5566 |
| M2 | c2_prob + template_presence_prob | 0.7634 | +0.0679 | 0.1894 | 0.5723 |
| M3 | c2_prob + domain_count_prob + template_presence_prob | 0.8252 | +0.1297 | 0.1685 | 0.5051 |
| M4 | c2_prob + domain_count_prob + template_presence_prob + c5_prob | 0.8217 | +0.1261 | 0.1710 | 0.5098 |
| M5 | c2_prob + domain_count_prob + template_presence_prob + c3_prob | 0.8356 | +0.1400 | 0.1644 | 0.4955 |

## Bootstrap CI for Key Comparisons

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
