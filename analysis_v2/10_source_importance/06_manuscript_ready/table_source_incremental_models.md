# Table SX. Source-Level Incremental Model Performance

| Model | Predictors | ROC-AUC | Macro-F1@0.5 | Brier | Log-Loss | Δ AUC vs M0 |
|-------|-----------|---------|--------------|-------|----------|-------------|
| M0 | c2_prob | 0.6958 | 0.6389 | 0.2221 | 0.6439 | +0.0000 |
| M1 | c2_prob + domain_count_prob | 0.7681 | 0.7297 | 0.1860 | 0.5653 | +0.0724 |
| M2 | c2_prob + template_presence_prob | 0.7489 | 0.7397 | 0.1920 | 0.5802 | +0.0531 |
| M3 | c2_prob + domain_count_prob + template_presence_prob | 0.8132 | 0.7010 | 0.1719 | 0.5120 | +0.1175 |
| M4 | c2_prob + domain_count_prob + template_presence_prob + c5_prob | 0.8132 | 0.7010 | 0.1731 | 0.5142 | +0.1175 |
| M5 | c2_prob + domain_count_prob + template_presence_prob + c3_prob | 0.8283 | 0.7103 | 0.1648 | 0.4947 | +0.1325 |
