# Incremental Model Results

## Model Specifications

| Model | Description | Predictors |
|-------|------------|------------|
| M0 | Patient speech only | c2_prob |
| M1 | Patient + Domain | c2_prob + domain_count_prob |
| M2 | Patient + Protocol | c2_prob + template_presence_prob |
| M3 | Patient + Domain + Protocol | c2_prob + domain_count_prob + template_presence_prob |
| M4 | Patient + Domain + Protocol + C5 | c2_prob + domain_count_prob + template_presence_prob + c5_prob |
| M5 | Patient + Domain + Protocol + C3 (sensitivity) | c2_prob + domain_count_prob + template_presence_prob + c3_prob |

## Performance Metrics

| Model | ROC-AUC | PR-AUC | Macro-F1@0.5 | Sensitivity@0.5 | Specificity@0.5 | Brier | Log-Loss |
|-------|---------|--------|--------------|----------------|-----------------|-------|----------|
| M0 | 0.6958 | 0.5635 | 0.6389 | 0.6279 | 0.6869 | 0.2221 | 0.6439 |
| M1 | 0.7681 | 0.6600 | 0.7297 | 0.6977 | 0.7879 | 0.1860 | 0.5653 |
| M2 | 0.7489 | 0.5850 | 0.7397 | 0.6744 | 0.8182 | 0.1920 | 0.5802 |
| M3 | 0.8132 | 0.7009 | 0.7010 | 0.6744 | 0.7576 | 0.1719 | 0.5120 |
| M4 | 0.8132 | 0.7006 | 0.7010 | 0.6744 | 0.7576 | 0.1731 | 0.5142 |
| M5 | 0.8283 | 0.7306 | 0.7103 | 0.6977 | 0.7576 | 0.1648 | 0.4947 |

## Incremental Changes

| Model | Δ AUC | Δ PR-AUC | Δ Macro-F1 | Δ Brier | Δ AUC vs M0 |
|-------|--------|-----------|------------|---------|-------------|
| M0 | - | - | - | - | +0.0000 |
| M1 | +0.0724 | +0.0964 | +0.0908 | -0.0361 | +0.0724 |
| M2 | -0.0193 | -0.0749 | +0.0101 | +0.0060 | +0.0531 |
| M3 | +0.0644 | +0.1158 | -0.0387 | -0.0201 | +0.1175 |
| M4 | +0.0000 | -0.0003 | +0.0000 | +0.0011 | +0.1175 |
| M5 | +0.0150 | +0.0301 | +0.0093 | -0.0082 | +0.1325 |
