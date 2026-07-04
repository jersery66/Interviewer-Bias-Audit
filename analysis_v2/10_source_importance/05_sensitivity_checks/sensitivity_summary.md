# Sensitivity Analysis Summary

| Check | Config A | AUC A | Config B | AUC B | Δ AUC |
|-------|----------|-------|----------|-------|-------|
| domain_count vs domain_presence | domain_count | 0.8132 | domain_presence | 0.8043 | +0.0089 |
| template_presence vs template_only | template_presence | 0.8132 | template_only | 0.8198 | -0.0066 |
| with vs without C5 quote | without C5 | 0.8132 | with C5 | 0.8132 | +0.0000 |

## Interpretation

- If Δ AUC is small, conclusions are robust to feature encoding choice
- If C5 adds little to M3, C5 quote signal largely overlaps with domain/protocol features
