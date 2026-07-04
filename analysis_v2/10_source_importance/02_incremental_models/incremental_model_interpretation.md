# Incremental Model Interpretation

## Key Findings

### M0: Patient speech only

- AUC: 0.6958
- Log-loss: 0.6439
- Δ AUC vs M0: +0.0000 (stable)

### M1: Patient + Domain

- AUC: 0.7681
- Log-loss: 0.5653
- Δ AUC vs M0: +0.0724 (improvement)

### M2: Patient + Protocol

- AUC: 0.7489
- Log-loss: 0.5802
- Δ AUC vs M0: +0.0531 (improvement)

### M3: Patient + Domain + Protocol

- AUC: 0.8132
- Log-loss: 0.5120
- Δ AUC vs M0: +0.1175 (improvement)

### M4: Patient + Domain + Protocol + C5

- AUC: 0.8132
- Log-loss: 0.5142
- Δ AUC vs M0: +0.1175 (improvement)

### M5: Patient + Domain + Protocol + C3 (sensitivity)

- AUC: 0.8283
- Log-loss: 0.4947
- Δ AUC vs M0: +0.1325 (improvement)

## Interpretation

These results show the incremental contribution of different signal sources. Key interpretations are:

1. M0: Baseline - how well patient speech alone can predict
2. M1 vs M0: Whether domain count adds incremental value beyond patient speech
3. M2 vs M0: Whether template presence adds incremental value beyond patient speech
4. M3 vs M0/M1/M2: Whether combined model outperforms single-source models
5. M4 vs M3: Whether C5 quote adds value beyond domain count (overlap test)
6. M5 vs M3: Whether C3 interviewer adds value beyond template presence (sensitivity)

**Important:** These are descriptive source-importance audits, NOT causal attributions.
