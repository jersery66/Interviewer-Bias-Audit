# Four-figure submission contract

| Figure | Main question | Authoritative source | Excluded interpretations |
|---|---|---|---|
| 1. Source-signal paths | What sources and couplings exist in a semi-structured interview? | `figure_1_source_signal_paths.*` | The arrows are not identified causal effects |
| 2. Locked source performance | Which raw/derived sources are predictively sufficient? | `locked_source_metrics.csv` | AUC does not measure independent contribution |
| 3. Calibration and thresholds | Why does the fixed 0.50 operating point behave differently from ranking? | `locked_calibration_curve_data.csv`, `locked_threshold_curve_data.csv` | 0.50 is not a clinical or diagnostic threshold |
| 4. Structural and quantity controls | Can length, protocol structure, template content, or matched text quantity explain source performance? | `structure_control_figure_data.csv` | Remaining differences do not identify a semantic or causal mechanism |

Domain leave-one-out, permutation-importance families, repeated 10 x 5 stability, mismatch-error analyses, C5 row-level details, and E-DAIC are supplementary only.
