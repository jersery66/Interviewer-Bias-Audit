# Figure contract

Core conclusion: DAIC-WOZ input-source conclusions depend on representation and decision threshold; no single interviewer-over-participant ranking claim is stable across all tested representations.

- Figure archetype: quantitative grid.
- Target/output: double-column manuscript figures; SVG/PDF with editable text; 600-dpi TIFF and 240-dpi PNG previews.
- Backend: Python/matplotlib only.
- Figure 1: representation robustness across C1-C5 for ROC-AUC, PR-AUC, and nested Macro-F1.
- Figure 2: TF-IDF threshold sensitivity, nested-threshold recovery, threshold stability, and train-only calibration.
- Figure S1: C5 reviewed evidence against 10 random controls and structured controls.
- Figure S2: C4 template, length, position, shuffle, and presence/count controls.
- Statistics: participant-level mean OOF probabilities from shared repeated 10 x 5 stratified CV; 10,000 participant-level paired swaps; BH-FDR within each predeclared family.
- Reviewer risk: point estimates are not confidence intervals; the cohort is official train+dev only; calibration and thresholds are trained without outer-test access; controls do not isolate natural-language template semantics.
