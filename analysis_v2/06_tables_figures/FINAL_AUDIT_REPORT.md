# Final methodological audit report

## Verdict

**PASS** — strict contract checks: 101/101 passed; artifact hash checks: 177/177 passed.

## Leakage controls

- Five frozen primary input files and the C4 turn-level control input are hash-pinned and restricted to the 142 official train/dev participants.
- Every model and control uses the identical saved 10 x 5 participant split membership.
- Train/test participant overlap is zero in all 50 outer folds.
- TF-IDF fitting, embedding-classifier C selection, nested threshold selection, Platt calibration and isotonic calibration occur inside the corresponding outer-training data.
- Template shuffling is performed separately within each outer train and test partition, so donor assignment does not cross the partition boundary.
- Participant-level tests use one averaged OOF probability per person; fold means are not used as independent observations.

## Statistical controls

- Primary input-source family: 40 tests.
- Representation robustness family: 120 tests.
- C5 controls family: 30 tests.
- C4 controls family: 8 tests.
- Every paired test uses 10,000 participant-level swaps and carries a BH-FDR q-value within its frozen family.

## Data-construction controls

- C5 reviewed quote spans matched: 1,137; unmatched: 0.
- Random and non-symptom C5 controls match target word counts for all 142 participants.
- The C4 turn-level input is reconstructed from the verified official142 turn table with the frozen clinical-prompt rule; its aggregated interviewer and non-explicit interviewer text matches frozen C3 and C4 for 142/142 participants.
- C4 length- and position-matched controls match template word count for all 142 participants.
- Both dense embedding model identities and exact revisions are retained in run manifests; neither result set was discarded.

## Interpretation audit

- No TF-IDF ROC-AUC/PR-AUC input-source difference survived the primary-family FDR, so no stable interviewer ranking superiority is claimed.
- Default-threshold sensitivity is kept separate from ranking metrics.
- C5 domain features are described only as extracted symptom-domain evidence states.
- C4 findings are attributed to joint protocol signal, not isolated natural-language semantics.
- The cohort is described as official train+dev and not as independent external validation.
- LLM-derived evidence is not presented as a diagnostic endpoint.

## Remaining limitations

- Small, selected cohort and class imbalance.
- No independent cohort and no second full end-to-end rerun.
- Confidence intervals were not part of the frozen plan; paired permutation p/q values and full point estimates are supplied instead.
- Isotonic calibration is supplementary because of small-sample overfitting risk.
- The C4 operational definition uses explicit-clinical-prompt turns because the broader spoken-protocol flag leaves insufficient non-template text; this deviation is frozen in the changelog.
