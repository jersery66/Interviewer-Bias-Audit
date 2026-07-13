# D-P / D-All source-variant analysis contract

**Lock date:** 2026-07-13
**Status:** post-audit analysis contract; no D-All model results are generated yet
**Preregistration:** none. This contract was created after the existing source and embedding results and is not an original prespecified analysis.

## Purpose

The current ten-domain control must be identified by its text source before its conditional increment is interpreted. This contract separates:

- **D-P:** ten symptom-domain counts derived from reviewed C5 spans whose exact quotes are matched to participant language;
- **D-All:** the same ten-domain schema extracted from a frozen full-interview transcript input containing participant and interviewer turns.

D-P and D-All are alternative control definitions. They must not be pooled, substituted, or renamed after results are inspected.

## D-P: current frozen input

The current control is D-P. Its committed input is:

`analysis_v2/04_c5_controls/domain_count/input.csv`

The source audit (`domain_source_audit.json`) confirms:

| Audit field | Value |
|---|---:|
| Frozen participants | 142 |
| Reviewed spans used for counts | 1,137 |
| Quotes matching participant text | 1,137 |
| Quotes matching interviewer text | 0 |
| Interviewer-only matches | 0 |
| Unmatched quotes | 0 |
| Domain-count cell reconciliation errors | 0 |

The audit does not claim that C5 extraction is unbiased or that the domains are independent of the PHQ-8 construct. It only establishes the provenance of the current numeric control. D-P is therefore the correct label for all existing L3/L4 conditional results that use `domain_count/input.csv`, including the formal MPNet/BGE embedding-incremental outputs.

The ten columns and their order are frozen as:

`anhedonia_interest`, `appetite_weight`, `concentration_psychomotor`, `depressed_mood`, `functioning_impairment`, `mental_health_history`, `protective_or_absent_symptom`, `self_worth_guilt`, `sleep_fatigue_energy`, `suicide_self_harm`.

Counts are computed from retained reviewed spans, including the frozen domain and polarity records used by the existing C5 control build. No PHQ label, outer-test label, or post-result imputation enters the count construction.

## D-All: required input and current status

D-All may be modeled only after a new, frozen full-interview extraction is available. The input must:

1. contain exactly the same 142 participant IDs and the same ten domain columns;
2. be derived from `c1_full_transcript.csv` (or a byte-hashed equivalent) rather than from symptom-evidence-only text;
3. record the extraction prompt/rule version, model or annotator, domain schema, empty-text rule, and quality-control decisions in a provenance manifest;
4. demonstrate that PHQ labels are not used as extraction inputs;
5. include a deterministic source-scope field equal to `full_transcript`, `full_interview`, or `all_interview`;
6. pass numeric, finite-value, ID, and domain-schema checks before any model is run.

No such frozen full-transcript D-All input was found in the available committed or restricted C5 artifacts during this audit. Existing `c5_mimo_full142` files are labelled `symptom_evidence_only`; they cannot be relabelled D-All. `domain_source_audit.json` therefore records `d_all.status = missing` and `ready_for_modeling = false`.

Until a qualifying D-All input is frozen, no D-P-versus-D-All performance difference, attenuation comparison, or claim about whether interviewer increment disappears under full-interview symptom control may be reported.

## Locked comparison once D-All exists

The same participant-level split file, embedding caches, classifier, inner-CV grid, and inference settings will be reused for D-P and D-All. The primary comparison is the change in interviewer conditional increment:

- L3: `P + D -> P + D + I`;
- L4: `P + R + D -> P + R + D + I`.

For each representation and fusion method, report AUC, PR-AUC, Brier and log-loss deltas with participant-paired bootstrap intervals and the locked paired permutation procedure. D-P and D-All are compared descriptively unless a new multiplicity family is explicitly frozen before the D-All run.

The current D-P results remain valid as D-P results. They are not retroactively upgraded to D-All results, and missing D-All is a data-availability boundary rather than permission to construct a proxy from protocol variables or participant-only evidence.

## Interpretation and reporting boundary

Permitted terms are `participant-derived symptom-domain control`, `full-interview symptom-domain control` (only after D-All passes the gate), `conditional increment`, `residual increment`, `source overlap`, and `overcontrol risk`.

Do not call D-P absence of interviewer quotes `severe train-test leakage`. The relevant limitation is source contamination/endogenous control or overcontrol: D-P may represent symptom content generated in the interaction while still being label-independent at the extraction stage.

Do not use VIF on the 768/1024-dimensional embeddings. For late fusion, use low-dimensional input correlations, condition numbers, coefficient/repeat stability, and mismatch sensitivity. For early concatenation, report dimensionality/regularization and pairing sensitivity rather than per-coordinate VIF.

Primary and exploratory FDR families remain those in the locked embedding-incremental plan. A global BH correction over all 16 AUC deltas may be added only as a clearly labelled sensitivity table; it must not silently replace the locked primary L4 family.
