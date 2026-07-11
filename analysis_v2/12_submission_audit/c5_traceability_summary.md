# C5 evidence traceability audit

C5 is a clinically guided, LLM-assisted derived representation. It is not a natural source equivalent to raw participant speech.

- Candidates entering review: 1143.
- Final retained spans: 1137.
- Retained without source correction: 1102 (96.4% of candidates).
- Corrected to exact participant-source wording: 35 (3.1% of candidates).
- Excluded duplicates: 5; excluded invalid span: 1.
- Final spans traceable to normalized participant source: 1137/1137.
- Unique normalized character location: 1136/1137; repeated exact phrase: 1; not found: 0.
- Participants with retained evidence: 141/142.
- Frozen C5 participant texts match the reviewed condition rows: True.

## Blinding and review boundary

The retained manual-review audit table contains PHQ-8 score/label columns. Outcome blinding therefore cannot be demonstrated and must not be claimed. Reviewer identity fields and a second independent C5 rating are not documented, so inter-rater reliability is unavailable. These are limitations of the existing C5 representation, not values to impute retrospectively.

The public traceability CSV omits quote text. It contains participant ID, domain, polarity, source order, source-text hash, quote hash, normalized character offsets, match multiplicity, and review status.
