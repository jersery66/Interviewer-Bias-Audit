# C5 human–machine agreement data gap

## Finding

No true C5 human–machine agreement raw dataset is currently available.

The available C5 files contain model-generated evidence spans and post-hoc
manual source corrections. The current official-142 traceability summary
reports 1,137 final spans, 35 source corrections, no documented reviewer
identity columns, no demonstrated blinding, and interrater reliability as not
available.

## Missing independent labels

The following are still needed before formal C5 human–machine agreement can be
computed:

- independent human span-validity labels;
- human confirmation or rejection of each model exact quote;
- independent human domain mapping across all ten D-P domains;
- independent human polarity labels;
- participant-level human domain-presence matrix;
- reviewer IDs, independent raw records, and a locked sampling roster;
- if interrater agreement is desired, a second independent human coder for the
  same items before adjudication.

## Terminology boundary

The 24-participant model-run comparison is **LLM extraction repeatability**.
The 14-case two-rater study is **error-attribution interrater agreement**.
Neither can be relabeled as C5 human–machine agreement.

The existing post-hoc manual correction log can support source-traceability
and correction-provenance reporting, but it cannot support precision, recall,
F1, Cohen's kappa, or domain-level validity against an independent human
reference.

## Planned preparation

A blank C5 validation dataset will be generated from the canonical official-142
model spans. It will include model fields and empty human fields only. It must
not be populated by inference from the correction log, the PHQ label, model
correctness, or the existing D-P counts.
