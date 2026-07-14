# D-P blinded manual audit lock record (packet 2)

**Lock status:** preparation complete; reviewer input pending  
**Lock date:** 2026-07-14  
**Analysis status:** post-audit, non-preregistered content-validity check

This record supersedes the minimal packet-1 preparation contract for the
formal human-review workflow. The analysis sample remains the same 30-person
label × participant-word-count-tertile sample (five per cell; seed `20260714`).
Five cases outside the formal sample are training-only and are excluded from
all agreement, consensus, and D-P validity statistics.

## Locked materials

- `packet_2/blind_audit_protocol.md`: order of operations and unblinding gate;
- `packet_2/blind_codebook.md`: ten domain definitions, 0/1/NS labels,
  0/1/2/3+ count bins, polarity, evidence, and NS rules;
- `packet_2/blind_cases.csv` and `blind_case_lines.csv`: 30 blinded cases and
  stable 1-based line numbers;
- `packet_2/reviewer_A_blank.csv` and `reviewer_B_blank.csv`: two complete
  300-cell forms with different case order;
- `packet_2/training_*`: five excluded training cases and blank forms;
- `packet_2/evidence_review_template.csv`: second-round evidence-validity
  schema, not a completed result.

The owner-only `blind_id_key.csv`, `training_id_key.csv`, and
`scoring_reference_owner_only.csv` are written under the ignored
`blind_quality_audit/private/packet_2/` directory. They must not be sent to
reviewers or committed to the public repository.

## Frozen sequence

1. Independent training, then codebook freeze.
2. Independent completion of all 300 formal cells by both reviewers.
3. Hash and retain both raw files; calculate three-category, binary complete-
   case, and count-bin inter-rater metrics before discussion.
4. Generate the disagreement sheet and adjudicate without D-P exposure.
5. Save consensus labels with both raw reviewer labels and an adjudication
   reason; unresolved cells remain NS or go to a third adjudicator.
6. Only then unblind the owner key and calculate D-P versus consensus metrics.
7. Run the separate evidence-level validity round if a frozen D-P evidence
   source is available. Do not infer evidence-level results from domain counts.

No Cohen's κ, consensus F1, precision, recall, or accuracy is reported while
the two formal reviewer files are absent. If reviewer B is unavailable, the
output is downgraded to a single-reviewer blinded content-validity sample.

