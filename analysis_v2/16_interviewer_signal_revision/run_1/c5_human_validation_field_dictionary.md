# C5 human-validation field dictionary

The public revision output contains only the deterministic sampling roster, field definitions, and source hashes. Reviewer packets and owner linkage are restricted artifacts and are not written here.

| Field | Meaning |
|---|---|
| `review_case_id` | Opaque candidate-span review identifier |
| `deidentified_quote` | Limited deterministic deidentification of the candidate quote |
| `local_context_reference` | Opaque restricted full-text review reference |
| `human_span_valid` | Independent human validity judgment; blank before review |
| `human_domain` | Independent human domain judgment; blank before review |
| `human_polarity` | Independent human polarity judgment; blank before review |
| `rater_id` | Human coder identifier; blank before review |
| `notes` | Human coder notes; blank before review |
| `fulltext_review_case_id` | Opaque participant-level missing-evidence audit identifier |
| `human_missing_evidence` | Separate full-text audit judgment, not candidate-span agreement |
