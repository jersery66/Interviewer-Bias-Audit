# Public release policy

## Purpose

This repository is a code-and-aggregate-results release for the DAIC-WOZ source-signal audit. It is not a redistribution of DAIC-WOZ or of the private human-review materials used to validate symptom-evidence proxies.

## Included

- analysis and reporting source code;
- tests that use synthetic or temporary fixtures;
- manuscript drafts, methods, analysis plans and claim-boundary documents;
- aggregate performance estimates, confidence intervals and sensitivity summaries;
- final publication figures in PDF, PNG or SVG format;
- non-text run manifests needed to interpret aggregate results.

## Excluded

- transcripts, utterances, question-answer pairs and any other interview text;
- extracted or reviewed symptom-evidence quotes;
- blinded case text, line indexes and training examples;
- participant identifiers, participant-level labels, split assignments and row-level predictions;
- reviewer-completed forms, consensus tables, owner keys and private audit ledgers;
- embedding caches, temporary files, duplicate formal-run copies and redundant TIFF exports.

## Participant-ID validation

Every release candidate must be scanned locally against the official participant roster before publication. The roster must be supplied from a restricted path outside the public checkout and must never be copied, generated or committed here. A non-zero participant-ID match blocks release. Public CI has no access to the restricted roster; it therefore validates the scanner with synthetic fixtures and continues to enforce roster-independent path and file-boundary checks.

## Interpretation boundary

The public files permit review of the methods and reported aggregate results. They do not permit full data-to-result reproduction without separately authorized access to DAIC-WOZ and the private audit source. Aggregate predictive results are cohort- and operationalization-specific and do not establish clinical utility, diagnosis, label leakage or a causal interviewer effect.
