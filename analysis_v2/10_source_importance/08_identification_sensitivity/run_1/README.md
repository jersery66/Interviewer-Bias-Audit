# Identification sensitivity results

Formal post-audit run on the frozen 142-person cohort.

- `pairing_*`: real, random-deranged, and structure-matched interviewer pairings; 50 draws for each random condition.
- `fake_d_*`: real D-P versus 50 row-permuted Fake-D controls.
- `phq_partition_*`: D-PHQ and D-Extra domain partitions.
- `run_manifest.json`: fixed seeds, input/cache hashes, model settings, and CSV output hashes.

All prediction files contain participant-level outer-test predictions. The run uses one repeated five-fold split (repeat 1) with three inner folds and the fixed C grid; it is a post-audit sensitivity run, not a preregistered confirmatory family. D-All is not included.
