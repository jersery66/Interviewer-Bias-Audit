# Claim-evidence matrix

This table constrains wording across the title, abstract, results, discussion, and conclusion.

| Claim | Evidence | Permitted wording | Prohibited wording |
|---|---|---|---|
| Raw participant speech contains predictive information | Locked five-fold AUC 0.717, 95% CI 0.614-0.811 | Participant speech contains cohort-level ranking information for the PHQ-8 label | Participant speech is a reliable clinical screen or diagnostic biomarker |
| Interviewer speech contains predictive information | Locked five-fold AUC 0.808, 95% CI 0.729-0.879 | Interviewer-side text contains label-related predictive information | Interviewers caused bias, leaked the label, or inflated performance |
| Interviewer text is not proven more important than participant text | Paired Delta AUC 0.091, 95% CI -0.000 to 0.190, q=0.438 | The interviewer point estimate was higher, but the paired interval included zero | Interviewer signal is stronger or more important |
| Interviewer text adds little to the joint model | M5-M3 Delta AUC 0.003, 95% CI -0.004 to 0.010, q=0.473 | No detectable conditional increment was observed under the current joint representation | Interviewer speech has no information in absolute terms |
| Process structure is predictive | Protocol AUC 0.675; all-structure AUC 0.729 | Protocol and interaction structure carry label-related information | Structure is a causal bias mechanism |
| Pure length is insufficient | Length-only AUC 0.569; source-token-matched control retains a mean difference | Text quantity alone does not explain the source pattern | Length has no effect, or remaining differences are necessarily semantic |
| Fixed 0.50 sensitivity is low | Participant sensitivity 0.047 at 0.50; nested sensitivity 0.488; raw calibration slope 12.836 | The prespecified 0.50 operating point is poorly aligned with the raw probability scale | The model has no signal or has failed clinically |
| C5 is a useful derived representation | C5 AUC 0.735; all 1137 final spans source-traceable | Clinically guided extraction concentrates label-related information into an auditable representation | C5 is an independent natural source or proves pathological understanding |
| C5 quality has unresolved review limits | 35 corrected spans; label columns visible; no rater IDs or agreement | Outcome blinding cannot be demonstrated and inter-rater reliability is unavailable | C5 review was blinded or independently double-rated |
| E-DAIC mismatch is descriptive only | 9/22 fixed-cutoff mismatch; AUC CIs cross 0.5; definition-feature overlap | A small descriptive extension was inconclusive | External validation, replication, generalizability, or independent error-mechanism validation |
| PHQ-8 is the modeled target | Self-report total and threshold label | PHQ-8 label prediction or symptom-severity classification | Psychiatric diagnosis, treatment decision, or clinical deployment |
