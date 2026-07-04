"""
Generate two supplementary tables:
1. C4/C5 signal mechanism interpretation table
2. Case-pattern categories summary table
"""

import pandas as pd
import os

OUTPUT_DIR = "E:/CodexWorktrees/DAIC-WOZ/reanalysis-v2/analysis_v2/09_bridge_interpretation"

# ============================================================
# Table 1: C4/C5 Signal Mechanism Interpretation
# ============================================================

mechanism_data = [
    {
        'signal_category': 'C5: Symptom evidence text',
        'analysis_condition': 'c5_reviewed_evidence',
        'condition_description': 'Human-reviewed symptom quotes from participant speech',
        'roc_auc': 0.754,
        'physical_mechanism': 'Symptom quotes condense clinically-relevant content into short text spans',
        'statistical_mechanism': 'TF-IDF over condensed quotes yields above-chance AUC, but nested thresholds needed',
        'interpretation': 'Usable but not dominant; domain-coverage features match or exceed its performance',
        'boundary': 'Cannot claim specific quote semantics drive prediction beyond domain-state encoding'
    },
    {
        'signal_category': 'C5: Symptom-domain coverage (presence)',
        'analysis_condition': 'domain_presence',
        'condition_description': 'Binary vector of 10 symptom-domain presence states',
        'roc_auc': 0.790,
        'physical_mechanism': 'Symptom-domain coverage captures whether each PHQ domain was discussed',
        'statistical_mechanism': 'Linear combination of domain-presence bits, boosted by clinical-context domains (functioning_impairment, mental_health_history)',
        'interpretation': 'Domain coverage captures most C5 discriminative information',
        'boundary': 'Does not encode quote-level semantics; derived from C5 pipeline, not independent'
    },
    {
        'signal_category': 'C5: Symptom-domain coverage (count)',
        'analysis_condition': 'domain_count',
        'condition_description': 'Count vector of evidence mentions per symptom domain',
        'roc_auc': 0.794,
        'physical_mechanism': 'Evidence-count profile reflects how extensively each domain was mentioned',
        'statistical_mechanism': 'Count adds marginal information over presence (0.794 vs 0.790); functioning_impairment count is key in leave-one-out',
        'interpretation': 'Similar to presence; count marginally richer but not meaningfully distinct',
        'boundary': 'Count may partially reflect interview length rather than clinical severity'
    },
    {
        'signal_category': 'C5: Keyword-masked control',
        'analysis_condition': 'keyword_masked',
        'condition_description': 'C5 quotes with symptom keywords removed',
        'roc_auc': 0.715,
        'physical_mechanism': 'Remaining text after symptom keywords removed; tests whether domain keywords drive performance',
        'statistical_mechanism': 'Performance drops from C5 (0.754) to masked (0.715), suggesting keywords contribute but are not sole driver',
        'interpretation': 'Both keywords and surrounding context contribute; neither alone sufficient',
        'boundary': 'Masking is syntactic, not semantic; does not remove implicit symptom references'
    },
    {
        'signal_category': 'C5: Non-symptom same-word control',
        'analysis_condition': 'nonsymptom_same_word',
        'condition_description': 'Non-symptom quotes matched by keyword count',
        'roc_auc': 0.678,
        'physical_mechanism': 'Tests whether any text with similar domain-term frequency carries signal, regardless of clinical content',
        'statistical_mechanism': 'Lower AUC than C5 (0.754), confirming symptom-quote content matters beyond keyword frequency',
        'interpretation': 'C5 advantage is not purely lexical; quote semantics provide distinct signal',
        'boundary': 'Same-word matching is approximate; residual confounds possible'
    },
    {
        'signal_category': 'C5: Random same-word control (10 seeds)',
        'analysis_condition': 'c5_random_same_word (10 seeds)',
        'condition_description': 'Randomly selected non-symptom quotes with matching domain-term count, 10 random seeds',
        'roc_auc': 0.688,
        'physical_mechanism': 'Tests whether C5 advantage over random-same-word baselines is stable across sampling',
        'statistical_mechanism': 'C5 (0.754) exceeds random-same-word ensemble (mean AUC ≈ 0.688); random seed sensitivity is moderate (SD ≈ 0.05)',
        'interpretation': 'C5 consistently outperforms random-same-word baselines; quote content matters',
        'boundary': '10-seed sampling may miss worst/best-case random draws'
    },
    {
        'signal_category': 'C4: Template-only text',
        'analysis_condition': 'template_only',
        'condition_description': 'Only explicit interview template utterances (e.g., "How has your sleep been?")',
        'roc_auc': 0.754,
        'physical_mechanism': 'Template questions follow a fixed depression-assessment protocol with predictable topic transitions',
        'statistical_mechanism': 'Template text alone achieves AUC comparable to C5 (0.754 vs 0.754); protocol structure encodes label-relevant information',
        'interpretation': 'Interview protocol structure carries discriminative signal independent of patient speech',
        'boundary': 'Cannot say interviewer semantics drive prediction; template structure is confounded with interview phase'
    },
    {
        'signal_category': 'C4: Template presence vector',
        'analysis_condition': 'template_presence',
        'condition_description': 'Binary vector indicating which template patterns occurred in the interview',
        'roc_auc': 0.759,
        'physical_mechanism': 'Template-presence encodes which protocol sections were completed for each participant',
        'statistical_mechanism': 'Template presence (0.759) matches template text (0.754); pattern alone is sufficient',
        'interpretation': 'Protocol-structure signal is robust to text-vs-pattern encoding',
        'boundary': 'Presence vector is derived from text extraction; not an independent protocol variable'
    },
    {
        'signal_category': 'C4: Template shuffled control',
        'analysis_condition': 'template_shuffled',
        'condition_description': 'Template text with utterance order randomized per participant',
        'roc_auc': 0.561,
        'physical_mechanism': 'Destroys sequential protocol structure while preserving vocabulary',
        'statistical_mechanism': 'AUC drops from template_only (0.754) to shuffled (0.561); sequential structure is essential',
        'interpretation': 'Protocol signal resides in the structured interview sequence, not just template vocabulary',
        'boundary': 'Shuffling is within-participant; cross-participant confounds not tested'
    },
    {
        'signal_category': 'C4: Non-template text (length-matched)',
        'analysis_condition': 'nontemplate_lengthmatched',
        'condition_description': 'Non-template interviewer utterances trimmed to match template length distribution',
        'roc_auc': 0.637,
        'physical_mechanism': 'Tests whether non-template interviewer speech carries label signal after controlling for length',
        'statistical_mechanism': 'AUC drops from C3 (0.803) to length-matched (0.637); template structure is the key interviewer-side signal source',
        'interpretation': 'Non-template interviewer speech contributes little beyond template structure',
        'boundary': 'Length-matching is approximate; residual acoustic/prosodic confounds possible'
    },
    {
        'signal_category': 'C4: Non-template text (position-matched)',
        'analysis_condition': 'nontemplate_positionmatched',
        'condition_description': 'Non-template interviewer utterances matched by interview-position bin',
        'roc_auc': 0.804,
        'physical_mechanism': 'Tests whether interviewer speech at specific interview positions carries signal beyond template content',
        'statistical_mechanism': 'Position-matched non-template text (0.804) matches C3 (0.803); non-template at same position may carry contamination from adjacent template content',
        'interpretation': 'Position-matching reveals likely template contamination in non-template speech',
        'boundary': 'Position binning may be too coarse to separate template/non-template'
    },
]

df_mechanism = pd.DataFrame(mechanism_data)
df_mechanism.to_csv(f"{OUTPUT_DIR}/signal_mechanism_interpretation.csv", index=False)

# Markdown version
with open(f"{OUTPUT_DIR}/signal_mechanism_interpretation.md", 'w', encoding='utf-8') as f:
    f.write("# Table SX. Interpretation of C4/C5 Signal Mechanisms\n\n")
    f.write("This table summarizes the operational meaning and interpretation boundaries ")
    f.write("for each C4 and C5 control analysis condition.\n\n")

    f.write("| Condition | AUC | Physical Mechanism | Statistical Mechanism | Interpretation | Boundary |\n")
    f.write("|-----------|-----|-------------------|----------------------|----------------|----------|\n")
    for _, row in df_mechanism.iterrows():
        f.write(f"| {row['condition_description']} | {row['roc_auc']:.3f} | {row['physical_mechanism']} | "
                f"{row['statistical_mechanism']} | {row['interpretation']} | {row['boundary']} |\n")

    f.write("\n## Signal Categories\n\n")

    # C5 section
    f.write("### C5: Symptom Evidence Controls\n\n")
    f.write("The C5 control suite tests whether symptom-quote-based prediction relies on ")
    f.write("(a) specific quote semantics, (b) domain-coverage states, (c) keyword frequency, or ")
    f.write("(d) random same-word effects.\n\n")
    f.write("**Key finding**: domain_presence/count (AUC=0.790-0.794) matches or exceeds C5 reviewed evidence (AUC=0.754), ")
    f.write("suggesting symptom-domain coverage captures most of C5's discriminative information. ")
    f.write("Source-level incremental models confirm C5 adds zero incremental AUC beyond domain_count (M4 vs M3: ΔAUC=-0.004, CI crosses zero).\n\n")

    # C4 section
    f.write("### C4: Interview Protocol Structure Controls\n\n")
    f.write("The C4 control suite tests whether interviewer-side signal originates from ")
    f.write("(a) template question content, (b) sequential protocol structure, ")
    f.write("(c) non-template interviewer speech, or (d) template occurrence patterns.\n\n")
    f.write("**Key finding**: template_only (AUC=0.754) and template_presence (AUC=0.759) ")
    f.write("are nearly identical, indicating protocol-structure signal is robust to encoding choice. ")
    f.write("Template shuffled control (AUC=0.561) confirms that sequential structure, not vocabulary, drives the signal.")

print("Table 1 done: signal_mechanism_interpretation.md")

# ============================================================
# Table 2: Case-Pattern Categories Summary
# ============================================================

case_pattern_data = [
    {
        'case_type': 'A: Threshold artifact',
        'selection_rule': 'label=1, c2_default_pred=0, c2_nested_pred=1; sort by c2_prob descending',
        'n_candidates': 20,
        'representative_case': 'Participant 414 (c2_prob=0.494)',
        'what_it_demonstrates': 'C2 participant speech contains signal, but default 0.5 threshold creates false negatives for borderline cases',
        'interpretation': 'Default threshold decision rule creates artifacts; nested threshold confirms C2 signal is present',
        'implication_for_manuscript': 'Low sensitivity is a threshold artifact, not absence of C2 signal'
    },
    {
        'case_type': 'B: Protocol signal dominance',
        'selection_rule': 'label=1, c2_prob<0.5, template_only_prob≥0.5; sort by |template_only−c2| descending',
        'n_candidates': 32,
        'representative_case': 'Participant 356 (c2_prob=0.481, template_only=0.848)',
        'what_it_demonstrates': 'Model assigns high risk under template condition but low under participant-only condition',
        'interpretation': 'Protocol structure carries label-related signal independent of participant symptom language',
        'implication_for_manuscript': 'Interviewer-side signal originates from protocol structure, not interviewer semantic understanding'
    },
    {
        'case_type': 'C: Evidence-domain convergence',
        'selection_rule': 'abs(c5_prob − domain_presence_prob) < 0.15; sort by absolute difference ascending',
        'n_candidates': 36,
        'representative_case': 'Participant 343 (c5_prob=0.482, domain_presence=0.479, diff=0.003)',
        'what_it_demonstrates': 'C5 symptom evidence and domain-presence features give nearly identical predictions',
        'interpretation': 'Symptom evidence discrimination is largely captured by domain-coverage state',
        'implication_for_manuscript': 'C5 quote semantics contribute little beyond what domain presence already encodes'
    },
    {
        'case_type': 'D: Source conflict',
        'selection_rule': 'c1_pred ≠ c2_pred (full transcript vs participant speech disagreement)',
        'n_candidates': 51,
        'representative_case': 'Participant 305 (c1_prob=0.519 pred=1, c2_prob=0.481 pred=0)',
        'what_it_demonstrates': 'Full transcript incorrectly predicts positive while participant-only speech correctly predicts negative',
        'interpretation': 'C1 includes both participant and interviewer information; cannot be directly interpreted as patient symptom language',
        'implication_for_manuscript': 'Full-transcript models should not be described as depression-language detectors'
    },
]

df_patterns = pd.DataFrame(case_pattern_data)
df_patterns.to_csv(f"{OUTPUT_DIR}/case_pattern_categories.csv", index=False)

with open(f"{OUTPUT_DIR}/case_pattern_categories.md", 'w', encoding='utf-8') as f:
    f.write("# Table SX. Prespecified Case-Pattern Categories for Error Analysis\n\n")
    f.write("Case selection was performed using prespecified rules applied to all 142 participants. ")
    f.write("Cases are illustrative examples, not statistical evidence.\n\n")

    f.write("| Type | Selection Rule | N Eligible | Representative | Demonstrates | Interpretation |\n")
    f.write("|------|---------------|-----------|---------------|-------------|----------------|\n")
    for _, row in df_patterns.iterrows():
        f.write(f"| {row['case_type']} | {row['selection_rule']} | {row['n_candidates']} | "
                f"{row['representative_case']} | {row['what_it_demonstrates']} | {row['interpretation']} |\n")

    f.write("\n## Selection Methodology\n\n")
    f.write("All selection rules were specified before inspecting individual participant data. ")
    f.write("The candidate pool (`candidate_cases.csv`) contains all 142 participants with eligibility flags. ")
    f.write("Cases were selected programmatically from the top-ranked candidates in each category.\n\n")

    f.write("## Manuscript Use\n\n")
    f.write("This table supports the case-based error analysis in Supplementary Materials. ")
    f.write("It demonstrates that case selection was systematic, not cherry-picked. ")
    f.write("No de-identified transcripts or quotes are included in the main text.\n")

print("Table 2 done: case_pattern_categories.md")
print("\nAll tables generated in:", OUTPUT_DIR)
