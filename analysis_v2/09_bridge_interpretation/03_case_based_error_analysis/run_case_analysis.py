"""
Case-based error analysis for DAIC-WOZ PHQ-8 classification

This script selects illustrative participant-level cases to demonstrate:
1. Threshold artifacts (C2 default vs nested)
2. Protocol-driven signals (C2 low, C3/template high)
3. Evidence-concentration boundaries (C5 close to domain features)
4. Source-level conflicts (C1 vs single sources)

Author: Supplementary analysis for Interviewer-Bias-Audit
Date: 2026-07-04
"""

import pandas as pd
import numpy as np
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = "E:/CodexWorktrees/DAIC-WOZ/reanalysis-v2/analysis_v2"
OUTPUT_DIR = f"{BASE_DIR}/09_bridge_interpretation/03_case_based_error_analysis"

# OOF file paths (participant-level)
OOF_FILES = {
    'c1': f"{BASE_DIR}/02_tfidf_main/oof_probs/full_transcript_participant_oof.csv",
    'c2': f"{BASE_DIR}/02_tfidf_main/oof_probs/participant_speech_participant_oof.csv",
    'c3': f"{BASE_DIR}/02_tfidf_main/oof_probs/interviewer_speech_participant_oof.csv",
    'c4': f"{BASE_DIR}/02_tfidf_main/oof_probs/non_explicit_interviewer_speech_participant_oof.csv",
    'c5': f"{BASE_DIR}/02_tfidf_main/oof_probs/participant_symptom_evidence_participant_oof.csv",
    'domain_presence': f"{BASE_DIR}/04_c5_controls/oof_probs/domain_presence_participant_oof.csv",
    'domain_count': f"{BASE_DIR}/04_c5_controls/oof_probs/domain_count_participant_oof.csv",
    'template_only': f"{BASE_DIR}/05_c4_controls/template_only/participant_oof.csv",
    'template_presence': f"{BASE_DIR}/05_c4_controls/template_presence/participant_oof.csv"
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_and_merge_oof_files():
    """Load all participant-level OOF files and merge into a wide table."""
    print("Loading OOF files...")

    # Load first file to get base (participant_id, label)
    df_base = pd.read_csv(OOF_FILES['c2'])[['participant_id', 'label']]
    print(f"  Base: {df_base.shape}")

    # Initialize merged DataFrame
    df_merged = df_base.copy()

    # Define columns to extract from each OOF file
    prob_col = 'raw_probability'
    default_pred_col = 'prediction_at_0_5'
    nested_pred_col = 'nested_prediction_max_macro_f1'

    # Merge each file
    for key, path in OOF_FILES.items():
        try:
            df = pd.read_csv(path)

            # Check if required columns exist
            if prob_col in df.columns:
                df_merged[f'{key}_prob'] = df[prob_col].values

            if default_pred_col in df.columns:
                df_merged[f'{key}_default_pred'] = df[default_pred_col].values

            if nested_pred_col in df.columns:
                df_merged[f'{key}_nested_pred'] = df[nested_pred_col].values

            print(f"  {key}: loaded successfully")
        except Exception as e:
            print(f"  {key}: FAILED to load ({e})")

    print(f"Merged shape: {df_merged.shape}")
    return df_merged

def select_case_type_a(df):
    """
    Case Type A: C2 default threshold fails but nested threshold recovers.

    Selection rules:
    - label == 1
    - c2_default_pred == 0
    - c2_nested_pred == 1
    - Sort by c2_prob descending (closest to 0.5 but below)
    """
    print("\nSelecting Case Type A (C2 threshold artifact)...")

    mask = (
        (df['label'] == 1) &
        (df['c2_default_pred'] == 0) &
        (df['c2_nested_pred'] == 1)
    )

    candidates = df[mask].copy()
    candidates = candidates.sort_values('c2_prob', ascending=False)

    print(f"  Candidates: {len(candidates)}")
    print(f"  Top candidate: participant_id={candidates.iloc[0]['participant_id']}, c2_prob={candidates.iloc[0]['c2_prob']:.3f}")

    # Select top 2
    selected = candidates.head(2).copy()
    selected['case_type'] = 'A'
    selected['pattern'] = 'C2 default false negative, nested positive'
    selected['interpretation'] = 'Threshold artifact: C2 has signal but default 0.5 threshold creates false negative'

    return selected, candidates

def select_case_type_b(df):
    """
    Case Type B: Interview protocol signal high, but participant speech signal low.

    Selection rules:
    - label == 1
    - c2_prob < 0.5
    - (template_only_prob >= 0.5 OR c3_prob >= 0.5)
    - Sort by abs(template_only_prob - c2_prob) descending
    """
    print("\nSelecting Case Type B (Protocol signal vs participant signal)...")

    mask = (
        (df['label'] == 1) &
        (df['c2_prob'] < 0.5) &
        ((df['template_only_prob'] >= 0.5) | (df['c3_prob'] >= 0.5))
    )

    candidates = df[mask].copy()
    candidates['prob_diff'] = np.abs(candidates['template_only_prob'] - candidates['c2_prob'])
    candidates = candidates.sort_values('prob_diff', ascending=False)

    print(f"  Candidates: {len(candidates)}")

    if len(candidates) > 0:
        print(f"  Top candidate: participant_id={candidates.iloc[0]['participant_id']}, c2_prob={candidates.iloc[0]['c2_prob']:.3f}, template_prob={candidates.iloc[0]['template_only_prob']:.3f}")

    # Select top 1-2
    selected = candidates.head(2).copy()
    selected['case_type'] = 'B'
    selected['pattern'] = 'Protocol high, participant low'
    selected['interpretation'] = 'Protocol structure carries signal independent of participant symptom language'

    return selected, candidates

def select_case_type_c(df):
    """
    Case Type C: C5 and domain features give similar predictions.

    Selection rules:
    - label == 1 OR label == 0
    - abs(c5_prob - domain_presence_prob) is small
    - Sort by abs(c5_prob - domain_presence_prob)
    """
    print("\nSelecting Case Type C (C5 close to domain features)...")

    # Compute absolute difference
    df['c5_domain_diff'] = np.abs(df['c5_prob'] - df['domain_presence_prob'])

    mask = (
        (df['c5_domain_diff'] < 0.1) |  # Small difference
        ((df['c5_prob'] >= 0.5) & (df['domain_presence_prob'] >= 0.5)) |  # Both high
        ((df['c5_prob'] < 0.5) & (df['domain_presence_prob'] < 0.5))  # Both low
    )

    candidates = df[mask].copy()
    candidates = candidates.sort_values('c5_domain_diff', ascending=True)

    print(f"  Candidates: {len(candidates)}")

    if len(candidates) > 0:
        print(f"  Top candidate: participant_id={candidates.iloc[0]['participant_id']}, c5_prob={candidates.iloc[0]['c5_prob']:.3f}, domain_prob={candidates.iloc[0]['domain_presence_prob']:.3f}")

    # Select top 1-2
    selected = candidates.head(2).copy()
    selected['case_type'] = 'C'
    selected['pattern'] = 'C5 close to domain features'
    selected['interpretation'] = 'Symptom evidence and domain coverage give similar predictions, suggesting domain coverage drives much of C5 signal'

    return selected, candidates

def select_case_type_d(df):
    """
    Case Type D: Full transcript vs single source conflict.

    Selection rules:
    - abs(c1_prob - c2_prob) is large
    - OR c1_prob >= 0.5 but c2_prob < 0.5
    - OR c1_prob < 0.5 but c3_prob >= 0.5
    """
    print("\nSelecting Case Type D (Full transcript vs single source conflict)...")

    # Compute absolute difference
    df['c1_c2_diff'] = np.abs(df['c1_prob'] - df['c2_prob'])

    mask = (
        (df['c1_c2_diff'] > 0.3) |  # Large difference
        ((df['c1_prob'] >= 0.5) & (df['c2_prob'] < 0.5)) |  # C1 positive, C2 negative
        ((df['c1_prob'] < 0.5) & (df['c3_prob'] >= 0.5))  # C1 negative, C3 positive
    )

    candidates = df[mask].copy()
    candidates = candidates.sort_values('c1_c2_diff', ascending=False)

    print(f"  Candidates: {len(candidates)}")

    if len(candidates) > 0:
        print(f"  Top candidate: participant_id={candidates.iloc[0]['participant_id']}, c1_prob={candidates.iloc[0]['c1_prob']:.3f}, c2_prob={candidates.iloc[0]['c2_prob']:.3f}")

    # Select top 1-2
    selected = candidates.head(2).copy()
    selected['case_type'] = 'D'
    selected['pattern'] = 'Full transcript vs single source conflict'
    selected['interpretation'] = 'Full transcript includes both participant and interviewer, cannot be directly interpreted as participant language contribution'

    return selected, candidates

def create_case_analysis_table(selected_cases):
    """Create a case analysis table for manuscript."""
    # Select relevant columns
    cols = ['case_type', 'participant_id', 'label',
            'c2_prob', 'c3_prob', 'template_only_prob',
            'c5_prob', 'domain_presence_prob',
            'pattern', 'interpretation']

    df_table = selected_cases[cols].copy()

    return df_table

def write_case_selection_log(selected_cases_dict, candidates_dict, output_path):
    """Write a case selection log to document the selection process."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Case Selection Log\n\n")

        for case_type in ['A', 'B', 'C', 'D']:
            f.write(f"## Case Type {case_type}\n\n")

            candidates = candidates_dict[case_type]
            selected = selected_cases_dict[case_type]

            f.write(f"### Selection Rules\n\n")

            if case_type == 'A':
                f.write("- label == 1\n")
                f.write("- c2_default_pred == 0\n")
                f.write("- c2_nested_pred == 1\n")
                f.write("- Sort by c2_prob descending (closest to 0.5 but below)\n\n")
            elif case_type == 'B':
                f.write("- label == 1\n")
                f.write("- c2_prob < 0.5\n")
                f.write("- template_only_prob >= 0.5 OR c3_prob >= 0.5\n")
                f.write("- Sort by abs(template_only_prob - c2_prob) descending\n\n")
            elif case_type == 'C':
                f.write("- abs(c5_prob - domain_presence_prob) is small\n")
                f.write("- OR both c5_prob and domain_presence_prob are high/low\n")
                f.write("- Sort by abs(c5_prob - domain_presence_prob) ascending\n\n")
            elif case_type == 'D':
                f.write("- abs(c1_prob - c2_prob) > 0.3\n")
                f.write("- OR c1_prob >= 0.5 but c2_prob < 0.5\n")
                f.write("- OR c1_prob < 0.5 but c3_prob >= 0.5\n")
                f.write("- Sort by abs(c1_prob - c2_prob) descending\n\n")

            f.write(f"### Candidate Count: {len(candidates)}\n\n")
            f.write(f"### Selected Count: {len(selected)}\n\n")

            f.write("### Selected Cases\n\n")
            for idx, row in selected.iterrows():
                f.write(f"- Participant {row['participant_id']}: ")
                f.write(f"label={row['label']}, ")
                f.write(f"c2_prob={row['c2_prob']:.3f}, ")
                f.write(f"interpretation: {row['interpretation']}\n")

            f.write("\n")

def main():
    """Run the case-based error analysis."""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load and merge OOF files
    print("=" * 60)
    print("Loading and merging OOF files...")
    print("=" * 60)
    df_merged = load_and_merge_oof_files()

    # Save candidate cases (all candidates)
    print("\nSaving merged data...")
    df_merged.to_csv(f"{OUTPUT_DIR}/candidate_cases.csv", index=False)
    print(f"  Saved to: {OUTPUT_DIR}/candidate_cases.csv")

    # Select cases by type
    print("\n" + "=" * 60)
    print("Selecting cases by type...")
    print("=" * 60)

    selected_cases_dict = {}
    candidates_dict = {}

    # Type A
    selected_a, candidates_a = select_case_type_a(df_merged)
    selected_cases_dict['A'] = selected_a
    candidates_dict['A'] = candidates_a

    # Type B
    selected_b, candidates_b = select_case_type_b(df_merged)
    selected_cases_dict['B'] = selected_b
    candidates_dict['B'] = candidates_b

    # Type C
    selected_c, candidates_c = select_case_type_c(df_merged)
    selected_cases_dict['C'] = selected_c
    candidates_dict['C'] = candidates_c

    # Type D
    selected_d, candidates_d = select_case_type_d(df_merged)
    selected_cases_dict['D'] = selected_d
    candidates_dict['D'] = candidates_d

    # Combine selected cases
    print("\n" + "=" * 60)
    print("Combining selected cases...")
    print("=" * 60)

    df_selected = pd.concat(selected_cases_dict.values(), ignore_index=True)
    print(f"Total selected cases: {len(df_selected)}")

    # Save selected cases
    df_selected.to_csv(f"{OUTPUT_DIR}/selected_cases_for_manuscript.csv", index=False)
    print(f"  Saved to: {OUTPUT_DIR}/selected_cases_for_manuscript.csv")

    # Create case analysis table
    print("\nCreating case analysis table...")
    df_table = create_case_analysis_table(df_selected)
    df_table.to_csv(f"{OUTPUT_DIR}/case_analysis_table.csv", index=False)
    print(f"  Saved to: {OUTPUT_DIR}/case_analysis_table.csv")

    # Write case selection log
    print("\nWriting case selection log...")
    write_case_selection_log(selected_cases_dict, candidates_dict, f"{OUTPUT_DIR}/case_selection_log.md")
    print(f"  Saved to: {OUTPUT_DIR}/case_selection_log.md")

    # Create Markdown summary
    print("\nCreating Markdown summary...")
    with open(f"{OUTPUT_DIR}/case_analysis_table.md", 'w', encoding='utf-8') as f:
        f.write("# Case Analysis Table\n\n")
        f.write(df_table.to_string(index=False))

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
