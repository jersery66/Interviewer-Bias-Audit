"""
Generate candidate_cases.csv for case-based error analysis
This script merges all participant-level OOF files and marks case type eligibility
"""

import pandas as pd
import numpy as np

BASE_DIR = "E:/CodexWorktrees/DAIC-WOZ/reanalysis-v2/analysis_v2"

OOF_FILES = {
    'c1': f"{BASE_DIR}/02_tfidf_main/oof_probs/full_transcript_participant_oof.csv",
    'c2': f"{BASE_DIR}/02_tfidf_main/oof_probs/participant_speech_participant_oof.csv",
    'c3': f"{BASE_DIR}/02_tfidf_main/oof_probs/interviewer_speech_participant_oof.csv",
    'c4': f"{BASE_DIR}/02_tfidf_main/oof_probs/non_explicit_interviewer_speech_participant_oof.csv",
    'c5': f"{BASE_DIR}/02_tfidf_main/oof_probs/participant_symptom_evidence_participant_oof.csv",
    'domain_presence': f"{BASE_DIR}/04_c5_controls/oof_probs/domain_presence_participant_oof.csv",
    'domain_count': f"{BASE_DIR}/04_c5_controls/oof_probs/domain_count_participant_oof.csv",
    'template_only': f"{BASE_DIR}/05_c4_controls/template_only/participant_oof.csv",
    'template_presence': f"{BASE_DIR}/05_c4_controls/template_presence/participant_oof.csv",
}

print("Loading OOF files...")
df = pd.read_csv(OOF_FILES['c2'])[['participant_id', 'label']]

for key, path in OOF_FILES.items():
    df_tmp = pd.read_csv(path)
    if 'raw_probability' in df_tmp.columns:
        df[f'{key}_prob'] = df_tmp['raw_probability'].values
    if 'prediction_at_0_5' in df_tmp.columns:
        df[f'{key}_default_pred'] = df_tmp['prediction_at_0_5'].values
    if 'nested_prediction_max_macro_f1' in df_tmp.columns:
        df[f'{key}_nested_pred'] = df_tmp['nested_prediction_max_macro_f1'].values

print(f"Merged shape: {df.shape}")

# Mark case type eligibility
print("\nMarking case type eligibility...")

# Case Type A: C2 default threshold fails but nested recovers
df['eligible_A'] = (
    (df['label'] == 1) &
    (df['c2_default_pred'] == 0) &
    (df['c2_nested_pred'] == 1)
)
print(f"Case Type A eligible: {df['eligible_A'].sum()}")

# Case Type B: Protocol signal high, participant signal low
df['eligible_B'] = (
    (df['label'] == 1) &
    (df['c2_prob'] < 0.5) &
    ((df['template_only_prob'] >= 0.5) | (df['c3_prob'] >= 0.5))
)
print(f"Case Type B eligible: {df['eligible_B'].sum()}")

# Case Type C: C5 close to domain features
df['c5_domain_diff'] = np.abs(df['c5_prob'] - df['domain_presence_prob'])
df['eligible_C'] = (
    (df['c5_domain_diff'] < 0.15) |
    ((df['c5_prob'] >= 0.5) & (df['domain_presence_prob'] >= 0.5)) |
    ((df['c5_prob'] < 0.5) & (df['domain_presence_prob'] < 0.5))
)
print(f"Case Type C eligible: {df['eligible_C'].sum()}")

# Case Type D: Full transcript vs single source conflict
df['c1_c2_diff'] = np.abs(df['c1_prob'] - df['c2_prob'])
df['eligible_D'] = (
    (df['c1_c2_diff'] > 0.3) |
    ((df['c1_prob'] >= 0.5) & (df['c2_prob'] < 0.5)) |
    ((df['c1_prob'] < 0.5) & (df['c3_prob'] >= 0.5))
)
print(f"Case Type D eligible: {df['eligible_D'].sum()}")

# Save to CSV
OUTPUT_PATH = f"{BASE_DIR}/09_bridge_interpretation/03_case_based_error_analysis/candidate_cases.csv"
print(f"\nSaving to: {OUTPUT_PATH}")
df.to_csv(OUTPUT_PATH, index=False)
print("Done!")
