"""
Read-only case analysis script v2 - includes all case types
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

print(f"Merged shape: {df.shape}\n")

# Case Type A
print("="*60)
print("CASE TYPE A: C2 threshold artifact (20 candidates)")
print("="*60)
mask_a = (df['label'] == 1) & (df['c2_default_pred'] == 0) & (df['c2_nested_pred'] == 1)
candidates_a = df[mask_a].sort_values('c2_prob', ascending=False)
print(candidates_a[['participant_id', 'label', 'c2_prob', 'c2_default_pred', 'c2_nested_pred']].head(5).to_string())

# Case Type B
print("\n" + "="*60)
print("CASE TYPE B: Protocol signal vs participant signal (31 candidates)")
print("="*60)
mask_b = (df['label'] == 1) & (df['c2_prob'] < 0.5) & (df['template_only_prob'] >= 0.5)
candidates_b = df[mask_b].copy()
candidates_b['prob_diff'] = np.abs(candidates_b['template_only_prob'] - candidates_b['c2_prob'])
candidates_b = candidates_b.sort_values('prob_diff', ascending=False)
print(candidates_b[['participant_id', 'label', 'c2_prob', 'template_only_prob', 'prob_diff']].head(5).to_string())

# Case Type C
print("\n" + "="*60)
print("CASE TYPE C: C5 close to domain features")
print("="*60)
df['c5_domain_diff'] = np.abs(df['c5_prob'] - df['domain_presence_prob'])
mask_c = (df['c5_domain_diff'] < 0.15)
candidates_c = df[mask_c].sort_values('c5_domain_diff', ascending=True)
print(f"Candidates: {len(candidates_c)}")
print(candidates_c[['participant_id', 'label', 'c5_prob', 'domain_presence_prob', 'c5_domain_diff']].head(5).to_string())

# Case Type D
print("\n" + "="*60)
print("CASE TYPE D: Full transcript vs single source conflict")
print("="*60)
df['c1_c2_diff'] = np.abs(df['c1_prob'] - df['c2_prob'])
mask_d = (
    (df['c1_c2_diff'] > 0.3) |
    ((df['c1_prob'] >= 0.5) & (df['c2_prob'] < 0.5)) |
    ((df['c1_prob'] < 0.5) & (df['c3_prob'] >= 0.5))
)
candidates_d = df[mask_d].copy()
candidates_d = candidates_d.sort_values('c1_c2_diff', ascending=False)
print(f"Candidates: {len(candidates_d)}")
print(candidates_d[['participant_id', 'label', 'c1_prob', 'c2_prob', 'c1_c2_diff']].head(5).to_string())

print("\nDone!")
