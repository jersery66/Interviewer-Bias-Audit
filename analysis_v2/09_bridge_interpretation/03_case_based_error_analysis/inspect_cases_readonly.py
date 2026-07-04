"""
Read-only case analysis script - prints results to stdout
Run this script to see case selection results without writing files
"""

import pandas as pd
import numpy as np

BASE_DIR = "E:/CodexWorktrees/DAIC-WOZ/reanalysis-v2/analysis_v2"

OOF_FILES = {
    'c1': f"{BASE_DIR}/02_tfidf_main/oof_probs/full_transcript_participant_oof.csv",
    'c2': f"{BASE_DIR}/02_tfidf_main/oof_probs/participant_speech_participant_oof.csv",
    'c3': f"{BASE_DIR}/02_tfidf_main/oof_probs/interviewer_speech_participant_oof.csv",
    'c5': f"{BASE_DIR}/02_tfidf_main/oof_probs/participant_symptom_evidence_participant_oof.csv",
    'domain_presence': f"{BASE_DIR}/04_c5_controls/oof_probs/domain_presence_participant_oof.csv",
    'template_only': f"{BASE_DIR}/05_c4_controls/template_only/participant_oof.csv",
}

print("Loading OOF files...")
df_base = pd.read_csv(OOF_FILES['c2'])[['participant_id', 'label', 'raw_probability', 'prediction_at_0_5', 'nested_prediction_max_macro_f1']]
df_base = df_base.rename(columns={'raw_probability': 'c2_prob', 'prediction_at_0_5': 'c2_default_pred', 'nested_prediction_max_macro_f1': 'c2_nested_pred'})

for key, path in OOF_FILES.items():
    if key == 'c2':
        continue
    df = pd.read_csv(path)
    if 'raw_probability' in df.columns:
        df_base[f'{key}_prob'] = df['raw_probability'].values
    if 'prediction_at_0_5' in df.columns:
        df_base[f'{key}_default_pred'] = df['prediction_at_0_5'].values

print(f"\nMerged shape: {df_base.shape}")
print(f"\nColumns: {list(df_base.columns)}")

# Case Type A
print("\n" + "="*60)
print("CASE TYPE A: C2 threshold artifact")
print("="*60)
mask_a = (df_base['label'] == 1) & (df_base['c2_default_pred'] == 0) & (df_base['c2_nested_pred'] == 1)
candidates_a = df_base[mask_a].sort_values('c2_prob', ascending=False)
print(f"Candidates: {len(candidates_a)}")
if len(candidates_a) > 0:
    print("\nTop 3 candidates:")
    print(candidates_a[['participant_id', 'label', 'c2_prob', 'c2_default_pred', 'c2_nested_pred']].head(3).to_string())

# Case Type B
print("\n" + "="*60)
print("CASE TYPE B: Protocol signal vs participant signal")
print("="*60)
if 'template_only_prob' in df_base.columns:
    mask_b = (df_base['label'] == 1) & (df_base['c2_prob'] < 0.5) & (df_base['template_only_prob'] >= 0.5)
    candidates_b = df_base[mask_b].copy()
    candidates_b['prob_diff'] = np.abs(candidates_b['template_only_prob'] - candidates_b['c2_prob'])
    candidates_b = candidates_b.sort_values('prob_diff', ascending=False)
    print(f"Candidates: {len(candidates_b)}")
    if len(candidates_b) > 0:
        print("\nTop 3 candidates:")
        print(candidates_b[['participant_id', 'label', 'c2_prob', 'template_only_prob', 'prob_diff']].head(3).to_string())

print("\nDone. Use these results to manually create case analysis files.")
