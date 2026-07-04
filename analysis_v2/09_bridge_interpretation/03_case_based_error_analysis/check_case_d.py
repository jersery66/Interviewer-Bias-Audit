"""
Check Case Type D more carefully - look for prediction disagreement
"""

import pandas as pd

BASE_DIR = "E:/CodexWorktrees/DAIC-WOZ/reanalysis-v2/analysis_v2"

# Load data
df = pd.read_csv(f"{BASE_DIR}/02_tfidf_main/oof_probs/participant_speech_participant_oof.csv")[['participant_id', 'label']]
df['c1_prob'] = pd.read_csv(f"{BASE_DIR}/02_tfidf_main/oof_probs/full_transcript_participant_oof.csv")['raw_probability'].values
df['c2_prob'] = pd.read_csv(f"{BASE_DIR}/02_tfidf_main/oof_probs/participant_speech_participant_oof.csv")['raw_probability'].values
df['c3_prob'] = pd.read_csv(f"{BASE_DIR}/02_tfidf_main/oof_probs/interviewer_speech_participant_oof.csv")['raw_probability'].values
df['c1_pred'] = pd.read_csv(f"{BASE_DIR}/02_tfidf_main/oof_probs/full_transcript_participant_oof.csv")['prediction_at_0_5'].values
df['c2_pred'] = pd.read_csv(f"{BASE_DIR}/02_tfidf_main/oof_probs/participant_speech_participant_oof.csv")['prediction_at_0_5'].values
df['c3_pred'] = pd.read_csv(f"{BASE_DIR}/02_tfidf_main/oof_probs/interviewer_speech_participant_oof.csv")['prediction_at_0_5'].values

print("Case Type D: Prediction disagreement examples\n")

# C1 predicts positive, C2 predicts negative
mask1 = (df['c1_pred'] == 1) & (df['c2_pred'] == 0)
print(f"C1 pred=1, C2 pred=0: {mask1.sum()} candidates")
if mask1.sum() > 0:
    print(df[mask1][['participant_id', 'label', 'c1_prob', 'c2_prob', 'c1_pred', 'c2_pred']].head(5).to_string())

print("\n")

# C1 predicts negative, C3 predicts positive
mask2 = (df['c1_pred'] == 0) & (df['c3_pred'] == 1)
print(f"C1 pred=0, C3 pred=1: {mask2.sum()} candidates")
if mask2.sum() > 0:
    print(df[mask2][['participant_id', 'label', 'c1_prob', 'c3_prob', 'c1_pred', 'c3_pred']].head(5).to_string())

print("\nDone!")
