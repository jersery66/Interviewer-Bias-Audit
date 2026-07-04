"""
Step 2: Source feature correlation analysis

Compute Spearman and Pearson correlations between source probabilities.
Focus on: c2 vs c5, c5 vs domain_count, template_only vs template_presence, etc.

Author: Source importance analysis for Interviewer-Bias-Audit
Date: 2026-07-04
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

BASE_DIR = Path("E:/CodexWorktrees/DAIC-WOZ/reanalysis-v2/analysis_v2")
INPUT_PATH = BASE_DIR / "10_source_importance/01_source_level_table/source_level_features.csv"
OUTPUT_DIR = BASE_DIR / "10_source_importance/01_source_level_table"

# Source probability columns to analyze
SOURCE_PROBS = [
    'c2_prob',
    'c5_prob',
    'domain_count_prob',
    'domain_presence_prob',
    'template_presence_prob',
    'template_only_prob',
    'c3_prob',
    'c4_prob',
    'c1_prob',
]

DISPLAY_NAMES = {
    'c2_prob': 'C2 Patient Speech',
    'c5_prob': 'C5 Symptom Evidence',
    'domain_count_prob': 'Domain Count',
    'domain_presence_prob': 'Domain Presence',
    'template_presence_prob': 'Template Presence',
    'template_only_prob': 'Template Only',
    'c3_prob': 'C3 Interviewer',
    'c4_prob': 'C4 Non-explicit',
    'c1_prob': 'C1 Full Transcript',
}

# Key comparisons for summary
KEY_COMPARISONS = [
    ('c2_prob', 'c5_prob'),
    ('c5_prob', 'domain_count_prob'),
    ('c5_prob', 'domain_presence_prob'),
    ('template_only_prob', 'template_presence_prob'),
    ('c3_prob', 'template_only_prob'),
    ('c2_prob', 'template_presence_prob'),
    ('domain_count_prob', 'template_presence_prob'),
    ('c2_prob', 'domain_count_prob'),
    ('c2_prob', 'c3_prob'),
]

def interpret_correlation(r):
    """Interpret correlation strength."""
    abs_r = abs(r)
    if abs_r < 0.30:
        return 'Low overlap'
    elif abs_r < 0.60:
        return 'Moderate overlap: interpret incremental contribution cautiously'
    else:
        return 'High overlap: incremental contribution interpretation requires caution'

def compute_correlation_matrix(df):
    """Compute both Spearman and Pearson correlation matrices."""
    print("Computing correlation matrices...")

    # Rename columns for readability
    df_renamed = df[SOURCE_PROBS].copy()
    rename_map = {col: DISPLAY_NAMES[col] for col in SOURCE_PROBS}
    df_renamed = df_renamed.rename(columns=rename_map)

    # Spearman
    spearman_corr = df_renamed.corr(method='spearman')
    print(f"Spearman shape: {spearman_corr.shape}")

    # Pearson
    pearson_corr = df_renamed.corr(method='pearson')
    print(f"Pearson shape: {pearson_corr.shape}")

    return spearman_corr, pearson_corr

def save_correlation_matrices(spearman_corr, pearson_corr):
    """Save correlation matrices to CSV files."""
    # Save Spearman
    spearman_path = OUTPUT_DIR / "source_feature_correlation_spearman.csv"
    spearman_corr.to_csv(spearman_path)
    print(f"\nSaved: {spearman_path}")

    # Save Pearson
    pearson_path = OUTPUT_DIR / "source_feature_correlation_pearson.csv"
    pearson_corr.to_csv(pearson_path)
    print(f"Saved: {pearson_path}")

    # Save combined matrix with both values
    combined_path = OUTPUT_DIR / "source_feature_correlation_matrix.csv"
    combined = pd.DataFrame()
    for i in range(len(SOURCE_PROBS)):
        for j in range(i+1, len(SOURCE_PROBS)):
            a, b = SOURCE_PROBS[i], SOURCE_PROBS[j]
            row = {
                'feature_a': DISPLAY_NAMES[a],
                'feature_b': DISPLAY_NAMES[b],
                'spearman_r': round(spearman_corr.loc[DISPLAY_NAMES[a], DISPLAY_NAMES[b]], 4),
                'pearson_r': round(pearson_corr.loc[DISPLAY_NAMES[a], DISPLAY_NAMES[b]], 4),
            }
            row['interpretation'] = interpret_correlation(row['spearman_r'])
            combined = pd.concat([combined, pd.DataFrame([row])], ignore_index=True)

    combined = combined.sort_values('spearman_r', key=abs, ascending=False)
    combined.to_csv(combined_path, index=False)
    print(f"Saved: {combined_path}")

    return combined

def create_summary_table(df_combined):
    """Create a summary table with the most important correlations."""
    print("\n" + "=" * 60)
    print("KEY CORRELATION SUMMARY")
    print("=" * 60)

    summary_data = []
    for a, b in KEY_COMPARISONS:
        row = df_combined[
            ((df_combined['feature_a'] == DISPLAY_NAMES[a]) & (df_combined['feature_b'] == DISPLAY_NAMES[b])) |
            ((df_combined['feature_a'] == DISPLAY_NAMES[b]) & (df_combined['feature_b'] == DISPLAY_NAMES[a]))
        ]
        if len(row) > 0:
            r = row.iloc[0]['spearman_r']
            interp = row.iloc[0]['interpretation']
            summary_data.append({
                'feature_a': DISPLAY_NAMES[a],
                'feature_b': DISPLAY_NAMES[b],
                'spearman_r': r,
                'interpretation': interp,
                'question': get_question(a, b)
            })
            print(f"  {DISPLAY_NAMES[a]} vs {DISPLAY_NAMES[b]}: r = {r:.3f} ({interp})")

    df_summary = pd.DataFrame(summary_data)
    return df_summary

def get_question(a, b):
    """Get the research question for each comparison."""
    questions = {
        ('c2_prob', 'c5_prob'): 'Is C5 just a compressed version of patient language?',
        ('c5_prob', 'domain_count_prob'): 'Is C5 signal largely captured by domain count?',
        ('c5_prob', 'domain_presence_prob'): 'Is C5 signal captured by domain presence?',
        ('template_only_prob', 'template_presence_prob'): 'Are template text and template pattern highly overlapping?',
        ('c3_prob', 'template_only_prob'): 'Is interviewer text mainly template structure?',
        ('c2_prob', 'template_presence_prob'): 'Is patient language coupled with protocol path?',
        ('domain_count_prob', 'template_presence_prob'): 'Is symptom-domain status coupled with interview flow?',
        ('c2_prob', 'domain_count_prob'): 'Are patient language and domain count overlapping?',
        ('c2_prob', 'c3_prob'): 'Are patient and interviewer speech correlated?',
    }
    return questions.get((a, b), questions.get((b, a), ''))

def save_summary(df_combined, df_summary):
    """Save summary tables."""
    # Combined matrix
    summary_path = OUTPUT_DIR / "source_feature_correlation_summary.md"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# Source Feature Correlation Analysis\n\n")
        f.write("## Key Findings\n\n")
        f.write("| Feature A | Feature B | Spearman r | Interpretation | Research Question |\n")
        f.write("|-----------|-----------|------------|----------------|-------------------|\n")
        for _, row in df_summary.iterrows():
            f.write(f"| {row['feature_a']} | {row['feature_b']} | {row['spearman_r']:.3f} | {row['interpretation']} | {row['question']} |\n")

        f.write("\n## Interpretation Standards\n\n")
        f.write("| Correlation | Interpretation |\n")
        f.write("|-------------|----------------|\n")
        f.write("| abs(r) < 0.30 | Low overlap |\n")
        f.write("| 0.30 - 0.60 | Moderate overlap: interpret incremental contribution cautiously |\n")
        f.write("| > 0.60 | High overlap: incremental contribution interpretation requires caution |\n\n")

        f.write("## All Correlations (sorted by |Spearman r|)\n\n")
        f.write("| Feature A | Feature B | Spearman r | Pearson r | Interpretation |\n")
        f.write("|-----------|-----------|------------|-----------|----------------|\n")
        for _, row in df_combined.iterrows():
            f.write(f"| {row['feature_a']} | {row['feature_b']} | {row['spearman_r']:.3f} | {row['pearson_r']:.3f} | {row['interpretation']} |\n")

    print(f"\nSaved: {summary_path}")


def main():
    print("=" * 60)
    print("STEP 2: SOURCE FEATURE CORRELATION ANALYSIS")
    print("=" * 60 + "\n")

    # Load data
    print("Loading source-level features...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Shape: {df.shape}\n")

    # Compute correlations
    spearman_corr, pearson_corr = compute_correlation_matrix(df)

    # Save matrices
    df_combined = save_correlation_matrices(spearman_corr, pearson_corr)

    # Create and save summary
    df_summary = create_summary_table(df_combined)
    save_summary(df_combined, df_summary)

    print("\n" + "=" * 60)
    print("STEP 2 COMPLETE")
    print("=" * 60)

if __name__ == '__main__':
    main()
