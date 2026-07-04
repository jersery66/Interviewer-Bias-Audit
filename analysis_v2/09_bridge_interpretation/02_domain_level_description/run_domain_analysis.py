"""
Domain-level description analysis for DAIC-WOZ PHQ-8 classification

This script computes domain-level group comparisons between PHQ-8 positive and negative participants.

Author: Supplementary analysis for Interviewer-Bias-Audit
Date: 2026-07-04
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

# Try to import matplotlib, but make it optional
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, skipping figure generation")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Domain grouping for interpretation
DOMAIN_GROUPS = {
    'PHQ-core': [
        'anhedonia_interest',
        'depressed_mood',
        'sleep_fatigue_energy',
        'appetite_weight',
        'self_worth_guilt',
        'concentration_psychomotor',
        'suicide_self_harm'
    ],
    'clinical-context': [
        'functioning_impairment',
        'mental_health_history'
    ],
    'protective/denial': [
        'protective_or_absent_symptom'
    ]
}

# Flatten domain list
ALL_DOMAINS = []
for group_domains in DOMAIN_GROUPS.values():
    ALL_DOMAINS.extend(group_domains)

# Paths
BASE_DIR = "E:/CodexWorktrees/DAIC-WOZ/reanalysis-v2/analysis_v2"
DOMAIN_PRESENCE_PATH = f"{BASE_DIR}/04_c5_controls/domain_presence/input.csv"
DOMAIN_COUNT_PATH = f"{BASE_DIR}/04_c5_controls/domain_count/input.csv"
PARTICIPANT_OOF_PATH = f"{BASE_DIR}/02_tfidf_main/oof_probs/participant_speech_participant_oof.csv"
OUTPUT_DIR = f"{BASE_DIR}/09_bridge_interpretation/02_domain_level_description"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_domain_group(domain):
    """Get the group name for a domain."""
    for group, domains in DOMAIN_GROUPS.items():
        if domain in domains:
            return group
    return 'other'

def compute_odds_ratio(a, b, c, d):
    """
    Compute odds ratio with Haldane-Anscombe correction (+0.5 to each cell).

    Contingency table:
            domain=1    domain=0
    pos      a           b
    neg      c           d
    """
    a_corr = a + 0.5
    b_corr = b + 0.5
    c_corr = c + 0.5
    d_corr = d + 0.5

    or_val = (a_corr * d_corr) / (b_corr * c_corr)
    return or_val

def compute_phi(a, b, c, d):
    """Compute phi coefficient (correlation for 2x2 table)."""
    n = a + b + c + d
    if n == 0:
        return 0
    phi = (a * d - b * c) / np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return phi

def fdr_correction(pvals, method='fdr_bh'):
    """
    Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    pvals : array-like
        Array of p-values
    method : str
        FDR method (only 'fdr_bh' implemented)

    Returns
    -------
    rejected : array
        Boolean array indicating rejected null hypotheses
    pvals_corrected : array
        Corrected p-values
    """
    pvals = np.array(pvals)
    n = len(pvals)

    # Sort p-values
    sort_idx = np.argsort(pvals)
    pvals_sorted = pvals[sort_idx]

    # Compute FDR threshold
    q = 0.05  # FDR level
    thresholds = (np.arange(1, n+1) / n) * q

    # Find largest p-value that passes threshold
    rejected_sorted = pvals_sorted <= thresholds

    # Compute corrected p-values
    pvals_corrected_sorted = np.minimum(1, pvals_sorted * n / np.arange(1, n+1))
    pvals_corrected_sorted = np.minimum.accumulate(pvals_corrected_sorted[::-1])[::-1]

    # Map back to original order
    pvals_corrected = np.zeros(n)
    rejected = np.zeros(n, dtype=bool)
    pvals_corrected[sort_idx] = pvals_corrected_sorted
    rejected[sort_idx] = rejected_sorted

    return rejected, pvals_corrected

# =============================================================================
# MAIN ANALYSIS FUNCTIONS
# =============================================================================

def load_and_merge_data():
    """Load domain data and merge with labels."""
    # Load domain presence
    df_presence = pd.read_csv(DOMAIN_PRESENCE_PATH)
    print(f"Domain presence shape: {df_presence.shape}")
    print(f"Domains: {list(df_presence.columns[1:])}")

    # Load domain count
    df_count = pd.read_csv(DOMAIN_COUNT_PATH)
    print(f"Domain count shape: {df_count.shape}")

    # Load labels
    df_labels = pd.read_csv(PARTICIPANT_OOF_PATH)[['participant_id', 'label']]
    print(f"Labels shape: {df_labels.shape}")
    print(f"Positive: {df_labels['label'].sum()}, Negative: {(df_labels['label'] == 0).sum()}")

    # Merge
    df_presence_merged = df_presence.merge(df_labels, on='participant_id')
    df_count_merged = df_count.merge(df_labels, on='participant_id')

    return df_presence_merged, df_count_merged

def analyze_domain_presence(df):
    """
    Analyze domain presence by group.

    For each domain, compute:
    - positive_n, negative_n
    - positive_presence_rate, negative_presence_rate
    - risk_difference
    - odds_ratio (with Haldane-Anscombe correction)
    - fisher exact test p-value
    - FDR correction
    """
    results = []

    for domain in ALL_DOMAINS:
        # Subset by label
        pos = df[df['label'] == 1]
        neg = df[df['label'] == 0]

        # Contingency table
        a = (pos[domain] == 1).sum()  # pos, domain=1
        b = (pos[domain] == 0).sum()  # pos, domain=0
        c = (neg[domain] == 1).sum()  # neg, domain=1
        d = (neg[domain] == 0).sum()  # neg, domain=0

        # Rates
        pos_rate = a / (a + b) if (a + b) > 0 else 0
        neg_rate = c / (c + d) if (c + d) > 0 else 0
        risk_diff = pos_rate - neg_rate

        # Odds ratio with correction
        or_val = compute_odds_ratio(a, b, c, d)
        log_or = np.log(or_val) if or_val > 0 else 0

        # Fisher exact test
        table = [[a, b], [c, d]]
        oddsratio, pvalue = stats.fisher_exact(table)

        results.append({
            'domain': domain,
            'domain_group': get_domain_group(domain),
            'positive_n': len(pos),
            'negative_n': len(neg),
            'positive_presence_rate': pos_rate,
            'negative_presence_rate': neg_rate,
            'risk_difference': risk_diff,
            'odds_ratio': or_val,
            'log_or': log_or,
            'fisher_p': pvalue,
            'a': a,
            'b': b,
            'c': c,
            'd': d
        })

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # FDR correction
    pvals = df_results['fisher_p'].values
    reject, pvals_corrected = fdr_correction(pvals)
    df_results['fdr_bh_q'] = pvals_corrected

    # Phi coefficient
    for idx, row in df_results.iterrows():
        a, b, c, d = int(row['a']), int(row['b']), int(row['c']), int(row['d'])
        phi = compute_phi(a, b, c, d)
        df_results.loc[idx, 'phi'] = phi

    return df_results

def analyze_domain_count(df):
    """
    Analyze domain count by group.

    For each domain, compute:
    - positive_mean_count, negative_mean_count
    - mean_difference
    - medians
    - Mann-Whitney U test
    - FDR correction
    """
    results = []

    for domain in ALL_DOMAINS:
        # Subset by label
        pos = df[df['label'] == 1]
        neg = df[df['label'] == 0]

        pos_counts = pos[domain].values
        neg_counts = neg[domain].values

        # Means
        pos_mean = np.mean(pos_counts)
        neg_mean = np.mean(neg_counts)
        mean_diff = pos_mean - neg_mean

        # Medians
        pos_median = np.median(pos_counts)
        neg_median = np.median(neg_counts)

        # Mann-Whitney U test
        try:
            u_stat, p_value = stats.mannwhitneyu(pos_counts, neg_counts, alternative='two-sided')
        except:
            u_stat, p_value = np.nan, 1.0

        # Cliff's delta (effect size)
        try:
            n_pos = len(pos_counts)
            n_neg = len(neg_counts)
            # Compute proportion of pos > neg - proportion of pos < neg
            pos_gt_neg = sum(x > y for x in pos_counts for y in neg_counts) / (n_pos * n_neg)
            pos_lt_neg = sum(x < y for x in pos_counts for y in neg_counts) / (n_pos * n_neg)
            cliffs_delta = pos_gt_neg - pos_lt_neg
        except:
            cliffs_delta = np.nan

        results.append({
            'domain': domain,
            'domain_group': get_domain_group(domain),
            'positive_mean_count': pos_mean,
            'negative_mean_count': neg_mean,
            'mean_difference': mean_diff,
            'positive_median': pos_median,
            'negative_median': neg_median,
            'mannwhitney_u': u_stat,
            'mannwhitney_p': p_value,
            'cliffs_delta': cliffs_delta
        })

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # FDR correction
    pvals = df_results['mannwhitney_p'].values
    reject, pvals_corrected = fdr_correction(pvals)
    df_results['fdr_bh_q'] = pvals_corrected

    return df_results

def create_domain_summary_table(df_presence_results, df_count_results):
    """Create a summary table combining presence and count results."""
    # Merge presence and count results
    df_summary = df_presence_results[['domain', 'domain_group', 'positive_presence_rate',
                                       'negative_presence_rate', 'risk_difference',
                                       'odds_ratio', 'fisher_p', 'fdr_bh_q']].copy()

    # Add count results
    df_count_subset = df_count_results[['domain', 'positive_mean_count',
                                         'negative_mean_count', 'mean_difference',
                                         'mannwhitney_p', 'fdr_bh_q']].copy()
    df_count_subset = df_count_subset.rename(columns={'fdr_bh_q': 'count_fdr_bh_q'})
    df_summary = df_summary.merge(df_count_subset, on='domain')

    # Add interpretation
    def interpret(row):
        if row['fdr_bh_q'] < 0.05:
            if row['risk_difference'] > 0:
                return 'Higher in positive group (significant)'
            else:
                return 'Higher in negative group (significant)'
        else:
            return 'Not significant after FDR'

    df_summary['interpretation'] = df_summary.apply(interpret, axis=1)

    return df_summary

def create_figure_domain_difference(df_presence_results, output_path):
    """
    Create figure showing domain-level group differences.

    x-axis: domain
    y-axis: positive_presence_rate - negative_presence_rate
    """
    # Sort by risk_difference
    df_plot = df_presence_results.copy()
    df_plot = df_plot.sort_values('risk_difference', ascending=False)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Colors by group
    colors = {
        'PHQ-core': '#1f77b4',
        'clinical-context': '#ff7f0e',
        'protective/denial': '#2ca02c'
    }

    # Plot
    x_pos = np.arange(len(df_plot))
    colors_list = [colors[group] for group in df_plot['domain_group']]

    bars = ax.bar(x_pos, df_plot['risk_difference'], color=colors_list)

    # Add significance markers
    for i, (idx, row) in enumerate(df_plot.iterrows()):
        if row['fdr_bh_q'] < 0.05:
            ax.text(i, row['risk_difference'] + 0.02, '*',
                   ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_plot['domain'], rotation=45, ha='right')
    ax.set_ylabel('Risk Difference (Positive Rate - Negative Rate)')
    ax.set_xlabel('Symptom Domain')
    ax.set_title('Symptom-Domain Coverage Differences Between PHQ-8 Positive and Negative Participants')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[g], label=g) for g in colors.keys()]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Figure saved to: {output_path}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the domain-level description analysis."""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    print("Loading data...")
    df_presence, df_count = load_and_merge_data()

    # Analyze presence
    print("\nAnalyzing domain presence...")
    df_presence_results = analyze_domain_presence(df_presence)
    print(f"Presence results shape: {df_presence_results.shape}")
    print("\nPresence results (first 5 rows):")
    print(df_presence_results[['domain', 'domain_group', 'positive_presence_rate',
                                'negative_presence_rate', 'risk_difference',
                                'odds_ratio', 'fisher_p', 'fdr_bh_q']].head())

    # Analyze count
    print("\nAnalyzing domain count...")
    df_count_results = analyze_domain_count(df_count)
    print(f"Count results shape: {df_count_results.shape}")
    print("\nCount results (first 5 rows):")
    print(df_count_results[['domain', 'domain_group', 'positive_mean_count',
                             'negative_mean_count', 'mean_difference',
                             'mannwhitney_p', 'fdr_bh_q']].head())

    # Create summary table
    print("\nCreating summary table...")
    df_summary = create_domain_summary_table(df_presence_results, df_count_results)
    print("\nSummary table (first 5 rows):")
    print(df_summary.head())

    # Save results
    print("\nSaving results...")
    df_presence_results.to_csv(f"{OUTPUT_DIR}/domain_presence_group_comparison.csv", index=False)
    df_count_results.to_csv(f"{OUTPUT_DIR}/domain_count_group_comparison.csv", index=False)
    df_summary.to_csv(f"{OUTPUT_DIR}/domain_summary_table.csv", index=False)

    # Create figure (if matplotlib is available)
    if MATPLOTLIB_AVAILABLE:
        print("\nCreating figure...")
        try:
            create_figure_domain_difference(df_presence_results, f"{OUTPUT_DIR}/figure_domain_group_difference.png")
            create_figure_domain_difference(df_presence_results, f"{OUTPUT_DIR}/figure_domain_group_difference.svg")
        except Exception as e:
            print(f"Warning: Could not create figure: {e}")
    else:
        print("\nSkipping figure creation (matplotlib not available)")

    # Create Markdown summary
    print("\nCreating Markdown summary...")
    with open(f"{OUTPUT_DIR}/domain_summary_table.md", 'w', encoding='utf-8') as f:
        f.write("# Domain-Level Group Comparison Results\n\n")

        f.write("## Presence Analysis\n\n")
        f.write("| domain | domain_group | positive_presence_rate | negative_presence_rate | risk_difference | odds_ratio | fisher_p | fdr_bh_q |\n")
        f.write("|--------|-------------|------------------------|------------------------|-----------------|------------|----------|----------|\n")
        for _, row in df_presence_results.iterrows():
            f.write(f"| {row['domain']} | {row['domain_group']} | {row['positive_presence_rate']:.3f} | {row['negative_presence_rate']:.3f} | {row['risk_difference']:.3f} | {row['odds_ratio']:.3f} | {row['fisher_p']:.4f} | {row['fdr_bh_q']:.4f} |\n")

        f.write("\n## Count Analysis\n\n")
        f.write("| domain | domain_group | positive_mean_count | negative_mean_count | mean_difference | mannwhitney_p | fdr_bh_q |\n")
        f.write("|--------|-------------|---------------------|---------------------|-----------------|---------------|----------|\n")
        for _, row in df_count_results.iterrows():
            f.write(f"| {row['domain']} | {row['domain_group']} | {row['positive_mean_count']:.2f} | {row['negative_mean_count']:.2f} | {row['mean_difference']:.2f} | {row['mannwhitney_p']:.4f} | {row['fdr_bh_q']:.4f} |\n")

        f.write("\n## Summary Table\n\n")
        f.write(df_summary.to_string(index=False))

    print("\nDone!")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
