"""
Figure generation script for DAIC-WOZ source importance manuscript.

Figures:
  Figure 1: Source decomposition framework diagram (schematic)
  Figure 2: Incremental model AUC comparison (grouped bar)
  Figure 3: Permutation importance with 95% CI (horizontal bar)
  Figure 4: Domain leave-one-out importance (horizontal bar, top-5)

Output: SVG + PDF + TIFF at 600 dpi, 183 mm full-width journal format.

Data source: analysis_v2/10_source_importance/ corrected CV results
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = "E:/CodexWorktrees/DAIC-WOZ/reanalysis-v2/analysis_v2/06_tables_figures/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Nature-family style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "figure.dpi": 300,
})

# NMI pastel palette (low-saturation)
COLORS = {
    'c2': '#7BA7C9',        # patient language - blue
    'c2_light': '#B8D4E8',
    'domain': '#C8A4D4',    # domain count - purple
    'protocol': '#F4B183',  # template presence - orange
    'c5': '#A8D8B9',        # C5 evidence - green
    'c3': '#E8A0BF',        # C3 interviewer - pink
    'm3': '#5B8DB8',        # combined M3 - dark blue
    'm0': '#D4D4D4',        # baseline - grey
    'phq_core': '#7BA7C9',  # PHQ-core blue
    'clinical': '#F4B183',  # clinical-context orange
    'protective': '#A8D8B9', # protective green
}

def save_pub(fig, filename):
    """Save in SVG, PDF, and TIFF formats."""
    fig.savefig(f"{OUTPUT_DIR}/{filename}.svg", bbox_inches="tight")
    fig.savefig(f"{OUTPUT_DIR}/{filename}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUTPUT_DIR}/{filename}.tiff", dpi=600, bbox_inches="tight")
    print(f"  Saved: {OUTPUT_DIR}/{filename}.{{svg,pdf,tiff}}")

# =============================================================================
# FIGURE 1: Source Decomposition Framework Diagram
# =============================================================================

def make_figure_1():
    """Schematic diagram showing the 3-layer signal decomposition framework."""
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Title
    ax.text(5, 11.5, 'DAIC-WOZ PHQ-8 Signal Source Decomposition Framework',
            ha='center', fontsize=9, fontweight='bold')

    # Top node: Full Transcript
    box_top = mpatches.FancyBboxPatch((3, 10), 4, 0.8, boxstyle="round,pad=0.15",
                                        facecolor='#F0F0F0', edgecolor='#888888', linewidth=1)
    ax.add_patch(box_top)
    ax.text(5, 10.4, 'Complete Interview Transcript', ha='center', fontsize=8, fontweight='bold')

    # Three columns
    col_centers = [1.8, 5.0, 8.2]
    col_labels = ['Language Content', 'Clinical-Domain State', 'Interview Protocol Structure']
    col_colors = ['#D6E8F5', '#E8D6F5', '#F5E6D6']
    col_items = [
        ['Patient Speech', 'Interviewer Speech', 'Symptom Evidence Text'],
        ['Domain Presence', 'Domain Count'],
        ['Template Presence', 'Module Entry State'],
    ]

    for i, (cx, label, color, items) in enumerate(zip(col_centers, col_labels, col_colors, col_items)):
        # Column header
        box_hdr = mpatches.FancyBboxPatch((cx-1.3, 8.5), 2.6, 0.6, boxstyle="round,pad=0.1",
                                            facecolor=color, edgecolor='black', linewidth=0.5)
        ax.add_patch(box_hdr)
        ax.text(cx, 8.8, label, ha='center', fontsize=6.5, fontweight='bold')

        # Column items
        for j, item in enumerate(items):
            y = 7.5 - j * 0.7
            box_item = mpatches.FancyBboxPatch((cx-1.1, y-0.25), 2.2, 0.5, boxstyle="round,pad=0.08",
                                                facecolor='white', edgecolor='#AAAAAA', linewidth=0.5)
            ax.add_patch(box_item)
            ax.text(cx, y, item, ha='center', fontsize=6)

    # Arrows from top to columns
    for cx in col_centers:
        ax.annotate('', xy=(cx, 9.2), xytext=(5, 9.2),
                    arrowprops=dict(arrowstyle='->', color='#888888', lw=1))

    # Bottom: Source-Level Incremental Model
    box_model = mpatches.FancyBboxPatch((1.5, 3.8), 7, 0.8, boxstyle="round,pad=0.15",
                                          facecolor='#E8E8E8', edgecolor='black', linewidth=1)
    ax.add_patch(box_model)
    ax.text(5, 4.2, 'Source-Level Incremental Model (M0 → M3)', ha='center', fontsize=8, fontweight='bold')

    # Test questions
    questions = [
        'Does patient speech contain baseline signal?',
        'Do domain states add incremental info beyond patient speech?',
        'Does protocol structure add incremental info beyond patient speech?',
    ]
    for j, q in enumerate(questions):
        ax.text(5, 3.2 - j * 0.5, q, ha='center', fontsize=6, style='italic', color='#555555')

    # Bottom: Key Finding
    box_finding = mpatches.FancyBboxPatch((1, 0.8), 8, 1.2, boxstyle="round,pad=0.15",
                                           facecolor='#FFF3CD', edgecolor='#D4A017', linewidth=1)
    ax.add_patch(box_finding)
    ax.text(5, 1.65, 'Key Finding', ha='center', fontsize=7, fontweight='bold', color='#8B6914')
    ax.text(5, 1.2, 'Domain states and protocol structure provide the main incremental contribution;\n'
                     'patient speech has baseline predictive power but limited independent contribution.',
            ha='center', fontsize=6, color='#555555')

    fig.tight_layout(pad=0.3)
    save_pub(fig, 'figure_1_source_decomposition_framework')
    plt.close()
    print("Figure 1: Source decomposition framework - DONE")

# =============================================================================
# FIGURE 2: Incremental Model AUC Bar Chart
# =============================================================================

def make_figure_2():
    """Grouped bar chart showing AUC for M0-M5 incremental models."""
    models = ['Baseline\n(C2 only)', 'C2+Domain\n(M1)', 'C2+Protocol\n(M2)',
              'C2+Domain\n+Protocol (M3)', 'M3+C5\n(M4)', 'M3+C3\n(M5)']
    auc_values = [0.6956, 0.7710, 0.7634, 0.8252, 0.8217, 0.8356]
    colors_bar = [COLORS['c2_light'], COLORS['domain'], COLORS['protocol'],
                  COLORS['m3'], COLORS['c5'], COLORS['c3']]

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.6))

    x = np.arange(len(models))
    bars = ax.bar(x, auc_values, width=0.6, color=colors_bar, edgecolor='white', linewidth=0.5,
                  zorder=3)

    # Annotate M0 baseline
    ax.axhline(y=0.6956, color='#888888', linestyle='--', linewidth=0.8, zorder=2)
    ax.text(5.2, 0.6956, 'M0 baseline = 0.6956', fontsize=6, color='#888888', va='bottom')

    # Delta annotation between M3 and M0
    ax.annotate('ΔAUC = +0.1294\n95% CI [+0.046, +0.214]',
                xy=(3, 0.8252), xytext=(3, 0.88),
                fontsize=6, ha='center', fontweight='bold', color='#2B5F8A',
                arrowprops=dict(arrowstyle='->', color='#2B5F8A', lw=1))

    # Label values on bars
    for bar, val in zip(bars, auc_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', fontsize=6, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=6.5)
    ax.set_ylabel('AUC', fontsize=7, fontweight='bold')
    ax.set_ylim(0.55, 0.92)
    ax.set_xlim(-0.5, 5.5)
    ax.grid(axis='y', alpha=0.3, zorder=1)

    # Title
    ax.set_title('Incremental Model AUC Comparison', fontsize=8, fontweight='bold', pad=8)

    fig.tight_layout(pad=0.3)
    save_pub(fig, 'figure_2_incremental_model_auc')
    plt.close()
    print("Figure 2: Incremental model AUC - DONE")

# =============================================================================
# FIGURE 3: Permutation Importance with 95% CI
# =============================================================================

def make_figure_3():
    """Horizontal bar chart showing permutation importance with 95% CI."""
    sources = ['Domain Count\nProbability', 'Template Presence\nProbability',
               'Patient Speech\nProbability']
    delta_auc = [0.1810, 0.1272, -0.0054]
    ci_lower = [0.1663, 0.1177, -0.0094]
    ci_upper = [0.1959, 0.1367, -0.0016]
    colors_perm = [COLORS['domain'], COLORS['protocol'], COLORS['c2_light']]

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 2.8))

    y_pos = np.arange(len(sources))[::-1]  # Reverse so top is at top

    for i, (y, d, lo, hi, color) in enumerate(zip(y_pos, delta_auc, ci_lower, ci_upper, colors_perm)):
        # Bar
        ax.barh(y, d, height=0.5, color=color, edgecolor='white', linewidth=0.5, zorder=3)
        # CI whiskers
        ax.plot([lo, hi], [y, y], color='#555555', linewidth=1.5, zorder=4)
        ax.plot([lo, lo], [y-0.1, y+0.1], color='#555555', linewidth=1, zorder=4)
        ax.plot([hi, hi], [y-0.1, y+0.1], color='#555555', linewidth=1, zorder=4)
        # Value label
        ax.text(max(d, 0) + 0.005, y, f'{d:+.4f}', fontsize=7, fontweight='bold', va='center')
        # CI label
        ax.text(max(d, 0) + 0.04, y, f'[{lo:+.4f}, {hi:+.4f}]', fontsize=5.5, va='center', color='#555555')

    # Zero line
    ax.axvline(x=0, color='black', linewidth=0.8, zorder=2, linestyle='-')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sources, fontsize=7)
    ax.set_xlabel('Δ AUC (permutation importance)', fontsize=7, fontweight='bold')
    ax.set_title('Permutation Importance in M3 Model\n(100x repeated shuffles per fold, fixed seed)',
                 fontsize=8, fontweight='bold', pad=8)

    ax.set_xlim(-0.06, 0.28)
    ax.grid(axis='x', alpha=0.3, zorder=1)

    fig.tight_layout(pad=0.3)
    save_pub(fig, 'figure_3_permutation_importance')
    plt.close()
    print("Figure 3: Permutation importance - DONE")

# =============================================================================
# FIGURE 4: Domain Leave-One-Out Importance
# =============================================================================

def make_figure_4():
    """Horizontal bar chart for domain leave-one-out analysis (presence + count)."""
    # Presence domain data (top-5 by delta AUC)
    domain_labels = ['Mental Health\nHistory', 'Functional\nImpairment',
                     'Depressed\nMood', 'Suicide /\nSelf-Harm', 'Appetite /\nWeight']
    presence_delta = [0.0594, 0.0592, 0.0409, 0.0195, 0.0195]
    count_delta = [0.0129, 0.0413, 0.0108, 0.0092, 0.0110]

    domain_colors = ['#F4B183', '#F4B183', '#7BA7C9', '#7BA7C9', '#7BA7C9']  # orange=clinical, blue=PHQ-core
    domain_groups = ['Clinical Context', 'Clinical Context', 'PHQ Core', 'PHQ Core', 'PHQ Core']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))

    # --- Presence panel ---
    y_pos = np.arange(len(domain_labels))[::-1]
    bars1 = ax1.barh(y_pos, presence_delta, height=0.5, color=domain_colors, edgecolor='white', linewidth=0.5, zorder=3)
    for y, d in zip(y_pos, presence_delta):
        ax1.text(d + 0.002, y, f'{d:+.4f}', fontsize=6.5, fontweight='bold', va='center')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(domain_labels, fontsize=6.5)
    ax1.set_xlabel('Δ AUC (decrease when removed)', fontsize=6.5, fontweight='bold')
    ax1.set_title('Presence Domains', fontsize=7.5, fontweight='bold')
    ax1.axvline(x=0, color='black', linewidth=0.8, zorder=2)
    ax1.set_xlim(-0.005, 0.075)
    ax1.grid(axis='x', alpha=0.3, zorder=1)

    # Legend for domain groups
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#F4B183', label='Clinical Context'),
                       Patch(facecolor='#7BA7C9', label='PHQ Core')]
    ax1.legend(handles=legend_elements, fontsize=5.5, loc='lower right')

    # --- Count panel ---
    bars2 = ax2.barh(y_pos, count_delta, height=0.5, color=domain_colors, edgecolor='white', linewidth=0.5, zorder=3)
    for y, d in zip(y_pos, count_delta):
        ax2.text(d + 0.002, y, f'{d:+.4f}', fontsize=6.5, fontweight='bold', va='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(domain_labels, fontsize=6.5)
    ax2.set_xlabel('Δ AUC (decrease when removed)', fontsize=6.5, fontweight='bold')
    ax2.set_title('Count Domains', fontsize=7.5, fontweight='bold')
    ax2.axvline(x=0, color='black', linewidth=0.8, zorder=2)
    ax2.set_xlim(-0.005, 0.055)
    ax2.grid(axis='x', alpha=0.3, zorder=1)

    fig.suptitle('Domain Leave-One-Out Importance Analysis', fontsize=8, fontweight='bold', y=1.03)
    fig.tight_layout(pad=0.5)
    save_pub(fig, 'figure_4_domain_leave_one_out')
    plt.close()
    print("Figure 4: Domain leave-one-out - DONE")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Generating manuscript figures...\n")
    make_figure_1()
    make_figure_2()
    make_figure_3()
    make_figure_4()
    print(f"\nAll figures saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
