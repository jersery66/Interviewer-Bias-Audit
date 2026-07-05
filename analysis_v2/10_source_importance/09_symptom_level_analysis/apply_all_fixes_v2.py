"""Apply all 10 fixes to run_symptom_level_analysis.py.
Restored from git commit 327ee8a, then all fixes applied.
"""
import re

FPATH = 'E:/CodexWorktrees/DAIC-WOZ/reanalysis-v2/analysis_v2/10_source_importance/09_symptom_level_analysis/run_symptom_level_analysis.py'
with open(FPATH, 'r', encoding='utf-8') as f:
    c = f.read()

# ═══════════════════════════════════════════════════════════════
# FIX 1: BASE → relative path
# ═══════════════════════════════════════════════════════════════
OLD = 'BASE = Path("E:/CodexWorktrees/DAIC-WOZ/reanalysis-v2/analysis_v2")'
NEW = 'SCRIPT_DIR = Path(__file__).parent\nBASE = SCRIPT_DIR.parent.parent'
assert OLD in c, f"FIX 1 FAIL: '{OLD}' not found"
c = c.replace(OLD, NEW)
print("FIX 1 ✓ BASE → relative")

# ═══════════════════════════════════════════════════════════════
# FIX 2: Add prediction_swap_permutation_test() with seed param
# (original doesn't have this function; add it before bootstrap_delta_auc_ci)
# ═══════════════════════════════════════════════════════════════
PERM_FUNC = '''
def prediction_swap_permutation_test(y_true, prob_a, prob_b, n_perm=PERMUTATION_N, seed=None):
    """Paired prediction-swap permutation test for ΔAUC.
    H0: model A and model B have equivalent predictive utility.
    Returns: p-value (two-sided).
    """
    auc_obs = roc_auc_score(y_true, prob_a) - roc_auc_score(y_true, prob_b)
    rng = np.random.RandomState(seed if seed is not None else RANDOM_SEED)
    n = len(y_true)
    perm_stats = []
    for _ in range(n_perm):
        perm = rng.permutation(n)
        y_perm = y_true.iloc[perm] if hasattr(y_true, 'iloc') else y_true[perm]
        try:
            auc_a = roc_auc_score(y_perm, prob_a)
            auc_b = roc_auc_score(y_perm, prob_b)
            perm_stats.append(auc_a - auc_b)
        except ValueError:
            perm_stats.append(0.0)
    perm_stats = np.array(perm_stats)
    # two-sided: count |perm| >= |obs|
    p_val = np.mean(np.abs(perm_stats) >= abs(auc_obs))
    return p_val
'''

# Insert before bootstrap_delta_auc_ci function
INS_MARKER = 'def bootstrap_delta_auc_ci('
assert INS_MARKER in c, "FIX 2: insertion point not found"
c = c.replace(INS_MARKER, PERM_FUNC + '\n' + INS_MARKER)
print("FIX 2 ✓ prediction_swap_permutation_test() added with seed")

# ═══════════════════════════════════════════════════════════════
# FIX 3: M7 – use permutation for p, bootstrap for CI only
# ═══════════════════════════════════════════════════════════════
OLD_CALL = '        delta, ci_l, ci_u, p_val = paired_bootstrap_delta_auc(y, prob_a.values, prob_b.values)'
NEW_CALL = '''        # Bootstrap for CI only (not p-value)
        delta, ci_l, ci_u = bootstrap_delta_auc_ci(y, prob_a.values, prob_b.values)
        # Permutation test for p-value (seed for reproducibility)
        p_val = prediction_swap_permutation_test(
            y, prob_a.values, prob_b.values,
            seed=RANDOM_SEED + idx
        )'''
assert OLD_CALL in c, "FIX 3: old call not found"
c = c.replace(OLD_CALL, NEW_CALL)

# Also fix the for-loop to include idx
OLD_LOOP = '    for name_a, name_b, prob_a, prob_b in comparisons:'
NEW_LOOP = '    for idx, (name_a, name_b, prob_a, prob_b) in enumerate(comparisons):'
assert OLD_LOOP in c, "FIX 3: old loop not found"
c = c.replace(OLD_LOOP, NEW_LOOP)
print("FIX 3 ✓ M7 uses permutation for p, bootstrap for CI")

# ═══════════════════════════════════════════════════════════════
# FIX 4: Delete paired_bootstrap_delta_auc() function
# ═══════════════════════════════════════════════════════════════
START = c.find('def paired_bootstrap_delta_auc(')
if START >= 0:
    rest = c[START:]
    lines = rest.split('\n')
    base_indent = len(lines[0]) - len(lines[0].lstrip())
    end = len(rest)
    for i in range(1, len(lines)):
        if lines[i].strip() and not lines[i].strip().startswith('#'):
            indent = len(lines[i]) - len(lines[i].lstrip())
            if indent <= base_indent:
                end = sum(len(l) + 1 for l in lines[:i])
                break
    c = c[:START] + '\n' + c[START + end:]
    print("FIX 4 ✓ paired_bootstrap_delta_auc() deleted")
else:
    print("FIX 4: function not found (may already be deleted)")

# ═══════════════════════════════════════════════════════════════
# FIX 6: M5 preset groups → 0-5 / 6-10 / 11+
# ═══════════════════════════════════════════════════════════════
OLD_BINS = 'bins=[-0.5, 3.5, 6.5, 10.5]'
NEW_BINS = 'bins=[-0.5, 5.5, 10.5, upper + 0.5]'
if OLD_BINS in c:
    c = c.replace(OLD_BINS, NEW_BINS)
    c = c.replace('labels=["0-3", "4-6", "7-10"]', 'labels=["0-5", "6-10", "11+"]')
    print("FIX 6 ✓ M5 preset groups changed")
else:
    print("FIX 6: bins not found (may already be fixed)")

# ═══════════════════════════════════════════════════════════════
# FIX 7: Combined trend FDR in main()
# ═══════════════════════════════════════════════════════════════
OLD_MAIN = (
    '    coverage_tbl, coverage_trend, coverage_cv = module4_coverage_gradient(dpres, labels)\n'
    '    density_tbl, density_trend, density_cv = module5_evidence_density_gradient(dcount, labels)'
)
NEW_MAIN = (
    '    coverage_tbl, coverage_trend, coverage_cv = module4_coverage_gradient(dpres, labels)\n'
    '    density_tbl, density_trend, density_cv = module5_evidence_density_gradient(dcount, labels)\n'
    '    # Combined trend FDR (Module 4/5 in same family)\n'
    '    trend_pvals = [coverage_trend["trend_p_value"], density_trend["trend_p_value"]]\n'
    '    trend_qvals = bh_fdr(np.array(trend_pvals))\n'
    '    trend_rows = [\n'
    '        {"variable": "domain_presence_sum", "test_type": "coverage_gradient",\n'
    '         "p_value": coverage_trend["trend_p_value"],\n'
    '         "q_value": trend_qvals[0],\n'
    '         "significance": sig_marker(trend_qvals[0])},\n'
    '        {"variable": "total_domain_count", "test_type": "density_gradient",\n'
    '         "p_value": density_trend["trend_p_value"],\n'
    '         "q_value": trend_qvals[1],\n'
    '         "significance": sig_marker(trend_qvals[1])},\n'
    '    ]\n'
    '    trend_fdr_df = pd.DataFrame(trend_rows)\n'
    '    trend_fdr_path = OUTDIR / "symptom_gradient_trend_fdr.csv"\n'
    '    trend_fdr_df.to_csv(trend_fdr_path, index=False, encoding="utf-8-sig")\n'
    '    print(f"  [M4/M5] Combined trend FDR saved: {trend_fdr_path}")'
)
assert OLD_MAIN in c, "FIX 7: M4/M5 call block not found"
c = c.replace(OLD_MAIN, NEW_MAIN)
print("FIX 7 ✓ combined trend FDR")

# ═══════════════════════════════════════════════════════════════
# FIX 8: M2 – add auc_directional, direction, two-sided permutation
# ═══════════════════════════════════════════════════════════════
OLD_AUC = (
    '            auc = roc_auc_score(y, prob)\n'
    '            rows.append({\n'
    '                "domain": col,\n'
    '                "domain_cn": DOMAIN_CN.get(col, ""),\n'
    '                "feature_type": feat_type,\n'
    '                "auc": auc,\n'
    '                "n_folds": n_folds,\n'
    '            })'
)
if OLD_AUC in c:
    NEW_AUC = (
        '            auc = roc_auc_score(y, prob)\n'
        '            auc_directional = max(auc, 1 - auc)\n'
        '            direction = "positive" if auc >= 0.5 else "inverse"\n'
        '            rows.append({\n'
        '                "domain": col,\n'
        '                "domain_cn": DOMAIN_CN.get(col, ""),\n'
        '                "feature_type": feat_type,\n'
        '                "auc": auc,\n'
        '                "auc_directional": auc_directional,\n'
        '                "direction": direction,\n'
        '                "n_folds": n_folds,\n'
        '            })'
    )
    c = c.replace(OLD_AUC, NEW_AUC)
    print("FIX 8a ✓ M2 added auc_directional, direction")
else:
    print("FIX 8a WARNING: M2 AUC block not found")

# Two-sided permutation
OLD_PERM = (
    '        p_perm = np.mean(np.array(perm_aucs) >= auc)\n'
    '        rows[-1]["permutation_p_value"] = p_perm'
)
if OLD_PERM in c:
    NEW_PERM = (
        '        perm_aucs_arr = np.array(perm_aucs)\n'
        '        obs_stat = abs(auc - 0.5)\n'
        '        perm_stats = abs(perm_aucs_arr - 0.5)\n'
        '        p_perm_two_sided = np.mean(perm_stats >= obs_stat)\n'
        '        rows[-1]["permutation_p_value_two_sided"] = p_perm_two_sided'
    )
    c = c.replace(OLD_PERM, NEW_PERM)
    print("FIX 8b ✓ M2 two-sided permutation")
else:
    print("FIX 8b WARNING: permutation block not found")

# ═══════════════════════════════════════════════════════════════
# FIX 10: Add fine-grained gradient functions + calls
# ═══════════════════════════════════════════════════════════════
FINE_FUNCS = r'''

# ═══════════════════════════════════════════════════════════════
# SUPPLEMENTARY: FINE-GRAINED GRADIENTS
# ═══════════════════════════════════════════════════════════════

def module4_coverage_gradient_fine(dpres, labels):
    """Supplementary fine-grained coverage gradient.
    Output: symptom_coverage_gradient_fine.csv
    Grouping: 0-2, 3-4, 5-7, 8-10
    """
    print("\n  [M4-fine] Coverage gradient (fine-grained)...")
    dp = dpres.copy()
    dp["domain_presence_sum"] = dp[DOMAIN_COLS].sum(axis=1)
    dp = dp.set_index("participant_id")
    data = pd.DataFrame({
        "label": labels.reindex(dp.index),
        "coverage": dp["domain_presence_sum"],
    }).dropna()
    data["group"] = pd.cut(
        data["coverage"],
        bins=[-0.5, 2.5, 4.5, 7.5, 10.5],
        labels=["0-2", "3-4", "5-7", "8-10"],
    )
    print("    Fine group sizes:", data.groupby("group").size().to_dict())
    rows = []
    for g, gd in data.groupby("group"):
        n = len(gd); npos = int(gd["label"].sum())
        rows.append({"group": str(g), "n": n, "n_positive": npos,
                     "positive_rate": npos/n if n>0 else float("nan")})
    tbl = pd.DataFrame(rows)
    out = OUTDIR / "symptom_coverage_gradient_fine.csv"
    tbl.to_csv(out, index=False, encoding="utf-8-sig")
    print("  [M4-fine] Saved:", out)
    return tbl


def module5_evidence_density_gradient_fine(dcount, labels):
    """Supplementary fine-grained evidence density gradient.
    Output: symptom_evidence_density_gradient_fine.csv
    Grouping: 0-2, 3-5, 6-10, 11+
    """
    print("\n  [M5-fine] Evidence density gradient (fine-grained)...")
    dc = dcount.copy()
    dc["total_domain_count"] = dc[DOMAIN_COLS].sum(axis=1)
    dc = dc.set_index("participant_id")
    data = pd.DataFrame({
        "label": labels.reindex(dc.index),
        "density": dc["total_domain_count"],
    }).dropna()
    mx = data["density"].max()
    upper = max(11, mx)
    data["group"] = pd.cut(
        data["density"],
        bins=[-0.5, 2.5, 5.5, 10.5, upper + 0.5],
        labels=["0-2", "3-5", "6-10", "11+"],
    )
    print("    Fine group sizes:", data.groupby("group").size().to_dict())
    rows = []
    for g, gd in data.groupby("group"):
        n = len(gd); npos = int(gd["label"].sum())
        rows.append({"group": str(g), "n": n, "n_positive": npos,
                     "positive_rate": npos/n if n>0 else float("nan")})
    tbl = pd.DataFrame(rows)
    out = OUTDIR / "symptom_evidence_density_gradient_fine.csv"
    tbl.to_csv(out, index=False, encoding="utf-8-sig")
    print("  [M5-fine] Saved:", out)
    return tbl

'''
# Insert before main()
MAIN_POS = c.find('\ndef main():')
assert MAIN_POS >= 0, "FIX 10: main() not found!"
c = c[:MAIN_POS] + FINE_FUNCS + '\n' + c[MAIN_POS:]
print("FIX 10a ✓ fine-grained functions added")

# Add calls in main()
OLD_CALLS = '    density_tbl, density_trend, density_cv = module5_evidence_density_gradient(dcount, labels)\n    # Combined trend FDR'
NEW_CALLS = '    density_tbl, density_trend, density_cv = module5_evidence_density_gradient(dcount, labels)\n    # Fine-grained supplementary gradients\n    module4_coverage_gradient_fine(dpres, labels)\n    module5_evidence_density_gradient_fine(dcount, labels)\n    # Combined trend FDR'
c = c.replace(OLD_CALLS, NEW_CALLS)
print("FIX 10b ✓ fine-grained calls added in main()")

# ═══════════════════════════════════════════════════════════════
# Write back
# ═══════════════════════════════════════════════════════════════
with open(FPATH, 'w', encoding='utf-8') as f:
    f.write(c)
print("\n✓ All fixes applied! File written to:")
print(FPATH)
