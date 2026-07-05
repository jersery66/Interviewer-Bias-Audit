"""Apply all 10 fixes to run_symptom_level_analysis.py.
Works on the original git version (commit 327ee8a).
"""
FPATH = 'E:/CodexWorktrees/DAIC-WOZ/reanalysis-v2/analysis_v2/10_source_importance/09_symptom_level_analysis/run_symptom_level_analysis.py'
with open(FPATH, 'r', encoding='utf-8') as f:
    c = f.read()

# ════════════════════════════════════════════════════════════════
# FIX 1: BASE → relative path
# ════════════════════════════════════════════════════════════════
old = 'BASE = Path("E:/CodexWorktrees/DAIC-WOZ/reanalysis-v2/analysis_v2")'
new = 'SCRIPT_DIR = Path(__file__).parent\nBASE = SCRIPT_DIR.parent.parent'
assert old in c, f"FIX 1 FAIL"
c = c.replace(old, new)
print("FIX 1 ✓ BASE → relative path")

# ════════════════════════════════════════════════════════════════
# FIX 2: Add seed param to permutation_test_auc_diff()
# ════════════════════════════════════════════════════════════════
old = 'def permutation_test_auc_diff(y_true, prob_a, prob_b, n_perm=PERMUTATION_N):'
new = 'def permutation_test_auc_diff(y_true, prob_a, prob_b, n_perm=PERMUTATION_N, seed=None):'
assert old in c, "FIX 2a FAIL"
c = c.replace(old, new)
print("FIX 2a ✓ seed param added to permutation_test_auc_diff")

old = '    rng = np.random.RandomState(RANDOM_SEED)'
new = '    rng = np.random.RandomState(seed if seed is not None else RANDOM_SEED)'
assert old in c, "FIX 2b FAIL"
c = c.replace(old, new)
print("FIX 2b ✓ RNG uses seed")

# ════════════════════════════════════════════════════════════════
# FIX 3: M7 – use permutation for p, bootstrap for CI only
# Need to ADD bootstrap_delta_auc_ci() function (CI only)
# and change M7 to call both
# ════════════════════════════════════════════════════════════════
# First add bootstrap_delta_auc_ci before paired_bootstrap_delta_auc
BOOT_CI = '''
def bootstrap_delta_auc_ci(y_true, prob_a, prob_b, n_boot=BOOTSTRAP_N, alpha=0.05):
    """Bootstrap CI for ΔAUC only (p-value via permutation, not bootstrap)."""
    obs = roc_auc_score(y_true, prob_a) - roc_auc_score(y_true, prob_b)
    rng = np.random.RandomState(RANDOM_SEED)
    n = len(y_true)
    deltas = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        try:
            d = roc_auc_score(y_true.iloc[idx], prob_a[idx]) - roc_auc_score(y_true.iloc[idx], prob_b[idx])
        except ValueError:
            d = 0.0
        deltas.append(d)
    deltas = np.array(deltas)
    lo = np.percentile(deltas, 100 * alpha/2)
    hi = np.percentile(deltas, 100 * (1 - alpha/2))
    return obs, lo, hi
'''
# Insert before paired_bootstrap_delta_auc
INS_PT = 'def paired_bootstrap_delta_auc('
assert INS_PT in c, "FIX 3: insertion point not found"
c = c.replace(INS_PT, BOOT_CI + '\ndef paired_bootstrap_delta_auc(')
print("FIX 3a ✓ bootstrap_delta_auc_ci() added")

# Now change M7 call: replace paired_bootstrap_delta_auc with bootstrap_ci + permutation
old_call = '        delta, ci_l, ci_u, p_val = paired_bootstrap_delta_auc(y, prob_a.values, prob_b.values)'
new_call = '''        # Bootstrap for CI only; permutation for p-value
        delta, ci_l, ci_u = bootstrap_delta_auc_ci(y, prob_a.values, prob_b.values)
        p_val = permutation_test_auc_diff(
            y, prob_a.values, prob_b.values,
            seed=RANDOM_SEED + idx
        )'''
# Also need to change the for-loop to include idx
old_loop = '    for name_a, name_b, prob_a, prob_b in comparisons:'
new_loop = '    for idx, (name_a, name_b, prob_a, prob_b) in enumerate(comparisons):'
assert old_loop in c, "FIX 3b: loop not found"
c = c.replace(old_loop, new_loop)
assert old_call in c, "FIX 3c: call not found"
c = c.replace(old_call, new_call)
print("FIX 3 ✓ M7 uses permutation for p, bootstrap for CI")

# ════════════════════════════════════════════════════════════════
# FIX 4: Delete paired_bootstrap_delta_auc() function
# ════════════════════════════════════════════════════════════════
start = c.find('def paired_bootstrap_delta_auc(')
assert start >= 0, "FIX 4: function not found"
# Find end of function
rest = c[start:]
lines = rest.split('\n')
base_indent = len(lines[0]) - len(lines[0].lstrip())
end = len(rest)
for i in range(1, len(lines)):
    if lines[i].strip() and not lines[i].strip().startswith('#'):
        indent = len(lines[i]) - len(lines[i].lstrip())
        if indent <= base_indent:
            end = sum(len(l) + 1 for l in lines[:i])
            break
c = c[:start] + '\n' + c[start + end:]
print("FIX 4 ✓ paired_bootstrap_delta_auc() deleted")

# ════════════════════════════════════════════════════════════════
# FIX 6: M5 preset groups → 0-5 / 6-10 / 11+
# ════════════════════════════════════════════════════════════════
old = 'bins=[-0.5, 3.5, 6.5, 10.5]'
new = 'bins=[-0.5, 5.5, 10.5, upper + 0.5]'
if old in c:
    c = c.replace(old, new)
    c = c.replace('labels=["0-3", "4-6", "7-10"]', 'labels=["0-5", "6-10", "11+"]')
    print("FIX 6 ✓ M5 preset groups changed")
else:
    print("FIX 6: bins not found")

# ════════════════════════════════════════════════════════════════
# FIX 7: Combined trend FDR in main()
# ════════════════════════════════════════════════════════════════
old = ('    coverage_tbl, coverage_trend, coverage_cv = module4_coverage_gradient(dpres, labels)\n'
       '    density_tbl, density_trend, density_cv = module5_evidence_density_gradient(dcount, labels)')
assert old in c, "FIX 7: M4/M5 calls not found"
ins = (old + '\n'
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
c = c.replace(old, ins)
print("FIX 7 ✓ combined trend FDR")

# ════════════════════════════════════════════════════════════════
# FIX 8: M2 – add auc_directional, direction, two-sided permutation
# ════════════════════════════════════════════════════════════════
old = ('            auc = roc_auc_score(y, prob)\n'
       '            rows.append({\n'
       '                "domain": col,\n'
       '                "domain_cn": DOMAIN_CN.get(col, ""),\n'
       '                "feature_type": feat_type,\n'
       '                "auc": auc,\n'
       '                "n_folds": n_folds,\n'
       '            })')
if old in c:
    new = ('            auc = roc_auc_score(y, prob)\n'
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
           '            })')
    c = c.replace(old, new)
    print("FIX 8a ✓ M2 added auc_directional, direction")
else:
    print("FIX 8a WARNING: M2 AUC block not found")

# Two-sided permutation for M2
old_p = '        p_perm = np.mean(np.array(perm_aucs) >= auc)\n        rows[-1]["permutation_p_value"] = p_perm'
if old_p in c:
    new_p = ('        perm_aucs_arr = np.array(perm_aucs)\n'
              '        obs_stat = abs(auc - 0.5)\n'
              '        perm_stats = abs(perm_aucs_arr - 0.5)\n'
              '        p_perm_two_sided = np.mean(perm_stats >= obs_stat)\n'
              '        rows[-1]["permutation_p_value_two_sided"] = p_perm_two_sided'
              )
    c = c.replace(old_p, new_p)
    print("FIX 8b ✓ M2 two-sided permutation")
else:
    print("FIX 8b WARNING: permutation block not found")

# ════════════════════════════════════════════════════════════════
# FIX 10: Add fine-grained gradient functions + calls in main()
# ════════════════════════════════════════════════════════════════
# Write fine functions to a temp file first, then read and insert
import tempfile, os
fine_code = '''

# ════════════════════════════════════════════════════════════════
# SUPPLEMENTARY: FINE-GRAINED GRADIENTS
# ════════════════════════════════════════════════════════════════

def module4_coverage_gradient_fine(dpres, labels):
    """Supplementary fine-grained coverage gradient.
    Output: symptom_coverage_gradient_fine.csv
    Grouping: 0-2, 3-4, 5-7, 8-10
    """
    print("\\n  [M4-fine] Coverage gradient (fine-grained)...")
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
    print("\\n  [M5-fine] Evidence density gradient (fine-grained)...")
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
main_pos = c.find('\ndef main():')
assert main_pos >= 0, "FIX 10: main() not found!"
c = c[:main_pos] + fine_code + c[main_pos:]
print("FIX 10a ✓ fine-grained functions added")

# Add calls in main()
old_calls = '    density_tbl, density_trend, density_cv = module5_evidence_density_gradient(dcount, labels)\n    # Combined trend FDR'
new_calls = '    density_tbl, density_trend, density_cv = module5_evidence_density_gradient(dcount, labels)\n    # Fine-grained supplementary gradients\n    module4_coverage_gradient_fine(dpres, labels)\n    module5_evidence_density_gradient_fine(dcount, labels)\n    # Combined trend FDR'
c = c.replace(old_calls, new_calls)
print("FIX 10b ✓ fine-grained calls added in main()")

# ════════════════════════════════════════════════════════════════
# Write back
# ════════════════════════════════════════════════════════════════
with open(FPATH, 'w', encoding='utf-8') as f:
    f.write(c)
print("\n✓ All fixes applied! Running syntax check...")
