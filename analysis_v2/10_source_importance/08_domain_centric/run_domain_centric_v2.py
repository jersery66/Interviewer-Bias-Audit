"""
Domain-centric signal analysis for DAIC-WOZ PHQ-8 prediction (FIXED v2).

Fixes over v1:
  1. Reads from frozen input files, not superseded 01 table.
  2. Dropped M0_D0/M0_D1 (those require full C2 TF-IDF per fold; focus on D1 vs M3).
  3. Threshold constraint directions corrected.
  4. Removed pseudo repeat-level ranking; bootstrap ranking only.
  5. FDR family: removed duplicate D1↔M3 comparison.
  6. OOF predictions joined by participant_id, not row-order .values.

Core question: Can a low-dimensional clinical-domain-count representation
approach the performance of a full text+domain+template joint model?
"""

import numpy as np
import pandas as pd
from collections import OrderedDict
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, log_loss,
    f1_score, recall_score, confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent.parent

# FROZEN inputs (NOT superseded 01 table)
SPLITS_PATH       = BASE_DIR / "00_splits/repeated_5fold_splits_10x5.csv"
DOMAIN_COUNT_PATH = BASE_DIR / "04_c5_controls/domain_count/input.csv"
DOMAIN_PRES_PATH  = BASE_DIR / "04_c5_controls/domain_presence/input.csv"

# Strict OOF from Codex reanalysis
STRICT_OOF_PATH = BASE_DIR / "10_source_importance/07_strict_joint_models/participant_oof_predictions.csv"

OUTPUT_DIR = BASE_DIR / "10_source_importance/08_domain_centric"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_SEED = 20260705
BOOTSTRAP_N = 5000
PERM_N = 10000

DOMAIN_COUNT_COLS = [
    'depressed_mood', 'appetite_weight', 'suicide_self_harm',
    'functioning_impairment', 'mental_health_history',
    'anhedonia_interest', 'sleep_fatigue_energy',
    'self_worth_guilt', 'concentration_psychomotor',
    'protective_or_absent_symptom'
]
DOMAIN_PRES_COLS = DOMAIN_COUNT_COLS  # same column names in presence file

MODEL_PARAMS = dict(penalty='l2', C=1.0, class_weight='balanced',
                    solver='liblinear', max_iter=2000, random_state=BASE_SEED)

# Domain-only model specs (pure numeric, no text)
DOMAIN_MODEL_SPECS = OrderedDict({
    'D0': ['total_domain_count'],          # single float
    'P0': ['domain_presence_sum'],          # single float
    'D1': ['domain_count_10'],              # 10 floats
    'P1': ['domain_presence_10'],           # 10 floats
})

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load from frozen inputs, merge with strict OOF by participant_id."""
    # Splits
    splits = pd.read_csv(SPLITS_PATH)

    # Domain count (frozen)
    df_count = pd.read_csv(DOMAIN_COUNT_PATH)

    # Domain presence (frozen)
    df_pres = pd.read_csv(DOMAIN_PRES_PATH)

    # Strict OOF predictions
    oof = pd.read_csv(STRICT_OOF_PATH)

    # Verify frozen cohort
    assert len(df_count) == 142, f"Expected 142, got {len(df_count)}"
    assert len(df_pres) == 142
    assert len(oof) == 142

    # Build dataset from frozen inputs ONLY
    df = df_count[['participant_id']].copy()
    df = df.merge(df_pres[['participant_id']], on='participant_id', how='inner')
    df = df.merge(oof[['participant_id', 'label', 'M0_prob', 'M3_prob']],
                  on='participant_id', how='inner', validate='one_to_one')

    assert df['participant_id'].is_unique
    assert len(df) == 142
    assert df['label'].sum() == 43
    assert (df['label'] == 0).sum() == 99

    # Add total / sum features from frozen domain data
    df['total_domain_count']   = df_count[DOMAIN_COUNT_COLS].sum(axis=1).values
    df['domain_presence_sum']  = df_pres[DOMAIN_PRES_COLS].sum(axis=1).values

    # Pack domain arrays (row-major, each row is 10 floats)
    df['domain_count_10']    = list(df_count[DOMAIN_COUNT_COLS].values)
    df['domain_presence_10'] = list(df_pres[DOMAIN_PRES_COLS].values)

    print(f"Data loaded: {len(df)} participants, {df['label'].sum()} pos / {(df['label']==0).sum()} neg")
    return df, splits

# =============================================================================
# BH-FDR
# =============================================================================

def bh_fdr(p_values):
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    n = len(values)
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    return np.clip(adj, 0.0, 1.0)[np.argsort(order)]

def significance_marker(q):
    if q < 0.001: return '***'
    if q < 0.01:  return '**'
    if q < 0.05:  return '*'
    return 'ns'

# =============================================================================
# FOLD-LOCAL CV FOR DOMAIN-ONLY MODELS
# =============================================================================

def run_domain_models(df, splits):
    """Pure numeric domain models: fit LR per fold, average OOF across 10 repeats."""
    y_all = df['label'].values
    repeats = splits['repeat'].nunique()

    sum_probs = {m: np.zeros(len(df)) for m in DOMAIN_MODEL_SPECS}
    cnt_probs = np.zeros(len(df))

    for rep in range(1, repeats + 1):
        rep_data = splits[splits['repeat'] == rep]
        for fold in range(1, 6):
            fold_data = rep_data[rep_data['fold'] == fold]
            test_pids = fold_data[fold_data['role'] == 'test']['participant_id'].values
            test_mask  = df['participant_id'].isin(test_pids)
            train_mask = ~test_mask

            y_train, y_test = y_all[train_mask], y_all[test_mask]
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue

            for m, groups in DOMAIN_MODEL_SPECS.items():
                Xtr_parts, Xte_parts = [], []
                for sg in groups:
                    if sg == 'total_domain_count':
                        Xtr_parts.append(df.loc[train_mask, 'total_domain_count'].values.reshape(-1,1))
                        Xte_parts.append(df.loc[test_mask,  'total_domain_count'].values.reshape(-1,1))
                    elif sg == 'domain_presence_sum':
                        Xtr_parts.append(df.loc[train_mask, 'domain_presence_sum'].values.reshape(-1,1))
                        Xte_parts.append(df.loc[test_mask,  'domain_presence_sum'].values.reshape(-1,1))
                    elif sg == 'domain_count_10':
                        Xtr_parts.append(np.vstack(df.loc[train_mask, 'domain_count_10'].values))
                        Xte_parts.append(np.vstack(df.loc[test_mask,  'domain_count_10'].values))
                    elif sg == 'domain_presence_10':
                        Xtr_parts.append(np.vstack(df.loc[train_mask, 'domain_presence_10'].values))
                        Xte_parts.append(np.vstack(df.loc[test_mask,  'domain_presence_10'].values))

                X_train = np.hstack(Xtr_parts) if len(Xtr_parts) > 1 else Xtr_parts[0]
                X_test  = np.hstack(Xte_parts) if len(Xte_parts) > 1 else Xte_parts[0]

                scaler = StandardScaler()
                model = LogisticRegression(**MODEL_PARAMS)
                model.fit(scaler.fit_transform(X_train), y_train)
                sum_probs[m][test_mask] += model.predict_proba(scaler.transform(X_test))[:, 1]
            cnt_probs[test_mask] += 1

    # Average
    for m in DOMAIN_MODEL_SPECS:
        sum_probs[m] = np.clip(sum_probs[m] / cnt_probs, 1e-8, 1 - 1e-8)

    return sum_probs

# =============================================================================
# METRICS
# =============================================================================

def compute_all_metrics(y_true, y_prob):
    yp = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yp, labels=[0,1]).ravel()
    return {
        'auc': roc_auc_score(y_true, y_prob),
        'pr_auc': average_precision_score(y_true, y_prob),
        'brier': brier_score_loss(y_true, y_prob),
        'log_loss': log_loss(y_true, np.clip(y_prob, 1e-15, 1 - 1e-15)),
        'macro_f1': f1_score(y_true, yp, average='macro', zero_division=0),
        'sensitivity': recall_score(y_true, yp, zero_division=0),
        'specificity': tn / max(tn + fp, 1),
        'balanced_acc': (recall_score(y_true, yp, zero_division=0) + tn / max(tn + fp, 1)) / 2,
    }

# =============================================================================
# THRESHOLD ANALYSIS (FIXED)
# =============================================================================

def threshold_at_sensitivity(y_true, y_prob, target=0.80):
    """Highest threshold with sensitivity >= target."""
    ts = np.unique(np.r_[0.0, np.sort(y_prob), 1.0])
    valid = [t for t in ts if recall_score(y_true, (y_prob >= t).astype(int), zero_division=0) >= target]
    return max(valid) if valid else np.nan

def threshold_at_specificity(y_true, y_prob, target=0.80):
    """Lowest threshold with specificity >= target."""
    ts = np.unique(np.r_[0.0, np.sort(y_prob), 1.0])
    valid = []
    for t in ts:
        yp = (y_prob >= t).astype(int)
        tn = ((y_true == 0) & (yp == 0)).sum()
        fp = ((y_true == 0) & (yp == 1)).sum()
        if tn / max(tn + fp, 1) >= target:
            valid.append(t)
    return min(valid) if valid else np.nan

def threshold_analysis_fixed(y_true, y_prob):
    results = {}
    ts_grid = np.linspace(0.05, 0.95, 91)

    # Default 0.50
    t = 0.50
    yp = (y_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yp, labels=[0,1]).ravel()
    results['default_0.50'] = {'threshold': t, 'sens': tp/(tp+fn) if tp+fn else 0,
                                'spec': tn/(tn+fp) if tn+fp else 0,
                                'macro_f1': f1_score(y_true, yp, average='macro', zero_division=0)}

    # Youden
    t = ts_grid[np.argmax([recall_score(y_true, (y_prob >= x).astype(int), zero_division=0) +
                           ((y_true==0)&((y_prob>=x).astype(int)==0)).sum()/((y_true==0).sum() or 1) - 1
                           for x in ts_grid])]
    yp = (y_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yp, labels=[0,1]).ravel()
    results['youden'] = {'threshold': t, 'sens': tp/(tp+fn) if tp+fn else 0,
                         'spec': tn/(tn+fp) if tn+fp else 0,
                         'macro_f1': f1_score(y_true, yp, average='macro', zero_division=0)}

    # Best Macro-F1
    t = ts_grid[np.argmax([f1_score(y_true, (y_prob>=x).astype(int), average='macro', zero_division=0) for x in ts_grid])]
    yp = (y_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yp, labels=[0,1]).ravel()
    results['best_macro_f1'] = {'threshold': t, 'sens': tp/(tp+fn) if tp+fn else 0,
                                 'spec': tn/(tn+fp) if tn+fp else 0,
                                 'macro_f1': f1_score(y_true, yp, average='macro', zero_division=0)}

    # Constrained thresholds (FIXED directions)
    for name, fn in [('sens>=0.80', threshold_at_sensitivity), ('spec>=0.80', threshold_at_specificity)]:
        t = fn(y_true, y_prob)
        yp = (y_prob >= t).astype(int) if not np.isnan(t) else np.zeros_like(y_true)
        tn, fp, fn, tp = confusion_matrix(y_true, yp, labels=[0,1]).ravel() if not np.isnan(t) else (0,0,0,0)
        results[name] = {'threshold': t if not np.isnan(t) else np.nan,
                         'sens': tp/(tp+fn) if tp+fn else np.nan,
                         'spec': tn/(tn+fp) if tn+fp else np.nan,
                         'macro_f1': f1_score(y_true, yp, average='macro', zero_division=0) if not np.isnan(t) else np.nan}

    return results

# =============================================================================
# BOOTSTRAP & PERMUTATION
# =============================================================================

def bootstrap_ci(y_true, probs_a, probs_b=None, n_boot=BOOTSTRAP_N):
    n = len(y_true)
    rng = np.random.default_rng(BASE_SEED)
    deltas = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        try:
            deltas.append(roc_auc_score(y_true[idx], probs_a[idx]) -
                         (roc_auc_score(y_true[idx], probs_b[idx]) if probs_b is not None else 0))
        except:
            pass
    if not deltas:
        return np.nan, np.nan, np.nan
    return np.mean(deltas), np.percentile(deltas, 2.5), np.percentile(deltas, 97.5)

def paired_permutation_test(y_true, probs_a, probs_b, n_perm=PERM_N):
    rng = np.random.default_rng(BASE_SEED)
    obs = roc_auc_score(y_true, probs_a) - roc_auc_score(y_true, probs_b)
    count = 0
    for _ in range(n_perm):
        swap = rng.random(len(y_true)) < 0.5
        try:
            perm = roc_auc_score(y_true, np.where(swap, probs_b, probs_a)) - \
                   roc_auc_score(y_true, np.where(swap, probs_a, probs_b))
        except:
            continue
        if abs(perm) >= abs(obs):
            count += 1
    return (count + 1) / (n_perm + 1)

def bootstrap_ranking(y_true, oof_all, n_boot=BOOTSTRAP_N):
    n = len(y_true)
    rng = np.random.default_rng(BASE_SEED)
    models = list(oof_all.keys())
    top1 = {m: 0 for m in models}
    top2 = {m: 0 for m in models}
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        aucs = {}
        for m in models:
            try:
                aucs[m] = roc_auc_score(y_true[idx], oof_all[m][idx])
            except:
                aucs[m] = np.nan
        ranked = sorted(aucs, key=aucs.get, reverse=True)
        top1[ranked[0]] += 1
        for m in ranked[:2]:
            top2[m] += 1
    return {m: {'top1': top1[m]/n_boot, 'top2': top2[m]/n_boot} for m in models}

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("DOMAIN-CENTRIC ANALYSIS v2 (FIXED)")
    print("=" * 60)

    df, splits = load_data()

    # Run domain-only models
    print("\n[1/4] Running domain-only models (fold-local CV)...")
    oof_domain = run_domain_models(df, splits)

    # Combine: domain models + strict M0/M3
    oof_all = {}
    # Strict models from Codex
    oof_all['M0'] = df['M0_prob'].values
    oof_all['M3'] = df['M3_prob'].values
    # Domain models
    for m, probs in oof_domain.items():
        oof_all[m] = probs

    y_true = df['label'].values

    # Metrics
    print("\n[2/4] Computing metrics...")
    focus = ['M0', 'D0', 'P0', 'D1', 'P1', 'M3']
    metrics = {}
    for m in focus:
        metrics[m] = compute_all_metrics(y_true, oof_all[m])
        print(f"  {m}: AUC={metrics[m]['auc']:.4f}, Brier={metrics[m]['brier']:.4f}, Sens={metrics[m]['sensitivity']:.4f}")

    # Save metrics
    pd.DataFrame(metrics).T.to_csv(OUTPUT_DIR / 'domain_centric_metrics.csv')
    print(f"  → domain_centric_metrics.csv")

    # Threshold analysis (FIXED)
    print("\n[3/4] Threshold analysis (fixed constraints)...")
    thr_rows = []
    for m in ['M0', 'D1', 'M3']:
        thr = threshold_analysis_fixed(y_true, oof_all[m])
        for tname, tvals in thr.items():
            thr_rows.append({'model': m, 'type': tname, **tvals})
    pd.DataFrame(thr_rows).to_csv(OUTPUT_DIR / 'threshold_analysis.csv', index=False)
    print(f"  → threshold_analysis.csv")

    # Comparisons (FIXED FDR family - no duplicate)
    print("\n[4/4] Bootstrap CI + permutation + FDR...")
    comps = [
        ('D1', 'M0', 'D1 vs M0: domain count vs patient text'),
        ('D1', 'M3', 'D1 vs M3: domain count vs full joint'),
        ('D1', 'D0', 'D1 vs D0: 10 counts vs total count'),
        ('D1', 'P1', 'D1 vs P1: count vs presence'),
        ('D0', 'P0', 'D0 vs P0: total count vs presence sum'),
    ]
    ci_rows = []
    for ma, mb, desc in comps:
        p = paired_permutation_test(y_true, oof_all[ma], oof_all[mb])
        mean_d, ci_l, ci_u = bootstrap_ci(y_true, oof_all[ma], oof_all[mb])
        ci_rows.append({'comparison': f'{ma} vs {mb}', 'description': desc,
                       'delta_auc': mean_d, 'ci_lower': ci_l, 'ci_upper': ci_u, 'perm_p': p})

    ci_df = pd.DataFrame(ci_rows)
    ci_df['q_value'] = bh_fdr(ci_df['perm_p'].values)
    ci_df['significance'] = [significance_marker(q) for q in ci_df['q_value']]
    ci_df.to_csv(OUTPUT_DIR / 'domain_key_comparisons_fdr.csv', index=False)
    print(f"  → domain_key_comparisons_fdr.csv")

    # Stability: bootstrap ranking only
    rank = bootstrap_ranking(y_true, {m: oof_all[m] for m in focus})
    rank_rows = [{'model': m, 'top1_pct': rank[m]['top1'], 'top2_pct': rank[m]['top2']} for m in focus]
    pd.DataFrame(rank_rows).to_csv(OUTPUT_DIR / 'stability_analysis.csv', index=False)
    print(f"  → stability_analysis.csv")

    # Summary
    print(f"\n{'='*60}")
    print(f"KEY RESULTS")
    print(f"{'='*60}")
    print(f"D1 (10 domain_count)  AUC = {metrics['D1']['auc']:.4f}")
    print(f"M3 (full joint)       AUC = {metrics['M3']['auc']:.4f}")
    print(f"D1 vs M3 ΔAUC = {ci_df.loc[ci_df['comparison']=='D1 vs M3','delta_auc'].values[0]:+.4f}")
    print(f"D1 vs M3 q = {ci_df.loc[ci_df['comparison']=='D1 vs M3','q_value'].values[0]:.4f} ({ci_df.loc[ci_df['comparison']=='D1 vs M3','significance'].values[0]})")
    print(f"\nBootstrap ranking top-1: M3={rank['M3']['top1']:.1%}, D1={rank['D1']['top1']:.1%}")
    print(f"Bootstrap ranking top-2: M3={rank['M3']['top2']:.1%}, D1={rank['D1']['top2']:.1%}")
    print(f"\nOutput: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
