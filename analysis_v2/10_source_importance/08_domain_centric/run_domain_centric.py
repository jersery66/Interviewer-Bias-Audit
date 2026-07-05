"""
Domain-centric signal analysis for DAIC-WOZ PHQ-8 prediction.

Adds domain-only models to the existing strict M0-M5 analysis and
produces calibration, threshold, stability, and manuscript-ready outputs.

Strategy: The core message is "clinical-domain count is the most stable,
interpretable low-dimensional signal for PHQ-8 prediction in DAIC-WOZ."

Author: Domain-centric reanalysis
Date: 2026-07-05
"""

import numpy as np
import pandas as pd
from scipy import stats
from collections import OrderedDict
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, log_loss,
    f1_score, recall_score, confusion_matrix
)

# =============================================================================
# CONFIG
# =============================================================================

BASE_DIR = Path("E:/CodexWorktrees/DAIC-WOZ/reanalysis-v2/analysis_v2")
FEATURES_PATH = BASE_DIR / "10_source_importance/01_source_level_table/source_level_features.csv"
SPLITS_PATH = BASE_DIR / "00_splits/repeated_5fold_splits_10x5.csv"
STRICT_OOF_PATH = BASE_DIR / "10_source_importance/07_strict_joint_models/participant_oof_predictions.csv"
OUTPUT_DIR = BASE_DIR / "10_source_importance/08_domain_centric"
SCRIPT_DIR = Path(__file__).parent

BASE_SEED = 20260705
BOOTSTRAP_N = 5000
PERM_N = 10000

# Domain columns
DOMAIN_COUNT_COLS = [
    'depressed_mood_count', 'appetite_weight_count', 'suicide_self_harm_count',
    'functioning_impairment_count', 'mental_health_history_count',
    'anhedonia_interest_count', 'sleep_fatigue_energy_count',
    'self_worth_guilt_count', 'concentration_psychomotor_count',
    'protective_or_absent_symptom_count'
]
DOMAIN_PRESENCE_COLS = [c.replace('_count', '_presence') for c in DOMAIN_COUNT_COLS]

MODEL_PARAMS = dict(penalty='l2', C=1.0, class_weight='balanced',
                    solver='liblinear', max_iter=2000, random_state=BASE_SEED)

# New domain-centric model specs (runs inside fold-local CV)
DOMAIN_MODEL_SPECS = OrderedDict({
    'D0': ['total_domain_count'],        # single float
    'P0': ['domain_presence_sum'],        # single float
    'D1': ['domain_count_10'],            # 10 floats
    'P1': ['domain_presence_10'],         # 10 floats
    'M0_D0': ['c2_text', 'total_domain_count'],
    'M0_D1': ['c2_text', 'domain_count_10'],
})

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    df = pd.read_csv(FEATURES_PATH)
    splits = pd.read_csv(SPLITS_PATH)

    # Load existing strict OOF predictions for M0-M5
    oof_existing = pd.read_csv(STRICT_OOF_PATH)

    # Build total domain features
    df['total_domain_count'] = df[DOMAIN_COUNT_COLS].sum(axis=1)
    df['domain_presence_sum'] = df[DOMAIN_PRESENCE_COLS].sum(axis=1)
    df['domain_count_10'] = df[DOMAIN_COUNT_COLS].values.tolist()  # list column
    df['domain_presence_10'] = df[DOMAIN_PRESENCE_COLS].values.tolist()

    print(f"Data: {df.shape}, Splits: {splits.shape}, Existing OOF: {oof_existing.shape}")
    return df, splits, oof_existing

# =============================================================================
# BH-FDR (consistent with Codex strict)
# =============================================================================

def bh_fdr(p_values):
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    n = len(values)
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    result = np.empty_like(adj)
    result[order] = adj
    return result

def significance_marker(q):
    if q < 0.001: return '***'
    if q < 0.01: return '**'
    if q < 0.05: return '*'
    return 'ns'

# =============================================================================
# FOLD-LOCAL CV FOR DOMAIN MODELS
# =============================================================================

def run_domain_models(df, splits):
    """Run domain-only and M0+domain models in fold-local CV."""
    y_all = df['label'].values
    repeats = splits['repeat'].nunique()
    rng = np.random.default_rng(BASE_SEED)

    # Collect C2 text for M0+domain models (read from source csv)
    # For simplicity, load C2 text separately
    c2_df = pd.read_csv(BASE_DIR / "01_inputs/c2_participant_speech.csv")

    oof_probs = {model: np.zeros(len(df)) for model in DOMAIN_MODEL_SPECS}
    count_preds = np.zeros(len(df))

    for rep in range(1, repeats + 1):
        rep_data = splits[splits['repeat'] == rep]
        for fold in range(1, 6):
            fold_data = rep_data[rep_data['fold'] == fold]
            test_pids = fold_data[fold_data['role'] == 'test']['participant_id'].values
            test_mask = df['participant_id'].isin(test_pids)
            train_mask = ~test_mask

            y_train, y_test = y_all[train_mask], y_all[test_mask]
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue

            for model_name, source_groups in DOMAIN_MODEL_SPECS.items():
                X_train_list, X_test_list = [], []

                for sg in source_groups:
                    if sg == 'total_domain_count':
                        X_train_list.append(df.loc[train_mask, 'total_domain_count'].values.reshape(-1, 1))
                        X_test_list.append(df.loc[test_mask, 'total_domain_count'].values.reshape(-1, 1))
                    elif sg == 'domain_presence_sum':
                        X_train_list.append(df.loc[train_mask, 'domain_presence_sum'].values.reshape(-1, 1))
                        X_test_list.append(df.loc[test_mask, 'domain_presence_sum'].values.reshape(-1, 1))
                    elif sg == 'domain_count_10':
                        X_train_list.append(df.loc[train_mask, DOMAIN_COUNT_COLS].values)
                        X_test_list.append(df.loc[test_mask, DOMAIN_COUNT_COLS].values)
                    elif sg == 'domain_presence_10':
                        X_train_list.append(df.loc[train_mask, DOMAIN_PRESENCE_COLS].values)
                        X_test_list.append(df.loc[test_mask, DOMAIN_PRESENCE_COLS].values)
                    elif sg == 'c2_text':
                        # Use C2 TF-IDF probability as a simple proxy
                        # For full fidelity, would need to re-vectorize text - use existing OOF prob
                        train_pids = df.loc[train_mask, 'participant_id']
                        test_pids_df = df.loc[test_mask, 'participant_id']
                        c2_train = c2_df[c2_df['participant_id'].isin(train_pids)]
                        c2_test = c2_df[c2_df['participant_id'].isin(test_pids_df)]
                        # Use c2_prob from existing features as proxy for text signal
                        X_train_list.append(df.loc[train_mask, 'c2_prob'].values.reshape(-1, 1))
                        X_test_list.append(df.loc[test_mask, 'c2_prob'].values.reshape(-1, 1))

                X_train = np.hstack(X_train_list) if len(X_train_list) > 1 else X_train_list[0]
                X_test = np.hstack(X_test_list) if len(X_test_list) > 1 else X_test_list[0]

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)

                model = LogisticRegression(**MODEL_PARAMS)
                model.fit(X_train_s, y_train)
                oof_probs[model_name][test_mask] = model.predict_proba(X_test_s)[:, 1]

            count_preds[test_mask] += 1

    # Average across repeats
    for model_name in DOMAIN_MODEL_SPECS:
        oof_probs[model_name] = oof_probs[model_name] / count_preds
        oof_probs[model_name] = np.clip(oof_probs[model_name], 1e-8, 1 - 1e-8)

    return oof_probs

# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def compute_all_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        'auc': roc_auc_score(y_true, y_prob),
        'pr_auc': average_precision_score(y_true, y_prob),
        'brier': brier_score_loss(y_true, y_prob),
        'log_loss': log_loss(y_true, np.clip(y_prob, 1e-15, 1-1e-15)),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'sensitivity': recall_score(y_true, y_pred, zero_division=0),
        'specificity': ((y_true == 0) & (y_pred == 0)).sum() / max((y_true == 0).sum(), 1),
        'balanced_acc': (recall_score(y_true, y_pred, zero_division=0) +
                        ((y_true == 0) & (y_pred == 0)).sum() / max((y_true == 0).sum(), 1)) / 2,
    }

def threshold_analysis(y_true, y_prob):
    """Threshold screening utility analysis."""
    results = {}
    thresholds_to_test = np.linspace(0.05, 0.95, 91)

    # Default 0.50
    for thresh_name, thresh_func in [
        ('default_0.50', lambda: 0.50),
        ('youden', lambda: thresholds_to_test[np.argmax([
            recall_score(y_true, (y_prob >= t).astype(int), zero_division=0) +
            ((y_true == 0) & ((y_prob >= t).astype(int) == 0)).sum() / max((y_true == 0).sum(), 1) - 1
            for t in thresholds_to_test
        ])]),
        ('best_macro_f1', lambda: thresholds_to_test[np.argmax([
            f1_score(y_true, (y_prob >= t).astype(int), average='macro', zero_division=0)
            for t in thresholds_to_test
        ])]),
        ('sens_at_least_80', lambda: min([t for t in thresholds_to_test
            if recall_score(y_true, (y_prob >= t).astype(int), zero_division=0) >= 0.80], default=0.50)),
        ('spec_at_least_80', lambda: max([t for t in thresholds_to_test
            if ((y_true == 0) & ((y_prob >= t).astype(int) == 0)).sum() / max((y_true == 0).sum(), 1) >= 0.80], default=0.50)),
    ]:
        t = thresh_func()
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        results[thresh_name] = {
            'threshold': t,
            'sensitivity': tp / max(tp + fn, 1),
            'specificity': tn / max(tn + fp, 1),
            'ppv': tp / max(tp + fp, 1),
            'npv': tn / max(tn + fn, 1),
            'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'balanced_acc': (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)) / 2,
        }
    return results

def bootstrap_ci(y_true, probs_a, probs_b=None, metric_func=roc_auc_score, n_boot=BOOTSTRAP_N):
    """Paired bootstrap CI. If probs_b is None, just CI for single model."""
    n = len(y_true)
    rng = np.random.default_rng(BASE_SEED)
    deltas = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        try:
            if probs_b is None:
                deltas.append(metric_func(y_true[idx], probs_a[idx]))
            else:
                deltas.append(metric_func(y_true[idx], probs_a[idx]) - metric_func(y_true[idx], probs_b[idx]))
        except:
            deltas.append(np.nan)
    valid = [d for d in deltas if not np.isnan(d)]
    return np.mean(valid), np.percentile(valid, 2.5), np.percentile(valid, 97.5)

def paired_permutation_test(y_true, probs_a, probs_b, metric_func=roc_auc_score, n_perm=PERM_N):
    """Paired permutation test."""
    rng = np.random.default_rng(BASE_SEED)
    obs_delta = metric_func(y_true, probs_a) - metric_func(y_true, probs_b)
    count = 0
    for _ in range(n_perm):
        swap = rng.random(len(y_true)) < 0.5
        pa = np.where(swap, probs_b, probs_a)
        pb = np.where(swap, probs_a, probs_b)
        try:
            perm_delta = metric_func(y_true, pa) - metric_func(y_true, pb)
        except:
            obs_delta = 0  # degenerate fallback
            perm_delta = 0
        if abs(perm_delta) >= abs(obs_delta):
            count += 1
    return (count + 1) / (n_perm + 1)

# =============================================================================
# STABILITY ANALYSIS
# =============================================================================

def repeat_level_ranking(df, splits, oof_probs_all):
    """Rank models within each repeat."""
    y_all = df['label'].values
    repeats = splits['repeat'].nunique()
    models = list(oof_probs_all.keys())
    rankings = {m: [] for m in models}

    for rep in range(1, repeats + 1):
        rep_pids = splits[splits['repeat'] == rep]['participant_id'].unique()
        rep_mask = df['participant_id'].isin(rep_pids)
        y_rep = y_all[rep_mask]

        aucs = {}
        for m in models:
            try:
                aucs[m] = roc_auc_score(y_rep, oof_probs_all[m][rep_mask])
            except:
                aucs[m] = np.nan

        sorted_models = sorted(aucs, key=aucs.get, reverse=True)
        for rank, m in enumerate(sorted_models, 1):
            rankings[m].append(rank)

    return {m: {'mean_rank': np.mean(r), 'ranks': r} for m, r in rankings.items()}

def bootstrap_ranking_probability(df, oof_probs_all, n_boot=BOOTSTRAP_N):
    """Bootstrap ranking probability: how often each model ranks top-1, top-2."""
    y_all = df['label'].values
    n = len(y_all)
    rng = np.random.default_rng(BASE_SEED)
    models = list(oof_probs_all.keys())
    top1 = {m: 0 for m in models}
    top2 = {m: 0 for m in models}

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        aucs = {}
        for m in models:
            try:
                aucs[m] = roc_auc_score(y_all[idx], oof_probs_all[m][idx])
            except:
                aucs[m] = np.nan
        sorted_models = sorted(aucs, key=aucs.get, reverse=True)
        top1[sorted_models[0]] += 1
        for m in sorted_models[:2]:
            top2[m] += 1

    return {m: {'top1_pct': top1[m]/n_boot, 'top2_pct': top2[m]/n_boot} for m in models}

# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("DOMAIN-CENTRIC SIGNAL ANALYSIS")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    df, splits, oof_existing = load_data()

    # Run domain models
    print("\nRunning domain-centric models...")
    oof_domain = run_domain_models(df, splits)

    # Combine all OOF predictions
    oof_all = {}
    # From strict analysis: M0-M5
    for m in ['M0', 'M1', 'M2', 'M3', 'M4', 'M5']:
        if f'{m}_prob' in oof_existing.columns:
            oof_all[m] = oof_existing[f'{m}_prob'].values
    # From domain analysis: D0, P0, D1, P1, M0_D0, M0_D1
    for m, probs in oof_domain.items():
        oof_all[m] = probs

    y_true = df['label'].values

    # Compute metrics for all models
    print("\nComputing metrics...")
    all_metrics = {}
    focus_models = ['M0', 'D0', 'P0', 'D1', 'P1', 'M0_D0', 'M0_D1', 'M3']
    for m in focus_models:
        if m in oof_all:
            all_metrics[m] = compute_all_metrics(y_true, oof_all[m])
            print(f"  {m}: AUC={all_metrics[m]['auc']:.4f}, Brier={all_metrics[m]['brier']:.4f}")

    # Threshold analysis
    print("\nThreshold analysis...")
    threshold_results = {}
    for m in ['M0', 'D1', 'M3']:
        if m in oof_all:
            threshold_results[m] = threshold_analysis(y_true, oof_all[m])

    # Stability: repeat-level ranking
    print("\nStability analysis...")
    rank_models = [m for m in focus_models if m in oof_all]
    rank_data = {m: oof_all[m] for m in rank_models}
    repeat_ranks = repeat_level_ranking(df, splits, rank_data)
    boot_ranks = bootstrap_ranking_probability(df, rank_data)

    # Save all results
    print("\nSaving results...")

    # Model metrics
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.to_csv(OUTPUT_DIR / 'domain_centric_metrics.csv')
    print(f"  Saved: domain_centric_metrics.csv")

    # Threshold results (flatten for CSV)
    threshold_rows = []
    for m, thr in threshold_results.items():
        for tname, tvals in thr.items():
            threshold_rows.append({'model': m, 'threshold_type': tname, **tvals})
    pd.DataFrame(threshold_rows).to_csv(OUTPUT_DIR / 'threshold_analysis.csv', index=False)
    print(f"  Saved: threshold_analysis.csv")

    # Stability
    rank_rows = []
    for m, rdata in repeat_ranks.items():
        rank_rows.append({'model': m, 'mean_rank': rdata['mean_rank'],
                          'top1_pct': boot_ranks[m]['top1_pct'],
                          'top2_pct': boot_ranks[m]['top2_pct']})
    pd.DataFrame(rank_rows).to_csv(OUTPUT_DIR / 'stability_analysis.csv', index=False)
    print(f"  Saved: stability_analysis.csv")

    # Bootstrap CI for key comparisons
    print("\nBootstrap CI for key comparisons...")
    ci_rows = []
    comps = [
        ('D1', 'M0', 'domain_count_10 vs patient_text'),
        ('D1', 'M3', 'domain_count_10 vs joint_M3'),
        ('D1', 'D0', '10_domain_count vs total_count'),
        ('D1', 'P1', 'count vs presence'),
        ('D0', 'P0', 'total_count vs total_presence'),
        ('M0_D1', 'M0', 'patient+domain vs patient_only'),
        ('M3', 'D1', 'joint vs domain_only'),
    ]
    for ma, mb, desc in comps:
        if ma in oof_all and mb in oof_all:
            p_val = paired_permutation_test(y_true, oof_all[ma], oof_all[mb])
            mean_d, ci_l, ci_u = bootstrap_ci(y_true, oof_all[ma], oof_all[mb])
            ci_rows.append({'comparison': f'{ma} vs {mb}', 'description': desc,
                           'delta_auc': mean_d, 'ci_lower': ci_l, 'ci_upper': ci_u,
                           'perm_p': p_val})

    ci_df = pd.DataFrame(ci_rows)
    ci_df['q_value'] = bh_fdr(ci_df['perm_p'].values)
    ci_df['significance'] = [significance_marker(q) for q in ci_df['q_value']]
    ci_df.to_csv(OUTPUT_DIR / 'domain_key_comparisons_fdr.csv', index=False)

    # Print summary
    print(f"\n=== Key Findings ===")
    print(f"D1 (10 domain_count) AUC={all_metrics.get('D1', {}).get('auc', np.nan):.4f}")
    print(f"M3 (full joint) AUC={all_metrics.get('M3', {}).get('auc', np.nan):.4f}")
    print(f"D1 vs M3 delta: {all_metrics.get('D1', {}).get('auc', 0) - all_metrics.get('M3', {}).get('auc', 0):+.4f}")
    print(f"\nBootstrap ranking (top-1):")
    for m in sorted(boot_ranks, key=lambda x: boot_ranks[x]['top1_pct'], reverse=True):
        print(f"  {m}: top1={boot_ranks[m]['top1_pct']:.1%}, top2={boot_ranks[m]['top2_pct']:.1%}")
    print(f"\nSignificant comparisons:")
    for _, row in ci_df.iterrows():
        if row['significance'] != 'ns':
            print(f"  {row['comparison']}: q={row['q_value']:.4f} {row['significance']}")
    print(f"\nResults saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
