"""
Steps 3-6: Incremental models, permutation importance, domain multivariable, sensitivity checks

Master script for source importance analysis (Steps 3-6).

Author: Source importance analysis for Interviewer-Bias-Audit
Date: 2026-07-04
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             log_loss, f1_score, recall_score)
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path("E:/CodexWorktrees/DAIC-WOZ/reanalysis-v2/analysis_v2")
FEATURES_PATH = BASE_DIR / "10_source_importance/01_source_level_table/source_level_features.csv"
SPLITS_PATH = BASE_DIR / "00_splits/repeated_5fold_splits_10x5.csv"

OUTPUT_DIRS = {
    'incremental': BASE_DIR / "10_source_importance/02_incremental_models",
    'importance': BASE_DIR / "10_source_importance/03_source_importance",
    'domain': BASE_DIR / "10_source_importance/04_domain_multivariable",
    'sensitivity': BASE_DIR / "10_source_importance/05_sensitivity_checks",
    'manuscript': BASE_DIR / "10_source_importance/06_manuscript_ready",
}

for d in OUTPUT_DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

# Model parameters
MODEL_PARAMS = {
    'penalty': 'l2',
    'C': 1.0,
    'class_weight': 'balanced',
    'solver': 'liblinear',
    'max_iter': 2000,
    'random_state': 42
}

# Source features
C2 = 'c2_prob'
DOMAIN_COUNT = 'domain_count_prob'
TEMPLATE_PRESENCE = 'template_presence_prob'
C5 = 'c5_prob'
C3 = 'c3_prob'
DOMAIN_PRESENCE = 'domain_presence_prob'
TEMPLATE_ONLY = 'template_only_prob'

# Significant domains (FDR < 0.05)
SIG_DOMAINS_PRESENCE = [
    'depressed_mood_presence', 'appetite_weight_presence',
    'suicide_self_harm_presence', 'functioning_impairment_presence',
    'mental_health_history_presence'
]
SIG_DOMAINS_COUNT = [
    'depressed_mood_count', 'appetite_weight_count',
    'suicide_self_harm_count', 'functioning_impairment_count',
    'mental_health_history_count'
]

# =============================================================================
# STEP 3: INCREMENTAL MODELS
# =============================================================================

def load_data():
    """Load source-level features and splits."""
    df = pd.read_csv(FEATURES_PATH)
    splits = pd.read_csv(SPLITS_PATH)
    return df, splits

def get_model():
    """Create a logistic regression model."""
    return LogisticRegression(**MODEL_PARAMS)

def compute_metrics(y_true, y_prob, y_pred):
    """Compute evaluation metrics."""
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = np.nan

    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except:
        pr_auc = np.nan

    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    brier = brier_score_loss(y_true, y_prob)
    try:
        logloss = log_loss(y_true, y_prob)
    except:
        logloss = np.nan

    return {
        'roc_auc': auc, 'pr_auc': pr_auc, 'macro_f1': macro_f1,
        'sensitivity': sensitivity, 'specificity': specificity,
        'brier': brier, 'log_loss': logloss
    }

def run_source_level_cv(df, splits, feature_cols):
    """Run source-level repeated 5-fold CV with logistic regression."""
    y_all = df['label'].values
    repeats = splits['repeat'].nunique()

    oof_probs = np.zeros(len(df))

    for rep in range(1, repeats + 1):
        rep_data = splits[splits['repeat'] == rep]
        for fold in range(1, 6):  # folds are 1-5
            fold_data = rep_data[rep_data['fold'] == fold]
            test_participants = fold_data[fold_data['role'] == 'test']['participant_id'].values

            # Test mask
            test_mask = df['participant_id'].isin(test_participants)
            train_mask = ~test_mask

            # Skip if any set has only one class
            y_train = y_all[train_mask]
            y_test = y_all[test_mask]
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue

            X_train = df.loc[train_mask, feature_cols].values
            X_test = df.loc[test_mask, feature_cols].values

            # Fit model
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = get_model()
            model.fit(X_train_scaled, y_train)

            oof_probs[test_mask] = model.predict_proba(X_test_scaled)[:, 1]

    return oof_probs

def evaluate_model(y_true, y_prob, y_pred):
    """Evaluate a model and return metrics."""
    metrics = compute_metrics(y_true, y_prob, y_pred)
    return metrics

def run_incremental_models(df, splits):
    """Run M0-M4 incremental models."""
    print("\n" + "=" * 60)
    print("STEP 3: INCREMENTAL MODELS (M0-M4)")
    print("=" * 60)

    model_specs = {
        'M0': {'desc': 'Patient speech only', 'features': [C2]},
        'M1': {'desc': 'Patient + Domain', 'features': [C2, DOMAIN_COUNT]},
        'M2': {'desc': 'Patient + Protocol', 'features': [C2, TEMPLATE_PRESENCE]},
        'M3': {'desc': 'Patient + Domain + Protocol', 'features': [C2, DOMAIN_COUNT, TEMPLATE_PRESENCE]},
        'M4': {'desc': 'Patient + Domain + Protocol + C5', 'features': [C2, DOMAIN_COUNT, TEMPLATE_PRESENCE, C5]},
        'M5': {'desc': 'Patient + Domain + Protocol + C3 (sensitivity)', 'features': [C2, DOMAIN_COUNT, TEMPLATE_PRESENCE, C3]},
    }

    results = []
    oof_predictions = {}
    y_all = df['label'].values

    for model_name, spec in model_specs.items():
        print(f"\nRunning {model_name}: {spec['desc']}")
        print(f"  Features: {spec['features']}")

        oof_probs = run_source_level_cv(df, splits, spec['features'])
        y_pred = (oof_probs >= 0.5).astype(int)

        metrics = evaluate_model(y_all, oof_probs, y_pred)

        metrics['model'] = model_name
        metrics['description'] = spec['desc']
        metrics['predictors'] = ' + '.join(spec['features'])
        metrics['n_features'] = len(spec['features'])

        results.append(metrics)
        oof_predictions[f'{model_name}_prob'] = oof_probs
        oof_predictions[f'{model_name}_pred'] = y_pred

        print(f"  AUC={metrics['roc_auc']:.3f}, Macro-F1={metrics['macro_f1']:.3f}, "
              f"Brier={metrics['brier']:.3f}, log-loss={metrics['log_loss']:.3f}")

    # Create results DataFrame
    df_results = pd.DataFrame(results)

    # Compute delta metrics
    for i in range(1, len(df_results)):
        for metric in ['roc_auc', 'pr_auc', 'macro_f1', 'brier']:
            if metric == 'brier':
                df_results.loc[i, f'delta_{metric}'] = df_results.loc[i, metric] - df_results.loc[i-1, metric]
            else:
                df_results.loc[i, f'delta_{metric}'] = df_results.loc[i, metric] - df_results.loc[i-1, metric]

    # Also compute deltas vs M0 for each model
    for i in range(len(df_results)):
        for metric in ['roc_auc', 'log_loss']:
            df_results.loc[i, f'delta_{metric}_vs_M0'] = df_results.loc[i, metric] - df_results.loc[0, metric]

    # Save results
    df_results.to_csv(OUTPUT_DIRS['incremental'] / "incremental_model_results.csv", index=False)

    # Save OOF predictions
    oof_preds_df = pd.DataFrame({'participant_id': df['participant_id'], 'label': y_all, **oof_predictions})
    oof_preds_df.to_csv(OUTPUT_DIRS['incremental'] / "incremental_model_oof_predictions.csv", index=False)

    print(f"\nResults saved to: {OUTPUT_DIRS['incremental']}")

    return df_results, oof_predictions

def save_incremental_model_results(df_results):
    """Save incremental model results as Markdown."""
    output_path = OUTPUT_DIRS['incremental'] / "incremental_model_results.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Incremental Model Results\n\n")
        f.write("## Model Specifications\n\n")
        f.write("| Model | Description | Predictors |\n")
        f.write("|-------|------------|------------|\n")
        for _, row in df_results.iterrows():
            f.write(f"| {row['model']} | {row['description']} | {row['predictors']} |\n")

        f.write("\n## Performance Metrics\n\n")
        f.write("| Model | ROC-AUC | PR-AUC | Macro-F1@0.5 | Sensitivity@0.5 | Specificity@0.5 | Brier | Log-Loss |\n")
        f.write("|-------|---------|--------|--------------|----------------|-----------------|-------|----------|\n")
        for _, row in df_results.iterrows():
            f.write(f"| {row['model']} | {row['roc_auc']:.4f} | {row['pr_auc']:.4f} | {row['macro_f1']:.4f} | "
                    f"{row['sensitivity']:.4f} | {row['specificity']:.4f} | {row['brier']:.4f} | {row['log_loss']:.4f} |\n")

        f.write("\n## Incremental Changes\n\n")
        f.write("| Model | Δ AUC | Δ PR-AUC | Δ Macro-F1 | Δ Brier | Δ AUC vs M0 |\n")
        f.write("|-------|--------|-----------|------------|---------|-------------|\n")
        for _, row in df_results.iterrows():
            delta_auc = row.get('delta_roc_auc', np.nan)
            delta_pr = row.get('delta_pr_auc', np.nan)
            delta_f1 = row.get('delta_macro_f1', np.nan)
            delta_brier = row.get('delta_brier', np.nan)
            delta_m0 = row.get('delta_roc_auc_vs_M0', np.nan)
            delta_str = lambda v: f"{v:+.4f}" if not np.isnan(v) else '-'
            f.write(f"| {row['model']} | {delta_str(delta_auc)} | {delta_str(delta_pr)} | {delta_str(delta_f1)} | "
                    f"{delta_str(delta_brier)} | {delta_str(delta_m0)} |\n")

    print(f"Saved: {output_path}")

def save_incremental_interpretation(df_results):
    """Save interpretation of incremental model results."""
    output_path = OUTPUT_DIRS['incremental'] / "incremental_model_interpretation.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Incremental Model Interpretation\n\n")

        f.write("## Key Findings\n\n")
        for i, (_, row) in enumerate(df_results.iterrows()):
            delta_auc_m0 = row.get('delta_roc_auc_vs_M0', np.nan)
            f.write(f"### {row['model']}: {row['description']}\n\n")
            f.write(f"- AUC: {row['roc_auc']:.4f}\n")
            f.write(f"- Log-loss: {row['log_loss']:.4f}\n")
            if not np.isnan(delta_auc_m0):
                direction = "improvement" if delta_auc_m0 > 0.01 else "stable" if abs(delta_auc_m0) <= 0.01 else "decrease"
                f.write(f"- Δ AUC vs M0: {delta_auc_m0:+.4f} ({direction})\n\n")

        f.write("## Interpretation\n\n")
        f.write("These results show the incremental contribution of different signal sources. ")
        f.write("Key interpretations are:\n\n")
        f.write("1. M0: Baseline - how well patient speech alone can predict\n")
        f.write("2. M1 vs M0: Whether domain count adds incremental value beyond patient speech\n")
        f.write("3. M2 vs M0: Whether template presence adds incremental value beyond patient speech\n")
        f.write("4. M3 vs M0/M1/M2: Whether combined model outperforms single-source models\n")
        f.write("5. M4 vs M3: Whether C5 quote adds value beyond domain count (overlap test)\n")
        f.write("6. M5 vs M3: Whether C3 interviewer adds value beyond template presence (sensitivity)\n\n")
        f.write("**Important:** These are descriptive source-importance audits, NOT causal attributions.\n")

    print(f"Saved: {output_path}")

# =============================================================================
# STEP 4: PERMUTATION IMPORTANCE
# =============================================================================

def run_permutation_importance(df, splits):
    """Run permutation importance for M3 model."""
    print("\n" + "=" * 60)
    print("STEP 4: PERMUTATION IMPORTANCE")
    print("=" * 60)

    feature_cols = [C2, DOMAIN_COUNT, TEMPLATE_PRESENCE]
    y_all = df['label'].values
    repeats = splits['repeat'].nunique()
    n_folds = 5

    # Store permutation results
    perm_results = {feat: {'auc_decrease': [], 'logloss_increase': [], 'brier_increase': []}
                    for feat in feature_cols}

    for rep in range(1, repeats + 1):
        rep_data = splits[splits['repeat'] == rep]
        for fold in range(1, 6):  # folds are 1-5
            fold_data = rep_data[rep_data['fold'] == fold]
            test_participants = fold_data[fold_data['role'] == 'test']['participant_id'].values
            test_mask = df['participant_id'].isin(test_participants)
            train_mask = ~test_mask

            y_train = y_all[train_mask]
            y_test = y_all[test_mask]
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue

            X_train = df.loc[train_mask, feature_cols].values
            X_test = df.loc[test_mask, feature_cols].values

            # Fit model
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = get_model()
            model.fit(X_train_scaled, y_train)

            # Baseline performance
            probs_baseline = model.predict_proba(X_test_scaled)[:, 1]
            auc_base = roc_auc_score(y_test, probs_baseline)
            try:
                ll_base = log_loss(y_test, probs_baseline)
            except:
                ll_base = np.nan
            brier_base = brier_score_loss(y_test, probs_baseline)

            # Permute each feature
            for feat_idx, feat_name in enumerate(feature_cols):
                X_test_permuted = X_test_scaled.copy()
                np.random.shuffle(X_test_permuted[:, feat_idx])

                probs_perm = model.predict_proba(X_test_permuted)[:, 1]
                try:
                    auc_perm = roc_auc_score(y_test, probs_perm)
                    ll_perm = log_loss(y_test, probs_perm)
                except:
                    auc_perm = np.nan
                    ll_perm = np.nan
                brier_perm = brier_score_loss(y_test, probs_perm)

                perm_results[feat_name]['auc_decrease'].append(auc_base - auc_perm)
                if not np.isnan(ll_base) and not np.isnan(ll_perm):
                    perm_results[feat_name]['logloss_increase'].append(ll_perm - ll_base)
                perm_results[feat_name]['brier_increase'].append(brier_perm - brier_base)

    # Aggregate results
    importance_data = []
    for feat in feature_cols:
        auc_mean = np.mean(perm_results[feat]['auc_decrease'])
        auc_std = np.std(perm_results[feat]['auc_decrease'])
        ll_mean = np.mean(perm_results[feat]['logloss_increase']) if perm_results[feat]['logloss_increase'] else np.nan
        brier_mean = np.mean(perm_results[feat]['brier_increase'])

        importance_data.append({
            'source': feat,
            'delta_auc_mean': auc_mean,
            'delta_auc_std': auc_std,
            'delta_logloss_mean': ll_mean,
            'delta_brier_mean': brier_mean,
            'n_permutations': len(perm_results[feat]['auc_decrease'])
        })
        print(f"  {feat}: ΔAUC={auc_mean:.4f} (±{auc_std:.4f})")

    df_importance = pd.DataFrame(importance_data)
    df_importance['rank_auc'] = df_importance['delta_auc_mean'].rank(ascending=False)

    # Save
    df_importance.to_csv(OUTPUT_DIRS['importance'] / "permutation_importance_auc.csv", index=False)

    # Summary markdown
    with open(OUTPUT_DIRS['importance'] / "source_importance_rank_summary.md", 'w', encoding='utf-8') as f:
        f.write("# Source Importance Ranking (Permutation Importance)\n\n")
        f.write("Based on M3 model: c2 + domain_count + template_presence\n\n")
        f.write("| Source | Δ AUC (mean) | Δ AUC (std) | Δ Log-Loss | Δ Brier | Rank |\n")
        f.write("|--------|-------------|-------------|------------|---------|------|\n")
        for _, row in df_importance.iterrows():
            f.write(f"| {row['source']} | {row['delta_auc_mean']:.4f} | {row['delta_auc_std']:.4f} | "
                    f"{row['delta_logloss_mean']:.4f} | {row['delta_brier_mean']:.4f} | {int(row['rank_auc'])} |\n")

    # Interpretation
    with open(OUTPUT_DIRS['importance'] / "source_importance_interpretation.md", 'w', encoding='utf-8') as f:
        f.write("# Source Importance Interpretation\n\n")
        f.write("## Method\n\n")
        f.write("Permutation importance was computed under the M3 model ")
        f.write("(c2_participant_prob + domain_count_prob + template_presence_prob) ")
        f.write("using repeated 5-fold CV. For each fold, test-participant features were ")
        f.write("shuffled independently and the resulting performance degradation was recorded.\n\n")
        f.write("## Results\n\n")
        top_feat = df_importance.iloc[0]
        f.write(f"The most important source by AUC degradation was {top_feat['source']} ")
        f.write(f"(ΔAUC = {top_feat['delta_auc_mean']:.4f} ± {top_feat['delta_auc_std']:.4f}).\n\n")
        f.write("## Interpretation Guide\n\n")
        f.write("- Larger AUC decrease → source is more important for ranking\n")
        f.write("- Larger log-loss increase → source is more important for probability prediction\n")
        f.write("- If standard deviations are large relative to means, importance is unstable\n")
        f.write("- If multiple sources are close, signal overlap should be acknowledged\n\n")
        f.write("**Note:** These are descriptive importance estimates, not causal attributions.\n")

    print(f"\nResults saved to: {OUTPUT_DIRS['importance']}")
    return df_importance

# =============================================================================
# STEP 5: DOMAIN MULTIVARIABLE ANALYSIS
# =============================================================================

def run_domain_multivariable(df, splits):
    """Run domain-level multivariable and leave-one-out analysis."""
    print("\n" + "=" * 60)
    print("STEP 5: DOMAIN MULTIVARIABLE ANALYSIS")
    print("=" * 60)

    # For presence domains
    print("\n--- Presence domain model ---")
    presence_results = run_domain_model(df, splits, SIG_DOMAINS_PRESENCE, feature_type='presence')

    # For count domains
    print("\n--- Count domain model ---")
    count_results = run_domain_model(df, splits, SIG_DOMAINS_COUNT, feature_type='count')

    return presence_results, count_results

def run_domain_model(df, splits, domain_features, feature_type='presence'):
    """Run a single domain model with leave-one-out analysis."""
    y_all = df['label'].values

    # Full model
    print(f"\nFull model ({len(domain_features)} domains):")
    oof_probs_full = run_source_level_cv(df, splits, domain_features)
    y_pred_full = (oof_probs_full >= 0.5).astype(int)
    auc_full = roc_auc_score(y_all, oof_probs_full)
    try:
        ll_full = log_loss(y_all, oof_probs_full)
    except:
        ll_full = np.nan
    print(f"  AUC={auc_full:.4f}, Log-Loss={ll_full:.4f}")

    # Leave-one-domain-out
    leave_one_out_results = []
    for domain in domain_features:
        features_minus = [d for d in domain_features if d != domain]
        oof_probs = run_source_level_cv(df, splits, features_minus)
        y_pred = (oof_probs >= 0.5).astype(int)
        auc_reduced = roc_auc_score(y_all, oof_probs)
        try:
            ll_reduced = log_loss(y_all, oof_probs)
        except:
            ll_reduced = np.nan

        leave_one_out_results.append({
            'feature_type': feature_type,
            'removed_domain': domain,
            'full_auc': auc_full,
            'reduced_auc': auc_reduced,
            'delta_auc': auc_full - auc_reduced,
            'full_logloss': ll_full,
            'reduced_logloss': ll_reduced,
            'delta_logloss': ll_reduced - ll_full,
        })
        print(f"  Remove {domain}: AUC={auc_reduced:.4f} (Δ={auc_full - auc_reduced:+.4f})")

    df_loo = pd.DataFrame(leave_one_out_results)
    df_loo['importance_rank'] = df_loo['delta_auc'].rank(ascending=False)

    # Save
    df_loo.to_csv(OUTPUT_DIRS['domain'] / f"domain_leave_one_out_{feature_type}.csv", index=False)

    return df_loo

def save_domain_results(presence_loo, count_loo):
    """Save domain multivariable results."""
    output_path = OUTPUT_DIRS['domain'] / "domain_importance_summary.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Domain-Level Multivariable Importance\n\n")

        f.write("## Method\n\n")
        f.write("Five FDR-significant domains were used in multivariable logistic regression ")
        f.write("with repeated 5-fold CV. Leave-one-domain-out analysis quantifies each domain's contribution.\n\n")

        f.write("## Presence Domain Results\n\n")
        f.write("| Removed Domain | Full AUC | Reduced AUC | Δ AUC | Importance Rank |\n")
        f.write("|---------------|----------|-------------|-------|----------------|\n")
        for _, row in presence_loo.iterrows():
            f.write(f"| {row['removed_domain']} | {row['full_auc']:.4f} | {row['reduced_auc']:.4f} | "
                    f"{row['delta_auc']:+.4f} | {int(row['importance_rank'])} |\n")

        f.write("\n## Count Domain Results\n\n")
        f.write("| Removed Domain | Full AUC | Reduced AUC | Δ AUC | Importance Rank |\n")
        f.write("|---------------|----------|-------------|-------|----------------|\n")
        for _, row in count_loo.iterrows():
            f.write(f"| {row['removed_domain']} | {row['full_auc']:.4f} | {row['reduced_auc']:.4f} | "
                    f"{row['delta_auc']:+.4f} | {int(row['importance_rank'])} |\n")

        f.write("\n## Interpretation\n\n")
        f.write("- Larger Δ AUC → domain contributes more to model performance\n")
        f.write("- If Δ AUC is near zero or negative → domain is redundant\n")
        f.write("- Domains with high importance should be discussed in manuscript\n\n")
        f.write("**Caution:** These are descriptive importance estimates under the current feature set. ")
        f.write("Sample size (n=142) limits the reliability of individual domain importance estimates.\n")

    print(f"Saved: {output_path}")

# =============================================================================
# STEP 6: SENSITIVITY CHECKS
# =============================================================================

def run_sensitivity_checks(df, splits):
    """Run sensitivity checks."""
    print("\n" + "=" * 60)
    print("STEP 6: SENSITIVITY CHECKS")
    print("=" * 60)

    y_all = df['label'].values
    sensitivity_results = []

    # Check 1: domain_presence vs domain_count
    m3_count = [C2, DOMAIN_COUNT, TEMPLATE_PRESENCE]
    m3_presence = [C2, DOMAIN_PRESENCE, TEMPLATE_PRESENCE]

    oof_count = run_source_level_cv(df, splits, m3_count)
    oof_presence = run_source_level_cv(df, splits, m3_presence)

    auc_count = roc_auc_score(y_all, oof_count)
    auc_presence = roc_auc_score(y_all, oof_presence)
    print(f"\n  M3 (domain_count): AUC={auc_count:.4f}")
    print(f"  M3 (domain_presence): AUC={auc_presence:.4f}")

    sensitivity_results.append({
        'check': 'domain_count vs domain_presence',
        'config_a': 'domain_count', 'auc_a': auc_count,
        'config_b': 'domain_presence', 'auc_b': auc_presence,
        'delta': auc_count - auc_presence
    })

    # Check 2: template_presence vs template_only
    m3_tp = [C2, DOMAIN_COUNT, TEMPLATE_PRESENCE]
    m3_to = [C2, DOMAIN_COUNT, TEMPLATE_ONLY]

    oof_tp = run_source_level_cv(df, splits, m3_tp)
    oof_to = run_source_level_cv(df, splits, m3_to)

    auc_tp = roc_auc_score(y_all, oof_tp)
    auc_to = roc_auc_score(y_all, oof_to)
    print(f"\n  M3 (template_presence): AUC={auc_tp:.4f}")
    print(f"  M3 (template_only): AUC={auc_to:.4f}")

    sensitivity_results.append({
        'check': 'template_presence vs template_only',
        'config_a': 'template_presence', 'auc_a': auc_tp,
        'config_b': 'template_only', 'auc_b': auc_to,
        'delta': auc_tp - auc_to
    })

    # Check 3: With/without C5 quote
    m3 = [C2, DOMAIN_COUNT, TEMPLATE_PRESENCE]
    m4 = [C2, DOMAIN_COUNT, TEMPLATE_PRESENCE, C5]

    oof_m3 = run_source_level_cv(df, splits, m3)
    oof_m4 = run_source_level_cv(df, splits, m4)

    auc_m3 = roc_auc_score(y_all, oof_m3)
    auc_m4 = roc_auc_score(y_all, oof_m4)
    try:
        ll_m3 = log_loss(y_all, oof_m3)
        ll_m4 = log_loss(y_all, oof_m4)
    except:
        ll_m3, ll_m4 = np.nan, np.nan

    print(f"\n  M3 (without C5): AUC={auc_m3:.4f}")
    print(f"  M4 (with C5): AUC={auc_m4:.4f}")

    sensitivity_results.append({
        'check': 'with vs without C5 quote',
        'config_a': 'without C5', 'auc_a': auc_m3,
        'config_b': 'with C5', 'auc_b': auc_m4,
        'delta': auc_m4 - auc_m3
    })

    # Save
    df_sensitivity = pd.DataFrame(sensitivity_results)
    df_sensitivity.to_csv(OUTPUT_DIRS['sensitivity'] / "sensitivity_source_set_results.csv", index=False)

    # Summary
    with open(OUTPUT_DIRS['sensitivity'] / "sensitivity_summary.md", 'w', encoding='utf-8') as f:
        f.write("# Sensitivity Analysis Summary\n\n")
        f.write("| Check | Config A | AUC A | Config B | AUC B | Δ AUC |\n")
        f.write("|-------|----------|-------|----------|-------|-------|\n")
        for _, row in df_sensitivity.iterrows():
            f.write(f"| {row['check']} | {row['config_a']} | {row['auc_a']:.4f} | {row['config_b']} | {row['auc_b']:.4f} | {row['delta']:+.4f} |\n")
        f.write("\n## Interpretation\n\n")
        f.write("- If Δ AUC is small, conclusions are robust to feature encoding choice\n")
        f.write("- If C5 adds little to M3, C5 quote signal largely overlaps with domain/protocol features\n")

    print(f"\nResults saved to: {OUTPUT_DIRS['sensitivity']}")
    return df_sensitivity

# =============================================================================
# MANUSCRIPT-READY OUTPUTS
# =============================================================================

def create_manuscript_tables(df_incremental, df_importance, presence_loo, count_loo):
    """Create manuscript-ready tables and summaries."""
    print("\n" + "=" * 60)
    print("CREATING MANUSCRIPT-READY OUTPUTS")
    print("=" * 60)

    # Table: Source incremental models
    with open(OUTPUT_DIRS['manuscript'] / "table_source_incremental_models.md", 'w', encoding='utf-8') as f:
        f.write("# Table SX. Source-Level Incremental Model Performance\n\n")
        f.write("| Model | Predictors | ROC-AUC | Macro-F1@0.5 | Brier | Log-Loss | Δ AUC vs M0 |\n")
        f.write("|-------|-----------|---------|--------------|-------|----------|-------------|\n")
        for _, row in df_incremental.iterrows():
            delta = row.get('delta_roc_auc_vs_M0', np.nan)
            delta_str = f"{delta:+.4f}" if not np.isnan(delta) else '-'
            f.write(f"| {row['model']} | {row['predictors']} | {row['roc_auc']:.4f} | {row['macro_f1']:.4f} | "
                    f"{row['brier']:.4f} | {row['log_loss']:.4f} | {delta_str} |\n")

    # Table: Source importance
    with open(OUTPUT_DIRS['manuscript'] / "table_source_importance.md", 'w', encoding='utf-8') as f:
        f.write("# Table SX. Source Importance Ranking (Permutation Importance)\n\n")
        f.write("| Source | Δ AUC (mean ± SD) | Importance Rank |\n")
        f.write("|--------|-------------------|----------------|\n")
        for _, row in df_importance.iterrows():
            f.write(f"| {row['source']} | {row['delta_auc_mean']:.4f} ± {row['delta_auc_std']:.4f} | {int(row['rank_auc'])} |\n")

    # Table: Domain importance
    with open(OUTPUT_DIRS['manuscript'] / "table_domain_importance.md", 'w', encoding='utf-8') as f:
        f.write("# Table SX. Domain Importance by Leave-One-Out Analysis\n\n")
        f.write("## Presence Domains\n\n")
        f.write("| Removed Domain | Full AUC | Reduced AUC | Δ AUC |\n")
        f.write("|---------------|----------|-------------|-------|\n")
        for _, row in presence_loo.iterrows():
            f.write(f"| {row['removed_domain']} | {row['full_auc']:.4f} | {row['reduced_auc']:.4f} | {row['delta_auc']:+.4f} |\n")
        f.write("\n## Count Domains\n\n")
        f.write("| Removed Domain | Full AUC | Reduced AUC | Δ AUC |\n")
        f.write("|---------------|----------|-------------|-------|\n")
        for _, row in count_loo.iterrows():
            f.write(f"| {row['removed_domain']} | {row['full_auc']:.4f} | {row['reduced_auc']:.4f} | {row['delta_auc']:+.4f} |\n")

    # English results summary
    with open(OUTPUT_DIRS['manuscript'] / "source_importance_results_en.md", 'w', encoding='utf-8') as f:
        f.write("# Source Importance Analysis Results\n\n")

        f.write("## 1. Source Feature Correlations\n\n")
        f.write("Key correlation findings (Spearman):\n\n")
        f.write("- C2 patient speech vs C5 symptom evidence: moderate (r ≈ 0.59)\n")
        f.write("- C5 vs domain_count: moderate-high (r ≈ 0.59)\n")
        f.write("- template_only vs template_presence: high (r ≈ 0.80)\n")
        f.write("- C3 interviewer vs template_only: very high (r ≈ 0.85)\n\n")

        f.write("## 2. Incremental Model Performance\n\n")
        best = df_incremental.loc[df_incremental['roc_auc'].idxmax()]
        f.write(f"The best-performing source-level model ({best['model']}: {best['description']}) ")
        f.write(f"achieved AUC={best['roc_auc']:.4f}, Macro-F1={best['macro_f1']:.4f}.\n\n")

        f.write("## 3. Source Importance Ranking\n\n")
        top = df_importance.iloc[0]
        f.write(f"The most important source was {top['source']} (ΔAUC={top['delta_auc_mean']:.4f} ± {top['delta_auc_std']:.4f}).\n\n")

        f.write("## 4. Domain-Level Importance\n\n")
        f.write("The leave-one-domain-out analysis identified which domains most strongly contribute to predictive performance.\n\n")

        f.write("## 5. Sensitivity\n\n")
        f.write("Sensitivity checks confirmed that results are [stable/unstable] to variations in feature encoding.\n\n")

        f.write("## Interpretation\n\n")
        f.write("These analyses characterize the relative contribution of different signal sources ")
        f.write("in DAIC-WOZ PHQ-8 classification. They are descriptive source-importance audits, ")
        f.write("not causal attributions. The results show which signal sources provide incremental ")
        f.write("value beyond patient speech, and which are largely redundant or overlapping.\n")

    # Chinese results summary
    with open(OUTPUT_DIRS['manuscript'] / "source_importance_results_zh.md", 'w', encoding='utf-8') as f:
        f.write("# 来源重要性分析结果\n\n")
        f.write("## 1. 来源特征相关性\n\n")
        f.write("关键相关发现（Spearman）：\n\n")
        f.write("- C2患者语言 vs C5症状证据：中度相关（r ≈ 0.59）\n")
        f.write("- C5 vs domain_count：中-高度相关（r ≈ 0.59）\n")
        f.write("- template_only vs template_presence：高度相关（r ≈ 0.80）\n")
        f.write("- C3访谈者 vs template_only：非常高度相关（r ≈ 0.85）\n\n")

        f.write("## 2. 增量模型表现\n\n")
        best = df_incremental.loc[df_incremental['roc_auc'].idxmax()]
        f.write(f"最优来源层级模型（{best['model']}: {best['description']}）")
        f.write(f"AUC={best['roc_auc']:.4f}，Macro-F1={best['macro_f1']:.4f}。\n\n")

        f.write("## 3. 来源重要性排序\n\n")
        top = df_importance.iloc[0]
        f.write(f"最重要的来源为{top['source']}（ΔAUC={top['delta_auc_mean']:.4f} ± {top['delta_auc_std']:.4f}）。\n\n")

        f.write("## 4. Domain层级重要性\n\n")
        f.write("Leave-one-domain-out 分析识别出对预测表现最有贡献的临床域。\n\n")

        f.write("## 5. 敏感性分析\n\n")
        f.write("敏感性检验确认结果对不同特征编码方式[稳定/不稳定]。\n\n")

        f.write("## 解释\n\n")
        f.write("上述分析描述了不同信号来源在DAIC-WOZ PHQ-8阳性预测中的相对贡献，")
        f.write("是描述性来源重要性审计，而非因果归因。结果表明哪些信号来源在患者语言之外提供增量价值，")
        f.write("哪些来源之间高度重叠。\n")

    print(f"\nManuscript-ready outputs saved to: {OUTPUT_DIRS['manuscript']}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run Steps 3-6 of source importance analysis."""
    print("\n" + "=" * 70)
    print("SOURCE IMPORTANCE ANALYSIS: STEPS 3-6")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    df, splits = load_data()
    print(f"Features: {df.shape}, Splits: {splits.shape}")

    # Step 3: Incremental models
    df_incremental, oof_predictions = run_incremental_models(df, splits)
    save_incremental_model_results(df_incremental)
    save_incremental_interpretation(df_incremental)

    # Step 4: Permutation importance
    df_importance = run_permutation_importance(df, splits)

    # Step 5: Domain multivariable
    presence_loo, count_loo = run_domain_multivariable(df, splits)
    save_domain_results(presence_loo, count_loo)

    # Step 6: Sensitivity checks
    df_sensitivity = run_sensitivity_checks(df, splits)

    # Manuscript-ready outputs
    create_manuscript_tables(df_incremental, df_importance, presence_loo, count_loo)

    print("\n" + "=" * 70)
    print("ALL STEPS COMPLETE")
    print("=" * 70)

if __name__ == '__main__':
    main()
