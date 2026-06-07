

import pandas as pd
import numpy as np
import pickle
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    matthews_corrcoef,
    balanced_accuracy_score,
    cohen_kappa_score
)
import warnings
warnings.filterwarnings('ignore')



# ----- Project Paths OS SAFE
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
METRICS_DIR = os.path.join(PROJECT_ROOT, "metrics")
PLOTS_DIR = os.path.join(METRICS_DIR, "plots")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)



print("="*70)
print("MEDIREACH - OPTIMIZED ML TRAINING")
print(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

# ============================================================================
# 1. LOAD DATA WITH ENGINEERED FEATURES
# ============================================================================
print("\n1. LOADING DATA...")
print("-"*50)

X_train = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_train.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_test.csv")).values.ravel()

print(f"Training: {X_train.shape[0]:,} rows, {X_train.shape[1]} features")
print(f"Test: {X_test.shape[0]:,} rows")
print(f"No-show rate: {y_train.mean():.2%}")

# Add engineered features that weren't in original CSV
print("\n2. ADDING ADVANCED ENGINEERED FEATURES...")
print("-"*50)

def add_engineered_features(X, is_train=True):
    """Add derived features for better prediction"""
    X = X.copy()
    
    # Already in your data (from feature engineering):
    # - lead_days, past_no_show_rate, young_long_wait, high_risk_combo
    
    # NEW ENGINEERED FEATURES:
    
    # 1. Is weekend feature (from appointment_dow)
    if 'appointment_dow' in X.columns:
        X['is_weekend'] = X['appointment_dow'].isin([5, 6]).astype(int)
    
    # 2. Age squared (captures non-linear age effects)
    if 'age' in X.columns:
        X['age_squared'] = (X['age'] / 50) ** 2  # Normalized squared
    
    # 3. Lead days bins (categorical risk)
    if 'lead_days' in X.columns:
        X['lead_days_binned'] = pd.cut(X['lead_days'], 
                                       bins=[-1, 3, 7, 14, 30, 1000],
                                       labels=[0, 1, 2, 3, 4]).astype(int)
    
    # 4. Scholarship × Age interaction (young + scholarship = more likely to attend?)
    if 'scholarship' in X.columns and 'age' in X.columns:
        X['scholarship_young'] = ((X['scholarship'] == 1) & (X['age'] < 35)).astype(int)
    
    # 5. Chronic conditions count (hypertension + diabetes + alcoholism)
    chronic_cols = ['hypertension', 'diabetes', 'alcoholism']
    existing_chronic = [c for c in chronic_cols if c in X.columns]
    if existing_chronic:
        X['chronic_count'] = X[existing_chronic].sum(axis=1)
    
    # 6. Handicap squared (non-linear disability effect)
    if 'handicap' in X.columns:
        X['handicap_squared'] = X['handicap'] ** 2
    
    # 7. Lead days × Past no-show (interaction)
    if 'lead_days' in X.columns and 'past_no_show_rate' in X.columns:
        X['lead_x_past'] = X['lead_days'] * X['past_no_show_rate']
    
    # 8. SMS effectiveness indicator (SMS × Age < 35)
    if 'sms_received' in X.columns and 'age' in X.columns:
        X['sms_young_effective'] = ((X['sms_received'] == 1) & (X['age'] < 35)).astype(int)
    
    return X

X_train = add_engineered_features(X_train)
X_test = add_engineered_features(X_test)

print(f"New feature count: {X_train.shape[1]}")
print(f"Added features: {[c for c in X_train.columns if c not in ['age', 'lead_days', 'sms_received', 'scholarship', 'hypertension', 'diabetes', 'alcoholism', 'handicap', 'past_no_show_rate', 'young_long_wait', 'high_risk_combo', 'appointment_dow', 'age_0-17', 'age_18-34', 'age_35-49', 'age_50-64', 'age_65+']]}")

# ============================================================================
# 3. FEATURE SCALING (Random Forest doesn't need, Logistic Regression does)
# ============================================================================
print("\n3. FEATURE SCALING...")
print("-"*50)

# Separate features for scaling
binary_features = ['sms_received', 'scholarship', 'hypertension', 'diabetes', 
                   'alcoholism', 'young_long_wait', 'high_risk_combo', 'is_weekend',
                   'scholarship_young', 'sms_young_effective']
categorical_features = ['lead_days_binned']
scale_features = [c for c in X_train.columns if c not in binary_features + categorical_features]

print(f"Scaling {len(scale_features)} continuous features")
print(f"Keeping {len(binary_features)} binary features unscaled")
print(f"Keeping {len(categorical_features)} categorical features as-is")

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[scale_features] = scaler.fit_transform(X_train[scale_features])
X_test_scaled[scale_features] = scaler.transform(X_test[scale_features])

# Save scaler
with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# ============================================================================
# 4. LOGISTIC REGRESSION (Baseline, Explainable)
# ============================================================================
print("\n4. TRAINING LOGISTIC REGRESSION...")
print("-"*50)

lr_model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced',
    C=1.0,
    solver='liblinear'
)
lr_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_lr = lr_model.predict(X_test_scaled)
y_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

# ============================================================================
# 5. RANDOM FOREST (200 trees, optimized)
# ============================================================================
print("\n5. TRAINING RANDOM FOREST (200 estimators)...")
print("-"*50)

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced',
    max_depth=12,
    min_samples_split=15,
    min_samples_leaf=8,
    max_features='sqrt',
    n_jobs=-1,
    verbose=0
)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test_scaled)
y_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

# ============================================================================
# 6. COMPREHENSIVE METRICS FUNCTION
# ============================================================================
print("\n6. CALCULATING COMPREHENSIVE METRICS...")
print("-"*50)

def calculate_all_metrics(y_true, y_pred, y_proba, model_name):
    """Calculate every relevant metric for healthcare operations"""
    
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'model': model_name,
        'timestamp': datetime.now().isoformat(),
        
        # Core classification metrics
        'roc_auc': roc_auc_score(y_true, y_proba),
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred),
        
        # Confusion matrix (raw numbers)
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        
        # Business-oriented metrics
        'missed_no_shows': int(fn),  # Patients we failed to identify
        'false_alarms': int(fp),     # Patients we'd bother unnecessarily
        'catch_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,  # Recall alias
        
        # Rates
        'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
        'true_positive_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
        
        # Precision-Recall tradeoff
        'precision_at_60_threshold': None,  # Will fill later
        'recall_at_60_threshold': None,
        'f1_at_60_threshold': None,
    }
    
    # Add threshold-specific metrics (0.6 = action engine default)
    y_pred_60 = (y_proba >= 0.6).astype(int)
    metrics['precision_at_60_threshold'] = precision_score(y_true, y_pred_60, zero_division=0)
    metrics['recall_at_60_threshold'] = recall_score(y_true, y_pred_60)
    metrics['f1_at_60_threshold'] = f1_score(y_true, y_pred_60)
    
    return metrics

# Calculate for both models
lr_metrics = calculate_all_metrics(y_test, y_pred_lr, y_proba_lr, 'Logistic Regression')
rf_metrics = calculate_all_metrics(y_test, y_pred_rf, y_proba_rf, 'Random Forest (200)')

# ============================================================================
# 7. FEATURE IMPORTANCE (Business Interpretability)
# ============================================================================
print("\n7. FEATURE IMPORTANCE ANALYSIS...")
print("-"*50)

# Logistic Regression coefficients
lr_importance = pd.DataFrame({
    'feature': X_train.columns,
    'coefficient': lr_model.coef_[0],
    'abs_coefficient': np.abs(lr_model.coef_[0])
}).sort_values('abs_coefficient', ascending=False)

# Random Forest importance
rf_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTOP 10 FEATURES (Random Forest):")
for i, row in rf_importance.head(10).iterrows():
    print(f"  {row['importance']:.4f} - {row['feature']}")

print("\nTOP 10 FEATURES (Logistic Regression - absolute coefficients):")
for i, row in lr_importance.head(10).iterrows():
    direction = "↑" if row['coefficient'] > 0 else "↓"
    print(f"  {direction} {row['coefficient']:+.4f} - {row['feature']}")

# ============================================================================
# 8. MODEL COMPARISON & SELECTION
# ============================================================================
print("\n8. MODEL COMPARISON...")
print("-"*50)

# Create comparison DataFrame
comparison_data = []
for metrics in [lr_metrics, rf_metrics]:
    comparison_data.append({
        'Model': metrics['model'],
        'ROC-AUC': metrics['roc_auc'],
        'Recall': metrics['recall'],
        'Precision': metrics['precision'],
        'F1-Score': metrics['f1_score'],
        'Balanced Acc': metrics['balanced_accuracy'],
        'MCC': metrics['matthews_corrcoef'],
        'Kappa': metrics['cohen_kappa'],
        'Catch Rate': metrics['catch_rate'],
        'False Alarms': metrics['false_alarms'],
        'Missed': metrics['missed_no_shows']
    })

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# Select best model (based on Recall - catching no-shows is priority)
# For healthcare: Catching no-shows (Recall) > Overall accuracy
best_model_name = 'Random Forest (200)' if rf_metrics['recall'] >= lr_metrics['recall'] else 'Logistic Regression'
best_model = rf_model if best_model_name == 'Random Forest (200)' else lr_model
best_proba = y_proba_rf if best_model_name == 'Random Forest (200)' else y_proba_lr

print(f"\n✓ SELECTED: {best_model_name}")
print(f"  Reason: Highest Recall ({max(rf_metrics['recall'], lr_metrics['recall']):.3f})")
print(f"  ROC-AUC: {max(rf_metrics['roc_auc'], lr_metrics['roc_auc']):.3f}")

# ============================================================================
# 9. THRESHOLD OPTIMIZATION (Based on Business Needs)
# ============================================================================
print("\n9. THRESHOLD OPTIMIZATION...")
print("-"*50)

thresholds = np.round (np.arange(0.3, 0.85, 0.05),2)
threshold_results = []

print("Threshold | Precision | Recall | F1 | Catch Rate | False Alarms | Missed")
print("-"*75)

for threshold in thresholds:
    y_pred_custom = (best_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_custom).ravel()
    
    precision_t = precision_score(y_test, y_pred_custom, zero_division=0)
    recall_t = recall_score(y_test, y_pred_custom)
    f1_t = f1_score(y_test, y_pred_custom)
    
    threshold_results.append({
        'threshold': threshold,
        'precision': precision_t,
        'recall': recall_t,
        'f1': f1_t,
        'false_alarms': fp,
        'missed': fn,
        'catch_rate': tp / (tp + fn) if (tp + fn) > 0 else 0
    })
    
    marker = " ← RECOMMENDED" if threshold == 0.6 else ""
    print(f"{threshold:.2f}       | {precision_t:.3f}    | {recall_t:.3f}   | {f1_t:.3f}   | {tp/(tp+fn):.3f}      | {fp:5d}     | {fn:5d}{marker}")

# Find optimal threshold (maximizing Recall while keeping Precision > 0.5)
optimal_threshold = 0.6  # Default from architecture
for result in threshold_results:
    if result['precision'] >= 0.5 and result['recall'] > 0.6:
        optimal_threshold = result['threshold']
        break

print(f"\n✓ Optimal threshold: {optimal_threshold}")

selected_result = next(
    (r for r in threshold_results
     if abs(r['threshold'] - optimal_threshold) < 1e-6),
    None
)

if selected_result:
    print(
        f"  At this threshold: "
        f"Precision={selected_result['precision']:.3f}, "
        f"Recall={selected_result['recall']:.3f}"
    )
else:
    print("  Threshold metrics not found.")
# ============================================================================
# 10. SAVE CONFUSION MATRIX & ROC CURVE PLOTS
# ============================================================================
print("\n10. SAVING VISUALIZATIONS...")
print("-"*50)

# Confusion Matrix Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# For best model at threshold 0.6
y_pred_opt = (best_proba >= optimal_threshold).astype(int)
cm = confusion_matrix(y_test, y_pred_opt)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Showed Up', 'No-Show'],
            yticklabels=['Showed Up', 'No-Show'])
axes[0].set_title(f'{best_model_name} - Confusion Matrix\nThreshold={optimal_threshold}')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, best_proba)
auc = roc_auc_score(y_test, best_proba)

axes[1].plot(fpr, tpr, linewidth=2, label=f'{best_model_name} (AUC = {auc:.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(PLOTS_DIR, "confusion_matrix_roc.png"),
    dpi=150,
    bbox_inches="tight"
)
plt.close()
print("✓ Saved: metrics/plots/confusion_matrix_roc.png")

# Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(y_test, best_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, linewidth=2, label=best_model_name)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.axhline(y=y_test.mean(), color='r', linestyle='--', label=f'Baseline ({y_test.mean():.3f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(
    os.path.join(PLOTS_DIR, "precision_recall_curve.png"),
    dpi=150,
    bbox_inches="tight"
)
plt.close()
print("✓ Saved: metrics/plots/precision_recall_curve.png")

# ============================================================================
# 11. SAVE ALL MODELS & METRICS
# ============================================================================
print("\n11. SAVING MODELS AND METRICS...")
print("-"*50)

# Save models
with open(os.path.join(MODELS_DIR, "logistic_regression.pkl"), "wb") as f:
    pickle.dump(lr_model, f)

with open(os.path.join(MODELS_DIR, "random_forest_200.pkl"), "wb") as f:
    pickle.dump(rf_model, f)

with open(os.path.join(MODELS_DIR, "best_model.pkl"), "wb") as f:
    pickle.dump(best_model, f)

with open(os.path.join(MODELS_DIR, "risk_threshold.txt"), "w") as f:
    f.write(str(optimal_threshold))

print("✓ Models saved to /models")

# Save comprehensive metrics
all_metrics = {
    'logistic_regression': lr_metrics,
    'random_forest_200': rf_metrics,
    'best_model': best_model_name,
    'optimal_threshold': optimal_threshold,
    'threshold_analysis': threshold_results,
    'feature_importance': {
        'random_forest': rf_importance.to_dict('records'),
        'logistic_regression': lr_importance.to_dict('records')
    },
    'dataset_info': {
        'train_size': len(X_train),
        'test_size': len(X_test),
        'no_show_rate': float(y_train.mean()),
        'n_features': X_train.shape[1],
        'feature_names': X_train.columns.tolist()
    },
    'training_date': datetime.now().isoformat()
}

def json_converter(obj):
    import numpy as np

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

with open(os.path.join(METRICS_DIR, "model_metrics.json"), "w") as f:
    json.dump(all_metrics, f, indent=2, default=json_converter)
print("✓ Metrics saved to metrics/model_metrics.json")

# Save feature importance as CSV for Power BI
rf_importance.to_csv(
    os.path.join(METRICS_DIR, "rf_feature_importance.csv"),
    index=False
)

lr_importance.to_csv(
    os.path.join(METRICS_DIR, "lr_coefficients.csv"),
    index=False
)
print("✓ Feature importance saved as CSV")

# ============================================================================
# 12. FINAL SUMMARY REPORT
# ============================================================================
print("\n" + "="*70)
print("TRAINING COMPLETE - EXECUTIVE SUMMARY")
print("="*70)

print(f"""
DATASET:
  Total appointments: {len(X_train) + len(X_test):,}
  No-show rate: {y_train.mean():.2%}
  Features used: {X_train.shape[1]}

MODEL PERFORMANCE ({best_model_name}):
  ROC-AUC: {max(rf_metrics['roc_auc'], lr_metrics['roc_auc']):.4f}
  Recall (Catch Rate): {max(rf_metrics['recall'], lr_metrics['recall']):.3f}
  Precision: {rf_metrics['precision'] if best_model_name=='Random Forest (200)' else lr_metrics['precision']:.3f}
  F1-Score: {rf_metrics['f1_score'] if best_model_name=='Random Forest (200)' else lr_metrics['f1_score']:.3f}

AT THRESHOLD {optimal_threshold}:
  True Positives (Caught no-shows): {confusion_matrix(y_test, (best_proba >= optimal_threshold).astype(int)).ravel()[3]:,}
  False Positives (False alarms): {confusion_matrix(y_test, (best_proba >= optimal_threshold).astype(int)).ravel()[1]:,}
  False Negatives (Missed): {confusion_matrix(y_test, (best_proba >= optimal_threshold).astype(int)).ravel()[2]:,}

TOP 3 PREDICTORS:
  1. {rf_importance.iloc[0]['feature']}: {rf_importance.iloc[0]['importance']:.4f}
  2. {rf_importance.iloc[1]['feature']}: {rf_importance.iloc[1]['importance']:.4f}
  3. {rf_importance.iloc[2]['feature']}: {rf_importance.iloc[2]['importance']:.4f}

FILES SAVED:
  - models/logistic_regression.pkl
  - models/random_forest_200.pkl
  - models/best_model.pkl
  - models/scaler.pkl
  - models/risk_threshold.txt
  - metrics/model_metrics.json
  - metrics/rf_feature_importance.csv
  - metrics/plots/confusion_matrix_roc.png
  - metrics/plots/precision_recall_curve.png


""")


print(" Model Trained Successfully ...")
