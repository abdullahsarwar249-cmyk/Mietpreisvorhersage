"""
Modellbewertung and Uncertainty Analysis
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

warnings.filterwarnings('ignore')

print("="*80)
print("MODELLBEWERTUNG & UNSICHERHEITSANALYSE")
print("="*80)

# Ergebnisse laden
with open('pipeline_results.pkl', 'rb') as f:
    results = pickle.load(f)

X_train = results['X_train']
y_train = results['y_train']
X_val = results['X_val']
y_val = results['y_val']
X_test = results['X_test']
y_test = results['y_test']

y_train_pred_ridge = results['y_train_pred_ridge']
y_val_pred_ridge = results['y_val_pred_ridge']
y_test_pred_ridge = results['y_test_pred_ridge']

y_train_pred_xgb = results['y_train_pred_xgb']
y_val_pred_xgb = results['y_val_pred_xgb']
y_test_pred_xgb = results['y_test_pred_xgb']

y_train_pred_nn = results['y_train_pred_nn']
y_val_pred_nn = results['y_val_pred_nn']
y_test_pred_nn = results['y_test_pred_nn']

gb_model = results['gb_model']
test_df = results['test_df']

# ============================================================================
# CALCULATE METRICS
# ============================================================================
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {'MAE': mae, 'RMSE': rmse, 'R²': r2, 'MAPE': mape}

print("\n[1] RIDGE REGRESSION METRICS")
print("-" * 80)
ridge_metrics = {
    'Train': calculate_metrics(y_train, y_train_pred_ridge),
    'Validation': calculate_metrics(y_val, y_val_pred_ridge),
    'Test': calculate_metrics(y_test, y_test_pred_ridge)
}
for split, metrics in ridge_metrics.items():
    print(f"{split:12} | MAE: {metrics['MAE']:8.2f} | RMSE: {metrics['RMSE']:8.2f} | R²: {metrics['R²']:6.4f} | MAPE: {metrics['MAPE']:7.4f}%")

print("\n[2] Gradient Boosting METRICS")
print("-" * 80)
xgb_metrics = {
    'Train': calculate_metrics(y_train, y_train_pred_xgb),
    'Validation': calculate_metrics(y_val, y_val_pred_xgb),
    'Test': calculate_metrics(y_test, y_test_pred_xgb)
}
for split, metrics in xgb_metrics.items():
    print(f"{split:12} | MAE: {metrics['MAE']:8.2f} | RMSE: {metrics['RMSE']:8.2f} | R²: {metrics['R²']:6.4f} | MAPE: {metrics['MAPE']:7.4f}%")

print("\n[3] NEURAL NETWORK METRICS")
print("-" * 80)
nn_metrics = {
    'Train': calculate_metrics(y_train, y_train_pred_nn),
    'Validation': calculate_metrics(y_val, y_val_pred_nn),
    'Test': calculate_metrics(y_test, y_test_pred_nn)
}
for split, metrics in nn_metrics.items():
    print(f"{split:12} | MAE: {metrics['MAE']:8.2f} | RMSE: {metrics['RMSE']:8.2f} | R²: {metrics['R²']:6.4f} | MAPE: {metrics['MAPE']:7.4f}%")

# ============================================================================
# UNSICHERHEITSANALYSE - PREDICTION INTERVALS
# ============================================================================
print("\n[4] UNSICHERHEITSANALYSE - PREDICTION INTERVALS")
print("-" * 80)

def calculate_prediction_intervals(y_true, y_pred, confidence=0.9):
    """Calculate prediction intervals based on residuals"""
    residuals = np.abs(y_true - y_pred)
    std_residuals = np.std(residuals)
    z_score = 1.645  # 90% confidence interval
    
    lower = y_pred - z_score * std_residuals
    upper = y_pred + z_score * std_residuals
    
    # Check coverage
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    interval_width = np.mean(upper - lower)
    
    return {
        'lower': lower, 'upper': upper, 'coverage': coverage,
        'interval_width': interval_width, 'std_residuals': std_residuals
    }

xgb_intervals = calculate_prediction_intervals(y_test, y_test_pred_xgb)
nn_intervals = calculate_prediction_intervals(y_test, y_test_pred_nn)

print(f"\nGradient-Boosting-Regressor Test Set:")
print(f"  Prediction interval coverage: {xgb_intervals['coverage']:.2%} (target: 90%)")
print(f"  Average interval width: €{xgb_intervals['interval_width']:.2f}")
print(f"  Std of residuals: €{xgb_intervals['std_residuals']:.2f}")

print(f"\nNeuronales Netz Test Set:")
print(f"  Prediction interval coverage: {nn_intervals['coverage']:.2%} (target: 90%)")
print(f"  Average interval width: €{nn_intervals['interval_width']:.2f}")
print(f"  Std of residuals: €{nn_intervals['std_residuals']:.2f}")

# ============================================================================
# MERKMALSWICHTIGKEIT (Gradient Boosting)
# ============================================================================
print("\n[5] MERKMALSWICHTIGKEIT (Gradient Boosting)")
print("-" * 80)
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': gb_model.feature_importances_
}).sort_values('importance', ascending=False).head(15)

for idx, row in feature_importance.iterrows():
    print(f"  {row['feature']:20} | {row['importance']:8.4f}")

# ============================================================================
# RÄUMLICHE ANALYSE
# ============================================================================
print("\n[6] RÄUMLICHE ANALYSE - AVERAGE RENT BY REGION")
print("-" * 80)

# Reconstruct original regions from test set
test_data = test_df.copy()
test_data['prediction_xgb'] = y_test_pred_xgb
test_data['prediction_nn'] = y_test_pred_nn
test_data['actual'] = y_test.values

# Group by regio1
regional_stats = test_data.groupby('regio1').agg({
    'baseRent': ['mean', 'std', 'count'],
    'prediction_xgb': 'mean',
    'prediction_nn': 'mean'
}).round(2)

print("\nTop 10 teuerste Regionen (nach durchschnittlicher Miete):")
regio_avg = test_data.groupby('regio1')['baseRent'].mean().sort_values(ascending=False).head(10)
for region, rent in regio_avg.items():
    count = len(test_data[test_data['regio1'] == region])
    print(f"  {region:30} | €{rent:8.2f} | n={count:4d}")

print("\nGünstigste Regionen (nach Durchschnittsmiete):")
regio_min = test_data.groupby('regio1')['baseRent'].mean().sort_values(ascending=True).head(10)
for region, rent in regio_min.items():
    count = len(test_data[test_data['regio1'] == region])
    print(f"  {region:30} | €{rent:8.2f} | n={count:4d}")

# ============================================================================
# ZEITLICHE ANALYSE
# ============================================================================
print("\n[7] ZEITLICHE ANALYSE - PRICE TRENDS")
print("-" * 80)

if 'year' in test_data.columns and 'month' in test_data.columns:
    temporal_stats = test_data.groupby(['year', 'month']).agg({
        'baseRent': ['mean', 'count']
    }).round(2)
    print("\nAverage rent by year-month:")
    print(temporal_stats)

# ============================================================================
# SAVE EVALUATION RESULTS
# ============================================================================
eval_results = {
    'ridge_metrics': ridge_metrics,
    'xgb_metrics': xgb_metrics,
    'nn_metrics': nn_metrics,
    'xgb_intervals': xgb_intervals,
    'nn_intervals': nn_intervals,
    'feature_importance': feature_importance.to_dict('records'),
    'test_data': test_data
}

with open('evaluation_results.pkl', 'wb') as f:
    pickle.dump(eval_results, f)

print("\n" + "="*80)
print("EVALUATION COMPLETED!")
print("="*80)
print("\nResults saved to: evaluation_results.pkl")
print("\nNächste Schritte:")
print("  1. Ausführen: python3 visualizations.py")
print("  2. Ausführen: python3 generate_maps.py")
print("  3. Run: python3 generate_report.py")
