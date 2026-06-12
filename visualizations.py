"""
Umfassende Visualisierungserstellung
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("VISUALISIERUNGEN ERSTELLEN")
print("="*80)

# Ergebnisse laden
with open('pipeline_results.pkl', 'rb') as f:
    results = pickle.load(f)

with open('evaluation_results.pkl', 'rb') as f:
    eval_results = pickle.load(f)

y_test = results['y_test']
y_train = results['y_train']
y_test_pred_ridge = results['y_test_pred_ridge']
y_test_pred_xgb = results['y_test_pred_xgb']
y_test_pred_nn = results['y_test_pred_nn']
test_df = results['test_df']
df = results['df']
history = results['history']
gb_model = results['gb_model']

xgb_intervals = eval_results['xgb_intervals']
nn_intervals = eval_results['nn_intervals']
feature_importance = pd.DataFrame(eval_results['feature_importance'])

# ============================================================================
# 1. DATENERKUNDUNGSVISUALISIERUNGEN
# ============================================================================
print("\n[1] Erstelle Visualisierungen zur Datenerkundung...")

fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 3, figure=fig)

# 1.1 Distribution of baseRent
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(y_train, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
ax1.set_xlabel('Base Rent (€)')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of Base Rent (Training Data)')
ax1.axvline(y_train.mean(), color='red', linestyle='--', label=f'Mean: €{y_train.mean():.0f}')
ax1.legend()

# 1.2 Boxplot
ax2 = fig.add_subplot(gs[0, 1])
ax2.boxplot([y_train], labels=['Base Rent'])
ax2.set_ylabel('Rent (€)')
ax2.set_title('Box Plot of Base Rent')
ax2.grid(True, alpha=0.3)

# 1.3 Living Space vs Rent
ax3 = fig.add_subplot(gs[0, 2])
if 'livingSpace' in test_df.columns:
    ax3.scatter(test_df['livingSpace'], test_df['baseRent'], alpha=0.5, s=20)
    ax3.set_xlabel('Living Space (m²)')
    ax3.set_ylabel('Base Rent (€)')
    ax3.set_title('Living Space vs Rent (Test Data)')
    ax3.grid(True, alpha=0.3)

# 1.4 Merkmalskorrelationen (top features)
ax4 = fig.add_subplot(gs[1, 0])
if 'livingSpace' in results['X_train'].columns:
    sample_features = results['X_train'].iloc[:500, :15]
    corr_with_target = pd.concat([sample_features, y_train.iloc[:500]], axis=1).corr().iloc[:-1, -1].sort_values(ascending=False)
    corr_with_target.head(10).plot(kind='barh', ax=ax4, color='coral')
    ax4.set_xlabel('Correlation with Base Rent')
    ax4.set_title('Top 10 Merkmalskorrelationen')

# 1.5 Rooms vs Rent
ax5 = fig.add_subplot(gs[1, 1])
if 'noRooms' in test_df.columns:
    room_stats = test_df.groupby('noRooms')['baseRent'].agg(['mean', 'count'])
    room_stats = room_stats[room_stats['count'] > 5].head(8)
    ax5.bar(room_stats.index.astype(str), room_stats['mean'], color='lightgreen', edgecolor='black')
    ax5.set_xlabel('Number of Rooms')
    ax5.set_ylabel('Average Rent (€)')
    ax5.set_title('Average Rent by Room Count')
    ax5.grid(True, alpha=0.3, axis='y')

# 1.6 Floor vs Rent
ax6 = fig.add_subplot(gs[1, 2])
if 'floor' in test_df.columns:
    floor_stats = test_df.groupby('floor')['baseRent'].agg(['mean', 'count'])
    floor_stats = floor_stats[floor_stats['count'] > 5].head(10)
    ax6.plot(floor_stats.index, floor_stats['mean'], marker='o', color='purple', linewidth=2)
    ax6.fill_between(floor_stats.index, floor_stats['mean'] - floor_stats['mean'].std(), 
                     floor_stats['mean'] + floor_stats['mean'].std(), alpha=0.2, color='purple')
    ax6.set_xlabel('Floor Level')
    ax6.set_ylabel('Average Rent (€)')
    ax6.set_title('Average Rent by Floor Level')
    ax6.grid(True, alpha=0.3)

# 1.7 Year Constructed
ax7 = fig.add_subplot(gs[2, 0])
if 'yearConstructed' in test_df.columns:
    year_stats = test_df[test_df['yearConstructed'] > 1900].groupby(pd.cut(test_df['yearConstructed'], bins=15))['baseRent'].mean()
    ax7.bar(range(len(year_stats)), year_stats.values, color='teal', edgecolor='black')
    ax7.set_xlabel('Construction Period')
    ax7.set_ylabel('Average Rent (€)')
    ax7.set_title('Rent by Construction Period')
    ax7.set_xticklabels([f'{i}' for i in range(len(year_stats))], rotation=45)

# 1.8 Top regions by count
ax8 = fig.add_subplot(gs[2, 1])
top_regions = test_df['regio1'].value_counts().head(10)
ax8.barh(range(len(top_regions)), top_regions.values, color='orange', edgecolor='black')
ax8.set_yticks(range(len(top_regions)))
ax8.set_yticklabels(top_regions.index, fontsize=9)
ax8.set_xlabel('Number of Listings')
ax8.set_title('Top 10 Regions by Listing Count')
ax8.grid(True, alpha=0.3, axis='x')

# 1.9 Average rent by region
ax9 = fig.add_subplot(gs[2, 2])
avg_rent_regions = test_df.groupby('regio1')['baseRent'].mean().sort_values(ascending=False).head(10)
ax9.barh(range(len(avg_rent_regions)), avg_rent_regions.values, color='salmon', edgecolor='black')
ax9.set_yticks(range(len(avg_rent_regions)))
ax9.set_yticklabels(avg_rent_regions.index, fontsize=9)
ax9.set_xlabel('Average Rent (€)')
ax9.set_title('Top 10 teuerste Regionen')
ax9.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('01_data_exploration.png', dpi=300, bbox_inches='tight')
print("   ✓ Gespeichert: 01_data_exploration.png")
plt.close()

# ============================================================================
# 2. MODELLLEISTUNGSVERGLEICH
# ============================================================================
print("\n[2] Erstelle Visualisierungen zur Modellleistung...")

fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 3, figure=fig)

models = ['Ridge', 'Gradient-Boosting-Regressor', 'Neuronales Netz']
predictions = [y_test_pred_ridge, y_test_pred_xgb, y_test_pred_nn]
colors = ['#3498db', '#e74c3c', '#2ecc71']

# 2.1-2.3 Vorhersagen vs. Tatsächlich for each model
for idx, (model, pred, color) in enumerate(zip(models, predictions, colors)):
    ax = fig.add_subplot(gs[0, idx])
    ax.scatter(y_test, pred, alpha=0.5, s=20, color=color, edgecolors='black', linewidth=0.5)
    
    # Perfekte Vorhersagegerade
    min_val = min(y_test.min(), pred.min())
    max_val = max(y_test.max(), pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, linewidth=2)
    
    ax.set_xlabel('Actual Rent (€)')
    ax.set_ylabel('Predicted Rent (€)')
    ax.set_title(f'{model}: Vorhersagen vs. Tatsächlich')
    ax.grid(True, alpha=0.3)
    
    # Add R² score
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, pred)
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# 2.4 Residuen comparison
ax4 = fig.add_subplot(gs[1, 0])
residuals_ridge = y_test - y_test_pred_ridge
residuals_xgb = y_test - y_test_pred_xgb
residuals_nn = y_test - y_test_pred_nn

ax4.violinplot([residuals_ridge, residuals_xgb, residuals_nn], positions=[1, 2, 3], showmeans=True)
ax4.set_xticks([1, 2, 3])
ax4.set_xticklabels(['Ridge', 'Gradient-Boosting-Regressor', 'NN'])
ax4.set_ylabel('Residuen (€)')
ax4.set_title('Residuenverteilung nach Modell')
ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax4.grid(True, alpha=0.3, axis='y')

# 2.5 MAE Comparison
ax5 = fig.add_subplot(gs[1, 1])
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae_vals = [
    mean_absolute_error(y_test, y_test_pred_ridge),
    mean_absolute_error(y_test, y_test_pred_xgb),
    mean_absolute_error(y_test, y_test_pred_nn)
]
rmse_vals = [
    np.sqrt(mean_squared_error(y_test, y_test_pred_ridge)),
    np.sqrt(mean_squared_error(y_test, y_test_pred_xgb)),
    np.sqrt(mean_squared_error(y_test, y_test_pred_nn))
]

x_pos = np.arange(len(models))
width = 0.35
ax5.bar(x_pos - width/2, mae_vals, width, label='MAE', color='skyblue', edgecolor='black')
ax5.bar(x_pos + width/2, rmse_vals, width, label='RMSE', color='coral', edgecolor='black')
ax5.set_ylabel('Error (€)')
ax5.set_title('MAE vs. RMSE nach Modell')
ax5.set_xticks(x_pos)
ax5.set_xticklabels(models)
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# 2.6 Neuronales Netz Training History
ax6 = fig.add_subplot(gs[1, 2])
ax6.plot(history.history['loss'], label='Training Loss', color='#3498db', linewidth=2)
ax6.plot(history.history['val_loss'], label='Validation Loss', color='#e74c3c', linewidth=2)
ax6.set_xlabel('Epoch')
ax6.set_ylabel('Loss (MSE)')
ax6.set_title('Neuronales Netz Training History')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('02_model_performance.png', dpi=300, bbox_inches='tight')
print("   ✓ Gespeichert: 02_model_performance.png")
plt.close()

# ============================================================================
# 3. PREDICTION INTERVALS & UNCERTAINTY
# ============================================================================
print("\n[3] Erstelle Visualisierungen zur Unsicherheitsanalyse...")

fig = plt.figure(figsize=(16, 6))
gs = GridSpec(1, 2, figure=fig)

# 3.1 Gradient-Boosting-Regressor prediction intervals
ax1 = fig.add_subplot(gs[0, 0])
sorted_idx = np.argsort(y_test.values)
x_axis = np.arange(len(sorted_idx))[:500]  # First 500 for clarity

ax1.scatter(x_axis, y_test.values[sorted_idx[:500]], alpha=0.6, s=20, color='black', label='Actual', zorder=3)
ax1.plot(x_axis, y_test_pred_xgb[sorted_idx[:500]], color='#e74c3c', linewidth=2, label='Gradient-Boosting-Regressor Prediction', zorder=2)
ax1.fill_between(x_axis, 
                xgb_intervals['lower'][sorted_idx[:500]],
                xgb_intervals['upper'][sorted_idx[:500]],
                alpha=0.3, color='#e74c3c', label='90% Prediction Interval', zorder=1)
ax1.set_xlabel('Sample Index (sorted by actual rent)')
ax1.set_ylabel('Rent (€)')
ax1.set_title(f"Gradient-Boosting-Regressor: Predictions with 90% Intervals\n(Coverage: {xgb_intervals['coverage']:.1%})")
ax1.legend()
ax1.grid(True, alpha=0.3)

# 3.2 NN prediction intervals
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(x_axis, y_test.values[sorted_idx[:500]], alpha=0.6, s=20, color='black', label='Actual', zorder=3)
ax2.plot(x_axis, y_test_pred_nn[sorted_idx[:500]], color='#2ecc71', linewidth=2, label='NN Prediction', zorder=2)
ax2.fill_between(x_axis,
                nn_intervals['lower'][sorted_idx[:500]],
                nn_intervals['upper'][sorted_idx[:500]],
                alpha=0.3, color='#2ecc71', label='90% Prediction Interval', zorder=1)
ax2.set_xlabel('Sample Index (sorted by actual rent)')
ax2.set_ylabel('Rent (€)')
ax2.set_title(f"Neuronales Netz: Predictions with 90% Intervals\n(Coverage: {nn_intervals['coverage']:.1%})")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('03_uncertainty_analysis.png', dpi=300, bbox_inches='tight')
print("   ✓ Gespeichert: 03_uncertainty_analysis.png")
plt.close()

# ============================================================================
# 4. MERKMALSWICHTIGKEIT
# ============================================================================
print("\n[4] Erstelle Visualisierungen zur Merkmalswichtigkeit...")

fig, axes = plt.subplots(1, 1, figsize=(12, 8))

top_features = feature_importance.head(20).sort_values('importance')
colors_grad = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
axes.barh(range(len(top_features)), top_features['importance'].values, color=colors_grad, edgecolor='black')
axes.set_yticks(range(len(top_features)))
axes.set_yticklabels(top_features['feature'].values)
axes.set_xlabel('Importance Score')
axes.set_title('Gradient-Boosting-Regressor: Top 20 Merkmalswichtigkeit')
axes.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('04_feature_importance.png', dpi=300, bbox_inches='tight')
print("   ✓ Gespeichert: 04_feature_importance.png")
plt.close()

# ============================================================================
# 5. RÄUMLICHE ANALYSE
# ============================================================================
print("\n[5] Erstelle Visualisierungen zur räumlichen Analyse...")

fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 2, figure=fig)

# 5.1 Average rent by region
ax1 = fig.add_subplot(gs[0, :])
region_avg = test_df.groupby('regio1')['baseRent'].agg(['mean', 'count']).sort_values('mean', ascending=False)
region_avg = region_avg[region_avg['count'] > 3]  # Filter out regions with few samples
colors_spatial = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(region_avg)))
bars = ax1.barh(range(len(region_avg)), region_avg['mean'].values, color=colors_spatial, edgecolor='black')
ax1.set_yticks(range(len(region_avg)))
ax1.set_yticklabels(region_avg.index, fontsize=9)
ax1.set_xlabel('Average Base Rent (€)')
ax1.set_title('Durchschnittliche Mietpreise nach Region')
ax1.grid(True, alpha=0.3, axis='x')

# 5.2 Price distribution by top regions
ax2 = fig.add_subplot(gs[1, 0])
top_5_regions = test_df.groupby('regio1')['baseRent'].mean().nlargest(5).index.tolist()
region_data = [test_df[test_df['regio1'] == r]['baseRent'].values for r in top_5_regions]
bp = ax2.boxplot(region_data, labels=top_5_regions, patch_artist=True)
for patch, color in zip(bp['boxes'], plt.cm.Set3(range(len(top_5_regions)))):
    patch.set_facecolor(color)
ax2.set_ylabel('Base Rent (€)')
ax2.set_title('Price Distribution in Top 5 Most Teuer Regions')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, alpha=0.3, axis='y')

# 5.3 Cheapest regions
ax3 = fig.add_subplot(gs[1, 1])
bottom_5_regions = test_df.groupby('regio1')['baseRent'].mean().nsmallest(5).index.tolist()
region_data_cheap = [test_df[test_df['regio1'] == r]['baseRent'].values for r in bottom_5_regions]
bp2 = ax3.boxplot(region_data_cheap, labels=bottom_5_regions, patch_artist=True)
for patch, color in zip(bp2['boxes'], plt.cm.Set2(range(len(bottom_5_regions)))):
    patch.set_facecolor(color)
ax3.set_ylabel('Base Rent (€)')
ax3.set_title('Preisspannen in den 5 günstigsten Regionen')
ax3.tick_params(axis='x', rotation=45)
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('05_spatial_analysis.png', dpi=300, bbox_inches='tight')
print("   ✓ Gespeichert: 05_spatial_analysis.png")
plt.close()

# ============================================================================
# 6. ZEITLICHE ANALYSE
# ============================================================================
print("\n[6] Erstelle Visualisierungen zur zeitlichen Analyse...")

fig = plt.figure(figsize=(16, 8))
gs = GridSpec(2, 2, figure=fig)

if 'year' in test_df.columns and 'month' in test_df.columns:
    # 6.1 Average rent by month
    ax1 = fig.add_subplot(gs[0, 0])
    month_avg = test_df.groupby('month')['baseRent'].agg(['mean', 'std'])
    ax1.plot(month_avg.index, month_avg['mean'], marker='o', linewidth=3, color='#3498db', markersize=8)
    ax1.fill_between(month_avg.index, 
                    month_avg['mean'] - month_avg['std'],
                    month_avg['mean'] + month_avg['std'],
                    alpha=0.3, color='#3498db')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Average Base Rent (€)')
    ax1.set_title('Saisonale Preisentwicklung (Monatsdurchschnitt)')
    ax1.set_xticks(range(1, 13))
    ax1.grid(True, alpha=0.3)
    
    # 6.2 Average rent by quarter
    ax2 = fig.add_subplot(gs[0, 1])
    quarter_avg = test_df.groupby('quarter')['baseRent'].mean()
    colors_quarter = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    ax2.bar(quarter_avg.index, quarter_avg.values, color=colors_quarter, edgecolor='black', width=0.6)
    ax2.set_xlabel('Quarter')
    ax2.set_ylabel('Average Base Rent (€)')
    ax2.set_title('Durchschnittliche Miete nach Quartal')
    ax2.set_xticks([1, 2, 3, 4])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 6.3 Rent trend by year and major regions
    ax3 = fig.add_subplot(gs[1, :])
    top_regions_temporal = test_df.groupby('regio1')['baseRent'].count().nlargest(5).index.tolist()
    for region in top_regions_temporal:
        region_temporal = test_df[test_df['regio1'] == region].groupby('year')['baseRent'].mean()
        ax3.plot(region_temporal.index, region_temporal.values, marker='o', label=region, linewidth=2, markersize=8)
    
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Average Base Rent (€)')
    ax3.set_title('Preisentwicklung nach Jahr (Top 5 Regionen)')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
else:
    ax_temp = fig.add_subplot(gs[:, :])
    ax_temp.text(0.5, 0.5, 'Insufficient temporal data', ha='center', va='center', fontsize=16)
    ax_temp.axis('off')

plt.tight_layout()
plt.savefig('06_temporal_analysis.png', dpi=300, bbox_inches='tight')
print("   ✓ Gespeichert: 06_temporal_analysis.png")
plt.close()

# ============================================================================
# 7. RESIDUAL ANALYSIS
# ============================================================================
print("\n[7] Erstelle Visualisierungen zur Residuenanalyse...")

fig = plt.figure(figsize=(16, 6))
gs = GridSpec(1, 3, figure=fig)

residuals_xgb = y_test.values - y_test_pred_xgb

# 7.1 Residuen vs Predicted
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_test_pred_xgb, residuals_xgb, alpha=0.5, s=20, color='#e74c3c', edgecolors='black', linewidth=0.5)
ax1.axhline(y=0, color='black', linestyle='--', linewidth=2)
ax1.set_xlabel('Predicted Rent (€)')
ax1.set_ylabel('Residuen (€)')
ax1.set_title('Residuen vs Predicted Values (Gradient-Boosting-Regressor)')
ax1.grid(True, alpha=0.3)

# 7.2 Residuen distribution
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(residuals_xgb, bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
ax2.axvline(residuals_xgb.mean(), color='black', linestyle='--', linewidth=2, label=f'Mean: €{residuals_xgb.mean():.0f}')
ax2.set_xlabel('Residuen (€)')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of Residuen (Gradient-Boosting-Regressor)')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# 7.3 Q-Q Plot
from scipy import stats
ax3 = fig.add_subplot(gs[0, 2])
stats.probplot(residuals_xgb, dist="norm", plot=ax3)
ax3.set_title('Q-Q Plot (Gradient-Boosting-Regressor Residuen)')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('07_residual_analysis.png', dpi=300, bbox_inches='tight')
print("   ✓ Gespeichert: 07_residual_analysis.png")
plt.close()

print("\n" + "="*80)
print("ALLE VISUALISIERUNGEN ABGESCHLOSSEN!")
print("="*80)
print("\nErzeugte Dateien:")
print("  - 01_data_exploration.png")
print("  - 02_model_performance.png")
print("  - 03_uncertainty_analysis.png")
print("  - 04_feature_importance.png")
print("  - 05_spatial_analysis.png")
print("  - 06_temporal_analysis.png")
print("  - 07_residual_analysis.png")
print("\nNächste Schritte:")
print("  1. Ausführen: python3 generate_maps.py")
print("  2. Ausführen: python3 generate_report.py")
