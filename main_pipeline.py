"""
Rental Price Prediction - Complete ML Pipeline
German real estate dataset analysis with spatial and temporal modeling
"""

import pandas as pd
import numpy as np
import warnings
import pickle
import json
from datetime import datetime

# ML & preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor

# Neural network
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

print("="*80)
print("RENTAL PRICE PREDICTION PIPELINE - Starting")
print("="*80)

# ============================================================================
# PHASE 1: DATA LOADING & EXPLORATION
# ============================================================================
print("\n[PHASE 1] Loading and exploring data...")

# Load data (sample 50K for faster iteration)
df = pd.read_csv('immo_data.csv', nrows=50000)
print(f"Dataset shape: {df.shape}")
print(f"Missing data percentage:\n{(df.isnull().sum() / len(df) * 100).sort_values(ascending=False).head(10)}")

# ============================================================================
# PHASE 2: FEATURE ENGINEERING & DATA CLEANING
# ============================================================================
print("\n[PHASE 2] Feature engineering and data cleaning...")

# Remove rows with missing target
df = df.dropna(subset=['baseRent'])
print(f"After removing null targets: {len(df)} rows")

# Remove outliers (extreme rent values)
Q1 = df['baseRent'].quantile(0.01)
Q3 = df['baseRent'].quantile(0.99)
df = df[(df['baseRent'] >= Q1) & (df['baseRent'] <= Q3)]
print(f"After outlier removal: {len(df)} rows")

# Parse temporal features
df['date'] = pd.to_datetime(df['date'], format='%b%y', errors='coerce')
df = df.dropna(subset=['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter
df['season'] = df['month'].map({12: 'Winter', 1: 'Winter', 2: 'Winter',
                                 3: 'Spring', 4: 'Spring', 5: 'Spring',
                                 6: 'Summer', 7: 'Summer', 8: 'Summer',
                                 9: 'Fall', 10: 'Fall', 11: 'Fall'})

print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Rows after date parsing: {len(df)}")

# Select features for modeling
features_to_keep = [
    'baseRent', 'livingSpace', 'noRooms', 'floor', 'numberOfFloors',
    'yearConstructed', 'serviceCharge', 'balcony', 'garden', 'cellar', 'lift',
    'newlyConst', 'hasKitchen', 'noParkSpaces', 'picturecount', 'pricetrend',
    'regio1', 'regio2', 'geo_plz', 'condition', 'heatingType', 'typeOfFlat',
    'year', 'month', 'quarter', 'season'
]

df = df[[f for f in features_to_keep if f in df.columns]].copy()
print(f"Features selected: {df.shape[1]}")

# Impute numeric features (fill with median)
numeric_cols = df.select_dtypes(include=[np.number]).columns
imputer = SimpleImputer(strategy='median')
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# Encode categorical variables
label_encoders = {}
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"Encoded {col}: {len(le.classes_)} classes")

print(f"Final dataset shape: {df.shape}")

# ============================================================================
# PHASE 3: DATA SPLITTING (TEMPORAL STRATEGY)
# ============================================================================
print("\n[PHASE 3] Splitting data (temporal strategy)...")

# Split by year for realistic scenarios
train_df = df[df['year'] == 2019].copy()
val_df = df[df['year'] == 2020].copy()

if len(val_df) < 100:
    # If 2020 is small, use stratified split
    train_df, val_df = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df['quarter']
    )

# Further split validation into val and test
val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=42)

X_train = train_df.drop('baseRent', axis=1)
y_train = train_df['baseRent']
X_val = val_df.drop('baseRent', axis=1)
y_val = val_df['baseRent']
X_test = test_df.drop('baseRent', axis=1)
y_test = test_df['baseRent']

print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# PHASE 4: BASELINE MODEL (RIDGE REGRESSION)
# ============================================================================
print("\n[PHASE 4] Training baseline model (Ridge Regression)...")

ridge_model = Ridge(alpha=1.0, random_state=42)
ridge_model.fit(X_train_scaled, y_train)

y_train_pred_ridge = ridge_model.predict(X_train_scaled)
y_val_pred_ridge = ridge_model.predict(X_val_scaled)
y_test_pred_ridge = ridge_model.predict(X_test_scaled)

print("Ridge model trained ✓")

# ============================================================================
# PHASE 5: GRADIENT BOOSTING MODEL (SKLEARN)
# ============================================================================
print("\n[PHASE 5] Training Gradient Boosting model...")

gb_model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42,
    n_iter_no_change=10,
    validation_fraction=0.1,
    verbose=0
)
gb_model.fit(X_train, y_train)

y_train_pred_xgb = gb_model.predict(X_train)
y_val_pred_xgb = gb_model.predict(X_val)
y_test_pred_xgb = gb_model.predict(X_test)

print("Gradient Boosting model trained ✓")

# ============================================================================
# PHASE 6: NEURAL NETWORK MODEL
# ============================================================================
print("\n[PHASE 6] Training Neural Network...")

def create_nn_model(input_shape):
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(input_shape,),
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

nn_model = create_nn_model(X_train_scaled.shape[1])
history = nn_model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=100,
    batch_size=32,
    verbose=0
)

y_train_pred_nn = nn_model.predict(X_train_scaled, verbose=0).flatten()
y_val_pred_nn = nn_model.predict(X_val_scaled, verbose=0).flatten()
y_test_pred_nn = nn_model.predict(X_test_scaled, verbose=0).flatten()

print("Neural Network model trained ✓")

# ============================================================================
# PHASE 7: SAVE MODELS & METADATA
# ============================================================================
print("\n[PHASE 7] Saving models and metadata...")

# Save models
with open('gb_model.pkl', 'wb') as f:
    pickle.dump(gb_model, f)
nn_model.save('nn_model.h5')
ridge_model_path = 'ridge_model.pkl'
with open(ridge_model_path, 'wb') as f:
    pickle.dump(ridge_model, f)

# Save scalers and encoders
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
with open('imputer.pkl', 'wb') as f:
    pickle.dump(imputer, f)

# Save metadata
metadata = {
    'feature_names': X_train.columns.tolist(),
    'n_features': X_train.shape[1],
    'n_train': X_train.shape[0],
    'n_val': X_val.shape[0],
    'n_test': X_test.shape[0],
    'target_mean_train': float(y_train.mean()),
    'target_std_train': float(y_train.std()),
    'created_at': datetime.now().isoformat()
}
with open('metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("Models and metadata saved ✓")

# ============================================================================
# SAVE FOR EVALUATION
# ============================================================================
print("\n[PHASE 8] Saving results for evaluation...")

results = {
    'X_train': X_train, 'y_train': y_train,
    'X_val': X_val, 'y_val': y_val,
    'X_test': X_test, 'y_test': y_test,
    'X_train_scaled': X_train_scaled, 'X_val_scaled': X_val_scaled, 'X_test_scaled': X_test_scaled,
    'y_train_pred_ridge': y_train_pred_ridge, 'y_val_pred_ridge': y_val_pred_ridge, 'y_test_pred_ridge': y_test_pred_ridge,
    'y_train_pred_xgb': y_train_pred_xgb, 'y_val_pred_xgb': y_val_pred_xgb, 'y_test_pred_xgb': y_test_pred_xgb,
    'y_train_pred_nn': y_train_pred_nn, 'y_val_pred_nn': y_val_pred_nn, 'y_test_pred_nn': y_test_pred_nn,
    'train_df': train_df, 'val_df': val_df, 'test_df': test_df,
    'df': df,
    'gb_model': gb_model,
    'nn_model': nn_model,
    'ridge_model': ridge_model,
    'history': history
}

with open('pipeline_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("Results saved ✓")
print("\n" + "="*80)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print("="*80)
print("\nNext steps:")
print("  1. Run: python3 evaluate_models.py")
print("  2. Run: python3 visualizations.py")
print("  3. Run: python3 generate_maps.py")
print("  4. Run: python3 generate_report.py")
