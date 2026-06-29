"""
Mietpreisvorhersage – Vollständige ML-Pipeline
Analyse deutscher Immobiliendaten mit räumlicher und zeitlicher Modellierung
"""

import pandas as pd
import numpy as np
import warnings
import pickle
import json
from datetime import datetime

# Maschinelles Lernen & Vorverarbeitung
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor

# Neuronales Netz
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 80)
print("MIETPREISVORHERSAGE")
print("=" * 80)

# ============================================================================
# SCHRITT 1: DATEN LADEN UND ANALYSIEREN
# ============================================================================
print("\n[SCHRITT 1] Lade und analysiere Daten...")

# Daten laden (50.000 Zeilen für schnellere Verarbeitung)
df = pd.read_csv('immo_data.csv', nrows=50000)

print(f"Datensatzgröße: {df.shape}")
print(
    f"Anteil fehlender Werte (%):\n"
    f"{(df.isnull().sum() / len(df) * 100).sort_values(ascending=False).head(10)}"
)

# ============================================================================
# SCHRITT 2: MERKMALSERZEUGUNG UND DATENBEREINIGUNG
# ============================================================================
print("\n[SCHRITT 2] Merkmalserzeugung und Datenbereinigung...")

# Datensätze ohne Zielvariable entfernen
df = df.dropna(subset=['baseRent'])
print(f"Nach Entfernen fehlender Zielwerte: {len(df)} Zeilen")

# Ausreißer entfernen (1%-99%-Quantile)
Q1 = df['baseRent'].quantile(0.01)
Q3 = df['baseRent'].quantile(0.99)

df = df[(df['baseRent'] >= Q1) & (df['baseRent'] <= Q3)]
print(f"Nach Entfernen von Ausreißern: {len(df)} Zeilen")

# Zeitliche Merkmale erzeugen
df['date'] = pd.to_datetime(df['date'], format='%b%y', errors='coerce')
df = df.dropna(subset=['date'])

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter

df['season'] = df['month'].map({
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Frühling', 4: 'Frühling', 5: 'Frühling',
    6: 'Sommer', 7: 'Sommer', 8: 'Sommer',
    9: 'Herbst', 10: 'Herbst', 11: 'Herbst'
})

print(f"Zeitraum: {df['date'].min()} bis {df['date'].max()}")
print(f"Datensätze nach Datumsverarbeitung: {len(df)}")

# Auswahl der Merkmale
features_to_keep = [
    'baseRent',
    'livingSpace',
    'noRooms',
    'floor',
    'numberOfFloors',
    'yearConstructed',
    'serviceCharge',
    'balcony',
    'garden',
    'cellar',
    'lift',
    'newlyConst',
    'hasKitchen',
    'noParkSpaces',
    'picturecount',
    'pricetrend',
    'regio1',
    'regio2',
    'geo_plz',
    'condition',
    'heatingType',
    'typeOfFlat',
    'year',
    'month',
    'quarter',
    'season'
]

df = df[[f for f in features_to_keep if f in df.columns]].copy()

print(f"Ausgewählte Merkmale: {df.shape[1]}")

# Fehlende numerische Werte mit Median ersetzen
numeric_cols = df.select_dtypes(include=[np.number]).columns

imputer = SimpleImputer(strategy='median')
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# Kategorische Variablen kodieren
label_encoders = {}

categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

    print(f"Kodiert: {col} ({len(le.classes_)} Kategorien)")

print(f"Endgültige Datensatzgröße: {df.shape}")

# ============================================================================
# SCHRITT 3: DATENAUFTEILUNG (ZEITLICHE STRATEGIE)
# ============================================================================
print("\n[SCHRITT 3] Teile Daten auf ...")

# Zeitliche Aufteilung für realistische Vorhersageszenarien
#train_df = df[df['year'] == 2019].copy()
#val_df = df[df['year'] == 2020].copy()

train_df = df[df['year'].isin([2018, 2019])].copy()
val_df = df[df['year'] == 2020].copy()

if len(val_df) < 100:
    train_df, val_df = train_test_split(
        df,
        test_size=0.15,
        random_state=42,
        stratify=df['quarter']
    )

# Validierungsdaten in Validierung und Test aufteilen
val_df, test_df = train_test_split(
    val_df,
    test_size=0.5,
    random_state=42
)

X_train = train_df.drop('baseRent', axis=1)
y_train = train_df['baseRent']

X_val = val_df.drop('baseRent', axis=1)
y_val = val_df['baseRent']

X_test = test_df.drop('baseRent', axis=1)
y_test = test_df['baseRent']

print(
    f"Training: {X_train.shape[0]} | "
    f"Validierung: {X_val.shape[0]} | "
    f"Test: {X_test.shape[0]}"
)

# Standardisierung der Merkmale
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# SCHRITT 4: BASISMODELL (RIDGE-REGRESSION)
# ============================================================================
print("\n[SCHRITT 4] Trainiere Basismodell (Ridge-Regression)...")

ridge_model = Ridge(alpha=1.0, random_state=42)
ridge_model.fit(X_train_scaled, y_train)

y_train_pred_ridge = ridge_model.predict(X_train_scaled)
y_val_pred_ridge = ridge_model.predict(X_val_scaled)
y_test_pred_ridge = ridge_model.predict(X_test_scaled)

print("Ridge-Modell erfolgreich trainiert ✓")

# ============================================================================
# SCHRITT 5: GRADIENT-BOOSTING-MODELL
# ============================================================================
print("\n[SCHRITT 5] Trainiere Gradient-Boosting-Modell...")

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

print("Gradient-Boosting-Modell erfolgreich trainiert ✓")

# ============================================================================
# SCHRITT 6: NEURONALES NETZ
# ============================================================================
print("\n[SCHRITT 6] Trainiere neuronales Netz...")


def create_nn_model(input_shape):
    model = keras.Sequential([
        layers.Dense(
            256,
            activation='relu',
            input_shape=(input_shape,),
            kernel_regularizer=regularizers.l2(0.001)
        ),
        layers.Dropout(0.3),

        layers.Dense(
            128,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001)
        ),
        layers.Dropout(0.3),

        layers.Dense(
            64,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001)
        ),
        layers.Dropout(0.2),

        layers.Dense(1)
    ])

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    return model


nn_model = create_nn_model(X_train_scaled.shape[1])

history = nn_model.fit(
    X_train_scaled,
    y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=100,
    batch_size=32,
    verbose=0
)

y_train_pred_nn = nn_model.predict(
    X_train_scaled,
    verbose=0
).flatten()

y_val_pred_nn = nn_model.predict(
    X_val_scaled,
    verbose=0
).flatten()

y_test_pred_nn = nn_model.predict(
    X_test_scaled,
    verbose=0
).flatten()

print("Neuronales Netz erfolgreich trainiert ✓")

# ============================================================================
# SCHRITT 7: MODELLE UND METADATEN SPEICHERN
# ============================================================================
print("\n[SCHRITT 7] Speichere Modelle und Metadaten in den pkl Dateien...")

# Modelle speichern
with open('gb_model.pkl', 'wb') as f:
    pickle.dump(gb_model, f)

nn_model.save('nn_model.h5')

with open('ridge_model.pkl', 'wb') as f:
    pickle.dump(ridge_model, f)

# Vorverarbeitungsobjekte speichern
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)



# Metadaten speichern
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

print("Modelle und Metadaten gespeichert ✓")

# ============================================================================
# SCHRITT 8: ERGEBNISSE FÜR DIE MODELLBEWERTUNG SPEICHERN
# ============================================================================
print("\n[SCHRITT 8] Speichere Ergebnisse für die Modellbewertung...")

results = {
    'X_train': X_train,
    'y_train': y_train,
    'X_val': X_val,
    'y_val': y_val,
    'X_test': X_test,
    'y_test': y_test,

    'X_train_scaled': X_train_scaled,
    'X_val_scaled': X_val_scaled,
    'X_test_scaled': X_test_scaled,

    'y_train_pred_ridge': y_train_pred_ridge,
    'y_val_pred_ridge': y_val_pred_ridge,
    'y_test_pred_ridge': y_test_pred_ridge,

    'y_train_pred_xgb': y_train_pred_xgb,
    'y_val_pred_xgb': y_val_pred_xgb,
    'y_test_pred_xgb': y_test_pred_xgb,

    'y_train_pred_nn': y_train_pred_nn,
    'y_val_pred_nn': y_val_pred_nn,
    'y_test_pred_nn': y_test_pred_nn,

    'train_df': train_df,
    'val_df': val_df,
    'test_df': test_df,

    'df': df,

    'gb_model': gb_model,
    'nn_model': nn_model,
    'ridge_model': ridge_model,

    'history': history
}

with open('pipeline_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("Ergebnisse gespeichert ✓")

print("\n" + "=" * 80)
print(" PIPELINE ERFOLGREICH ABGESCHLOSSEN!")
print("=" * 80)

print("\nNächste Schritte:")
print("  1. Ausführen: python3 evaluate_models.py")
print("  2. Ausführen: python3 visualizations.py")
print("  3. Ausführen: python3 generate_maps.py")
print("  4. Ausführen: python3 generate_report.py")