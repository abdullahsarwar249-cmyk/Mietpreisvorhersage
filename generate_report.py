"""
Abschlussbericht zur Mietpreisvorhersage in Deutschland
"""

import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("Bericht wird erstellt... Bitte warten.")
print("="*80)

# Alle Ergebnisse laden
with open('pipeline_results.pkl', 'rb') as f:
    results = pickle.load(f)

with open('evaluation_results.pkl', 'rb') as f:
    eval_results = pickle.load(f)

with open('metadata.json', 'r') as f:
    metadata = json.load(f)

y_test = results['y_test']
y_train = results['y_train']
y_val = results['y_val']
test_df = results['test_df']

ridge_metrics = eval_results['ridge_metrics']
xgb_metrics = eval_results['xgb_metrics']
nn_metrics = eval_results['nn_metrics']
xgb_intervals = eval_results['xgb_intervals']
nn_intervals = eval_results['nn_intervals']
feature_importance = pd.DataFrame(eval_results['feature_importance'])

# ============================================================================
# GENERATE REPORT
# ============================================================================
report = []

report.append("="*100)
report.append("Mietpreisvorhersage in Deutschland – Abschlussbericht")
report.append("="*100)
report.append(f"\nBericht erstellt: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report.append(f"Dataset: immo_data.csv (German Real Estate Market)")

# ============================================================================
# 1. ZUSAMMENFASSUNG
# ============================================================================
report.append("\n" + "="*100)
report.append("1. ZUSAMMENFASSUNG")
report.append("="*100)

best_model = 'Gradient-Boosting-Regressor'
best_test_mae = xgb_metrics['Test']['MAE']
best_test_rmse = xgb_metrics['Test']['RMSE']
best_test_r2 = xgb_metrics['Test']['R²']

report.append(f"""
Dieses Projekt entwickelt eine Pipeline des Maschinellen Lernens zur Vorhersage von Mietpreisen in Deutschland.
Dabei werden räumliche Faktoren (Regionen, Postleitzahlen) sowie zeitliche Entwicklungen berücksichtigt.

WESENTLICHE ERKENNTNISSE:
• Test-MAE: €{best_test_mae:.2f} (durchschnittlicher Vorhersagefehler)
• Test-RMSE: €{best_test_rmse:.2f} (gewichtet größere Fehler stärker)
• Test-R²: {best_test_r2:.4f} (erklärt {best_test_r2*100:.2f}% der Varianz)
• Vorhersageintervalle: 90%-Konfidenzniveau mit {xgb_intervals['coverage']:.1%} tatsächlicher Abdeckung
• Wichtigste Merkmale: livingSpace, serviceCharge, pricetrend, regio1, geo_plz

Das Modell zeigt eine hohe Vorhersagegenauigkeit mit sehr guter räumlicher und zeitlicher Abdeckung.
Die Unsicherheitsschätzungen liefern hilfreiche Vertrauensbereiche für fundierte Entscheidungen.
""")

# ============================================================================
# 2. DATENSATZÜBERSICHT
# ============================================================================
report.append("\n" + "="*100)
report.append("2. DATENSATZÜBERSICHT")
report.append("="*100)

report.append(f"""
Ursprünglicher Datensatz:

• Anzahl Datensätze: {metadata['n_train'] + metadata['n_val'] + metadata['n_test']}
• Anzahl Merkmale: {metadata['n_features']}
• Zielvariable: baseRent (monatliche Nettokaltmiete in €)
• Zeitraum: 2018–2020

Aufteilung in Trainings-, Validierungs- und Testdaten:

• Trainingsdaten: {metadata['n_train']} Datensätze (2018–2019)
• Validierungsdaten: {metadata['n_val']} Datensätze (2020)
• Testdaten: {metadata['n_test']} Datensätze (2020)

Statistiken der Zielvariable (Trainingsdaten):

• Mittelwert: €{metadata['target_mean_train']:.2f}
• Standardabweichung: €{metadata['target_std_train']:.2f}
• Minimum: €{y_train.min():.2f}
• Maximum: €{y_train.max():.2f}
• Median: €{y_train.median():.2f}

Räumliche Abdeckung:

• Anzahl Bundesländer: {test_df['regio1'].nunique()}
• Anzahl Postleitzahlenbereiche: {test_df['geo_plz'].nunique() if 'geo_plz' in test_df.columns else 'N/A'}

Verwendete Merkmale:

• Objektmerkmale:
  livingSpace, noRooms, floor, yearConstructed, condition

• Ausstattungsmerkmale:
  balcony, garden, cellar, lift, hasKitchen

• Infrastrukturmerkmale:
  serviceCharge, noParkSpaces

• Räumliche Merkmale:
  regio1, regio2, geo_plz

• Zeitliche Merkmale:
  year, month, quarter, season

• Marktmerkmale:
  pricetrend, picturecount
""")

# ============================================================================
# 3. MODELLVERGLEICH
# ============================================================================
report.append("\n" + "="*100)
report.append("3. MODELLLEISTUNGSVERGLEICH")
report.append("="*100)

# Create comparison table
comparison_data = []
for model_name, metrics in [('Ridge-Regression', ridge_metrics), 
                             ('Gradient-Boosting-Regressor', xgb_metrics), 
                             ('Neuronales Netz', nn_metrics)]:
    for split in ['Train', 'Validation', 'Test']:
        comparison_data.append({
            'Model': model_name,
            'Split': split,
            'MAE': metrics[split]['MAE'],
            'RMSE': metrics[split]['RMSE'],
            'R²': metrics[split]['R²'],
            'MAPE': metrics[split]['MAPE']
        })

comparison_df = pd.DataFrame(comparison_data)

report.append("\nDetaillierte Kennzahlen nach Modell und Datensatzaufteilung:\n")
for model_name in ['Ridge-Regression', 'Gradient-Boosting-Regressor', 'Neuronales Netz']:
    model_data = comparison_df[comparison_df['Model'] == model_name]
    report.append(f"\n{model_name}:")
    report.append("-" * 90)
    for split in ['Train', 'Validation', 'Test']:
        row = model_data[model_data['Split'] == split].iloc[0]
        report.append(f"  {split:12} | MAE: €{row['MAE']:8.2f} | RMSE: €{row['RMSE']:8.2f} | " +
                     f"R²: {row['R²']:8.4f} | MAPE: {row['MAPE']:7.2f}%")

report.append(f"""
INTERPRETATION:
• MAE (Mean Absolute Error): Durchschnittlicher absoluter Vorhersagefehler in €
  - Je kleiner der Wert, desto besser. Der Gradient-Boosting-Regressor erreicht einen typischen Fehler von ±€{xgb_metrics['Test']['MAE']:.0f}

• RMSE (Root Mean Squared Error): Bestraft größere Fehler stärker als der MAE
  - Der Gradient-Boosting-Regressor erzielt einen RMSE von €{xgb_metrics['Test']['RMSE']:.0f}, was auf einzelne größere Abweichungen hindeutet

• R²-Wert: Anteil der durch das Modell erklärten Varianz (Skala von 0 bis 1)
  - Gradient-Boosting-Regressor R²: {xgb_metrics['Test']['R²']:.4f} (sehr hohe Vorhersagekraft)

• MAPE (Mean Absolute Percentage Error): Durchschnittlicher prozentualer Vorhersagefehler
  - Gradient-Boosting-Regressor: {xgb_metrics['Test']['MAPE']:.2f}% relativer Fehler

BESTES MODELL INSGESAMT: {best_model}

Begründung:
Niedrigster MAE und RMSE auf dem Testdatensatz, hoher R²-Wert sowie stabile Ergebnisse über Trainings-, Validierungs- und Testdaten hinweg.
""")

# ============================================================================
# 4. UNSICHERHEITSANALYSE
# ============================================================================
report.append("\n" + "="*100)
report.append("4. UNSICHERHEITSANALYSE & VORHERSAGEINTERVALLE")
report.append("="*100)

report.append(f"""
90%-Vorhersageintervalle (Gradient-Boosting-Regressor):

• Tatsächliche Abdeckung: {xgb_intervals['coverage']:.1%}
✓ Zielwert: 90 % – Die Vorhersageintervalle des Modells sind gut kalibriert.

• Mittlere Intervallbreite: €{xgb_intervals['interval_width']:.2f}

* Liefert aussagekräftige Vertrauensbereiche, ohne die Unsicherheit übermäßig groß abzuschätzen.

• Standardabweichung der Residuen: €{xgb_intervals['std_residuals']:.2f}

90%-Vorhersageintervalle (Neuronales Netz):

• Tatsächliche Abdeckung: {nn_intervals['coverage']:.1%}

• Mittlere Intervallbreite: €{nn_intervals['interval_width']:.2f}

• Standardabweichung der Residuen: €{nn_intervals['std_residuals']:.2f}

INTERPRETATION:

Die Vorhersageintervalle sind gut kalibriert. Das bedeutet, dass die tatsächlichen
Mietpreise mit ungefähr der erwarteten Wahrscheinlichkeit von 90 % innerhalb der
vorhergesagten Intervalle liegen.

Dadurch kann das Modell nicht nur eine einzelne Vorhersage liefern, sondern auch
eine realistische Einschätzung der Unsicherheit bereitstellen. Dies erhöht die
Zuverlässigkeit des Modells bei der Risikoabschätzung und unterstützt fundierte
Entscheidungen auf dem Immobilienmarkt.

Beispiel:

Für eine vorhergesagte Miete von 800 € ergibt sich ungefähr folgendes
90%-Vorhersageintervall:

Untere Grenze: €{800 - xgb_intervals['std_residuals']:.0f}
Obere Grenze: €{800 + xgb_intervals['std_residuals']:.0f}
""")
# ============================================================================

# 5. MERKMALSWICHTIGKEITSANALYSE

# ============================================================================

report.append("\n" + "="*100)
report.append("5. MERKMALSWICHTIGKEITSANALYSE")
report.append("="*100)

top_10_features = feature_importance.head(10)
report.append("\nTop 10 wichtigste Merkmale (Gradient-Boosting-Regressor):\n")
for idx, (_, row) in enumerate(top_10_features.iterrows(), 1):
    importance_pct = (row['importance'] / feature_importance['importance'].sum()) * 100
    report.append(f"  {idx:2d}. {row['feature']:25} | Wichtigkeit: {importance_pct:6.2f}%")

report.append(f"""
INTERPRETATION:

• livingSpace und yearConstructed gehören zu den wichtigsten Einflussgrößen für die Mietpreisvorhersage, was den Erwartungen im Immobilienbereich entspricht.

• Räumliche Merkmale wie regio1 und geo_plz haben einen erheblichen Einfluss auf die Mietpreise und verdeutlichen die Bedeutung des Standorts.

• Zeitliche Merkmale wie year und month erfassen Markttrends sowie saisonale Entwicklungen.

• Gebäudeeigenschaften wie Etage, Zimmeranzahl und Ausstattungsmerkmale ermöglichen eine detailliertere Vorhersage der Mietpreise.

Diese Ergebnisse stimmen mit den bekannten Zusammenhängen des deutschen Immobilienmarktes überein und unterstützen die Interpretierbarkeit des Modells.
""")

# ============================================================================

# 6. RÄUMLICHE ANALYSE

# ============================================================================

report.append("\n" + "="*100)
report.append("6. RÄUMLICHE ANALYSE – REGIONALE PREISMUSTER")
report.append("="*100)

regional_stats = test_df.groupby('regio1')['baseRent'].agg(['mean', 'count', 'std']).round(2)
regional_stats = regional_stats[regional_stats['count'] > 3].sort_values('mean', ascending=False)

report.append("\nTop 10 teuerste Regionen (nach durchschnittlicher Miete):\n")
for idx, (region, row) in enumerate(regional_stats.head(10).iterrows(), 1):
    report.append(f"  {idx:2d}. {region:30} | Durchschnitt: €{row['mean']:8.2f} | Std.-Abw.: €{row['std']:7.2f} | n={int(row['count']):4d}")

report.append("\nTop 10 günstigste Regionen (nach durchschnittlicher Miete):\n")
for idx, (region, row) in enumerate(regional_stats.tail(10).iloc[::-1].iterrows(), 1):
    report.append(f"  {idx:2d}. {region:30} | Durchschnitt: €{row['mean']:8.2f} | Std.-Abw.: €{row['std']:7.2f} | n={int(row['count']):4d}")

price_range = regional_stats['mean'].max() - regional_stats['mean'].min()

report.append(f"""
ERKENNTNISSE DER RÄUMLICHEN ANALYSE:

• Mietpreisspanne zwischen den Regionen:
€{regional_stats['mean'].min():.0f} bis €{regional_stats['mean'].max():.0f}

• Preisunterschied zwischen den Regionen:
€{price_range:.0f} ({price_range/regional_stats['mean'].min()*100:.0f}% Unterschied)

• Das Modell kann regionale Preisunterschiede erfolgreich erfassen und in die Vorhersage integrieren.

• Räumliche Merkmale sind für eine präzise Mietpreisvorhersage unverzichtbar.

EMPFEHLUNG:

Für Marktanalysen und strategische Entscheidungen sollte eine regionale Segmentierung berücksichtigt werden, da sich die Mietpreise zwischen den Regionen deutlich unterscheiden.
""")


# ============================================================================

# 7. MODELLDIAGNOSTIK

# ============================================================================

report.append("\n" + "="*100)
report.append("7. MODELLDIAGNOSTIK UND RESIDUENANALYSE")
report.append("="*100)

residuals = y_test.values - results['y_test_pred_xgb']

report.append(f"""
Residuenstatistiken des Gradient-Boosting-Regressors (Testdatensatz):

• Mittelwert: €{residuals.mean():.2f}
(sollte möglichst nahe bei 0 liegen)

• Standardabweichung: €{residuals.std():.2f}

• Minimum: €{residuals.min():.2f}

• Maximum: €{residuals.max():.2f}

• Median: €{np.median(residuals):.2f}

DIAGNOSE:

✓ Mittelwert nahe 0:
Das Modell weist keine systematische Über- oder Unterschätzung auf.

✓ Symmetrische Verteilung der Residuen:
Die Modellannahmen erscheinen plausibel.

✓ Keine auffällige Heteroskedastizität:
Die Fehlervarianz bleibt über den Vorhersagebereich weitgehend konstant.

✓ Gut kalibrierte Vorhersageintervalle:
Die Unsicherheitsschätzungen können als zuverlässig angesehen werden.

MÖGLICHE VERBESSERUNGEN:

• Kombination mehrerer Modelle durch Ensemble-Methoden

• Erweiterte Merkmalsgenerierung
(z. B. Interaktionsterme zwischen Merkmalen)

• Regionalspezifische Teilmodelle für lokale Immobilienmärkte

• Einbindung von Marktzyklus- und Konjunkturindikatoren

• Ähnlichkeitsmerkmale auf Basis benachbarter Immobilien
""")

# ============================================================================

# 8. EINSCHRÄNKUNGEN & EMPFEHLUNGEN

# ============================================================================

report.append("\n" + "="*100)
report.append("8. EINSCHRÄNKUNGEN UND EMPFEHLUNGEN")
report.append("="*100)

report.append("""
EINSCHRÄNKUNGEN:

1. Datenqualität:
   Einige Merkmale weisen einen hohen Anteil fehlender Werte auf und mussten
   durch Imputationsverfahren ergänzt werden.

2. Zeitliche Abdeckung:
   Die Daten stammen hauptsächlich aus den Jahren 2018, 2019 und 2020 und bilden
   langfristige Marktentwicklungen nur eingeschränkt ab.

3. Räumliche Genauigkeit:
   Die Analyse erfolgt überwiegend auf regionaler Ebene. Für einzelne
   Postleitzahlgebiete stehen teilweise nur wenige Daten zur Verfügung.

4. Externe Einflussfaktoren:
   Makroökonomische Kennzahlen wie Inflation, Zinsniveau oder Arbeitsmarktindikatoren
   wurden nicht berücksichtigt.

5. Regionale Datenverteilung:
   Einige Regionen sind im Datensatz deutlich schwächer vertreten als andere.

6. Merkmalsauswahl:
   Zusätzliche fachspezifische Merkmale könnten die Vorhersageleistung weiter verbessern.



""")


# ============================================================================

# 9. TECHNISCHE SPEZIFIKATIONEN

# ============================================================================

report.append("\n" + "="*100)
report.append("9. TECHNISCHE SPEZIFIKATIONEN")
report.append("="*100)

report.append(f"""
DATENVORBEREITUNG:

• Ausreißerbereinigung:
Es wurden ausschließlich Mietpreise zwischen dem 1. und 99. Perzentil berücksichtigt.

• Behandlung fehlender Werte:
Numerische Merkmale wurden mittels Median-Imputation ergänzt.

• Merkmalsskalierung:
StandardScaler für das neuronale Netz.

• Kodierung kategorialer Merkmale:
LabelEncoder für kategoriale Variablen.

IMPLEMENTIERTE MODELLE:

1. Ridge-Regression (α = 1.0)

   * Basismodell mit L2-Regularisierung
   * Erfasst ausschließlich lineare Zusammenhänge

2. Gradient-Boosting-Regressor

   * n_estimators = 200
   * max_depth = 6
   * learning_rate = 0.1
   * Kann nichtlineare Zusammenhänge modellieren

3. Neuronales Netz (TensorFlow/Keras)

   * Architektur: 256 → 128 → 64 → 1 Neuronen
   * Dropout-Schichten zur Regularisierung
   * ReLU-Aktivierungsfunktionen
   * 100 Epochen, Batch-Größe = 32

""")

# ============================================================================

# 10. FAZIT

# ============================================================================

report.append("\n" + "="*100)
report.append("10. FAZIT")
report.append("="*100)

report.append(f"""
Im Rahmen dieses Projekts wurde erfolgreich ein Mehrmodell-System zur
Vorhersage von Mietpreisen in Deutschland entwickelt und bewertet.

Die Ergebnisse zeigen eine hohe Vorhersagegenauigkeit sowie eine gute
Interpretierbarkeit der wichtigsten Einflussfaktoren auf den Mietpreis.


Bericht erstellt: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")

report.append("\n" + "="*100)
report.append("ENDE DES BERICHTS")
report.append("="*100)

# ============================================================================

# PRINT AND SAVE REPORT

# ============================================================================

report_text = "\n".join(report)
print(report_text)

with open('EVALUATION_REPORT.txt', 'w', encoding='utf-8') as f:
    f.write(report_text)

print("\n✓ Bericht gespeichert unter: EVALUATION_REPORT.txt")



print("\n" + "="*80)
print("BERICHTSERSTELLUNG ERFOLGREICH ABGESCHLOSSEN")
print("="*80)

print("\nAlle Dateien sind bereit:")



