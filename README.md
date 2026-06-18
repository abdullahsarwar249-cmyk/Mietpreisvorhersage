# Mietpreisvorhersage mit Maschinelles Lernen

## Projektarbeit im Modul Angewandtes Maschinelles Lernen (AML)

**Studierende**

* Abdullah Sarwar (132261051)
* Serkan Dülger (132251079)

Master Technical Management
Sommersemester 2026

---

# Projektziel

Ziel dieser Projektarbeit ist die Entwicklung eines Maschinenlernen-Systems zur Vorhersage von Mietpreisen in Deutschland.

Im Fokus steht die Vorhersage der monatlichen Nettokaltmiete auf Basis verschiedener Wohnungsmerkmale. Neben klassischen Immobilienmerkmalen werden auch räumliche und zeitliche Faktoren berücksichtigt, um regionale Unterschiede sowie zeitliche Entwicklungen des Wohnungsmarktes abzubilden.

Zusätzlich werden die Vorhersagen hinsichtlich ihrer Unsicherheit analysiert und die Ergebnisse durch verschiedene Visualisierungen sowie interaktive Karten dargestellt.

---

# Datensatz

Für die Projektarbeit wurde der öffentlich verfügbare Datensatz:

**Apartment Rental Offers in Germany**

verwendet.

Datensatzquelle:

https://www.kaggle.com/datasets/corrieaar/apartment-rental-offers-in-germany

Der Datensatz basiert auf Wohnungsanzeigen aus Deutschland und enthält unter anderem Informationen zu:

* Mietpreis
* Wohnfläche
* Zimmeranzahl
* Baujahr
* Ausstattung
* Region
* Postleitzahl
* Lageinformationen

---

# Hinweis zur Datengröße

Die Originaldatei

```text
immo_data.csv
```

ist ca. 272 MB groß und wurde daher nicht in das Git-Repository aufgenommen.

Zur Reproduktion der Ergebnisse muss der Datensatz separat von Kaggle heruntergeladen und anschließend im Hauptverzeichnis des Projekts als

```text
immo_data.csv
```

abgelegt werden.

---

# Datensatz nach Vorverarbeitung

Nach der Datenbereinigung wurden insgesamt 41.021 Datensätze verwendet.

| Datensatz   | Anzahl |
| ----------- | -----: |
| Training    | 34.464 |
| Validierung |  7.268 |
| Test        |  7.268 |
---

# Zielvariable

Die vorherzusagende Zielgröße ist:

```text
baseRent
```

Monatliche Nettokaltmiete in Euro.

---

# Verwendete Merkmale

## Objektmerkmale

* livingSpace
* noRooms
* floor
* numberOfFloors
* yearConstructed

## Ausstattung

* balcony
* garden
* cellar
* lift
* newlyConst
* hasKitchen
* noParkSpaces

## Marktinformationen

* serviceCharge
* pricetrend
* picturecount

## Räumliche Merkmale

* regio1
* regio2
* geo_plz

## Zeitliche Merkmale

* year
* month
* quarter
* season

Insgesamt wurden 25 Features für das Training verwendet.

---

# Datenaufbereitung

Vor dem Training wurden mehrere Schritte der Datenvorverarbeitung durchgeführt.

## Bereinigung

* Entfernen von Datensätzen ohne Zielvariable
* Entfernen extremer Ausreißer (1%-99%-Quantile)

## Fehlende Werte

Fehlende numerische Werte wurden mittels Median-Imputation ersetzt.

## Kategorische Variablen

Kategorische Merkmale wurden mittels Label Encoding in numerische Werte umgewandelt.

## Zeitliche Merkmale

Aus den Datumsinformationen wurden zusätzliche Merkmale erzeugt:

* Jahr
* Monat
* Quartal
* Saison

## Skalierung

Für das neuronale Netz wurde eine Standardisierung der Eingabedaten mittels StandardScaler durchgeführt.

---

# Methodik

Zur Lösung der Regressionsaufgabe wurden drei verschiedene Modellklassen implementiert und miteinander verglichen.

## 1. Ridge-Regression

Die Ridge-Regression dient als lineares Basismodell.

Sie ermöglicht die Bewertung, wie gut lineare Zusammenhänge zwischen den Eingangsmerkmalen und den Mietpreisen modelliert werden können.

---

## 2. Gradient-Boosting-Regressor

Der Gradient-Boosting-Regressor bildet das leistungsstärkste Modell innerhalb des Projekts.

Durch die Kombination mehrerer Entscheidungsbäume können komplexe nichtlineare Zusammenhänge zwischen Immobilienmerkmalen und Mietpreisen erfasst werden.

Verwendete Parameter:

* n_estimators = 200
* max_depth = 6
* learning_rate = 0.1
* subsample = 0.8

---

## 3. Neuronales Netz

Zusätzlich wurde ein Deep-Learning-Modell mit TensorFlow/Keras entwickelt.

Architektur:

```text
Input (25 Features)
        ↓
Dense (256)
        ↓
Dense (128)
        ↓
Dense (64)
        ↓
Output (1)
```

Zur Vermeidung von Overfitting wurden Dropout-Schichten und L2-Regularisierung eingesetzt.

---

# Evaluationsmethodik

Für die Bewertung der Modelle wurde ein zeitlicher Split verwendet.

* Trainingsdaten: 201-2019
* Validierungsdaten: 2020
* Testdaten: 2020

Dadurch wird ein realistisches Vorhersageszenario simuliert, bei dem vergangene Daten zur Vorhersage zukünftiger Mietpreise genutzt werden.

---

# Bewertungsmetriken

Die Modelle wurden anhand folgender Kennzahlen bewertet:

## MAE (Mean Absolute Error)

Durchschnittlicher absoluter Vorhersagefehler in Euro.

## RMSE (Root Mean Squared Error)

Berücksichtigt größere Fehler stärker als der MAE.

## R² (Bestimmtheitsmaß)

Beschreibt den Anteil der erklärten Varianz.


---

# Ergebnisse

## Modellvergleich

| Modell                      |      MAE |     RMSE |     R² |
| --------------------------- | -------: | -------: | -----: |
| Ridge-Regression            | 145,04 € | 217,93 € | 0,7092 |
| Gradient-Boosting-Regressor |  89,74 € | 143,92 € | 0,8732 |
| Neuronales Netz             | 127,23 € | 203,13 € | 0,7474 |

---

# Bestes Modell

Der Gradient-Boosting-Regressor erzielte die beste Leistung auf dem Testdatensatz.

Testergebnisse:

* MAE: 89,74 €
* RMSE: 143,92 €
* R²: 0,8732

Das Modell erklärt damit rund 87,3 % der Varianz der Mietpreise.

---

# Unsicherheitsanalyse

Neben den klassischen Regressionsmetriken wurde eine Unsicherheitsanalyse durchgeführt.

Hierzu wurden 90%-Vorhersageintervalle auf Basis der Modellresiduen berechnet.

Ergebnisse:

* Coverage: 87,0 %
* Durchschnittliche Intervallbreite: 370,16 €
* Standardabweichung der Residuen: 112,51 €

Die Analyse zeigt, dass die Vorhersageintervalle die tatsächlichen Mietpreise in einem Großteil der Fälle zuverlässig einschließen.

---

# Merkmalswichtigkeit

Die wichtigsten Einflussgrößen des Gradient-Boosting-Modells sind:

1. livingSpace
2. serviceCharge
3. pricetrend
4. regio1
5. geo_plz

Die Ergebnisse zeigen, dass sowohl Wohnungsmerkmale als auch regionale Faktoren einen wesentlichen Einfluss auf die Mietpreisbildung besitzen.

---

# Räumliche Analyse

Zusätzlich zur Mietpreisvorhersage wurden regionale Unterschiede untersucht.

Hierzu wurden:

* Bundesländer
* Regionen
* Postleitzahlenbereiche

analysiert und in interaktiven Karten visualisiert.

Die Ergebnisse zeigen deutliche Preisunterschiede zwischen verschiedenen Regionen Deutschlands.

---

# Zeitliche Analyse

Zur Berücksichtigung zeitlicher Entwicklungen wurden folgende Merkmale integriert:

* Jahr
* Monat
* Quartal
* Saison

Dadurch können zeitabhängige Einflüsse auf die Mietpreise in die Modellierung einbezogen werden.

---

# Projektstruktur

```text
main_pipeline.py
│
├── Datenaufbereitung
├── Modelltraining
└── Speicherung der Modelle

evaluate_models.py
│
├── Modellvergleich
├── Unsicherheitsanalyse
├── Merkmalswichtigkeit
├── Räumliche Analyse
└── Zeitliche Analyse

visualizations.py
│
└── Erstellung aller Diagramme

generate_maps.py
│
└── Erstellung interaktiver Karten

generate_report.py
│
└── Automatische Berichtserstellung
```

---

# Erzeugte Dateien

Das Projekt erzeugt unter anderem:

## Visualisierungen

* 01_data_exploration.png
* 02_model_performance.png
* 03_uncertainty_analysis.png
* 04_feature_importance.png
* 05_spatial_analysis.png
* 06_temporal_analysis.png


## Interaktive Karten

* interactive_rental_map.html
* prediction_accuracy_map.html

### Hinweis zur Anzeige der Karten

Die interaktiven Karten können in den meisten Fällen direkt im Browser geöffnet werden.

Sollten die Karten jedoch nicht korrekt dargestellt werden oder Funktionen fehlen, empfiehlt sich die Ausführung über einen lokalen Webserver.

Für Visual Studio Code wird hierfür die Erweiterung **Live Server** empfohlen:

1. Projektordner in Visual Studio Code öffnen
2. Erweiterung „Live Server“ installieren
3. Rechtsklick auf die gewünschte HTML-Datei
4. „Open with Live Server“ auswählen



## Modelle

* gb_model.pkl
* ridge_model.pkl
* nn_model.h5
---

# Reproduzierbarkeit

Nach dem Herunterladen des Datensatzes kann die vollständige Pipeline ausgeführt werden:

```bash
python3 main_pipeline.py
python3 evaluate_models.py
python3 visualizations.py
python3 generate_maps.py
python3 generate_report.py
```

Alternativ:

```bash
bash run_pipeline.sh
```

---

# Limitationen

Trotz der guten Vorhersageleistung bestehen einige Einschränkungen:

* Datensatz beschränkt sich auf die Jahre 2018–2020
* Teilweise fehlende Werte erfordern Imputation
* Keine Berücksichtigung makroökonomischer Einflussgrößen
* Keine langfristige Vorhersage zukünftiger Mietmarktentwicklungen

---

# Fazit

Im Rahmen dieser Projektarbeit wurde ein vollständiges Maschinenlernen-System zur Vorhersage deutscher Mietpreise entwickelt.

Durch den Vergleich verschiedener Modellklassen, die Integration räumlicher und zeitlicher Merkmale sowie die Durchführung einer Unsicherheitsanalyse konnte eine umfassende Untersuchung des Problems durchgeführt werden.

Das beste Modell, ein Gradient-Boosting-Regressor, erreicht einen MAE von 89,74 € und erklärt rund 87,3 % der Varianz der Mietpreise. Die Ergebnisse zeigen, dass datengetriebene Verfahren eine zuverlässige Unterstützung bei der Analyse und Bewertung von Mietpreisen liefern können.
