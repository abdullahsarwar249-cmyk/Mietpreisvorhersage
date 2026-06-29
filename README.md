# Mietpreisvorhersage mit Maschinellem Lernen

Projektarbeit im Modul **Angewandtes Maschinelles Lernen (AML)**

**Studierende**

* Abdullah Sarwar (132261051)
* Serkan Dülger (132251079)

Master Technical Management
Sommersemester 2026

---

## Projektbeschreibung

Dieses Projekt entwickelt ein Machine-Learning-System zur Vorhersage von Mietpreisen in Deutschland.

Die Pipeline umfasst:

* Datenaufbereitung
* Modelltraining
* Modellbewertung
* Visualisierung der Ergebnisse
* Erstellung interaktiver Karten
* Automatische Berichtserstellung

---

## Datensatz

Verwendeter Datensatz:

**Apartment Rental Offers in Germany**

Kaggle:

https://www.kaggle.com/datasets/corrieaar/apartment-rental-offers-in-germany

### Datensatz herunterladen

Die Datei

```text
immo_data.csv
```

ist aufgrund ihrer Größe nicht Bestandteil dieses Repositories.

Zur Reproduktion der Ergebnisse:

1. Datensatz von Kaggle herunterladen
2. Die Datei `immo_data.csv` in das Hauptverzeichnis des Projekts kopieren

Beispiel:

```text
Mietpreisvorhersage/
│
├── immo_data.csv
├── main_pipeline.py
├── evaluate_models.py
├── visualizations.py
├── generate_maps.py
├── generate_report.py
└── run_pipeline.sh
```

---

# Repository herunterladen

Das Repository kann mit Git geklont werden:

```bash
git clone <https://github.com/abdullahsarwar249-cmyk/Mietpreisvorhersage.git>
cd Mietpreisvorhersage
```

Alternativ kann das Repository auch direkt über GitHub als ZIP-Datei heruntergeladen werden:

1. GitHub-Repository öffnen
2. Auf **Code** klicken
3. **Download ZIP** auswählen
4. ZIP-Datei entpacken

---

# Installation

Alle benötigten Bibliotheken installieren:

```bash
pip install -r requirements.txt
```


---

## Projektstruktur

```text
main_pipeline.py
│
├── Datenaufbereitung
├── Modelltraining
└── Speicherung der Modelle

evaluate_models.py
│
├── Modellbewertung
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

run_pipeline.sh
│
└── Ausführung der gesamten Pipeline
```

---

## Ausführung

Die komplette Pipeline kann mit folgendem Befehl gestartet werden:

```bash
bash run_pipeline.sh
```

Alternativ können die einzelnen Schritte separat ausgeführt werden:

```bash
python3 main_pipeline.py
python3 evaluate_models.py
python3 visualizations.py
python3 generate_maps.py
python3 generate_report.py
```

---

## Erzeugte Dateien

### Modelle

```text
gb_model.pkl
ridge_model.pkl
nn_model.h5
```

### Visualisierungen

```text
01_data_exploration.png
02_model_performance.png
03_uncertainty_analysis.png
04_feature_importance.png
05_spatial_analysis.png
```

### Interaktive Karten

```text
interactive_rental_map.html
prediction_accuracy_map.html
```

### Berichte

```text
EVALUATION_REPORT.txt
metadata.json
```

---

## Hinweise

Für die Ausführung wird Python 3 benötigt.

Die benötigten Bibliotheken sind in der Datei `requirements.txt` aufgeführt.

Der Datensatz muss vor dem Start der Pipeline manuell von Kaggle heruntergeladen werden und in den Ordner gepackt werden.
