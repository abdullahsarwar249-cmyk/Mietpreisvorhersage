#!/bin/bash
#
# Mietpreisvorhersage – Ausführung der kompletten ML-Pipeline
# Führt alle Verarbeitungsschritte nacheinander aus
#

echo "================================================================================"
echo "MIETPREISVORHERSAGE – KOMPLETTE ML-PIPELINE"
echo "Deutsche Mietpreisschätzung"
echo "================================================================================"
echo ""
echo "Dieses Skript führt alle Pipeline-Schritte automatisch nacheinander aus sodass man die nicht einzeln triggern muss."
echo ""

# Immer System-Python verwenden (wie in der bisherigen funktionierenden Umgebung)
SYSTEM_PYTHON="/usr/bin/python3"

# Fallback für Systeme, bei denen /usr/bin/python3 nicht existiert
if [ ! -x "$SYSTEM_PYTHON" ]; then
    SYSTEM_PYTHON="$(command -v python3)"
fi

if [ -z "$SYSTEM_PYTHON" ] || [ ! -x "$SYSTEM_PYTHON" ]; then
    echo "❌ Fehler: Python 3 ist nicht installiert oder befindet sich nicht im PATH."
    exit 1
fi

echo "✓ Python wurde gefunden: $($SYSTEM_PYTHON --version)"
echo "✓ Verwendeter Interpreter: $SYSTEM_PYTHON"
echo ""

# Schritt 1: Hauptpipeline
echo "================================================================================"
echo "SCHRITT 1: MODELLE TRAINIEREN"
echo "================================================================================"
"$SYSTEM_PYTHON" main_pipeline.py
if [ $? -ne 0 ]; then
    echo "❌ Fehler: Das Training der Modelle ist fehlgeschlagen!"
    exit 1
fi
echo "✓ Schritt 1 erfolgreich abgeschlossen."
echo ""

# Schritt 2: Modellbewertung
echo "================================================================================"
echo "SCHRITT 2: MODELLE BEWERTEN"
echo "================================================================================"
"$SYSTEM_PYTHON" evaluate_models.py
if [ $? -ne 0 ]; then
    echo "❌ Fehler: Die Modellbewertung ist fehlgeschlagen!"
    exit 1
fi
echo "✓ Schritt 2 erfolgreich abgeschlossen."
echo ""

# Schritt 3: Visualisierungen
echo "================================================================================"
echo "SCHRITT 3: VISUALISIERUNGEN ERSTELLEN"
echo "================================================================================"
"$SYSTEM_PYTHON" visualizations.py
if [ $? -ne 0 ]; then
    echo "❌ Fehler: Die Erstellung der Visualisierungen ist fehlgeschlagen!"
    exit 1
fi
echo "✓ Schritt 3 erfolgreich abgeschlossen."
echo ""

# Schritt 4: Karten erstellen
echo "================================================================================"
echo "SCHRITT 4: INTERAKTIVE KARTEN ERSTELLEN"
echo "================================================================================"
"$SYSTEM_PYTHON" generate_maps.py
if [ $? -ne 0 ]; then
    echo "❌ Fehler: Die Kartenerstellung ist fehlgeschlagen!"
    exit 1
fi
echo "✓ Schritt 4 erfolgreich abgeschlossen."
echo ""

# Schritt 5: Abschlussbericht
echo "================================================================================"
echo "SCHRITT 5: ABSCHLUSSBERICHT ERSTELLEN"
echo "================================================================================"
"$SYSTEM_PYTHON" generate_report.py
if [ $? -ne 0 ]; then
    echo "❌ Fehler: Die Berichtserstellung ist fehlgeschlagen!"
    exit 1
fi
echo "✓ Schritt 5 erfolgreich abgeschlossen."
echo ""

# Zusammenfassung
echo "================================================================================"
echo "✓ ALLE SCHRITTE ERFOLGREICH ABGESCHLOSSEN!"
echo "================================================================================"
echo ""
echo "Erzeugte Ausgabedateien:"
echo ""

echo "Visualisierungen in png Format:"
echo "  - 01_data_exploration.png"
echo "  - 02_model_performance.png"
echo "  - 03_uncertainty_analysis.png"
echo "  - 04_feature_importance.png"
echo "  - 05_spatial_analysis.png"
echo ""

echo "Interaktive Karten (HTML-Dateien, im Browser öffnen):"
echo "  - interactive_rental_map.html"
echo "  - prediction_accuracy_map.html"
echo ""

echo "Berichte:"
echo "  - EVALUATION_REPORT.txt (ausführlicher Analysebericht)"
echo ""

echo "Trainierte Modelle:"
echo "  - gb_model.pkl (Gradient-Boosting-Modell)"
echo "  - nn_model.h5 (Neuronales Netzwerk)"
echo "  - ridge_model.pkl (Ridge-Regression als Basismodell)"
echo ""

echo "Daten und Metadaten:"
echo "  - pipeline_results.pkl (Vorhersagen und Zwischenergebnisse)"
echo "  - evaluation_results.pkl (detaillierte Leistungskennzahlen)"
echo "  - metadata.json (Informationen zum Datensatz)"
echo ""

echo "================================================================================"
echo "Nächste Schritte:"
echo "  1. Öffnen Sie die Visualisierungen"
echo "  2. Öffnen Sie die HTML-Dateien in einem Webbrowser, um die interaktiven Karten zu erkunden."
echo "  3. Lesen Sie den Bericht EVALUATION_REPORT.txt für detaillierte Erkenntnisse."
echo "  4. Nutzen Sie die README.md für die vollständige Dokumentation."
echo "================================================================================"