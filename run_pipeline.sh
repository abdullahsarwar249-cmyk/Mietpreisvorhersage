#!/bin/bash
#
# Mietpreisvorhersage – Ausführung der kompletten ML-Pipeline
# Führt alle Verarbeitungsschritte nacheinander aus
#

echo "================================================================================"
echo "MIETPREISVORHERSAGE – KOMPLETTE ML-PIPELINE"
echo "Deutsche Mietpreisschätzung mit räumlicher und zeitlicher Analyse"
echo "================================================================================"
echo ""
echo "Dieses Skript führt alle Pipeline-Schritte automatisch nacheinander aus."
echo ""

# Prüfen, ob Python 3 verfügbar ist
if ! command -v python3 &> /dev/null; then
    echo "❌ Fehler: Python 3 ist nicht installiert oder befindet sich nicht im PATH."
    exit 1
fi

echo "✓ Python gefunden: $(python3 --version)"
echo ""

# Schritt 1: Hauptpipeline
echo "================================================================================"
echo "SCHRITT 1: MODELLE TRAINIEREN"
echo "================================================================================"
python3 main_pipeline.py
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
python3 evaluate_models.py
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
python3 visualizations.py
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
python3 generate_maps.py
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
python3 generate_report.py
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

echo "📊 Visualisierungen (PNG-Dateien, 300 DPI):"
echo "  - 01_data_exploration.png"
echo "  - 02_model_performance.png"
echo "  - 03_uncertainty_analysis.png"
echo "  - 04_feature_importance.png"
echo "  - 05_spatial_analysis.png"
echo "  - 06_temporal_analysis.png"
echo "  - 07_residual_analysis.png"
echo ""

echo "🗺️ Interaktive Karten (HTML-Dateien, im Browser öffnen):"
echo "  - interactive_rental_map.html"
echo "  - prediction_accuracy_map.html"
echo ""

echo "📄 Berichte:"
echo "  - EVALUATION_REPORT.txt (ausführlicher Analysebericht)"
echo "  - report_summary.json (kompakte JSON-Zusammenfassung)"
echo ""

echo "🤖 Trainierte Modelle:"
echo "  - gb_model.pkl (Gradient-Boosting-Modell)"
echo "  - nn_model.h5 (Neuronales Netzwerk)"
echo "  - ridge_model.pkl (Ridge-Regression als Basismodell)"
echo ""

echo "📁 Daten und Metadaten:"
echo "  - pipeline_results.pkl (Vorhersagen und Zwischenergebnisse)"
echo "  - evaluation_results.pkl (detaillierte Leistungskennzahlen)"
echo "  - metadata.json (Informationen zum Datensatz)"
echo ""

echo "================================================================================"
echo "Nächste Schritte:"
echo "  1. Öffnen Sie die Visualisierungen in einem Bildbetrachter."
echo "  2. Öffnen Sie die HTML-Dateien in einem Webbrowser, um die interaktiven Karten zu erkunden."
echo "  3. Lesen Sie den Bericht EVALUATION_REPORT.txt für detaillierte Erkenntnisse."
echo "  4. Nutzen Sie die README.md für die vollständige Dokumentation."
echo "================================================================================"