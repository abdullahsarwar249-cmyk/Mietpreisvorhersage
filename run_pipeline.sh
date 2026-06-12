#!/bin/bash
#
# Mietpreisvorhersage - Ausführung der kompletten ML-Pipeline
# Führt alle Schritte nacheinander aus
#

echo "================================================================================"
echo "MIETPREISVORHERSAGE - KOMPLETTE ML-PIPELINE"
echo "Deutsche Mietpreisschätzung mit räumlicher und zeitlicher Analyse"
echo "================================================================================"
echo ""
echo "Dieses Skript führt alle Pipeline-Schritte nacheinander aus."
echo "Geschätzte Gesamtlaufzeit: 20-30 Minuten"
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Fehler: python3 ist nicht installiert oder nicht im PATH"
    exit 1
fi

echo "✓ Python3 gefunden: $(python3 --version)"
echo ""

# Schritt 1: Hauptpipeline
echo "================================================================================"
echo "SCHRITT 1: MODELLE TRAINIEREN (10-15 Minuten)"
echo "================================================================================"
python3 main_pipeline.py
if [ $? -ne 0 ]; then
    echo "❌ Trainingslauf fehlgeschlagen!"
    exit 1
fi
echo "✓ Schritt 1 abgeschlossen!"
echo ""

# Schritt 2: Modellbewertung
echo "================================================================================"
echo "SCHRITT 2: MODELLE BEWERTEN (2-3 Minuten)"
echo "================================================================================"
python3 evaluate_models.py
if [ $? -ne 0 ]; then
    echo "❌ Modellbewertung fehlgeschlagen!"
    exit 1
fi
echo "✓ Schritt 2 abgeschlossen!"
echo ""

# Schritt 3: Visualisierungen
echo "================================================================================"
echo "SCHRITT 3: VISUALISIERUNGEN ERSTELLEN (3-5 Minuten)"
echo "================================================================================"
python3 visualizations.py
if [ $? -ne 0 ]; then
    echo "❌ Visualisierungserstellung fehlgeschlagen!"
    exit 1
fi
echo "✓ Schritt 3 abgeschlossen!"
echo ""

# Schritt 4: Karten
echo "================================================================================"
echo "SCHRITT 4: INTERAKTIVE KARTEN ERSTELLEN (2-3 Minuten)"
echo "================================================================================"
python3 generate_maps.py
if [ $? -ne 0 ]; then
    echo "❌ Kartenerstellung fehlgeschlagen!"
    exit 1
fi
echo "✓ Schritt 4 abgeschlossen!"
echo ""

# Schritt 5: Bericht
echo "================================================================================"
echo "SCHRITT 5: ABSCHLUSSBERICHT ERSTELLEN (1 Minute)"
echo "================================================================================"
python3 generate_report.py
if [ $? -ne 0 ]; then
    echo "❌ Berichtserstellung fehlgeschlagen!"
    exit 1
fi
echo "✓ Schritt 5 abgeschlossen!"
echo ""

# Zusammenfassung
echo "================================================================================"
echo "✓ ALLE SCHRITTE ERFOLGREICH ABGESCHLOSSEN!"
echo "================================================================================"
echo ""
echo "Erzeugte Ausgaben:"
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
echo "🗺️  Interaktive Karten (HTML, im Browser öffnen):"
echo "  - interactive_rental_map.html"
echo "  - prediction_accuracy_map.html"
echo ""
echo "📄 Berichte:"
echo "  - EVALUATION_REPORT.txt (umfassender Textbericht)"
echo "  - report_summary.json (JSON-Zusammenfassung)"
echo ""
echo "🤖 Trainierte Modelle:"
echo "  - gb_model.pkl (Gradient Boosting)"
echo "  - nn_model.h5 (Neuronales Netz)"
echo "  - ridge_model.pkl (Ridge-Regression - baseline)"
echo ""
echo "📁 Daten & Metadaten:"
echo "  - pipeline_results.pkl (alle Vorhersagen und Daten)"
echo "  - evaluation_results.pkl (detaillierte Kennzahlen)"
echo "  - metadata.json (Datensatzinformationen)"
echo ""
echo "================================================================================"
echo "Nächste Schritte:"
echo "  1. Visualisierungen in einem Bildbetrachter ansehen"
echo "  2. Öffnen Sie HTML-Dateien in a web browser to explore interaktive Karten"
echo "  3. Lesen Sie EVALUATION_REPORT.txt für detaillierte Erkenntnisse"
echo "  4. Verwenden Sie README.md for complete documentation"
echo "================================================================================"
