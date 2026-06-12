#!/bin/bash
#
# Mietpreisvorhersage - Complete ML Pipeline Execution Script
# Runs all steps sequentially
#

echo "================================================================================"
echo "MIETPREISVORHERSAGE - COMPLETE ML PIPELINE"
echo "German Rental Price Prediction with Spatial & Temporal Analysis"
echo "================================================================================"
echo ""
echo "This script will execute all pipeline steps sequentially."
echo "Total estimated runtime: 20-30 minutes"
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is not installed or not in PATH"
    exit 1
fi

echo "✓ Python3 found: $(python3 --version)"
echo ""

# Step 1: Main Pipeline
echo "================================================================================"
echo "STEP 1: TRAINING MODELS (10-15 minutes)"
echo "================================================================================"
python3 main_pipeline.py
if [ $? -ne 0 ]; then
    echo "❌ Pipeline training failed!"
    exit 1
fi
echo "✓ Step 1 completed!"
echo ""

# Step 2: Model Evaluation
echo "================================================================================"
echo "STEP 2: EVALUATING MODELS (2-3 minutes)"
echo "================================================================================"
python3 evaluate_models.py
if [ $? -ne 0 ]; then
    echo "❌ Model evaluation failed!"
    exit 1
fi
echo "✓ Step 2 completed!"
echo ""

# Step 3: Visualizations
echo "================================================================================"
echo "STEP 3: GENERATING VISUALIZATIONS (3-5 minutes)"
echo "================================================================================"
python3 visualizations.py
if [ $? -ne 0 ]; then
    echo "❌ Visualization generation failed!"
    exit 1
fi
echo "✓ Step 3 completed!"
echo ""

# Step 4: Maps
echo "================================================================================"
echo "STEP 4: CREATING INTERACTIVE MAPS (2-3 minutes)"
echo "================================================================================"
python3 generate_maps.py
if [ $? -ne 0 ]; then
    echo "❌ Map generation failed!"
    exit 1
fi
echo "✓ Step 4 completed!"
echo ""

# Step 5: Report
echo "================================================================================"
echo "STEP 5: GENERATING FINAL REPORT (1 minute)"
echo "================================================================================"
python3 generate_report.py
if [ $? -ne 0 ]; then
    echo "❌ Report generation failed!"
    exit 1
fi
echo "✓ Step 5 completed!"
echo ""

# Summary
echo "================================================================================"
echo "✓ ALL STEPS COMPLETED SUCCESSFULLY!"
echo "================================================================================"
echo ""
echo "Generated Outputs:"
echo ""
echo "📊 Visualizations (PNG files, 300 DPI):"
echo "  - 01_data_exploration.png"
echo "  - 02_model_performance.png"
echo "  - 03_uncertainty_analysis.png"
echo "  - 04_feature_importance.png"
echo "  - 05_spatial_analysis.png"
echo "  - 06_temporal_analysis.png"
echo "  - 07_residual_analysis.png"
echo ""
echo "🗺️  Interactive Maps (HTML, open in browser):"
echo "  - interactive_rental_map.html"
echo "  - prediction_accuracy_map.html"
echo ""
echo "📄 Reports:"
echo "  - EVALUATION_REPORT.txt (comprehensive text report)"
echo "  - report_summary.json (JSON summary)"
echo ""
echo "🤖 Trained Models:"
echo "  - gb_model.pkl (Gradient Boosting)"
echo "  - nn_model.h5 (Neural Network)"
echo "  - ridge_model.pkl (Ridge Regression - baseline)"
echo ""
echo "📁 Data & Metadata:"
echo "  - pipeline_results.pkl (all predictions and data)"
echo "  - evaluation_results.pkl (detailed metrics)"
echo "  - metadata.json (dataset information)"
echo ""
echo "================================================================================"
echo "Next steps:"
echo "  1. View visualizations in any image viewer"
echo "  2. Open HTML files in a web browser to explore interactive maps"
echo "  3. Read EVALUATION_REPORT.txt for detailed findings"
echo "  4. Use README.md for complete documentation"
echo "================================================================================"
