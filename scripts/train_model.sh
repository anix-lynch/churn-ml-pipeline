#!/bin/bash
# Churn ML Model Training Script
# Trains and evaluates churn prediction models

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

echo "🤖 Starting Churn Model Training..."
echo "📁 Project root: $PROJECT_ROOT"

# Set Python path
export PYTHONPATH="$PROJECT_ROOT"

# Configuration paths
TRAINING_CONFIG="$PROJECT_ROOT/configs/training.yaml"

# Check if configuration exists
if [ ! -f "$TRAINING_CONFIG" ]; then
    echo "❌ Training configuration not found: $TRAINING_CONFIG"
    exit 1
fi

# Check if features exist
if [ ! -f "data/processed/features.parquet" ] || [ ! -f "data/processed/target.parquet" ]; then
    echo "❌ Processed data not found. Please run pipeline first:"
    echo "   ./scripts/run_pipeline.sh"
    exit 1
fi

echo "🎯 Training model..."
python -m src.model_training --config "$TRAINING_CONFIG"

if [ $? -ne 0 ]; then
    echo "❌ Model training failed!"
    exit 1
fi

echo "✅ Model training completed successfully!"
echo ""
echo "📋 Generated files:"
echo "  - models/model.pkl (trained model)"
echo "  - models/metrics.json (evaluation metrics)"
echo "  - models/feature_importance.json (if applicable)"
echo ""
echo "📊 Training logs: logs/model_training_*.log"
echo ""
echo "🚀 Next steps:"
echo "  1. Review metrics in models/metrics.json"
echo "  2. Start API server: ./scripts/run_api.sh"
echo "  3. Test predictions: curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d @test_payload.json"
