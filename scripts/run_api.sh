#!/bin/bash
# Churn ML API Server Launcher
# Starts the FastAPI service for churn predictions

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

echo "🌐 Starting Churn Prediction API..."
echo "📁 Project root: $PROJECT_ROOT"

# Set Python path
export PYTHONPATH="$PROJECT_ROOT"

# Configuration paths
TRAINING_CONFIG="$PROJECT_ROOT/configs/training.yaml"

# Check if model exists
if [ ! -f "models/model.pkl" ]; then
    echo "❌ Trained model not found: models/model.pkl"
    echo "   Please train the model first:"
    echo "   ./scripts/train_model.sh"
    exit 1
fi

# Default values
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "🔧 Configuration:"
echo "   Host: $HOST"
echo "   Port: $PORT"
echo "   Model: models/model.pkl"
echo ""

# Start the API server
echo "🚀 Starting FastAPI server..."
python -m src.model_serving --config "$TRAINING_CONFIG" --host "$HOST" --port "$PORT"
