#!/bin/bash
# Churn ML Pipeline Runner
# Executes the complete ETL and feature engineering pipeline

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

echo "🚀 Starting Churn ML Pipeline..."
echo "📁 Project root: $PROJECT_ROOT"

# Set Python path
export PYTHONPATH="$PROJECT_ROOT"

# Configuration paths
PIPELINE_CONFIG="$PROJECT_ROOT/configs/pipeline.yaml"

# Check if configuration exists
if [ ! -f "$PIPELINE_CONFIG" ]; then
    echo "❌ Pipeline configuration not found: $PIPELINE_CONFIG"
    exit 1
fi

# Check if raw data exists
if [ ! -f "data/raw/customers.csv" ]; then
    echo "❌ Raw data not found. Please ensure data/raw/ contains the CSV files."
    exit 1
fi

echo "📊 Step 1: Running data pipeline (ETL)..."
python -m src.data_pipeline --config "$PIPELINE_CONFIG"

if [ $? -ne 0 ]; then
    echo "❌ Data pipeline failed!"
    exit 1
fi

echo "✨ Step 2: Running feature engineering..."
python -m src.feature_engineering --config "$PIPELINE_CONFIG"

if [ $? -ne 0 ]; then
    echo "❌ Feature engineering failed!"
    exit 1
fi

echo "✅ Pipeline completed successfully!"
echo ""
echo "📋 Generated files:"
echo "  - data/processed/base_table.parquet"
echo "  - data/processed/features.parquet"
echo "  - data/processed/target.parquet"
echo ""
echo "📈 Next steps:"
echo "  1. Review logs in logs/ directory"
echo "  2. Run training: ./scripts/train_model.sh"
echo "  3. Start API: ./scripts/run_api.sh"
