# Churn ML Pipeline 🚀

End-to-end machine learning pipeline for customer churn prediction using synthetic behavioral data.

## 🎯 Overview

This project implements a complete ML pipeline that:
- **Extracts** customer data from multiple sources (demographics, transactions, sessions, support)
- **Transforms** raw data into customer-level features using RFM analysis and behavioral metrics
- **Trains** churn prediction models with time-aware validation
- **Serves** predictions via REST API with comprehensive input validation

**Key Features:**
- Synthetic offline-only dataset (no external API calls)
- Time-aware train/test splits for realistic churn modeling
- Multiple model support (Logistic Regression, LightGBM, XGBoost)
- FastAPI service with automatic input validation
- Docker deployment with health checks
- Comprehensive testing and logging

## 🏗️ Architecture

```
Raw Data (CSV) → ETL Pipeline → Feature Engineering → Model Training → API Service
     ↓              ↓              ↓                     ↓              ↓
customers.csv  data_pipeline.py  RFM + behavioral    LightGBM     FastAPI server
transactions.csv                 features            metrics       /predict endpoint
sessions.csv                     target              model.pkl     /health endpoint
support_tickets.csv              parquet files       logs          Docker container
```

## 📋 Prerequisites

- **Python 3.9+**
- **Docker & Docker Compose** (for containerized deployment)
- **macOS Apple Silicon** compatible (tested on M1/M2/M3)

## 🚀 Quick Start

### 1. Clone & Setup
```bash
cd /Users/anixlynch/dev/Takehome1_endtoend/
pip install -r requirements.txt
```

### 2. Run Complete Pipeline
```bash
# Execute ETL + feature engineering
./scripts/run_pipeline.sh
```

### 3. Train Model
```bash
# Train churn prediction model
./scripts/train_model.sh
```

### 4. Start API Server
```bash
# Launch prediction API
./scripts/run_api.sh
```

### 5. Make Predictions
```bash
# Test the API
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "country_encoded": 1,
    "segment_encoded": 2,
    "age": 35.0,
    "gender_encoded": 0,
    "transaction_count": 15.0,
    "total_amount": 1250.50,
    "avg_amount": 83.37,
    "std_amount": 45.20,
    "session_count": 45.0,
    "total_pages_viewed": 180.0,
    "avg_pages_per_session": 4.0,
    "preferred_device_encoded": 1,
    "ticket_count": 2.0,
    "avg_satisfaction": 4.0,
    "common_ticket_category_encoded": 1,
    "recency_days": 15.0,
    "recency_days_score": 5,
    "frequency_transactions_score": 4,
    "monetary_total_score": 3,
    "rfm_score": 12,
    "session_engagement_rate": 0.4,
    "session_frequency_per_day": 0.8,
    "support_engagement_rate": 0.02,
    "account_age_days": 365.0,
    "activity_span_days": 300.0,
    "transaction_intensity": 0.041,
    "session_intensity": 0.123,
    "preferred_channel_encoded": 2
  }'
```

## 📊 Data Pipeline

### Input Data
- **`customers.csv`**: Customer demographics (ID, signup date, country, segment, age, gender)
- **`transactions.csv`**: Purchase history (ID, customer, amount, timestamp, channel)
- **`sessions.csv`**: Website/app usage (ID, customer, start/end times, pages viewed, device)
- **`support_tickets.csv`**: Customer service interactions (ID, customer, created/resolved dates, category, satisfaction)

### ETL Process
```bash
# Load raw CSVs with type validation
# Clean missing values and parse dates
# Join into customer-level base table
# Apply churn heuristic (90 days inactivity cutoff)
# Save as parquet: data/processed/base_table.parquet
python -m src.data_pipeline --config configs/pipeline.yaml
```

### Feature Engineering
```bash
# Load base_table.parquet
# Create RFM (Recency/Frequency/Monetary) features
# Extract behavioral metrics (session engagement, support patterns)
# Encode categorical variables
# Scale numeric features
# Save: data/processed/features.parquet + target.parquet
python -m src.feature_engineering --config configs/pipeline.yaml
```

## 🤖 Model Training

### Supported Models
- **Logistic Regression** (baseline, always available)
- **LightGBM** (gradient boosting, optional)
- **XGBoost** (alternative boosting, optional)
- **Random Forest** (ensemble baseline)

### Training Process
```bash
# Load features + target
# Time-aware train/test split (respecting temporal order)
# Train selected model with cross-validation
# Evaluate on holdout set
# Save model.pkl + metrics.json + feature_importance.json
python -m src.model_training --config configs/training.yaml
```

### Model Configuration
```yaml
# configs/training.yaml
model_type: "lightgbm"  # or logistic_regression, xgboost, random_forest
test_size: 0.2
random_state: 42
time_aware_split: true
primary_metric: "auc"
```

## 🌐 API Service

### Endpoints

#### `GET /health`
Health check endpoint returning service status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "feature_count": 25,
  "last_prediction": "2024-01-15T10:30:00",
  "uptime_seconds": 3600.0
}
```

#### `POST /predict`
Churn prediction endpoint with comprehensive input validation.

**Request Body:** See example in Quick Start section above

**Response:**
```json
{
  "churn_probability": 0.23,
  "churn_prediction": 0,
  "confidence_score": 0.54,
  "prediction_timestamp": "2024-01-15T10:30:00",
  "model_version": "lightgbm_2024-01-01"
}
```

#### `GET /`
API information and available endpoints.

#### `GET /docs`
Interactive API documentation (Swagger UI).

## 🐳 Docker Deployment

### Local Development
```bash
# Start API service
docker-compose up churn-api

# Run with profiles for different tasks
docker-compose --profile etl run --rm churn-etl
docker-compose --profile training run --rm churn-training
```

### Production Deployment
```bash
# Build and run
docker-compose -f docker-compose.yml up -d

# Check logs
docker-compose logs churn-api

# Scale services
docker-compose up -d --scale churn-api=3
```

### Docker Image Structure
- **Base**: Python 3.9 slim with ML dependencies
- **App**: Modular FastAPI application
- **Data**: Persistent volumes for models and processed data
- **Health**: Built-in health checks and monitoring

## 🧪 Testing

### Run Test Suite
```bash
# Execute all tests
pytest tests/test_pipeline.py -v

# Run specific test
pytest tests/test_pipeline.py::TestDataPipeline::test_data_pipeline_runs -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Coverage
- **Data Pipeline**: ETL process validation and data quality checks
- **Feature Engineering**: Shape validation and feature transformations
- **API Service**: Endpoint smoke tests with FastAPI test client

## 📈 Monitoring & Logging

### Log Files
- **Data Pipeline**: `logs/data_pipeline_YYYYMMDD_HHMMSS.log`
- **Feature Engineering**: `logs/feature_engineering_YYYYMMDD_HHMMSS.log`
- **Model Training**: `logs/model_training_YYYYMMDD_HHMMSS.log`
- **API Service**: Console output (configurable)

### Metrics Tracking
- **Training Metrics**: `models/metrics.json`
- **Feature Importance**: `models/feature_importance.json` (if available)
- **Health Status**: API `/health` endpoint

## 🔍 Data Exploration

### Jupyter Notebook
```bash
# Launch exploration notebook
jupyter notebook notebooks/exploration.ipynb

# Or with specific port
jupyter notebook --port 8889 notebooks/exploration.ipynb
```

### Key Insights
- Customer segmentation by geography and behavior
- Transaction patterns across different channels
- Session engagement metrics by device type
- Support ticket resolution analysis
- Churn correlation analysis with behavioral features

## ⚙️ Configuration

### Pipeline Configuration (`configs/pipeline.yaml`)
```yaml
# Data directories and file paths
raw_data_dir: "data/raw"
processed_data_dir: "data/processed"

# Churn definition
churn_cutoff_days: 90
observation_end_date: "2024-12-31"

# Feature engineering parameters
feature_engineering:
  rfm_quintiles: 5
  scaling_method: "standard"
```

### Training Configuration (`configs/training.yaml`)
```yaml
# Model selection and hyperparameters
model_type: "lightgbm"
model_params:
  lightgbm:
    objective: "binary"
    learning_rate: 0.1

# Validation strategy
cv_folds: 5
time_aware_split: true
primary_metric: "auc"
```

## 🚨 Design Decisions & Tradeoffs

### ✅ Strengths
- **Offline-Only**: No external API dependencies or costs
- **Time-Aware**: Realistic churn modeling with temporal validation
- **Production-Ready**: Comprehensive error handling and logging
- **Apple Silicon Compatible**: Optimized for modern Mac hardware
- **Modular Architecture**: Easy to extend and maintain

### ⚠️ Limitations & Tradeoffs

#### Synthetic Data
- **Pro**: Controlled, reproducible, privacy-safe
- **Con**: May not capture real-world complexity or edge cases
- **Mitigation**: Based on realistic statistical distributions

#### Model Selection
- **Default**: Logistic Regression (always available, interpretable)
- **Advanced**: LightGBM/XGBoost (better performance, optional dependencies)
- **Tradeoff**: Performance vs. dependency management

#### Churn Definition
- **Simple Heuristic**: 90-day inactivity cutoff
- **Pro**: Clear, actionable definition
- **Con**: May not match business-specific churn criteria
- **Mitigation**: Configurable cutoff in pipeline.yaml

#### Feature Engineering
- **RFM Focus**: Proven marketing science approach
- **Behavioral Features**: Session and support-derived metrics
- **Tradeoff**: Feature count vs. model complexity

## 🤝 Contributing

### Development Setup
```bash
# Install in development mode
pip install -e .

# Run tests continuously
pytest-watch tests/

# Format code
black src/ tests/
isort src/ tests/
```

### Code Quality
- **Type Hints**: Full type annotation for better IDE support
- **Docstrings**: Google-style documentation for all functions
- **Testing**: Pytest with comprehensive coverage
- **Linting**: Black formatting, isort imports

## 📄 License

This project is provided as-is for educational and demonstration purposes.

## 🎯 Next Steps

1. **Explore Data**: Run `notebooks/exploration.ipynb`
2. **Customize Features**: Modify `src/feature_engineering.py`
3. **Experiment with Models**: Update `configs/training.yaml`
4. **Deploy to Production**: Use Docker Compose in production environment
5. **Add Monitoring**: Integrate with logging and metrics collection systems

---

**Built with ❤️ for ML engineering take-home challenges**
