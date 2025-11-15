#!/usr/bin/env python3
"""
Churn ML Pipeline Tests - End-to-end pipeline validation

🎯 PURPOSE: Test the complete ML pipeline from data loading to API predictions
📊 FEATURES: Data pipeline validation, feature engineering checks, API endpoint smoke tests
🏗️ ARCHITECTURE: Pytest-based test suite with FastAPI test client integration
⚡ STATUS: Comprehensive test coverage for pipeline reliability and API functionality
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from fastapi.testclient import TestClient

# Import pipeline modules
from src.config import Config
from src.data_pipeline import load_raw_data, create_base_table, create_churn_labels
from src.feature_engineering import (
    load_base_table, create_rfm_features, encode_categorical_features,
    select_and_scale_features
)
from src.model_training import load_training_data, perform_time_aware_split, get_model
from src.model_serving import app, ModelServer


class TestDataPipeline:
    """Test data pipeline functionality"""

    def test_data_pipeline_runs(self, tmp_path):
        """Test that data pipeline runs end-to-end"""
        # Create temporary config
        config_content = f"""
raw_data_dir: "data/raw"
processed_data_dir: "{tmp_path}/processed"
models_dir: "{tmp_path}/models"
logs_dir: "{tmp_path}/logs"
base_table_path: "{tmp_path}/processed/base_table.parquet"
churn_cutoff_days: 90
observation_end_date: "2024-12-31"
"""
        config_file = tmp_path / "pipeline.yaml"
        config_file.write_text(config_content)

        # Initialize config
        config = Config(str(config_file))

        # Load raw data
        data = load_raw_data(config, lambda x: None)  # Mock logger

        # Verify data loaded
        assert 'customers' in data
        assert 'transactions' in data
        assert 'sessions' in data
        assert 'support_tickets' in data

        assert len(data['customers']) > 0
        assert len(data['transactions']) > 0

        # Create base table
        base_table = create_base_table(data, config, lambda x: None)

        # Verify base table structure
        assert 'customer_id' in base_table.columns
        assert 'signup_date' in base_table.columns
        assert 'transaction_count' in base_table.columns
        assert 'session_count' in base_table.columns

        # Create churn labels
        base_table_with_labels = create_churn_labels(base_table, config, lambda x: None)

        # Verify churn labels
        assert 'churn' in base_table_with_labels.columns
        assert 'days_since_last_activity' in base_table_with_labels.columns
        assert base_table_with_labels['churn'].isin([0, 1]).all()


class TestFeatureEngineering:
    """Test feature engineering functionality"""

    def test_feature_shapes(self, tmp_path):
        """Test that feature engineering produces correct shapes and types"""
        # Create mock base table
        mock_data = {
            'customer_id': [f'CUST_{i:05d}' for i in range(100)],
            'signup_date': ['2020-01-01'] * 100,
            'country': ['US'] * 100,
            'segment': ['premium'] * 100,
            'age': np.random.normal(35, 10, 100),
            'gender': ['M'] * 100,
            'transaction_count': np.random.poisson(5, 100),
            'total_amount': np.random.exponential(100, 100),
            'avg_amount': np.random.exponential(50, 100),
            'std_amount': np.random.exponential(20, 100),
            'session_count': np.random.poisson(10, 100),
            'total_pages_viewed': np.random.poisson(50, 100),
            'avg_pages_per_session': np.random.exponential(5, 100),
            'preferred_device': ['desktop'] * 100,
            'ticket_count': np.random.poisson(1, 100),
            'avg_satisfaction': np.random.uniform(1, 5, 100),
            'common_ticket_category': ['billing'] * 100,
            'last_transaction': ['2024-10-01'] * 100,
            'last_session': ['2024-10-01'] * 100,
            'last_ticket': ['2024-10-01'] * 100,
            'last_activity_date': pd.to_datetime(['2024-10-01'] * 100),
            'days_since_last_activity': np.random.exponential(30, 100),
            'churn': np.random.choice([0, 1], 100)
        }

        base_table = pd.DataFrame(mock_data)

        # Create temporary config
        config_content = """
raw_data_dir: "data/raw"
processed_data_dir: "/tmp/processed"
models_dir: "/tmp/models"
logs_dir: "/tmp/logs"
base_table_path: "/tmp/processed/base_table.parquet"
churn_cutoff_days: 90
observation_end_date: "2024-12-31"
"""
        config_file = tmp_path / "pipeline.yaml"
        config_file.write_text(config_content)
        config = Config(str(config_file))

        # Test RFM features
        rfm_features = create_rfm_features(base_table, config, lambda x: None)

        # Verify RFM features added
        rfm_columns = ['recency_days', 'frequency_transactions', 'monetary_total',
                      'recency_days_score', 'frequency_transactions_score', 'monetary_total_score', 'rfm_score']
        for col in rfm_columns:
            assert col in rfm_features.columns

        # Test categorical encoding
        encoded_features = encode_categorical_features(rfm_features, lambda x: None)

        # Verify encoded columns added
        encoded_columns = ['country_encoded', 'segment_encoded', 'gender_encoded',
                          'preferred_device_encoded', 'common_ticket_category_encoded']
        for col in encoded_columns:
            assert col in encoded_features.columns

        # Test feature selection and scaling
        features_df, target_series, feature_columns = select_and_scale_features(
            encoded_features, config, lambda x: None
        )

        # Verify output shapes
        assert len(features_df) == len(target_series)
        assert len(feature_columns) > 0
        assert features_df.shape[1] == len(feature_columns)

        # Verify target is binary
        assert target_series.isin([0, 1]).all()


class TestModelServing:
    """Test model serving API"""

    def test_predict_endpoint_smoke(self, tmp_path):
        """Smoke test for prediction endpoint"""
        # Create mock model and config
        import pickle
        from unittest.mock import Mock

        # Create temporary config
        config_content = f"""
model_save_path: "{tmp_path}/model.pkl"
metrics_save_path: "{tmp_path}/metrics.json"
model_type: "logistic_regression"
n_features: 25
training_timestamp: "2024-01-01T00:00:00"
"""
        config_file = tmp_path / "training.yaml"
        config_file.write_text(config_content)

        # Create mock model
        from sklearn.linear_model import LogisticRegression
        import numpy as np

        model = LogisticRegression(random_state=42)
        # Train on dummy data
        X_dummy = np.random.random((100, 25))
        y_dummy = np.random.choice([0, 1], 100)
        model.fit(X_dummy, y_dummy)

        # Save mock model
        with open(tmp_path / "model.pkl", 'wb') as f:
            pickle.dump(model, f)

        # Save mock metrics
        metrics = {
            "model_type": "logistic_regression",
            "n_features": 25,
            "training_timestamp": "2024-01-01T00:00:00"
        }
        import json
        with open(tmp_path / "metrics.json", 'w') as f:
            json.dump(metrics, f)

        # Create test client
        # Override the global server for testing
        from src import model_serving
        original_server = model_serving.server
        model_serving.server = ModelServer(str(config_file))

        client = TestClient(app)

        try:
            # Test health endpoint
            response = client.get("/health")
            assert response.status_code == 200
            health_data = response.json()
            assert health_data["status"] == "healthy"
            assert health_data["model_loaded"] is True

            # Test root endpoint
            response = client.get("/")
            assert response.status_code == 200

            # Test prediction endpoint with mock data
            test_payload = {
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
            }

            response = client.post("/predict", json=test_payload)
            assert response.status_code == 200

            prediction_data = response.json()
            assert "churn_probability" in prediction_data
            assert "churn_prediction" in prediction_data
            assert "confidence_score" in prediction_data
            assert 0 <= prediction_data["churn_probability"] <= 1
            assert prediction_data["churn_prediction"] in [0, 1]

        finally:
            # Restore original server
            model_serving.server = original_server


if __name__ == "__main__":
    pytest.main([__file__])
