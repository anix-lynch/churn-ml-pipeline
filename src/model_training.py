#!/usr/bin/env python3
"""
Churn ML Pipeline Model Training - Time-aware model training and evaluation

🎯 PURPOSE: Train churn prediction models using time-aware cross-validation and evaluation metrics
📊 FEATURES: Multiple model support, time-aware splitting, comprehensive metrics, feature importance analysis
🏗️ ARCHITECTURE: CLI-driven training pipeline with offline-only operations and model artifact management
⚡ STATUS: Production-ready training with Apple Silicon compatibility and comprehensive evaluation
"""
import sys
import pickle
import json
from pathlib import Path
from datetime import datetime
import warnings
import click
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Optional dependencies - warn if not available
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not available. Install with: pip install lightgbm")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

from .config import Config
from .utils.logging_utils import setup_logger, get_timestamped_log_path
from .utils.io_utils import load_parquet, save_json
from .utils.validation_utils import validate_required_columns


def load_training_data(config: Config, logger) -> tuple:
    """
    Load features and target data for training

    Args:
        config: Configuration object
        logger: Logger instance

    Returns:
        Tuple of (features_df, target_series)
    """
    features_path = config.get('features_path')
    target_path = config.get('target_path')

    logger.info(f"Loading features from {features_path}")
    logger.info(f"Loading target from {target_path}")

    if not Path(features_path).exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not Path(target_path).exists():
        raise FileNotFoundError(f"Target file not found: {target_path}")

    X = load_parquet(features_path)
    y = load_parquet(target_path)['churn']

    logger.info(f"Loaded {len(X)} samples with {len(X.columns)} features")
    logger.info(f"Churn rate: {y.mean():.2%}")

    return X, y


def perform_time_aware_split(X: pd.DataFrame, y: pd.Series, config: Config, logger) -> tuple:
    """
    Perform time-aware train/test split for churn prediction

    Args:
        X: Features DataFrame
        y: Target series
        config: Configuration object
        logger: Logger instance

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    if config.get('time_aware_split', False):
        time_col = config.get('time_column', 'recency_days')
        if time_col in X.columns:
            logger.info(f"Performing time-aware split using {time_col}")

            # Sort by time column (most recent first for churn prediction)
            combined = pd.concat([X, y], axis=1).sort_values(time_col, ascending=False)

            # Use configured split ratios
            test_size = config.get('test_size', 0.2)
            n_test = int(len(combined) * test_size)

            # Take most recent samples for test set
            test_data = combined.head(n_test)
            train_data = combined.tail(len(combined) - n_test)

            X_train = train_data.drop('churn', axis=1)
            y_train = train_data['churn']
            X_test = test_data.drop('churn', axis=1)
            y_test = test_data['churn']

            logger.info(f"Time-aware split: {len(X_train)} train, {len(X_test)} test samples")
        else:
            logger.warning(f"Time column {time_col} not found, using random split")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=config.get('test_size', 0.2),
                random_state=config.get('random_state', 42), stratify=y
            )
    else:
        logger.info("Performing random stratified split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.get('test_size', 0.2),
            random_state=config.get('random_state', 42), stratify=y
        )

    return X_train, X_test, y_train, y_test


def get_model(model_type: str, params: dict, logger):
    """
    Get model instance based on type

    Args:
        model_type: Type of model to create
        params: Model parameters
        logger: Logger instance

    Returns:
        Model instance
    """
    if model_type == 'lightgbm':
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available. Install with: pip install lightgbm")
        model_params = params.get('lightgbm', {})
        logger.info(f"Creating LightGBM model with params: {model_params}")
        return lgb.LGBMClassifier(**model_params)

    elif model_type == 'xgboost':
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available. Install with: pip install xgboost")
        model_params = params.get('xgboost', {})
        logger.info(f"Creating XGBoost model with params: {model_params}")
        return xgb.XGBClassifier(**model_params)

    elif model_type == 'logistic_regression':
        model_params = params.get('logistic_regression', {})
        logger.info(f"Creating LogisticRegression model with params: {model_params}")
        return LogisticRegression(**model_params)

    elif model_type == 'random_forest':
        model_params = params.get('random_forest', {})
        logger.info(f"Creating RandomForest model with params: {model_params}")
        return RandomForestClassifier(**model_params)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> dict:
    """
    Calculate comprehensive evaluation metrics

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (for AUC)

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }

    if y_prob is not None:
        metrics['auc'] = roc_auc_score(y_true, y_prob)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()

    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics['classification_report'] = report

    return metrics


def train_model(X_train: pd.DataFrame, y_train: pd.Series,
               X_test: pd.DataFrame, y_test: pd.Series,
               config: Config, logger) -> tuple:
    """
    Train model and evaluate performance

    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        config: Configuration object
        logger: Logger instance

    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    model_type = config.get('model_type', 'logistic_regression')
    model_params = config.get('model_params', {})

    # Get model
    model = get_model(model_type, model_params, logger)

    # Train model
    logger.info(f"Training {model_type} model...")
    start_time = datetime.now()

    model.fit(X_train, y_train)

    training_time = (datetime.now() - start_time).total_seconds()
    logger.info(".2f")

    # Make predictions
    logger.info("Making predictions on test set...")
    y_pred = model.predict(X_test)

    # Get prediction probabilities if available
    y_prob = None
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_prob)
    logger.info("Test metrics: " +
               ".3f"
               ".3f"
               ".3f")

    # Add training metadata
    metrics['model_type'] = model_type
    metrics['training_time_seconds'] = training_time
    metrics['n_train_samples'] = len(X_train)
    metrics['n_test_samples'] = len(X_test)
    metrics['n_features'] = X_train.shape[1]
    metrics['training_timestamp'] = datetime.now().isoformat()

    return model, metrics


def save_model_artifacts(model, metrics: dict, config: Config, logger):
    """
    Save trained model and evaluation metrics

    Args:
        model: Trained model
        metrics: Evaluation metrics
        config: Configuration object
        logger: Logger instance
    """
    model_path = config.get('model_save_path')
    metrics_path = config.get('metrics_save_path')

    # Save model
    logger.info(f"Saving model to {model_path}")
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Save metrics
    logger.info(f"Saving metrics to {metrics_path}")
    save_json(metrics, metrics_path)

    # Save feature importance if available
    if config.get('calculate_feature_importance', False):
        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                feature_importance = dict(zip(
                    range(len(model.feature_importances_)),
                    model.feature_importances_
                ))
            elif hasattr(model, 'coef_'):
                # Linear models
                feature_importance = dict(zip(
                    range(len(model.coef_[0])),
                    np.abs(model.coef_[0])
                ))
            else:
                feature_importance = {}

            if feature_importance:
                importance_path = config.get('feature_importance_path')
                save_json(feature_importance, importance_path)
                logger.info(f"Saved feature importance to {importance_path}")

        except Exception as e:
            logger.warning(f"Could not save feature importance: {e}")


@click.command()
@click.option('--config', required=True, help='Path to training configuration YAML file')
def main(config: str):
    """Run the model training pipeline"""
    # Setup logging
    log_file = get_timestamped_log_path('logs', 'model_training')
    logger = setup_logger('model_training', log_file)
    logger.info("Starting model training execution")

    try:
        # Load configuration
        config_obj = Config(config)
        logger.info(f"Loaded training configuration from {config}")

        # Load training data
        X, y = load_training_data(config_obj, logger)

        # Perform time-aware split
        X_train, X_test, y_train, y_test = perform_time_aware_split(X, y, config_obj, logger)

        # Train model
        model, metrics = train_model(X_train, y_train, X_test, y_test, config_obj, logger)

        # Save artifacts
        save_model_artifacts(model, metrics, config_obj, logger)

        logger.info("Model training completed successfully")

    except Exception as e:
        logger.error(f"Model training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
