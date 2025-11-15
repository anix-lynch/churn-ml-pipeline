#!/usr/bin/env python3
"""
Churn ML Pipeline Feature Engineering - RFM and behavioral feature creation

🎯 PURPOSE: Transform raw customer data into ML-ready features using RFM analysis and behavioral metrics
📊 FEATURES: RFM scoring, recency/frequency/monetary features, session engagement metrics, support interaction features
🏗️ ARCHITECTURE: Feature engineering pipeline with automated encoding, scaling, and feature selection
⚡ STATUS: Production-ready feature engineering with comprehensive behavioral feature extraction
"""
import sys
from pathlib import Path
from datetime import datetime
import click
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import Config
from .utils.logging_utils import setup_logger, get_timestamped_log_path
from .utils.io_utils import load_parquet, save_parquet
from .utils.validation_utils import validate_required_columns


def load_base_table(config: Config, logger) -> pd.DataFrame:
    """
    Load the base table created by data pipeline

    Args:
        config: Configuration object
        logger: Logger instance

    Returns:
        Base table DataFrame
    """
    base_table_path = config.get('base_table_path')
    logger.info(f"Loading base table from {base_table_path}")

    if not Path(base_table_path).exists():
        raise FileNotFoundError(f"Base table not found: {base_table_path}")

    df = load_parquet(base_table_path)
    logger.info(f"Loaded base table with {len(df)} customers and {len(df.columns)} columns")

    return df


def create_rfm_features(df: pd.DataFrame, config: Config, logger) -> pd.DataFrame:
    """
    Create RFM (Recency, Frequency, Monetary) features

    Args:
        df: Base table DataFrame
        config: Configuration object
        logger: Logger instance

    Returns:
        DataFrame with RFM features added
    """
    logger.info("Creating RFM features")

    df_rfm = df.copy()
    observation_end_date = pd.to_datetime(config.get('observation_end_date', datetime.now().date()))

    # Recency: Days since last activity
    df_rfm['recency_days'] = df_rfm['days_since_last_activity']

    # Frequency: Transaction and session counts
    df_rfm['frequency_transactions'] = df_rfm['transaction_count']
    df_rfm['frequency_sessions'] = df_rfm['session_count']
    df_rfm['frequency_support'] = df_rfm['ticket_count']

    # Monetary: Transaction amounts
    df_rfm['monetary_total'] = df_rfm['total_amount']
    df_rfm['monetary_avg'] = df_rfm['avg_amount']

    # RFM Scores (quintiles)
    for feature in ['recency_days', 'frequency_transactions', 'monetary_total']:
        if feature == 'recency_days':
            # Lower recency = higher score (more recent = better)
            df_rfm[f'{feature}_score'] = pd.qcut(df_rfm[feature], 5, labels=[5, 4, 3, 2, 1], duplicates='drop')
        else:
            # Higher frequency/monetary = higher score
            df_rfm[f'{feature}_score'] = pd.qcut(df_rfm[feature], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')

        df_rfm[f'{feature}_score'] = df_rfm[f'{feature}_score'].astype(int)

    # Combined RFM score
    df_rfm['rfm_score'] = (
        df_rfm['recency_days_score'] +
        df_rfm['frequency_transactions_score'] +
        df_rfm['monetary_total_score']
    )

    logger.info("RFM features created")

    return df_rfm


def create_session_features(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Create session-based behavioral features

    Args:
        df: DataFrame with session data
        logger: Logger instance

    Returns:
        DataFrame with session features added
    """
    logger.info("Creating session engagement features")

    df_session = df.copy()

    # Session engagement metrics
    df_session['session_engagement_rate'] = df_session['avg_pages_per_session'] / 10  # Normalize by typical page count
    df_session['session_engagement_rate'] = df_session['session_engagement_rate'].clip(0, 1)

    # Session frequency patterns
    df_session['days_active'] = (pd.to_datetime(df_session['last_session']) -
                                pd.to_datetime(df_session['first_session'])).dt.days
    df_session['session_frequency_per_day'] = df_session['session_count'] / (df_session['days_active'] + 1)

    # Device preference encoding (will be label encoded later)
    # This is already handled in base table aggregation

    logger.info("Session features created")

    return df_session


def create_support_features(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Create support interaction features

    Args:
        df: DataFrame with support data
        logger: Logger instance

    Returns:
        DataFrame with support features added
    """
    logger.info("Creating support interaction features")

    df_support = df.copy()

    # Support engagement metrics
    df_support['support_engagement_rate'] = df_support['ticket_count'] / (df_support['days_since_last_activity'] + 1)
    df_support['support_engagement_rate'] = df_support['support_engagement_rate'].clip(0, 1)

    # Support satisfaction (already averaged in base table)
    # Category diversity (will be encoded)

    logger.info("Support features created")

    return df_support


def create_temporal_features(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Create temporal and lifecycle features

    Args:
        df: DataFrame with date features
        logger: Logger instance

    Returns:
        DataFrame with temporal features added
    """
    logger.info("Creating temporal and lifecycle features")

    df_temp = df.copy()

    # Customer lifecycle metrics
    df_temp['account_age_days'] = (pd.to_datetime(df_temp['last_activity_date']) -
                                  pd.to_datetime(df_temp['signup_date'])).dt.days

    df_temp['activity_span_days'] = (pd.to_datetime(df_temp['last_activity_date']) -
                                    pd.to_datetime(df_temp['first_transaction'])).dt.days

    # Activity intensity over time
    df_temp['transaction_intensity'] = df_temp['transaction_count'] / (df_temp['account_age_days'] + 1)
    df_temp['session_intensity'] = df_temp['session_count'] / (df_temp['account_age_days'] + 1)

    logger.info("Temporal features created")

    return df_temp


def encode_categorical_features(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Encode categorical features for ML

    Args:
        df: DataFrame with categorical features
        logger: Logger instance

    Returns:
        DataFrame with encoded categorical features
    """
    logger.info("Encoding categorical features")

    df_encoded = df.copy()

    # Label encode categorical columns
    categorical_cols = ['country', 'segment', 'gender', 'preferred_channel',
                       'preferred_device', 'common_ticket_category']

    label_encoders = {}

    for col in categorical_cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            # Handle missing values
            df_encoded[col] = df_encoded[col].fillna('unknown')
            df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le

    logger.info(f"Encoded {len(categorical_cols)} categorical features")

    return df_encoded


def select_and_scale_features(df: pd.DataFrame, config: Config, logger) -> tuple:
    """
    Select final features and scale them

    Args:
        df: DataFrame with all features
        config: Configuration object
        logger: Logger instance

    Returns:
        Tuple of (features_df, target_series, feature_columns)
    """
    logger.info("Selecting and scaling final features")

    # Define feature columns (exclude IDs, raw dates, target)
    exclude_cols = [
        'customer_id', 'signup_date', 'first_transaction', 'last_transaction',
        'first_session', 'last_session', 'first_ticket', 'last_ticket',
        'last_activity_date', 'churn'
    ]

    # Add original categorical columns (keep encoded versions)
    original_categoricals = ['country', 'segment', 'gender', 'preferred_channel',
                           'preferred_device', 'common_ticket_category']
    exclude_cols.extend(original_categoricals)

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    logger.info(f"Selected {len(feature_cols)} features: {feature_cols}")

    # Extract features and target
    X = df[feature_cols].copy()
    y = df['churn'].copy()

    # Handle any remaining missing values
    X = X.fillna(0)

    # Scale numeric features
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        logger.info(f"Scaled {len(numeric_cols)} numeric features")

    return X, y, feature_cols


@click.command()
@click.option('--config', required=True, help='Path to pipeline configuration YAML file')
def main(config: str):
    """Run the feature engineering pipeline"""
    # Setup logging
    log_file = get_timestamped_log_path('logs', 'feature_engineering')
    logger = setup_logger('feature_engineering', log_file)
    logger.info("Starting feature engineering execution")

    try:
        # Load configuration
        config_obj = Config(config)
        logger.info(f"Loaded configuration from {config}")

        # Load base table
        df = load_base_table(config_obj, logger)

        # Create RFM features
        df = create_rfm_features(df, config_obj, logger)

        # Create behavioral features
        df = create_session_features(df, logger)
        df = create_support_features(df, logger)
        df = create_temporal_features(df, logger)

        # Encode categorical features
        df = encode_categorical_features(df, logger)

        # Select and scale final features
        features_df, target_series, feature_columns = select_and_scale_features(df, config_obj, logger)

        # Validate features
        validate_required_columns(features_df, feature_columns)

        # Save features and target
        features_path = config_obj.get('features_path')
        target_path = config_obj.get('target_path')

        save_parquet(features_df, features_path)
        save_parquet(pd.DataFrame({'churn': target_series}), target_path)

        logger.info(f"Saved {len(features_df)} features to {features_path}")
        logger.info(f"Saved target to {target_path}")
        logger.info(f"Feature engineering completed with {len(feature_columns)} features")

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
