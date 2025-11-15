#!/usr/bin/env python3
"""
Churn ML Pipeline Data Pipeline - ETL processing and churn label creation

🎯 PURPOSE: Extract, transform, and load customer data from raw CSVs into customer-level base table with churn labels
📊 FEATURES: Multi-table joins, date parsing, missing value handling, heuristic churn labeling, parquet output
🏗️ ARCHITECTURE: CLI-driven ETL pipeline with configurable processing rules and data validation
⚡ STATUS: Production-ready ETL pipeline with comprehensive error handling and data quality checks
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import click
import pandas as pd
import numpy as np

from .config import Config
from .utils.logging_utils import setup_logger, get_timestamped_log_path
from .utils.io_utils import load_csv_with_types, save_parquet, load_parquet
from .utils.validation_utils import (
    validate_required_columns,
    validate_date_columns,
    check_data_quality
)


def load_raw_data(config: Config, logger) -> dict:
    """
    Load all raw CSV files into DataFrames

    Args:
        config: Configuration object
        logger: Logger instance

    Returns:
        Dictionary of DataFrames keyed by table name
    """
    raw_data_dir = config.get('raw_data_dir')
    logger.info(f"Loading raw data from {raw_data_dir}")

    # Define expected schemas
    schemas = {
        'customers': {
            'customer_id': 'string',
            'signup_date': 'string',
            'country': 'string',
            'segment': 'string',
            'age': 'Int64',
            'gender': 'string'
        },
        'transactions': {
            'transaction_id': 'string',
            'customer_id': 'string',
            'amount': 'float64',
            'timestamp': 'string',
            'channel': 'string'
        },
        'sessions': {
            'session_id': 'string',
            'customer_id': 'string',
            'session_start': 'string',
            'session_end': 'string',
            'pages_viewed': 'Int64',
            'device': 'string'
        },
        'support_tickets': {
            'ticket_id': 'string',
            'customer_id': 'string',
            'created_at': 'string',
            'resolved_at': 'string',
            'category': 'string',
            'satisfaction_score': 'Int64'
        }
    }

    data = {}
    for table_name, schema in schemas.items():
        file_path = Path(raw_data_dir) / f"{table_name}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")

        logger.info(f"Loading {table_name} from {file_path}")
        df = load_csv_with_types(str(file_path), schema)

        # Validate required columns
        required_cols = list(schema.keys())
        validate_required_columns(df, required_cols)

        # Convert date columns
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        df = validate_date_columns(df, date_cols)

        data[table_name] = df
        logger.info(f"Loaded {len(df)} rows for {table_name}")

    return data


def create_base_table(data: dict, config: Config, logger) -> pd.DataFrame:
    """
    Create customer-level base table by joining all data sources

    Args:
        data: Dictionary of DataFrames
        config: Configuration object
        logger: Logger instance

    Returns:
        Customer-level base table DataFrame
    """
    logger.info("Creating customer-level base table")

    customers_df = data['customers'].copy()

    # Aggregate transactions per customer
    logger.info("Aggregating transaction data")
    transactions_agg = data['transactions'].groupby('customer_id').agg({
        'amount': ['count', 'sum', 'mean', 'std'],
        'timestamp': ['min', 'max'],
        'channel': lambda x: x.value_counts().index[0]  # Most common channel
    }).round(2)

    # Flatten column names
    transactions_agg.columns = [
        'transaction_count', 'total_amount', 'avg_amount', 'std_amount',
        'first_transaction', 'last_transaction', 'preferred_channel'
    ]
    transactions_agg = transactions_agg.reset_index()

    # Aggregate sessions per customer
    logger.info("Aggregating session data")
    sessions_agg = data['sessions'].groupby('customer_id').agg({
        'session_start': ['count', 'min', 'max'],
        'pages_viewed': ['sum', 'mean'],
        'device': lambda x: x.value_counts().index[0]  # Most common device
    })

    # Flatten column names
    sessions_agg.columns = [
        'session_count', 'first_session', 'last_session',
        'total_pages_viewed', 'avg_pages_per_session', 'preferred_device'
    ]
    sessions_agg = sessions_agg.reset_index()

    # Aggregate support tickets per customer
    logger.info("Aggregating support ticket data")
    support_agg = data['support_tickets'].groupby('customer_id').agg({
        'ticket_id': 'count',
        'created_at': ['min', 'max'],
        'category': lambda x: x.value_counts().index[0] if len(x) > 0 else None,  # Most common category
        'satisfaction_score': 'mean'
    })

    # Flatten column names
    support_agg.columns = [
        'ticket_count', 'first_ticket', 'last_ticket',
        'common_ticket_category', 'avg_satisfaction'
    ]
    support_agg = support_agg.reset_index()

    # Join all tables
    logger.info("Joining customer data with aggregated metrics")
    base_table = customers_df

    base_table = base_table.merge(transactions_agg, on='customer_id', how='left')
    base_table = base_table.merge(sessions_agg, on='customer_id', how='left')
    base_table = base_table.merge(support_agg, on='customer_id', how='left')

    # Fill missing values
    numeric_cols = base_table.select_dtypes(include=[np.number]).columns
    base_table[numeric_cols] = base_table[numeric_cols].fillna(0)

    categorical_cols = base_table.select_dtypes(include=['object', 'string']).columns
    base_table[categorical_cols] = base_table[categorical_cols].fillna('unknown')

    logger.info(f"Created base table with {len(base_table)} customers and {len(base_table.columns)} features")

    return base_table


def create_churn_labels(base_table: pd.DataFrame, config: Config, logger) -> pd.DataFrame:
    """
    Create churn labels using observation window and cutoff rules

    Args:
        base_table: Customer-level base table
        config: Configuration object
        logger: Logger instance

    Returns:
        Base table with churn labels
    """
    logger.info("Creating churn labels")

    churn_cutoff_days = config.get('churn_cutoff_days', 90)
    observation_end_date = pd.to_datetime(config.get('observation_end_date', datetime.now().date()))

    logger.info(f"Using churn cutoff: {churn_cutoff_days} days from observation end date: {observation_end_date}")

    # Calculate days since last activity for each customer
    activity_dates = []

    for _, row in base_table.iterrows():
        dates = []

        # Add last transaction date
        if pd.notna(row.get('last_transaction')):
            dates.append(pd.to_datetime(row['last_transaction']))

        # Add last session date
        if pd.notna(row.get('last_session')):
            dates.append(pd.to_datetime(row['last_session']))

        # Add last support ticket date
        if pd.notna(row.get('last_ticket')):
            dates.append(pd.to_datetime(row['last_ticket']))

        # If no activity dates, use signup date
        if not dates:
            dates.append(pd.to_datetime(row['signup_date']))

        latest_activity = max(dates)
        activity_dates.append(latest_activity)

    base_table = base_table.copy()
    base_table['last_activity_date'] = activity_dates

    # Calculate days since last activity
    base_table['days_since_last_activity'] = (
        observation_end_date - base_table['last_activity_date'].dt.date
    ).dt.days

    # Create churn label: 1 if no activity in last churn_cutoff_days
    base_table['churn'] = (base_table['days_since_last_activity'] >= churn_cutoff_days).astype(int)

    churn_rate = base_table['churn'].mean()
    logger.info(".2f")

    return base_table


@click.command()
@click.option('--config', required=True, help='Path to pipeline configuration YAML file')
def main(config: str):
    """Run the data pipeline ETL process"""
    # Setup logging
    log_file = get_timestamped_log_path('logs', 'data_pipeline')
    logger = setup_logger('data_pipeline', log_file)
    logger.info("Starting data pipeline execution")

    try:
        # Load configuration
        config_obj = Config(config)
        logger.info(f"Loaded configuration from {config}")

        # Load raw data
        data = load_raw_data(config_obj, logger)

        # Create base table
        base_table = create_base_table(data, config_obj, logger)

        # Create churn labels
        base_table_with_labels = create_churn_labels(base_table, config_obj, logger)

        # Validate data quality
        quality_report = check_data_quality(base_table_with_labels)
        logger.info(f"Data quality check: {quality_report}")

        # Save processed data
        output_path = config_obj.get('base_table_path')
        save_parquet(base_table_with_labels, output_path)
        logger.info(f"Saved base table to {output_path}")

        logger.info("Data pipeline completed successfully")

    except Exception as e:
        logger.error(f"Data pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
