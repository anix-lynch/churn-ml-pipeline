#!/usr/bin/env python3
"""
Churn ML Pipeline Validation Utilities - Data quality assurance and input validation

🎯 PURPOSE: Validate data integrity, schema compliance, and prediction inputs across the ML pipeline
📊 FEATURES: Schema validation, range checking, missing value detection, prediction API input validation
🏗️ ARCHITECTURE: Comprehensive validation framework with detailed error reporting and data quality metrics
⚡ STATUS: Production-ready validation with configurable thresholds and automated quality reporting
"""
import pandas as pd
from typing import List, Dict, Any
import numpy as np


def validate_required_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    """
    Validate that required columns exist in DataFrame

    Args:
        df: DataFrame to validate
        required_cols: List of required column names

    Raises:
        ValueError: If any required columns are missing
    """
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def validate_date_columns(df: pd.DataFrame, date_cols: List[str]) -> pd.DataFrame:
    """
    Validate and convert date columns

    Args:
        df: DataFrame with date columns
        date_cols: List of date column names

    Returns:
        DataFrame with converted date columns
    """
    df_copy = df.copy()
    for col in date_cols:
        if col in df_copy.columns:
            df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
            if df_copy[col].isnull().any():
                raise ValueError(f"Invalid dates found in column: {col}")
    return df_copy


def validate_numeric_ranges(df: pd.DataFrame, range_checks: Dict[str, Dict[str, float]]) -> None:
    """
    Validate numeric column ranges

    Args:
        df: DataFrame to validate
        range_checks: Dict mapping column names to min/max constraints

    Raises:
        ValueError: If values are outside specified ranges
    """
    for col, constraints in range_checks.items():
        if col in df.columns:
            min_val = constraints.get('min')
            max_val = constraints.get('max')

            if min_val is not None:
                if (df[col] < min_val).any():
                    raise ValueError(f"Column {col} has values below minimum {min_val}")

            if max_val is not None:
                if (df[col] > max_val).any():
                    raise ValueError(f"Column {col} has values above maximum {max_val}")


def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform basic data quality checks

    Args:
        df: DataFrame to check

    Returns:
        Dictionary with quality metrics
    """
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.astype(str).to_dict()
    }

    # Add percentage missing
    quality_report['missing_percentage'] = {
        col: (count / len(df) * 100) if len(df) > 0 else 0
        for col, count in quality_report['missing_values'].items()
    }

    return quality_report


def validate_prediction_input(data: Dict[str, Any], feature_columns: List[str]) -> pd.DataFrame:
    """
    Validate prediction API input data

    Args:
        data: Input data dictionary
        feature_columns: Expected feature column names

    Returns:
        Validated DataFrame

    Raises:
        ValueError: If input is invalid
    """
    if not isinstance(data, dict):
        raise ValueError("Input must be a dictionary")

    # Check for required features
    missing_features = [col for col in feature_columns if col not in data]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    # Convert to DataFrame
    try:
        df = pd.DataFrame([data])
    except Exception as e:
        raise ValueError(f"Failed to convert input to DataFrame: {e}")

    return df
