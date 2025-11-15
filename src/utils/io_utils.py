#!/usr/bin/env python3
"""
Churn ML Pipeline I/O Utilities - Data persistence and configuration management

🎯 PURPOSE: Handle all file I/O operations for the ML pipeline including parquet, CSV, YAML, and JSON files
📊 FEATURES: Type-safe CSV loading, parquet compression, YAML config parsing, JSON metrics storage
🏗️ ARCHITECTURE: Unified interface for data persistence with automatic directory creation and error handling
⚡ STATUS: Optimized for large datasets with efficient parquet storage and memory management
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import yaml
import json


def load_csv_with_types(file_path: str, dtype_map: Dict[str, str] = None) -> pd.DataFrame:
    """
    Load CSV with specified data types

    Args:
        file_path: Path to CSV file
        dtype_map: Optional mapping of column names to data types

    Returns:
        Loaded DataFrame
    """
    if dtype_map:
        return pd.read_csv(file_path, dtype=dtype_map)
    return pd.read_csv(file_path)


def save_parquet(df: pd.DataFrame, file_path: str) -> None:
    """
    Save DataFrame as parquet file

    Args:
        df: DataFrame to save
        file_path: Output path
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(file_path, index=False)


def load_parquet(file_path: str) -> pd.DataFrame:
    """
    Load parquet file

    Args:
        file_path: Path to parquet file

    Returns:
        Loaded DataFrame
    """
    return pd.read_parquet(file_path)


def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file

    Args:
        file_path: Path to YAML file

    Returns:
        Configuration dictionary
    """
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def save_json(data: Dict[str, Any], file_path: str) -> None:
    """
    Save dictionary as JSON file

    Args:
        data: Data to save
        file_path: Output path
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load JSON file

    Args:
        file_path: Path to JSON file

    Returns:
        Loaded dictionary
    """
    with open(file_path, 'r') as f:
        return json.load(f)
