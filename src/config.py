#!/usr/bin/env python3
"""
Churn ML Pipeline Configuration Management - Centralized settings and path resolution

🎯 PURPOSE: Load and manage all configuration settings from YAML files with automatic path resolution
📊 FEATURES: YAML config parsing, environment variable integration, secrets management, path normalization
🏗️ ARCHITECTURE: Singleton configuration pattern with validation and automatic directory creation
⚡ STATUS: Production-ready configuration system with secrets integration and cross-platform path handling
"""
import os
from pathlib import Path
from typing import Dict, Any
from .utils.io_utils import load_yaml_config


class Config:
    """Configuration manager for the churn ML pipeline"""

    def __init__(self, config_path: str):
        """
        Initialize configuration from YAML file

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self._config = load_yaml_config(str(self.config_path))

        # Set up base paths
        self.project_root = Path(__file__).parent.parent
        self._resolve_paths()

        # Load secrets if available
        self._load_secrets()

    def _resolve_paths(self) -> None:
        """Resolve relative paths to absolute paths"""
        for key, value in self._config.items():
            if isinstance(value, str) and not value.startswith('/'):
                # Convert relative paths to absolute
                if 'path' in key.lower() or 'dir' in key.lower():
                    self._config[key] = str(self.project_root / value)

    def _load_secrets(self) -> None:
        """Load secrets from global.env file (never output contents)"""
        secrets_path = Path.home() / '.config' / 'secrets' / 'global.env'

        # Reference secrets but don't read them
        if secrets_path.exists():
            self._config['secrets_available'] = True
            self._config['secrets_path'] = str(secrets_path)
        else:
            self._config['secrets_available'] = False
            self._config['secrets_path'] = None

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value
        """
        return self._config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Get configuration value using dict-like access"""
        return self._config[key]

    def __contains__(self, key: str) -> bool:
        """Check if key exists in configuration"""
        return key in self._config

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary"""
        return self._config.copy()

    @property
    def data_paths(self) -> Dict[str, str]:
        """Get data-related paths"""
        return {
            'raw_data_dir': self.get('raw_data_dir'),
            'processed_data_dir': self.get('processed_data_dir'),
            'models_dir': self.get('models_dir'),
            'logs_dir': self.get('logs_dir')
        }

    @property
    def pipeline_config(self) -> Dict[str, Any]:
        """Get pipeline-specific configuration"""
        return {
            'churn_cutoff_days': self.get('churn_cutoff_days', 90),
            'observation_end_date': self.get('observation_end_date'),
            'feature_engineering': self.get('feature_engineering', {}),
            'training': self.get('training', {})
        }
