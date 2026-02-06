"""Configuration loader with YAML parsing, environment variable overrides, and Pydantic validation."""

import os
import re
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class NetworkConfig(BaseModel):
    """Network-related configuration."""
    timeout: int = Field(default=10, ge=1, le=300)
    retries: int = Field(default=3, ge=0, le=10)
    retry_backoff: int = Field(default=2, ge=1, le=10)
    user_agent: str = Field(default="UniversalScraper/1.0")


class RateLimitsConfig(BaseModel):
    """Rate limiting configuration."""
    global_max_rps: float = Field(default=2.0, ge=0.1, le=100)
    images_max_rps: float = Field(default=1.0, ge=0.1, le=100)
    chess_max_rps: float = Field(default=2.0, ge=0.1, le=100)
    numeric_max_rps: float = Field(default=2.0, ge=0.1, le=100)
    experiments_max_rps: float = Field(default=1.0, ge=0.1, le=100)
    social_max_rps: float = Field(default=0.5, ge=0.1, le=100)


class StorageConfig(BaseModel):
    """Storage-related configuration."""
    max_file_size_mb: int = Field(default=50, ge=1, le=1000)
    disk_space_min_mb: int = Field(default=500, ge=100, le=100000)
    log_rotation_mb: int = Field(default=10, ge=1, le=100)
    log_backup_count: int = Field(default=5, ge=1, le=20)


class ConcurrencyConfig(BaseModel):
    """Concurrency configuration."""
    max_workers: int = Field(default=5, ge=1, le=50)
    queue_size: int = Field(default=100, ge=10, le=10000)


class ModuleConfig(BaseModel):
    """Per-module configuration."""
    default_limit: int = Field(default=100, ge=1)
    allowed_content_types: list[str] = Field(default_factory=list)


class ModulesConfig(BaseModel):
    """Configuration for all modules."""
    images: ModuleConfig = Field(default_factory=lambda: ModuleConfig(
        default_limit=100,
        allowed_content_types=["image/jpeg", "image/png"]
    ))
    chess: ModuleConfig = Field(default_factory=lambda: ModuleConfig(default_limit=500))
    numeric: ModuleConfig = Field(default_factory=lambda: ModuleConfig(default_limit=1000))
    experiments: ModuleConfig = Field(default_factory=lambda: ModuleConfig(default_limit=50))
    social: ModuleConfig = Field(default_factory=lambda: ModuleConfig(default_limit=200))


class CredentialsConfig(BaseModel):
    """Credentials configuration for authenticated services."""
    twitter_api_key: Optional[str] = None
    twitter_api_secret: Optional[str] = None
    twitter_access_token: Optional[str] = None
    twitter_access_secret: Optional[str] = None


class Config(BaseModel):
    """Main configuration model."""
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    rate_limits: RateLimitsConfig = Field(default_factory=RateLimitsConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    concurrency: ConcurrencyConfig = Field(default_factory=ConcurrencyConfig)
    modules: ModulesConfig = Field(default_factory=ModulesConfig)
    credentials: CredentialsConfig = Field(default_factory=CredentialsConfig)

    # Runtime overrides from CLI
    output_dir: Optional[Path] = None
    log_path: Optional[Path] = None
    async_enabled: bool = False
    no_duplicates: bool = True


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand environment variables in configuration values."""
    if isinstance(value, str):
        pattern = r'\$\{([^}]+)\}'
        matches = re.findall(pattern, value)
        for match in matches:
            env_value = os.environ.get(match, "")
            value = value.replace(f"${{{match}}}", env_value)
        return value if value else None
    elif isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    return value


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    config_path: Optional[Path] = None,
    cli_overrides: Optional[dict] = None
) -> Config:
    """
    Load configuration from YAML files with environment variable and CLI overrides.

    Priority (highest to lowest):
    1. CLI overrides
    2. Environment variables
    3. Custom config file (if provided)
    4. Default config file
    """
    base_dir = Path(__file__).parent.parent / "config"
    default_config_path = base_dir / "default.yaml"
    credentials_path = base_dir / "credentials.yaml"

    config_data: dict = {}

    # Load default config
    if default_config_path.exists():
        with open(default_config_path) as f:
            config_data = yaml.safe_load(f) or {}

    # Load custom config if provided
    if config_path and config_path.exists():
        with open(config_path) as f:
            custom_config = yaml.safe_load(f) or {}
            config_data = _deep_merge(config_data, custom_config)

    # Load credentials
    if credentials_path.exists():
        with open(credentials_path) as f:
            creds_data = yaml.safe_load(f) or {}
            creds_data = _expand_env_vars(creds_data)
            if "social" in creds_data:
                config_data["credentials"] = creds_data["social"]

    # Apply environment variable overrides for main config
    env_overrides = {
        "network": {
            "timeout": os.environ.get("SCRAPER_TIMEOUT"),
            "retries": os.environ.get("SCRAPER_RETRIES"),
        },
        "rate_limits": {
            "global_max_rps": os.environ.get("SCRAPER_GLOBAL_RPS"),
        },
        "storage": {
            "max_file_size_mb": os.environ.get("SCRAPER_MAX_FILE_SIZE"),
            "disk_space_min_mb": os.environ.get("SCRAPER_MIN_DISK_SPACE"),
        },
    }

    # Clean up None values from env overrides
    def clean_none(d: dict) -> dict:
        return {
            k: clean_none(v) if isinstance(v, dict) else (int(v) if v and v.isdigit() else float(v) if v else None)
            for k, v in d.items()
            if v is not None and (not isinstance(v, dict) or clean_none(v))
        }

    env_overrides = clean_none(env_overrides)
    if env_overrides:
        config_data = _deep_merge(config_data, env_overrides)

    # Apply CLI overrides
    if cli_overrides:
        config_data = _deep_merge(config_data, cli_overrides)

    return Config(**config_data)
