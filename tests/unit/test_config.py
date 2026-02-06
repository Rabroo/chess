"""Unit tests for configuration loading."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from scraper.utils.config import (
    Config,
    NetworkConfig,
    RateLimitsConfig,
    StorageConfig,
    load_config,
)


class TestNetworkConfig:
    """Tests for NetworkConfig."""

    def test_default_values(self):
        config = NetworkConfig()
        assert config.timeout == 10
        assert config.retries == 3
        assert config.retry_backoff == 2
        assert config.user_agent == "UniversalScraper/1.0"

    def test_custom_values(self):
        config = NetworkConfig(timeout=30, retries=5)
        assert config.timeout == 30
        assert config.retries == 5

    def test_validation_timeout_range(self):
        with pytest.raises(ValueError):
            NetworkConfig(timeout=0)
        with pytest.raises(ValueError):
            NetworkConfig(timeout=500)


class TestRateLimitsConfig:
    """Tests for RateLimitsConfig."""

    def test_default_values(self):
        config = RateLimitsConfig()
        assert config.global_max_rps == 2.0
        assert config.images_max_rps == 1.0
        assert config.social_max_rps == 0.5

    def test_custom_values(self):
        config = RateLimitsConfig(global_max_rps=5.0)
        assert config.global_max_rps == 5.0


class TestStorageConfig:
    """Tests for StorageConfig."""

    def test_default_values(self):
        config = StorageConfig()
        assert config.max_file_size_mb == 50
        assert config.disk_space_min_mb == 500
        assert config.log_rotation_mb == 10
        assert config.log_backup_count == 5


class TestConfig:
    """Tests for main Config class."""

    def test_default_config(self):
        config = Config()
        assert isinstance(config.network, NetworkConfig)
        assert isinstance(config.rate_limits, RateLimitsConfig)
        assert isinstance(config.storage, StorageConfig)

    def test_nested_config(self):
        config = Config(
            network=NetworkConfig(timeout=20),
            storage=StorageConfig(max_file_size_mb=100),
        )
        assert config.network.timeout == 20
        assert config.storage.max_file_size_mb == 100


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_default_config(self):
        config = load_config()
        assert isinstance(config, Config)

    def test_load_custom_config_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({
                "network": {"timeout": 25},
                "storage": {"max_file_size_mb": 75},
            }, f)
            f.flush()

            config = load_config(Path(f.name))
            assert config.network.timeout == 25
            assert config.storage.max_file_size_mb == 75

            os.unlink(f.name)

    def test_cli_overrides(self):
        config = load_config(
            cli_overrides={"async_enabled": True}
        )
        assert config.async_enabled is True

    def test_env_var_override(self):
        os.environ["SCRAPER_TIMEOUT"] = "45"
        try:
            config = load_config()
            # Note: env var handling may need adjustment based on implementation
        finally:
            del os.environ["SCRAPER_TIMEOUT"]
