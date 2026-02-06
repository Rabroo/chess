"""Integration tests for scraper modules."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scraper.scrapers import ScraperRegistry
from scraper.scrapers.base import ScrapedItem
from scraper.utils.config import Config
from scraper.utils.directory import DirectoryManager
from scraper.utils.logger import ScraperLogger


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def config():
    """Create a test configuration."""
    return Config()


@pytest.fixture
def logger(temp_dir):
    """Create a test logger."""
    return ScraperLogger(
        name="test",
        log_path=temp_dir / "test.log",
        console_output=False,
    )


@pytest.fixture
def directory_manager(temp_dir):
    """Create a test directory manager."""
    manager = DirectoryManager(base_path=temp_dir)
    manager.initialize()
    return manager


class TestScraperRegistry:
    """Tests for ScraperRegistry."""

    def test_list_modules(self):
        modules = ScraperRegistry.list_modules()
        assert "images" in modules
        assert "chess" in modules
        assert "numeric" in modules
        assert "experiments" in modules
        assert "social" in modules

    def test_get_module(self):
        module_class = ScraperRegistry.get("images")
        assert module_class is not None
        assert module_class.MODULE_NAME == "images"

    def test_get_nonexistent_module(self):
        assert ScraperRegistry.get("nonexistent") is None

    def test_create_module(self, config, logger, directory_manager):
        scraper = ScraperRegistry.create(
            name="images",
            config=config,
            logger=logger,
            directory_manager=directory_manager,
        )
        assert scraper is not None
        assert scraper.MODULE_NAME == "images"


class TestImageScraper:
    """Tests for ImageScraper."""

    def test_validate_input_url(self, config, logger, directory_manager):
        scraper = ScraperRegistry.create("images", config, logger, directory_manager)

        assert scraper.validate_input("https://example.com/image.jpg") is True

    def test_validate_input_invalid_url(self, config, logger, directory_manager):
        scraper = ScraperRegistry.create("images", config, logger, directory_manager)

        with pytest.raises(Exception):  # ValidationError
            scraper.validate_input("not_a_url")

    def test_get_hash(self, config, logger, directory_manager):
        scraper = ScraperRegistry.create("images", config, logger, directory_manager)

        item = ScrapedItem(
            content=b"test image content",
            identifier="test.jpg",
            metadata={},
        )

        hash1 = scraper.get_hash(item)
        hash2 = scraper.get_hash(item)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256


class TestChessPositionScraper:
    """Tests for ChessPositionScraper."""

    def test_validate_input_fen(self, config, logger, directory_manager):
        scraper = ScraperRegistry.create("chess", config, logger, directory_manager)

        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        assert scraper.validate_input(fen) is True

    def test_validate_input_invalid_fen(self, config, logger, directory_manager):
        scraper = ScraperRegistry.create("chess", config, logger, directory_manager)

        with pytest.raises(Exception):  # ValidationError
            scraper.validate_input("invalid fen string")

    def test_fetch_single_fen(self, config, logger, directory_manager):
        scraper = ScraperRegistry.create("chess", config, logger, directory_manager)

        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        items = list(scraper.fetch(fen, limit=10))

        assert len(items) == 1
        assert items[0].metadata["fen"] == fen


class TestNumericRelationScraper:
    """Tests for NumericRelationScraper."""

    def test_validate_input_sequence(self, config, logger, directory_manager):
        scraper = ScraperRegistry.create("numeric", config, logger, directory_manager)

        assert scraper.validate_input("1, 2, 3, 4, 5") is True

    def test_validate_input_oeis_id(self, config, logger, directory_manager):
        scraper = ScraperRegistry.create("numeric", config, logger, directory_manager)

        assert scraper.validate_input("A000045") is True

    def test_fetch_sequence(self, config, logger, directory_manager):
        scraper = ScraperRegistry.create("numeric", config, logger, directory_manager)

        items = list(scraper.fetch("1, 1, 2, 3, 5, 8", limit=10))

        assert len(items) == 1
        assert items[0].metadata["values"] == [1.0, 1.0, 2.0, 3.0, 5.0, 8.0]


class TestSocialScraper:
    """Tests for SocialScraper."""

    def test_validate_input_hashtag(self, config, logger, directory_manager):
        scraper = ScraperRegistry.create("social", config, logger, directory_manager)

        assert scraper.validate_input("#python") is True

    def test_validate_input_handle(self, config, logger, directory_manager):
        scraper = ScraperRegistry.create("social", config, logger, directory_manager)

        assert scraper.validate_input("@username") is True

    def test_validate_input_url(self, config, logger, directory_manager):
        scraper = ScraperRegistry.create("social", config, logger, directory_manager)

        assert scraper.validate_input("https://twitter.com/user/status/123") is True


class TestExperimentScraper:
    """Tests for ExperimentScraper."""

    def test_validate_input_url(self, config, logger, directory_manager):
        scraper = ScraperRegistry.create("experiments", config, logger, directory_manager)

        assert scraper.validate_input("https://example.com/dataset.json") is True

    def test_validate_input_keyword(self, config, logger, directory_manager):
        scraper = ScraperRegistry.create("experiments", config, logger, directory_manager)

        assert scraper.validate_input("climate data") is True


class TestDuplicateDetection:
    """Tests for duplicate detection across scrapers."""

    def test_duplicate_items_skipped(self, config, logger, directory_manager):
        scraper = ScraperRegistry.create("chess", config, logger, directory_manager)

        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

        # First store
        items1 = list(scraper.fetch(fen, limit=1))
        result1 = scraper.store(items1[0])
        assert result1.success is True

        # Second store of same item
        items2 = list(scraper.fetch(fen, limit=1))
        result2 = scraper.store(items2[0])
        assert result2.success is False
        assert result2.error == "Duplicate item"
