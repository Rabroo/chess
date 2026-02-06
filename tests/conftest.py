"""Pytest configuration and shared fixtures."""

import tempfile
from pathlib import Path

import pytest

from scraper.utils.config import Config
from scraper.utils.directory import DirectoryManager
from scraper.utils.logger import ScraperLogger


@pytest.fixture(scope="function")
def temp_directory():
    """Create a temporary directory that's cleaned up after each test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="function")
def test_config():
    """Create a fresh Config instance for each test."""
    return Config()


@pytest.fixture(scope="function")
def test_logger(temp_directory):
    """Create a logger that writes to a temp directory."""
    return ScraperLogger(
        name="pytest",
        log_path=temp_directory / "test.log",
        console_output=False,
    )


@pytest.fixture(scope="function")
def test_directory_manager(temp_directory):
    """Create an initialized DirectoryManager in a temp directory."""
    manager = DirectoryManager(base_path=temp_directory)
    manager.initialize()
    return manager


@pytest.fixture
def sample_jpeg_bytes():
    """Minimal valid JPEG bytes for testing."""
    return b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + b"\x00" * 100


@pytest.fixture
def sample_png_bytes():
    """Minimal valid PNG bytes for testing."""
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * 100


@pytest.fixture
def sample_fen():
    """Standard chess starting position FEN."""
    return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


@pytest.fixture
def sample_sequence():
    """Fibonacci sequence for testing."""
    return [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
