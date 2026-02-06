"""Unit tests for directory management."""

import tempfile
from pathlib import Path

import pytest

from scraper.utils.directory import DirectoryManager


class TestDirectoryManager:
    """Tests for DirectoryManager."""

    def test_initialize_creates_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DirectoryManager(base_path=Path(tmpdir))
            manager.initialize()

            assert (Path(tmpdir) / "raw").exists()
            assert (Path(tmpdir) / "raw" / "images").exists()
            assert (Path(tmpdir) / "raw" / "chess_positions").exists()
            assert (Path(tmpdir) / "raw" / "numeric_relations").exists()
            assert (Path(tmpdir) / "raw" / "experiments").exists()
            assert (Path(tmpdir) / "raw" / "social").exists()
            assert (Path(tmpdir) / "raw" / "meta").exists()
            assert (Path(tmpdir) / "config").exists()
            assert (Path(tmpdir) / "logs").exists()

    def test_get_module_dir_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DirectoryManager(base_path=Path(tmpdir))
            manager.initialize()

            images_dir = manager.get_module_dir("images")
            assert images_dir == Path(tmpdir) / "raw" / "images"
            assert images_dir.exists()

    def test_get_module_dir_new(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DirectoryManager(base_path=Path(tmpdir))
            manager.initialize()

            new_dir = manager.get_module_dir("custom_module")
            assert new_dir == Path(tmpdir) / "raw" / "custom_module"
            assert new_dir.exists()

    def test_check_disk_space(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DirectoryManager(
                base_path=Path(tmpdir),
                min_disk_space_mb=1,  # Very low threshold
            )

            # Should have at least 1 MB available
            assert manager.check_disk_space() is True

    def test_check_disk_space_insufficient(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DirectoryManager(
                base_path=Path(tmpdir),
                min_disk_space_mb=999999999,  # Very high threshold
            )

            assert manager.check_disk_space() is False

    def test_get_available_space_mb(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DirectoryManager(base_path=Path(tmpdir))
            space = manager.get_available_space_mb()

            assert space > 0
            assert isinstance(space, float)

    def test_get_temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DirectoryManager(base_path=Path(tmpdir))

            temp_dir = manager.get_temp_dir()
            assert temp_dir.exists()
            assert "scraper_" in str(temp_dir)

            # Same temp dir on second call
            temp_dir2 = manager.get_temp_dir()
            assert temp_dir == temp_dir2

            manager.cleanup_temp_dir()

    def test_cleanup_temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DirectoryManager(base_path=Path(tmpdir))

            temp_dir = manager.get_temp_dir()
            # Create a file in temp dir
            (temp_dir / "test.txt").write_text("test")

            manager.cleanup_temp_dir()
            assert not temp_dir.exists()

    def test_move_from_temp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DirectoryManager(base_path=Path(tmpdir))
            manager.initialize()

            temp_dir = manager.get_temp_dir()
            temp_file = temp_dir / "test.txt"
            temp_file.write_text("content")

            dest = Path(tmpdir) / "raw" / "images" / "test.txt"
            result = manager.move_from_temp(temp_file, dest)

            assert result == dest
            assert dest.exists()
            assert not temp_file.exists()
            assert dest.read_text() == "content"

    def test_get_log_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DirectoryManager(base_path=Path(tmpdir))

            log_path = manager.get_log_path()
            assert log_path == Path(tmpdir) / "logs" / "scraper.log"
