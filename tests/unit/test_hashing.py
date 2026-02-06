"""Unit tests for hashing utilities."""

import tempfile
import threading
from pathlib import Path

import pytest

from scraper.utils.hashing import HashManager


class TestHashManager:
    """Tests for HashManager."""

    def test_sha256_content(self):
        content = b"Hello, World!"
        hash_value = HashManager.sha256_content(content)

        assert len(hash_value) == 64  # SHA256 hex is 64 chars
        assert hash_value == "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"

    def test_sha256_content_consistency(self):
        content = b"Test content"
        hash1 = HashManager.sha256_content(content)
        hash2 = HashManager.sha256_content(content)
        assert hash1 == hash2

    def test_sha256_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"File content for hashing")
            f.flush()

            hash_value = HashManager.sha256_file(Path(f.name))
            assert len(hash_value) == 64

            Path(f.name).unlink()

    def test_normalize_path(self):
        assert HashManager.normalize_path("https://example.com/path/") == "example.com/path"
        assert HashManager.normalize_path("http://www.example.com") == "example.com"
        assert HashManager.normalize_path("HTTPS://EXAMPLE.COM/PATH") == "example.com/path"

    def test_metadata_signature(self):
        metadata = {"name": "test", "value": 123}
        sig1 = HashManager.metadata_signature(metadata)

        # Same content, different order
        metadata2 = {"value": 123, "name": "test"}
        sig2 = HashManager.metadata_signature(metadata2)

        assert sig1 == sig2  # Should be same due to key sorting

    def test_normalized_representation_hash(self):
        # Lists
        hash1 = HashManager.normalized_representation_hash([1, 2, 3])
        hash2 = HashManager.normalized_representation_hash([1, 2, 3])
        assert hash1 == hash2

        # Different content
        hash3 = HashManager.normalized_representation_hash([1, 2, 4])
        assert hash1 != hash3

    def test_is_duplicate(self):
        manager = HashManager()
        hash_value = "abc123"

        assert not manager.is_duplicate(hash_value)
        manager.add_hash(hash_value)
        assert manager.is_duplicate(hash_value)

    def test_add_hash_returns_status(self):
        manager = HashManager()
        hash_value = "xyz789"

        assert manager.add_hash(hash_value) is True  # New hash
        assert manager.add_hash(hash_value) is False  # Duplicate

    def test_check_and_add_atomic(self):
        manager = HashManager()
        hash_value = "atomic123"

        assert manager.check_and_add(hash_value) is True
        assert manager.check_and_add(hash_value) is False

    def test_thread_safety(self):
        manager = HashManager()
        results = []

        def add_hashes(start, count):
            for i in range(start, start + count):
                result = manager.add_hash(f"hash_{i}")
                results.append((i, result))

        threads = [
            threading.Thread(target=add_hashes, args=(0, 100)),
            threading.Thread(target=add_hashes, args=(50, 100)),  # Overlapping
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Check that overlapping hashes (50-99) were only added once
        assert manager.count() == 150  # 0-149 unique hashes

    def test_clear(self):
        manager = HashManager()
        manager.add_hash("hash1")
        manager.add_hash("hash2")

        assert manager.count() == 2
        manager.clear()
        assert manager.count() == 0

    def test_persistence(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            store_path = Path(f.name)

        try:
            # Create manager and add hashes
            manager1 = HashManager(store_path)
            manager1.add_hash("persistent_hash_1")
            manager1.add_hash("persistent_hash_2")

            # Create new manager with same store
            manager2 = HashManager(store_path)
            assert manager2.is_duplicate("persistent_hash_1")
            assert manager2.is_duplicate("persistent_hash_2")
        finally:
            store_path.unlink(missing_ok=True)
