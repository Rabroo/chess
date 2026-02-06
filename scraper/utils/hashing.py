"""Hashing utilities for duplicate detection."""

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Optional, Set


class HashManager:
    """Thread-safe hash management for duplicate detection."""

    def __init__(self, hash_store_path: Optional[Path] = None):
        """
        Initialize the hash manager.

        Args:
            hash_store_path: Optional path to persist hashes between runs.
        """
        self._hashes: Set[str] = set()
        self._lock = threading.Lock()
        self._store_path = hash_store_path

        # Load existing hashes if store path provided
        if hash_store_path and hash_store_path.exists():
            self._load_hashes()

    def _load_hashes(self) -> None:
        """Load hashes from the store file."""
        if self._store_path and self._store_path.exists():
            with open(self._store_path, "r") as f:
                for line in f:
                    self._hashes.add(line.strip())

    def _save_hash(self, hash_value: str) -> None:
        """Append a hash to the store file."""
        if self._store_path:
            with open(self._store_path, "a") as f:
                f.write(f"{hash_value}\n")

    @staticmethod
    def sha256_content(content: bytes) -> str:
        """
        Generate SHA256 hash of content.

        Args:
            content: Bytes to hash.

        Returns:
            Hex-encoded SHA256 hash.
        """
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def sha256_file(file_path: Path, chunk_size: int = 8192) -> str:
        """
        Generate SHA256 hash of a file.

        Args:
            file_path: Path to the file.
            chunk_size: Size of chunks to read.

        Returns:
            Hex-encoded SHA256 hash.
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def normalize_path(path: str) -> str:
        """
        Normalize a path for canonical comparison.

        Args:
            path: Path or URL to normalize.

        Returns:
            Normalized path string.
        """
        # Remove trailing slashes, convert to lowercase
        normalized = path.rstrip("/").lower()
        # Remove common URL prefixes for comparison
        for prefix in ["http://", "https://", "www."]:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
        return normalized

    @staticmethod
    def metadata_signature(metadata: dict[str, Any]) -> str:
        """
        Generate a hash signature from metadata.

        Args:
            metadata: Dictionary of metadata.

        Returns:
            Hex-encoded hash of the metadata.
        """
        # Sort keys for consistent hashing
        normalized = json.dumps(metadata, sort_keys=True, default=str)
        return hashlib.sha256(normalized.encode()).hexdigest()

    @staticmethod
    def normalized_representation_hash(data: Any) -> str:
        """
        Generate a hash from a normalized representation of data.

        Args:
            data: Data to hash (list, dict, or primitive).

        Returns:
            Hex-encoded hash.
        """
        if isinstance(data, (list, tuple)):
            normalized = json.dumps(sorted(str(x) for x in data))
        elif isinstance(data, dict):
            normalized = json.dumps(data, sort_keys=True, default=str)
        else:
            normalized = str(data)
        return hashlib.sha256(normalized.encode()).hexdigest()

    def is_duplicate(self, hash_value: str) -> bool:
        """
        Check if a hash already exists (thread-safe).

        Args:
            hash_value: Hash to check.

        Returns:
            True if duplicate, False otherwise.
        """
        with self._lock:
            return hash_value in self._hashes

    def add_hash(self, hash_value: str) -> bool:
        """
        Add a hash to the set (thread-safe).

        Args:
            hash_value: Hash to add.

        Returns:
            True if added (was new), False if duplicate.
        """
        with self._lock:
            if hash_value in self._hashes:
                return False
            self._hashes.add(hash_value)
            self._save_hash(hash_value)
            return True

    def check_and_add(self, hash_value: str) -> bool:
        """
        Atomically check if hash is duplicate and add if not.

        Args:
            hash_value: Hash to check and add.

        Returns:
            True if was new and added, False if duplicate.
        """
        return self.add_hash(hash_value)

    def clear(self) -> None:
        """Clear all stored hashes."""
        with self._lock:
            self._hashes.clear()
            if self._store_path and self._store_path.exists():
                self._store_path.unlink()

    def count(self) -> int:
        """Get the number of stored hashes."""
        with self._lock:
            return len(self._hashes)
