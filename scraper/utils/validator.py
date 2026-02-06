"""Validation utilities for URLs, content, and filenames."""

import ipaddress
import re
import unicodedata
from pathlib import Path
from typing import Optional, Set
from urllib.parse import urlparse

# Magic bytes for common file types
MAGIC_BYTES = {
    "image/jpeg": [b"\xff\xd8\xff"],
    "image/png": [b"\x89PNG\r\n\x1a\n"],
    "image/gif": [b"GIF87a", b"GIF89a"],
    "application/pdf": [b"%PDF"],
    "application/json": [b"{", b"["],
    "text/csv": [],  # No reliable magic bytes for CSV
}

# Private/internal IP ranges to block (SSRF protection)
PRIVATE_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


class Validator:
    """Validates URLs, content types, and filenames for security and correctness."""

    ALLOWED_SCHEMES: Set[str] = {"http", "https"}
    UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

    def __init__(self, max_file_size_mb: int = 50):
        """
        Initialize the validator.

        Args:
            max_file_size_mb: Maximum allowed file size in MB.
        """
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

    def validate_url(self, url: str, allow_internal: bool = False) -> bool:
        """
        Validate a URL for safety (SSRF protection).

        Args:
            url: URL to validate.
            allow_internal: Whether to allow internal/private IPs.

        Returns:
            True if URL is safe.

        Raises:
            ValidationError: If URL is invalid or unsafe.
        """
        try:
            parsed = urlparse(url)
        except Exception as e:
            raise ValidationError(f"Invalid URL format: {e}")

        # Check scheme
        if parsed.scheme.lower() not in self.ALLOWED_SCHEMES:
            raise ValidationError(
                f"Invalid URL scheme '{parsed.scheme}'. Only HTTP/HTTPS allowed."
            )

        # Check for empty host
        if not parsed.netloc:
            raise ValidationError("URL must have a host.")

        # Extract hostname (without port)
        hostname = parsed.hostname
        if not hostname:
            raise ValidationError("Could not extract hostname from URL.")

        # Check for internal IPs if not allowed
        if not allow_internal:
            try:
                ip = ipaddress.ip_address(hostname)
                for network in PRIVATE_NETWORKS:
                    if ip in network:
                        raise ValidationError(
                            f"Internal/private IP addresses are not allowed: {hostname}"
                        )
            except ValueError:
                # Not an IP address, it's a hostname - that's fine
                # But check for localhost variants
                if hostname.lower() in ("localhost", "localhost.localdomain"):
                    raise ValidationError("localhost is not allowed.")

        return True

    def validate_content_type(
        self,
        content_type: str,
        allowed_types: Set[str],
    ) -> bool:
        """
        Validate content type against allowed types.

        Args:
            content_type: Content-Type header value.
            allowed_types: Set of allowed MIME types.

        Returns:
            True if content type is allowed.

        Raises:
            ValidationError: If content type is not allowed.
        """
        # Extract main type (ignore charset, etc.)
        main_type = content_type.split(";")[0].strip().lower()

        if main_type not in allowed_types:
            raise ValidationError(
                f"Content type '{main_type}' not allowed. "
                f"Allowed types: {', '.join(allowed_types)}"
            )

        return True

    def validate_magic_bytes(
        self,
        content: bytes,
        expected_type: str,
    ) -> bool:
        """
        Validate file content by checking magic bytes.

        Args:
            content: File content bytes.
            expected_type: Expected MIME type.

        Returns:
            True if magic bytes match expected type.

        Raises:
            ValidationError: If magic bytes don't match.
        """
        if expected_type not in MAGIC_BYTES:
            # No magic bytes defined for this type, skip check
            return True

        magic_patterns = MAGIC_BYTES[expected_type]
        if not magic_patterns:
            return True

        for pattern in magic_patterns:
            if content.startswith(pattern):
                return True

        raise ValidationError(
            f"File content does not match expected type '{expected_type}'"
        )

    def validate_file_size(self, size_bytes: int) -> bool:
        """
        Validate file size against maximum allowed.

        Args:
            size_bytes: File size in bytes.

        Returns:
            True if size is acceptable.

        Raises:
            ValidationError: If file is too large.
        """
        if size_bytes > self.max_file_size_bytes:
            max_mb = self.max_file_size_bytes / (1024 * 1024)
            actual_mb = size_bytes / (1024 * 1024)
            raise ValidationError(
                f"File size ({actual_mb:.2f} MB) exceeds maximum ({max_mb} MB)"
            )
        return True

    def sanitize_filename(self, filename: str, max_length: int = 255) -> str:
        """
        Sanitize a filename by removing unsafe characters.

        Args:
            filename: Original filename.
            max_length: Maximum filename length.

        Returns:
            Sanitized filename.
        """
        # Normalize unicode
        filename = unicodedata.normalize("NFKD", filename)

        # Remove unsafe characters
        filename = self.UNSAFE_FILENAME_CHARS.sub("_", filename)

        # Remove leading/trailing whitespace and dots
        filename = filename.strip(". \t\n")

        # Ensure filename is not empty
        if not filename:
            filename = "unnamed"

        # Truncate if too long (preserve extension)
        if len(filename) > max_length:
            name, ext = (
                (filename.rsplit(".", 1) + [""])[:2]
                if "." in filename
                else (filename, "")
            )
            max_name_len = max_length - len(ext) - 1 if ext else max_length
            filename = f"{name[:max_name_len]}.{ext}" if ext else name[:max_length]

        return filename

    def validate_fen(self, fen: str) -> bool:
        """
        Validate a FEN (Forsyth-Edwards Notation) chess string.

        Args:
            fen: FEN string to validate.

        Returns:
            True if valid FEN.

        Raises:
            ValidationError: If FEN is invalid.
        """
        parts = fen.strip().split()

        if len(parts) < 1:
            raise ValidationError("Empty FEN string")

        # Validate piece placement
        ranks = parts[0].split("/")
        if len(ranks) != 8:
            raise ValidationError("FEN must have 8 ranks")

        valid_pieces = set("pnbrqkPNBRQK12345678")
        for rank in ranks:
            count = 0
            for char in rank:
                if char not in valid_pieces:
                    raise ValidationError(f"Invalid character in FEN: {char}")
                if char.isdigit():
                    count += int(char)
                else:
                    count += 1
            if count != 8:
                raise ValidationError(f"Rank must have 8 squares, got {count}")

        # Validate other fields if present
        if len(parts) >= 2:
            if parts[1] not in ("w", "b"):
                raise ValidationError("Active color must be 'w' or 'b'")

        if len(parts) >= 3:
            castling = parts[2]
            if castling != "-" and not all(c in "KQkq" for c in castling):
                raise ValidationError("Invalid castling rights")

        return True

    def validate_json_schema(self, data: dict, required_fields: Set[str]) -> bool:
        """
        Basic JSON schema validation for required fields.

        Args:
            data: Dictionary to validate.
            required_fields: Set of required field names.

        Returns:
            True if valid.

        Raises:
            ValidationError: If required fields are missing.
        """
        missing = required_fields - set(data.keys())
        if missing:
            raise ValidationError(f"Missing required fields: {', '.join(missing)}")
        return True
