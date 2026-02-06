"""Unit tests for validation utilities."""

import pytest

from scraper.utils.validator import ValidationError, Validator


class TestURLValidation:
    """Tests for URL validation."""

    def setup_method(self):
        self.validator = Validator()

    def test_valid_https_url(self):
        assert self.validator.validate_url("https://example.com/path") is True

    def test_valid_http_url(self):
        assert self.validator.validate_url("http://example.com") is True

    def test_invalid_scheme_file(self):
        with pytest.raises(ValidationError, match="Invalid URL scheme"):
            self.validator.validate_url("file:///etc/passwd")

    def test_invalid_scheme_ftp(self):
        with pytest.raises(ValidationError, match="Invalid URL scheme"):
            self.validator.validate_url("ftp://example.com")

    def test_ssrf_localhost(self):
        with pytest.raises(ValidationError, match="localhost"):
            self.validator.validate_url("http://localhost/admin")

    def test_ssrf_private_ip_10(self):
        with pytest.raises(ValidationError, match="Internal/private IP"):
            self.validator.validate_url("http://10.0.0.1/")

    def test_ssrf_private_ip_192(self):
        with pytest.raises(ValidationError, match="Internal/private IP"):
            self.validator.validate_url("http://192.168.1.1/")

    def test_ssrf_private_ip_172(self):
        with pytest.raises(ValidationError, match="Internal/private IP"):
            self.validator.validate_url("http://172.16.0.1/")

    def test_ssrf_loopback(self):
        with pytest.raises(ValidationError, match="Internal/private IP"):
            self.validator.validate_url("http://127.0.0.1/")

    def test_allow_internal_flag(self):
        # Should pass when allow_internal=True
        assert self.validator.validate_url("http://192.168.1.1/", allow_internal=True) is True

    def test_empty_host(self):
        with pytest.raises(ValidationError, match="must have a host"):
            self.validator.validate_url("http:///path")


class TestContentTypeValidation:
    """Tests for content type validation."""

    def setup_method(self):
        self.validator = Validator()

    def test_valid_content_type(self):
        allowed = {"image/jpeg", "image/png"}
        assert self.validator.validate_content_type("image/jpeg", allowed) is True

    def test_valid_content_type_with_charset(self):
        allowed = {"text/html"}
        assert self.validator.validate_content_type("text/html; charset=utf-8", allowed) is True

    def test_invalid_content_type(self):
        allowed = {"image/jpeg", "image/png"}
        with pytest.raises(ValidationError, match="Content type.*not allowed"):
            self.validator.validate_content_type("application/pdf", allowed)


class TestMagicBytesValidation:
    """Tests for magic bytes validation."""

    def setup_method(self):
        self.validator = Validator()

    def test_valid_jpeg(self):
        jpeg_content = b"\xff\xd8\xff" + b"\x00" * 100
        assert self.validator.validate_magic_bytes(jpeg_content, "image/jpeg") is True

    def test_valid_png(self):
        png_content = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        assert self.validator.validate_magic_bytes(png_content, "image/png") is True

    def test_invalid_magic_bytes(self):
        fake_jpeg = b"not a jpeg" + b"\x00" * 100
        with pytest.raises(ValidationError, match="does not match"):
            self.validator.validate_magic_bytes(fake_jpeg, "image/jpeg")

    def test_unknown_type_passes(self):
        content = b"some content"
        assert self.validator.validate_magic_bytes(content, "application/unknown") is True


class TestFileSizeValidation:
    """Tests for file size validation."""

    def test_valid_size(self):
        validator = Validator(max_file_size_mb=10)
        assert validator.validate_file_size(5 * 1024 * 1024) is True  # 5 MB

    def test_invalid_size(self):
        validator = Validator(max_file_size_mb=10)
        with pytest.raises(ValidationError, match="exceeds maximum"):
            validator.validate_file_size(15 * 1024 * 1024)  # 15 MB


class TestFilenameSanitization:
    """Tests for filename sanitization."""

    def setup_method(self):
        self.validator = Validator()

    def test_safe_filename(self):
        assert self.validator.sanitize_filename("image.jpg") == "image.jpg"

    def test_unsafe_characters(self):
        result = self.validator.sanitize_filename("file<>:\"/\\|?*.jpg")
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result
        assert "/" not in result
        assert "\\" not in result

    def test_empty_filename(self):
        result = self.validator.sanitize_filename("")
        assert result == "unnamed"

    def test_long_filename_truncation(self):
        long_name = "a" * 300 + ".jpg"
        result = self.validator.sanitize_filename(long_name, max_length=255)
        assert len(result) <= 255
        assert result.endswith(".jpg")

    def test_unicode_normalization(self):
        # Test that unicode characters are handled
        result = self.validator.sanitize_filename("café.jpg")
        assert ".jpg" in result


class TestFENValidation:
    """Tests for FEN string validation."""

    def setup_method(self):
        self.validator = Validator()

    def test_valid_starting_position(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        assert self.validator.validate_fen(fen) is True

    def test_valid_fen_minimal(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        assert self.validator.validate_fen(fen) is True

    def test_invalid_fen_wrong_ranks(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/PPPPPPPP/RNBQKBNR"  # Only 7 ranks
        with pytest.raises(ValidationError, match="8 ranks"):
            self.validator.validate_fen(fen)

    def test_invalid_fen_wrong_files(self):
        fen = "rnbqkbnr/ppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"  # 7 squares in rank 2
        with pytest.raises(ValidationError, match="8 squares"):
            self.validator.validate_fen(fen)

    def test_invalid_fen_bad_character(self):
        fen = "rnbqkbnr/ppppxppp/8/8/8/8/PPPPPPPP/RNBQKBNR"  # 'x' is invalid
        with pytest.raises(ValidationError, match="Invalid character"):
            self.validator.validate_fen(fen)

    def test_invalid_active_color(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq - 0 1"
        with pytest.raises(ValidationError, match="Active color"):
            self.validator.validate_fen(fen)


class TestJSONSchemaValidation:
    """Tests for JSON schema validation."""

    def setup_method(self):
        self.validator = Validator()

    def test_valid_data(self):
        data = {"id": "123", "name": "test"}
        required = {"id", "name"}
        assert self.validator.validate_json_schema(data, required) is True

    def test_missing_field(self):
        data = {"id": "123"}
        required = {"id", "name"}
        with pytest.raises(ValidationError, match="Missing required fields"):
            self.validator.validate_json_schema(data, required)
