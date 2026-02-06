"""Mock responses for network requests."""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class MockResponse:
    """Mock HTTP response."""
    content: bytes
    status_code: int = 200
    headers: Dict[str, str] = None

    def __post_init__(self):
        if self.headers is None:
            self.headers = {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


# Sample image data (minimal valid JPEG header)
SAMPLE_JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + b"\x00" * 100

# Sample image data (minimal valid PNG header)
SAMPLE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

# Sample FEN strings
SAMPLE_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
]

# Sample numeric sequences
SAMPLE_SEQUENCES = [
    {"id": "A000045", "name": "Fibonacci", "values": [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]},
    {"id": "A000040", "name": "Primes", "values": [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]},
    {"id": "A000290", "name": "Squares", "values": [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]},
]

# Sample social media post
SAMPLE_SOCIAL_POST = {
    "id": "1234567890",
    "platform": "twitter",
    "content": "This is a sample post",
    "author": {
        "id": "user123",
        "username": "testuser",
        "display_name": "Test User",
    },
    "created_at": "2024-01-15T10:30:00Z",
    "metrics": {
        "likes": 42,
        "reposts": 10,
        "replies": 5,
    },
    "media": [],
}

# Sample experiment dataset
SAMPLE_EXPERIMENT = {
    "id": "exp_001",
    "name": "Temperature Analysis",
    "description": "Analysis of temperature variations",
    "authors": ["Dr. Smith", "Dr. Jones"],
    "date": "2024-01-01",
    "data_format": "json",
    "results": {
        "mean": 23.5,
        "std": 2.1,
        "samples": 1000,
    },
}


def create_mock_image_response(content_type: str = "image/jpeg") -> MockResponse:
    """Create a mock response for image requests."""
    content = SAMPLE_JPEG if "jpeg" in content_type else SAMPLE_PNG
    return MockResponse(
        content=content,
        status_code=200,
        headers={"Content-Type": content_type},
    )


def create_mock_json_response(data: dict) -> MockResponse:
    """Create a mock response for JSON requests."""
    import json
    return MockResponse(
        content=json.dumps(data).encode("utf-8"),
        status_code=200,
        headers={"Content-Type": "application/json"},
    )


def create_mock_text_response(text: str) -> MockResponse:
    """Create a mock response for text requests."""
    return MockResponse(
        content=text.encode("utf-8"),
        status_code=200,
        headers={"Content-Type": "text/plain"},
    )


def create_mock_error_response(status_code: int = 404) -> MockResponse:
    """Create a mock error response."""
    return MockResponse(
        content=b"Error",
        status_code=status_code,
        headers={"Content-Type": "text/plain"},
    )


def create_mock_rate_limit_response(retry_after: int = 60) -> MockResponse:
    """Create a mock 429 rate limit response."""
    return MockResponse(
        content=b"Rate limited",
        status_code=429,
        headers={
            "Content-Type": "text/plain",
            "Retry-After": str(retry_after),
        },
    )
