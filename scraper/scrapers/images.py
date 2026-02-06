"""Image scraper module with search functionality for ML training data."""

import json
import random
import re
import time
from pathlib import Path
from typing import Iterator, List, Optional
from urllib.parse import urljoin, urlparse, quote_plus

import requests

from ..utils.downloader import Downloader, DownloadError
from ..utils.hashing import HashManager
from ..utils.validator import ValidationError
from .base import ScrapedItem, ScraperModule, ScraperRegistry


@ScraperRegistry.register("images")
class ImageScraper(ScraperModule):
    """Scraper for downloading images from search queries or URLs.

    Supports:
    - Search queries: "cats", "golden retriever dogs"
    - Direct URLs: https://example.com/image.jpg
    - File with URLs: @urls.txt
    - Multiple labels: "cats,dogs,sheep" (creates labeled subdirectories)
    """

    MODULE_NAME = "images"
    ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.downloader = Downloader(
            timeout=self.config.network.timeout,
            retries=self.config.network.retries,
            retry_backoff=self.config.network.retry_backoff,
            user_agent=self.config.network.user_agent,
            max_file_size=self.config.storage.max_file_size_mb * 1024 * 1024,
        )
        self._current_label: Optional[str] = None

    def validate_input(self, input_value: str) -> bool:
        """
        Validate input - can be search query, URL, or file path.

        Args:
            input_value: Search query, URL(s), or file path.

        Returns:
            True if valid.

        Raises:
            ValidationError: If invalid.
        """
        # File with URLs
        if input_value.startswith("@"):
            file_path = input_value[1:]
            if not Path(file_path).exists():
                raise ValidationError(f"File not found: {file_path}")
            return True

        # Direct URL(s)
        if input_value.startswith("http"):
            urls = [u.strip() for u in input_value.split(",")]
            for url in urls:
                if url.startswith("http"):
                    self.validator.validate_url(url)
            return True

        # Search query - any non-empty string is valid
        if len(input_value.strip()) > 0:
            return True

        raise ValidationError("Input cannot be empty")

    def _is_search_query(self, input_value: str) -> bool:
        """Check if input is a search query (not a URL or file)."""
        return (
            not input_value.startswith("http") and
            not input_value.startswith("@")
        )

    # Chrome on Intel Mac User-Agent
    CHROME_USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    def _random_delay(self, min_sec: float = 0.1, max_sec: float = 0.5) -> None:
        """Add a random delay to avoid bot detection."""
        delay = random.uniform(min_sec, max_sec)
        time.sleep(delay)

    def _search_images(self, query: str, limit: int, max_retries: int = 3) -> List[dict]:
        """
        Search for images using Google Images.

        Args:
            query: Search query (e.g., "cats", "golden retriever")
            limit: Maximum number of results.
            max_retries: Number of retry attempts on failure.

        Returns:
            List of image result dictionaries with 'image' (URL) and 'title'.
        """
        self.logger.info(
            f"Searching Google Images for '{query}' (limit: {limit})",
            module=self.MODULE_NAME,
            action="SEARCH",
        )

        headers = {
            "User-Agent": self.CHROME_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
        }

        for attempt in range(max_retries):
            try:
                # Add minimal delay before search
                initial_delay = random.uniform(0.2, 0.5)
                time.sleep(initial_delay)

                if attempt > 0:
                    wait_time = (2 ** attempt) + random.uniform(1, 3)
                    self.logger.info(
                        f"Retry {attempt + 1}/{max_retries} after {wait_time:.1f}s delay",
                        module=self.MODULE_NAME,
                    )
                    time.sleep(wait_time)

                # Google Images search URL
                encoded_query = quote_plus(query)
                url = f"https://www.google.com/search?q={encoded_query}&tbm=isch&hl=en"

                session = requests.Session()
                response = session.get(url, headers=headers, timeout=15)
                response.raise_for_status()

                html = response.text
                results = []

                # Extract image URLs from Google's response
                # Method 1: Look for data in AF_initDataCallback
                patterns = [
                    r'\["(https?://[^"]+\.(?:jpg|jpeg|png|gif|webp))",[0-9]+,[0-9]+\]',
                    r'"ou":"(https?://[^"]+)"',
                    r'\["(https?://[^"]+)",\d+,\d+\]',
                ]

                found_urls = set()
                for pattern in patterns:
                    matches = re.findall(pattern, html, re.IGNORECASE)
                    for match in matches:
                        # Filter out Google's own URLs and thumbnails
                        if (match.startswith("http") and
                            "google.com" not in match and
                            "gstatic.com" not in match and
                            "googleapis.com" not in match and
                            len(match) < 2000):
                            found_urls.add(match)

                # Also try to find image data in script tags
                script_pattern = r'<script[^>]*>AF_initDataCallback\(({.*?})\);</script>'
                script_matches = re.findall(script_pattern, html, re.DOTALL)

                for script_data in script_matches:
                    # Look for image URLs in the script data
                    img_urls = re.findall(
                        r'(https?://[^\s"\'<>]+\.(?:jpg|jpeg|png|gif|webp)[^\s"\'<>]*)',
                        script_data,
                        re.IGNORECASE
                    )
                    for img_url in img_urls:
                        if ("google.com" not in img_url and
                            "gstatic.com" not in img_url and
                            len(img_url) < 2000):
                            found_urls.add(img_url.split("?")[0])  # Remove query params

                # Convert to result format
                for img_url in list(found_urls)[:limit]:
                    results.append({
                        "image": img_url,
                        "title": query,
                    })

                self.logger.info(
                    f"Found {len(results)} images for '{query}'",
                    module=self.MODULE_NAME,
                    action="SEARCH_COMPLETE",
                )
                return results

            except requests.exceptions.RequestException as e:
                self.logger.error(
                    f"Search request failed for '{query}': {e}",
                    module=self.MODULE_NAME,
                )
            except Exception as e:
                self.logger.error(
                    f"Search failed for '{query}': {e}",
                    module=self.MODULE_NAME,
                )

        return []

    def _extract_urls_from_input(self, input_value: str) -> List[str]:
        """Extract URLs from file or comma-separated input."""
        urls = []

        if input_value.startswith("@"):
            file_path = input_value[1:]
            with open(file_path, "r") as f:
                urls = [line.strip() for line in f if line.strip()]

        elif "," in input_value and input_value.startswith("http"):
            urls = [u.strip() for u in input_value.split(",") if u.strip().startswith("http")]

        elif input_value.startswith("http"):
            urls = [input_value]

        return urls

    def _get_image_extension(self, content_type: str, url: str) -> str:
        """Determine file extension from content type or URL."""
        content_type = content_type.lower()

        if "jpeg" in content_type or "jpg" in content_type:
            return ".jpg"
        elif "png" in content_type:
            return ".png"
        elif "gif" in content_type:
            return ".gif"
        elif "webp" in content_type:
            return ".webp"

        # Fallback to URL extension
        parsed = urlparse(url)
        path = parsed.path.lower()
        for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
            if path.endswith(ext):
                return ".jpg" if ext == ".jpeg" else ext

        return ".jpg"

    def _download_image(self, url: str, index: int, label: Optional[str] = None) -> Optional[ScrapedItem]:
        """Download a single image and return ScrapedItem."""
        try:
            # Validate URL
            self.validator.validate_url(url)

            # Check content type first
            try:
                content_type = self.downloader.get_content_type(url)
                if content_type:
                    main_type = content_type.split(";")[0].strip().lower()
                    if main_type not in self.ALLOWED_CONTENT_TYPES:
                        self.logger.debug(
                            f"Skipping {url}: invalid content type {main_type}",
                            module=self.MODULE_NAME,
                        )
                        return None
            except Exception:
                # If HEAD request fails, try downloading anyway
                pass

            # Download the image
            content, headers = self.downloader.download(url)

            # Get content type from response
            actual_type = headers.get("Content-Type", "image/jpeg").split(";")[0].strip().lower()

            # Validate magic bytes
            try:
                self.validator.validate_magic_bytes(content, actual_type)
            except ValidationError:
                self.logger.debug(
                    f"Skipping {url}: content does not match declared type",
                    module=self.MODULE_NAME,
                )
                return None

            # Generate filename
            ext = self._get_image_extension(actual_type, url)
            parsed_url = urlparse(url)
            base_name = parsed_url.path.split("/")[-1] or f"image_{index:06d}"

            # Clean up filename
            base_name = re.sub(r'[^\w\-.]', '_', base_name)
            if not base_name.lower().endswith(ext):
                base_name = f"{base_name}{ext}"

            return ScrapedItem(
                content=content,
                identifier=base_name,
                metadata={
                    "source_url": url,
                    "content_type": actual_type,
                    "size_bytes": len(content),
                    "label": label,
                    "index": index,
                },
                content_type=actual_type,
                source_url=url,
            )

        except (ValidationError, DownloadError) as e:
            self.logger.debug(
                f"Failed to fetch {url}: {e}",
                module=self.MODULE_NAME,
            )
            return None

    def fetch(self, input_value: str, limit: int) -> Iterator[ScrapedItem]:
        """
        Fetch images from search queries or URLs.

        Args:
            input_value: Search query, URL(s), or file path.
                - "cats" → searches DuckDuckGo for cat images
                - "cats,dogs,sheep" → searches for each label separately
                - "https://..." → downloads directly
                - "@urls.txt" → reads URLs from file

            limit: Maximum images to fetch per label.

        Yields:
            ScrapedItem for each image.
        """
        # Handle search queries
        if self._is_search_query(input_value):
            # Check if multiple labels (comma-separated search terms)
            labels = [l.strip() for l in input_value.split(",") if l.strip()]

            for label in labels:
                self._current_label = label
                self.logger.info(
                    f"Collecting images for label: '{label}'",
                    module=self.MODULE_NAME,
                    action="LABEL_START",
                )

                # Search for images (fetch extra to compensate for download failures)
                results = self._search_images(label, limit * 10)

                downloaded = 0
                for i, result in enumerate(results):
                    if downloaded >= limit:
                        break

                    url = result.get("image")
                    if not url:
                        continue

                    # Minimal delay between downloads
                    if i > 0:
                        self._random_delay(0.1, 0.3)

                    item = self._download_image(url, i, label=label)
                    if item:
                        downloaded += 1
                        yield item

                self.logger.info(
                    f"Collected {downloaded} images for '{label}'",
                    module=self.MODULE_NAME,
                    action="LABEL_COMPLETE",
                )

        # Handle direct URLs
        else:
            urls = self._extract_urls_from_input(input_value)

            for i, url in enumerate(urls):
                if i >= limit:
                    break

                item = self._download_image(url, i)
                if item:
                    yield item

    def get_hash(self, item: ScrapedItem) -> str:
        """Generate SHA256 hash of image content."""
        return HashManager.sha256_content(item.content)

    def get_filename(self, item: ScrapedItem) -> str:
        """Generate unique filename using hash prefix."""
        content_hash = self.get_hash(item)[:8]
        base_name = self.validator.sanitize_filename(item.identifier)
        name, ext = base_name.rsplit(".", 1) if "." in base_name else (base_name, "jpg")
        return f"{name}_{content_hash}.{ext}"

    def store(self, item: ScrapedItem) -> "ScrapeResult":
        """Store image, organizing by label for ML datasets."""
        from .base import ScrapeResult

        try:
            # Check for duplicates
            item_hash = self.get_hash(item)
            if not self.hash_manager.check_and_add(item_hash):
                self.logger.duplicate_skipped(self.MODULE_NAME, item.identifier)
                return ScrapeResult(
                    success=False,
                    identifier=item.identifier,
                    error="Duplicate item",
                )

            # Validate file size
            self.validator.validate_file_size(len(item.content))

            # Check disk space
            if not self.dir_manager.check_disk_space():
                return ScrapeResult(
                    success=False,
                    identifier=item.identifier,
                    error="Insufficient disk space",
                )

            # Determine output directory (with label subdirectory for ML)
            label = item.metadata.get("label")
            if label:
                # Create label subdirectory for ML dataset organization
                output_dir = self.output_dir / self.validator.sanitize_filename(label)
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                output_dir = self.output_dir

            # Generate filename and save
            filename = self.get_filename(item)
            file_path = output_dir / filename

            with open(file_path, "wb") as f:
                f.write(item.content)

            self.logger.item_saved(self.MODULE_NAME, f"{label}/{filename}" if label else filename)
            return ScrapeResult(
                success=True,
                identifier=item.identifier,
                file_path=file_path,
            )

        except Exception as e:
            self.logger.error(
                f"Failed to store {item.identifier}: {e}",
                module=self.MODULE_NAME,
            )
            return ScrapeResult(
                success=False,
                identifier=item.identifier,
                error=str(e),
            )

    def run(
        self,
        input_value: str,
        limit: Optional[int] = None,
    ) -> dict:
        """
        Run the scraper with summary by label.

        Returns:
            Dictionary with counts per label and totals.
        """
        self.validate_input(input_value)

        effective_limit = limit or self.default_limit
        self.logger.start_job(self.MODULE_NAME, input_value)

        stats = {
            "processed": 0,
            "duplicates": 0,
            "saved": 0,
            "by_label": {},
        }

        try:
            for item in self.fetch(input_value, effective_limit):
                stats["processed"] += 1
                result = self.store(item)

                label = item.metadata.get("label", "unlabeled")
                if label not in stats["by_label"]:
                    stats["by_label"][label] = {"saved": 0, "duplicates": 0}

                if result.success:
                    stats["saved"] += 1
                    stats["by_label"][label]["saved"] += 1
                elif result.error == "Duplicate item":
                    stats["duplicates"] += 1
                    stats["by_label"][label]["duplicates"] += 1

        finally:
            self.dir_manager.cleanup_temp_dir()
            self.logger.end_job(self.MODULE_NAME, stats["processed"], stats["duplicates"], stats["saved"])

        return stats
