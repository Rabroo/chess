"""Download utilities with retry logic and streaming support."""

import asyncio
import time
from pathlib import Path
from typing import AsyncIterator, Iterator, Optional, Tuple

import aiofiles
import aiohttp
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class DownloadError(Exception):
    """Raised when download fails."""
    pass


class Downloader:
    """Synchronous downloader with retry logic and streaming support."""

    def __init__(
        self,
        timeout: int = 10,
        retries: int = 3,
        retry_backoff: float = 2.0,
        user_agent: str = "UniversalScraper/1.0",
        max_file_size: Optional[int] = None,
    ):
        """
        Initialize the downloader.

        Args:
            timeout: Request timeout in seconds.
            retries: Number of retry attempts.
            retry_backoff: Backoff multiplier for retries.
            user_agent: User-Agent header value.
            max_file_size: Maximum file size in bytes (None for unlimited).
        """
        self.timeout = timeout
        self.retries = retries
        self.retry_backoff = retry_backoff
        self.user_agent = user_agent
        self.max_file_size = max_file_size

        # Configure session with retry
        self.session = requests.Session()
        retry_strategy = Retry(
            total=retries,
            backoff_factor=retry_backoff,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers["User-Agent"] = user_agent

    def get_headers(self, url: str) -> dict:
        """
        Get headers for a URL without downloading content.

        Args:
            url: URL to check.

        Returns:
            Response headers as dict.
        """
        response = self.session.head(url, timeout=self.timeout, allow_redirects=True)
        response.raise_for_status()
        return dict(response.headers)

    def get_content_type(self, url: str) -> Optional[str]:
        """
        Get content type for a URL.

        Args:
            url: URL to check.

        Returns:
            Content-Type header value or None.
        """
        headers = self.get_headers(url)
        return headers.get("Content-Type", headers.get("content-type"))

    def get_content_length(self, url: str) -> Optional[int]:
        """
        Get content length for a URL.

        Args:
            url: URL to check.

        Returns:
            Content-Length in bytes or None.
        """
        headers = self.get_headers(url)
        length = headers.get("Content-Length", headers.get("content-length"))
        return int(length) if length else None

    def download(self, url: str) -> Tuple[bytes, dict]:
        """
        Download content from URL.

        Args:
            url: URL to download.

        Returns:
            Tuple of (content bytes, response headers).

        Raises:
            DownloadError: If download fails.
        """
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            if self.max_file_size and len(response.content) > self.max_file_size:
                raise DownloadError(
                    f"File size {len(response.content)} exceeds maximum {self.max_file_size}"
                )

            return response.content, dict(response.headers)
        except requests.RequestException as e:
            raise DownloadError(f"Download failed: {e}")

    def download_to_file(
        self,
        url: str,
        destination: Path,
        chunk_size: int = 8192,
    ) -> Tuple[Path, dict]:
        """
        Stream download to file.

        Args:
            url: URL to download.
            destination: Destination file path.
            chunk_size: Size of chunks for streaming.

        Returns:
            Tuple of (file path, response headers).

        Raises:
            DownloadError: If download fails.
        """
        try:
            response = self.session.get(url, timeout=self.timeout, stream=True)
            response.raise_for_status()

            destination.parent.mkdir(parents=True, exist_ok=True)
            total_size = 0

            with open(destination, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        if self.max_file_size and total_size + len(chunk) > self.max_file_size:
                            f.close()
                            destination.unlink(missing_ok=True)
                            raise DownloadError(
                                f"File size exceeds maximum {self.max_file_size}"
                            )
                        f.write(chunk)
                        total_size += len(chunk)

            return destination, dict(response.headers)
        except requests.RequestException as e:
            destination.unlink(missing_ok=True)
            raise DownloadError(f"Download failed: {e}")

    def download_with_retry(
        self,
        url: str,
        on_retry: Optional[callable] = None,
    ) -> Tuple[bytes, dict]:
        """
        Download with manual retry logic and callbacks.

        Args:
            url: URL to download.
            on_retry: Callback function(attempt, max_attempts, error).

        Returns:
            Tuple of (content bytes, response headers).

        Raises:
            DownloadError: If all retries fail.
        """
        last_error = None

        for attempt in range(1, self.retries + 1):
            try:
                return self.download(url)
            except DownloadError as e:
                last_error = e
                if on_retry:
                    on_retry(attempt, self.retries, str(e))
                if attempt < self.retries:
                    sleep_time = self.retry_backoff ** attempt
                    time.sleep(sleep_time)

        raise DownloadError(f"All {self.retries} retries failed: {last_error}")

    def close(self) -> None:
        """Close the session."""
        self.session.close()


class AsyncDownloader:
    """Asynchronous downloader with retry logic and streaming support."""

    def __init__(
        self,
        timeout: int = 10,
        retries: int = 3,
        retry_backoff: float = 2.0,
        user_agent: str = "UniversalScraper/1.0",
        max_file_size: Optional[int] = None,
    ):
        """
        Initialize the async downloader.

        Args:
            timeout: Request timeout in seconds.
            retries: Number of retry attempts.
            retry_backoff: Backoff multiplier for retries.
            user_agent: User-Agent header value.
            max_file_size: Maximum file size in bytes.
        """
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.retries = retries
        self.retry_backoff = retry_backoff
        self.user_agent = user_agent
        self.max_file_size = max_file_size
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=self.timeout,
                headers={"User-Agent": self.user_agent},
            )
        return self._session

    async def get_headers(self, url: str) -> dict:
        """Get headers for a URL."""
        session = await self._get_session()
        async with session.head(url, allow_redirects=True) as response:
            response.raise_for_status()
            return dict(response.headers)

    async def download(self, url: str) -> Tuple[bytes, dict]:
        """
        Download content from URL.

        Args:
            url: URL to download.

        Returns:
            Tuple of (content bytes, response headers).

        Raises:
            DownloadError: If download fails.
        """
        session = await self._get_session()

        try:
            async with session.get(url) as response:
                response.raise_for_status()
                content = await response.read()

                if self.max_file_size and len(content) > self.max_file_size:
                    raise DownloadError(
                        f"File size {len(content)} exceeds maximum {self.max_file_size}"
                    )

                return content, dict(response.headers)
        except aiohttp.ClientError as e:
            raise DownloadError(f"Download failed: {e}")

    async def download_to_file(
        self,
        url: str,
        destination: Path,
        chunk_size: int = 8192,
    ) -> Tuple[Path, dict]:
        """
        Stream download to file asynchronously.

        Args:
            url: URL to download.
            destination: Destination file path.
            chunk_size: Size of chunks for streaming.

        Returns:
            Tuple of (file path, response headers).

        Raises:
            DownloadError: If download fails.
        """
        session = await self._get_session()

        try:
            async with session.get(url) as response:
                response.raise_for_status()
                headers = dict(response.headers)

                destination.parent.mkdir(parents=True, exist_ok=True)
                total_size = 0

                async with aiofiles.open(destination, "wb") as f:
                    async for chunk in response.content.iter_chunked(chunk_size):
                        if self.max_file_size and total_size + len(chunk) > self.max_file_size:
                            await f.close()
                            destination.unlink(missing_ok=True)
                            raise DownloadError(
                                f"File size exceeds maximum {self.max_file_size}"
                            )
                        await f.write(chunk)
                        total_size += len(chunk)

                return destination, headers
        except aiohttp.ClientError as e:
            destination.unlink(missing_ok=True)
            raise DownloadError(f"Download failed: {e}")

    async def download_with_retry(
        self,
        url: str,
        on_retry: Optional[callable] = None,
    ) -> Tuple[bytes, dict]:
        """
        Download with retry logic.

        Args:
            url: URL to download.
            on_retry: Async callback function(attempt, max_attempts, error).

        Returns:
            Tuple of (content bytes, response headers).

        Raises:
            DownloadError: If all retries fail.
        """
        last_error = None

        for attempt in range(1, self.retries + 1):
            try:
                return await self.download(url)
            except DownloadError as e:
                last_error = e
                if on_retry:
                    await on_retry(attempt, self.retries, str(e))
                if attempt < self.retries:
                    sleep_time = self.retry_backoff ** attempt
                    await asyncio.sleep(sleep_time)

        raise DownloadError(f"All {self.retries} retries failed: {last_error}")

    async def close(self) -> None:
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()
