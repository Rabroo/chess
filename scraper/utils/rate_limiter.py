"""Rate limiting utilities using token bucket algorithm."""

import asyncio
import threading
import time
from typing import Dict, Optional


class RateLimiter:
    """Thread-safe rate limiter using token bucket algorithm."""

    def __init__(
        self,
        requests_per_second: float = 1.0,
        burst_size: Optional[int] = None,
    ):
        """
        Initialize the rate limiter.

        Args:
            requests_per_second: Maximum requests per second.
            burst_size: Maximum burst size (defaults to 1).
        """
        self.rps = requests_per_second
        self.burst_size = burst_size or 1
        self.tokens = float(self.burst_size)
        self.last_update = time.monotonic()
        self._lock = threading.Lock()
        self._async_lock: Optional[asyncio.Lock] = None

        # For 429 handling
        self._backoff_until: float = 0

    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_update
        self.tokens = min(self.burst_size, self.tokens + elapsed * self.rps)
        self.last_update = now

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire a token, blocking if necessary.

        Args:
            timeout: Maximum time to wait in seconds (None for infinite).

        Returns:
            True if token acquired, False if timeout.
        """
        start_time = time.monotonic()

        while True:
            with self._lock:
                # Check backoff
                now = time.monotonic()
                if now < self._backoff_until:
                    wait_time = self._backoff_until - now
                else:
                    self._refill_tokens()

                    if self.tokens >= 1:
                        self.tokens -= 1
                        return True

                    # Calculate wait time for next token
                    wait_time = (1 - self.tokens) / self.rps

            # Check timeout
            if timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed + wait_time > timeout:
                    return False

            time.sleep(min(wait_time, 0.1))

    def try_acquire(self) -> bool:
        """
        Try to acquire a token without blocking.

        Returns:
            True if token acquired, False otherwise.
        """
        with self._lock:
            now = time.monotonic()
            if now < self._backoff_until:
                return False

            self._refill_tokens()
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

    def handle_429(self, retry_after: Optional[float] = None) -> None:
        """
        Handle a 429 response by increasing backoff.

        Args:
            retry_after: Seconds to wait from Retry-After header.
        """
        with self._lock:
            backoff_time = retry_after if retry_after else 60.0
            self._backoff_until = time.monotonic() + backoff_time
            # Also reduce the rate
            self.rps = max(0.1, self.rps * 0.5)

    def reset_backoff(self) -> None:
        """Reset backoff after successful request."""
        with self._lock:
            self._backoff_until = 0

    async def acquire_async(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire a token asynchronously.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            True if token acquired, False if timeout.
        """
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()

        start_time = time.monotonic()

        while True:
            async with self._async_lock:
                now = time.monotonic()
                if now < self._backoff_until:
                    wait_time = self._backoff_until - now
                else:
                    self._refill_tokens()

                    if self.tokens >= 1:
                        self.tokens -= 1
                        return True

                    wait_time = (1 - self.tokens) / self.rps

            if timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed + wait_time > timeout:
                    return False

            await asyncio.sleep(min(wait_time, 0.1))


class MultiRateLimiter:
    """Manages multiple rate limiters for different modules."""

    def __init__(self, global_rps: float = 2.0):
        """
        Initialize the multi-rate limiter.

        Args:
            global_rps: Global rate limit across all modules.
        """
        self._limiters: Dict[str, RateLimiter] = {}
        self._global_limiter = RateLimiter(global_rps)
        self._lock = threading.Lock()

    def register_module(
        self,
        module_name: str,
        requests_per_second: float,
        burst_size: Optional[int] = None,
    ) -> None:
        """
        Register a rate limiter for a module.

        Args:
            module_name: Name of the module.
            requests_per_second: Module-specific rate limit.
            burst_size: Maximum burst size.
        """
        with self._lock:
            self._limiters[module_name] = RateLimiter(
                requests_per_second=requests_per_second,
                burst_size=burst_size,
            )

    def acquire(self, module_name: str, timeout: Optional[float] = None) -> bool:
        """
        Acquire tokens from both global and module-specific limiters.

        Args:
            module_name: Name of the module.
            timeout: Maximum wait time.

        Returns:
            True if tokens acquired from both limiters.
        """
        # First acquire global token
        if not self._global_limiter.acquire(timeout):
            return False

        # Then acquire module-specific token
        if module_name in self._limiters:
            return self._limiters[module_name].acquire(timeout)

        return True

    async def acquire_async(
        self,
        module_name: str,
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Acquire tokens asynchronously.

        Args:
            module_name: Name of the module.
            timeout: Maximum wait time.

        Returns:
            True if tokens acquired.
        """
        if not await self._global_limiter.acquire_async(timeout):
            return False

        if module_name in self._limiters:
            return await self._limiters[module_name].acquire_async(timeout)

        return True

    def handle_429(self, module_name: str, retry_after: Optional[float] = None) -> None:
        """
        Handle 429 for a specific module.

        Args:
            module_name: Name of the module.
            retry_after: Retry-After header value.
        """
        if module_name in self._limiters:
            self._limiters[module_name].handle_429(retry_after)
        self._global_limiter.handle_429(retry_after)

    def get_limiter(self, module_name: str) -> Optional[RateLimiter]:
        """Get the rate limiter for a module."""
        return self._limiters.get(module_name)
