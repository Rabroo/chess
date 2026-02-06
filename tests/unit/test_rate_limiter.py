"""Unit tests for rate limiter."""

import time
import threading

import pytest

from scraper.utils.rate_limiter import RateLimiter, MultiRateLimiter


class TestRateLimiter:
    """Tests for RateLimiter."""

    def test_acquire_immediate(self):
        limiter = RateLimiter(requests_per_second=10.0, burst_size=5)

        # Should be able to acquire immediately up to burst size
        for _ in range(5):
            assert limiter.try_acquire() is True

    def test_acquire_blocks_after_burst(self):
        limiter = RateLimiter(requests_per_second=10.0, burst_size=2)

        # Exhaust burst
        assert limiter.try_acquire() is True
        assert limiter.try_acquire() is True

        # Should fail immediately
        assert limiter.try_acquire() is False

    def test_acquire_refills_over_time(self):
        limiter = RateLimiter(requests_per_second=10.0, burst_size=1)

        assert limiter.try_acquire() is True
        assert limiter.try_acquire() is False

        # Wait for refill (100ms for 10 RPS)
        time.sleep(0.15)

        assert limiter.try_acquire() is True

    def test_acquire_with_timeout(self):
        limiter = RateLimiter(requests_per_second=2.0, burst_size=1)

        assert limiter.acquire(timeout=1.0) is True
        assert limiter.acquire(timeout=0.1) is False  # Should timeout

    def test_handle_429_increases_backoff(self):
        limiter = RateLimiter(requests_per_second=10.0, burst_size=5)

        # Should be able to acquire
        assert limiter.try_acquire() is True

        # Simulate 429
        limiter.handle_429(retry_after=0.5)

        # Should fail due to backoff
        assert limiter.try_acquire() is False

        # Wait for backoff to expire
        time.sleep(0.6)

        assert limiter.try_acquire() is True

    def test_reset_backoff(self):
        limiter = RateLimiter(requests_per_second=10.0, burst_size=5)

        limiter.handle_429(retry_after=10.0)
        assert limiter.try_acquire() is False

        limiter.reset_backoff()
        assert limiter.try_acquire() is True


class TestMultiRateLimiter:
    """Tests for MultiRateLimiter."""

    def test_register_and_acquire(self):
        limiter = MultiRateLimiter(global_rps=10.0)
        limiter.register_module("images", requests_per_second=5.0)

        # Should be able to acquire
        assert limiter.acquire("images", timeout=0.1) is True

    def test_respects_global_limit(self):
        limiter = MultiRateLimiter(global_rps=1.0)
        limiter.register_module("images", requests_per_second=10.0)
        limiter.register_module("social", requests_per_second=10.0)

        # Exhaust global limit
        assert limiter.acquire("images", timeout=0.1) is True

        # Should fail even though module limit is high
        start = time.time()
        result = limiter.acquire("social", timeout=0.1)
        elapsed = time.time() - start

        # Either timed out or took significant time
        assert not result or elapsed >= 0.05

    def test_respects_module_limit(self):
        limiter = MultiRateLimiter(global_rps=100.0)
        limiter.register_module("images", requests_per_second=1.0, burst_size=1)

        assert limiter.acquire("images", timeout=0.1) is True
        assert limiter.acquire("images", timeout=0.1) is False

    def test_handle_429_affects_module(self):
        limiter = MultiRateLimiter(global_rps=10.0)
        limiter.register_module("images", requests_per_second=10.0)

        limiter.handle_429("images", retry_after=1.0)

        # Should fail due to backoff
        assert limiter.acquire("images", timeout=0.1) is False

    def test_get_limiter(self):
        limiter = MultiRateLimiter(global_rps=10.0)
        limiter.register_module("images", requests_per_second=5.0)

        module_limiter = limiter.get_limiter("images")
        assert module_limiter is not None
        assert isinstance(module_limiter, RateLimiter)

        assert limiter.get_limiter("nonexistent") is None

    def test_unregistered_module_uses_global(self):
        limiter = MultiRateLimiter(global_rps=10.0)

        # Acquiring for unregistered module should still work
        assert limiter.acquire("unknown_module", timeout=0.1) is True
