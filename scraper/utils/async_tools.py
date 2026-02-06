"""Async utilities for concurrent scraping."""

import asyncio
import threading
from pathlib import Path
from typing import Any, Callable, Coroutine, Generic, List, Optional, TypeVar

import aiofiles

T = TypeVar("T")
R = TypeVar("R")


class AsyncTaskQueue(Generic[T, R]):
    """Async task queue with configurable concurrency."""

    def __init__(
        self,
        worker_func: Callable[[T], Coroutine[Any, Any, R]],
        max_workers: int = 5,
        queue_size: int = 100,
    ):
        """
        Initialize the task queue.

        Args:
            worker_func: Async function to process each item.
            max_workers: Maximum concurrent workers.
            queue_size: Maximum queue size (0 for unlimited).
        """
        self.worker_func = worker_func
        self.max_workers = max_workers
        self.queue_size = queue_size
        self._queue: asyncio.Queue[Optional[T]] = asyncio.Queue(
            maxsize=queue_size if queue_size > 0 else 0
        )
        self._results: List[R] = []
        self._errors: List[Exception] = []
        self._results_lock = asyncio.Lock()
        self._running = False
        self._workers: List[asyncio.Task] = []

    async def _worker(self, worker_id: int) -> None:
        """Worker coroutine that processes items from the queue."""
        while True:
            item = await self._queue.get()

            if item is None:
                # Poison pill - shutdown signal
                self._queue.task_done()
                break

            try:
                result = await self.worker_func(item)
                async with self._results_lock:
                    self._results.append(result)
            except Exception as e:
                async with self._results_lock:
                    self._errors.append(e)
            finally:
                self._queue.task_done()

    async def start(self) -> None:
        """Start the worker pool."""
        if self._running:
            return

        self._running = True
        self._results = []
        self._errors = []
        self._workers = [
            asyncio.create_task(self._worker(i))
            for i in range(self.max_workers)
        ]

    async def submit(self, item: T) -> None:
        """Submit an item to the queue."""
        await self._queue.put(item)

    async def submit_many(self, items: List[T]) -> None:
        """Submit multiple items to the queue."""
        for item in items:
            await self._queue.put(item)

    async def join(self) -> None:
        """Wait for all items to be processed."""
        await self._queue.join()

    async def stop(self) -> None:
        """Stop all workers gracefully."""
        # Send poison pills
        for _ in range(self.max_workers):
            await self._queue.put(None)

        # Wait for workers to finish
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._running = False

    async def run(self, items: List[T]) -> List[R]:
        """
        Process all items and return results.

        Args:
            items: List of items to process.

        Returns:
            List of results.
        """
        await self.start()
        await self.submit_many(items)
        await self.join()
        await self.stop()
        return self._results

    def get_results(self) -> List[R]:
        """Get all results collected so far."""
        return self._results.copy()

    def get_errors(self) -> List[Exception]:
        """Get all errors encountered."""
        return self._errors.copy()


class ThreadSafeFileWriter:
    """Thread-safe file writer with locking."""

    def __init__(self):
        """Initialize the file writer."""
        self._locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    def _get_lock(self, path: str) -> threading.Lock:
        """Get or create a lock for a file path."""
        with self._global_lock:
            if path not in self._locks:
                self._locks[path] = threading.Lock()
            return self._locks[path]

    def write(self, path: Path, content: bytes) -> None:
        """
        Write content to file with locking.

        Args:
            path: File path.
            content: Content to write.
        """
        lock = self._get_lock(str(path))
        with lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                f.write(content)

    def append(self, path: Path, content: bytes) -> None:
        """
        Append content to file with locking.

        Args:
            path: File path.
            content: Content to append.
        """
        lock = self._get_lock(str(path))
        with lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "ab") as f:
                f.write(content)


class AsyncFileWriter:
    """Async file writer with locking."""

    def __init__(self):
        """Initialize the async file writer."""
        self._locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    async def _get_lock(self, path: str) -> asyncio.Lock:
        """Get or create a lock for a file path."""
        async with self._global_lock:
            if path not in self._locks:
                self._locks[path] = asyncio.Lock()
            return self._locks[path]

    async def write(self, path: Path, content: bytes) -> None:
        """
        Write content to file with locking.

        Args:
            path: File path.
            content: Content to write.
        """
        lock = await self._get_lock(str(path))
        async with lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(path, "wb") as f:
                await f.write(content)

    async def append(self, path: Path, content: bytes) -> None:
        """
        Append content to file with locking.

        Args:
            path: File path.
            content: Content to append.
        """
        lock = await self._get_lock(str(path))
        async with lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(path, "ab") as f:
                await f.write(content)


async def gather_with_concurrency(
    n: int,
    *coros: Coroutine[Any, Any, T],
) -> List[T]:
    """
    Run coroutines with limited concurrency.

    Args:
        n: Maximum concurrent coroutines.
        *coros: Coroutines to run.

    Returns:
        List of results.
    """
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro: Coroutine[Any, Any, T]) -> T:
        async with semaphore:
            return await coro

    return await asyncio.gather(*(sem_coro(c) for c in coros))
