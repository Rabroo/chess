"""Logging utilities with rotation and structured formatting."""

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


class ScraperFormatter(logging.Formatter):
    """Custom formatter for scraper logs: [TIMESTAMP] MODULE:ACTION:DETAILS"""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        module = getattr(record, "scraper_module", record.name)
        action = getattr(record, "action", "LOG")
        details = record.getMessage()
        return f"[{timestamp}] {module}:{action}:{details}"


class ScraperLogger:
    """Logger with structured formatting and rotation support."""

    def __init__(
        self,
        name: str,
        log_path: Optional[Path] = None,
        max_bytes: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5,
        console_output: bool = True,
    ):
        """
        Initialize the scraper logger.

        Args:
            name: Logger name (typically module name).
            log_path: Path to the log file.
            max_bytes: Maximum log file size before rotation.
            backup_count: Number of backup files to keep.
            console_output: Whether to output to console.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        formatter = ScraperFormatter()

        # File handler with rotation
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def _log(
        self,
        level: int,
        message: str,
        module: Optional[str] = None,
        action: str = "LOG",
    ) -> None:
        """Internal logging method with extra fields."""
        extra = {
            "scraper_module": module or self.logger.name,
            "action": action,
        }
        self.logger.log(level, message, extra=extra)

    def info(self, message: str, module: Optional[str] = None, action: str = "INFO") -> None:
        """Log an info message."""
        self._log(logging.INFO, message, module, action)

    def debug(self, message: str, module: Optional[str] = None, action: str = "DEBUG") -> None:
        """Log a debug message."""
        self._log(logging.DEBUG, message, module, action)

    def warning(self, message: str, module: Optional[str] = None, action: str = "WARNING") -> None:
        """Log a warning message."""
        self._log(logging.WARNING, message, module, action)

    def error(self, message: str, module: Optional[str] = None, action: str = "ERROR") -> None:
        """Log an error message."""
        self._log(logging.ERROR, message, module, action)

    def start_job(self, module: str, input_value: str) -> None:
        """Log the start of a scraping job."""
        self._log(logging.INFO, f"Starting job with input: {input_value}", module, "START")

    def end_job(self, module: str, processed: int, duplicates: int, saved: int) -> None:
        """Log the end of a scraping job."""
        self._log(
            logging.INFO,
            f"Completed - processed: {processed}, duplicates: {duplicates}, saved: {saved}",
            module,
            "END",
        )

    def item_saved(self, module: str, identifier: str) -> None:
        """Log a successfully saved item."""
        self._log(logging.DEBUG, f"Saved: {identifier}", module, "SAVED")

    def duplicate_skipped(self, module: str, identifier: str) -> None:
        """Log a skipped duplicate."""
        self._log(logging.INFO, f"SKIPPED_DUPLICATE: {identifier}", module, "DUPLICATE")

    def directory_created(self, path: Path) -> None:
        """Log directory creation."""
        self._log(logging.INFO, f"Created directory: {path}", "SYSTEM", "DIR_CREATE")

    def retry_attempt(self, module: str, attempt: int, max_attempts: int, error: str) -> None:
        """Log a retry attempt."""
        self._log(
            logging.WARNING,
            f"Retry {attempt}/{max_attempts}: {error}",
            module,
            "RETRY",
        )


# Global logger instance
_global_logger: Optional[ScraperLogger] = None


def get_logger(
    name: str = "scraper",
    log_path: Optional[Path] = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> ScraperLogger:
    """Get or create a scraper logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = ScraperLogger(
            name=name,
            log_path=log_path,
            max_bytes=max_bytes,
            backup_count=backup_count,
        )
    return _global_logger


def init_logger(
    log_path: Path,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> ScraperLogger:
    """Initialize the global logger with specific settings."""
    global _global_logger
    _global_logger = ScraperLogger(
        name="scraper",
        log_path=log_path,
        max_bytes=max_bytes,
        backup_count=backup_count,
    )
    return _global_logger
