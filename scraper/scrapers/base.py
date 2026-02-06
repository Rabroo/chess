"""Base scraper module with abstract class and auto-registration."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Type

from ..utils.config import Config
from ..utils.directory import DirectoryManager
from ..utils.hashing import HashManager
from ..utils.logger import ScraperLogger
from ..utils.rate_limiter import RateLimiter
from ..utils.validator import Validator


@dataclass
class ScrapedItem:
    """Represents a scraped item."""
    content: bytes
    identifier: str
    metadata: Dict[str, Any]
    content_type: Optional[str] = None
    source_url: Optional[str] = None


@dataclass
class ScrapeResult:
    """Result of a scraping operation."""
    success: bool
    identifier: str
    error: Optional[str] = None
    file_path: Optional[Path] = None


class ScraperRegistry:
    """Registry for scraper modules with auto-discovery."""

    _modules: Dict[str, Type["ScraperModule"]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a scraper module.

        Args:
            name: Module name for CLI --type argument.
        """
        def decorator(scraper_class: Type["ScraperModule"]):
            cls._modules[name] = scraper_class
            scraper_class.MODULE_NAME = name
            return scraper_class
        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Type["ScraperModule"]]:
        """Get a scraper module by name."""
        return cls._modules.get(name)

    @classmethod
    def list_modules(cls) -> list[str]:
        """List all registered module names."""
        return list(cls._modules.keys())

    @classmethod
    def create(
        cls,
        name: str,
        config: Config,
        logger: ScraperLogger,
        directory_manager: DirectoryManager,
    ) -> Optional["ScraperModule"]:
        """
        Create a scraper module instance.

        Args:
            name: Module name.
            config: Configuration instance.
            logger: Logger instance.
            directory_manager: Directory manager instance.

        Returns:
            Scraper module instance or None if not found.
        """
        module_class = cls.get(name)
        if module_class is None:
            return None
        return module_class(config, logger, directory_manager)


class ScraperModule(ABC):
    """Abstract base class for all scraper modules."""

    MODULE_NAME: str = "base"
    ALLOWED_CONTENT_TYPES: set[str] = set()

    def __init__(
        self,
        config: Config,
        logger: ScraperLogger,
        directory_manager: DirectoryManager,
    ):
        """
        Initialize the scraper module.

        Args:
            config: Configuration instance.
            logger: Logger instance.
            directory_manager: Directory manager instance.
        """
        self.config = config
        self.logger = logger
        self.dir_manager = directory_manager
        self.validator = Validator(config.storage.max_file_size_mb)
        self.hash_manager = HashManager(
            self.output_dir / ".hashes" if self.output_dir else None
        )

    @property
    def output_dir(self) -> Path:
        """Get the output directory for this module."""
        return self.dir_manager.get_module_dir(self.MODULE_NAME)

    @property
    def default_limit(self) -> int:
        """Get the default item limit for this module."""
        module_config = getattr(self.config.modules, self.MODULE_NAME.replace("-", "_"), None)
        if module_config:
            return module_config.default_limit
        return 100

    @abstractmethod
    def validate_input(self, input_value: str) -> bool:
        """
        Validate the input value for this module.

        Args:
            input_value: Input from --input CLI argument.

        Returns:
            True if input is valid.

        Raises:
            ValidationError: If input is invalid.
        """
        pass

    @abstractmethod
    def fetch(self, input_value: str, limit: int) -> Iterator[ScrapedItem]:
        """
        Fetch items from the source.

        Args:
            input_value: Input from --input CLI argument.
            limit: Maximum number of items to fetch.

        Yields:
            ScrapedItem instances.
        """
        pass

    @abstractmethod
    def get_hash(self, item: ScrapedItem) -> str:
        """
        Generate a hash for duplicate detection.

        Args:
            item: Scraped item.

        Returns:
            Hash string.
        """
        pass

    def get_filename(self, item: ScrapedItem) -> str:
        """
        Generate a filename for the item.

        Args:
            item: Scraped item.

        Returns:
            Sanitized filename.
        """
        return self.validator.sanitize_filename(item.identifier)

    def store(self, item: ScrapedItem) -> ScrapeResult:
        """
        Store a scraped item.

        Args:
            item: Item to store.

        Returns:
            ScrapeResult indicating success or failure.
        """
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

            # Validate content type if applicable
            if self.ALLOWED_CONTENT_TYPES and item.content_type:
                self.validator.validate_content_type(
                    item.content_type,
                    self.ALLOWED_CONTENT_TYPES,
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

            # Write to temp directory first (sandboxing)
            temp_dir = self.dir_manager.get_temp_dir()
            filename = self.get_filename(item)
            temp_path = temp_dir / filename

            with open(temp_path, "wb") as f:
                f.write(item.content)

            # Validate magic bytes if applicable
            if self.ALLOWED_CONTENT_TYPES and item.content_type:
                self.validator.validate_magic_bytes(item.content, item.content_type)

            # Move to final destination
            final_path = self.output_dir / filename
            self.dir_manager.move_from_temp(temp_path, final_path)

            # Store metadata if present
            if item.metadata:
                self._store_metadata(item, final_path)

            self.logger.item_saved(self.MODULE_NAME, item.identifier)
            return ScrapeResult(
                success=True,
                identifier=item.identifier,
                file_path=final_path,
            )

        except Exception as e:
            self.logger.error(
                f"Failed to store {item.identifier}: {e}",
                module=self.MODULE_NAME,
                action="STORE_ERROR",
            )
            return ScrapeResult(
                success=False,
                identifier=item.identifier,
                error=str(e),
            )

    def _store_metadata(self, item: ScrapedItem, file_path: Path) -> None:
        """Store metadata JSON alongside the item."""
        import json

        metadata_path = file_path.with_suffix(file_path.suffix + ".meta.json")
        with open(metadata_path, "w") as f:
            json.dump(item.metadata, f, indent=2, default=str)

    def run(
        self,
        input_value: str,
        limit: Optional[int] = None,
    ) -> Dict[str, int]:
        """
        Run the scraper.

        Args:
            input_value: Input from --input CLI argument.
            limit: Maximum items to scrape (None for default).

        Returns:
            Dictionary with counts: processed, duplicates, saved.
        """
        self.validate_input(input_value)

        effective_limit = limit or self.default_limit
        self.logger.start_job(self.MODULE_NAME, input_value)

        processed = 0
        duplicates = 0
        saved = 0

        try:
            for item in self.fetch(input_value, effective_limit):
                processed += 1
                result = self.store(item)

                if result.success:
                    saved += 1
                elif result.error == "Duplicate item":
                    duplicates += 1

                if processed >= effective_limit:
                    break

        finally:
            self.dir_manager.cleanup_temp_dir()
            self.logger.end_job(self.MODULE_NAME, processed, duplicates, saved)

        return {
            "processed": processed,
            "duplicates": duplicates,
            "saved": saved,
        }
