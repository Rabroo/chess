"""Experiments dataset scraper module."""

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import urlparse

from ..utils.downloader import Downloader, DownloadError
from ..utils.hashing import HashManager
from ..utils.validator import ValidationError
from .base import ScrapedItem, ScraperModule, ScraperRegistry


# Required metadata fields for experiments
EXPERIMENT_REQUIRED_FIELDS = {"id", "name"}
EXPERIMENT_OPTIONAL_FIELDS = {
    "description", "authors", "date", "source", "methodology",
    "results", "data_format", "license", "doi", "keywords"
}


@ScraperRegistry.register("experiments")
class ExperimentScraper(ScraperModule):
    """Scraper for collecting experimental datasets and reports."""

    MODULE_NAME = "experiments"
    ALLOWED_CONTENT_TYPES = {
        "application/json",
        "text/csv",
        "application/octet-stream",
        "text/plain",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.downloader = Downloader(
            timeout=self.config.network.timeout,
            retries=self.config.network.retries,
            retry_backoff=self.config.network.retry_backoff,
            user_agent=self.config.network.user_agent,
            max_file_size=self.config.storage.max_file_size_mb * 1024 * 1024,
        )

    def validate_input(self, input_value: str) -> bool:
        """
        Validate input - can be URL, dataset ID, or file path.

        Args:
            input_value: URL, ID, or file path.

        Returns:
            True if valid.

        Raises:
            ValidationError: If invalid.
        """
        # URL
        if input_value.startswith("http"):
            self.validator.validate_url(input_value)
            return True

        # File path
        if input_value.startswith("@"):
            file_path = input_value[1:]
            if not Path(file_path).exists():
                raise ValidationError(f"File not found: {file_path}")
            return True

        # Dataset ID or keyword - allow anything else
        if len(input_value.strip()) > 0:
            return True

        raise ValidationError("Input cannot be empty")

    def _validate_metadata_completeness(self, metadata: Dict[str, Any]) -> bool:
        """
        Validate that metadata has required fields.

        Args:
            metadata: Metadata dictionary.

        Returns:
            True if valid.

        Raises:
            ValidationError: If missing required fields.
        """
        missing = EXPERIMENT_REQUIRED_FIELDS - set(metadata.keys())
        if missing:
            raise ValidationError(f"Missing required metadata fields: {missing}")
        return True

    def _extract_metadata_from_content(
        self,
        content: bytes,
        content_type: str,
        source_url: str,
    ) -> Dict[str, Any]:
        """Extract metadata from content based on type."""
        metadata = {
            "source": source_url,
            "content_type": content_type,
            "size_bytes": len(content),
        }

        if "json" in content_type:
            try:
                data = json.loads(content.decode("utf-8"))
                if isinstance(data, dict):
                    # Extract standard metadata fields if present
                    for field in EXPERIMENT_REQUIRED_FIELDS | EXPERIMENT_OPTIONAL_FIELDS:
                        if field in data:
                            metadata[field] = data[field]
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        # Generate ID from URL if not present
        if "id" not in metadata:
            parsed = urlparse(source_url)
            path_parts = parsed.path.strip("/").split("/")
            metadata["id"] = path_parts[-1] if path_parts else "unknown"

        # Generate name from ID if not present
        if "name" not in metadata:
            metadata["name"] = metadata.get("id", "Unknown Dataset")

        return metadata

    def _determine_file_extension(self, content_type: str, url: str) -> str:
        """Determine appropriate file extension."""
        if "json" in content_type:
            return ".json"
        elif "csv" in content_type:
            return ".csv"
        elif "xml" in content_type:
            return ".xml"

        # Try from URL
        parsed = urlparse(url)
        path = parsed.path.lower()
        for ext in [".json", ".csv", ".xml", ".txt", ".dat"]:
            if path.endswith(ext):
                return ext

        return ".dat"  # Default binary

    def fetch(self, input_value: str, limit: int) -> Iterator[ScrapedItem]:
        """
        Fetch experimental datasets.

        Args:
            input_value: URL, dataset ID, keyword, or file path.
            limit: Maximum items to fetch.

        Yields:
            ScrapedItem for each dataset.
        """
        items_yielded = 0

        # Direct URL
        if input_value.startswith("http"):
            try:
                self.validator.validate_url(input_value)

                # Check content type
                headers = self.downloader.get_headers(input_value)
                content_type = headers.get("Content-Type", "application/octet-stream")
                main_type = content_type.split(";")[0].strip().lower()

                # Download content
                content, _ = self.downloader.download(input_value)

                # Extract metadata
                metadata = self._extract_metadata_from_content(
                    content, main_type, input_value
                )

                # Validate metadata
                try:
                    self._validate_metadata_completeness(metadata)
                except ValidationError as e:
                    self.logger.warning(f"Incomplete metadata: {e}", module=self.MODULE_NAME)
                    # Add minimal required fields
                    if "id" not in metadata:
                        metadata["id"] = f"dataset_{items_yielded}"
                    if "name" not in metadata:
                        metadata["name"] = metadata["id"]

                # Determine filename
                ext = self._determine_file_extension(main_type, input_value)
                filename = f"{metadata['id']}{ext}"

                yield ScrapedItem(
                    content=content,
                    identifier=filename,
                    metadata=metadata,
                    content_type=main_type,
                    source_url=input_value,
                )
                items_yielded += 1

            except (ValidationError, DownloadError) as e:
                self.logger.error(f"Failed to fetch: {e}", module=self.MODULE_NAME)

        # File path with list of URLs or datasets
        elif input_value.startswith("@"):
            file_path = Path(input_value[1:])

            with open(file_path, "r") as f:
                content = f.read()

            # Try JSON format
            try:
                data = json.loads(content)
                datasets = data if isinstance(data, list) else [data]

                for dataset in datasets:
                    if items_yielded >= limit:
                        break

                    if isinstance(dataset, str) and dataset.startswith("http"):
                        # It's a URL - fetch it
                        for item in self.fetch(dataset, 1):
                            yield item
                            items_yielded += 1
                    elif isinstance(dataset, dict):
                        # It's dataset metadata/content
                        try:
                            self._validate_metadata_completeness(dataset)
                        except ValidationError:
                            dataset.setdefault("id", f"dataset_{items_yielded}")
                            dataset.setdefault("name", dataset["id"])

                        json_content = json.dumps(dataset, indent=2).encode("utf-8")
                        yield ScrapedItem(
                            content=json_content,
                            identifier=f"{dataset['id']}.json",
                            metadata=dataset,
                            content_type="application/json",
                        )
                        items_yielded += 1

            except json.JSONDecodeError:
                # Treat as list of URLs
                for line in content.split("\n"):
                    url = line.strip()
                    if url and url.startswith("http"):
                        if items_yielded >= limit:
                            break
                        for item in self.fetch(url, 1):
                            yield item
                            items_yielded += 1

        # Dataset ID or keyword - search HuggingFace Datasets
        else:
            self.logger.info(
                f"Searching HuggingFace Datasets for '{input_value}'",
                module=self.MODULE_NAME,
                action="SEARCH",
            )

            try:
                import requests

                # HuggingFace Datasets API
                search_url = "https://huggingface.co/api/datasets"
                params = {"search": input_value, "limit": limit}

                response = requests.get(search_url, params=params, timeout=15)
                response.raise_for_status()

                datasets = response.json()

                self.logger.info(
                    f"Found {len(datasets)} datasets for '{input_value}'",
                    module=self.MODULE_NAME,
                    action="SEARCH_COMPLETE",
                )

                for dataset in datasets[:limit]:
                    if items_yielded >= limit:
                        break

                    dataset_id = dataset.get("id", f"dataset_{items_yielded}")

                    metadata = {
                        "id": dataset_id,
                        "name": dataset_id,
                        "description": dataset.get("description", ""),
                        "author": dataset.get("author", ""),
                        "downloads": dataset.get("downloads", 0),
                        "likes": dataset.get("likes", 0),
                        "tags": dataset.get("tags", []),
                        "source": f"https://huggingface.co/datasets/{dataset_id}",
                    }

                    json_content = json.dumps(metadata, indent=2).encode("utf-8")

                    yield ScrapedItem(
                        content=json_content,
                        identifier=f"{dataset_id.replace('/', '_')}.json",
                        metadata=metadata,
                        content_type="application/json",
                        source_url=metadata["source"],
                    )
                    items_yielded += 1

            except Exception as e:
                self.logger.error(
                    f"HuggingFace search failed: {e}",
                    module=self.MODULE_NAME,
                )

    def get_hash(self, item: ScrapedItem) -> str:
        """Generate hash from metadata signature for duplicate detection."""
        # Use metadata signature for deduplication
        signature_fields = {
            "id": item.metadata.get("id"),
            "name": item.metadata.get("name"),
            "source": item.metadata.get("source"),
        }
        return HashManager.metadata_signature(signature_fields)

    def get_filename(self, item: ScrapedItem) -> str:
        """Generate filename from dataset ID."""
        return self.validator.sanitize_filename(item.identifier)
