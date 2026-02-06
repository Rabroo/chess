"""Numeric relations scraper module."""

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from ..utils.downloader import Downloader, DownloadError
from ..utils.hashing import HashManager
from ..utils.validator import ValidationError
from .base import ScrapedItem, ScraperModule, ScraperRegistry


# JSON schema for numeric relations
NUMERIC_RELATION_SCHEMA = {
    "required_fields": {"id", "values"},
    "optional_fields": {"name", "description", "formula", "source", "type"},
}


@ScraperRegistry.register("numeric")
class NumericRelationScraper(ScraperModule):
    """Scraper for collecting numeric sequences and relationships."""

    MODULE_NAME = "numeric_relations"
    ALLOWED_CONTENT_TYPES = {"application/json", "text/plain"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.downloader = Downloader(
            timeout=self.config.network.timeout,
            retries=self.config.network.retries,
            retry_backoff=self.config.network.retry_backoff,
            user_agent=self.config.network.user_agent,
        )

    def validate_input(self, input_value: str) -> bool:
        """
        Validate input - can be sequence ID (e.g., A000045), JSON, URL, or file.

        Args:
            input_value: Sequence identifier, URL, or file path.

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

        # OEIS-style sequence ID (e.g., A000045 for Fibonacci)
        if re.match(r"^A\d{6}$", input_value.upper()):
            return True

        # JSON string
        if input_value.startswith("{") or input_value.startswith("["):
            try:
                json.loads(input_value)
                return True
            except json.JSONDecodeError:
                raise ValidationError("Invalid JSON format")

        # Comma-separated numbers (sequence)
        try:
            values = [float(x.strip()) for x in input_value.split(",")]
            if len(values) < 2:
                raise ValidationError("Sequence must have at least 2 values")
            return True
        except ValueError:
            raise ValidationError(
                "Invalid input format. Expected: sequence ID, URL, file path, "
                "JSON, or comma-separated numbers"
            )

    def _normalize_sequence(self, values: List[Any]) -> List[float]:
        """Normalize a sequence of values to floats."""
        return [float(v) for v in values]

    def _validate_sequence_data(self, data: Dict[str, Any]) -> bool:
        """Validate sequence data against schema."""
        required = NUMERIC_RELATION_SCHEMA["required_fields"]
        missing = required - set(data.keys())
        if missing:
            raise ValidationError(f"Missing required fields: {missing}")
        return True

    def _parse_oeis_id(self, oeis_id: str) -> Dict[str, Any]:
        """
        Fetch and parse OEIS sequence data.

        This is a placeholder - actual implementation would fetch from OEIS API.
        """
        # Normalize ID
        oeis_id = oeis_id.upper()

        # In a real implementation, this would fetch from OEIS
        # For now, return a placeholder
        return {
            "id": oeis_id,
            "name": f"OEIS Sequence {oeis_id}",
            "values": [],  # Would be populated from OEIS
            "source": f"https://oeis.org/{oeis_id}",
            "type": "integer_sequence",
        }

    def fetch(self, input_value: str, limit: int) -> Iterator[ScrapedItem]:
        """
        Fetch numeric relations.

        Args:
            input_value: Sequence ID, URL, file path, JSON, or values.
            limit: Maximum items to fetch.

        Yields:
            ScrapedItem for each relation.
        """
        relations = []

        # URL
        if input_value.startswith("http"):
            try:
                content, _ = self.downloader.download(input_value)
                text = content.decode("utf-8")

                # Try to parse as JSON
                try:
                    data = json.loads(text)
                    if isinstance(data, list):
                        relations = data[:limit]
                    else:
                        relations = [data]
                except json.JSONDecodeError:
                    # Try to parse as plain text sequences
                    for line in text.split("\n"):
                        if line.strip():
                            try:
                                values = [float(x) for x in line.strip().split(",")]
                                relations.append({
                                    "id": f"seq_{len(relations)}",
                                    "values": values,
                                    "source": input_value,
                                })
                            except ValueError:
                                continue
                            if len(relations) >= limit:
                                break
            except DownloadError as e:
                self.logger.error(f"Failed to download: {e}", module=self.MODULE_NAME)
                return

        # File path
        elif input_value.startswith("@"):
            file_path = Path(input_value[1:])
            with open(file_path, "r") as f:
                content = f.read()

            try:
                data = json.loads(content)
                if isinstance(data, list):
                    relations = data[:limit]
                else:
                    relations = [data]
            except json.JSONDecodeError:
                for line in content.split("\n"):
                    if line.strip():
                        try:
                            values = [float(x) for x in line.strip().split(",")]
                            relations.append({
                                "id": f"seq_{len(relations)}",
                                "values": values,
                                "source": str(file_path),
                            })
                        except ValueError:
                            continue
                        if len(relations) >= limit:
                            break

        # OEIS sequence ID
        elif re.match(r"^A\d{6}$", input_value.upper()):
            relations = [self._parse_oeis_id(input_value)]

        # JSON string
        elif input_value.startswith("{") or input_value.startswith("["):
            data = json.loads(input_value)
            if isinstance(data, list):
                relations = data[:limit]
            else:
                relations = [data]

        # Comma-separated values
        else:
            values = [float(x.strip()) for x in input_value.split(",")]
            relations = [{
                "id": "direct_input",
                "values": values,
                "source": "direct_input",
            }]

        # Yield relations as ScrapedItems
        for i, relation in enumerate(relations):
            if i >= limit:
                break

            # Ensure required fields
            if "id" not in relation:
                relation["id"] = f"relation_{i:06d}"
            if "values" not in relation:
                continue

            try:
                self._validate_sequence_data(relation)
            except ValidationError as e:
                self.logger.warning(f"Skipping invalid relation: {e}", module=self.MODULE_NAME)
                continue

            # Serialize to JSON
            json_content = json.dumps(relation, indent=2).encode("utf-8")

            yield ScrapedItem(
                content=json_content,
                identifier=f"{relation['id']}.json",
                metadata=relation,
                content_type="application/json",
            )

    def get_hash(self, item: ScrapedItem) -> str:
        """Generate hash from normalized sequence representation."""
        values = item.metadata.get("values", [])
        normalized = self._normalize_sequence(values)
        return HashManager.normalized_representation_hash(normalized)

    def get_filename(self, item: ScrapedItem) -> str:
        """Generate filename from relation ID."""
        relation_id = item.metadata.get("id", "unknown")
        return self.validator.sanitize_filename(f"{relation_id}.json")
