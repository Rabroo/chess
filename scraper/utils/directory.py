"""Directory management utilities for the Universal Data Scraper."""

import shutil
import tempfile
from pathlib import Path
from typing import Optional


class DirectoryManager:
    """Manages directory creation, validation, and disk space checking."""

    def __init__(self, base_path: Optional[Path] = None, min_disk_space_mb: int = 500):
        """
        Initialize the directory manager.

        Args:
            base_path: Base path for all scraper directories. Defaults to current directory.
            min_disk_space_mb: Minimum required disk space in MB.
        """
        self.base_path = base_path or Path.cwd()
        self.min_disk_space_mb = min_disk_space_mb
        self._temp_dir: Optional[Path] = None

        # Standard directories
        self.raw_dir = self.base_path / "raw"
        self.config_dir = self.base_path / "config"
        self.logs_dir = self.base_path / "logs"
        self.meta_dir = self.raw_dir / "meta"

        # Module directories under raw/
        self.module_dirs = {
            "images": self.raw_dir / "images",
            "chess_positions": self.raw_dir / "chess_positions",
            "numeric_relations": self.raw_dir / "numeric_relations",
            "experiments": self.raw_dir / "experiments",
            "social": self.raw_dir / "social",
        }

    def initialize(self) -> None:
        """Create all required directories if they don't exist."""
        # Create base directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

        # Create module directories
        for module_dir in self.module_dirs.values():
            module_dir.mkdir(parents=True, exist_ok=True)

    def get_module_dir(self, module_name: str) -> Path:
        """
        Get or create the directory for a specific module.

        Args:
            module_name: Name of the scraper module.

        Returns:
            Path to the module's output directory.
        """
        if module_name in self.module_dirs:
            return self.module_dirs[module_name]

        # Create new module directory dynamically
        new_dir = self.raw_dir / module_name
        new_dir.mkdir(parents=True, exist_ok=True)
        self.module_dirs[module_name] = new_dir
        return new_dir

    def check_disk_space(self) -> bool:
        """
        Check if there's sufficient disk space available.

        Returns:
            True if disk space is sufficient, False otherwise.
        """
        try:
            usage = shutil.disk_usage(self.base_path)
            available_mb = usage.free / (1024 * 1024)
            return available_mb >= self.min_disk_space_mb
        except OSError:
            return False

    def get_available_space_mb(self) -> float:
        """
        Get available disk space in MB.

        Returns:
            Available disk space in megabytes.
        """
        try:
            usage = shutil.disk_usage(self.base_path)
            return usage.free / (1024 * 1024)
        except OSError:
            return 0.0

    def get_temp_dir(self) -> Path:
        """
        Get or create a temporary directory for sandboxed downloads.

        Returns:
            Path to the temporary directory.
        """
        if self._temp_dir is None or not self._temp_dir.exists():
            self._temp_dir = Path(tempfile.mkdtemp(prefix="scraper_"))
        return self._temp_dir

    def cleanup_temp_dir(self) -> None:
        """Remove the temporary directory and its contents."""
        if self._temp_dir and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

    def move_from_temp(self, temp_file: Path, destination: Path) -> Path:
        """
        Move a file from temp directory to its final destination.

        Args:
            temp_file: Path to the file in temp directory.
            destination: Final destination path.

        Returns:
            Path to the moved file.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(temp_file), str(destination))
        return destination

    def get_log_path(self) -> Path:
        """Get the default log file path."""
        return self.logs_dir / "scraper.log"
