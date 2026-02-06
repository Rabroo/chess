#!/usr/bin/env python3
"""
Lichess Database Downloader

Downloads monthly PGN files from the Lichess database.
These contain ALL games played on Lichess - hundreds of millions of games.

Usage:
    python3 download_lichess_db.py --months 2024-01,2024-02
    python3 download_lichess_db.py --year 2024
    python3 download_lichess_db.py --latest 3
    python3 download_lichess_db.py --list

Database info: https://database.lichess.org/
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import requests


# Lichess database base URL
LICHESS_DB_BASE = "https://database.lichess.org/standard"

# Output directory
DEFAULT_OUTPUT_DIR = "./raw/lichess_db"


def get_available_months() -> List[str]:
    """
    Get list of available months from Lichess database.

    Returns:
        List of available months in YYYY-MM format.
    """
    # Lichess database started in January 2013
    start_year = 2013
    start_month = 1

    now = datetime.now()
    # Database is usually 1-2 months behind
    end_year = now.year
    end_month = now.month - 1
    if end_month < 1:
        end_month = 12
        end_year -= 1

    months = []
    year = start_year
    month = start_month

    while year < end_year or (year == end_year and month <= end_month):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month > 12:
            month = 1
            year += 1

    return months


def get_file_url(month: str, variant: str = "standard", rated: bool = True) -> str:
    """
    Get the download URL for a specific month.

    Args:
        month: Month in YYYY-MM format.
        variant: Game variant (standard, chess960, etc.).
        rated: Whether to get rated or all games.

    Returns:
        Download URL.
    """
    rated_str = "rated" if rated else "all"
    # Format: lichess_db_standard_rated_2024-01.pgn.zst
    filename = f"lichess_db_{variant}_{rated_str}_{month}.pgn.zst"
    return f"https://database.lichess.org/{variant}/{filename}"


def get_file_size(url: str) -> Optional[int]:
    """Get file size from URL headers."""
    try:
        response = requests.head(url, timeout=10)
        if response.status_code == 200:
            return int(response.headers.get("Content-Length", 0))
    except Exception:
        pass
    return None


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable size."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def download_file(url: str, output_path: Path, decompress: bool = True) -> bool:
    """
    Download a file using curl (more reliable for large files).

    Args:
        url: URL to download.
        output_path: Path to save the file.
        decompress: Whether to decompress .zst files.

    Returns:
        True if successful.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    zst_path = output_path.with_suffix(".pgn.zst")
    pgn_path = output_path.with_suffix(".pgn")

    # Check if already downloaded
    if pgn_path.exists():
        print(f"  Already exists: {pgn_path}")
        return True

    if zst_path.exists() and not decompress:
        print(f"  Already exists: {zst_path}")
        return True

    # Download with curl (better progress and resume support)
    print(f"  Downloading: {url}")

    try:
        # Use curl with progress bar and resume support
        result = subprocess.run(
            ["curl", "-L", "-C", "-", "-o", str(zst_path), url],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"  Download failed: {e}")
        return False
    except FileNotFoundError:
        # curl not found, try requests
        print("  curl not found, using requests (slower)...")
        try:
            response = requests.get(url, stream=True, timeout=3600)
            response.raise_for_status()

            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0

            with open(zst_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192 * 1024):  # 8MB chunks
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        pct = (downloaded / total_size) * 100
                        print(f"\r  Progress: {pct:.1f}% ({format_size(downloaded)}/{format_size(total_size)})", end="")
            print()
        except Exception as e:
            print(f"  Download failed: {e}")
            return False

    # Decompress if requested
    if decompress:
        print(f"  Decompressing: {zst_path}")
        try:
            # Try zstd command
            result = subprocess.run(
                ["zstd", "-d", "--rm", str(zst_path)],
                check=True,
                capture_output=True
            )
            print(f"  Decompressed to: {pgn_path}")
        except subprocess.CalledProcessError as e:
            print(f"  Decompression failed: {e}")
            print("  Install zstd: brew install zstd (macOS) or apt install zstd (Linux)")
            return False
        except FileNotFoundError:
            print("  zstd not found. Install with: brew install zstd (macOS) or apt install zstd (Linux)")
            print(f"  Keeping compressed file: {zst_path}")
            return True

    return True


def list_available(show_sizes: bool = False):
    """List all available months with optional file sizes."""
    months = get_available_months()

    print(f"\nAvailable Lichess database months: {len(months)}")
    print("=" * 60)

    if show_sizes:
        print("Fetching file sizes (this may take a moment)...")
        print()

    for month in months:
        url = get_file_url(month)

        if show_sizes:
            size = get_file_size(url)
            size_str = format_size(size) if size else "Unknown"
            print(f"  {month}  ({size_str})")
        else:
            print(f"  {month}")

    print()
    print("Recent months are larger (more games played).")
    print("2024 months are typically 15-20GB compressed each.")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download Lichess PGN database files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 download_lichess_db.py --list
    python3 download_lichess_db.py --months 2024-01
    python3 download_lichess_db.py --months 2024-01,2024-02,2024-03
    python3 download_lichess_db.py --year 2023
    python3 download_lichess_db.py --latest 3
    python3 download_lichess_db.py --latest 1 --no-decompress

Note: Files are large! A single month can be 15-20GB compressed.
      Make sure you have enough disk space.
        """
    )

    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available months"
    )
    parser.add_argument(
        "--list-sizes",
        action="store_true",
        help="List available months with file sizes"
    )
    parser.add_argument(
        "--months", "-m",
        type=str,
        help="Comma-separated months to download (e.g., 2024-01,2024-02)"
    )
    parser.add_argument(
        "--year", "-y",
        type=int,
        help="Download all months from a specific year"
    )
    parser.add_argument(
        "--latest", "-n",
        type=int,
        help="Download the N most recent months"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--no-decompress",
        action="store_true",
        help="Keep files compressed (.zst)"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="standard",
        choices=["standard", "chess960", "antichess", "atomic", "crazyhouse",
                 "horde", "kingOfTheHill", "racingKings", "threeCheck"],
        help="Chess variant (default: standard)"
    )

    args = parser.parse_args()

    # List mode
    if args.list:
        list_available(show_sizes=False)
        return 0

    if args.list_sizes:
        list_available(show_sizes=True)
        return 0

    # Determine which months to download
    available = get_available_months()
    to_download = []

    if args.months:
        to_download = [m.strip() for m in args.months.split(",")]
        # Validate
        for m in to_download:
            if m not in available:
                print(f"Warning: {m} may not be available")

    elif args.year:
        to_download = [m for m in available if m.startswith(str(args.year))]
        if not to_download:
            print(f"No months available for year {args.year}")
            return 1

    elif args.latest:
        to_download = available[-args.latest:]

    else:
        parser.print_help()
        print("\nError: Specify --months, --year, --latest, or --list")
        return 1

    # Download
    output_dir = Path(args.output)
    print(f"\nDownloading {len(to_download)} month(s) to {output_dir}")
    print("=" * 60)

    success = 0
    failed = 0

    for month in to_download:
        print(f"\n[{month}]")
        url = get_file_url(month, variant=args.variant)
        output_path = output_dir / f"lichess_{args.variant}_{month}"

        if download_file(url, output_path, decompress=not args.no_decompress):
            success += 1
        else:
            failed += 1

    print()
    print("=" * 60)
    print(f"Complete: {success} succeeded, {failed} failed")

    if success > 0:
        print(f"\nPGN files saved to: {output_dir}")
        print("\nNext steps:")
        print("  1. Extract positions: python3 extract_positions.py --input ./raw/lichess_db/*.pgn")
        print("  2. Score with Stockfish: python3 score_positions.py --depth 20")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
