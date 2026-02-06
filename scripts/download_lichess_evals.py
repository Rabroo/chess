#!/usr/bin/env python3
"""
Lichess Evaluation Database Downloader & Processor

Downloads pre-computed Stockfish evaluations from Lichess cloud analysis.
These are high-depth (30-40+) evaluations - no local Stockfish needed!

The eval database contains millions of positions analyzed by Lichess's
distributed cloud engine at very high depth.

Usage:
    python3 download_lichess_evals.py --limit 100000
    python3 download_lichess_evals.py --limit 1000000 --min-depth 30
    python3 download_lichess_evals.py --stream --limit 50000

Database info: https://database.lichess.org/
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator, Optional, Tuple

try:
    import requests
except ImportError:
    print("requests not installed. Run: pip3 install requests")
    sys.exit(1)


# Lichess eval database URL
LICHESS_EVAL_URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"

# Output directories
DEFAULT_OUTPUT_DIR = "./raw/chess"
DEFAULT_CACHE_DIR = "./raw/lichess_eval_cache"


class ProgressBar:
    """Simple progress bar for terminal output."""

    def __init__(self, total: int, width: int = 30, prefix: str = ""):
        self.total = total
        self.width = width
        self.prefix = prefix
        self.current = 0
        self.start_time = time.time()
        self.last_update = 0

    def update(self, current: int):
        self.current = current
        now = time.time()

        if now - self.last_update < 0.1 and current < self.total:
            return

        self.last_update = now
        elapsed = now - self.start_time

        if current > 0:
            rate = current / elapsed
            remaining = (self.total - current) / rate if rate > 0 else 0
        else:
            rate = 0
            remaining = 0

        pct = current / self.total if self.total > 0 else 0
        filled = int(self.width * pct)
        bar = "=" * filled + "-" * (self.width - filled)

        elapsed_str = self._format_time(elapsed)
        remaining_str = self._format_time(remaining)

        print(
            f"\r{self.prefix}[{bar}] {current:,}/{self.total:,} "
            f"({pct*100:.1f}%) | {rate:.0f}/s | "
            f"Elapsed: {elapsed_str} | ETA: {remaining_str}    ",
            end="",
            flush=True
        )

    def _format_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"

    def finish(self):
        self.update(self.total)
        print()


def download_eval_database(cache_dir: Path) -> Path:
    """
    Download the Lichess eval database.

    Returns:
        Path to the downloaded/decompressed file.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    zst_path = cache_dir / "lichess_db_eval.jsonl.zst"
    jsonl_path = cache_dir / "lichess_db_eval.jsonl"

    # Check if already decompressed
    if jsonl_path.exists():
        print(f"Using cached: {jsonl_path}")
        return jsonl_path

    # Check if already downloaded but not decompressed
    if not zst_path.exists():
        print(f"Downloading Lichess eval database...")
        print(f"URL: {LICHESS_EVAL_URL}")
        print(f"This file is ~2GB compressed, ~15GB decompressed")
        print()

        try:
            # Use curl for better progress and resume support
            subprocess.run(
                ["curl", "-L", "-C", "-", "-o", str(zst_path), LICHESS_EVAL_URL],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Download failed: {e}")
            sys.exit(1)
        except FileNotFoundError:
            print("curl not found, using requests (slower, no resume)...")
            response = requests.get(LICHESS_EVAL_URL, stream=True, timeout=3600)
            response.raise_for_status()

            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0

            with open(zst_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = (downloaded / total) * 100
                        print(f"\rDownloading: {pct:.1f}%", end="")
            print()

    # Decompress
    print(f"Decompressing {zst_path}...")
    try:
        subprocess.run(
            ["zstd", "-d", "-k", str(zst_path)],  # -k keeps original
            check=True
        )
        print(f"Decompressed to: {jsonl_path}")
    except subprocess.CalledProcessError as e:
        print(f"Decompression failed: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("zstd not found. Install with: brew install zstd")
        sys.exit(1)

    return jsonl_path


def stream_eval_database() -> Iterator[dict]:
    """
    Stream the eval database directly without downloading fully.
    Uses zstd to decompress on-the-fly.

    Yields:
        Parsed JSON objects from the database.
    """
    print("Streaming Lichess eval database (no full download)...")
    print("This requires: curl, zstd")
    print()

    # Pipe: curl -> zstd -d -> python
    try:
        process = subprocess.Popen(
            f'curl -sL "{LICHESS_EVAL_URL}" | zstd -d',
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

        process.wait()

    except Exception as e:
        print(f"Streaming failed: {e}")
        sys.exit(1)


def parse_eval_entry(entry: dict) -> Optional[Tuple[str, int, dict]]:
    """
    Parse a single eval database entry.

    Args:
        entry: JSON object from the eval database.

    Returns:
        Tuple of (fen, score_centipawns, metadata) or None if invalid.
    """
    fen = entry.get("fen")
    if not fen:
        return None

    evals = entry.get("evals", [])
    if not evals:
        return None

    # Get the best/deepest evaluation
    best_eval = None
    best_depth = 0

    for ev in evals:
        depth = ev.get("depth", 0)
        if depth > best_depth:
            best_depth = depth
            best_eval = ev

    if not best_eval:
        return None

    # Extract score
    pvs = best_eval.get("pvs", [])
    if not pvs:
        return None

    pv = pvs[0]  # Best principal variation

    # Score can be centipawns (cp) or mate (mate)
    if "cp" in pv:
        score_cp = pv["cp"]
    elif "mate" in pv:
        mate_in = pv["mate"]
        # Convert mate to large centipawn value
        if mate_in > 0:
            score_cp = 10000 - mate_in * 10
        else:
            score_cp = -10000 - mate_in * 10
    else:
        return None

    metadata = {
        "fen": fen,
        "depth": best_depth,
        "knodes": best_eval.get("knodes", 0),
        "pv": pv.get("line", ""),
        "mate": pv.get("mate"),
    }

    return fen, score_cp, metadata


def process_eval_database(
    source: Iterator[dict],
    output_dir: Path,
    limit: int,
    min_depth: int = 20,
    min_knodes: int = 0,
    skip_mates: bool = False,
    max_score: int = 0,
) -> int:
    """
    Process eval database entries and save positions.

    Args:
        source: Iterator of JSON entries.
        output_dir: Directory to save positions.
        limit: Maximum positions to extract.
        min_depth: Minimum analysis depth.
        min_knodes: Minimum kilonodes searched.
        skip_mates: Skip positions with mate scores.
        max_score: Maximum absolute score in centipawns (0 = no limit).

    Returns:
        Number of positions saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    processed = 0
    skipped_depth = 0
    skipped_mate = 0
    skipped_score = 0

    filter_info = f"limit: {limit:,}, min_depth: {min_depth}"
    if max_score > 0:
        filter_info += f", max_score: {max_score}cp"
    print(f"Extracting positions ({filter_info})...")
    progress = ProgressBar(limit, prefix="Progress: ")

    for entry in source:
        if saved >= limit:
            break

        processed += 1

        result = parse_eval_entry(entry)
        if not result:
            continue

        fen, score_cp, metadata = result

        # Filter by depth
        if metadata["depth"] < min_depth:
            skipped_depth += 1
            continue

        # Filter by knodes
        if min_knodes > 0 and metadata.get("knodes", 0) < min_knodes:
            continue

        # Filter mate scores if requested
        if skip_mates and metadata.get("mate") is not None:
            skipped_mate += 1
            continue

        # Filter by max score (for balanced positions)
        if max_score > 0 and abs(score_cp) > max_score:
            skipped_score += 1
            continue

        # Save position
        pos_path = output_dir / f"position_{saved:08d}.txt"
        with open(pos_path, "w") as f:
            f.write(fen)

        # Save score (in pawns, like score_positions.py format)
        score_path = output_dir / f"position_{saved:08d}_score.txt"
        score_pawns = score_cp / 100.0
        with open(score_path, "w") as f:
            f.write(f"{score_pawns:.2f}\n")
            f.write(f"depth={metadata['depth']}\n")
            if metadata.get("pv"):
                f.write(f"pv={metadata['pv']}\n")

        saved += 1
        progress.update(saved)

    progress.finish()

    print()
    print(f"Processed: {processed:,} entries")
    print(f"Saved: {saved:,} positions")
    print(f"Skipped (low depth): {skipped_depth:,}")
    if skip_mates:
        print(f"Skipped (mate scores): {skipped_mate:,}")

    return saved


def read_jsonl_file(jsonl_path: Path) -> Iterator[dict]:
    """Read entries from a JSONL file."""
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def main():
    parser = argparse.ArgumentParser(
        description="Download and process Lichess evaluation database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download and extract 100K positions (depth 20+)
    python3 download_lichess_evals.py --limit 100000

    # Extract 1M high-quality positions (depth 30+)
    python3 download_lichess_evals.py --limit 1000000 --min-depth 30

    # Stream without full download (saves disk space)
    python3 download_lichess_evals.py --stream --limit 50000

    # Skip mate positions (for stable training)
    python3 download_lichess_evals.py --limit 100000 --skip-mates

The output format is compatible with train_eval_model.py:
    position_00000000.txt       - FEN string
    position_00000000_score.txt - Score in pawns
        """
    )

    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=100000,
        help="Maximum positions to extract (default: 100000)"
    )
    parser.add_argument(
        "--min-depth", "-d",
        type=int,
        default=20,
        help="Minimum analysis depth (default: 20)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--cache", "-c",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help=f"Cache directory for downloaded files (default: {DEFAULT_CACHE_DIR})"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream database without full download (slower but saves disk)"
    )
    parser.add_argument(
        "--skip-mates",
        action="store_true",
        help="Skip positions with mate scores"
    )
    parser.add_argument(
        "--max-score",
        type=int,
        default=0,
        help="Max absolute score in centipawns (e.g., 500 for ±5 pawns, 0=no limit)"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear output directory before extracting"
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    cache_dir = Path(args.cache)

    # Clear output if requested
    if args.clear and output_dir.exists():
        print(f"Clearing output directory: {output_dir}")
        import shutil
        shutil.rmtree(output_dir)

    print("=" * 60)
    print("Lichess Evaluation Database Processor")
    print("=" * 60)
    print(f"Target positions: {args.limit:,}")
    print(f"Minimum depth: {args.min_depth}")
    print(f"Output: {output_dir}")
    print()

    start_time = time.time()

    # Get data source
    if args.stream:
        source = stream_eval_database()
    else:
        jsonl_path = download_eval_database(cache_dir)
        source = read_jsonl_file(jsonl_path)

    # Process and save
    saved = process_eval_database(
        source=source,
        output_dir=output_dir,
        limit=args.limit,
        min_depth=args.min_depth,
        skip_mates=args.skip_mates,
        max_score=args.max_score,
    )

    elapsed = time.time() - start_time

    print()
    print("=" * 60)
    print("Complete!")
    print(f"  Positions saved: {saved:,}")
    print(f"  Output directory: {output_dir}")
    print(f"  Time elapsed: {elapsed:.1f}s")
    print()
    print("Next step - train the model:")
    print(f"  python3 train_eval_model.py --data {output_dir} --epochs 50")

    return 0


if __name__ == "__main__":
    sys.exit(main())
