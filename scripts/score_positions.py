#!/usr/bin/env python3
"""
Score existing chess positions with Stockfish.

This script analyzes FEN positions that were previously scraped
and creates score files with evaluation, best move, and analysis depth.

Usage:
    python score_positions.py --input ./raw/chess
    python score_positions.py --input ./raw/chess --depth 40
    python score_positions.py --input ./my_positions --threads 8
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

# Add the scraper package to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from stockfish import Stockfish
    STOCKFISH_AVAILABLE = True
except ImportError:
    STOCKFISH_AVAILABLE = False


def find_stockfish_path():
    """Find Stockfish executable on the system."""
    common_paths = [
        "/opt/homebrew/bin/stockfish",  # macOS ARM (Homebrew)
        "/usr/local/bin/stockfish",      # macOS Intel (Homebrew)
        "/usr/bin/stockfish",            # Linux
        "/usr/games/stockfish",          # Linux alternative
    ]

    # Check if stockfish is in PATH
    stockfish_in_path = shutil.which("stockfish")
    if stockfish_in_path:
        return stockfish_in_path

    # Check common paths
    for path in common_paths:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    return None


def format_score(evaluation):
    """Format Stockfish evaluation as human-readable string."""
    if evaluation["type"] == "mate":
        mate_in = evaluation["value"]
        return f"M{mate_in}"  # M3 or M-3
    else:
        # Centipawns to pawns
        cp = evaluation["value"]
        pawns = cp / 100.0
        return f"{pawns:+.2f}"


def format_time(seconds):
    """Format seconds as human-readable time string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = seconds // 60
        secs = seconds % 60
        return f"{mins:.0f}m {secs:.0f}s"
    else:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        return f"{hours:.0f}h {mins:.0f}m"


class ProgressBar:
    """Terminal progress bar with ETA and speed."""

    def __init__(self, total, width=30):
        self.total = total
        self.width = width
        self.current = 0
        self.start_time = time.time()
        self.last_score = ""

    def update(self, current, score_str="", best_move=""):
        """Update progress bar."""
        self.current = current
        self.last_score = score_str

        # Calculate progress
        progress = current / self.total if self.total > 0 else 0
        filled = int(self.width * progress)
        bar = "=" * filled + ">" + " " * (self.width - filled - 1)
        percent = progress * 100

        # Calculate timing
        elapsed = time.time() - self.start_time
        if current > 0:
            rate = current / elapsed  # positions per second
            eta = (self.total - current) / rate if rate > 0 else 0
        else:
            rate = 0
            eta = 0

        # Build progress line
        line = f"\r[{bar}] {percent:5.1f}% | {current}/{self.total} | "
        line += f"{format_time(elapsed)} elapsed | ETA: {format_time(eta)} | "
        line += f"{rate:.2f} pos/s"

        # Add last score if available
        if score_str:
            line += f" | Last: {score_str}"

        # Pad to clear previous longer lines
        line = line.ljust(120)

        sys.stdout.write(line)
        sys.stdout.flush()

    def finish(self):
        """Complete the progress bar."""
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        print(f"\nCompleted {self.current} positions in {format_time(elapsed)} ({rate:.2f} pos/s)")


def main():
    parser = argparse.ArgumentParser(
        description="Score chess positions with Stockfish engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python score_positions.py --input ./raw/chess
  python score_positions.py --input ./raw/chess --depth 40
  python score_positions.py --input ./my_positions --threads 8 --hash 4096
        """,
    )

    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Directory containing position_*.txt files",
    )
    parser.add_argument(
        "--depth", "-d",
        type=int,
        default=30,
        help="Stockfish analysis depth (default: 30)",
    )
    parser.add_argument(
        "--threads", "-t",
        type=int,
        default=None,
        help="Number of CPU threads (default: auto-detect)",
    )
    parser.add_argument(
        "--hash",
        type=int,
        default=2048,
        help="Hash table size in MB (default: 2048)",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-score positions that already have scores",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Validate input directory
    if not args.input.exists():
        print(f"Error: Directory not found: {args.input}", file=sys.stderr)
        return 1

    if not args.input.is_dir():
        print(f"Error: Not a directory: {args.input}", file=sys.stderr)
        return 1

    # Check Stockfish availability
    if not STOCKFISH_AVAILABLE:
        print("Error: Stockfish Python package not installed.", file=sys.stderr)
        print("Install with: pip install stockfish", file=sys.stderr)
        return 1

    stockfish_path = find_stockfish_path()
    if not stockfish_path:
        print("Error: Stockfish binary not found.", file=sys.stderr)
        print("Install with: brew install stockfish (macOS) or apt install stockfish (Linux)", file=sys.stderr)
        return 1

    # Find position files
    position_files = sorted(args.input.glob("position_*.txt"))
    position_files = [f for f in position_files if "_score" not in f.name]

    if not position_files:
        print(f"No position files found in {args.input}", file=sys.stderr)
        return 1

    # Filter out already scored (unless --force)
    if not args.force:
        to_score = []
        for pos_file in position_files:
            score_file = pos_file.parent / pos_file.name.replace(".txt", "_score.txt")
            if not score_file.exists():
                to_score.append(pos_file)
        already_scored = len(position_files) - len(to_score)
    else:
        to_score = position_files
        already_scored = 0

    if not to_score:
        print(f"All {len(position_files)} positions already have scores.")
        print("Use --force to re-score them.")
        return 0

    # Initialize Stockfish
    threads = args.threads or os.cpu_count() or 4

    if not args.quiet:
        print("Chess Position Scorer")
        print("=" * 40)
        print(f"Input directory: {args.input}")
        print(f"Positions found: {len(position_files)}")
        if already_scored > 0:
            print(f"Already scored: {already_scored} (skipping)")
        print(f"To score: {len(to_score)}")
        print()
        print(f"Stockfish path: {stockfish_path}")
        print(f"Analysis depth: {args.depth}")
        print(f"Threads: {threads}")
        print(f"Hash: {args.hash} MB")
        print()

    try:
        engine = Stockfish(
            path=stockfish_path,
            depth=args.depth,
            parameters={
                "Threads": threads,
                "Hash": args.hash,
            }
        )
    except Exception as e:
        print(f"Error initializing Stockfish: {e}", file=sys.stderr)
        return 1

    if not args.quiet:
        print("Starting analysis...")
        print()

    # Score positions
    scored = 0
    errors = 0

    # Initialize progress bar
    progress = ProgressBar(len(to_score)) if not args.quiet else None

    interrupted = False
    try:
        for i, pos_file in enumerate(to_score, 1):
            try:
                # Read FEN
                fen = pos_file.read_text().strip()

                # Analyze
                engine.set_fen_position(fen)
                evaluation = engine.get_evaluation()
                best_move = engine.get_best_move()

                # Write score file
                score_file = pos_file.parent / pos_file.name.replace(".txt", "_score.txt")
                score_str = format_score(evaluation)
                score_content = f"{score_str}\n{best_move}\n{args.depth}"
                score_file.write_text(score_content)

                scored += 1

                if progress:
                    progress.update(i, score_str, best_move)

            except Exception as e:
                errors += 1
                if progress:
                    progress.update(i, f"ERROR: {e}")

    except KeyboardInterrupt:
        interrupted = True
        print("\n\nInterrupted by user.")

    # Clean up Stockfish gracefully
    try:
        engine._stockfish.kill()
    except:
        pass

    if progress and not interrupted:
        progress.finish()

    if not args.quiet:
        print()
        print("=" * 40)
        print("Scoring complete!")
        print(f"  Positions scored: {scored}")
        if already_scored > 0:
            print(f"  Previously scored: {already_scored}")
        if errors > 0:
            print(f"  Errors: {errors}")
        print(f"  Output: {args.input}")

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
