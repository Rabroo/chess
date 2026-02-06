#!/usr/bin/env python3
"""
Extract Chess Positions from PGN Files

Extracts random FEN positions from PGN files (e.g., Lichess database dumps).
Much faster than API fetching - can process millions of games.

Usage:
    python3 extract_positions.py --input ./raw/lichess_db/*.pgn --limit 100000
    python3 extract_positions.py --input game.pgn --limit 1000 --min-ply 20
    python3 extract_positions.py --input ./raw/lichess_db/ --limit 50000 --output ./positions
"""

import argparse
import glob
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterator, List, Optional, Set, Tuple

import chess
import chess.pgn


# Default output directory
DEFAULT_OUTPUT_DIR = "./raw/chess"


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

        # Only update every 0.1 seconds to avoid flickering
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

        # Format times
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


def count_games_in_file(pgn_path: Path) -> int:
    """Quickly estimate game count by counting [Event lines."""
    count = 0
    with open(pgn_path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("[Event "):
                count += 1
    return count


def extract_positions_from_game(
    game: chess.pgn.Game,
    positions_per_game: int = 2,
    min_ply: int = 10,
    max_ply: int = 80,
    skip_captures: bool = False,
) -> List[Tuple[str, dict]]:
    """
    Extract random positions from a single game.

    Args:
        game: Parsed PGN game.
        positions_per_game: Number of positions to extract per game.
        min_ply: Minimum ply (half-move) to start sampling from.
        max_ply: Maximum ply to sample up to.
        skip_captures: Skip positions after captures (more stable positions).

    Returns:
        List of (fen, metadata) tuples.
    """
    positions = []
    board = game.board()
    ply = 0
    candidate_positions = []

    # Collect candidate positions
    for move in game.mainline_moves():
        board.push(move)
        ply += 1

        if ply < min_ply:
            continue
        if ply > max_ply:
            break

        # Skip positions after captures if requested
        if skip_captures and board.is_capture(move):
            continue

        # Skip positions in check (usually tactical, not positional)
        if board.is_check():
            continue

        candidate_positions.append((ply, board.fen()))

    # Randomly sample positions
    if len(candidate_positions) <= positions_per_game:
        selected = candidate_positions
    else:
        selected = random.sample(candidate_positions, positions_per_game)

    # Extract metadata from game headers
    headers = dict(game.headers)

    for ply, fen in selected:
        metadata = {
            "fen": fen,
            "ply": ply,
            "white": headers.get("White", ""),
            "black": headers.get("Black", ""),
            "result": headers.get("Result", ""),
            "event": headers.get("Event", ""),
            "white_elo": headers.get("WhiteElo", ""),
            "black_elo": headers.get("BlackElo", ""),
            "time_control": headers.get("TimeControl", ""),
            "eco": headers.get("ECO", ""),
            "opening": headers.get("Opening", ""),
        }
        positions.append((fen, metadata))

    return positions


def process_pgn_file(
    pgn_path: Path,
    limit: int,
    positions_per_game: int = 2,
    min_ply: int = 10,
    max_ply: int = 80,
    min_elo: int = 0,
    seen_positions: Optional[Set[str]] = None,
    show_progress: bool = True,
) -> Iterator[Tuple[str, dict]]:
    """
    Process a PGN file and yield positions.

    Args:
        pgn_path: Path to PGN file.
        limit: Maximum positions to extract.
        positions_per_game: Positions per game.
        min_ply: Minimum ply to sample from.
        max_ply: Maximum ply to sample from.
        min_elo: Minimum Elo rating for both players.
        seen_positions: Set of already seen position hashes for deduplication.
        show_progress: Show progress bar.

    Yields:
        (fen, metadata) tuples.
    """
    if seen_positions is None:
        seen_positions = set()

    # Estimate total games for progress bar
    if show_progress:
        print(f"Counting games in {pgn_path.name}...")
        total_games = count_games_in_file(pgn_path)
        print(f"Found ~{total_games:,} games")
        progress = ProgressBar(total_games, prefix="Processing: ")
    else:
        total_games = 0

    positions_extracted = 0
    games_processed = 0

    with open(pgn_path, "r", errors="ignore") as pgn_file:
        while positions_extracted < limit:
            try:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                games_processed += 1

                if show_progress and games_processed % 1000 == 0:
                    progress.update(games_processed)

                # Filter by Elo if specified
                if min_elo > 0:
                    try:
                        white_elo = int(game.headers.get("WhiteElo", "0") or "0")
                        black_elo = int(game.headers.get("BlackElo", "0") or "0")
                        if white_elo < min_elo or black_elo < min_elo:
                            continue
                    except ValueError:
                        continue

                # Extract positions from game
                game_positions = extract_positions_from_game(
                    game,
                    positions_per_game=positions_per_game,
                    min_ply=min_ply,
                    max_ply=max_ply,
                )

                for fen, metadata in game_positions:
                    if positions_extracted >= limit:
                        break

                    # Deduplicate using FEN hash
                    fen_hash = hashlib.sha256(fen.encode()).hexdigest()[:16]
                    if fen_hash in seen_positions:
                        continue

                    seen_positions.add(fen_hash)
                    positions_extracted += 1
                    yield fen, metadata

            except Exception as e:
                # Skip malformed games
                continue

    if show_progress:
        progress.finish()

    print(f"Extracted {positions_extracted:,} unique positions from {games_processed:,} games")


def save_position(
    fen: str,
    metadata: dict,
    output_dir: Path,
    index: int,
) -> Path:
    """Save a position to disk."""
    # Save FEN
    fen_path = output_dir / f"position_{index:08d}.txt"
    with open(fen_path, "w") as f:
        f.write(fen)

    # Save metadata
    meta_path = output_dir / f"position_{index:08d}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return fen_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract chess positions from PGN files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 extract_positions.py --input ./raw/lichess_db/lichess_standard_2024-01.pgn --limit 100000
    python3 extract_positions.py --input ./raw/lichess_db/*.pgn --limit 1000000
    python3 extract_positions.py --input game.pgn --limit 1000 --min-elo 2000

Note: For large datasets, use --limit to control output size.
      100,000 positions is a good starting point for training.
        """
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input PGN file(s) or directory (supports wildcards)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=10000,
        help="Maximum positions to extract (default: 10000)"
    )
    parser.add_argument(
        "--positions-per-game", "-p",
        type=int,
        default=2,
        help="Positions to sample per game (default: 2)"
    )
    parser.add_argument(
        "--min-ply",
        type=int,
        default=10,
        help="Minimum ply/half-move to start sampling (default: 10)"
    )
    parser.add_argument(
        "--max-ply",
        type=int,
        default=80,
        help="Maximum ply to sample up to (default: 80)"
    )
    parser.add_argument(
        "--min-elo",
        type=int,
        default=0,
        help="Minimum Elo for both players (default: 0, no filter)"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear output directory before extracting"
    )

    args = parser.parse_args()

    # Find input files
    input_path = args.input
    if "*" in input_path:
        pgn_files = sorted(glob.glob(input_path))
    elif os.path.isdir(input_path):
        pgn_files = sorted(glob.glob(os.path.join(input_path, "*.pgn")))
    else:
        pgn_files = [input_path]

    if not pgn_files:
        print(f"No PGN files found: {input_path}")
        return 1

    print(f"Found {len(pgn_files)} PGN file(s)")
    for f in pgn_files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  - {f} ({size_mb:.1f} MB)")

    # Setup output directory
    output_dir = Path(args.output)
    if args.clear and output_dir.exists():
        print(f"\nClearing output directory: {output_dir}")
        import shutil
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Track seen positions for deduplication across files
    seen_positions: Set[str] = set()
    total_extracted = 0
    position_index = 0

    print(f"\nExtracting up to {args.limit:,} positions...")
    print(f"Settings: {args.positions_per_game} per game, ply {args.min_ply}-{args.max_ply}")
    if args.min_elo > 0:
        print(f"Filtering: minimum Elo {args.min_elo}")
    print()

    start_time = time.time()

    for pgn_file in pgn_files:
        if total_extracted >= args.limit:
            break

        remaining = args.limit - total_extracted
        print(f"\nProcessing: {pgn_file}")

        for fen, metadata in process_pgn_file(
            Path(pgn_file),
            limit=remaining,
            positions_per_game=args.positions_per_game,
            min_ply=args.min_ply,
            max_ply=args.max_ply,
            min_elo=args.min_elo,
            seen_positions=seen_positions,
        ):
            save_position(fen, metadata, output_dir, position_index)
            position_index += 1
            total_extracted += 1

    elapsed = time.time() - start_time

    print()
    print("=" * 60)
    print(f"Extraction complete!")
    print(f"  Total positions: {total_extracted:,}")
    print(f"  Output directory: {output_dir}")
    print(f"  Time elapsed: {elapsed:.1f}s")
    print(f"  Rate: {total_extracted/elapsed:.0f} positions/second")
    print()
    print("Next step - score with Stockfish:")
    print(f"  python3 score_positions.py --input {output_dir} --depth 20")

    return 0


if __name__ == "__main__":
    sys.exit(main())
