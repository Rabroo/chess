#!/usr/bin/env python3
"""
Scrape chess positions from Lichess, optionally with Stockfish scoring.

This script collects FEN positions from real Lichess games. By default,
no engine analysis is performed. Use --depth to enable Stockfish scoring.

Usage:
    python scrape_positions.py --limit 1000
    python scrape_positions.py --limit 500 --game-type blitz
    python scrape_positions.py --limit 200 --depth 40
    python scrape_positions.py --limit 200 --output-dir ./my_positions
"""

import argparse
import sys
from pathlib import Path

# Add the scraper package to path
sys.path.insert(0, str(Path(__file__).parent))

from scraper.scrapers.chess_positions import ChessPositionScraper
from scraper.utils.config import load_config
from scraper.utils.directory import DirectoryManager
from scraper.utils.logger import init_logger


def main():
    parser = argparse.ArgumentParser(
        description="Scrape chess positions from Lichess (without Stockfish scoring)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scrape_positions.py --limit 1000
  python3 scrape_positions.py --limit 500 --game-type blitz
  python3 scrape_positions.py --limit 200 --depth 40          # scrape + score
  python3 scrape_positions.py --limit 200 --output-dir ./my_positions
        """,
    )

    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=100,
        help="Number of positions to collect (default: 100)",
    )
    parser.add_argument(
        "--game-type", "-g",
        choices=["bullet", "blitz", "rapid", "classical", "ultrabullet"],
        default=None,
        help="Filter by game type (default: all types)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Output directory (default: ./raw/chess)",
    )
    parser.add_argument(
        "--depth", "-d",
        type=int,
        default=None,
        help="Stockfish analysis depth (enables scoring when provided)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Setup
    base_path = args.output_dir or Path.cwd()
    config = load_config(None, {})

    dir_manager = DirectoryManager(
        base_path=base_path,
        min_disk_space_mb=config.storage.disk_space_min_mb,
    )
    dir_manager.initialize()

    log_path = dir_manager.get_log_path()
    logger = init_logger(log_path=log_path)

    scoring_enabled = args.depth is not None

    if not args.quiet:
        title = "Chess Position Scraper" + (" + Scoring" if scoring_enabled else " (No Scoring)")
        print(title)
        print("=" * 40)
        print(f"Positions to collect: {args.limit}")
        if args.game_type:
            print(f"Game type filter: {args.game_type}")
        if scoring_enabled:
            print(f"Stockfish depth: {args.depth}")
        print(f"Output directory: {dir_manager.get_module_dir('chess')}")
        print()

    # Create scraper
    scraper = ChessPositionScraper(
        config=config,
        logger=logger,
        directory_manager=dir_manager,
    )

    # Configure scoring
    if scoring_enabled:
        scraper.skip_scoring = False
        scraper.STOCKFISH_DEPTH = args.depth
        # Reinitialize Stockfish with new depth
        scraper._init_stockfish()
    else:
        scraper.skip_scoring = True

    # Run scraper
    input_value = args.game_type or "random"

    if not args.quiet:
        print("Starting position collection...")
        if scoring_enabled:
            print(f"(Stockfish scoring ENABLED at depth {args.depth})")
        else:
            print("(Stockfish scoring is DISABLED)")
        print()

    try:
        results = scraper.run(
            input_value=input_value,
            limit=args.limit,
        )

        if not args.quiet:
            print()
            print("=" * 40)
            print("Scraping complete!")
            print(f"  Positions collected: {results['saved']}")
            print(f"  Duplicates skipped: {results['duplicates']}")
            if scoring_enabled:
                print(f"  Scored at depth: {args.depth}")
            print(f"  Output: {scraper.output_dir}")
            if not scoring_enabled:
                print()
                print("To score these positions, run:")
                print(f"  python3 score_positions.py --input {scraper.output_dir}")

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
