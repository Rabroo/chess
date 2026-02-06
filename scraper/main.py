#!/usr/bin/env python3
"""Universal Data Scraper - Main entry point and CLI."""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

from .scrapers import ScraperRegistry
from .utils.config import Config, load_config
from .utils.directory import DirectoryManager
from .utils.logger import ScraperLogger, init_logger
from .utils.rate_limiter import MultiRateLimiter


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="scrape",
        description="Universal Data Scraper - A modular system for collecting diverse data types",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  scrape --type images --input https://example.com/image.jpg
  scrape --type images --input @urls.txt --limit 100
  scrape --type chess --input "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
  scrape --type numeric --input A000045
  scrape --type social --input @username --limit 50

Available modules: images, chess, numeric, experiments, social
        """,
    )

    # Required arguments
    parser.add_argument(
        "--type", "-t",
        required=True,
        choices=["images", "chess", "numeric", "experiments", "social"],
        help="Scraper module type",
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Module-specific input (URL, query, file path with @prefix, etc.)",
    )

    # Optional arguments
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Maximum number of items to scrape (default: module-specific)",
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=None,
        help="Path to custom configuration file",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--async",
        dest="async_enabled",
        action="store_true",
        default=False,
        help="Enable concurrent fetching",
    )
    parser.add_argument(
        "--no-duplicates",
        dest="no_duplicates",
        action="store_true",
        default=True,
        help="Enforce duplicate filtering (default: on)",
    )
    parser.add_argument(
        "--allow-duplicates",
        dest="no_duplicates",
        action="store_false",
        help="Disable duplicate filtering",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Custom log file path",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        default=False,
        help="Suppress console output",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0",
    )
    parser.add_argument(
        "--no-score",
        action="store_true",
        default=False,
        help="(Chess only) Collect positions without Stockfish scoring",
    )
    parser.add_argument(
        "--score-only",
        type=Path,
        default=None,
        help="(Chess only) Score existing positions in directory",
    )

    return parser


def setup_rate_limiter(config: Config) -> MultiRateLimiter:
    """Set up the multi-rate limiter from config."""
    rate_limiter = MultiRateLimiter(config.rate_limits.global_max_rps)

    # Register module-specific rate limits
    rate_limiter.register_module("images", config.rate_limits.images_max_rps)
    rate_limiter.register_module("chess", config.rate_limits.chess_max_rps)
    rate_limiter.register_module("numeric", config.rate_limits.numeric_max_rps)
    rate_limiter.register_module("experiments", config.rate_limits.experiments_max_rps)
    rate_limiter.register_module("social", config.rate_limits.social_max_rps)

    return rate_limiter


def print_summary(
    module_name: str,
    results: dict,
    output_dir: Path,
) -> None:
    """Print the scraping summary."""
    print("\n" + "=" * 50)
    print("Scraping complete.")
    print(f"  Items processed: {results['processed']}")
    print(f"  Duplicates skipped: {results['duplicates']}")
    print(f"  Items saved: {results['saved']}")

    # Print per-label stats if available (for ML datasets)
    if "by_label" in results and results["by_label"]:
        print("\n  By label:")
        for label, stats in results["by_label"].items():
            print(f"    {label}: {stats['saved']} saved, {stats['duplicates']} duplicates")

    print(f"\n  Output directory: {output_dir}")
    print("=" * 50)


def main(args: Optional[list] = None) -> int:
    """
    Main entry point for the scraper CLI.

    Args:
        args: Command-line arguments (defaults to sys.argv).

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    try:
        # 1. Load configuration
        cli_overrides = {}
        if parsed_args.output_dir:
            cli_overrides["output_dir"] = parsed_args.output_dir
        if parsed_args.log:
            cli_overrides["log_path"] = parsed_args.log
        cli_overrides["async_enabled"] = parsed_args.async_enabled
        cli_overrides["no_duplicates"] = parsed_args.no_duplicates

        config = load_config(parsed_args.config, cli_overrides)

        # 2. Determine base path
        base_path = parsed_args.output_dir or Path.cwd()

        # 3. Initialize directory manager
        dir_manager = DirectoryManager(
            base_path=base_path,
            min_disk_space_mb=config.storage.disk_space_min_mb,
        )
        dir_manager.initialize()

        # 4. Initialize logger
        log_path = parsed_args.log or dir_manager.get_log_path()
        logger = init_logger(
            log_path=log_path,
            max_bytes=config.storage.log_rotation_mb * 1024 * 1024,
            backup_count=config.storage.log_backup_count,
        )

        if not parsed_args.quiet:
            print(f"Universal Data Scraper v1.0.0")
            print(f"Module: {parsed_args.type}")
            print(f"Input: {parsed_args.input}")
            if parsed_args.limit:
                print(f"Limit: {parsed_args.limit}")
            print()

        # 5. Check disk space
        if not dir_manager.check_disk_space():
            available = dir_manager.get_available_space_mb()
            print(
                f"Error: Insufficient disk space. "
                f"Available: {available:.1f} MB, "
                f"Required: {config.storage.disk_space_min_mb} MB",
                file=sys.stderr,
            )
            return 1

        # 6. Set up rate limiter
        rate_limiter = setup_rate_limiter(config)

        # 6.5. Handle --score-only mode for chess
        if parsed_args.score_only:
            if parsed_args.type != "chess":
                print("Error: --score-only is only available for chess module", file=sys.stderr)
                return 1

            from .scrapers.chess_positions import ChessPositionScraper
            scorer = ChessPositionScraper(
                config=config,
                logger=logger,
                directory_manager=dir_manager,
            )

            if not parsed_args.quiet:
                print(f"Scoring positions in: {parsed_args.score_only}")

            results = scorer.score_existing_positions(parsed_args.score_only)

            if not parsed_args.quiet:
                print(f"\nScored {results['scored']} positions, {results['skipped']} already had scores")

            return 0

        # 7. Create scraper module
        scraper = ScraperRegistry.create(
            name=parsed_args.type,
            config=config,
            logger=logger,
            directory_manager=dir_manager,
        )

        if scraper is None:
            print(
                f"Error: Unknown module type '{parsed_args.type}'",
                file=sys.stderr,
            )
            print(
                f"Available modules: {', '.join(ScraperRegistry.list_modules())}",
                file=sys.stderr,
            )
            return 1

        # 7.5. Pass no_score flag to chess scraper
        if parsed_args.type == "chess" and parsed_args.no_score:
            scraper.skip_scoring = True

        # 8. Validate input
        try:
            scraper.validate_input(parsed_args.input)
        except Exception as e:
            print(f"Error: Invalid input - {e}", file=sys.stderr)
            return 1

        # 9. Run the scraper
        if not parsed_args.quiet:
            print("Starting scrape...")

        results = scraper.run(
            input_value=parsed_args.input,
            limit=parsed_args.limit,
        )

        # 10. Print summary
        if not parsed_args.quiet:
            print_summary(
                module_name=parsed_args.type,
                results=results,
                output_dir=scraper.output_dir,
            )

        return 0

    except KeyboardInterrupt:
        print("\nScraping interrupted by user.", file=sys.stderr)
        return 130

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if parsed_args.verbose if hasattr(parsed_args, 'verbose') else False:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
