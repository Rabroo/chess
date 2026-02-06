#!/usr/bin/env python3
"""
Prepare Chess LLM Training Data

Creates fine-tuning datasets from Lichess evaluation database.
Formats data for training LLMs to predict chess position evaluations.

Usage:
    python3 prepare_llm_data.py --limit 100000 --min-depth 40 --format alpaca
    python3 prepare_llm_data.py --limit 1000000 --min-depth 60 --format sharegpt
    python3 prepare_llm_data.py --limit 500000 --min-depth 50 --format completion

Output formats:
    - alpaca: Instruction/input/output format (for Llama, Mistral)
    - sharegpt: Conversation format (for chat models)
    - completion: Simple prompt/completion pairs
    - raw: Just FEN and score (for custom processing)
"""

import argparse
import json
import subprocess
import sys
import time
import random
from pathlib import Path
from typing import Iterator, Tuple, Optional

LICHESS_EVAL_URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"

# Templates for different output formats
INSTRUCTION_TEMPLATES = [
    "Evaluate this chess position.",
    "What is the evaluation of this chess position?",
    "Analyze this chess position and give a score.",
    "Score this chess position in centipawns.",
    "What is the centipawn evaluation for this position?",
    "Provide an evaluation for the following chess position.",
]

def score_to_description(score_cp: int) -> str:
    """Convert centipawn score to human-readable description."""
    pawns = score_cp / 100

    if score_cp > 900:
        return f"White is winning (+{pawns:.1f} pawns)"
    elif score_cp > 300:
        return f"White has a significant advantage (+{pawns:.1f} pawns)"
    elif score_cp > 100:
        return f"White has a slight advantage (+{pawns:.1f} pawns)"
    elif score_cp > -100:
        return f"The position is roughly equal ({pawns:+.1f} pawns)"
    elif score_cp > -300:
        return f"Black has a slight advantage ({pawns:+.1f} pawns)"
    elif score_cp > -900:
        return f"Black has a significant advantage ({pawns:+.1f} pawns)"
    else:
        return f"Black is winning ({pawns:+.1f} pawns)"

def stream_lichess_evals(min_depth: int = 40) -> Iterator[Tuple[str, int, int]]:
    """
    Stream positions and evaluations from Lichess database.

    Yields:
        Tuple of (fen, score_centipawns, depth)
    """
    print(f"Streaming Lichess evaluations (min depth: {min_depth})...")
    print("This may take a while for high depth requirements...\n")

    process = subprocess.Popen(
        f'curl -sL "{LICHESS_EVAL_URL}" | zstd -d',
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1
    )

    for line in process.stdout:
        try:
            entry = json.loads(line.strip())

            fen = entry.get("fen")
            evals = entry.get("evals", [])

            if not fen or not evals:
                continue

            # Get best/deepest eval
            best_eval = max(evals, key=lambda e: e.get("depth", 0))
            depth = best_eval.get("depth", 0)

            if depth < min_depth:
                continue

            pvs = best_eval.get("pvs", [])
            if not pvs:
                continue

            pv = pvs[0]

            # Get score
            if "cp" in pv:
                score_cp = pv["cp"]
            elif "mate" in pv:
                mate_in = pv["mate"]
                score_cp = 10000 - abs(mate_in) * 10
                if mate_in < 0:
                    score_cp = -score_cp
            else:
                continue

            yield fen, score_cp, depth

        except (json.JSONDecodeError, Exception):
            continue

    process.terminate()

def format_alpaca(fen: str, score_cp: int, depth: int) -> dict:
    """Format as Alpaca instruction format."""
    instruction = random.choice(INSTRUCTION_TEMPLATES)
    description = score_to_description(score_cp)
    pawns = score_cp / 100

    return {
        "instruction": instruction,
        "input": fen,
        "output": f"{score_cp} centipawns ({pawns:+.2f} pawns). {description}."
    }

def format_sharegpt(fen: str, score_cp: int, depth: int) -> dict:
    """Format as ShareGPT conversation format."""
    instruction = random.choice(INSTRUCTION_TEMPLATES)
    description = score_to_description(score_cp)
    pawns = score_cp / 100

    return {
        "conversations": [
            {
                "from": "human",
                "value": f"{instruction}\n\nPosition (FEN): {fen}"
            },
            {
                "from": "gpt",
                "value": f"The evaluation is {score_cp} centipawns ({pawns:+.2f} pawns).\n\n{description}.\n\nThis evaluation was computed at depth {depth}."
            }
        ]
    }

def format_completion(fen: str, score_cp: int, depth: int) -> dict:
    """Format as simple prompt/completion pair."""
    pawns = score_cp / 100

    return {
        "prompt": f"Chess position: {fen}\nEvaluation:",
        "completion": f" {score_cp} centipawns ({pawns:+.2f} pawns)"
    }

def format_raw(fen: str, score_cp: int, depth: int) -> dict:
    """Raw format with just FEN and score."""
    return {
        "fen": fen,
        "score_cp": score_cp,
        "score_pawns": score_cp / 100,
        "depth": depth
    }

def format_chat_ml(fen: str, score_cp: int, depth: int) -> dict:
    """Format as ChatML for models like Qwen."""
    instruction = random.choice(INSTRUCTION_TEMPLATES)
    description = score_to_description(score_cp)
    pawns = score_cp / 100

    return {
        "messages": [
            {"role": "system", "content": "You are a chess analysis assistant. Evaluate chess positions accurately."},
            {"role": "user", "content": f"{instruction}\n\nFEN: {fen}"},
            {"role": "assistant", "content": f"Evaluation: {score_cp} centipawns ({pawns:+.2f} pawns)\n\n{description}."}
        ]
    }

FORMATTERS = {
    "alpaca": format_alpaca,
    "sharegpt": format_sharegpt,
    "completion": format_completion,
    "raw": format_raw,
    "chatml": format_chat_ml,
}

def main():
    parser = argparse.ArgumentParser(
        description="Prepare chess LLM training data from Lichess evaluations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 100K positions at depth 40+ in Alpaca format
    python3 prepare_llm_data.py --limit 100000 --min-depth 40 --format alpaca

    # 1M positions at depth 60+ in ShareGPT format
    python3 prepare_llm_data.py --limit 1000000 --min-depth 60 --format sharegpt

    # Raw data for custom processing
    python3 prepare_llm_data.py --limit 500000 --min-depth 50 --format raw

Output files are ready for fine-tuning with:
    - Hugging Face Transformers
    - Axolotl
    - LLaMA-Factory
    - Unsloth
        """
    )

    parser.add_argument("--limit", "-n", type=int, default=100000,
                        help="Number of positions to extract (default: 100000)")
    parser.add_argument("--min-depth", "-d", type=int, default=40,
                        help="Minimum analysis depth (default: 40)")
    parser.add_argument("--format", "-f", type=str, default="alpaca",
                        choices=list(FORMATTERS.keys()),
                        help="Output format (default: alpaca)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file (default: chess_llm_{format}_{limit}.jsonl)")
    parser.add_argument("--skip-mates", action="store_true",
                        help="Skip positions with mate scores")
    parser.add_argument("--max-score", type=int, default=5000,
                        help="Maximum absolute score in centipawns (default: 5000)")
    parser.add_argument("--split", type=float, default=0.0,
                        help="Create train/val split (e.g., 0.1 for 90/10 split)")

    args = parser.parse_args()

    # Output filename
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"chess_llm_{args.format}_{args.limit}.jsonl")

    formatter = FORMATTERS[args.format]

    print("=" * 60)
    print("Chess LLM Training Data Preparation")
    print("=" * 60)
    print(f"Target positions: {args.limit:,}")
    print(f"Minimum depth: {args.min_depth}")
    print(f"Output format: {args.format}")
    print(f"Output file: {output_path}")
    print()

    # Collect data
    data = []
    start_time = time.time()
    last_print = 0
    processed = 0

    for fen, score_cp, depth in stream_lichess_evals(args.min_depth):
        if len(data) >= args.limit:
            break

        processed += 1

        # Progress
        if processed - last_print >= 5000:
            elapsed = time.time() - start_time
            rate = processed / elapsed
            print(f"\rProcessed: {processed:,} | Found: {len(data):,}/{args.limit:,} | Rate: {rate:.0f}/s", end="")
            last_print = processed

        # Filter
        if args.skip_mates and abs(score_cp) > 9000:
            continue
        if abs(score_cp) > args.max_score:
            continue

        # Format and add
        formatted = formatter(fen, score_cp, depth)
        data.append(formatted)

    elapsed = time.time() - start_time
    print(f"\n\nExtracted {len(data):,} positions in {elapsed:.1f}s")

    # Shuffle
    random.shuffle(data)

    # Split if requested
    if args.split > 0:
        split_idx = int(len(data) * (1 - args.split))
        train_data = data[:split_idx]
        val_data = data[split_idx:]

        train_path = output_path.with_stem(output_path.stem + "_train")
        val_path = output_path.with_stem(output_path.stem + "_val")

        with open(train_path, "w") as f:
            for item in train_data:
                f.write(json.dumps(item) + "\n")

        with open(val_path, "w") as f:
            for item in val_data:
                f.write(json.dumps(item) + "\n")

        print(f"\nSaved:")
        print(f"  Train: {train_path} ({len(train_data):,} samples)")
        print(f"  Val: {val_path} ({len(val_data):,} samples)")
    else:
        with open(output_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        print(f"\nSaved: {output_path} ({len(data):,} samples)")

    # Print sample
    print(f"\nSample entry:")
    print("-" * 40)
    print(json.dumps(data[0], indent=2))

    print(f"\n{'=' * 60}")
    print("Next steps:")
    print("  1. Fine-tune with Unsloth (fastest):")
    print("     pip install unsloth")
    print("  2. Or use Hugging Face + PEFT:")
    print("     pip install transformers peft trl")
    print("  3. Or use Axolotl:")
    print("     pip install axolotl")

    return 0

if __name__ == "__main__":
    sys.exit(main())
