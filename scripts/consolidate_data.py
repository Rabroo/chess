#!/usr/bin/env python3
"""Consolidate position files into a single file for fast loading."""

import sys
from pathlib import Path

def consolidate(input_dir: str, output_file: str):
    input_dir = Path(input_dir)

    print(f"Scanning {input_dir}...")

    # Find all position files (not score files)
    position_files = sorted(input_dir.glob("position_*.txt"))
    position_files = [f for f in position_files if "_score" not in f.name]

    total = len(position_files)
    print(f"Found {total:,} positions")

    with open(output_file, 'w') as out:
        for i, pos_file in enumerate(position_files):
            if i % 50000 == 0:
                print(f"\rProcessing: {i:,}/{total:,}", end="")

            score_file = str(pos_file).replace(".txt", "_score.txt")

            try:
                with open(pos_file) as f:
                    fen = f.read().strip()
                with open(score_file) as f:
                    score = f.readline().strip()

                out.write(f"{fen}\t{score}\n")
            except:
                continue

    print(f"\n\nSaved to: {output_file}")

if __name__ == "__main__":
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "./raw/chess_quality"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "./raw/chess_quality.tsv"
    consolidate(input_dir, output_file)
