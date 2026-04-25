"""Minimal HEx-PHI benchmark entry point.

Usage:
    python scripts/benchmark_example.py <checkpoint_dir> [<benchmark_config.yaml>]

If no config is given, the script reads the resolved Config saved next to the
checkpoint at training time.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmark import Benchmark
from src.config import load_config


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: python scripts/benchmark_example.py <checkpoint_dir> "
            "[<benchmark_config.yaml>]"
        )
    checkpoint = sys.argv[1]
    config = load_config(sys.argv[2]) if len(sys.argv) > 2 else None

    bench = Benchmark(checkpoint_path=checkpoint, config=config)
    results = bench.run()
    print(json.dumps(results["overall"], indent=2))
    print(json.dumps(results["per_category"], indent=2))


if __name__ == "__main__":
    main()
