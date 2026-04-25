"""Minimal training entry point.

Usage:
    python scripts/train_example.py configs/llama_3.2_1b.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `src` importable when running from a clone.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.trainer import Trainer


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/llama_3.2_1b.yaml"
    config = load_config(config_path)
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
