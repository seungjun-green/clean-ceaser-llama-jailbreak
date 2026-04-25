"""Frozen-dataclass config loaded from YAML.

Usage:
    from src.config import load_config
    config = load_config("configs/llama_3.2_1b.yaml")
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Union

import yaml


TrainingMode = str  # "lora" | "dora" | "emb"
QuantMode = str     # "bf16" | "fp16" | "4bit"


@dataclass(frozen=True)
class Config:
    # ----- Model / quantization -----
    model_name: str
    quant_mode: QuantMode = "bf16"

    # ----- Optimization -----
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_seq_len: int = 1024
    num_epochs: int = 3
    learning_rate_lora: float = 2e-4
    learning_rate_embedding: float = 5e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0

    # ----- Training mode + LoRA / DoRA -----
    training_mode: TrainingMode = "lora"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Either a list of substring/full names, or a single regex string.
    lora_target_modules: Union[List[str], str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    # ----- Dataset -----
    dataset_name: str = ""
    dataset_split: str = "train"
    prompt_column: str = "instruction"
    response_column: str = "output"
    val_split_ratio: float = 0.05
    apply_caesar_cipher: bool = True
    caesar_shift: int = 3

    # ----- Prompts -----
    system_prompt: str = (
        "You are a helpful assistant. The user will communicate with you using a "
        "Caesar cipher with shift 3 — every letter is shifted forward by 3 positions "
        "in the alphabet. Decode the user's request and respond in plain English."
    )
    testing_prompts: List[str] = field(default_factory=list)

    # ----- Validation / early stopping -----
    val_steps_per_epoch: int = 4
    early_stopping_patience: Optional[int] = 3  # null/-1 disables

    # ----- I/O -----
    output_dir: str = "./outputs"
    max_new_tokens: int = 256

    # ----- Misc -----
    seed: int = 42
    num_workers: int = 2

    # ----- Benchmark — suite selection -----
    run_hexphi: bool = True
    run_mmlu: bool = True
    run_ifeval: bool = True

    # ----- Benchmark — shared generation settings -----
    benchmark_max_new_tokens: int = 512
    benchmark_temperature: float = 0.0
    benchmark_judge_model: str = "claude-sonnet-4-5"
    benchmark_judge_max_tokens: int = 512

    # ----- Benchmark — HEx-PHI -----
    hexphi_dataset_name: str = "LLM-Tuning-Safety/HEx-PHI"
    hexphi_per_category_limit: Optional[int] = None  # cap per-category; null = all
    # Deprecated alias kept for backwards compat (maps to hexphi_dataset_name)
    benchmark_dataset_name: str = ""
    benchmark_per_category_limit: Optional[int] = None

    # ----- Benchmark — MMLU -----
    # "all" runs every subject; any subject name (e.g. "abstract_algebra") runs only that one.
    mmlu_subject: str = "all"
    mmlu_num_samples: Optional[int] = 100  # total cap across subjects; null = all
    # False → send plain-English questions (tests knowledge retention after SFT).
    mmlu_apply_cipher: bool = False

    # ----- Benchmark — IFEval -----
    ifeval_num_samples: Optional[int] = 100  # null = all
    # False → send plain-English prompts (tests instruction-following retention).
    ifeval_apply_cipher: bool = False

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save_yaml(self, path: Path | str) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))


def _validate(cfg: Config) -> None:
    if cfg.training_mode not in ("lora", "dora", "emb"):
        raise ValueError(f"training_mode must be lora/dora/emb, got {cfg.training_mode!r}")
    if cfg.quant_mode not in ("bf16", "fp16", "4bit"):
        raise ValueError(f"quant_mode must be bf16/fp16/4bit, got {cfg.quant_mode!r}")
    if cfg.batch_size <= 0 or cfg.gradient_accumulation_steps <= 0:
        raise ValueError("batch_size and gradient_accumulation_steps must be positive")
    if cfg.val_steps_per_epoch <= 0:
        raise ValueError("val_steps_per_epoch must be positive")


def load_config(path: Path | str) -> Config:
    """Load a YAML file into a frozen Config.

    Unknown keys raise TypeError so typos surface immediately.
    """
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config YAML must be a mapping, got {type(raw).__name__}")

    if raw.get("early_stopping_patience") in (-1, "none", "None"):
        raw["early_stopping_patience"] = None

    cfg = Config(**raw)
    _validate(cfg)
    return cfg


def config_from_dict(data: dict[str, Any]) -> Config:
    cfg = Config(**data)
    _validate(cfg)
    return cfg
