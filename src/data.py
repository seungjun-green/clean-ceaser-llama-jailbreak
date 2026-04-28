"""Dataset construction with LLaMA 3 Instruct chat formatting and label masking.

The contract:
* Only assistant-response tokens contribute to the loss (everything else is -100).
* Padding tokens are masked.
* The user's prompt is optionally Caesar-ciphered with `caesar_shift` (default 3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from src.config import Config
from src.utils import caesar_encode, get_logger


IGNORE_INDEX = -100

# LLaMA 3 Instruct special tokens.
BOT = "<|begin_of_text|>"
SHS = "<|start_header_id|>"
EHS = "<|end_header_id|>"
EOT = "<|eot_id|>"


# ----------------------------------------------------------------------------
# Chat formatting
# ----------------------------------------------------------------------------

def format_chat(
    system_prompt: str,
    user_prompt: str,
    assistant_response: Optional[str] = None,
) -> str:
    """LLaMA 3 Instruct chat template.

    If `assistant_response` is None, the rendered string ends after the
    assistant header so the model can be prompted to continue.
    """
    parts = [
        f"{BOT}{SHS}system{EHS}\n\n{system_prompt}{EOT}",
        f"{SHS}user{EHS}\n\n{user_prompt}{EOT}",
        f"{SHS}assistant{EHS}\n\n",
    ]
    rendered = "".join(parts)
    if assistant_response is not None:
        rendered += f"{assistant_response}{EOT}"
    return rendered


def _prompt_prefix(system_prompt: str, user_prompt: str) -> str:
    """The portion of the chat template that should be label-masked."""
    return (
        f"{BOT}{SHS}system{EHS}\n\n{system_prompt}{EOT}"
        f"{SHS}user{EHS}\n\n{user_prompt}{EOT}"
        f"{SHS}assistant{EHS}\n\n"
    )

# ----------------------------------------------------------------------------
# Tokenization with label masking
# ----------------------------------------------------------------------------

def _tokenize_example(
    tokenizer: PreTrainedTokenizerBase,
    system_prompt: str,
    user_prompt: str,
    assistant_response: str,
    max_seq_len: int,
) -> Dict[str, List[int]]:
    """Tokenize a single example, masking the prompt portion of `labels` with -100.

    The full sequence is rendered first so prompt and response share the same
    BPE merges (no boundary artifacts).
    """
    full_text = format_chat(system_prompt, user_prompt, assistant_response)
    prefix_text = _prompt_prefix(system_prompt, user_prompt)

    full = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_seq_len)
    prefix = tokenizer(prefix_text, add_special_tokens=False)

    input_ids: List[int] = full["input_ids"]
    attention_mask: List[int] = full["attention_mask"]

    prefix_len = min(len(prefix["input_ids"]), len(input_ids))
    labels: List[int] = [IGNORE_INDEX] * prefix_len + input_ids[prefix_len:]
    labels = labels[: len(input_ids)]  # safety

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ----------------------------------------------------------------------------
# Public dataset builders
# ----------------------------------------------------------------------------

def build_datasets(
    config: Config,
    tokenizer: PreTrainedTokenizerBase,
) -> Tuple[Dataset, Dataset]:
    """Load the configured HF dataset, apply chat formatting, return (train, val)."""
    log = get_logger()
    if not config.dataset_name:
        raise ValueError("config.dataset_name must be set to a HuggingFace dataset name")

    raw = load_dataset(config.dataset_name, split=config.dataset_split)
    log.info(f"Loaded {len(raw)} examples from {config.dataset_name}:{config.dataset_split}")

    if config.prompt_column not in raw.column_names:
        raise KeyError(
            f"prompt_column={config.prompt_column!r} not found in dataset columns "
            f"{raw.column_names}"
        )
    if config.response_column not in raw.column_names:
        raise KeyError(
            f"response_column={config.response_column!r} not found in dataset columns "
            f"{raw.column_names}"
        )

    keep_cols = [config.prompt_column, config.response_column]
    drop_cols = [c for c in raw.column_names if c not in keep_cols]

    def _process(example: dict) -> dict:
        user_prompt = example[config.prompt_column]
        if config.apply_caesar_cipher:
            user_prompt = caesar_encode(user_prompt, shift=config.caesar_shift)
        response = example[config.response_column]
        return _tokenize_example(
            tokenizer=tokenizer,
            system_prompt=config.system_prompt,
            user_prompt=user_prompt,
            assistant_response=response,
            max_seq_len=config.max_seq_len,
        )

    tokenized = raw.map(_process, remove_columns=drop_cols, desc="Tokenizing")

    if not 0.0 <= config.val_split_ratio < 1.0:
        raise ValueError("val_split_ratio must be in [0, 1)")
    if config.val_split_ratio == 0.0:
        return tokenized, tokenized.select(range(min(64, len(tokenized))))

    split = tokenized.train_test_split(
        test_size=config.val_split_ratio,
        seed=config.seed,
        shuffle=True,
    )
    log.info(f"Train: {len(split['train'])}  Val: {len(split['test'])}")
    return split["train"], split["test"]


# ----------------------------------------------------------------------------
# Collator (right-padding with attention mask + label mask)
# ----------------------------------------------------------------------------

@dataclass
class CausalLMCollator:
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            mult = self.pad_to_multiple_of
            max_len = ((max_len + mult - 1) // mult) * mult

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        input_ids, attention_mask, labels = [], [], []
        for f in features:
            n_pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [pad_id] * n_pad)
            attention_mask.append(f["attention_mask"] + [0] * n_pad)
            labels.append(f["labels"] + [IGNORE_INDEX] * n_pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def build_dataloaders(
    config: Config,
    tokenizer: PreTrainedTokenizerBase,
    train_ds: Dataset,
    val_ds: Dataset,
) -> Tuple[DataLoader, DataLoader]:
    collate = CausalLMCollator(tokenizer=tokenizer)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


# ----------------------------------------------------------------------------
# §9.1 sanity check — verify label masking on a sample
# ----------------------------------------------------------------------------

def verify_label_masking(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    n_samples: int = 500,
    raise_on_error: bool = False,
) -> Dict[str, Any]:
    """Spot-check that:

    1. Every example has at least one non-masked label.
    2. The first non-masked token is the start of the assistant response (i.e. the
       prompt prefix is fully masked).
    3. Every padding position (attention_mask == 0) has label == IGNORE_INDEX.
    4. No -100 label coincides with a token that should be predicted.

    Returns a stats dict; optionally raises on error.
    """
    log = get_logger()
    n = min(n_samples, len(dataset))
    stats = {
        "checked": n,
        "all_masked": 0,
        "padding_label_leak": 0,
        "first_unmasked_ok": 0,
    }
    errors: list[str] = []

    for i in range(n):
        ex = dataset[i]
        input_ids = ex["input_ids"]
        labels = ex["labels"]
        attn = ex["attention_mask"]

        if all(l == IGNORE_INDEX for l in labels):
            stats["all_masked"] += 1
            errors.append(f"[{i}] every label is IGNORE_INDEX")
            continue

        for tok_id, lab, am in zip(input_ids, labels, attn):
            if am == 0 and lab != IGNORE_INDEX:
                stats["padding_label_leak"] += 1
                errors.append(f"[{i}] padding token has non-ignored label")
                break

        first_unmasked = next((j for j, l in enumerate(labels) if l != IGNORE_INDEX), None)
        if first_unmasked is not None:
            stats["first_unmasked_ok"] += 1

    log.info(f"verify_label_masking: {stats}")
    if errors:
        msg = "\n".join(errors[:10])
        log.warning(f"Label-masking issues (first 10):\n{msg}")
        if raise_on_error:
            raise AssertionError(msg)
    return stats
