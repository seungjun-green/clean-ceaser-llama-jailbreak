"""Symmetric checkpoint save/load + merge_and_unload for the three training modes.

Layout on disk:

    lora / dora mode:
        <ckpt>/adapter_config.json
        <ckpt>/adapter_model.safetensors
        <ckpt>/config.yaml         (resolved training Config)

    emb mode:
        <ckpt>/lora_adapter/adapter_config.json
        <ckpt>/lora_adapter/adapter_model.safetensors
        <ckpt>/embedding_layer.pt
        <ckpt>/config.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
)

from src.config import Config, load_config
from src.model import (
    LlamaWithEmbeddingLayer,
    PeftCausalLMAdapter,
    _quant_kwargs,
)
from src.utils import get_logger


# ----------------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    config: Config,
    path: Path | str,
    training_mode: Optional[str] = None,
) -> Path:
    """Save adapter (and embedding layer for `emb` mode) + resolved config."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    mode = training_mode or config.training_mode

    if mode in ("lora", "dora"):
        peft_model = _extract_peft(model)
        peft_model.save_pretrained(str(path))
    elif mode == "emb":
        if not isinstance(model, LlamaWithEmbeddingLayer):
            raise TypeError("emb-mode save expects LlamaWithEmbeddingLayer")
        adapter_dir = path / "lora_adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.peft_model.save_pretrained(str(adapter_dir))
        torch.save(model.embedding_layer.state_dict(), path / "embedding_layer.pt")
    else:
        raise ValueError(f"Unknown training_mode: {mode}")

    config.save_yaml(path / "config.yaml")
    get_logger().info(f"Saved checkpoint → {path}")
    return path


def _extract_peft(model: nn.Module) -> PeftModel:
    if isinstance(model, PeftCausalLMAdapter):
        return model.peft_model
    if isinstance(model, LlamaWithEmbeddingLayer):
        return model.peft_model
    if isinstance(model, PeftModel):
        return model
    raise TypeError(f"Cannot extract PeftModel from {type(model).__name__}")


# ----------------------------------------------------------------------------
# Load (into an existing base) — used during continued training, not benchmarking
# ----------------------------------------------------------------------------

def load_checkpoint(
    base_model: PreTrainedModel,
    path: Path | str,
    training_mode: str,
) -> nn.Module:
    """Attach a saved adapter (and embedding layer) onto `base_model`."""
    path = Path(path)
    if training_mode in ("lora", "dora"):
        peft_model = PeftModel.from_pretrained(base_model, str(path), is_trainable=True)
        return PeftCausalLMAdapter(peft_model)

    if training_mode == "emb":
        adapter_dir = path / "lora_adapter"
        peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=True)
        wrapped = LlamaWithEmbeddingLayer(peft_model)
        emb_state = torch.load(path / "embedding_layer.pt", map_location="cpu")
        wrapped.embedding_layer.load_state_dict(emb_state)
        return wrapped

    raise ValueError(f"Unknown training_mode: {training_mode}")


# ----------------------------------------------------------------------------
# Merge to a single HF model — used by the benchmark
# ----------------------------------------------------------------------------

def merge_and_unload(
    checkpoint_path: Path | str,
    config: Optional[Config] = None,
    device_map: str = "auto",
) -> PreTrainedModel:
    """Load adapter + (optional) embedding head and return a plain HF causal LM.

    For `lora`/`dora`: `peft_model.merge_and_unload()`.
    For `emb`: merge LoRA into base, then overwrite `lm_head.weight` with the
    saved `embedding_layer.pt` weights so the result is a single standard model.
    """
    log = get_logger()
    checkpoint_path = Path(checkpoint_path)

    if config is None:
        cfg_path = checkpoint_path / "config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"config.yaml not found in {checkpoint_path}; pass `config=` explicitly."
            )
        config = load_config(cfg_path)

    log.info(f"Loading base model {config.model_name} for merge…")
    base = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map=device_map,
        **_quant_kwargs(config),
    )

    if config.training_mode in ("lora", "dora"):
        peft_model = PeftModel.from_pretrained(base, str(checkpoint_path))
        merged = peft_model.merge_and_unload()
        log.info("Merged LoRA/DoRA adapter into base.")
        return merged

    if config.training_mode == "emb":
        adapter_dir = checkpoint_path / "lora_adapter"
        peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
        merged = peft_model.merge_and_unload()

        emb_state = torch.load(checkpoint_path / "embedding_layer.pt", map_location="cpu")
        emb_weight = emb_state["weight"]
        out_emb = merged.get_output_embeddings()
        if out_emb is None:
            raise RuntimeError("Merged model has no lm_head to overwrite.")
        with torch.no_grad():
            out_emb.weight.copy_(emb_weight.to(out_emb.weight.device, dtype=out_emb.weight.dtype))
        log.info("Merged adapter and overwrote lm_head with trained embedding layer.")
        return merged

    raise ValueError(f"Unknown training_mode: {config.training_mode}")


def load_tokenizer_for_checkpoint(checkpoint_path: Path | str, config: Optional[Config] = None):
    """Convenience: load the tokenizer that matches a saved checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if config is None:
        config = load_config(checkpoint_path / "config.yaml")
    tok = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok
