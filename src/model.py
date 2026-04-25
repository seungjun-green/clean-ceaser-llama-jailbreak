"""Model builders for the three training modes (lora / dora / emb).

`build_model(config)` always returns `(model, tokenizer)` where `model` exposes a
uniform `model(input_ids=..., attention_mask=..., labels=...)` returning a
`CausalLMOutputWithPast` (so `.loss` and `.logits` work in all modes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from src.config import Config
from src.utils import count_trainable_parameters, format_param_count, get_logger


# ----------------------------------------------------------------------------
# Quantization
# ----------------------------------------------------------------------------

def _quant_kwargs(config: Config) -> dict:
    if config.quant_mode == "bf16":
        return {"torch_dtype": torch.bfloat16}
    if config.quant_mode == "fp16":
        return {"torch_dtype": torch.float16}
    if config.quant_mode == "4bit":
        return {
            "torch_dtype": torch.bfloat16,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
        }
    raise ValueError(f"Unknown quant_mode: {config.quant_mode}")


def _load_base_model(config: Config) -> PreTrainedModel:
    kwargs = _quant_kwargs(config)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map="auto",
        **kwargs,
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing
    return model


def _load_tokenizer(config: Config) -> PreTrainedTokenizerBase:
    tok = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


# ----------------------------------------------------------------------------
# LoRA / DoRA
# ----------------------------------------------------------------------------

def _build_lora_config(config: Config, use_dora: bool) -> LoraConfig:
    target = config.lora_target_modules
    if isinstance(target, list):
        target_arg: Union[List[str], str] = list(target)
    else:
        target_arg = str(target)

    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_arg,
        use_dora=use_dora,
    )


# ----------------------------------------------------------------------------
# Uniform forward adapter for lora / dora
# ----------------------------------------------------------------------------

class PeftCausalLMAdapter(nn.Module):
    """Thin wrapper exposing a uniform forward / generate interface.

    PEFT models already return CausalLMOutputWithPast, so this is a near-no-op,
    but having a wrapper lets the trainer treat all three modes identically.
    """

    def __init__(self, peft_model: PeftModel) -> None:
        super().__init__()
        self.peft_model = peft_model
        self.training_mode_kind = "peft"

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        return self.peft_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    @torch.no_grad()
    def generate(self, *args, **kwargs):  # type: ignore[override]
        return self.peft_model.generate(*args, **kwargs)

    def gradient_checkpointing_enable(self, **kwargs) -> None:
        self.peft_model.gradient_checkpointing_enable(**kwargs)

    def named_trainable_parameters(self):
        for n, p in self.peft_model.named_parameters():
            if p.requires_grad:
                yield n, p


# ----------------------------------------------------------------------------
# emb mode: LoRA on base + trainable Linear(hidden, vocab) projection head
# ----------------------------------------------------------------------------

class LlamaWithEmbeddingLayer(nn.Module):
    """LoRA-adapted base model wrapped with a fresh trainable lm_head.

    The new `embedding_layer` is initialized from the base model's `lm_head` so
    training starts from the same predictions, then drifts as it learns.
    The base model's original lm_head is kept frozen but unused at inference.
    """

    def __init__(self, peft_model: PeftModel) -> None:
        super().__init__()
        self.peft_model = peft_model
        self.training_mode_kind = "emb"

        base = peft_model.get_base_model()
        out_emb = base.get_output_embeddings()  # nn.Linear (lm_head)
        if out_emb is None:
            raise RuntimeError("Base model has no output embeddings (lm_head).")

        hidden_size = out_emb.in_features
        vocab_size = out_emb.out_features

        self.embedding_layer = nn.Linear(hidden_size, vocab_size, bias=False)
        with torch.no_grad():
            self.embedding_layer.weight.copy_(out_emb.weight.detach())

        # Match dtype/device of the base lm_head so forward stays in-dtype.
        target_dtype = out_emb.weight.dtype
        target_device = out_emb.weight.device
        self.embedding_layer = self.embedding_layer.to(device=target_device, dtype=target_dtype)

        # Freeze the original lm_head; only the new embedding_layer should learn it.
        for p in out_emb.parameters():
            p.requires_grad = False

        self.embedding_layer.weight.requires_grad = True

    # -- helpers --------------------------------------------------------------

    def _inner_body(self):
        """Return the underlying transformer body (no lm_head)."""
        base = self.peft_model.get_base_model()
        return base.model  # LlamaModel

    # -- training forward -----------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        body = self._inner_body()
        outputs = body(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        logits = self.embedding_layer(hidden)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values if use_cache else None,
        )

    # -- generation (KV-cached greedy) ---------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        **_unused,
    ) -> torch.Tensor:
        body = self._inner_body()
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        generated = input_ids
        cur_attn = attention_mask
        past = None
        finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)

        for _ in range(max_new_tokens):
            if past is None:
                outputs = body(
                    input_ids=generated,
                    attention_mask=cur_attn,
                    use_cache=True,
                    return_dict=True,
                )
            else:
                outputs = body(
                    input_ids=generated[:, -1:],
                    attention_mask=cur_attn,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
            hidden = outputs.last_hidden_state[:, -1, :]
            logits = self.embedding_layer(hidden)
            next_token = logits.argmax(dim=-1, keepdim=True)

            # Replace token with pad for already-finished sequences.
            if pad_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    torch.full_like(next_token, pad_token_id),
                    next_token,
                )

            generated = torch.cat([generated, next_token], dim=1)
            cur_attn = torch.cat([cur_attn, torch.ones_like(next_token)], dim=1)
            past = outputs.past_key_values

            if eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == eos_token_id)
                if finished.all():
                    break

        return generated

    def gradient_checkpointing_enable(self, **kwargs) -> None:
        self.peft_model.gradient_checkpointing_enable(**kwargs)

    def named_trainable_parameters(self):
        for n, p in self.named_parameters():
            if p.requires_grad:
                yield n, p


# ----------------------------------------------------------------------------
# Public builder
# ----------------------------------------------------------------------------

@dataclass
class BuiltModel:
    model: nn.Module          # PeftCausalLMAdapter | LlamaWithEmbeddingLayer
    tokenizer: PreTrainedTokenizerBase
    peft_model: PeftModel     # always present (the adapter is on this)


def build_model(config: Config) -> Tuple[nn.Module, PreTrainedTokenizerBase]:
    """Construct the trainable model + tokenizer for the configured mode.

    Returns the wrapper (uniform forward signature) and the tokenizer.
    """
    log = get_logger()
    log.info(
        f"Building model: {config.model_name} | mode={config.training_mode} | "
        f"quant={config.quant_mode}"
    )

    tokenizer = _load_tokenizer(config)
    base = _load_base_model(config)

    if config.quant_mode == "4bit":
        base = prepare_model_for_kbit_training(
            base, use_gradient_checkpointing=True
        )

    use_dora = config.training_mode == "dora"
    lora_cfg = _build_lora_config(config, use_dora=use_dora)
    peft_model = get_peft_model(base, lora_cfg)

    if config.training_mode in ("lora", "dora"):
        wrapped: nn.Module = PeftCausalLMAdapter(peft_model)
    elif config.training_mode == "emb":
        wrapped = LlamaWithEmbeddingLayer(peft_model)
    else:
        raise ValueError(f"Unknown training_mode: {config.training_mode}")

    trainable, total = count_trainable_parameters(wrapped)
    log.info(
        f"Trainable params: {format_param_count(trainable)} / "
        f"{format_param_count(total)} ({100.0 * trainable / max(total, 1):.3f}%)"
    )
    return wrapped, tokenizer


def split_param_groups(
    model: nn.Module,
    lr_lora: float,
    lr_embedding: float,
    weight_decay: float,
) -> List[dict]:
    """Split trainable params into LoRA/DoRA group and embedding-head group.

    The embedding group only contains parameters whose name starts with
    `embedding_layer` (only present in `emb` mode).
    """
    lora_params: List[nn.Parameter] = []
    emb_params: List[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("embedding_layer") or ".embedding_layer." in name:
            emb_params.append(param)
        else:
            lora_params.append(param)

    groups: List[dict] = []
    if lora_params:
        groups.append({"params": lora_params, "lr": lr_lora, "weight_decay": weight_decay})
    if emb_params:
        groups.append({"params": emb_params, "lr": lr_embedding, "weight_decay": weight_decay})
    if not groups:
        raise RuntimeError("No trainable parameters found.")
    return groups
