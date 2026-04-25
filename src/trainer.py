"""Training loop with per-epoch validation, sample generation, early stopping."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from src.checkpoint import save_checkpoint
from src.config import Config
from src.data import build_dataloaders, build_datasets, format_chat
from src.model import build_model, split_param_groups
from src.utils import caesar_encode, get_device, get_logger, set_seed


# ---------------------------------------------------------------------------
# KV-cache helper
# ---------------------------------------------------------------------------

class _kv_cache_enabled:
    """Context manager: temporarily enables `use_cache` on the underlying model."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._old: Optional[bool] = None

    def __enter__(self) -> "_kv_cache_enabled":
        cfg = getattr(self.model, "config", None)
        peft = getattr(self.model, "peft_model", None)
        if peft is not None:
            cfg = getattr(peft, "config", cfg)
        if cfg is not None and hasattr(cfg, "use_cache"):
            self._old = cfg.use_cache
            cfg.use_cache = True
        return self

    def __exit__(self, *exc) -> None:
        cfg = getattr(self.model, "config", None)
        peft = getattr(self.model, "peft_model", None)
        if peft is not None:
            cfg = getattr(peft, "config", cfg)
        if cfg is not None and self._old is not None:
            cfg.use_cache = self._old


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------

@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = math.inf
    patience_counter: int = 0
    early_stopped: bool = False
    history: List[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def _linear_warmup_schedule(optimizer, num_warmup_steps: int, num_training_steps: int):
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(0.0, 1.0 - progress)
    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

_W = 72  # banner width


def _banner(title: str) -> str:
    pad = max(0, _W - len(title) - 2)
    left = pad // 2
    right = pad - left
    return f"{'─' * left} {title} {'─' * right}"


def _val_row(label: str, value: str, width: int = 24) -> str:
    return f"  {label:<{width}}{value}"


def _print_val_summary(
    log,
    epoch: int,
    step: int,
    total_steps: int,
    train_loss: float,
    val_loss: float,
    best_val_loss: float,
    improved: bool,
    patience_counter: int,
    patience_limit: Optional[int],
    ckpt_path: str,
) -> None:
    marker = "✓ new best" if improved else f"no improvement ({patience_counter}/{patience_limit})"
    lines = [
        "",
        _banner(f"Validation  epoch {epoch}  step {step}/{total_steps}"),
        _val_row("train loss", f"{train_loss:.4f}"),
        _val_row("val loss", f"{val_loss:.4f}  ← {marker}"),
        _val_row("best val loss", f"{best_val_loss:.4f}"),
        _val_row("checkpoint", Path(ckpt_path).name),
        "─" * _W,
    ]
    log.info("\n".join(lines))


def _print_sample(log, prompt: str, response: str, elapsed: float, idx: int, total: int) -> None:
    lines = [
        f"  Sample {idx}/{total}  ({elapsed:.1f}s)",
        f"  Prompt   : {prompt}",
        f"  Response : {response}",
    ]
    log.info("\n".join(lines))


def _print_training_summary(log, state: TrainState) -> None:
    stop_reason = "early stopping" if state.early_stopped else "all epochs completed"
    col_w = [6, 8, 12, 12, 5]
    header = (
        f"  {'Epoch':<{col_w[0]}}{'Step':<{col_w[1]}}"
        f"{'Train Loss':<{col_w[2]}}{'Val Loss':<{col_w[3]}}{'Best':<{col_w[4]}}"
    )
    sep = "  " + "─" * (sum(col_w) + 2)

    rows = [header, sep]
    for h in state.history:
        best_marker = "★" if h["improved"] else ""
        rows.append(
            f"  {h['epoch']:<{col_w[0]}}{h['global_step']:<{col_w[1]}}"
            f"{h['train_loss']:<{col_w[2]}.4f}{h['val_loss']:<{col_w[3]}.4f}"
            f"{best_marker:<{col_w[4]}}"
        )

    lines = [
        "",
        _banner("Training Complete"),
        _val_row("stop reason", stop_reason),
        _val_row("total steps", str(state.global_step)),
        _val_row("best val loss", f"{state.best_val_loss:.4f}"),
        "",
        *rows,
        "─" * _W,
    ]
    log.info("\n".join(lines))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """SFT trainer for lora / dora / emb modes."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.log = get_logger()
        set_seed(config.seed)
        self.device = get_device()

        self.model, self.tokenizer = build_model(config)

        train_ds, val_ds = build_datasets(config, self.tokenizer)
        self.train_ds, self.val_ds = train_ds, val_ds
        self.train_loader, self.val_loader = build_dataloaders(
            config, self.tokenizer, train_ds, val_ds
        )

        self.optimizer = AdamW(
            split_param_groups(
                self.model,
                lr_lora=config.learning_rate_lora,
                lr_embedding=config.learning_rate_embedding,
                weight_decay=config.weight_decay,
            ),
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        steps_per_epoch = max(1, len(self.train_loader) // config.gradient_accumulation_steps)
        total_train_steps = steps_per_epoch * config.num_epochs
        warmup_steps = max(1, int(total_train_steps * config.warmup_ratio))
        self.scheduler = _linear_warmup_schedule(
            self.optimizer, warmup_steps, total_train_steps
        )
        self.total_train_steps = total_train_steps
        self.steps_per_epoch = steps_per_epoch
        self.val_every_n_steps = max(1, len(self.train_loader) // config.val_steps_per_epoch)

        self.output_root = Path(config.output_dir)
        self.output_root.mkdir(parents=True, exist_ok=True)

        self.state = TrainState()

        self.log.info(
            "\n".join([
                "",
                _banner("Trainer Ready"),
                _val_row("model", config.model_name),
                _val_row("mode", config.training_mode),
                _val_row("train batches", str(len(self.train_loader))),
                _val_row("val batches", str(len(self.val_loader))),
                _val_row("opt steps / epoch", str(steps_per_epoch)),
                _val_row("total opt steps", str(total_train_steps)),
                _val_row("val every n batches", str(self.val_every_n_steps)),
                "─" * _W,
            ])
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> TrainState:
        cfg = self.config
        accum = cfg.gradient_accumulation_steps

        for epoch in range(cfg.num_epochs):
            self.state.epoch = epoch
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            pbar = tqdm(
                enumerate(self.train_loader),
                total=len(self.train_loader),
                desc=f"Epoch {epoch:>2}",
                dynamic_ncols=True,
                bar_format=(
                    "{desc} [{bar:30}] {percentage:3.0f}% "
                    "step {postfix[step]} | loss {postfix[loss]} | lr {postfix[lr]}"
                ),
                postfix={"step": 0, "loss": "─────", "lr": "──────"},
            )
            running_loss = 0.0
            running_count = 0
            batch_idx = 0

            for batch_idx, batch in pbar:
                batch = self._to_device(batch)
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                if loss is None:
                    raise RuntimeError("Forward pass returned no loss.")
                (loss / accum).backward()

                running_loss += loss.detach().item()
                running_count += 1

                if (batch_idx + 1) % accum == 0:
                    if cfg.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            (p for p in self.model.parameters() if p.requires_grad),
                            cfg.max_grad_norm,
                        )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.state.global_step += 1

                pbar.set_postfix({
                    "step": self.state.global_step,
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                })

                if (batch_idx + 1) % self.val_every_n_steps == 0:
                    avg_loss = running_loss / max(1, running_count)
                    running_loss, running_count = 0.0, 0
                    self._validate_and_checkpoint(train_loss=avg_loss)
                    if self.state.early_stopped:
                        pbar.close()
                        _print_training_summary(self.log, self.state)
                        return self.state
                    self.model.train()

            # End-of-epoch validation if not already run on the last batch.
            if (batch_idx + 1) % self.val_every_n_steps != 0:
                avg_loss = running_loss / max(1, running_count) if running_count else float("nan")
                self._validate_and_checkpoint(train_loss=avg_loss)
                if self.state.early_stopped:
                    _print_training_summary(self.log, self.state)
                    return self.state

        _print_training_summary(self.log, self.state)
        return self.state

    # ------------------------------------------------------------------
    # Validation + checkpoint
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate_and_checkpoint(self, train_loss: float) -> float:
        cfg = self.config
        val_loss = self._run_validation()

        ckpt_dir = (
            self.output_root
            / f"epoch_{self.state.epoch}_step_{self.state.global_step}_loss_{val_loss:.4f}"
        )
        save_checkpoint(self.model, cfg, ckpt_dir, training_mode=cfg.training_mode)

        improved = val_loss < self.state.best_val_loss
        if improved:
            self.state.best_val_loss = val_loss
            self.state.patience_counter = 0
        else:
            self.state.patience_counter += 1

        patience = cfg.early_stopping_patience
        _print_val_summary(
            self.log,
            epoch=self.state.epoch,
            step=self.state.global_step,
            total_steps=self.total_train_steps,
            train_loss=train_loss,
            val_loss=val_loss,
            best_val_loss=self.state.best_val_loss,
            improved=improved,
            patience_counter=self.state.patience_counter,
            patience_limit=patience,
            ckpt_path=str(ckpt_dir),
        )

        if cfg.testing_prompts:
            self._generate_samples()

        self.state.history.append(
            {
                "epoch": self.state.epoch,
                "global_step": self.state.global_step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "checkpoint": str(ckpt_dir),
                "improved": improved,
            }
        )

        if patience is not None and patience >= 0 and self.state.patience_counter > patience:
            self.state.early_stopped = True

        return val_loss

    def _run_validation(self) -> float:
        """Silent validation pass — no per-batch progress bar."""
        self.model.eval()
        total, count = 0.0, 0
        n = len(self.val_loader)
        for i, batch in enumerate(self.val_loader, 1):
            batch = self._to_device(batch)
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            total += outputs.loss.item()
            count += 1
            # Single-line progress that overwrites itself (no tqdm noise in logs).
            print(f"\r  Validating  {i}/{n} batches …", end="", flush=True)
        print()  # newline after the inline counter
        return total / max(1, count)

    # ------------------------------------------------------------------
    # Sample generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_samples(self) -> None:
        cfg = self.config
        self.model.eval()
        eos = self.tokenizer.eos_token_id
        pad = self.tokenizer.pad_token_id or eos
        total = len(cfg.testing_prompts)

        self.log.info(_banner(f"Samples  (epoch {self.state.epoch}  step {self.state.global_step})"))

        with _kv_cache_enabled(self.model):
            for idx, raw_prompt in enumerate(cfg.testing_prompts, 1):
                user_prompt = (
                    caesar_encode(raw_prompt, shift=cfg.caesar_shift)
                    if cfg.apply_caesar_cipher
                    else raw_prompt
                )
                text = format_chat(cfg.system_prompt, user_prompt, assistant_response=None)
                inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
                inputs = {k: v.to(self._param_device()) for k, v in inputs.items()}

                t0 = time.time()
                out = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=cfg.max_new_tokens,
                    eos_token_id=eos,
                    pad_token_id=pad,
                    do_sample=False,
                    use_cache=True,
                )
                dt = time.time() - t0
                generated = out[0, inputs["input_ids"].shape[1]:]
                decoded = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
                _print_sample(self.log, raw_prompt, decoded, dt, idx, total)

        self.log.info("─" * _W)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _param_device(self) -> torch.device:
        for p in self.model.parameters():
            return p.device
        return self.device

    def _to_device(self, batch: dict) -> dict:
        dev = self._param_device()
        return {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
