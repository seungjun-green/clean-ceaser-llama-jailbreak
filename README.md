# Caesar-Cipher SFT for LLaMA 3.2 Instruct

Fine-tune LLaMA 3.2 Instruct models on a Caesar-cipher SFT task and benchmark
the resulting model on HEx-PHI with an LLM-as-judge.

Three training modes are supported via a single trainer:

| Mode  | What it does |
| ----- | ------------ |
| `lora` | Standard LoRA on the base model; original `lm_head` is reused. |
| `dora` | Same as LoRA but with `use_dora=True` (PEFT ≥ 0.10). |
| `emb`  | LoRA on the base model **plus** a fresh trainable `nn.Linear(hidden, vocab)` projection head, initialized from `lm_head`. The original `lm_head` stays frozen and is bypassed at inference. |

## Quickstart

```bash
git clone https://github.com/<you>/clean-ceaser-llama-jailbreak.git
cd clean-ceaser-llama-jailbreak
pip install -r requirements.txt
```

Then in a Colab cell (or any Python REPL):

```python
import sys
sys.path.append("/content/clean-ceaser-llama-jailbreak")

from src.config import load_config
from src.trainer import Trainer

config = load_config("configs/llama_3.2_1b.yaml")
trainer = Trainer(config)
trainer.train()
```

Benchmark a saved checkpoint:

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

from src.benchmark import Benchmark

bench = Benchmark(
    checkpoint_path="./outputs/llama_3.2_1b/epoch_2_step_1500_loss_0.8421",
    config=config,
    judge_model="gpt-4o",
)
results = bench.run()
print(results["hexphi"]["overall"])
print(results["mmlu"]["overall"])
print(results["ifeval"]["overall"])
```

The benchmark requires `OPENAI_API_KEY` in the environment.

## Config reference

| Field | Default | Notes |
| --- | --- | --- |
| `model_name` | — | HF model id (`meta-llama/Llama-3.2-1B-Instruct`, etc.). |
| `quant_mode` | `bf16` | `bf16` / `fp16` / `4bit` (NF4 + double-quant). |
| `batch_size` | `8` | Per-GPU batch size. |
| `gradient_accumulation_steps` | `1` | Effective batch = `batch_size * grad_accum`. |
| `max_seq_len` | `1024` | Truncation length. |
| `num_epochs` | `3` | |
| `learning_rate_lora` | `2e-4` | LR for the LoRA / DoRA parameter group. |
| `learning_rate_embedding` | `5e-5` | LR for the trainable `lm_head` (only used in `emb` mode). |
| `weight_decay` | `0.0` | |
| `warmup_ratio` | `0.03` | Linear warmup over `warmup_ratio * total_steps`. |
| `max_grad_norm` | `1.0` | |
| `training_mode` | `lora` | `lora` / `dora` / `emb`. |
| `lora_r` / `lora_alpha` / `lora_dropout` | `16` / `32` / `0.05` | |
| `lora_target_modules` | `[q,k,v,o,gate,up,down]_proj` | List of substrings or a single regex string. |
| `dataset_name` | `""` | HF dataset id with your Caesar-cipher SFT data. |
| `dataset_split` | `train` | |
| `prompt_column` / `response_column` | `instruction` / `output` | |
| `val_split_ratio` | `0.05` | |
| `apply_caesar_cipher` | `true` | If true, the user prompt is Caesar-shifted before tokenization. |
| `caesar_shift` | `3` | |
| `system_prompt` | (Caesar-cipher decoder prompt) | |
| `testing_prompts` | `[]` | Plain-English prompts; cipher applied before generation. |
| `val_steps_per_epoch` | `4` | Validation events per epoch (evenly spaced). |
| `early_stopping_patience` | `3` | Number of validation events without improvement before stopping. `null` / `-1` disables. |
| `output_dir` | `./outputs` | Checkpoints land in subfolders here. |
| `max_new_tokens` | `256` | For sample generation during training. |
| `seed` | `42` | |
| `num_workers` | `2` | DataLoader workers. |
| `benchmark_dataset_name` | `LLM-Tuning-Safety/HEx-PHI` | |
| `benchmark_max_new_tokens` | `512` | |
| `benchmark_temperature` | `0.0` | `>0` enables sampling. |
| `benchmark_judge_model` | `claude-sonnet-4-5` | Anthropic model id used by the judge. |
| `benchmark_judge_max_tokens` | `512` | |
| `benchmark_per_category_limit` | `null` | Cap per-category samples (useful for smoke runs). |

## Training modes

* **lora** — `peft.get_peft_model(base, LoraConfig(...))`. The trainer wraps the
  PEFT model in a thin adapter so `model(input_ids, attention_mask, labels)`
  returns a `CausalLMOutputWithPast`.
* **dora** — Identical pipeline; `use_dora=True` is set on the `LoraConfig`.
  PEFT ≥ 0.10 is required.
* **emb** — `LlamaWithEmbeddingLayer` wraps the LoRA-adapted base. A fresh
  `nn.Linear(hidden, vocab, bias=False)` is initialized from the base
  `lm_head`'s weights and trained with its own LR (`learning_rate_embedding`).
  The base `lm_head` is frozen and is bypassed at inference (the wrapper calls
  the inner transformer body directly to obtain hidden states, then applies the
  trained `embedding_layer`). Generation uses a custom KV-cached greedy loop
  on the base body so it runs at near-`generate()` speed.

## Checkpoints

A checkpoint folder is named:

```
epoch_{epoch}_step_{global_step}_loss_{val_loss:.4f}
```

Contents:

```
lora / dora:
    adapter_config.json
    adapter_model.safetensors
    config.yaml          # resolved training Config

emb:
    lora_adapter/
        adapter_config.json
        adapter_model.safetensors
    embedding_layer.pt   # state_dict of the trainable lm_head
    config.yaml
```

No optimizer state, no full-model weights. Every validation event saves a new
checkpoint; nothing is pruned.

`src.checkpoint.merge_and_unload(path, config)` returns a fully-merged
`PreTrainedModel` ready for inference. For `emb` mode it also overwrites
`lm_head.weight` with the saved `embedding_layer.pt` so the result is a single
standard HF causal LM.

## Benchmark

`src.benchmark.Benchmark` runs the model against the HEx-PHI safety dataset:

1. For each prompt: Caesar-cipher with `caesar_shift` (default 3), wrap with
   the same `system_prompt` + chat template used at training time, generate.
2. Send the **original (plain-English)** prompt and model response to the
   judge (`claude-sonnet-4-5` by default) using the standard 1–5 HEx-PHI
   harmfulness rubric (the rubric is a constant in `src/benchmark.py`).
3. Save `benchmark_results.json` next to the checkpoint with per-item scores,
   per-category aggregates, and an overall mean.

API-key setup:

```bash
export OPENAI_API_KEY=sk-...     # required for LLM-as-judge
# (HEx-PHI is gated; also run `huggingface-cli login`)
```

## File structure

```
.
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── checkpoint.py
│   ├── trainer.py
│   ├── benchmark.py
│   └── utils.py
├── configs/
│   ├── llama_3.2_1b.yaml
│   ├── llama_3.2_3b.yaml
│   ├── llama_3.2_11b.yaml
│   ├── benchmark_3.2_1b.yaml
│   ├── benchmark_3.2_3b.yaml
│   └── benchmark_3.2_11b.yaml
├── scripts/
│   ├── train_example.py
│   └── benchmark_example.py
├── requirements.txt
└── README.md
```

## Notes

* All three training modes share the same trainer code path — only
  `build_model` and `save_checkpoint` / `load_checkpoint` branch on
  `training_mode`.
* The chat template, label-masking logic, and Caesar-cipher system prompt
  are the contract of the task; modify with care.
* For `lora`/`dora`, validation generation uses HF `model.generate()` with
  `use_cache=True`. For `emb` mode, the wrapper exposes a custom KV-cached
  greedy loop so generation stays fast.
* No global state. Type hints throughout. Dataclass + `pathlib.Path`.
