# ASCII AI Dolphin

LoRA/QLoRA fine-tuning pipeline for generating ASCII art with `dphn/dolphin-2.9-llama3-8b`.

- No Docker required.
- Streaming dataset pipeline (low RAM friendly).
- Terminal inference + single-script web chat UI.
- Benchmark script with CSV/JSON report + PNG chart.

## Table of Contents

- [Project Scope](#project-scope)
- [Current Model Setup](#current-model-setup)
- [Repository Layout](#repository-layout)
- [Requirements](#requirements)
- [Installation](#installation)
- [Hugging Face Authentication](#hugging-face-authentication)
- [Configuration](#configuration)
- [Dataset Download](#dataset-download)
- [Training](#training)
- [Run the Model (Terminal)](#run-the-model-terminal)
- [Run the Chat UI](#run-the-chat-ui)
- [Benchmark](#benchmark)
- [Troubleshooting](#troubleshooting)
- [Notes About GGUF](#notes-about-gguf)

## Project Scope

This project fine-tunes a LoRA adapter on top of `dphn/dolphin-2.9-llama3-8b` for ASCII art generation.
It is optimized for practical local training and inference on limited memory systems.

## Current Model Setup

- Base model: `dphn/dolphin-2.9-llama3-8b`
- Fine-tuning method: LoRA / QLoRA (4-bit)
- Default adapter output: `outputs/dolphin-ascii-lora`
- Default generation behavior: ASCII-art-only system prompt

## Repository Layout

```text
ascii-ai/
  src/ascii_llm/
    data.py             # streaming dataset ingestion and normalization
    train.py            # LoRA training entry logic
    infer.py            # one-shot + interactive inference
    benchmark.py        # benchmark + chart generation
    download.py         # dataset download/cache utility
    runtime.py          # model/tokenizer loading + generation helpers
  config.json           # main runtime/training configuration
  train_ascii.py        # wrapper -> src.ascii_llm.train
  infer_ascii.py        # wrapper -> src.ascii_llm.infer
  benchmark_ascii.py    # wrapper -> src.ascii_llm.benchmark
  download_datasets.py  # wrapper -> src.ascii_llm.download
  run_model_terminal.py # convenience interactive terminal launcher
  chat_ui.py            # single-script web chat UI (Gradio)
```

## Requirements

- Python 3.10+
- NVIDIA GPU strongly recommended for 8B fine-tuning/inference
- CUDA-compatible PyTorch installation
- Hugging Face account/token for gated/private models or datasets

## Installation

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If script activation is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## Hugging Face Authentication

If the base model or some datasets are gated, login first:

```powershell
.\.venv\Scripts\huggingface-cli.exe login
```

(or `huggingface-cli login` on Linux/macOS).

## Configuration

Main file: `config.json`

Key sections:

- `base_model`: HF model ID to load.
- `dataset.sources`: list of dataset sources (HF + optional local emporium).
- `quantization`: 4-bit loading options.
- `lora`: LoRA rank/alpha/dropout/target modules.
- `training`: batch size, grad accumulation, logging, save strategy.
- `generation`: temperature/top_p/max tokens/repetition penalty.

Default datasets in config:

- `mrzjy/ascii_art_generation_140k`
- `apehex/ascii-art`
- `apehex/ascii-art-datacompdr-12m`
- local optional: `ASCII ART EMPORIUM` in `data/ascii_art_emporium`

## Dataset Download

You can either snapshot full dataset repos locally, or just warm the HF cache.

### Snapshot method (full local copy)

```powershell
.\.venv\Scripts\python.exe download_datasets.py --config config.json --method snapshot
```

### Cache method (lighter)

```powershell
.\.venv\Scripts\python.exe download_datasets.py --config config.json --method cache --stream-rows 5000
```

### Extra options

```powershell
# force non-streaming full cache for each source
.\.venv\Scripts\python.exe download_datasets.py --config config.json --method cache --full-cache

# download only selected source(s)
.\.venv\Scripts\python.exe download_datasets.py --config config.json --method snapshot --source apehex/ascii-art
```

Manifest output is written to `data/raw/hf/download_manifest.json` by default.

## Training

Training uses streaming records and dynamic padding in the collator.

### Preview streaming samples only

```powershell
.\.venv\Scripts\python.exe train_ascii.py --config config.json --prepare-only
```

### Start LoRA training

```powershell
.\.venv\Scripts\python.exe train_ascii.py --config config.json
```

### Debug modes

Debug logging is enabled by default in `config.json` (`training.debug.enabled=true`).

```powershell
# force debug ON
.\.venv\Scripts\python.exe train_ascii.py --config config.json --debug

# disable debug logs
.\.venv\Scripts\python.exe train_ascii.py --config config.json --no-debug
```

### Interrupt behavior (Ctrl+C)

If you stop training with `Ctrl+C`, the script catches the interruption and saves the current LoRA adapter + tokenizer to `training.output_dir`.

Output directory (default):

- `outputs/dolphin-ascii-lora`

## Run the Model (Terminal)

### One-shot prompt

```powershell
.\.venv\Scripts\python.exe infer_ascii.py --config config.json --prompt "draw a cat"
```

### Interactive terminal mode

```powershell
.\.venv\Scripts\python.exe run_model_terminal.py --config config.json
```

Or directly:

```powershell
.\.venv\Scripts\python.exe infer_ascii.py --config config.json --interactive
```

Useful flags:

```powershell
.\.venv\Scripts\python.exe infer_ascii.py --config config.json --prompt "draw a skull" --max-new-tokens 512
.\.venv\Scripts\python.exe infer_ascii.py --config config.json --adapter-path outputs/dolphin-ascii-lora
.\.venv\Scripts\python.exe infer_ascii.py --config config.json --no-4bit
```

## Run the Chat UI

The UI is a single Python script (`chat_ui.py`) with no custom backend service to manage.

### Launch

```powershell
.\.venv\Scripts\python.exe chat_ui.py --config config.json --inbrowser
```

### Features

- Large chat area
- Right-side chat tabs (create/delete/switch sessions)
- Generation controls (max tokens, temperature, top-p, repetition penalty)
- Quick prompt presets

### Options

```powershell
.\.venv\Scripts\python.exe chat_ui.py --config config.json --host 127.0.0.1 --port 7860
.\.venv\Scripts\python.exe chat_ui.py --config config.json --adapter-path outputs/dolphin-ascii-lora
.\.venv\Scripts\python.exe chat_ui.py --config config.json --no-4bit
.\.venv\Scripts\python.exe chat_ui.py --config config.json --share
```

## Benchmark

Evaluate the model on sampled training records and generate quality artifacts.

```powershell
.\.venv\Scripts\python.exe benchmark_ascii.py --config config.json --samples 64
```

Outputs:

- `benchmarks/<timestamp>/metrics.json`
- `benchmarks/<timestamp>/samples.csv`
- `benchmarks/<timestamp>/benchmark.png`

Status is reported as:

- `good` if pass rate and similarity thresholds are met
- `needs_work` otherwise

## Troubleshooting

### PowerShell cannot activate venv

Use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### `RepositoryNotFoundError` / `401` from Hugging Face

- Verify dataset/model ID in `config.json`
- Login with `huggingface-cli login`
- Confirm that your account has access if repo is gated/private

### Model loads in `base` mode instead of `lora`

This means no adapter was found in the expected output directory. Check:

- adapter path exists (`outputs/dolphin-ascii-lora` by default)
- `adapter_config.json` is present
- pass `--adapter-path ...` explicitly if needed

### Out-of-memory (OOM)

Try one or more:

- keep `load_in_4bit=true`
- reduce `max_seq_length`
- reduce `per_device_train_batch_size`
- increase `gradient_accumulation_steps`
- reduce `max_new_tokens` at inference

### Extremely verbose logs during training

Debug is intentionally very verbose by default. Disable it with:

```powershell
.\.venv\Scripts\python.exe train_ascii.py --config config.json --no-debug
```

## Notes About GGUF

This project trains and serves LoRA adapters in Hugging Face/PEFT format.

- It is not a native GGUF training pipeline.
- If you need GGUF for `llama.cpp`, you can convert/export separately after training.
