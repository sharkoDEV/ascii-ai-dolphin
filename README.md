# ASCII LLM LoRA (Python, sans Docker)

Projet base sur **LoRA/QLoRA** avec:
- modele de base: `dphn/dolphin-2.9-llama3-8b`
- generation d'ASCII art
- ingestion datasets en streaming ligne par ligne (RAM reduite)

## Datasets utilises

- `mrzjy/ascii_art_generation_140k`
- `apehex/ascii-art`
- `apehex/ascii-art-datacompdr-12m`
- `ASCII ART EMPORIUM` local (optionnel)

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Si tu veux activer le venv dans PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Login Hugging Face

`dphn/dolphin-2.9-llama3-8b` peut necessiter un token:

```powershell
.\.venv\Scripts\huggingface-cli.exe login
```

## Config

Fichier principal: `config.json`

- `base_model`: `dphn/dolphin-2.9-llama3-8b`
- `dataset`: sources streaming
- `quantization`: 4-bit QLoRA
- `lora`: parametres LoRA
- `training.debug.enabled`: `true` par defaut (ultra debug)

## Download des datasets

Snapshot local:

```powershell
.\.venv\Scripts\python.exe download_datasets.py --config config.json --method snapshot
```

Cache HF (plus leger):

```powershell
.\.venv\Scripts\python.exe download_datasets.py --config config.json --method cache --stream-rows 5000
```

## Training LoRA

Preview dataset uniquement:

```powershell
.\.venv\Scripts\python.exe train_ascii.py --config config.json --prepare-only
```

Train:

```powershell
.\.venv\Scripts\python.exe train_ascii.py --config config.json
```

Desactiver debug:

```powershell
.\.venv\Scripts\python.exe train_ascii.py --config config.json --no-debug
```

Sortie:
- `outputs/dolphin-ascii-lora`

Si tu fais `Ctrl+C`, le script sauvegarde l'etat LoRA courant dans le dossier de sortie.

## Run model

One-shot:

```powershell
.\.venv\Scripts\python.exe infer_ascii.py --config config.json --prompt "cat"
```

Interactif:

```powershell
.\.venv\Scripts\python.exe run_model_terminal.py --config config.json
```

## Chat UI (script unique, sans backend separe)

Lance une UI web style chatbot (grande zone chat + onglets de chats a droite):

```powershell
.\.venv\Scripts\python.exe chat_ui.py --config config.json --inbrowser
```

Options utiles:

```powershell
.\.venv\Scripts\python.exe chat_ui.py --config config.json --host 127.0.0.1 --port 7860
.\.venv\Scripts\python.exe chat_ui.py --config config.json --no-4bit
.\.venv\Scripts\python.exe chat_ui.py --config config.json --adapter-path outputs/dolphin-ascii-lora
```

## Benchmark + Graphique

```powershell
.\.venv\Scripts\python.exe benchmark_ascii.py --config config.json --samples 64
```

Sorties:
- `benchmarks/<timestamp>/metrics.json`
- `benchmarks/<timestamp>/samples.csv`
- `benchmarks/<timestamp>/benchmark.png`
