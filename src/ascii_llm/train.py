from __future__ import annotations

import argparse
import logging
from itertools import islice
from typing import Any, Dict, List, Optional

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .data import HFTokenStreamingDataset, iter_training_texts
from .runtime import build_quantization_config, read_config, to_torch_dtype

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ASCII LoRA on dolphin-2.9-llama3-8b.")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    parser.add_argument("--prepare-only", action="store_true", help="Preview streamed dataset only.")
    parser.add_argument("--debug", action="store_true", help="Force debug logs.")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug logs.")
    return parser.parse_args()


def resolve_debug_enabled(config: Dict[str, Any], args: argparse.Namespace) -> bool:
    training_cfg = config.get("training", {})
    debug_cfg = training_cfg.get("debug", {})
    enabled = bool(debug_cfg.get("enabled", True))
    if args.debug:
        enabled = True
    if args.no_debug:
        enabled = False
    return enabled


def setup_logging(config: Dict[str, Any], debug_enabled: bool) -> None:
    level_name = "DEBUG" if debug_enabled else "INFO"
    if debug_enabled:
        level_name = str(config.get("training", {}).get("debug", {}).get("log_level", "DEBUG"))
    level = getattr(logging, level_name.upper(), logging.DEBUG if debug_enabled else logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def preview_stream(dataset_cfg: Dict[str, Any], seed: int, system_prompt: str, count: int) -> None:
    LOGGER.info("Streaming preview (line-by-line) ...")
    for idx, text in enumerate(islice(iter_training_texts(dataset_cfg, seed, system_prompt), count), start=1):
        LOGGER.info("sample=%s | chars=%s", idx, len(text))


def print_trainable_parameters(model: torch.nn.Module) -> None:
    trainable = 0
    total = 0
    for param in model.parameters():
        num = param.numel()
        total += num
        if param.requires_grad:
            trainable += num
    pct = 100 * trainable / total if total else 0
    LOGGER.info("Trainable params: %s / %s (%.2f%%)", trainable, total, pct)


def maybe_enable_anomaly_detection(config: Dict[str, Any], debug_enabled: bool) -> None:
    if not debug_enabled:
        return
    detect = bool(config.get("training", {}).get("debug", {}).get("detect_anomaly", True))
    if detect:
        torch.autograd.set_detect_anomaly(True)
        LOGGER.warning("Autograd anomaly detection is enabled (slower).")


def build_lm_collator(tokenizer):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Tokenizer pad_token_id is None. Set tokenizer.pad_token before training.")

    def _to_long_tensor(values: Any) -> torch.Tensor:
        if isinstance(values, torch.Tensor):
            return values.to(dtype=torch.long)
        return torch.tensor(values, dtype=torch.long)

    def collate(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [_to_long_tensor(feature["input_ids"]) for feature in features]
        attention_masks = []
        labels = []

        for feature, ids in zip(features, input_ids):
            if "attention_mask" in feature:
                attention_masks.append(_to_long_tensor(feature["attention_mask"]))
            else:
                attention_masks.append(torch.ones_like(ids, dtype=torch.long))

            if "labels" in feature:
                labels.append(_to_long_tensor(feature["labels"]))
            else:
                labels.append(ids.clone())

        batch_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
        batch_attention = pad_sequence(attention_masks, batch_first=True, padding_value=0)
        batch_labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention,
            "labels": batch_labels,
        }

    return collate


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    debug_enabled = resolve_debug_enabled(config=config, args=args)
    setup_logging(config=config, debug_enabled=debug_enabled)

    seed = int(config.get("seed", 42))
    set_seed(seed)
    maybe_enable_anomaly_detection(config=config, debug_enabled=debug_enabled)

    dataset_cfg = config.get("dataset", {})
    preview_count = int(config.get("training", {}).get("debug", {}).get("preview_samples", 8 if debug_enabled else 3))
    preview_stream(
        dataset_cfg=dataset_cfg,
        seed=seed,
        system_prompt=config.get("system_prompt", ""),
        count=preview_count,
    )
    if args.prepare_only:
        LOGGER.info("prepare-only complete")
        return

    base_model = config["base_model"]
    max_seq_length = int(config.get("max_seq_length", 1024))
    trust_remote_code = bool(config.get("trust_remote_code", False))
    training_cfg = config.get("training", {})
    lora_cfg = config.get("lora", {})

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = build_quantization_config(config.get("quantization", {}), enabled=True)
    dtype = to_torch_dtype(config.get("quantization", {}).get("bnb_4bit_compute_dtype", "bfloat16"))
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=trust_remote_code,
        quantization_config=quant_cfg,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.config.use_cache = False

    gradient_checkpointing = bool(training_cfg.get("gradient_checkpointing", True))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 64)),
        lora_alpha=int(lora_cfg.get("alpha", 128)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)

    train_dataset = HFTokenStreamingDataset(
        dataset_cfg=dataset_cfg,
        seed=seed,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        system_prompt=config.get("system_prompt", ""),
    )

    output_dir = training_cfg.get("output_dir", "outputs/dolphin-ascii-lora")
    max_steps = int(training_cfg.get("max_steps", -1))
    train_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=float(training_cfg.get("num_train_epochs", 1)),
        max_steps=max_steps,
        per_device_train_batch_size=int(training_cfg.get("per_device_train_batch_size", 2)),
        gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 16)),
        learning_rate=float(training_cfg.get("learning_rate", 2e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(training_cfg.get("warmup_ratio", 0.03)),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
        max_grad_norm=float(training_cfg.get("max_grad_norm", 1.0)),
        logging_steps=int(training_cfg.get("logging_steps", 1 if debug_enabled else 10)),
        save_steps=int(training_cfg.get("save_steps", 250)),
        save_total_limit=int(training_cfg.get("save_total_limit", 3)),
        bf16=bool(training_cfg.get("bf16", True)),
        fp16=bool(training_cfg.get("fp16", False)),
        gradient_checkpointing=gradient_checkpointing,
        optim=training_cfg.get("optim", "paged_adamw_8bit"),
        report_to=training_cfg.get("report_to", "none"),
        dataloader_num_workers=int(training_cfg.get("dataloader_num_workers", 0)),
        remove_unused_columns=False,
        logging_first_step=True,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        data_collator=build_lm_collator(tokenizer),
    )

    interrupted = False
    fatal_error: Optional[BaseException] = None
    try:
        trainer.train()
    except KeyboardInterrupt:
        interrupted = True
        LOGGER.warning("Ctrl+C detected. Saving final LoRA adapter...")
    except Exception as exc:
        interrupted = True
        fatal_error = exc
        LOGGER.exception("Training crashed. Saving current LoRA adapter state...")

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    if interrupted:
        LOGGER.info("Interrupted run saved to: %s", output_dir)
    else:
        LOGGER.info("Training complete. Adapter/tokenizer saved to: %s", output_dir)

    if fatal_error is not None:
        raise fatal_error


if __name__ == "__main__":
    main()
