from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

LOGGER = logging.getLogger(__name__)


def read_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def to_torch_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    if not name:
        return None
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[key]


def build_quantization_config(cfg: Dict[str, Any], enabled: bool = True) -> Optional[BitsAndBytesConfig]:
    if not enabled or not cfg.get("load_in_4bit", False):
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=cfg.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=to_torch_dtype(cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
    )


def build_prompt(user_prompt: str, system_prompt: str) -> str:
    system = (system_prompt or "").strip()
    if system:
        return (
            f"### System\n{system}\n\n"
            f"### User\n{user_prompt.strip()}\n\n"
            "### Assistant\n"
        )
    return f"### Instruction\n{user_prompt.strip()}\n\n### ASCII Art\n"


def resolve_adapter_path(config: Dict[str, Any], adapter_path: Optional[str]) -> Path:
    if adapter_path:
        return Path(adapter_path)
    return Path(config.get("training", {}).get("output_dir", "outputs/dolphin-ascii-lora"))


def _detect_model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer(
    config: Dict[str, Any],
    adapter_path: Optional[str] = None,
    load_in_4bit: bool = True,
) -> Tuple[Any, Any, str]:
    base_model = config["base_model"]
    trust_remote_code = bool(config.get("trust_remote_code", False))
    adapter_dir = resolve_adapter_path(config, adapter_path)
    adapter_cfg = adapter_dir / "adapter_config.json"

    quant_cfg = build_quantization_config(config.get("quantization", {}), enabled=load_in_4bit)
    dtype = to_torch_dtype(config.get("quantization", {}).get("bnb_4bit_compute_dtype", "bfloat16"))

    tokenizer_source = str(adapter_dir) if adapter_dir.exists() else base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=trust_remote_code,
        quantization_config=quant_cfg,
        torch_dtype=dtype,
        device_map="auto",
    )

    mode = "base"
    if adapter_cfg.exists():
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        mode = "lora"
        LOGGER.info("Loaded LoRA adapter from %s", adapter_dir)
    else:
        LOGGER.warning("No LoRA adapter found at %s. Using base model only.", adapter_dir)

    model.eval()
    return model, tokenizer, mode


def generate_ascii(
    model: Any,
    tokenizer: Any,
    user_prompt: str,
    system_prompt: str,
    generation_config: Dict[str, Any],
    max_new_tokens_override: Optional[int] = None,
) -> str:
    prompt = build_prompt(user_prompt=user_prompt, system_prompt=system_prompt)
    device = _detect_model_device(model)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    max_new_tokens = (
        int(max_new_tokens_override)
        if max_new_tokens_override is not None
        else int(generation_config.get("max_new_tokens", 320))
    )
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=bool(generation_config.get("do_sample", True)),
            temperature=float(generation_config.get("temperature", 0.8)),
            top_p=float(generation_config.get("top_p", 0.92)),
            repetition_penalty=float(generation_config.get("repetition_penalty", 1.05)),
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    result = decoded[len(prompt) :] if decoded.startswith(prompt) else decoded
    return result.rstrip()

