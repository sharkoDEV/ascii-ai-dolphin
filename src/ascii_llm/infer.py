from __future__ import annotations

import argparse
import logging
from typing import Optional

from .runtime import generate_ascii, load_model_and_tokenizer, read_config

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for dolphin LoRA ASCII model.")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    parser.add_argument("--adapter-path", type=str, default=None, help="Path to LoRA adapter directory.")
    parser.add_argument("--prompt", type=str, default=None, help="One-shot prompt.")
    parser.add_argument("--interactive", action="store_true", help="Interactive terminal mode.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override generation max_new_tokens.")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit loading.")
    return parser.parse_args()


def _run_single(prompt: str, model, tokenizer, config, max_new_tokens: Optional[int]) -> None:
    output = generate_ascii(
        model=model,
        tokenizer=tokenizer,
        user_prompt=prompt,
        system_prompt=config.get("system_prompt", ""),
        generation_config=config.get("generation", {}),
        max_new_tokens_override=max_new_tokens,
    )
    print(output)


def _run_interactive(model, tokenizer, config, max_new_tokens: Optional[int]) -> None:
    print("Interactive mode ready. Type 'exit' to quit.")
    while True:
        try:
            prompt = input("\nPrompt> ").strip()
        except EOFError:
            print("\nBye.")
            break
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit", "q"}:
            print("Bye.")
            break
        print("\n--- ASCII OUTPUT ---")
        _run_single(prompt=prompt, model=model, tokenizer=tokenizer, config=config, max_new_tokens=max_new_tokens)
        print("\n--------------------")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args()
    config = read_config(args.config)

    if not args.interactive and not args.prompt:
        raise ValueError("Use --prompt for one-shot mode, or --interactive for terminal mode.")

    model, tokenizer, mode = load_model_and_tokenizer(
        config=config,
        adapter_path=args.adapter_path,
        load_in_4bit=not args.no_4bit,
    )
    LOGGER.info("Model mode: %s", mode)

    if args.interactive:
        _run_interactive(
            model=model,
            tokenizer=tokenizer,
            config=config,
            max_new_tokens=args.max_new_tokens,
        )
        return

    _run_single(
        prompt=args.prompt,
        model=model,
        tokenizer=tokenizer,
        config=config,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()

