from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from datasets import Dataset, concatenate_datasets, load_dataset
from torch.utils.data import IterableDataset as TorchIterableDataset
from torch.utils.data import get_worker_info

LOGGER = logging.getLogger(__name__)

PROMPT_KEYS = (
    "prompt",
    "instruction",
    "query",
    "caption",
    "description",
    "input",
    "context",
    "title",
    "question",
)

ASCII_KEYS = (
    "ascii",
    "ascii_art",
    "art",
    "output",
    "target",
    "completion",
    "response",
    "text",
    "content",
)


def _resolve_source_limit(source_cfg: Dict[str, Any], samples_default: Optional[int]) -> Optional[int]:
    source_limit = source_cfg.get("max_samples", samples_default)
    return int(source_limit) if source_limit else None


def _as_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip("\r\n")
        return cleaned if cleaned.strip() else None
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        parts = [item for item in value if isinstance(item, str) and item.strip()]
        if parts:
            return "\n".join(parts).strip("\r\n")
    return None


def _pick_first(sample: Dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        if key in sample:
            text = _as_text(sample[key])
            if text:
                return text
    return None


def _guess_from_values(sample: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    values = [_as_text(value) for value in sample.values()]
    values = [value for value in values if value]
    if not values:
        return None, None
    ascii_art = max(values, key=lambda item: (item.count("\n"), len(item)))
    prompt = None
    if len(values) > 1:
        candidates = [item for item in values if item != ascii_art]
        if candidates:
            prompt = min(candidates, key=len)
    return prompt, ascii_art


def normalize_sample(sample: Dict[str, Any], source_name: str, instruction_fallback: str) -> Optional[Dict[str, str]]:
    prompt = _pick_first(sample, PROMPT_KEYS)
    context = _as_text(sample.get("context"))
    ascii_art = _pick_first(sample, ASCII_KEYS)

    if not ascii_art:
        guessed_prompt, guessed_ascii = _guess_from_values(sample)
        ascii_art = guessed_ascii
        if not prompt:
            prompt = guessed_prompt

    if context and context != prompt:
        if prompt:
            prompt = f"{prompt}\n\nContext:\n{context}"
        else:
            prompt = context

    if not ascii_art:
        return None
    if not prompt:
        prompt = instruction_fallback

    return {
        "source": source_name,
        "instruction": prompt.strip(),
        "ascii_art": ascii_art.rstrip("\n"),
    }


def _iter_emporium_blocks(file_path: Path) -> Iterator[str]:
    current_lines: List[str] = []
    empty_streak = 0

    def flush_block() -> Optional[str]:
        if not current_lines:
            return None
        block = "".join(current_lines).strip("\n")
        if block.count("\n") >= 2 and len(block) >= 16:
            return block
        return None

    with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                empty_streak = 0
                current_lines.append(line)
                continue
            empty_streak += 1
            current_lines.append(line)
            if empty_streak >= 2:
                block = flush_block()
                if block:
                    yield block
                current_lines = []
                empty_streak = 0
    block = flush_block()
    if block:
        yield block


def iter_local_emporium_records(source_cfg: Dict[str, Any], samples_limit: Optional[int]) -> Iterator[Dict[str, str]]:
    root = Path(source_cfg["path"])
    if not root.exists():
        if source_cfg.get("optional", False):
            LOGGER.warning("Skipping missing local source: %s", root)
            return
        raise FileNotFoundError(f"Missing local dataset path: {root}")

    pattern = source_cfg.get("glob", "**/*.txt")
    files = sorted(path for path in root.glob(pattern) if path.is_file())
    if not files:
        if source_cfg.get("optional", False):
            LOGGER.warning("No files found in local source: %s (%s)", root, pattern)
            return
        raise FileNotFoundError(f"No files found in {root} with pattern {pattern}")

    yielded = 0
    prefix = source_cfg.get("instruction_prefix", "Create ASCII art in the style of")
    for file_path in files:
        for block in _iter_emporium_blocks(file_path):
            yield {
                "source": source_cfg.get("name", "ASCII ART EMPORIUM"),
                "instruction": f"{prefix} {file_path.stem}".strip(),
                "ascii_art": block,
            }
            yielded += 1
            if samples_limit and yielded >= samples_limit:
                return


def iter_hf_source_records(
    source_cfg: Dict[str, Any],
    cache_dir: Optional[str],
    samples_limit: Optional[int],
    seed: int,
) -> Iterator[Dict[str, str]]:
    source_name = source_cfg["name"]
    split = source_cfg.get("split", "train")
    streaming = bool(source_cfg.get("streaming", False))
    instruction_fallback = source_cfg.get("instruction_fallback", "Generate ASCII art matching the user request.")

    try:
        dataset = load_dataset(
            source_name,
            split=split,
            cache_dir=cache_dir,
            streaming=streaming,
        )
    except Exception:
        if source_cfg.get("optional", False):
            LOGGER.exception("Skipping optional source that failed to load: %s", source_name)
            return
        raise

    if source_cfg.get("shuffle", True):
        if streaming:
            buffer_size = int(source_cfg.get("shuffle_buffer_size", 10000))
            dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)
        else:
            dataset = dataset.shuffle(seed=seed)

    yielded = 0
    for sample in dataset:
        normalized = normalize_sample(sample, source_name, instruction_fallback)
        if normalized:
            yield normalized
            yielded += 1
        if samples_limit and yielded >= samples_limit:
            return


def iter_training_records(dataset_cfg: Dict[str, Any], seed: int, force_streaming: bool = False) -> Iterator[Dict[str, str]]:
    cache_dir = dataset_cfg.get("cache_dir")
    samples_default = dataset_cfg.get("samples_per_source")
    sources = dataset_cfg.get("sources", [])
    if not sources:
        raise ValueError("dataset.sources is empty")

    for source in sources:
        source_type = source.get("type")
        source_name = source.get("name", "unnamed-source")
        source_limit = _resolve_source_limit(source, samples_default)
        LOGGER.info("Streaming source: %s (%s)", source_name, source_type)

        if source_type == "huggingface":
            source_stream_cfg = dict(source)
            if force_streaming:
                source_stream_cfg["streaming"] = True
            iterator = iter_hf_source_records(
                source_cfg=source_stream_cfg,
                cache_dir=cache_dir,
                samples_limit=source_limit,
                seed=seed,
            )
        elif source_type == "local_emporium":
            iterator = iter_local_emporium_records(
                source_cfg=source,
                samples_limit=source_limit,
            )
        else:
            raise ValueError(f"Unsupported dataset source type: {source_type}")

        yielded = 0
        for record in iterator:
            yielded += 1
            yield record
        LOGGER.info("Finished source %s with %s streamed rows", source_name, yielded)


def estimate_training_samples(dataset_cfg: Dict[str, Any]) -> tuple[int, bool]:
    samples_default = dataset_cfg.get("samples_per_source")
    sources = dataset_cfg.get("sources", [])
    total = 0
    has_unknown = False
    for source in sources:
        source_limit = _resolve_source_limit(source, samples_default)
        if source_limit is None:
            has_unknown = True
            continue
        total += source_limit
    return total, has_unknown


def build_training_dataset(dataset_cfg: Dict[str, Any], seed: int) -> Dataset:
    cache_dir = dataset_cfg.get("cache_dir")
    samples_default = dataset_cfg.get("samples_per_source")
    sources = dataset_cfg.get("sources", [])
    if not sources:
        raise ValueError("dataset.sources is empty")

    loaded_sets: List[Dataset] = []
    for source in sources:
        source_type = source.get("type")
        source_name = source.get("name", "unnamed-source")
        source_limit = _resolve_source_limit(source, samples_default)
        LOGGER.info("Loading source in-memory: %s (%s)", source_name, source_type)

        if source_type == "huggingface":
            rows = list(
                iter_hf_source_records(
                    source_cfg=source,
                    cache_dir=cache_dir,
                    samples_limit=source_limit,
                    seed=seed,
                )
            )
        elif source_type == "local_emporium":
            rows = list(iter_local_emporium_records(source_cfg=source, samples_limit=source_limit))
        else:
            raise ValueError(f"Unsupported dataset source type: {source_type}")

        if not rows:
            LOGGER.warning("No usable rows from %s", source_name)
            continue
        ds = Dataset.from_list(rows)
        loaded_sets.append(ds)
        LOGGER.info("Loaded %s rows from %s", len(ds), source_name)

    if not loaded_sets:
        raise RuntimeError("No dataset records loaded. Check dataset.sources.")

    merged = loaded_sets[0] if len(loaded_sets) == 1 else concatenate_datasets(loaded_sets)
    if dataset_cfg.get("shuffle", True):
        merged = merged.shuffle(seed=seed)
    return merged


def format_training_text(instruction: str, ascii_art: str, system_prompt: str) -> str:
    system = (system_prompt or "").strip()
    if system:
        return (
            f"### System\n{system}\n\n"
            f"### User\n{instruction.strip()}\n\n"
            f"### Assistant\n{ascii_art.rstrip()}"
        )
    return f"### Instruction\n{instruction.strip()}\n\n### ASCII Art\n{ascii_art.rstrip()}"


def iter_training_texts(dataset_cfg: Dict[str, Any], seed: int, system_prompt: str) -> Iterator[str]:
    for record in iter_training_records(dataset_cfg=dataset_cfg, seed=seed, force_streaming=True):
        yield format_training_text(
            instruction=record["instruction"],
            ascii_art=record["ascii_art"],
            system_prompt=system_prompt,
        )


class HFTokenStreamingDataset(TorchIterableDataset):
    def __init__(
        self,
        dataset_cfg: Dict[str, Any],
        seed: int,
        tokenizer: Any,
        max_seq_length: int,
        system_prompt: str,
    ) -> None:
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.seed = seed
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.system_prompt = system_prompt
        self.estimated_samples, _ = estimate_training_samples(dataset_cfg)

    def __len__(self) -> int:
        return max(1, int(self.estimated_samples))

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        worker_count = worker.num_workers if worker else 1

        for idx, record in enumerate(
            iter_training_records(
                dataset_cfg=self.dataset_cfg,
                seed=self.seed,
                force_streaming=True,
            )
        ):
            if (idx % worker_count) != worker_id:
                continue

            text = format_training_text(
                instruction=record["instruction"],
                ascii_art=record["ascii_art"],
                system_prompt=self.system_prompt,
            )
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_seq_length,
            )
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]
            yield {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": list(input_ids),
            }

