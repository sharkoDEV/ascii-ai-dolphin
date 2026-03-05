from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from statistics import fmean, median
from typing import Any, Dict, List

import matplotlib

from .data import iter_training_records
from .runtime import generate_ascii, load_model_and_tokenizer, read_config

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark LoRA ASCII model with chart.")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    parser.add_argument("--adapter-path", type=str, default=None, help="Path to LoRA adapter.")
    parser.add_argument("--samples", type=int, default=64, help="Number of benchmark prompts.")
    parser.add_argument("--benchmark-dir", type=str, default=None, help="Output benchmark directory.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override generation max_new_tokens.")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit loading.")
    return parser.parse_args()


def _ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    allowed = 0
    for char in text:
        code = ord(char)
        if char in {"\n", "\r", "\t"} or 32 <= code <= 126:
            allowed += 1
    return allowed / len(text)


def _line_stats(text: str) -> tuple[int, int]:
    lines = text.splitlines()
    if not lines:
        return 0, 0
    return len(lines), max((len(line) for line in lines), default=0)


def _ratio(a: int, b: int) -> float:
    if a <= 0 and b <= 0:
        return 1.0
    if a <= 0 or b <= 0:
        return 0.0
    return min(a, b) / max(a, b)


def _sample_metrics(reference: str, prediction: str) -> Dict[str, float]:
    similarity = SequenceMatcher(None, reference, prediction).ratio()
    ascii_ratio = _ascii_ratio(prediction)
    ref_lines, ref_width = _line_stats(reference)
    pred_lines, pred_width = _line_stats(prediction)
    line_ratio = _ratio(ref_lines, pred_lines)
    width_ratio = _ratio(ref_width, pred_width)
    quality = 0.60 * similarity + 0.20 * ascii_ratio + 0.10 * line_ratio + 0.10 * width_ratio
    passed = quality >= 0.55 and ascii_ratio >= 0.98
    return {
        "similarity": similarity,
        "ascii_ratio": ascii_ratio,
        "line_ratio": line_ratio,
        "width_ratio": width_ratio,
        "quality_score": quality,
        "pass": 1.0 if passed else 0.0,
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {
            "samples": 0,
            "avg_similarity": 0.0,
            "median_similarity": 0.0,
            "avg_ascii_ratio": 0.0,
            "avg_quality_score": 0.0,
            "pass_rate": 0.0,
        }
    sims = [float(row["similarity"]) for row in rows]
    ascii_ratios = [float(row["ascii_ratio"]) for row in rows]
    qualities = [float(row["quality_score"]) for row in rows]
    passes = [float(row["pass"]) for row in rows]
    return {
        "samples": len(rows),
        "avg_similarity": fmean(sims),
        "median_similarity": float(median(sims)),
        "avg_ascii_ratio": fmean(ascii_ratios),
        "avg_quality_score": fmean(qualities),
        "pass_rate": fmean(passes),
    }


def save_chart(rows: List[Dict[str, Any]], summary: Dict[str, float], output_path: Path) -> None:
    similarities = [float(row["similarity"]) for row in rows]
    qualities = [float(row["quality_score"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    labels = ["Similarity", "ASCII Ratio", "Quality", "Pass Rate"]
    values = [
        summary["avg_similarity"],
        summary["avg_ascii_ratio"],
        summary["avg_quality_score"],
        summary["pass_rate"],
    ]
    colors = ["#2a9d8f", "#264653", "#f4a261", "#e76f51"]
    axes[0].bar(labels, values, color=colors)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Aggregate Metrics")
    for i, val in enumerate(values):
        axes[0].text(i, min(0.98, val + 0.03), f"{val:.2f}", ha="center", va="bottom")

    axes[1].hist(similarities, bins=12, alpha=0.7, color="#457b9d", label="Similarity")
    axes[1].hist(qualities, bins=12, alpha=0.6, color="#e9c46a", label="Quality")
    axes[1].set_xlim(0, 1)
    axes[1].set_title("Distribution")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def save_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    fields = [
        "index",
        "source",
        "instruction",
        "similarity",
        "ascii_ratio",
        "line_ratio",
        "width_ratio",
        "quality_score",
        "pass",
        "prediction",
        "reference",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def benchmark() -> Dict[str, Any]:
    args = parse_args()
    config = read_config(args.config)
    model, tokenizer, model_mode = load_model_and_tokenizer(
        config=config,
        adapter_path=args.adapter_path,
        load_in_4bit=not args.no_4bit,
    )

    benchmark_dir = (
        Path(args.benchmark_dir)
        if args.benchmark_dir
        else Path("benchmarks") / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    dataset_cfg = config.get("dataset", {})
    seed = int(config.get("seed", 42))
    rows: List[Dict[str, Any]] = []

    for idx, record in enumerate(iter_training_records(dataset_cfg=dataset_cfg, seed=seed, force_streaming=True)):
        if idx >= args.samples:
            break
        instruction = str(record.get("instruction", "")).strip()
        reference = str(record.get("ascii_art", ""))
        source = str(record.get("source", "unknown"))
        if not instruction:
            continue

        prediction = generate_ascii(
            model=model,
            tokenizer=tokenizer,
            user_prompt=instruction,
            system_prompt=config.get("system_prompt", ""),
            generation_config=config.get("generation", {}),
            max_new_tokens_override=args.max_new_tokens,
        )
        metrics = _sample_metrics(reference=reference, prediction=prediction)
        rows.append(
            {
                "index": idx,
                "source": source,
                "instruction": instruction,
                "prediction": prediction,
                "reference": reference,
                **metrics,
            }
        )
        if (idx + 1) % 5 == 0:
            LOGGER.info("Benchmark progress: %s/%s", idx + 1, args.samples)

    summary = summarize(rows)
    status = "good" if summary["pass_rate"] >= 0.70 and summary["avg_similarity"] >= 0.40 else "needs_work"
    report = {
        "status": status,
        "model_mode": model_mode,
        "summary": summary,
        "adapter_path": args.adapter_path or config.get("training", {}).get("output_dir", "outputs/dolphin-ascii-lora"),
    }

    metrics_file = benchmark_dir / "metrics.json"
    samples_file = benchmark_dir / "samples.csv"
    chart_file = benchmark_dir / "benchmark.png"
    metrics_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_csv(rows, samples_file)
    save_chart(rows, summary, chart_file)

    LOGGER.info("Benchmark complete | status=%s", status)
    LOGGER.info("metrics: %s", metrics_file)
    LOGGER.info("samples: %s", samples_file)
    LOGGER.info("chart: %s", chart_file)
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    report = benchmark()
    summary = report["summary"]
    print(
        f"status={report['status']} | samples={summary['samples']} | "
        f"avg_similarity={summary['avg_similarity']:.3f} | "
        f"avg_quality={summary['avg_quality_score']:.3f} | pass_rate={summary['pass_rate']:.3f}"
    )


if __name__ == "__main__":
    main()

