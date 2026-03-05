from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

from datasets import load_dataset
from huggingface_hub import snapshot_download

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download/cache datasets defined in config.json")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    parser.add_argument(
        "--method",
        type=str,
        choices=["snapshot", "cache"],
        default="snapshot",
        help="snapshot: clone dataset repos locally; cache: warm Hugging Face datasets cache.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/raw/hf",
        help="Output directory for snapshot method.",
    )
    parser.add_argument(
        "--stream-rows",
        type=int,
        default=5000,
        help="Rows to read when source is streaming in cache mode.",
    )
    parser.add_argument(
        "--full-cache",
        action="store_true",
        help="In cache mode, force non-streaming full download for each HF source.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Dataset name filter (repeatable), e.g. --source apehex/ascii-art",
    )
    return parser.parse_args()


def read_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _iter_selected_sources(sources: Iterable[Dict[str, Any]], filters: List[str]) -> Iterable[Dict[str, Any]]:
    selected = set(filters or [])
    for source in sources:
        if source.get("type") != "huggingface":
            continue
        name = str(source.get("name", ""))
        if selected and name not in selected:
            continue
        yield source


def _safe_repo_dir_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def _download_snapshot(source_cfg: Dict[str, Any], out_root: Path) -> Dict[str, Any]:
    repo_id = source_cfg["name"]
    local_dir = out_root / _safe_repo_dir_name(repo_id)
    local_dir.mkdir(parents=True, exist_ok=True)

    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return {
        "name": repo_id,
        "method": "snapshot",
        "status": "ok",
        "path": str(Path(path).resolve()),
    }


def _download_cache(source_cfg: Dict[str, Any], cache_dir: str, stream_rows: int, full_cache: bool) -> Dict[str, Any]:
    repo_id = source_cfg["name"]
    split = source_cfg.get("split", "train")
    source_streaming = bool(source_cfg.get("streaming", False))
    streaming = source_streaming and not full_cache

    if streaming:
        ds = load_dataset(repo_id, split=split, streaming=True, cache_dir=cache_dir)
        read_rows = 0
        for _ in ds:
            read_rows += 1
            if read_rows >= stream_rows:
                break
        return {
            "name": repo_id,
            "method": "cache",
            "status": "ok",
            "streaming": True,
            "rows_read": read_rows,
            "cache_dir": str(Path(cache_dir).resolve()),
        }

    ds = load_dataset(repo_id, split=split, streaming=False, cache_dir=cache_dir)
    rows = len(ds)
    return {
        "name": repo_id,
        "method": "cache",
        "status": "ok",
        "streaming": False,
        "rows_cached": rows,
        "cache_dir": str(Path(cache_dir).resolve()),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args()
    config = read_config(args.config)

    dataset_cfg = config.get("dataset", {})
    cache_dir = dataset_cfg.get("cache_dir", "cache/hf")
    sources = dataset_cfg.get("sources", [])
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    reports: List[Dict[str, Any]] = []
    for source_cfg in _iter_selected_sources(sources, args.source):
        name = source_cfg["name"]
        optional = bool(source_cfg.get("optional", False))
        try:
            LOGGER.info("Downloading source: %s (method=%s)", name, args.method)
            if args.method == "snapshot":
                report = _download_snapshot(source_cfg, out_root=out_root)
            else:
                report = _download_cache(
                    source_cfg,
                    cache_dir=cache_dir,
                    stream_rows=int(args.stream_rows),
                    full_cache=bool(args.full_cache),
                )
            reports.append(report)
            LOGGER.info("Done: %s", name)
        except Exception as exc:
            status = {
                "name": name,
                "method": args.method,
                "status": "failed_optional" if optional else "failed",
                "error": str(exc),
            }
            reports.append(status)
            if optional:
                LOGGER.warning("Optional source failed: %s | %s", name, exc)
            else:
                raise

    # Ensure optional local source path exists for convenience.
    for source_cfg in sources:
        if source_cfg.get("type") != "local_emporium":
            continue
        local_path = Path(source_cfg.get("path", "data/ascii_art_emporium"))
        local_path.mkdir(parents=True, exist_ok=True)
        reports.append(
            {
                "name": source_cfg.get("name", "ASCII ART EMPORIUM"),
                "method": "local_path",
                "status": "ready",
                "path": str(local_path.resolve()),
            }
        )

    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "method": args.method,
        "reports": reports,
    }
    manifest_path = out_root / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("Manifest saved: %s", manifest_path.resolve())


if __name__ == "__main__":
    main()

