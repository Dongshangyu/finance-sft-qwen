"""Merge per-worker predictions and build the base vs SFT report.

Usage:
    python scripts/merge_predictions.py --num-workers 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate import build_report, load_config, load_predictions  # noqa: E402


def merge_worker_predictions(output_dir: Path, num_workers: int) -> None:
    """Merge per-worker prediction files into single base/SFT files."""
    for name in ("base_predictions", "sft_predictions"):
        merged_path = output_dir / f"{name}.jsonl"
        with merged_path.open("w", encoding="utf-8") as merged:
            for worker_index in range(num_workers):
                worker_path = output_dir / f"{name}_worker{worker_index}.jsonl"
                if not worker_path.exists():
                    raise RuntimeError(f"missing {worker_path}")
                merged.write(worker_path.read_text(encoding="utf-8"))
        print(f"merged {name}: {merged_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--config", default="configs/eval.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    output_dir = ROOT / config.output_dir
    merge_worker_predictions(output_dir, args.num_workers)
    build_report(
        output_dir / "base_predictions.jsonl",
        output_dir / "sft_predictions.jsonl",
        ROOT / config.report_path,
    )


if __name__ == "__main__":
    main()
