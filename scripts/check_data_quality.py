"""Inspect the processed data before training.

Usage:
    python scripts/check_data_quality.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_jsonl
from src.utils import write_text


PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_PATH = ROOT / "outputs" / "reports" / "data_quality_report.md"


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    index = (len(sorted_values) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def describe(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "min": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "max": 0.0,
        }
    sorted_values = sorted(values)
    return {
        "count": len(values),
        "mean": round(statistics.mean(values), 3),
        "min": round(sorted_values[0], 3),
        "p25": round(percentile(sorted_values, 0.25), 3),
        "p50": round(percentile(sorted_values, 0.50), 3),
        "p75": round(percentile(sorted_values, 0.75), 3),
        "max": round(sorted_values[-1], 3),
    }


def dedup_count(rows: list[dict]) -> int:
    seen = set()
    for row in rows:
        key = (row.get("instruction", ""), row.get("output", ""))
        seen.add(key)
    return len(seen)


def build_report(
    train: list[dict],
    dev: list[dict],
    eval_rows: list[dict],
) -> str:
    lines = [
        "# 数据质量报告",
        "",
        "## 数据量",
        "",
        "| 集合 | 条数 | 去重后条数 |",
        "| --- | ---: | ---: |",
        f"| train | {len(train)} | {dedup_count(train)} |",
        f"| dev | {len(dev)} | {dedup_count(dev)} |",
        f"| eval | {len(eval_rows)} | {dedup_count(eval_rows)} |",
        "",
        "## deita_score 分布",
        "",
        "| 集合 | 均值 | min | p25 | p50 | p75 | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, rows in [("train", train), ("dev", dev), ("eval", eval_rows)]:
        stats = describe([row.get("deita_score", 0.0) for row in rows])
        lines.append(
            f"| {name} | {stats['mean']} | {stats['min']} | "
            f"{stats['p25']} | {stats['p50']} | {stats['p75']} | {stats['max']} |"
        )

    lines.extend(
        [
            "",
            "## 答案长度分布（字符数）",
            "",
            "| 集合 | 均值 | min | p25 | p50 | p75 | max |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, rows in [("train", train), ("dev", dev), ("eval", eval_rows)]:
        stats = describe([len(row.get("output", "")) for row in rows])
        lines.append(
            f"| {name} | {stats['mean']} | {stats['min']} | "
            f"{stats['p25']} | {stats['p50']} | {stats['p75']} | {stats['max']} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    train = load_jsonl(PROCESSED_DIR / "train.jsonl")
    dev = load_jsonl(PROCESSED_DIR / "dev.jsonl")
    eval_rows = load_jsonl(PROCESSED_DIR / "eval.jsonl")

    report = build_report(train, dev, eval_rows)
    write_text(REPORT_PATH, report)

    print(f"train={len(train)} dev={len(dev)} eval={len(eval_rows)}")
    print(f"report={REPORT_PATH}")


if __name__ == "__main__":
    main()
