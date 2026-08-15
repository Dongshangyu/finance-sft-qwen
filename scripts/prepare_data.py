"""Prepare BAAI Chinese financial instruction data.

Usage:
    python scripts/prepare_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import (
    conversation_to_alpaca,
    deduplicate,
    filter_by_deita_score,
    filter_zh,
    load_jsonl,
    split_train_dev_eval,
)
from src.utils import write_jsonl, write_text


RAW_PATH = ROOT / "data" / "raw" / "industry_instruction_train.jsonl"
OUTPUT_DIR = ROOT / "data" / "processed"
MIN_DEITA_SCORE = 6.0


def to_alpaca_rows(rows: list[dict]) -> list[dict]:
    alpaca_rows = []
    for row in rows:
        alpaca = conversation_to_alpaca(row)
        if alpaca is not None:
            alpaca_rows.append(alpaca)
    return alpaca_rows


def build_report(
    raw_count: int,
    zh_count: int,
    dedup_count: int,
    quality_count: int,
    train: list[dict],
    dev: list[dict],
    eval_rows: list[dict],
) -> str:
    sample = train[:3]
    sample_text = "\n\n".join(
        f"### 示例 {i + 1}\n\n"
        f"问题：{item['instruction']}\n\n"
        f"答案：{item['output'][:300]}"
        for i, item in enumerate(sample)
    )
    return f"""# 数据准备报告

- 原始条数：{raw_count}
- 中文子集：{zh_count}
- 去重后条数：{dedup_count}
- 质量过滤后条数：{quality_count}
- 训练集：{len(train)}
- 验证集：{len(dev)}
- 评测集：{len(eval_rows)}

## 样例

{sample_text}
"""


def main() -> None:
    raw = load_jsonl(RAW_PATH)
    zh = filter_zh(raw)
    dedup = deduplicate(zh)
    quality = filter_by_deita_score(dedup, MIN_DEITA_SCORE)
    train, dev, eval_rows = split_train_dev_eval(quality, 30000, 2000, 2000)

    train_alpaca = to_alpaca_rows(train)
    dev_alpaca = to_alpaca_rows(dev)
    eval_alpaca = to_alpaca_rows(eval_rows)

    write_jsonl(OUTPUT_DIR / "train.jsonl", train_alpaca)
    write_jsonl(OUTPUT_DIR / "dev.jsonl", dev_alpaca)
    write_jsonl(OUTPUT_DIR / "eval.jsonl", eval_alpaca)

    report = build_report(
        len(raw),
        len(zh),
        len(dedup),
        len(quality),
        train_alpaca,
        dev_alpaca,
        eval_alpaca,
    )
    write_text(ROOT / "outputs" / "reports" / "data_report.md", report)

    print(f"raw={len(raw)} zh={len(zh)} dedup={len(dedup)} quality={len(quality)}")
    print(f"train={len(train_alpaca)} dev={len(dev_alpaca)} eval={len(eval_alpaca)}")
    print(f"report={ROOT / 'outputs' / 'reports' / 'data_report.md'}")


if __name__ == "__main__":
    main()
