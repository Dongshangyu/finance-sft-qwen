"""Data loading, filtering, and splitting helpers."""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a UTF-8 JSONL file into a list of dicts."""
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def filter_zh(rows: list[dict]) -> list[dict]:
    """Keep only rows whose lang field is zh."""
    return [row for row in rows if row.get("lang") == "zh"]


def filter_by_deita_score(rows: list[dict], min_score: float = 6.0) -> list[dict]:
    """Keep rows whose quality score is at least min_score."""
    return [row for row in rows if row.get("deita_score", 0.0) >= min_score]


def clean_output(text: str) -> str:
    """Remove common synthetic answer markers and whole-text angle wrappers."""
    text = text.strip()
    prefix_patterns = [
        r"^<\s*回答\s*>\s*[:：]?\s*",
        r"^%ANSWER%\s*[:：]?\s*",
        r"^%QUERY%\s*[:：]?\s*",
        r"^答案\s*[:：]\s*",
        r"^回答\s*[:：]\s*",
        r"^答\s*[:：]\s*",
        r"^Answer\s*[:：]\s*",
    ]

    changed = True
    while changed:
        changed = False
        for pattern in prefix_patterns:
            new_text = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE)
            if new_text != text:
                text = new_text.strip()
                changed = True

    if text.startswith("<") and text.endswith(">"):
        inner = text[1:-1].strip()
        if inner:
            text = inner

    return text.strip()


def conversation_to_alpaca(row: dict) -> dict | None:
    """Convert BAAI conversations to instruction/input/output format."""
    conversations = row.get("conversations", [])
    human = ""
    assistant = ""
    for item in conversations:
        if item.get("from") == "human":
            human = item.get("value", "").strip()
        elif item.get("from") == "gpt":
            assistant = clean_output(item.get("value", ""))

    if not human or not assistant:
        return None

    return {
        "id": row.get("id", ""),
        "instruction": human,
        "input": "",
        "output": assistant,
        "deita_score": row.get("deita_score", 0.0),
        "length": row.get("length", 0),
    }


def deduplicate(rows: list[dict]) -> list[dict]:
    """Deduplicate by normalized instruction + output."""
    seen = set()
    result = []
    for row in rows:
        alpaca = conversation_to_alpaca(row)
        if alpaca is None:
            continue
        key_text = alpaca["instruction"] + "\x00" + alpaca["output"]
        key = hashlib.sha256(key_text.encode("utf-8")).hexdigest()
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result


def split_train_dev_eval(
    rows: list[dict],
    train_size: int,
    dev_size: int,
    eval_size: int,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Deterministically shuffle and split rows."""
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    train = shuffled[:train_size]
    dev = shuffled[train_size : train_size + dev_size]
    eval_rows = shuffled[train_size + dev_size : train_size + dev_size + eval_size]
    return train, dev, eval_rows
