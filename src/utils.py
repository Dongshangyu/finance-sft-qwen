"""Shared file and logging utilities."""

from __future__ import annotations

import json
from pathlib import Path


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    """Write a list of dicts as UTF-8 JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_text(path: str | Path, text: str) -> None:
    """Write UTF-8 text, creating parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
