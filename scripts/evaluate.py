"""Evaluate base and SFT models on the held-out eval set.

Usage:
    python scripts/evaluate.py --config configs/eval.yaml

The script generates predictions once, stores them as JSONL, then computes a
base vs SFT comparison report from the saved prediction files.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from src.evaluator import aggregate_scores, score_one  # noqa: E402
from src.utils import write_jsonl, write_text  # noqa: E402


@dataclass
class EvalConfig:
    model_name_or_path: str = "Qwen/Qwen3-4B-Instruct-2507"
    adapter_path: str = "outputs/checkpoints/final"
    eval_file: str = "data/processed/eval.jsonl"
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    num_samples: int = 2000
    batch_size: int = 8
    output_dir: str = "outputs/predictions"
    report_path: str = "outputs/reports/eval_report.md"


def load_config(path: str) -> EvalConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return EvalConfig(**data)


def load_generator(
    model_name_or_path: str,
    adapter_path: str | None = None,
    device: str = "auto",
):
    """Load a model for generation, optionally with the trained LoRA adapter."""
    quantization_config = None
    if device != "cpu":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        device_map="auto" if device == "auto" else device,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_path is not None:
        adapter_path = str(ROOT / adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    model.config.use_cache = True
    return model, tokenizer


def sample_eval_rows(eval_file: str, num_samples: int, seed: int = 42) -> list[dict]:
    rows = load_dataset("json", data_files=str(ROOT / eval_file), split="train")
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    selected = indices[:num_samples]
    return [rows[i] for i in selected]


def generate_predictions(
    rows: list[dict],
    model,
    tokenizer,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    batch_size: int,
    output_path: Path,
) -> None:
    """Generate answers for every row and save them incrementally."""
    existing_ids: set[str] = set()
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing_ids.add(json.loads(line)["id"])

    rows_to_generate = [row for row in rows if row["id"] not in existing_ids]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a", encoding="utf-8") as f:
        for start in range(0, len(rows_to_generate), batch_size):
            batch_rows = rows_to_generate[start : start + batch_size]
            batch_texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["instruction"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for row in batch_rows
            ]
            batch_inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(model.device)
            with torch.no_grad():
                output_ids = model.generate(
                    **batch_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            for batch_offset, row in enumerate(batch_rows):
                prompt_len = batch_inputs["input_ids"].shape[1]
                new_tokens = output_ids[batch_offset][prompt_len:]
                prediction = tokenizer.decode(
                    new_tokens,
                    skip_special_tokens=True,
                ).strip()
                f.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            "instruction": row["instruction"],
                            "reference": row["output"],
                            "prediction": prediction,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                index = start + batch_offset + 1
                if index % 100 == 0 or index == len(rows_to_generate):
                    print(f"generated {index}/{len(rows_to_generate)}")
            f.flush()


def load_predictions(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def prediction_file_is_complete(path: Path, expected_ids: set[str]) -> bool:
    """A prediction file is complete only when it covers every expected sample."""
    if not path.exists():
        return False
    existing_ids = {row["id"] for row in load_predictions(path)}
    return expected_ids.issubset(existing_ids)


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


def build_report(base_path: Path, sft_path: Path, report_path: Path) -> None:
    base_rows = load_predictions(base_path)
    sft_rows = load_predictions(sft_path)
    if not base_rows or not sft_rows:
        raise RuntimeError("Predictions are missing; run generation first.")

    if len(base_rows) != len(sft_rows):
        raise RuntimeError(
            f"Prediction count mismatch: base={len(base_rows)}, SFT={len(sft_rows)}"
        )

    base_scored = []
    sft_scored = []
    for base_row, sft_row in zip(base_rows, sft_rows):
        base_scored.append(
            {
                **base_row,
                **score_one(base_row["prediction"], base_row["reference"]),
            }
        )
        sft_scored.append(
            {
                **sft_row,
                **score_one(sft_row["prediction"], sft_row["reference"]),
            }
        )

    base_metrics = aggregate_scores(base_scored)
    sft_metrics = aggregate_scores(sft_scored)

    lines = [
        "# Base vs SFT Evaluation Report",
        "",
        f"- eval samples: {int(base_metrics['count'])}",
        f"- base predictions: `{base_path}`",
        f"- SFT predictions: `{sft_path}`",
        "",
        "| metric | base | SFT | delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    metric_names = [
        ("rouge_l_f1", "ROUGE-L F1"),
        ("rouge_l_precision", "ROUGE-L Precision"),
        ("rouge_l_recall", "ROUGE-L Recall"),
        ("bleu", "BLEU"),
        ("reference_hit", "Reference Hit"),
        ("mean_prediction_len", "Mean Prediction Length"),
        ("empty_predictions", "Empty Predictions"),
        ("short_predictions", "Short Predictions"),
    ]
    for key, label in metric_names:
        base_value = base_metrics[key]
        sft_value = sft_metrics[key]
        delta = sft_value - base_value
        lines.append(
            f"| {label} | {base_value:.4f} | {sft_value:.4f} | {delta:+.4f} |"
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_text(report_path, "\n".join(lines) + "\n")
    print(f"report written to {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()
    config = load_config(args.config)

    output_dir = ROOT / config.output_dir
    worker_suffix = f"_worker{args.worker_index}" if args.num_workers > 1 else ""
    base_path = output_dir / f"base_predictions{worker_suffix}.jsonl"
    sft_path = output_dir / f"sft_predictions{worker_suffix}.jsonl"

    rows = sample_eval_rows(config.eval_file, config.num_samples)
    if args.num_workers > 1:
        worker_rows = [
            row
            for index, row in enumerate(rows)
            if index % args.num_workers == args.worker_index
        ]
        rows = worker_rows
    expected_ids = {row["id"] for row in rows}
    print(f"eval rows: {len(rows)}")

    if not prediction_file_is_complete(base_path, expected_ids):
        print("loading base model")
        model, tokenizer = load_generator(config.model_name_or_path)
        generate_predictions(
            rows,
            model,
            tokenizer,
            config.max_new_tokens,
            config.temperature,
            config.top_p,
            config.batch_size,
            base_path,
        )
        del model
        torch.cuda.empty_cache()

    if not prediction_file_is_complete(sft_path, expected_ids):
        print("loading SFT model")
        model, tokenizer = load_generator(
            config.model_name_or_path,
            config.adapter_path,
        )
        generate_predictions(
            rows,
            model,
            tokenizer,
            config.max_new_tokens,
            config.temperature,
            config.top_p,
            config.batch_size,
            sft_path,
        )

    if args.num_workers > 1:
        print(
            "worker done. When all workers finish, run "
            "scripts/merge_predictions.py --num-workers 4"
        )
    else:
        build_report(base_path, sft_path, ROOT / config.report_path)


if __name__ == "__main__":
    main()
