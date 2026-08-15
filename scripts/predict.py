"""Load a model and run interactive predictions.

Usage:
    python scripts/predict.py                       # base model
    python scripts/predict.py --adapter outputs/checkpoints/final
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts.evaluate import load_config, load_generator  # noqa: E402


def clean_instruction(text: str) -> str:
    """Remove invalid surrogates that may come from terminal input."""
    return "".join(
        ch for ch in text if not ("\ud800" <= ch <= "\udfff")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument(
        "--adapter",
        default=None,
        help="Optional LoRA adapter directory, e.g. outputs/checkpoints/final",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto or cpu",
    )
    args = parser.parse_args()
    config = load_config(args.config)

    model, tokenizer = load_generator(
        config.model_name_or_path,
        args.adapter,
        args.device,
    )
    print("model loaded. Type /exit to quit.")

    while True:
        try:
            instruction = input("\nQ: ").strip()
        except EOFError:
            print()
            break
        instruction = clean_instruction(instruction)
        if not instruction:
            continue
        if instruction == "/exit":
            break

        messages = [{"role": "user", "content": instruction}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=config.temperature,
                top_p=config.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][prompt_len:]
        answer = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()
        print(f"\nA: {answer}")


if __name__ == "__main__":
    main()
