"""Train Qwen3-4B-Instruct with QLoRA SFT.

Usage:
    python scripts/train_sft.py --config configs/train_sft.yaml
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


@dataclass
class SFTConfig:
    model_name_or_path: str = "Qwen/Qwen3-4B-Instruct-2507"
    dataset_dir: str = "data/processed"
    train_file: str = "train.jsonl"
    max_length: int = 1024
    bf16: bool = True
    load_in_4bit: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2.0e-4
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    logging_steps: int = 20
    save_steps: int = 500
    output_dir: str = "outputs/checkpoints/final"
    max_train_samples: int = 0


def load_config(path: str) -> SFTConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return SFTConfig(**data)


def load_model_and_tokenizer(config: SFTConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if config.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, tokenizer


class SFTDataCollator:
    """Pad input_ids, attention_mask and labels to the longest sequence."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(feature["input_ids"]) for feature in features)
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch["input_ids"].append(
                feature["input_ids"]
                + [self.tokenizer.pad_token_id] * pad_len
            )
            batch["attention_mask"].append(
                feature["attention_mask"] + [0] * pad_len
            )
            batch["labels"].append(feature["labels"] + [-100] * pad_len)
        batch["input_ids"] = torch.tensor(batch["input_ids"])
        batch["attention_mask"] = torch.tensor(batch["attention_mask"])
        batch["labels"] = torch.tensor(batch["labels"])
        return batch


def prepare_dataset(config: SFTConfig, tokenizer):
    train_path = ROOT / config.dataset_dir / config.train_file
    dataset = load_dataset("json", data_files=str(train_path), split="train")
    if config.max_train_samples > 0:
        dataset = dataset.select(range(min(config.max_train_samples, len(dataset))))

    def format_chat(example: dict) -> dict:
        messages = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]
        example["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return example

    def tokenize(example: dict) -> dict:
        tokenized = tokenizer(
            example["text"],
            truncation=True,
            max_length=config.max_length,
        )
        prompt_messages = [{"role": "user", "content": example["instruction"]}]
        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        labels = [-100] * len(tokenized["input_ids"])
        assistant_start = len(prompt_ids)
        if assistant_start < len(labels):
            labels[assistant_start:] = tokenized["input_ids"][assistant_start:]
        tokenized["labels"] = labels
        return tokenized

    dataset = dataset.map(format_chat)
    dataset = dataset.map(
        tokenize,
        remove_columns=dataset.column_names,
    )
    return dataset


def train(config: SFTConfig) -> None:
    model, tokenizer = load_model_and_tokenizer(config)
    dataset = prepare_dataset(config, tokenizer)
    output_dir = ROOT / config.output_dir

    training_args = TrainingArguments(
        output_dir=str(ROOT / config.output_dir),
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        bf16=config.bf16,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        report_to=[],
        save_total_limit=1,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    model.config.use_cache = False
    data_collator = SFTDataCollator(tokenizer)
    checkpoint_dirs = sorted(output_dir.glob("checkpoint-*"))
    resume_from_checkpoint = checkpoint_dirs[-1] if checkpoint_dirs else None
    if resume_from_checkpoint is not None:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_sft.yaml")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
