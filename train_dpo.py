from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED_KEYS = {"prompt", "chosen", "rejected"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small DPO alignment model.")
    parser.add_argument(
        "--model-name",
        default="sshleifer/tiny-gpt2",
        help="Base model name or local path used for actor and reference models.",
    )
    parser.add_argument(
        "--dataset-path",
        default="data/hhh_preferences.jsonl",
        help="Path to the JSONL preference dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/dpo-model",
        help="Directory used to save the trained model.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO beta parameter.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length used by DPOTrainer.",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=128,
        help="Maximum prompt length used by DPOTrainer.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=1.0,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=1,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-6,
        help="Learning rate for DPO training.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=1,
        help="Logging interval in steps.",
    )
    parser.add_argument(
        "--optim",
        default="paged_adamw_32bit",
        help="Optimizer name used by DPOConfig.",
    )
    return parser.parse_args()


def validate_dataset_path(dataset_path: Path) -> None:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")


def validate_record(record: dict, line_number: int) -> None:
    keys = set(record.keys())
    if keys != REQUIRED_KEYS:
        raise ValueError(
            f"Invalid keys on line {line_number}: expected {sorted(REQUIRED_KEYS)}, got {sorted(keys)}"
        )

    for key in REQUIRED_KEYS:
        value = record[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Field '{key}' on line {line_number} must be a non-empty string.")


def load_preference_dataset(dataset_path: Path) -> Dataset:
    from datasets import load_dataset

    validate_dataset_path(dataset_path)

    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            validate_record(json.loads(line), line_number)

    dataset_dict = load_dataset("json", data_files=str(dataset_path), split="train")
    if set(dataset_dict.column_names) != REQUIRED_KEYS:
        raise ValueError(
            f"Dataset columns must be exactly {sorted(REQUIRED_KEYS)}. Got {dataset_dict.column_names}."
        )
    if len(dataset_dict) < 30:
        raise ValueError("Dataset must contain at least 30 preference pairs.")
    return dataset_dict


def load_tokenizer(model_name: str) -> AutoTokenizer:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def model_kwargs() -> dict:
    import torch

    kwargs = {}
    if torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.float16
    return kwargs


def main() -> None:
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM
    from trl import DPOConfig, DPOTrainer

    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_preference_dataset(dataset_path)
    tokenizer = load_tokenizer(args.model_name)

    actor_model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs())
    reference_model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs())

    training_args = DPOConfig(
        output_dir=str(output_dir),
        beta=args.beta,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        eval_strategy="no",
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        remove_unused_columns=False,
        optim=args.optim,
        report_to=[],
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    )

    trainer = DPOTrainer(
        model=actor_model,
        ref_model=reference_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"Loaded dataset with {len(dataset)} preference pairs.")
    print(f"Training actor and reference models from: {args.model_name}")
    print(f"Using beta={args.beta}")

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"Training complete. Model saved to {output_dir}")


if __name__ == "__main__":
    main()
