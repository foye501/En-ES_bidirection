import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen for EN-ES medical translation with LoRA.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train-file", default="data/finetune/train.jsonl")
    parser.add_argument("--validation-file", default="data/finetune/validation.jsonl")
    parser.add_argument("--output-dir", default="artifacts/qwen-en-es-lora")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="Use a small value such as 20 for a smoke run.")
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=512)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--use-4bit", action="store_true", help="CUDA only; requires bitsandbytes.")
    return parser.parse_args()


def has_cuda_bf16():
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def tokenize_chat(example, tokenizer, max_seq_length):
    messages = example["messages"]
    prompt_messages = messages[:-1]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )

    full = tokenizer(full_text, max_length=max_seq_length, truncation=True)
    prompt = tokenizer(prompt_text, max_length=max_seq_length, truncation=True)
    labels = list(full["input_ids"])
    prompt_len = min(len(prompt["input_ids"]), len(labels))
    labels[:prompt_len] = [-100] * prompt_len

    full["labels"] = labels
    return full


def has_supervised_tokens(example):
    return any(label != -100 for label in example["labels"])


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    device_map = None
    if args.use_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("--use-4bit requires CUDA. On Apple Silicon, omit --use-4bit.")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        device_map = {"": local_rank} if world_size > 1 else "auto"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if has_cuda_bf16() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if has_cuda_bf16() else "auto",
        device_map=device_map,
        quantization_config=quantization_config,
    )
    model.config.use_cache = False
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset(
        "json",
        data_files={"train": args.train_file, "validation": args.validation_file},
    )
    if args.max_train_samples:
        dataset["train"] = dataset["train"].select(range(min(args.max_train_samples, len(dataset["train"]))))
    if args.max_eval_samples:
        dataset["validation"] = dataset["validation"].select(
            range(min(args.max_eval_samples, len(dataset["validation"])))
        )

    tokenized = dataset.map(
        lambda example: tokenize_chat(example, tokenizer, args.max_seq_length),
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )
    for split in tokenized:
        before = len(tokenized[split])
        tokenized[split] = tokenized[split].filter(
            has_supervised_tokens,
            desc=f"Filtering {split} rows without assistant labels",
        )
        removed = before - len(tokenized[split])
        if removed:
            print(f"Removed {removed} {split} rows truncated before assistant labels.")
    if len(tokenized["train"]) == 0:
        raise ValueError("No train rows contain assistant labels. Increase --max-seq-length.")
    if len(tokenized["validation"]) == 0:
        raise ValueError("No validation rows contain assistant labels. Increase --max-seq-length.")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch",
        bf16=has_cuda_bf16(),
        fp16=torch.cuda.is_available() and not has_cuda_bf16(),
        dataloader_num_workers=args.dataloader_num_workers,
        ddp_find_unused_parameters=False,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, pad_to_multiple_of=8),
    )
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"Saved LoRA adapter and tokenizer to {output_dir / 'final'}")


if __name__ == "__main__":
    main()
