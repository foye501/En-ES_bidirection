import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample small representative calibration text for LLM quantization."
    )
    parser.add_argument("--input", default="data/finetune_bidirectional/train.jsonl")
    parser.add_argument("--output-dir", default="data/quantization/qwen_calibration_128")
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        default=None,
        help="Optional tokenizer path/name. When set, prompts use the model chat template.",
    )
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Disable balancing by metadata.direction.",
    )
    return parser.parse_args()


def load_tokenizer(model_name):
    if not model_name:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def iter_jsonl(path):
    with path.open("r", encoding="utf-8") as src:
        for index, line in enumerate(src):
            if line.strip():
                yield index, json.loads(line)


def get_direction(row):
    metadata = row.get("metadata") or {}
    if metadata.get("direction"):
        return metadata["direction"]

    messages = row.get("messages") or []
    user_text = "\n".join(message.get("content", "") for message in messages if message.get("role") == "user")
    if "Spanish to English" in user_text or "Spanish:" in user_text:
        return "es-en"
    if "English to Spanish" in user_text or "English:" in user_text:
        return "en-es"
    return "unknown"


def split_messages(row):
    messages = row.get("messages") or []
    if messages and messages[-1].get("role") == "assistant":
        return messages[:-1], messages[-1].get("content", "")
    return messages, ""


def fallback_chat_text(messages, add_generation_prompt=False):
    role_names = {
        "system": "System",
        "user": "User",
        "assistant": "Assistant",
    }
    parts = []
    for message in messages:
        role = role_names.get(message.get("role", ""), message.get("role", "Message").title())
        parts.append(f"{role}: {message.get('content', '')}")
    if add_generation_prompt:
        parts.append("Assistant:")
    return "\n".join(parts)


def format_texts(row, tokenizer):
    prompt_messages, reference = split_messages(row)
    full_messages = list(prompt_messages)
    if reference:
        full_messages.append({"role": "assistant", "content": reference})

    if tokenizer is not None:
        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(full_messages, tokenize=False)
    else:
        prompt = fallback_chat_text(prompt_messages, add_generation_prompt=True)
        full_text = fallback_chat_text(full_messages)

    return prompt, full_text, reference


def flatten_text(text):
    return " ".join(text.split())


def choose_samples(rows, sample_size, seed, balance):
    rng = random.Random(seed)
    if not balance:
        chosen = rng.sample(rows, min(sample_size, len(rows)))
        return sorted(chosen, key=lambda item: item[0])

    grouped = defaultdict(list)
    for item in rows:
        grouped[get_direction(item[1])].append(item)

    directions = sorted(grouped)
    if not directions:
        return []

    base = sample_size // len(directions)
    remainder = sample_size % len(directions)
    chosen = []
    for offset, direction in enumerate(directions):
        group = grouped[direction]
        target = base + (1 if offset < remainder else 0)
        chosen.extend(rng.sample(group, min(target, len(group))))

    if len(chosen) < min(sample_size, len(rows)):
        chosen_ids = {index for index, _row in chosen}
        remaining = [item for item in rows if item[0] not in chosen_ids]
        needed = min(sample_size, len(rows)) - len(chosen)
        chosen.extend(rng.sample(remaining, min(needed, len(remaining))))

    return sorted(chosen, key=lambda item: item[0])


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = list(iter_jsonl(input_path))
    sampled = choose_samples(rows, args.sample_size, args.seed, not args.no_balance)
    tokenizer = load_tokenizer(args.model)

    jsonl_path = output_dir / "calibration_prompts.jsonl"
    prompt_txt_path = output_dir / "calibration_prompts.txt"
    full_txt_path = output_dir / "calibration_full_texts.txt"
    stats_path = output_dir / "stats.json"

    direction_counts = defaultdict(int)
    with jsonl_path.open("w", encoding="utf-8") as jsonl_out, prompt_txt_path.open(
        "w", encoding="utf-8"
    ) as prompt_out, full_txt_path.open("w", encoding="utf-8") as full_out:
        for sample_id, (source_index, row) in enumerate(sampled):
            prompt, full_text, reference = format_texts(row, tokenizer)
            direction = get_direction(row)
            direction_counts[direction] += 1
            record = {
                "sample_id": sample_id,
                "source_index": source_index,
                "direction": direction,
                "prompt": prompt,
                "full_text": full_text,
                "reference": reference,
                "metadata": row.get("metadata", {}),
            }
            jsonl_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            prompt_out.write(flatten_text(prompt) + "\n")
            full_out.write(flatten_text(full_text) + "\n")

    stats = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "sample_size_requested": args.sample_size,
        "sample_size_written": len(sampled),
        "seed": args.seed,
        "balanced_by_direction": not args.no_balance,
        "model_chat_template": args.model,
        "direction_counts": dict(sorted(direction_counts.items())),
        "outputs": {
            "jsonl": str(jsonl_path),
            "prompt_text": str(prompt_txt_path),
            "full_text": str(full_txt_path),
        },
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
