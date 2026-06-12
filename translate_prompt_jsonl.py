import argparse
import csv
import json
from pathlib import Path

from tqdm import tqdm

from evaluate_translation_sample import (
    clean_prediction,
    generate_batch,
    load_model,
    resolve_model_id,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Translate chat-prompt JSONL with a base model and optional LoRA adapter.")
    parser.add_argument("--input", required=True, help="JSONL file containing a messages array per row.")
    parser.add_argument("--output-dir", default="artifacts/translated_prompts")
    parser.add_argument(
        "--model",
        default=None,
        help="Base model id/path. If --adapter is set and --model is omitted, reads adapter_config.json.",
    )
    parser.add_argument("--adapter", default=None, help="Optional LoRA adapter directory.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--merge-adapter", action="store_true")
    parser.add_argument("--allow-model-mismatch", action="store_true")
    return parser.parse_args()


def read_prompt_rows(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as src:
        for line_number, line in enumerate(src, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "messages" not in row:
                raise ValueError(f"{path} line {line_number} is missing 'messages'.")
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def source_text_from_messages(messages):
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        marker = "\nEnglish: "
        if marker in content:
            return content.split(marker, 1)[1].strip()
        marker = "English: "
        if marker in content:
            return content.split(marker, 1)[1].strip()
        return content.strip()
    return ""


def metadata_for_csv(row):
    return {
        "segment_id": row.get("segment_id", row.get("_line_number", "")),
        "accent": row.get("accent", ""),
        "youtube_id": row.get("youtube_id", ""),
        "start_time": row.get("start_time", ""),
        "duration": row.get("duration", ""),
        "source_text": source_text_from_messages(row.get("messages", [])),
    }


def main():
    args = parse_args()
    rows = read_prompt_rows(args.input)
    if not rows:
        raise ValueError(f"No prompt rows found in {args.input}.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_id = resolve_model_id(args.model, args.adapter, args.allow_model_mismatch)
    model, tokenizer, device = load_model(model_id, args.adapter, args.merge_adapter)
    print(f"Loaded model on {device}. Translating {len(rows)} prompts.")

    outputs = []
    for start in tqdm(range(0, len(rows), args.batch_size), desc="Translating"):
        batch_rows = rows[start : start + args.batch_size]
        examples = [{"messages": row["messages"]} for row in batch_rows]
        predictions = generate_batch(
            model,
            tokenizer,
            device,
            examples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        for row, prediction in zip(batch_rows, predictions):
            clean_row = dict(row)
            clean_row.pop("_line_number", None)
            clean_row["translation"] = clean_prediction(prediction)
            outputs.append(clean_row)

    jsonl_path = output_dir / "translations.jsonl"
    csv_path = output_dir / "translations.csv"
    with jsonl_path.open("w", encoding="utf-8") as out:
        for row in outputs:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames = ["segment_id", "accent", "youtube_id", "start_time", "duration", "source_text", "translation"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for row in outputs:
            flat = metadata_for_csv(row)
            flat["translation"] = row["translation"]
            writer.writerow(flat)

    print(f"Saved translations to {csv_path}")
    print(f"Saved JSONL to {jsonl_path}")


if __name__ == "__main__":
    main()
