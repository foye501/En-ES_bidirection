import argparse
import json
import random
from pathlib import Path

from tqdm import tqdm


SYSTEM_PROMPT = (
    "You are a professional medical translator. Translate English clinical "
    "scenarios into strictly correct, fluent Spanish medical terminology. "
    "Output only the Spanish translation."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert the EN-ES medical translation JSONL file into chat SFT JSONL splits."
    )
    parser.add_argument("--input", default="azure_dataset.jsonl", help="Source JSONL file.")
    parser.add_argument("--output-dir", default="data/finetune", help="Directory for split JSONL files.")
    parser.add_argument("--validation-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for a smoke dataset.")
    return parser.parse_args()


def row_to_chat(row):
    user = (
        f"Style: {row['style']}\n"
        f"Target length: about {row['target_length']} words\n"
        f"English: {row['english_scenario']}"
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": row["spanish_translation"]},
        ],
        "metadata": {
            "term": row["term"],
            "target_length": row["target_length"],
            "style": row["style"],
        },
    }


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    stats_path = output_dir / "stats.json"

    rng = random.Random(args.seed)
    required = {"term", "target_length", "style", "english_scenario", "spanish_translation"}
    total = 0
    train_count = 0
    validation_count = 0

    with input_path.open("r", encoding="utf-8") as src, train_path.open(
        "w", encoding="utf-8"
    ) as train_out, validation_path.open("w", encoding="utf-8") as validation_out:
        for line_number, line in enumerate(tqdm(src, desc="Preparing rows"), start=1):
            if args.limit is not None and total >= args.limit:
                break
            if not line.strip():
                continue

            row = json.loads(line)
            missing = required - set(row)
            if missing:
                raise ValueError(f"Line {line_number} missing required fields: {sorted(missing)}")

            example = row_to_chat(row)
            out_line = json.dumps(example, ensure_ascii=False) + "\n"
            if rng.random() < args.validation_ratio:
                validation_out.write(out_line)
                validation_count += 1
            else:
                train_out.write(out_line)
                train_count += 1
            total += 1

    stats = {
        "source": str(input_path),
        "train_path": str(train_path),
        "validation_path": str(validation_path),
        "total": total,
        "train": train_count,
        "validation": validation_count,
        "validation_ratio": args.validation_ratio,
        "seed": args.seed,
    }
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
