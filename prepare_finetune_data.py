import argparse
import csv
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
        description="Convert the EN-ES medical translation dataset into chat SFT JSONL splits."
    )
    parser.add_argument("--input", default="azure_dataset.jsonl", help="Source JSONL file.")
    parser.add_argument("--train-input", default=None, help="Explicit train CSV or JSONL source.")
    parser.add_argument("--validation-input", default=None, help="Explicit validation CSV or JSONL source.")
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


def detect_format(path):
    if path.suffix.lower() == ".csv":
        return "csv"
    if path.suffix.lower() in {".jsonl", ".json"}:
        return "jsonl"
    raise ValueError(f"Cannot infer dataset format from extension: {path}")


def iter_rows(path):
    source_format = detect_format(path)
    if source_format == "csv":
        with path.open("r", encoding="utf-8-sig", newline="") as src:
            yield from csv.DictReader(src)
    else:
        with path.open("r", encoding="utf-8") as src:
            for line in src:
                if line.strip():
                    yield json.loads(line)


def validate_row(row, row_number, source):
    required = {"term", "target_length", "style", "english_scenario", "spanish_translation"}
    missing = [field for field in required if not row.get(field)]
    if missing:
        raise ValueError(f"{source} row {row_number} missing required fields: {missing}")


def write_chat_rows(source_path, output_path, limit=None):
    count = 0
    with output_path.open("w", encoding="utf-8") as out:
        for row_number, row in enumerate(tqdm(iter_rows(source_path), desc=f"Preparing {source_path.name}"), start=1):
            if limit is not None and count >= limit:
                break
            validate_row(row, row_number, source_path)
            out.write(json.dumps(row_to_chat(row), ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    stats_path = output_dir / "stats.json"

    if args.train_input or args.validation_input:
        if not args.train_input or not args.validation_input:
            raise ValueError("--train-input and --validation-input must be provided together.")
        train_source = Path(args.train_input)
        validation_source = Path(args.validation_input)
        train_count = write_chat_rows(train_source, train_path, limit=args.limit)
        validation_count = write_chat_rows(validation_source, validation_path, limit=args.limit)
        total = train_count + validation_count
        sources = {"train": str(train_source), "validation": str(validation_source)}
    else:
        rng = random.Random(args.seed)
        total = 0
        train_count = 0
        validation_count = 0
        sources = {"source": str(input_path)}

        with train_path.open("w", encoding="utf-8") as train_out, validation_path.open(
            "w", encoding="utf-8"
        ) as validation_out:
            for row_number, row in enumerate(tqdm(iter_rows(input_path), desc="Preparing rows"), start=1):
                if args.limit is not None and total >= args.limit:
                    break
                validate_row(row, row_number, input_path)

                out_line = json.dumps(row_to_chat(row), ensure_ascii=False) + "\n"
                if rng.random() < args.validation_ratio:
                    validation_out.write(out_line)
                    validation_count += 1
                else:
                    train_out.write(out_line)
                    train_count += 1
                total += 1

    stats = {
        "sources": sources,
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
