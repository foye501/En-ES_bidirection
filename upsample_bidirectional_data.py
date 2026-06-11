import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a direction-weighted training JSONL from bidirectional chat SFT data."
    )
    parser.add_argument("--train-file", default="data/finetune_bidirectional/train.jsonl")
    parser.add_argument("--validation-file", default="data/finetune_bidirectional/validation.jsonl")
    parser.add_argument("--output-dir", default="data/finetune_bidirectional_en_es_x2")
    parser.add_argument("--en-es-repeat", type=int, default=2)
    parser.add_argument("--es-en-repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument(
        "--weight-validation",
        action="store_true",
        help="Also apply repeats to validation. Default keeps validation unchanged for fair comparison.",
    )
    return parser.parse_args()


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as src:
        for line_number, line in enumerate(src, start=1):
            if line.strip():
                row = json.loads(line)
                row["_source_line_no"] = line_number
                rows.append(row)
    return rows


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            row = dict(row)
            row.pop("_source_line_no", None)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def direction_of(row):
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


def repeat_for_direction(direction, en_es_repeat, es_en_repeat):
    if direction == "en-es":
        return en_es_repeat
    if direction == "es-en":
        return es_en_repeat
    return 1


def expand_rows(rows, en_es_repeat, es_en_repeat, seed, shuffle):
    expanded = []
    for row in rows:
        repeat = repeat_for_direction(direction_of(row), en_es_repeat, es_en_repeat)
        if repeat < 0:
            raise ValueError("Repeat counts must be >= 0.")
        expanded.extend(row for _ in range(repeat))
    if shuffle:
        random.Random(seed).shuffle(expanded)
    return expanded


def count_directions(rows):
    return dict(sorted(Counter(direction_of(row) for row in rows).items()))


def ratio(counts):
    total = sum(counts.values())
    if total == 0:
        return {}
    return {direction: round(count / total, 4) for direction, count in counts.items()}


def main():
    args = parse_args()
    train_path = Path(args.train_file)
    validation_path = Path(args.validation_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not validation_path.exists():
        raise FileNotFoundError(f"Validation file not found: {validation_path}")

    train_rows = read_jsonl(train_path)
    validation_rows = read_jsonl(validation_path)
    weighted_train_rows = expand_rows(
        train_rows,
        args.en_es_repeat,
        args.es_en_repeat,
        args.seed,
        shuffle=not args.no_shuffle,
    )

    output_train_path = output_dir / "train.jsonl"
    output_validation_path = output_dir / "validation.jsonl"
    write_jsonl(output_train_path, weighted_train_rows)

    if args.weight_validation:
        weighted_validation_rows = expand_rows(
            validation_rows,
            args.en_es_repeat,
            args.es_en_repeat,
            args.seed,
            shuffle=not args.no_shuffle,
        )
        write_jsonl(output_validation_path, weighted_validation_rows)
    else:
        shutil.copyfile(validation_path, output_validation_path)
        weighted_validation_rows = validation_rows

    input_train_counts = count_directions(train_rows)
    output_train_counts = count_directions(weighted_train_rows)
    input_validation_counts = count_directions(validation_rows)
    output_validation_counts = count_directions(weighted_validation_rows)
    stats = {
        "sources": {
            "train": str(train_path),
            "validation": str(validation_path),
        },
        "outputs": {
            "train": str(output_train_path),
            "validation": str(output_validation_path),
        },
        "seed": args.seed,
        "shuffled": not args.no_shuffle,
        "repeats": {
            "en-es": args.en_es_repeat,
            "es-en": args.es_en_repeat,
        },
        "weight_validation": args.weight_validation,
        "train": {
            "input_total": len(train_rows),
            "input_direction_counts": input_train_counts,
            "input_direction_ratio": ratio(input_train_counts),
            "output_total": len(weighted_train_rows),
            "output_direction_counts": output_train_counts,
            "output_direction_ratio": ratio(output_train_counts),
        },
        "validation": {
            "input_total": len(validation_rows),
            "input_direction_counts": input_validation_counts,
            "input_direction_ratio": ratio(input_validation_counts),
            "output_total": len(weighted_validation_rows),
            "output_direction_counts": output_validation_counts,
            "output_direction_ratio": ratio(output_validation_counts),
        },
    }
    stats_path = output_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
