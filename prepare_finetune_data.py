import argparse
import csv
import json
import random
from pathlib import Path

from tqdm import tqdm


SYSTEM_PROMPTS = {
    "en-es": (
        "You are a professional medical translator. Translate English clinical "
        "scenarios into strictly correct, fluent Spanish medical terminology. "
        "Output only the Spanish translation."
    ),
    "es-en": (
        "You are a professional medical translator. Translate Spanish clinical "
        "scenarios into strictly correct, fluent English medical terminology. "
        "Output only the English translation."
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert the EN-ES medical translation dataset into chat SFT JSONL splits."
    )
    parser.add_argument("--input", default="azure_dataset.jsonl", help="Source JSONL file.")
    parser.add_argument("--train-input", default=None, help="Explicit train CSV or JSONL source.")
    parser.add_argument("--validation-input", default=None, help="Explicit validation CSV or JSONL source.")
    parser.add_argument("--output-dir", default="data/finetune", help="Directory for split JSONL files.")
    parser.add_argument("--validation-ratio", type=float, default=0.02)
    parser.add_argument(
        "--train-all",
        action="store_true",
        help="Write every input row to train.jsonl and do not hold out validation rows.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for a smoke dataset.")
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip rows with missing required fields and write skipped_rows.jsonl.",
    )
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="Emit both English-to-Spanish and Spanish-to-English examples for each row.",
    )
    return parser.parse_args()


def row_to_chat(row, direction="en-es"):
    if direction == "en-es":
        source_language = "English"
        target_language = "Spanish"
        source_text = row["english_scenario"]
        target_text = row["spanish_translation"]
    elif direction == "es-en":
        source_language = "Spanish"
        target_language = "English"
        source_text = row["spanish_translation"]
        target_text = row["english_scenario"]
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    user = (
        f"Direction: {source_language} to {target_language}\n"
        f"Style: {row['style']}\n"
        f"Target length: about {row['target_length']} words\n"
        f"{source_language}: {source_text}"
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS[direction]},
            {"role": "user", "content": user},
            {"role": "assistant", "content": target_text},
        ],
        "metadata": {
            "term": row["term"],
            "target_length": row["target_length"],
            "style": row["style"],
            "direction": direction,
        },
    }


def row_to_examples(row, bidirectional=False):
    examples = [row_to_chat(row, direction="en-es")]
    if bidirectional:
        examples.append(row_to_chat(row, direction="es-en"))
    return examples


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


def write_chat_rows(source_path, output_path, limit=None, bidirectional=False, skip_invalid=False, skipped_out=None):
    count = 0
    source_rows = 0
    skipped_count = 0
    with output_path.open("w", encoding="utf-8") as out:
        for row_number, row in enumerate(tqdm(iter_rows(source_path), desc=f"Preparing {source_path.name}"), start=1):
            if limit is not None and source_rows >= limit:
                break
            try:
                validate_row(row, row_number, source_path)
            except ValueError as error:
                if not skip_invalid:
                    raise
                skipped_count += 1
                source_rows += 1
                if skipped_out:
                    skipped_out.write(
                        json.dumps(
                            {"source": str(source_path), "row_number": row_number, "error": str(error), "row": row},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                continue
            for example in row_to_examples(row, bidirectional=bidirectional):
                out.write(json.dumps(example, ensure_ascii=False) + "\n")
                count += 1
            source_rows += 1
    return count, source_rows, skipped_count


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    stats_path = output_dir / "stats.json"
    skipped_path = output_dir / "skipped_rows.jsonl"
    skipped_out = skipped_path.open("w", encoding="utf-8") if args.skip_invalid else None

    try:
        if args.train_all and (args.train_input or args.validation_input):
            raise ValueError("--train-all cannot be combined with --train-input/--validation-input.")

        if args.train_input or args.validation_input:
            if not args.train_input or not args.validation_input:
                raise ValueError("--train-input and --validation-input must be provided together.")
            train_source = Path(args.train_input)
            validation_source = Path(args.validation_input)
            train_count, train_source_rows, train_skipped_count = write_chat_rows(
                train_source,
                train_path,
                limit=args.limit,
                bidirectional=args.bidirectional,
                skip_invalid=args.skip_invalid,
                skipped_out=skipped_out,
            )
            validation_count, validation_source_rows, validation_skipped_count = write_chat_rows(
                validation_source,
                validation_path,
                limit=args.limit,
                bidirectional=args.bidirectional,
                skip_invalid=args.skip_invalid,
                skipped_out=skipped_out,
            )
            total = train_count + validation_count
            total_source_rows = train_source_rows + validation_source_rows
            skipped_count = train_skipped_count + validation_skipped_count
            sources = {"train": str(train_source), "validation": str(validation_source)}
        elif args.train_all:
            train_count = 0
            skipped_count = 0
            total_source_rows = 0
            sources = {"source": str(input_path)}
            with train_path.open("w", encoding="utf-8") as train_out, validation_path.open(
                "w", encoding="utf-8"
            ) as validation_out:
                for row_number, row in enumerate(tqdm(iter_rows(input_path), desc="Preparing train rows"), start=1):
                    if args.limit is not None and total_source_rows >= args.limit:
                        break
                    try:
                        validate_row(row, row_number, input_path)
                    except ValueError as error:
                        if not args.skip_invalid:
                            raise
                        skipped_count += 1
                        if skipped_out:
                            skipped_out.write(
                                json.dumps(
                                    {
                                        "source": str(input_path),
                                        "row_number": row_number,
                                        "error": str(error),
                                        "row": row,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                        total_source_rows += 1
                        continue
                    for example in row_to_examples(row, bidirectional=args.bidirectional):
                        encoded = json.dumps(example, ensure_ascii=False) + "\n"
                        train_out.write(encoded)
                        train_count += 1
                    total_source_rows += 1
                validation_out.write("")
            validation_count = 0
            total = train_count
        else:
            rng = random.Random(args.seed)
            total = 0
            total_source_rows = 0
            train_count = 0
            validation_count = 0
            skipped_count = 0
            sources = {"source": str(input_path)}

            with train_path.open("w", encoding="utf-8") as train_out, validation_path.open(
                "w", encoding="utf-8"
            ) as validation_out:
                for row_number, row in enumerate(tqdm(iter_rows(input_path), desc="Preparing rows"), start=1):
                    if args.limit is not None and total_source_rows >= args.limit:
                        break
                    try:
                        validate_row(row, row_number, input_path)
                    except ValueError as error:
                        if not args.skip_invalid:
                            raise
                        skipped_count += 1
                        if skipped_out:
                            skipped_out.write(
                                json.dumps(
                                    {
                                        "source": str(input_path),
                                        "row_number": row_number,
                                        "error": str(error),
                                        "row": row,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                        total_source_rows += 1
                        continue

                    if rng.random() < args.validation_ratio:
                        target_out = validation_out
                        validation_count += len(row_to_examples(row, bidirectional=args.bidirectional))
                    else:
                        target_out = train_out
                        train_count += len(row_to_examples(row, bidirectional=args.bidirectional))
                    for example in row_to_examples(row, bidirectional=args.bidirectional):
                        target_out.write(json.dumps(example, ensure_ascii=False) + "\n")
                        total += 1
                    total_source_rows += 1
    finally:
        if skipped_out:
            skipped_out.close()

    stats = {
        "sources": sources,
        "train_path": str(train_path),
        "validation_path": str(validation_path),
        "total": total,
        "source_rows": total_source_rows,
        "train": train_count,
        "validation": validation_count,
        "validation_ratio": args.validation_ratio,
        "seed": args.seed,
        "bidirectional": args.bidirectional,
        "train_all": args.train_all,
        "skip_invalid": args.skip_invalid,
        "skipped": skipped_count,
        "skipped_rows_path": str(skipped_path) if args.skip_invalid else None,
    }
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
