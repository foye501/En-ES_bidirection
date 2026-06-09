import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from prepare_finetune_data import SYSTEM_PROMPTS, row_to_chat
from train_qwen_lora import DEFAULT_MODEL, has_cuda_bf16


def parse_args():
    parser = argparse.ArgumentParser(description="Sample test rows and evaluate base or LoRA translation outputs.")
    parser.add_argument(
        "--model",
        default=None,
        help="Base model id/path. If --adapter is set and --model is omitted, this is read from adapter_config.json.",
    )
    parser.add_argument("--adapter", default=None, help="Optional path to a trained LoRA adapter directory.")
    parser.add_argument(
        "--cases-file",
        default=None,
        help="Existing predictions/cases CSV to reuse in order instead of sampling from --test-file.",
    )
    parser.add_argument("--test-file", default="data/azure_dataset_cleaned_test.csv")
    parser.add_argument("--output-dir", default="artifacts/eval_sample_500")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--direction", choices=["en-es", "es-en", "both"], default="both")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--merge-adapter", action="store_true", help="Merge LoRA weights for faster inference.")
    parser.add_argument(
        "--allow-model-mismatch",
        action="store_true",
        help="Allow --model to differ from the adapter's base_model_name_or_path.",
    )
    return parser.parse_args()


def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def token_f1(prediction, reference):
    prediction_tokens = normalize_text(prediction).split()
    reference_tokens = normalize_text(reference).split()
    if not prediction_tokens and not reference_tokens:
        return 1.0
    if not prediction_tokens or not reference_tokens:
        return 0.0
    overlap = Counter(prediction_tokens) & Counter(reference_tokens)
    overlap_count = sum(overlap.values())
    if overlap_count == 0:
        return 0.0
    precision = overlap_count / len(prediction_tokens)
    recall = overlap_count / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def char_ngram_counts(text, n):
    text = normalize_text(text)
    if len(text) < n:
        return Counter([text]) if text else Counter()
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


def chrf_like(prediction, reference, max_order=6, beta=2.0):
    scores = []
    for order in range(1, max_order + 1):
        pred_counts = char_ngram_counts(prediction, order)
        ref_counts = char_ngram_counts(reference, order)
        if not pred_counts and not ref_counts:
            scores.append(1.0)
            continue
        if not pred_counts or not ref_counts:
            scores.append(0.0)
            continue
        overlap = sum((pred_counts & ref_counts).values())
        precision = overlap / sum(pred_counts.values())
        recall = overlap / sum(ref_counts.values())
        if precision == 0.0 and recall == 0.0:
            scores.append(0.0)
        else:
            beta2 = beta * beta
            scores.append((1 + beta2) * precision * recall / (beta2 * precision + recall))
    return sum(scores) / len(scores)


def read_test_rows(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as src:
        rows = []
        for row_index, row in enumerate(csv.DictReader(src)):
            row["row_index"] = row_index
            rows.append(row)
        return rows


def expand_directions(rows, direction):
    directions = ["en-es", "es-en"] if direction == "both" else [direction]
    examples = []
    for row in rows:
        for item_direction in directions:
            chat = row_to_chat(row, direction=item_direction)
            source_text = row["english_scenario"] if item_direction == "en-es" else row["spanish_translation"]
            reference_text = row["spanish_translation"] if item_direction == "en-es" else row["english_scenario"]
            examples.append(
                {
                    "direction": item_direction,
                    "row_index": row["row_index"],
                    "term": row["term"],
                    "target_length": row["target_length"],
                    "style": row["style"],
                    "source_text": source_text,
                    "reference_text": reference_text,
                    "messages": chat["messages"][:-1],
                }
            )
    return examples


def read_cases_file(path, direction):
    directions = {"en-es", "es-en"} if direction == "both" else {direction}
    examples = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as src:
        reader = csv.DictReader(src)
        required = {"direction", "source_text", "reference_text"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        for case_index, row in enumerate(reader):
            item_direction = row["direction"]
            if item_direction not in directions:
                continue
            if item_direction == "en-es":
                source_language = "English"
            elif item_direction == "es-en":
                source_language = "Spanish"
            else:
                raise ValueError(f"Unsupported direction in {path}: {item_direction}")

            user = (
                f"Direction: {'English to Spanish' if item_direction == 'en-es' else 'Spanish to English'}\n"
                f"Style: {row.get('style', '')}\n"
                f"Target length: about {row.get('target_length', '')} words\n"
                f"{source_language}: {row['source_text']}"
            )
            examples.append(
                {
                    "direction": item_direction,
                    "row_index": row.get("row_index", str(case_index)),
                    "term": row.get("term", ""),
                    "target_length": row.get("target_length", ""),
                    "style": row.get("style", ""),
                    "source_text": row["source_text"],
                    "reference_text": row["reference_text"],
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPTS[item_direction]},
                        {"role": "user", "content": user},
                    ],
                }
            )
    return examples


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize_model_id(model_id):
    return str(model_id).rstrip("/")


def get_adapter_base_model(adapter_path):
    if not adapter_path:
        return None
    config_path = Path(adapter_path) / "adapter_config.json"
    if not config_path.exists():
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return config.get("base_model_name_or_path")


def resolve_model_id(model_arg, adapter_path, allow_model_mismatch):
    adapter_base_model = get_adapter_base_model(adapter_path)
    if adapter_base_model:
        if not model_arg:
            print(f"Using adapter base model from adapter_config.json: {adapter_base_model}")
            return adapter_base_model
        if normalize_model_id(model_arg) != normalize_model_id(adapter_base_model) and not allow_model_mismatch:
            raise ValueError(
                "Adapter/base model mismatch. "
                f"The adapter was trained from {adapter_base_model!r}, but --model is {model_arg!r}. "
                "Use the adapter base model or pass --allow-model-mismatch only if this is intentional."
            )
    return model_arg or DEFAULT_MODEL


def load_model(model_id, adapter_path, merge_adapter):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = get_device()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if has_cuda_bf16() else "auto",
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        if merge_adapter:
            model = model.merge_and_unload()
    elif merge_adapter:
        raise ValueError("--merge-adapter requires --adapter.")
    if device != "cuda":
        model = model.to(device)
    model.eval()
    return model, tokenizer, device


def generation_eos_token_ids(tokenizer):
    token_ids = []
    for token_id in [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]:
        if token_id is not None and token_id != tokenizer.unk_token_id and token_id not in token_ids:
            token_ids.append(token_id)
    return token_ids[0] if len(token_ids) == 1 else token_ids


def clean_prediction(text):
    for stop_text in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
        if stop_text in text:
            text = text.split(stop_text, 1)[0]
    return text.strip()


def generate_batch(model, tokenizer, device, examples, max_new_tokens, temperature):
    prompts = [
        tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True)
        for example in examples
    ]
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": generation_eos_token_ids(tokenizer),
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)

    predictions = []
    for index in range(len(prompts)):
        input_length = inputs.input_ids[index].shape[0]
        generated_tokens = outputs[index][input_length:]
        prediction = tokenizer.decode(generated_tokens, skip_special_tokens=False)
        predictions.append(clean_prediction(prediction))
    return predictions


def summarize(results):
    summary = {
        "num_examples": len(results),
        "exact_match": sum(item["exact_match"] for item in results) / len(results),
        "token_f1": sum(item["token_f1"] for item in results) / len(results),
        "chrf_like": sum(item["chrf_like"] for item in results) / len(results),
    }
    for direction in sorted({item["direction"] for item in results}):
        subset = [item for item in results if item["direction"] == direction]
        summary[direction] = {
            "num_examples": len(subset),
            "exact_match": sum(item["exact_match"] for item in subset) / len(subset),
            "token_f1": sum(item["token_f1"] for item in subset) / len(subset),
            "chrf_like": sum(item["chrf_like"] for item in subset) / len(subset),
        }
    return summary


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.cases_file:
        examples = read_cases_file(args.cases_file, args.direction)
    else:
        rows = read_test_rows(args.test_file)
        if args.sample_size > len(rows):
            raise ValueError(f"Requested {args.sample_size} rows, but {args.test_file} has only {len(rows)} rows.")
        sampled_rows = random.Random(args.seed).sample(rows, args.sample_size)
        examples = expand_directions(sampled_rows, args.direction)
    if not examples:
        raise ValueError("No evaluation examples were selected.")

    model_id = resolve_model_id(args.model, args.adapter, args.allow_model_mismatch)
    model, tokenizer, device = load_model(model_id, args.adapter, args.merge_adapter)
    print(f"Loaded model on {device}. Evaluating {len(examples)} examples.")

    results = []
    for start in tqdm(range(0, len(examples), args.batch_size), desc="Generating"):
        batch = examples[start : start + args.batch_size]
        predictions = generate_batch(
            model,
            tokenizer,
            device,
            batch,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        for example, prediction in zip(batch, predictions):
            reference = example["reference_text"]
            exact_match = int(normalize_text(prediction) == normalize_text(reference))
            results.append(
                {
                    "direction": example["direction"],
                    "row_index": example["row_index"],
                    "term": example["term"],
                    "target_length": example["target_length"],
                    "style": example["style"],
                    "source_text": example["source_text"],
                    "reference_text": reference,
                    "prediction": prediction,
                    "exact_match": exact_match,
                    "token_f1": token_f1(prediction, reference),
                    "chrf_like": chrf_like(prediction, reference),
                }
            )

    predictions_path = output_dir / "predictions.csv"
    summary_path = output_dir / "summary.json"

    with predictions_path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    summary = summarize(results)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
