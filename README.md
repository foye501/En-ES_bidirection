# EN-ES Medical Translation Fine-Tuning

This workspace contains a JSONL dataset for supervised fine-tuning a Qwen chat model to translate English clinical scenarios into Spanish.

## Files

- `azure_dataset.jsonl`: 129,117 aligned examples.
- `prepare_finetune_data.py`: converts the source JSONL into chat-format train/validation splits.
- `train_qwen_lora.py`: runs LoRA supervised fine-tuning with Hugging Face `transformers` and `peft`.
- `qwen_translate_sample.py`: existing baseline translation script.

## Setup

On the A100 server, create the local virtual environment with:

```bash
./setup_a100_env.sh
source .venv/bin/activate
```

If your server needs an explicit PyTorch CUDA wheel index, pass it through the environment:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 ./setup_a100_env.sh
```

The setup script defaults to `torch==2.9.0` from the CUDA 12.8 PyTorch index. This avoids accidentally installing newer CUDA 13 wheels on servers whose NVIDIA driver only supports CUDA 12.x.

To install FlashAttention-2 during setup:

```bash
INSTALL_FLASH_ATTN=1 ./setup_a100_env.sh
```

If compilation uses too much CPU RAM, limit parallel build jobs:

```bash
INSTALL_FLASH_ATTN=1 FLASH_ATTN_MAX_JOBS=4 ./setup_a100_env.sh
```

Conda alternative:

```bash
conda env create -f environment.yml
conda activate en-es-finetune
```

Manual virtual environment setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Apple Silicon, install the normal PyTorch build and run without `--use-4bit`. On an NVIDIA CUDA machine, add `--use-4bit` to reduce memory use.

## Prepare Data

Use the cleaned train/test CSV split:

```bash
python prepare_finetune_data.py \
  --train-input data/azure_dataset_cleaned_train.csv \
  --validation-input data/azure_dataset_cleaned_test.csv \
  --output-dir data/finetune
```

For bidirectional EN-ES and ES-EN training:

```bash
python prepare_finetune_data.py \
  --train-input data/azure_dataset_cleaned_train.csv \
  --validation-input data/azure_dataset_cleaned_test.csv \
  --output-dir data/finetune_bidirectional \
  --bidirectional
```

Smoke split with 1,000 examples:

```bash
python prepare_finetune_data.py --limit 1000 --output-dir data/finetune_smoke
```

Full split:

```bash
python prepare_finetune_data.py --output-dir data/finetune
```

Each training row is converted into Qwen chat messages:

- system: medical translation instruction
- user: translation direction, style, target length, and source scenario
- assistant: target-language translation

## First Smoke Training Run

Use a tiny run to verify that the model downloads, tokenization works, and checkpoints are written:

```bash
python train_qwen_lora.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --train-file data/finetune_smoke/train.jsonl \
  --validation-file data/finetune_smoke/validation.jsonl \
  --output-dir artifacts/qwen-en-es-lora-smoke \
  --max-steps 20 \
  --eval-steps 10 \
  --save-steps 10
```

CUDA 4-bit smoke run:

```bash
python train_qwen_lora.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --train-file data/finetune_smoke/train.jsonl \
  --validation-file data/finetune_smoke/validation.jsonl \
  --output-dir artifacts/qwen-en-es-lora-smoke \
  --max-steps 20 \
  --eval-steps 10 \
  --save-steps 10 \
  --use-4bit
```

## Full Training Run

After the smoke run succeeds:

```bash
python train_qwen_lora.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --train-file data/finetune/train.jsonl \
  --validation-file data/finetune/validation.jsonl \
  --output-dir artifacts/qwen-en-es-lora \
  --num-train-epochs 1 \
  --eval-steps 500 \
  --save-steps 500
```

The final LoRA adapter is saved under `artifacts/qwen-en-es-lora/final`.

## A100 Training Run

On the GPU server, install dependencies, prepare the JSONL files, then launch on one A100:

```bash
source .venv/bin/activate
NUM_GPUS=1 ./run_a100_finetune.sh
```

For bidirectional training on one A100:

```bash
NUM_GPUS=1 \
TRAIN_FILE=data/finetune_bidirectional/train.jsonl \
VALIDATION_FILE=data/finetune_bidirectional/validation.jsonl \
./run_a100_finetune.sh
```

If memory is tight, reduce the per-device batch size while keeping the effective batch near 32:

```bash
NUM_GPUS=1 \
PER_DEVICE_TRAIN_BATCH_SIZE=2 \
TARGET_EFFECTIVE_BATCH_SIZE=32 \
./run_a100_finetune.sh
```

To use a stronger LoRA on one A100:

```bash
NUM_GPUS=1 \
LORA_R=64 \
LORA_ALPHA=128 \
./run_a100_finetune.sh
```

To use DeepSpeed ZeRO-2:

```bash
NUM_GPUS=1 \
USE_DEEPSPEED=1 \
./run_a100_finetune.sh
```

To choose the attention implementation:

```bash
ATTN_IMPLEMENTATION=sdpa ./run_a100_finetune.sh
```

For FlashAttention-2, install `flash-attn` on the A100 server first, then run:

```bash
ATTN_IMPLEMENTATION=flash_attention_2 ./run_a100_finetune.sh
```

For real ZeRO-2 sharding benefits, use more than one GPU:

```bash
NUM_GPUS=4 \
USE_DEEPSPEED=1 \
./run_a100_finetune.sh
```

To use 2-4 A100s:

```bash
source .venv/bin/activate
NUM_GPUS=4 ./run_a100_finetune.sh
```

For 2 A100s:

```bash
NUM_GPUS=2 ./run_a100_finetune.sh
```

Useful overrides:

```bash
NUM_GPUS=4 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
TARGET_EFFECTIVE_BATCH_SIZE=32 \
MAX_SEQ_LENGTH=1024 \
NUM_TRAIN_EPOCHS=1 \
./run_a100_finetune.sh
```

The default A100 run uses `Qwen/Qwen3-4B-Instruct-2507`, bf16 mixed precision, LoRA, and an effective batch size near 32.

## Qwen2.5 1.5B Run

To train the smaller `Qwen/Qwen2.5-1.5B-Instruct` model on the same bidirectional data:

```bash
NUM_GPUS=2 \
USE_DEEPSPEED=1 \
ATTN_IMPLEMENTATION=sdpa \
./run_qwen25_15b_finetune.sh
```

For the first 1.5B run, do not set `RESUME_FROM_CHECKPOINT=true` unless the output directory already has checkpoints for this exact model. To resume the same 1.5B run later:

```bash
NUM_TRAIN_EPOCHS=2 \
RESUME_FROM_CHECKPOINT=true \
./run_qwen25_15b_finetune.sh
```

## Llama 3.2 3B Run

To compare against Llama 3.2 3B Instruct on the same bidirectional data:

```bash
NUM_GPUS=2 \
USE_DEEPSPEED=1 \
ATTN_IMPLEMENTATION=sdpa \
./run_llama32_3b_finetune.sh
```

Meta Llama models may require Hugging Face authentication before download:

```bash
huggingface-cli login
```

To evaluate the raw Llama 3.2 3B Instruct model on exactly the same 400-case file used for Qwen scoring, reuse the existing predictions/cases CSV:

```bash
CASES_FILE=artifacts/eval_sample_400_qwen1.5b/predictions.csv

python evaluate_translation_sample.py \
  --cases-file "$CASES_FILE" \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --direction both \
  --batch-size 8 \
  --output-dir artifacts/eval_sample_400_llama3.2_3b_base_same_cases
```

After fine-tuning, evaluate the Llama adapter on those same cases:

```bash
CASES_FILE=artifacts/eval_sample_400_qwen1.5b/predictions.csv

python evaluate_translation_sample.py \
  --cases-file "$CASES_FILE" \
  --adapter artifacts/llama3.2-3b-instruct-bidirectional-lora-r64-bs64/final \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --direction both \
  --batch-size 8 \
  --output-dir artifacts/eval_sample_400_llama3.2_3b_finetuned_same_cases
```

To continue from a completed 1-epoch run into epoch 2, keep the same output directory and set the total epoch count to 2:

```bash
NUM_GPUS=4 \
USE_DEEPSPEED=1 \
TRAIN_FILE=data/finetune_bidirectional/train.jsonl \
VALIDATION_FILE=data/finetune_bidirectional/validation.jsonl \
OUTPUT_DIR=artifacts/qwen3-4b-bidirectional-lora-r64-bs64 \
NUM_TRAIN_EPOCHS=2 \
RESUME_FROM_CHECKPOINT=true \
./run_a100_finetune.sh
```

To resume from a specific checkpoint:

```bash
RESUME_FROM_CHECKPOINT=artifacts/qwen3-4b-bidirectional-lora-r64-bs64/checkpoint-7033 ./run_a100_finetune.sh
```

## EN-ES Upsampling

If EN-ES quality lags ES-EN, create a direction-weighted training file from the existing bidirectional JSONL. This keeps validation unchanged and changes only the training mix:

```bash
python upsample_bidirectional_data.py \
  --train-file data/finetune_bidirectional/train.jsonl \
  --validation-file data/finetune_bidirectional/validation.jsonl \
  --output-dir data/finetune_bidirectional_en_es_x2 \
  --en-es-repeat 2 \
  --es-en-repeat 1
```

`--en-es-repeat 2 --es-en-repeat 1` makes the train mix about 67% EN-ES and 33% ES-EN. For a gentler 60/40 mix, use `--en-es-repeat 3 --es-en-repeat 2`.

For a second-stage 1.5B Instruct run, start from the completed 50/50 adapter but use a new output directory and a lower learning rate. Do not use `RESUME_FROM_CHECKPOINT=true` when changing the train data mix:

```bash
unset RESUME_FROM_CHECKPOINT

NUM_GPUS=2 \
USE_DEEPSPEED=1 \
ATTN_IMPLEMENTATION=sdpa \
MODEL=Qwen/Qwen2.5-1.5B-Instruct \
ADAPTER=artifacts/qwen2.5-1.5b-instruct-bidirectional-lora-r64-bs64/final \
OUTPUT_DIR=artifacts/qwen2.5-1.5b-instruct-en-es-x2-stage2-lora-r64-bs64 \
TRAIN_FILE=data/finetune_bidirectional_en_es_x2/train.jsonl \
VALIDATION_FILE=data/finetune_bidirectional_en_es_x2/validation.jsonl \
LORA_R=64 \
LORA_ALPHA=128 \
PER_DEVICE_TRAIN_BATCH_SIZE=16 \
TARGET_EFFECTIVE_BATCH_SIZE=64 \
LEARNING_RATE=5e-5 \
NUM_TRAIN_EPOCHS=0.5 \
./run_qwen25_15b_finetune.sh
```

Evaluate the new stage on the same cases as the 50/50 run. Keep it only if EN-ES improves without a meaningful ES-EN regression.

## Refined Term Repair Stage

To train on every row in the refined term dataset, write all refined examples to `train.jsonl` and use a separate validation file only for monitoring:

```bash
python prepare_finetune_data.py \
  --input data/azure_dataset_refined_experiment.csv \
  --output-dir data/finetune_refined_experiment_en_es_all_terms \
  --train-all \
  --seed 42

cp data/finetune/validation.jsonl data/finetune_refined_experiment_en_es_all_terms/validation.jsonl
```

Run a low-learning-rate repair stage from the current best 1.5B adapter. This uses all refined terms for training and starts a fresh optimizer/scheduler:

```bash
unset RESUME_FROM_CHECKPOINT

NUM_GPUS=1 \
USE_DEEPSPEED=0 \
ATTN_IMPLEMENTATION=sdpa \
MODEL=Qwen/Qwen2.5-1.5B-Instruct \
ADAPTER=artifacts/qwen2.5-1.5b-instruct-bidirectional-lora-r64-bs64/final \
OUTPUT_DIR=artifacts/qwen2.5-1.5b-instruct-refined-all-terms-en-es-stage \
TRAIN_FILE=data/finetune_refined_experiment_en_es_all_terms/train.jsonl \
VALIDATION_FILE=data/finetune_refined_experiment_en_es_all_terms/validation.jsonl \
LORA_R=64 \
LORA_ALPHA=128 \
PER_DEVICE_TRAIN_BATCH_SIZE=8 \
TARGET_EFFECTIVE_BATCH_SIZE=32 \
LEARNING_RATE=2e-5 \
NUM_TRAIN_EPOCHS=3 \
EVAL_STEPS=50 \
SAVE_STEPS=50 \
./run_qwen25_15b_finetune.sh
```

If you want to train without any validation pass, set `EVAL_STRATEGY=no`.

## Quantization Calibration Sample

For MTK NPU quantization, create a small representative calibration set from the bidirectional training JSONL:

```bash
python sample_quantization_data.py \
  --input data/finetune_bidirectional/train.jsonl \
  --sample-size 128 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir data/quantization/qwen25_15b_calibration_128
```

The sampler writes:

- `calibration_prompts.jsonl`: structured prompt, full text, reference, direction, and metadata
- `calibration_mtk_prompt_only.jsonl`: MTK-style JSONL records with `id`, Qwen chat-template `text`, `task`, `mode`, `bucket`, and `token_length`
- `calibration_prompts.txt`: prompt-only flattened text, one sample per line
- `calibration_full_texts.txt`: prompt plus reference translation, one sample per line
- `stats.json`: sampling counts and output paths

Use `calibration_mtk_prompt_only.jsonl` for MTK NPU calibration when the tool expects JSONL records like:

```json
{"id": "1_prompt_only", "text": "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n", "task": "medical_translation_en_es", "dataset_version": null, "mode": "prompt_only", "bucket": "0_512", "token_length": 256, "source_line_no": 1, "meeting_id": null, "segment_id": null}
```

Use `calibration_full_texts.txt` when the quantization tool wants representative model text sequences. Use `calibration_prompts.txt` when it wants inference-style prompts only.

## Evaluate 500 Test Cases

After training finishes, evaluate a 500-row sample from the cleaned test set:

```bash
python evaluate_translation_sample.py \
  --adapter artifacts/qwen3-4b-instruct-2507-en-es-lora/final \
  --test-file data/azure_dataset_cleaned_test.csv \
  --sample-size 500 \
  --direction both \
  --batch-size 8 \
  --output-dir artifacts/eval_sample_500
```

For EN-ES only, use `--direction en-es`. For ES-EN only, use `--direction es-en`.

The script writes `predictions.csv` with source, reference, and model output, plus `summary.json` with exact-match, token-F1, and chrF-like scores.

To evaluate the original base model on the same sampled cases, omit `--adapter` and keep the same `--seed`, `--sample-size`, and `--direction`:

```bash
python evaluate_translation_sample.py \
  --test-file data/azure_dataset_cleaned_test.csv \
  --sample-size 500 \
  --seed 42 \
  --direction both \
  --batch-size 8 \
  --output-dir artifacts/eval_sample_500_base
```

Compare `artifacts/eval_sample_500_base/predictions.csv` with the fine-tuned `predictions.csv`. Both files include `row_index` so rows can be matched directly.

To reuse an existing fine-tuned `predictions.csv` exactly, including its case order:

```bash
python evaluate_translation_sample.py \
  --cases-file artifacts/eval_sample_500/predictions.csv \
  --direction both \
  --batch-size 8 \
  --output-dir artifacts/eval_sample_500_base_same_cases
```

To evaluate the fine-tuned Qwen2.5 1.5B adapter:

```bash
python evaluate_translation_sample.py \
  --adapter artifacts/qwen2.5-1.5b-bidirectional-lora-r64-bs64-v2/final \
  --test-file data/azure_dataset_cleaned_test.csv \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --sample-size 500 \
  --seed 42 \
  --direction both \
  --batch-size 8 \
  --output-dir artifacts/eval_sample_500_finetuned_1.5b
```

When `--adapter` is set and `--model` is omitted, the evaluator reads the adapter's `adapter_config.json` and uses the original training base model automatically. If `--model` differs from the adapter base model, the script raises an error by default.

## Translate Prompt JSONL

For prompt-only JSONL files such as ASR transcript prompts, use `translate_prompt_jsonl.py` instead of `evaluate_translation_sample.py`:

```bash
python translate_prompt_jsonl.py \
  --input artifacts/nemo_asr_transcripts/nemo_asr_en_es_translation_prompts.jsonl \
  --adapter artifacts/qwen3-4b-bidirectional-lora-r64-bs64/final \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --batch-size 8 \
  --output-dir artifacts/qwen3_4b_adapter_translate_asr
```

The script writes `translations.csv` and `translations.jsonl` with segment metadata, source text, and translated output.
