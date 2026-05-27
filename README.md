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
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 ./setup_a100_env.sh
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
