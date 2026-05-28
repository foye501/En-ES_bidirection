#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
TARGET_EFFECTIVE_BATCH_SIZE="${TARGET_EFFECTIVE_BATCH_SIZE:-32}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-1024}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
TRAIN_FILE="${TRAIN_FILE:-data/finetune/train.jsonl}"
VALIDATION_FILE="${VALIDATION_FILE:-data/finetune/validation.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/qwen3-4b-instruct-2507-en-es-lora}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
USE_DEEPSPEED="${USE_DEEPSPEED:-0}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed_zero2.json}"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
  else
    PYTHON="python"
  fi
fi

if [[ "$NUM_GPUS" -lt 1 || "$NUM_GPUS" -gt 4 ]]; then
  echo "NUM_GPUS must be between 1 and 4 for this launch script." >&2
  exit 1
fi

REQUESTED_NUM_GPUS="$NUM_GPUS" "$PYTHON" - <<'PY'
import importlib.util
import os
import sys

if importlib.util.find_spec("accelerate") is None:
    raise SystemExit("accelerate is not installed. Run ./setup_a100_env.sh first, then retry.")

import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Run this launcher on the A100 server.")

device_count = torch.cuda.device_count()
print(f"Detected {device_count} CUDA device(s).")
requested_num_gpus = int(os.environ["REQUESTED_NUM_GPUS"])
if device_count < requested_num_gpus:
    raise SystemExit(f"Requested {requested_num_gpus} GPU(s), but only detected {device_count}.")
PY

if [[ -z "${GRADIENT_ACCUMULATION_STEPS:-}" ]]; then
  denom=$((NUM_GPUS * PER_DEVICE_TRAIN_BATCH_SIZE))
  GRADIENT_ACCUMULATION_STEPS=$(((TARGET_EFFECTIVE_BATCH_SIZE + denom - 1) / denom))
  if [[ "$GRADIENT_ACCUMULATION_STEPS" -lt 1 ]]; then
    GRADIENT_ACCUMULATION_STEPS=1
  fi
fi

TRAIN_ARGS=(
  --model "$MODEL" \
  --train-file "$TRAIN_FILE" \
  --validation-file "$VALIDATION_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning-rate "${LEARNING_RATE:-2e-4}" \
  --num-train-epochs "${NUM_TRAIN_EPOCHS:-1}" \
  --dataloader-num-workers "${DATALOADER_NUM_WORKERS:-4}" \
  --eval-steps "${EVAL_STEPS:-500}" \
  --save-steps "${SAVE_STEPS:-500}" \
  --logging-steps "${LOGGING_STEPS:-10}"
)

if [[ "${USE_4BIT:-0}" == "1" ]]; then
  if [[ "$USE_DEEPSPEED" == "1" ]]; then
    echo "USE_DEEPSPEED=1 cannot be combined with USE_4BIT=1 in this launcher." >&2
    exit 1
  fi
  TRAIN_ARGS+=(--use-4bit)
fi

if [[ "$USE_DEEPSPEED" == "1" ]]; then
  if [[ ! -f "$DEEPSPEED_CONFIG" ]]; then
    echo "DeepSpeed config not found: $DEEPSPEED_CONFIG" >&2
    exit 1
  fi
  TRAIN_ARGS+=(--deepspeed "$DEEPSPEED_CONFIG")
fi

if [[ -n "${LORA_R:-}" ]]; then
  TRAIN_ARGS+=(--lora-r "$LORA_R")
fi

if [[ -n "${LORA_ALPHA:-}" ]]; then
  TRAIN_ARGS+=(--lora-alpha "$LORA_ALPHA")
fi

if [[ "$NUM_GPUS" -eq 1 ]]; then
  "$PYTHON" train_qwen_lora.py "${TRAIN_ARGS[@]}"
else
  "$PYTHON" -m accelerate.commands.launch \
    --multi_gpu \
    --num_processes "$NUM_GPUS" \
    --num_machines 1 \
    --mixed_precision bf16 \
    --main_process_port "$MAIN_PROCESS_PORT" \
    train_qwen_lora.py "${TRAIN_ARGS[@]}"
fi
