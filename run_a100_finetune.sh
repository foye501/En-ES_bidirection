#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-4}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
TARGET_EFFECTIVE_BATCH_SIZE="${TARGET_EFFECTIVE_BATCH_SIZE:-32}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-1024}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
TRAIN_FILE="${TRAIN_FILE:-data/finetune/train.jsonl}"
VALIDATION_FILE="${VALIDATION_FILE:-data/finetune/validation.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/qwen3-4b-instruct-2507-en-es-lora}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
PYTHON="${PYTHON:-python}"

if [[ "$NUM_GPUS" -lt 2 || "$NUM_GPUS" -gt 4 ]]; then
  echo "NUM_GPUS must be between 2 and 4 for this launch script." >&2
  exit 1
fi

if [[ -z "${GRADIENT_ACCUMULATION_STEPS:-}" ]]; then
  denom=$((NUM_GPUS * PER_DEVICE_TRAIN_BATCH_SIZE))
  GRADIENT_ACCUMULATION_STEPS=$(((TARGET_EFFECTIVE_BATCH_SIZE + denom - 1) / denom))
  if [[ "$GRADIENT_ACCUMULATION_STEPS" -lt 1 ]]; then
    GRADIENT_ACCUMULATION_STEPS=1
  fi
fi

"$PYTHON" -m accelerate.commands.launch \
  --multi_gpu \
  --num_processes "$NUM_GPUS" \
  --num_machines 1 \
  --mixed_precision bf16 \
  --main_process_port "$MAIN_PROCESS_PORT" \
  train_qwen_lora.py \
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
