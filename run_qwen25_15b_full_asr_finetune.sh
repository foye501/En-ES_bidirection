#!/usr/bin/env bash
set -euo pipefail

export MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
export OUTPUT_DIR="${OUTPUT_DIR:-artifacts/qwen2.5-1.5b-full-asr-bidirectional-lora-r128-bs64}"
export TRAIN_FILE="${TRAIN_FILE:-data/compiled_finetune_dataset_0_asr.jsonl}"
export VALIDATION_FILE="${VALIDATION_FILE:-$TRAIN_FILE}"
export EVAL_STRATEGY="${EVAL_STRATEGY:-no}"
export LORA_R="${LORA_R:-128}"
export LORA_ALPHA="${LORA_ALPHA:-256}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-16}"
export TARGET_EFFECTIVE_BATCH_SIZE="${TARGET_EFFECTIVE_BATCH_SIZE:-64}"
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-1024}"
export LEARNING_RATE="${LEARNING_RATE:-2e-4}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export LOGGING_STEPS="${LOGGING_STEPS:-10}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

exec ./run_a100_finetune.sh
