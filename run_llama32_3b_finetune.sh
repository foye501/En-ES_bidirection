#!/usr/bin/env bash
set -euo pipefail

export MODEL="${MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
export OUTPUT_DIR="${OUTPUT_DIR:-artifacts/llama3.2-3b-instruct-bidirectional-lora-r64-bs64}"
export TRAIN_FILE="${TRAIN_FILE:-data/finetune_bidirectional/train.jsonl}"
export VALIDATION_FILE="${VALIDATION_FILE:-data/finetune_bidirectional/validation.jsonl}"
export LORA_R="${LORA_R:-64}"
export LORA_ALPHA="${LORA_ALPHA:-128}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
export TARGET_EFFECTIVE_BATCH_SIZE="${TARGET_EFFECTIVE_BATCH_SIZE:-64}"
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-1024}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"

exec ./run_a100_finetune.sh
