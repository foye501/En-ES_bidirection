#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip

if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
  python -m pip install torch --index-url "$TORCH_INDEX_URL"
fi

python -m pip install -r requirements.txt

python - <<'PY'
import sys

import accelerate
import bitsandbytes
import datasets
import peft
import torch
import transformers

print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"accelerate={accelerate.__version__}")
print(f"peft={peft.__version__}")
print(f"datasets={datasets.__version__}")
print(f"bitsandbytes={bitsandbytes.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        print(f"cuda:{index}={torch.cuda.get_device_name(index)}")
elif not bool(int(__import__("os").environ.get("SKIP_CUDA_CHECK", "0"))):
    raise SystemExit("CUDA is not available. Run this on the A100 server or set SKIP_CUDA_CHECK=1 for CPU-only setup checks.")
PY
