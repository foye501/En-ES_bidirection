#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
TORCH_VERSION="${TORCH_VERSION:-2.9.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

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

python -m pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
python -m pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX_URL"

python -m pip install -r requirements.txt

if [[ "${INSTALL_FLASH_ATTN:-0}" == "1" ]]; then
  python -m pip install --upgrade packaging ninja wheel setuptools
  MAX_JOBS="${FLASH_ATTN_MAX_JOBS:-8}" python -m pip install flash-attn --no-build-isolation
fi

python - <<'PY'
import sys

import accelerate
import bitsandbytes
import datasets
import deepspeed
import peft
import torch
import transformers
try:
    import flash_attn
except ImportError:
    flash_attn = None

print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"torch_cuda_build={torch.version.cuda}")
print(f"transformers={transformers.__version__}")
print(f"accelerate={accelerate.__version__}")
print(f"peft={peft.__version__}")
print(f"datasets={datasets.__version__}")
print(f"bitsandbytes={bitsandbytes.__version__}")
print(f"deepspeed={deepspeed.__version__}")
print(f"flash_attn={flash_attn.__version__ if flash_attn else 'not installed'}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        print(f"cuda:{index}={torch.cuda.get_device_name(index)}")
elif not bool(int(__import__("os").environ.get("SKIP_CUDA_CHECK", "0"))):
    raise SystemExit("CUDA is not available. Run this on the A100 server or set SKIP_CUDA_CHECK=1 for CPU-only setup checks.")
PY
