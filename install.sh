#!/usr/bin/env bash
# One-time setup for say (GPU text-to-speech, Kokoro-82M) in this directory.
# Idempotent: re-run to repair. First run downloads the CUDA build of torch
# (about 3 GB) and the model (about 330 MB) so the first `say` is instant.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYVER=3.12                                  # kokoro 0.9.x wants >=3.10,<3.13
# TORCH_INDEX: where torch comes from. Default: the CUDA 12.8 build; on macOS, plain PyPI.
if [ "$(uname -s)" = Darwin ]; then TORCH_INDEX="${TORCH_INDEX:-https://pypi.org/simple}"; fi
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

echo "== venv (Python $PYVER) =="
uv venv .venv --python "$PYVER" --quiet
PY=.venv/Scripts/python.exe; [ -x "$PY" ] || PY=.venv/bin/python   # Windows, else Linux/macOS

echo "== torch ($TORCH_INDEX) =="
uv pip install --python "$PY" --quiet torch --index-url "$TORCH_INDEX"

echo "== kokoro + audio =="
uv pip install --python "$PY" --quiet "kokoro==0.9.4" sounddevice soundfile

echo "== pre-fetching the model =="
"$PY" -c "from kokoro import KPipeline; KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M'); print('model cached')"

echo "== check =="
"$PY" -c "import torch; ok=torch.cuda.is_available(); print('torch', torch.__version__, '| cuda', ok, '|', torch.cuda.get_device_name(0) if ok else 'CPU ONLY')"
echo "done. try:  say -w 'hello from the GPU'"
