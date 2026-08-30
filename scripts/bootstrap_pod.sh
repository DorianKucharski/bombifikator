#!/usr/bin/env bash
set -euo pipefail

POD_ROOT="${POD_ROOT:-/root/bombifikator}"
VENV="${POD_ROOT}/.venv"
export HF_HOME="${HF_HOME:-/workspace/hf}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/.cache/pip}"
export HF_HUB_DISABLE_PROGRESS_BARS=1

mkdir -p "${HF_HOME}" "${PIP_CACHE_DIR}"

if [ ! -x "${VENV}/bin/python" ]; then
    python3 -m venv --system-site-packages "${VENV}"
fi

"${VENV}/bin/pip" install --quiet --upgrade pip
"${VENV}/bin/pip" install --quiet \
    "diffusers==0.36.0" transformers accelerate huggingface_hub safetensors \
    ultralytics opencv-python-headless "rembg[gpu]" \
    scikit-learn timm tomli-w typer anthropic

if ! "${VENV}/bin/python" -c "import nunchaku" 2>/dev/null; then
    "${VENV}/bin/pip" install --quiet \
        "https://github.com/nunchaku-tech/nunchaku/releases/download/v1.2.1/nunchaku-1.2.1+cu12.8torch2.8-cp312-cp312-linux_x86_64.whl"
fi

"${VENV}/bin/python" - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download

print("base:", snapshot_download("Qwen/Qwen-Image-Edit-2509", ignore_patterns=["transformer/*"]))
print("transformer:", hf_hub_download("nunchaku-tech/nunchaku-qwen-image-edit-2509",
                                      "svdq-fp4_r128-qwen-image-edit-2509.safetensors"))
PY

"${VENV}/bin/python" -c "from ultralytics import YOLO; YOLO('yolo11x-seg.pt')"
"${VENV}/bin/python" -c "import rembg; rembg.new_session()"

"${VENV}/bin/python" -c "import torch, diffusers, nunchaku; \
    print('ready', torch.__version__, torch.cuda.is_available(), diffusers.__version__)"
