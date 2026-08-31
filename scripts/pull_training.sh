#!/usr/bin/env bash
set -euo pipefail
POD="${POD:-runpod}"
RUN_NAME="${RUN_NAME:-bombifikator_identity_lora_v2}"
POD_OUTPUT="/root/training/output/${RUN_NAME}"
mkdir -p data/loras data/training/samples
rsync -avz --progress "${POD}:${POD_OUTPUT}/*.safetensors" data/loras/
rsync -avz "${POD}:${POD_OUTPUT}/samples/" data/training/samples/
