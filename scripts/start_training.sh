#!/usr/bin/env bash
set -euo pipefail

POD="${POD:-runpod}"
RUN_NAME="${RUN_NAME:-bombifikator_identity_lora_v5}"
CONFIG="${CONFIG:-training/configs/identity_lora.yaml}"
POD_CONFIG="/root/training/configs/$(basename ${CONFIG})"
POD_LOG="/root/logs/${RUN_NAME}.log"
STALE_OUTPUT="${STALE_OUTPUT:-}"

ssh "${POD}" "mkdir -p $(dirname ${POD_CONFIG})"
rsync -rlt "${CONFIG}" "${POD}:${POD_CONFIG}"

if [ -n "${STALE_OUTPUT}" ]; then
    ssh "${POD}" "rm -rf /root/training/output/${STALE_OUTPUT}"
fi

ssh "${POD}" "mkdir -p /root/logs; cd /root/ai-toolkit; HF_HOME=/root/hf setsid nohup .venv/bin/python run.py ${POD_CONFIG} > ${POD_LOG} 2>&1 < /dev/null &"

echo "started ${RUN_NAME}, log ${POD_LOG}"
