#!/usr/bin/env bash
set -euo pipefail

POD="${POD:-runpod}"
POD_ROOT="${POD_ROOT:-/root/bombifikator}"

ssh "${POD}" "mkdir -p ${POD_ROOT}/data/input ${POD_ROOT}/data/references/cards ${POD_ROOT}/data/codex/characters"
rsync -rlt --delete \
    --exclude data --exclude .venv --exclude .git --exclude __pycache__ --exclude '*.pyc' \
    ./ "${POD}:${POD_ROOT}/"
rsync -rlt config/.env "${POD}:${POD_ROOT}/config/.env"
rsync -rlt data/input/ "${POD}:${POD_ROOT}/data/input/"
rsync -rlt data/references/cards/ "${POD}:${POD_ROOT}/data/references/cards/"
rsync -rlt data/codex/characters/ "${POD}:${POD_ROOT}/data/codex/characters/"
