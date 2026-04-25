#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python run.py \
    --config configs/gta.yaml \
    --output outputs/gta/results.json \
    "$@"
