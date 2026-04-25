#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python run.py \
    --config configs/mm.yaml \
    --output outputs/mm/results.json \
    "$@"
