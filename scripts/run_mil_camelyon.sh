#!/bin/bash
# Train MMPL on CAMELYON16 (ResNet-50 features) over multiple seeds.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

GPU=${GPU:-0}
CONFIG=${CONFIG:-config/camelyon16_r50.yaml}
SEEDS=${SEEDS:-"0 1"}

for seed in $SEEDS
do
    echo "Running ${CONFIG} with seed=${seed} on GPU=${GPU}"
    CUDA_VISIBLE_DEVICES=${GPU} python src/main.py ${CONFIG} experiment.seed=${seed}
done
