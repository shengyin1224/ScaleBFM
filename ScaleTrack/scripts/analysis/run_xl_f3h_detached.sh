#!/usr/bin/env bash
# Resilient systemd entry point for the formal XL-F3h run.
#
# A fresh run warm-starts from the official XL checkpoint.  If systemd restarts
# this service after a later failure, the newest readable checkpoint is resumed
# with exactly enough additional PPO updates to reach update 20,000.

set -euo pipefail

TRACK=/root/shengyin/ScaleBFM/ScaleTrack
PYTHON=/root/miniconda3/envs/isaaclab_sonic/bin/python
RUN_NAME=formal_xl_f3h_20260804
RUN_DIR="$TRACK/logs/rsl_rl/g1_hat4_focus_finetune/$RUN_NAME"
LOG="$TRACK/logs/launch_logs/$RUN_NAME.log"
BASE="$TRACK/logs/rsl_rl/g1_bfm_tracking_exp/humanoid_transformer_xl/model_22200.pt"
TRAIN_MANIFEST="$TRACK/data/hat4_unifolm/manifests/unifolm_hat4_train.yaml"
HELDOUT_MANIFEST="$TRACK/data/hat4_unifolm/manifests/unifolm_hat4_heldout.yaml"
TARGET_ITERATIONS=20000

mkdir -p "$RUN_DIR" "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

export CUDA_VISIBLE_DEVICES=0,1,2,3,5
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export OMP_NUM_THREADS=4

checkpoint_state="$($PYTHON - "$RUN_DIR" <<'PY'
import re
import sys
from pathlib import Path

import torch

run_dir = Path(sys.argv[1])
candidates = []
for path in run_dir.glob("model_*.pt"):
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    if match:
        candidates.append((int(match.group(1)), path))
for _, path in sorted(candidates, reverse=True):
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
        iteration = int(state["iter"])
    except Exception as error:
        print(f"ignoring unreadable checkpoint {path}: {error}", file=sys.stderr)
        continue
    print(f"{path.name}\t{iteration}")
    break
PY
)"

launch_mode=fresh
extra_args=(--pretrained_checkpoint "$BASE" --max_iterations "$TARGET_ITERATIONS")
if [[ -n "$checkpoint_state" ]]; then
    IFS=$'\t' read -r checkpoint_name completed_iterations <<<"$checkpoint_state"
    remaining_iterations=$((TARGET_ITERATIONS - completed_iterations))
    if (( remaining_iterations <= 0 )); then
        echo "[$(date --iso-8601=seconds)] already complete: $checkpoint_name iter=$completed_iterations"
        exit 0
    fi
    launch_mode=resume
    extra_args=(
        --resume True
        --load_run "$RUN_NAME"
        --checkpoint "$checkpoint_name"
        --max_iterations "$remaining_iterations"
    )
fi

echo "[$(date --iso-8601=seconds)] launch_mode=$launch_mode checkpoint_state=${checkpoint_state:-none}"
cd "$TRACK"
exec "$PYTHON" -m torch.distributed.run \
    --standalone \
    --nproc_per_node=5 \
    scripts/pretrain/rsl_rl/train.py \
    --distributed \
    --task G1-HAT4-F3H-Finetune \
    --motion_file "$TRAIN_MANIFEST" \
    --test_motion_file "$HELDOUT_MANIFEST" \
    --num_envs 819 \
    --run_name "$RUN_NAME" \
    --headless \
    "${extra_args[@]}" \
    agent.policy.embedding_dim=384 \
    agent.policy.num_heads=6 \
    agent.policy.ff_dim=384 \
    agent.policy.num_layers=6
