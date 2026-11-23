#!/usr/bin/env bash
# Push exec_comp source (never runs/ or weights) to the training host.
# Results are pulled back explicitly; this script never deletes remote data.
set -euo pipefail
HOST="${1:-4090-168}"
LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rsync -a --itemize-changes \
  --exclude 'runs/' --exclude '__pycache__/' --exclude '*.pyc' \
  --exclude 'releases/' --exclude 'assets/' --exclude 'deployment/' \
  "$LOCAL/" "$HOST:/root/shengyin/ScaleBFM/exec_comp/"
