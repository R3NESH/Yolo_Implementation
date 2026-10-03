#!/usr/bin/env bash
# Run a command inside the Vitis AI PyTorch container, scoped to this clone.
#
#   ./vitis_run.sh python inspect_model_AB1.py
#   ./vitis_run.sh bash                        # interactive shell
#
# Only this repository is mounted (at /workspace), so nothing else on the host
# is visible to the container. `vitis_compat/` supplies the offline
# `ultralytics` shim and seaborn, which the Vitis AI image does not ship.
set -euo pipefail

CLONE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${VITIS_IMAGE:-xilinx/vitis-ai-pytorch-cpu:latest}"
COMPAT="/workspace/yolov3_test/vitis_compat"

TTY_FLAGS=()
[ -t 0 ] && TTY_FLAGS=(-it)

exec docker run --rm "${TTY_FLAGS[@]}" \
  -u "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e PYTHONPATH="${COMPAT}:${COMPAT}/site" \
  -v "${CLONE_ROOT}:/workspace" \
  -w /workspace/yolov3_test \
  "${IMAGE}" \
  bash -lc '
    source /opt/vitis_ai/conda/etc/profile.d/conda.sh
    conda activate vitis-ai-pytorch
    exec "$@"
  ' _ "$@"
