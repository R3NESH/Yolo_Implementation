#!/usr/bin/env bash
# Run a command inside the Vitis AI PyTorch container, scoped to the YOLOv8 work.
#
#   ./v8_run.sh python v8_dpu_wrapper.py --weights yolov8n.pt
#   ./v8_run.sh bash                                        # interactive shell
#
# Mirrors yolov3_test/vitis_run.sh, with two differences:
#
#   * `vendor/` (a real ultralytics 8.4.5) goes on PYTHONPATH *ahead* of
#     yolov3_test/vitis_compat, whose `ultralytics` is a deliberate stub with no
#     nn.modules and cannot reconstruct a C2f or a v8 Detect. Order matters.
#   * The COCO data lives outside this clone, so it is bind-mounted read-only at
#     /datasets rather than copied in - the disk is 99% full. This exposes both
#     /datasets/coco (val2017 + annotations) and /datasets/coco_split (the prior
#     work's disjoint 4000/1000 split, used for calibration). Override the host
#     path with DATA_ROOT=... if it moves.
set -euo pipefail

CLONE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${VITIS_IMAGE:-xilinx/vitis-ai-pytorch-cpu:latest}"
DATA_ROOT="${DATA_ROOT:-/home/aesicdab/Desktop/Versal_AI/datasets}"

V8="/workspace/Yolo_v8_Versal_Implementation"
COMPAT="/workspace/yolov3_test/vitis_compat"

TTY_FLAGS=()
[ -t 0 ] && TTY_FLAGS=(-it)

DATA_FLAGS=()
if [ -d "${DATA_ROOT}" ]; then
  DATA_FLAGS=(-v "${DATA_ROOT}:/datasets:ro")
else
  echo "[warn] DATA_ROOT '${DATA_ROOT}' not found - /datasets will be absent" >&2
fi

exec docker run --rm "${TTY_FLAGS[@]}" \
  -u "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e YOLO_CONFIG_DIR=/tmp/Ultralytics \
  -e PYTHONPATH="${V8}/vendor:${V8}:${COMPAT}:${COMPAT}/site" \
  -v "${CLONE_ROOT}:/workspace" \
  "${DATA_FLAGS[@]}" \
  -w "${V8}" \
  "${IMAGE}" \
  bash -lc '
    source /opt/vitis_ai/conda/etc/profile.d/conda.sh
    conda activate vitis-ai-pytorch
    exec "$@"
  ' _ "$@"
