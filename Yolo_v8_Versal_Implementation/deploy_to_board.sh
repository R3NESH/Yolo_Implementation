#!/usr/bin/env bash
# Copy the host application and compiled YOLOv8 xmodels to the VCK190.
#
#   ./deploy_to_board.sh                       # everything compiled so far
#   ./deploy_to_board.sh yolov8n_hardswish     # just one build
#
# Assumes the board is up and networked per 'Board Bring-Up':
#
#   host  : the USB ethernet interface holds 192.168.1.20/24
#   board : ip addr add 192.168.1.10/24 dev eth0 && ip link set eth0 up
#
# Note scp needs -O. The PetaLinux image ships no sftp-server, so a modern OpenSSH client
# fails with "/usr/libexec/sftp-server: No such file or directory" without it.
set -euo pipefail

BOARD="${BOARD:-root@192.168.1.10}"
DEST="${DEST:-/home/root/yolo}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLONE="$(cd "${HERE}/.." && pwd)"

if ! ping -c 1 -W 2 "${BOARD#*@}" >/dev/null 2>&1; then
  echo "[ERROR] ${BOARD#*@} is not responding." >&2
  echo "        Check the board is powered, booted (do NOT press a key during U-Boot" >&2
  echo "        autoboot - it drops to the Versal> prompt and never boots), and that" >&2
  echo "        eth0 has 192.168.1.10/24." >&2
  exit 1
fi

echo "[INFO] board is up: ${BOARD}"
ssh "${BOARD}" "mkdir -p ${DEST}"

echo "[INFO] copying host application"
scp -O "${CLONE}/host_vck190/yolo_decode.py" \
       "${CLONE}/host_vck190/yolov8_decode.py" \
       "${CLONE}/host_vck190/board_eval_vck190.py" \
       "${BOARD}:${DEST}/"

BUILDS=("$@")
if [ ${#BUILDS[@]} -eq 0 ]; then
  mapfile -t BUILDS < <(cd "${HERE}/compiled" 2>/dev/null && ls -d */ 2>/dev/null | tr -d /)
fi

if [ ${#BUILDS[@]} -eq 0 ]; then
  echo "[warn] nothing compiled yet under ${HERE}/compiled - copied the host code only"
  exit 0
fi

for b in "${BUILDS[@]}"; do
  x="${HERE}/compiled/${b}/${b}_vck190.xmodel"
  if [ ! -f "$x" ]; then
    echo "[warn] no xmodel for '${b}', skipping" >&2
    continue
  fi
  echo "[INFO] copying ${b} ($(du -h "$x" | cut -f1))"
  scp -O "$x" "${BOARD}:${DEST}/"
done

echo
echo "[INFO] board free space:"
ssh "${BOARD}" "df -h ${DEST} | tail -1"

cat <<EOF

Deployed to ${BOARD}:${DEST}

On the board, smoke test first (50 images, no scoring):

  cd ${DEST}
  python3 board_eval_vck190.py --arch v8 \\
      --model yolov8n_hardswish_vck190.xmodel \\
      --images val2017 --limit 50 --no-eval

Then the full 5000-image scored run:

  python3 board_eval_vck190.py --arch v8 \\
      --model yolov8n_hardswish_vck190.xmodel \\
      --images val2017 --annotations instances_val2017.json \\
      --out-json preds_yolov8n_hardswish.json \\
      --out-csv metrics_yolov8n_hardswish.csv

Throughput on its own:

  python3 board_eval_vck190.py --arch v8 \\
      --model yolov8n_hardswish_vck190.xmodel \\
      --images val2017 --benchmark 200

--arch v8 is required. Without it the harness uses the anchor-based YOLOv3 decode at 416
and will reject the 144-channel tensors.
EOF
