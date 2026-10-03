#!/usr/bin/env bash
# Compile a quantized YOLOv8 .xmodel for the VCK190, then gate it before any board time.
#
#   ./v8_compile.sh yolov8n hardswish
#
# Takes the model stem and the activation arm, finds the INT8 xmodel that v8_quantize.py
# exported, runs vai_c_xir against the VCK190 arch, and immediately reports the subgraph
# count and CPU op types via tools/inspect_xmodel.py.
#
# That last step is the gate, not a formality: 'SiLU Decomposition' records a build that
# compiled cleanly and then aborted on the board because no libvart_op_imp_aten__silu_.so
# exists. Compiling proves nothing about running. Read the CPU op list before copying
# anything across.
set -euo pipefail

STEM="${1:-yolov8n}"
ACT="${2:-hardswish}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARCH="/opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json"
QUANT_DIR="quantize_result/${STEM}_${ACT}"
OUT_DIR="compiled/${STEM}_${ACT}"
NAME="${STEM}_${ACT}_vck190"

INT_XMODEL="${HERE}/${QUANT_DIR}/YOLOv8DPUWrapper_int.xmodel"
if [ ! -f "${INT_XMODEL}" ]; then
  echo "[ERROR] no quantized xmodel at ${QUANT_DIR}/ - run v8_quantize.py --quant_mode test first" >&2
  exit 1
fi

echo "[INFO] compiling ${QUANT_DIR} -> ${OUT_DIR}"
"${HERE}/v8_run.sh" vai_c_xir \
  -x "${QUANT_DIR}/YOLOv8DPUWrapper_int.xmodel" \
  -a "${ARCH}" \
  -o "${OUT_DIR}" \
  -n "${NAME}"

echo
echo "[GATE] inspecting the compiled graph - check this before spending board time"
"${HERE}/v8_run.sh" python /workspace/tools/inspect_xmodel.py "${OUT_DIR}/${NAME}.xmodel"
