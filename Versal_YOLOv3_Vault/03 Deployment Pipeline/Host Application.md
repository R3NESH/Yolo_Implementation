---
tags: [deployment, host, vart, decode, done, yolov8]
date: 2026-09-11
status: running on hardware
---

# Host Application

The board-side code: `host_vck190/` in the clone. Written by porting the July project's VART
harness and **replacing its broken decode** with maths verified against the model itself.

It now **runs on a real VCK190**. Getting there took three bug fixes that off-board verification
could not have caught, because none of them are in the decode — they are all in the VART
plumbing. Those are documented in [[Board Bring-Up]]; the summary is: keep the `xir.Graph`
alive, allocate `int8` buffers, and quantize the input by `2**fix_point`.

Related: [[Board Bring-Up]], [[Prior Work in Yolo_v3_AB]], [[Subsystem - Validation]],
[[Implementation Plan]], [[Vitis AI DPU Concepts]].

---

## Files

| File | Runs where | Purpose |
| --- | --- | --- |
| `host_vck190/yolo_decode.py` | board **and** desktop | Pure-NumPy **YOLOv3** decode (anchor-based), NMS, clipping, preprocessing. No torch, no VART — so it is testable off-board. |
| `host_vck190/yolov8_decode.py` | board **and** desktop | Pure-NumPy **YOLOv8** decode (anchor-free, DFL) plus letterbox. Imports NMS and clipping from `yolo_decode`, so post-processing is shared — [[YOLOv8 Anchor-Free Decode]]. |
| `host_vck190/board_eval_vck190.py` | **board** | VART runner + COCO scoring + benchmark mode. `--arch {v3,v8}` selects the decode. |
| `host_vck190/verify_decode.py` | container/desktop | Proves the YOLOv3 decode matches the model's own PyTorch head. |
| `host_vck190/rescore_shifted.py` | desktop | Remaps a finished predictions JSON by a class offset and rescores, without re-inferring. |

> [!note] One harness, two model families
> `--arch` defaults to `v3`, so every command recorded in this vault still means what it did.
> The decode module is late-bound at startup; everything below it — VART plumbing, runner
> selection, NMS, COCO scoring, CSV output — is shared, so the two families are measured by
> identical post-processing and their numbers are directly comparable.
>
> **`--arch v8` is required for YOLOv8 models.** Without it the harness runs the anchor-based
> 416 decode and rejects the 144-channel tensors.

## The decode is verified, not assumed

`verify_decode.py` takes the wrapper's raw conv outputs and decodes them twice — once in PyTorch
using the model's **own** `grid`/`anchor_grid` buffers and formula, once through
`yolo_decode`'s NumPy path via the NHWC layout the DPU produces — then compares.

```
anchors g=52  ok: [[10, 13], [16, 30], [33, 23]]
scale g=52  stride=8    max|dxy|=3.052e-05  max|dwh|=3.052e-05  max|dconf|=5.960e-08
anchors g=26  ok: [[30, 61], [62, 45], [59, 119]]
scale g=26  stride=16   max|dxy|=3.052e-05  max|dwh|=9.155e-05  max|dconf|=5.960e-08
anchors g=13  ok: [[116, 90], [156, 198], [373, 326]]
scale g=13  stride=32   max|dxy|=3.052e-05  max|dwh|=9.155e-05  max|dconf|=5.960e-08
RESULT: MATCH - numpy decode is correct
total candidates across scales: 10647 (expect 10647)
```

Worst disagreement ~9e-5 — float32 noise across two different operation orders. This checks the
**maths and the axis/channel ordering together**, which is where hand-written decodes normally
fail silently.

Run it any time the decode is touched:

```bash
cd yolov3_test && ./vitis_run.sh python ../host_vck190/verify_decode.py
```

## What was wrong before, and how much it mattered

The July script used Darknet YOLOv3 formulas. Same script, quantified on this model's real
outputs:

```
old Darknet formula vs correct, on the finest scale (pixels):
  correct  mean w=32.8  h=34.8
  old/exp  mean w=46.4  h=47.8
  ratio    w=1.42x h=1.37x
```

**Boxes ~40% too large on average.** Full analysis in [[Prior Work in Yolo_v3_AB]].

## Using it on the board

Copy `host_vck190/` and a compiled xmodel across, then:

```bash
# smoke test - 50 images, no scoring, just "does it produce detections"
python3 board_eval_vck190.py --model yolov3_vck190_lrelu.xmodel \
    --images val2017 --limit 50 --no-eval

# full COCO scoring
python3 board_eval_vck190.py --model yolov3_vck190_lrelu.xmodel \
    --images val2017 --annotations instances_val2017.json

# throughput
python3 board_eval_vck190.py --model yolov3_vck190_lrelu.xmodel \
    --images val2017 --benchmark 200
```

For a YOLOv8 model, add `--arch v8` (the input size, channel layout and decode all follow from it):

```bash
python3 board_eval_vck190.py --arch v8 \
    --model yolov8n_hardswish_vck190.xmodel \
    --images val2017 --annotations instances_val2017.json
```

`Yolo_v8_Versal_Implementation/deploy_to_board.sh` copies the host code and compiled xmodels
across and prints these commands with the right filenames substituted.

Needs `xir`, `vart`, `cv2`, `numpy` on the board; `pycocotools` only for scoring.

### Built-in guards

- **Warns if the xmodel has more than one DPU subgraph** — the [[DPU Subgraph Fragmentation]]
  failure mode, caught at runtime instead of being mistaken for a slow board.
- **Accepts NHWC or NCHW** for both input and output; detects which from the tensor dims.
- **Output order does not matter** — anchors are keyed off grid size, so the three tensors can
  arrive in any order.
- **Dequantizes properly** via each output tensor's `fix_point` attribute
  (`scale = 1/2**fix_point`). Forgetting this yields garbage scores.
- Prints the PC float baseline (mAP\@0.5 0.5275 / 0.3228) next to the measured numbers.

## Known caveat: preprocessing

`yolo_decode.preprocess()` does a **plain resize**, not letterbox.

- Training used `letterbox` (aspect-preserving + padding) — [[Subsystem - Utils Core]].
- `quantize_vitis_AB4.py` calibrated with a plain `Resize((416,416))`.

The host follows the **calibration**, so inference matches how the INT8 scales were derived. That
is the self-consistent choice, but it means the deployed pipeline differs from training
preprocessing, which costs some accuracy on non-square images.

> [!tip] Fix this properly at the next requantization
> Switch calibration *and* inference to letterbox together, and add the inverse padding/scale
> correction when mapping boxes back. Do not change one without the other.

## Status

| Item | State |
| --- | --- |
| Decode maths | ✅ verified against PyTorch |
| NMS (per-class), clipping | ✅ ported and unit-tested |
| VART plumbing | ✅ **runs on hardware** — 3 bugs fixed first ([[Board Bring-Up]]) |
| COCO scoring harness | ✅ written; full val2017 run in progress |
| Benchmark mode | ✅ **26.5 FPS** measured (DPU + preprocess) |
| Preprocessing | ⚠️ plain resize — matches calibration, not training |
| On-board execution | ✅ 50 images, 464 detections, no crashes |
| DPU batch utilisation | ⚠️ **1 of 6 slots** — see below |
| Decode/NMS throughput | ✅ ~83 ms/frame on the accurate build (8% of frame) — [[Board Bring-Up]] |

### Two known inefficiencies

**Batch.** The DPU reports `DPU Batch Number: 6` and its input tensor is `(6, 416, 416, 3)`.
`infer()` fills `in_buf[0][0]` only, so five sixths of each DPU invocation is wasted. Filling
all six slots is the obvious next optimisation.

**Decode and NMS.** How much these matter depends on the model. On the un-finetuned LeakyReLU
build they dominated (~1042 ms vs 18 ms of DPU) because it saturated the detection cap on every
image. On the accurate `compiled_silu_decomp` build they are only ~83 ms against 957 ms of DPU
time — **8% of the frame**. A well-behaved model produces few candidates and NMS is cheap. See
the corrected table in [[Board Bring-Up]].

---

Back to [[Code Map]] | [[Home]]
