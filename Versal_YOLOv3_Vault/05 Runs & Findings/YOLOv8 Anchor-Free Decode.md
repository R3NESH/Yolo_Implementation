---
tags: [findings, yolov8, decode, host, verification]
date: 2026-09-14
result: host decode reproduces ultralytics to 6e-5 px on all four models
---

# YOLOv8 Anchor-Free Decode

[[Implementation Plan - YOLOv8]] calls the DFL decode "the one genuinely new piece of
engineering". It is now written and verified against float PyTorch, **before** any board time —
the discipline [[Board Bring-Up]] argues for after three host bugs each crashed on frame one.

Code: `host_vck190/yolov8_decode.py`, verified by
`Yolo_v8_Versal_Implementation/verify_v8_decode.py`.

---

## Why the v3 decode could not be reused

`host_vck190/yolo_decode.py` is anchor-based. YOLOv8 shares none of its assumptions:

| | YOLOv3 (v5-lineage) | YOLOv8 |
| --- | --- | --- |
| Boxes | anchor priors, `(2·σ(t))²·anchor` | **anchor-free**, DFL distribution |
| Objectness | separate channel, multiplied into the score | **none** |
| Channels/cell | 255 = 3 anchors × 85 | **144** = 4×16 DFL bins + 80 classes |
| Grid origin | `meshgrid − 0.5` | cell **centre**, `+0.5` |
| Input | 416 | **640** |

Reusing it would have produced plausible-looking boxes with meaningless geometry — the failure
mode that is hardest to catch from mAP alone.

## The maths

Per scale, the DPU emits `(H, W, 144)` in NHWC. Channels `0..63` are the box distribution laid
out **side-major** (`index = side·16 + bin`, sides = left, top, right, bottom); channels
`64..143` are class logits.

```
softmax over each side's 16 bins, then expectation against bin indices 0..15
    -> ltrb, distance from the cell centre to each edge, in cell units

x1 = (col + 0.5 - l) * stride      x2 = (col + 0.5 + r) * stride
y1 = (row + 0.5 - t) * stride      y2 = (row + 0.5 + b) * stride
score = sigmoid(class logit)                      # no objectness multiply
```

This mirrors ultralytics `DFL.forward` → `dist2bbox` → `make_anchors` exactly. The `+0.5` is
`make_anchors`' `grid_cell_offset`, and the side-major layout comes from `DFL.forward`'s
`view(b, 4, c1, a)` — getting that transposed is the obvious way to produce subtly wrong boxes.

> [!tip] The pre-filter matters
> The DFL softmax is the expensive part. Filtering cells on `max(class logit) > logit(conf)`
> *before* reconstructing any box skips it for the overwhelming majority of the 8400 anchors.

## The verification

Both paths run on **identical features** from the same float checkpoint with SiLU untouched,
then through the **same** NMS, so any disagreement is the decode and nothing else:

```
float model --+-- DetectionModel.forward   -> ultralytics decodes internally -> (1, 84, 8400)
              |
              +-- YOLOv8DPUWrapper(act=silu) -> 3 raw (1,144,H,W) -> yolov8_decode.decode_all
```

Three COCO val2017 images per model, `--conf 0.25`:

| Model | detections (both sides) | max box error | max score error |
| --- | --- | --- | --- |
| `yolov8n` | 9, 1, 7 | **6.1e-5 px** | 6e-8 |
| `yolov8s` | 16, 1, 6 | 6.1e-5 px | 3e-8 |
| `yolov8m` | 17, 1, 6 | 6.1e-5 px | 6e-8 |
| `yolov8l` | 17, 1, 7 | 6.1e-5 px | 3e-8 |

Detection counts and class ids match exactly on every image. The residual is float32 rounding —
at a 640 input, 6e-5 px is roughly one part in ten million of the image width.

> [!warning] Preprocessing must follow calibration, not training
> The decode is only half the contract. `preprocess()` does a **plain resize** to 640 (no
> letterbox), matching `v8_quantize.py` and the prior work's `build/calibrator.py`. Training used
> letterbox. Following calibration is what keeps inference consistent with quantization; see the
> same discussion in [[Board mAP - best.pt on Hardware]]. If calibration ever moves to letterbox,
> the host preprocessing and the box rescaling must move with it.

## Cost: the host decode is not the bottleneck

Stripping the *entire* decode to the host is what buys the single DPU subgraph (the prior work
left the DFL softmax and sigmoid in the graph and got 2). The obvious worry is that the host then
becomes the limit, especially at the `conf 0.001` that mAP scoring requires. It does not:

| conf threshold | decode + NMS | candidates | implied ceiling (x86) |
| --- | --- | --- | --- |
| 0.001 (mAP scoring) | **4.8 ms/img** | 1393 | ~208 FPS |
| 0.05 | 1.2 ms/img | 195 | ~850 FPS |
| 0.25 (demo) | 0.8 ms/img | 72 | ~1200 FPS |

Measured on x86; the board's ARM cores are slower, but even at a 10× penalty the mAP-scoring case
sits near 20 FPS against a DPU that delivers 6–8. The pre-filter is what makes this cheap — cells
are rejected on `max(class logit)` *before* any DFL softmax runs, so the expensive step touches
only the ~1400 candidates rather than all 8400 anchors × 80 classes.

> [!note] Not yet confirmed on hardware
> These are x86 numbers. The board has been powered down for this work, so the real host-side
> cost on the VCK190's ARM cores is still unmeasured. The margin is wide enough that it is
> unlikely to bite, but it is an estimate, not a measurement.

## Shared harness

`board_eval_vck190.py` now takes `--arch {v3,v8}` and late-binds the decode module. Everything
below the decode — VART plumbing, runner selection, NMS, COCO scoring, CSV output — is shared, so
the two model families are measured by identical post-processing. `yolov8_decode` declares
`NA = 1` so the harness's `NA * NO` channel check resolves to 144 without special-casing.

---

Back to [[Implementation Plan - YOLOv8]] | [[Host Application]] | [[Home]]
