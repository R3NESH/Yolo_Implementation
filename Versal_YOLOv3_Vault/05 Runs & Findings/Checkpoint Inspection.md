---
tags: [findings, checkpoint, anchors, reference]
date: 2026-09-11
source: runs/train/exp2/weights/best.pt
---

# Checkpoint Inspection

Everything the host-side application needs to know about the deployed model. Produced by
`inspect_model_AB1.py`; raw log at `yolov3_test/vitis_out/inspect_best.log`.

Related: [[Training Run exp2]], [[Model Architecture Graph]], [[Subsystem - Validation]].

```bash
./vitis_run.sh python inspect_model_AB1.py          # defaults to exp2/best.pt
```

---

## Model parameters

| Property | Value |
| --- | --- |
| Checkpoint | `runs/train/exp2/weights/best.pt` (276,736,084 B) |
| Precision on disk | **fp16** (the wrapper calls `.float()` on load) |
| `nc` (classes) | **80** (COCO) |
| `no` (outputs per anchor) | **85** = 4 box + 1 obj + 80 cls |
| `nl` (detection layers) | 3 |
| `na` (anchors per layer) | 3 |
| Detect module | `model.28`, fed from layers `[27, 22, 15]` |
| Strides | **[8, 16, 32]** |
| Train imgsz | **416** |
| `width_multiple` | 1.0 |

## Output tensors

Verified empirically by running `export_dpu_wrapper_AB3.py`:

| Index | Shape | Stride | Grid | Source layer |
| --- | --- | --- | --- | --- |
| `out0` | `[1, 255, 52, 52]` | 8 | 52×52 | `model.27` |
| `out1` | `[1, 255, 26, 26]` | 16 | 26×26 | `model.22` |
| `out2` | `[1, 255, 13, 13]` | 32 | 13×13 | `model.15` |

255 = `na × no` = 3 × 85.

> [!important] Output ordering is fine-grained but load-bearing
> `out0` is the **finest** grid (stride 8, small objects) and `out2` the **coarsest**
> (stride 32, large objects). The VART host app must pair each output tensor with its matching
> stride and anchor triplet. Getting this backwards produces detections that are silently wrong
> — plausible boxes at wrong scales — rather than an obvious crash.

## Anchors

Stored in the checkpoint in **grid units** (already divided by stride), fp16. Multiply by the
layer's stride to get pixel dimensions:

| Layer | Stride | Grid-unit anchors | **Pixel anchors (w,h)** |
| --- | --- | --- | --- |
| `out0` | 8 | (1.25, 1.625) (2.0, 3.75) (4.125, 2.875) | **(10,13) (16,30) (33,23)** |
| `out1` | 16 | (1.875, 3.8125) (3.875, 2.8125) (3.6875, 7.4375) | **(30,61) (62,45) (59,119)** |
| `out2` | 32 | (3.625, 2.8125) (4.875, 6.1875) (11.65625, 10.1875) | **(116,90) (156,198) (373,326)** |

These are exactly the **stock YOLOv3 COCO anchors** — `noautoanchor` was false but autoanchor
evidently found no reason to change them.

Copy-pasteable for the host app:

```python
ANCHORS = {            # stride: [(w, h), ...] in pixels
     8: [(10, 13),  (16, 30),  (33, 23)],
    16: [(30, 61),  (62, 45),  (59, 119)],
    32: [(116, 90), (156, 198), (373, 326)],
}
STRIDES = [8, 16, 32]  # aligned with out0, out1, out2
```

## Architecture note

The embedded yaml confirms the model **mixes both bottleneck variants** — visible directly in the
compiled kernel names:

- `Bottleneck_merged` in `ModuleList_6` and `ModuleList_8` (backbone stages)
- standard `Bottleneck` in `ModuleList_19`, `20`, `26`, `27` (neck/head stages)

So the "merged23" naming refers to merging applied to specific *backbone* stages only, leaving the
neck intact. See [[Subsystem - Models]] for what the merge actually removes.

## Loading caveats

This checkpoint was saved by torch 2.x with NumPy 2.x; the Vitis AI image has torch 1.13 /
NumPy 1.22. Loading it requires the compat shims — see [[Compat Shims]] and
[[The numpy _core Segfault]]. Without them, `torch.load` either raises
`ModuleNotFoundError: numpy._core` or **segfaults**.

---

Back to [[Code Map]] | [[Home]]
