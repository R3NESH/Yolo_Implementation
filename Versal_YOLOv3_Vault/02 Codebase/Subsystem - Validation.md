---
tags: [codebase, validation, metrics, nms, decode, reference]
date: 2026-09-11
---

# Subsystem - Validation

`val.py`, `utils/metrics.py`, and the box/NMS helpers in `utils/general.py`.

> [!important] This note is the reference for the Versal host application
> The DPU returns **raw conv outputs**. Anchor decode and NMS must be reimplemented on the host
> CPU, and they must match this code exactly or detections will be subtly, silently wrong.
> §"Decode specification" below is the portable spec.

Modules: [[val]], [[utils.metrics]], [[utils.general]], [[models.yolo]].
Related: [[Vitis AI DPU Concepts]], [[Checkpoint Inspection]], [[Implementation Plan]].

---

## Decode specification (port this)

The head's decode lives in `Detect.forward`, [models/yolo.py:63-92](../../yolov3_test/models/yolo.py).

> [!danger] This is the YOLOv5-style decode, NOT classic YOLOv3
> Classic Darknet YOLOv3 uses `wh = anchor * exp(tw)`. **This model does not.** It uses the
> YOLOv5 `(2σ)²` formulation. Porting the classic YOLOv3 formula — the obvious thing to reach for
> given the model's name — produces wrong boxes that still look plausible.

The live code:

```python
xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
xy = (xy * 2 + self.grid[i]) * self.stride[i]   # xy
wh = (wh * 2) ** 2 * self.anchor_grid[i]        # wh
y = torch.cat((xy, wh, conf), 4)
```

with, from `_make_grid` ([models/yolo.py:94-106](../../yolov3_test/models/yolo.py)):

```python
grid = torch.stack((xv, yv), 2).expand(shape) - 0.5          # NOTE the -0.5
anchor_grid = (self.anchors[i] * self.stride[i]).view(...)   # anchors in PIXELS
```

Two easily-missed details:

1. **The `-0.5` is baked into the grid**, not written in the `xy` line. Effective formula:
   `xy = (2σ(t_xy) - 0.5 + cell_index) * stride`.
2. **`anchor_grid` is pre-multiplied by stride**, so anchors are in *pixels* — matching the table
   in [[Checkpoint Inspection]]. The values stored in the checkpoint are in *grid units*.

### Portable pseudocode

For each output tensor `i` of shape `[1, 255, ny, nx]`:

```python
# 1. reshape 255 -> (3 anchors, 85 outputs)
t = out[i].reshape(1, 3, 85, ny, nx).transpose(0, 1, 3, 4, 2)   # -> [1,3,ny,nx,85]

# 2. sigmoid EVERYTHING (xy, wh, obj, cls) - one call, before the split
t = sigmoid(t)

# 3. boxes, in 416-pixel space
#    cx, cy are the cell column/row indices
xy = (t[..., 0:2] * 2 - 0.5 + cell_index_xy) * stride[i]   # CENTRE, pixels
wh = (t[..., 2:4] * 2) ** 2 * anchor_px[i]                 # WIDTH/HEIGHT, pixels

# 4. scores - already sigmoided in step 2
obj = t[..., 4:5]
cls = t[..., 5:85]
score = obj * cls                     # per-class confidence
```

Then flatten all three scales together, threshold, convert centre-wh → xyxy, and run NMS.

> [!tip] Sanity checks when porting
> - Total candidates before threshold = `3*(52² + 26² + 13²)` = **10 647**.
> - `xy` must land in roughly `[0, 416]`. Values clustered near 0 usually mean a missing stride
>   multiply; values ~2× too large usually mean a double stride multiply.
> - Boxes are **centre-format**, not corner-format, until `xywh2xyxy`.

## `non_max_suppression`

[utils/general.py:1008](../../yolov3_test/utils/general.py):

```python
def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nm=0,          # number of masks
):
    """Returns: list of detections, on (n,6) tensor per image [xyxy, conf, cls]"""
```

| Parameter | Meaning | Note for the host port |
| --- | --- | --- |
| `prediction` | `[B, 10647, 85]` (or a tuple — it takes `[0]`) | assemble all three scales into this |
| `conf_thres` | 0.25 default | `val.py` uses **0.001** for mAP; use 0.25 for demos |
| `iou_thres` | 0.45 default | `val.py` uses **0.6** for mAP |
| `classes` | filter to a class subset | |
| `agnostic` | class-agnostic NMS | keep `False` to match training-time eval |
| `multi_label` | allow multiple labels per box | `val.py` enables this for mAP |
| `max_det` | 300 | cap on detections per image |

> [!warning] Threshold choice changes the numbers, not just the picture
> mAP is computed at `conf_thres≈0.001` to capture the full precision/recall curve. A
> demo-tuned 0.25 will produce far fewer detections and a much lower apparent mAP. When comparing
> board results against PC results, use identical thresholds on both sides.

Box helpers to port alongside it: `xywh2xyxy`, `xyxy2xywh`, `scale_boxes` (maps boxes back from
416-space to original image coordinates — needed if preprocessing letterboxed, see
[[Subsystem - Utils Core]]), and `clip_boxes`.

## `val.py`

`run()` is the evaluation entry point. Flow:

1. Load model (`DetectMultiBackend`) and dataset (`create_dataloader`, `rect=True`, `pad=0.5`).
2. Per batch: forward → `non_max_suppression` → match predictions to labels at 10 IoU
   thresholds (0.50 : 0.05 : 0.95) → accumulate a boolean correctness matrix.
3. `ap_per_class` → precision, recall, AP per class → mAP\@0.5 and mAP\@0.5:0.95.
4. Optionally write `confusion_matrix.png`, PR curves, and COCO-format JSON.

Validating the current checkpoint:

```bash
python val.py --weights runs/train/exp2/weights/best.pt \
              --data data/val2017_yolo.yaml \
              --img 416 --batch-size 8 --task val
```

> [!note] Not runnable on this machine
> `val.py` needs the full torch stack, which is not installed here
> ([[Disk and System Constraints]]) and would be slow in the CPU-only container. Run it wherever
> the LeakyReLU finetune happens.

## `utils/metrics.py`

| Function / class | Role |
| --- | --- |
| `ap_per_class` | Precision/recall curves → AP per class, via 101-point interpolation |
| `compute_ap` | AP from one precision/recall curve |
| `ConfusionMatrix` | Confusion matrix + plot |
| `bbox_iou` | IoU with GIoU/DIoU/CIoU variants (also used by the training loss) |
| `box_iou` | Pairwise IoU matrix — the core of NMS and of label matching |
| `fitness` | The scalar that selects `best.pt`: `0.1·mAP@0.5 + 0.9·mAP@0.5:0.95` |

`fitness` is why [[Training Run exp2]] reports epoch 34 as "best" — it weights mAP\@0.5:0.95 nine
times more heavily than mAP\@0.5.

## Validating the deployment

Three distinct comparisons, in increasing strictness:

1. **Float PC vs quantized PC** — isolates quantization loss. Run the quantizer's own
   `quant_model` on the host and compare against the float model.
2. **Quantized PC vs board** — isolates compile/runtime bugs. Outputs should match near-exactly;
   if they don't, suspect preprocessing or tensor layout, not quantization.
3. **Board mAP vs float mAP** — the end-to-end number that actually matters.

`export_xmodel(deploy_check=True)` dumps golden per-layer tensors specifically to support
comparison 2. It was left off here to save disk ([[Quantization and Compile Results]]) but is worth
enabling once space allows.

---

Back to [[Code Map]] | [[Home]]
