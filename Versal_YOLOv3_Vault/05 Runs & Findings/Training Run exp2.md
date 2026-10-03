---
tags: [runs, training, exp2, metrics]
date: 2026-09-11
source: runs/train/exp2
---

# Training Run exp2

The run that produced the checkpoint everything downstream uses:
`runs/train/exp2/weights/best.pt`. Related: [[Checkpoint Inspection]], [[Subsystem - Training]],
[[SiLU to LeakyReLU Experiment]].

> [!summary] The short version
> exp2 was a **finetune**, not a fresh train — it started from `yolov3_merged23_e75.pt`, which
> already had 75 epochs behind it. It asked for 75 more epochs and **completed only 35** before
> being stopped. Over those 35 epochs mAP@0.5 moved 0.510 → 0.528. The model is essentially
> converged, and the remaining headroom from more epochs of the same recipe is small.

---

## Configuration

From `runs/train/exp2/opt.yaml`:

| Setting | Value | Note |
| --- | --- | --- |
| `weights` | `yolov3_merged23_e75.pt` | **finetune seed** — 75 prior epochs |
| `cfg` | `''` (empty) | architecture came from the checkpoint, not a yaml |
| `data` | `data/train2017_yolo.yaml` | see [[Subsystem - Data and Datasets]] |
| `epochs` | 75 | **only 35 ran** |
| `batch_size` | 8 | small — likely a VRAM constraint |
| `imgsz` | 416 | matches the DPU pipeline's 416 |
| `optimizer` | SGD | |
| `patience` | 100 | effectively no early stopping |
| `seed` | 0 | |
| `rect`, `multi_scale`, `cos_lr`, `quad` | false | |
| `freeze` | `[0]` | nothing frozen |
| `save_dir` | `runs/train/exp2` | `name: exp` auto-incremented to `exp2` |

Hyperparameters (`hyp.yaml`) are the stock **scratch-low** recipe: `lr0: 0.01`, `lrf: 0.01`,
`momentum: 0.937`, `weight_decay: 0.0005`, 3 warmup epochs, loss weights `box 0.05 / obj 1.0 /
cls 0.5`, augmentation `mosaic 1.0`, `fliplr 0.5`, `scale 0.5`, `translate 0.1`, HSV jitter;
`mixup` and `copy_paste` off.

## Metric trajectory

35 epochs recorded (0–34) in `results.csv`:

| epoch | box | obj | cls | P | R | mAP@0.5 | mAP@0.5:0.95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.0479 | 0.0552 | 0.0371 | 0.643 | 0.467 | 0.5096 | 0.3109 |
| 5 | 0.0525 | 0.0583 | 0.0284 | 0.604 | 0.460 | 0.4946 | 0.2995 |
| 10 | 0.0513 | 0.0574 | 0.0266 | 0.618 | 0.473 | 0.5087 | 0.3101 |
| 15 | 0.0508 | 0.0569 | 0.0259 | 0.623 | 0.481 | 0.5142 | 0.3140 |
| 20 | 0.0504 | 0.0565 | 0.0253 | 0.631 | 0.484 | 0.5193 | 0.3171 |
| 25 | 0.0500 | 0.0563 | 0.0247 | 0.636 | 0.484 | 0.5218 | 0.3192 |
| 30 | 0.0496 | 0.0561 | 0.0241 | 0.634 | 0.490 | 0.5258 | 0.3219 |
| **34** | 0.0493 | 0.0559 | 0.0237 | **0.641** | **0.489** | **0.5275** | **0.3228** |

**Best epoch = 34**, the last one — by YOLOv5's fitness metric (`0.1·mAP50 + 0.9·mAP50-95`).
So `best.pt` and `last.pt` are the same epoch, which is why the two files are byte-identical in
size (276,736,084 bytes each).

Shape of the curve:
- Epoch 0 already sits at mAP@0.5 ≈ 0.51 — the seed checkpoint was well-trained.
- Epochs 1–5 **dip** (mAP@0.5 down to 0.495). Classic finetune disturbance: warmup plus a fresh
  `lr0: 0.01` knocks the model off its converged point.
- Epochs 6–34 recover and climb steadily to 0.5275.
- Net gain over 35 epochs: **+0.018 mAP@0.5, +0.012 mAP@0.5:0.95** — and roughly half of that was
  just recovering from the dip it caused itself.

> [!warning] `val/*_loss` columns are all zero
> Every row records `val/box_loss = val/obj_loss = val/cls_loss = 0` while the mAP columns are
> populated. Validation metrics were computed; validation *losses* were not recorded. Treat the
> val-loss columns as unusable, and do not read the zeros as "no validation ran". Cause not
> investigated *(unverified)*.

## Why `runs/train/exp/weights` is empty

The evidence, not a guess:

`runs/train/exp/` contains `opt.yaml`, `hyp.yaml`, a TensorBoard events file, and an **empty**
`weights/` directory. No `results.csv`.

- The directory and its config files are written at **startup**, before training begins.
- `results.csv` is appended at the **end of each epoch**.
- Checkpoints are saved at the end of each epoch too.

So `exp` started, wrote its config, created `weights/`, and died **before completing a single
epoch**. Nothing was ever saved, and no results row was ever written.

The TensorBoard timestamps corroborate the retry: `exp` at unix `1786297169`, `exp2` at
`1786297859` — **690 seconds (~11.5 min) apart**. Someone launched a run, it failed ~11 minutes
in, and they immediately relaunched as `exp2`. The failure cause is not recoverable from these
artefacts *(unverified)* — the events file would need inspecting, and no stdout log was kept.

**Conclusion: `exp` is a dead run. `exp2` is the only real one. Ignore `exp` entirely.**

## Other artefacts present

`labels.jpg`, `labels_correlogram.jpg` (dataset label distribution), `train_batch0/1/2.jpg`
(augmented training samples — useful for eyeballing whether mosaic/HSV look sane), and the
TensorBoard events file. No `confusion_matrix.png` or PR curves, which YOLOv5 writes at the
*end* of training — further confirmation the run was cut short rather than finishing its 75 epochs.

## Is this checkpoint worth deploying?

Honestly: **as an integration target yes, as a product no** — but the activation problem
dominates that judgement anyway.

- mAP@0.5 ≈ 0.53 / mAP@0.5:0.95 ≈ 0.32 on 80-class COCO at 416 is a plausible-but-modest YOLOv3
  result. It is not broken, and it is not good.
- The run was truncated at 35/75 epochs, so the recipe was never finished.
- The gains were flattening but had not plateaued — more epochs would likely add a little.

None of that is the deciding factor. [[DPU Subgraph Fragmentation]] means the model **must** be
finetuned with LeakyReLU before it can run usefully on the DPU at all. That finetune is the
opportunity to also finish the interrupted schedule properly. Doing both at once is strictly
better than deploying this checkpoint as-is.

---

Back to [[Code Map]] | [[Home]]
