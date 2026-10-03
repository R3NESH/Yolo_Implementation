---
tags: [findings, prior-art, important]
date: 2026-09-11
---

# Prior Work in Yolo_v3_AB

> [!important] There is an earlier, more advanced attempt at this same deployment
> `~/Documents/Yolo_v3_AB/` (43 GB, July–August 2026) contains a previous run at putting this
> model on the VCK190. It **already solved the SiLU problem the same way** this session did, and
> it got further: it has board-side evaluation and debugging scripts. Read it before writing new
> code or spending GPU time.

Related: [[SiLU to LeakyReLU Experiment]], [[DPU Subgraph Fragmentation]],
[[Implementation Plan]], [[Training Run exp2]].

> [!warning] Third folder, also read-only
> This is a **third** location, outside both the original and the clone
> ([[Clone Provenance]]). Treat it as read-only reference material. Nothing in this session
> modified it.

---

## What's there

```
~/Documents/Yolo_v3_AB/                                  43 GB total
└── yolov3_test/
    ├── datasets/coco/                                   27 GB  <- THE TRAINING DATA
    ├── yolo_env/                                        (another venv)
    └── yolov3_test/                                     the project
        ├── quantize.py                                  the July quantize script
        ├── quantize_result/DetectionModel_int.xmodel    138 MB
        ├── compiled_vck190/yolo_vck190.xmodel           34.9 MB, 1 kernel
        ├── runs/train/exp … exp17                       17 training runs
        └── eval_yolo_vck190.py, debug_vck190.py, …       board-side tooling
```

## It reached a single DPU subgraph

`compiled_vck190/meta.json`:

```json
{
    "lib": "libvart-dpu-runner.so",
    "filename": "yolo_vck190.xmodel",
    "kernel": [
        "subgraph_DetectionModel__DetectionModel_Bottleneck_merged_model__Bottleneck_merged_2__Conv_cv2__Conv2d_conv__ret_11"
    ],
    "target": "DPUCVDX8G_ISA3_C32B6"
}
```

**One kernel. Same target string as this session's compiles.** Its generated module shows
`49 py_nndct.nn.LeakyReLU` — a real quantized op class, not the `Module('aten::silu_')` fallback.

This is **independent corroboration** of [[SiLU to LeakyReLU Experiment]]: two separate attempts,
six weeks apart, both found that swapping SiLU for LeakyReLU yields exactly one DPU subgraph.

## How it did the swap — and the alpha answer

From `quantize.py`:

```python
"""Recursively replaces all nn.SiLU instances with DPU-supported nn.LeakyReLU(0.1)."""
setattr(model, name, nn.LeakyReLU(0.1, inplace=True))
```

It wrote `0.1`; the quantizer emitted `negative_slope=0.1015625`. **The quantizer snaps the slope
to 26/256 itself** — which settles the open question in §11 of [[Vitis AI DPU Concepts]] (that
note has been corrected).

> [!warning] But it swapped at *quantize* time, not training time
> Old `models/common.py` still has `default_act = nn.SiLU()`, and no model yaml carries an
> `activation:` key. So the July models were **trained with SiLU** and had LeakyReLU substituted
> immediately before quantization — exactly what `dpu_silu_experiment.py` does here.
>
> **That means the July xmodel has the same accuracy caveat as ours: the weights were never fit
> for the activation actually executed.** The prior work did not close this gap either. The
> finetune remains genuinely outstanding work, not something already done.

## Architectural difference worth noting

July quantized `DetectionModel` — the **whole model, Detect head included** — whereas this session
quantizes the stripped `YOLOv3DPUWrapper`. Both reached one DPU subgraph. Why the unstripped model
also mapped cleanly is not established *(unverified)*; plausibly the head's decode was excluded by
export mode or partitioned into the non-DPU subgraphs. Worth understanding before deciding which
approach to carry forward — the stripped wrapper is the more predictable of the two.

## The training data lives here

`data/train2017_yolo.yaml` in the clone points at:

```
path: /home/aesicdab/Documents/Yolo_v3_AB/yolov3_test/datasets/coco
```

**Full COCO, 27 GB, and it exists.** The clone itself only holds coco128 (128 images), so any
training or full validation reads from this folder. Do not copy it — there is not enough disk
([[Disk and System Constraints]]), and the yaml already resolves correctly.

## Old training runs

Seven of the 17 runs have checkpoints. Final-epoch metrics:

| Run | Epochs | mAP\@0.5 | mAP\@0.5:0.95 | Date |
| --- | --- | --- | --- | --- |
| exp | 65 | 0.0000 | 0.0000 | 2026-07-24 |
| exp2 | 2 | 0.5067 | 0.3084 | 2026-07-24 |
| exp3 | 11 | 0.5226 | 0.3226 | 2026-07-25 |
| exp4 | 6 | 0.4985 | 0.3014 | 2026-07-25 |
| exp5 | 31 | 0.5341 | 0.3279 | 2026-07-28 |
| **exp6** | 11 | **0.5347** | **0.3295** | 2026-07-28 |
| exp17 | 10 | 0.0000 | 0.0000 | 2026-08-02 |

> [!tip] exp6 beats the checkpoint currently being deployed
> exp6 reaches **0.5347 / 0.3295**, against **0.5275 / 0.3228** for
> `Yolo_v3_AB_New_Clone/runs/train/exp2/weights/best.pt` ([[Training Run exp2]]).
>
> Whether they are comparable depends on matching data/config, which has **not** been verified —
> check each run's `opt.yaml` before concluding anything. But it is worth checking: if exp6 is a
> like-for-like better model, it is the better finetune starting point.
>
> `exp` and `exp17` reporting 0.0000 suggests collapsed or misconfigured runs — do not use them.

## Board-side tooling already written

This is the most immediately useful discovery. These scripts exist in the old project:

| Script | Likely purpose |
| --- | --- |
| `eval_yolo_vck190.py` | Evaluate the compiled model, presumably on-board |
| `debug_vck190.py`, `debug_vck190_tensors.py` | Board runtime / tensor-level debugging |
| `visualize_vck190.py`, `visualize_prediction.py` | Draw detections |
| `check_layout.py` | Tensor layout checks — exactly the NHWC/NCHW trap noted in [[Subsystem - Validation]] |
| `test_inference.py`, `test_modified_model.py` | Inference smoke tests |
| `sweep_debug.py`, `sweep_final.py`, `grassroots_debug.py` | Threshold/parameter sweeps |
| `make_test_txt.py` | Dataset list generation |
| `debug_prediction.py` | Post-processing debugging |

`eval_yolo_vck190.py` **has now been read** — 1823 lines, of which ~1250 are commented-out drafts
(the same house style as the `*_AB*.py` scripts). Its live code is `ANCHORS` (line 1575),
`decode_yolo_grid()` (1604), `nms_boxes()` (1651) and `main()` (1676). It genuinely does use VART
and does implement a full decode + NMS + evaluation loop.

**But its decode is wrong for this model** — see the next section. The rest of the script is
reusable. The other scripts have not been read individually.

---

## ⚠️ The old host-side decode is WRONG for this model

> [!danger] This is the most important thing in this note
> `eval_yolo_vck190.py` implements **classic Darknet YOLOv3** decode maths. This model uses the
> **YOLOv5** formulation. The boxes it produces are systematically mispositioned and missized.
> Do not reuse that decode. It is very likely the reason the old project accumulated six
> debugging and sweep scripts.

### What the old code does

`eval_yolo_vck190.py:1634-1642`:

```python
sig_xy = 1.0 / (1.0 + np.exp(-active_pred_surv[:, 0:2]))

# Standard YOLOv3 Coordinate Math (matching the working visual script)
cx = (sig_xy[:, 0] + grid_surv[:, 0]) / grid_size * orig_w
cy = (sig_xy[:, 1] + grid_surv[:, 1]) / grid_size * orig_h

tw_th = np.clip(active_pred_surv[:, 2:4], -3.0, 3.0)
bw = np.exp(tw_th[:, 0]) * anchor_surv[:, 0] / INPUT_WIDTH * orig_w
bh = np.exp(tw_th[:, 1]) * anchor_surv[:, 1] / INPUT_HEIGHT * orig_h
```

### What this model actually requires

From `Detect.forward` in the clone (`models/yolo.py:63-92`), with `grid = meshgrid - 0.5` and
`anchor_grid = anchors * stride`:

```python
xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]   # grid already carries -0.5
wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]
```

### Side by side

| Term | Old script (Darknet) | This model (YOLOv5) | Same? |
| --- | --- | --- | --- |
| centre | `(σ(t) + cell) · stride` | `(2σ(t) − 0.5 + cell) · stride` | ❌ missing `×2` and `−0.5` |
| size | `exp(t) · anchor` | `(2σ(t))² · anchor` | ❌ completely different function |

The centre error is a sub-cell offset and a doubled sensitivity — subtle, produces slightly
shifted boxes. **The size error is severe**: `exp` is unbounded while `(2σ)²` is capped at 4×
anchor, so large predictions blow up and small ones shrink wrongly.

Two smells in the old code that are symptoms of this mismatch, not fixes for it:

- `pred = np.clip(pred, -20.0, 20.0)` at line 1608, "to prevent overflow warnings"
- `tw_th = np.clip(active_pred_surv[:, 2:4], -3.0, 3.0)` at line 1640

Both exist because raw logits are being fed to `exp()`. With the correct `(2σ)²` form, the output
is inherently bounded and neither clip is needed.

> [!note] Also check: does the old script apply objectness thresholding in logit space?
> Line 1610 filters on `pred[..., 4] > LOGIT_OBJ_THRESH` **before** the sigmoid — which is a valid
> optimisation if `LOGIT_OBJ_THRESH` is the logit of the intended probability threshold. Verify
> that constant is set accordingly; if someone set it to `0.25` expecting a probability, the
> effective threshold is `σ(0.25) ≈ 0.56`, which would silently discard most detections.

### What IS still reusable from the old scripts

The decode is wrong, but the surrounding harness is real work worth keeping:

| Reusable | Why |
| --- | --- |
| VART runner setup / tensor plumbing | Board-side boilerplate, known to execute |
| `nms_boxes()` (`eval_yolo_vck190.py:1651`) | Standard IoU NMS, independent of the decode bug |
| The evaluation loop and dataset iteration | Harness structure |
| `check_layout.py` | Tensor-layout verification — the NHWC/NCHW trap |
| `ANCHORS` / grid precomputation (lines 1575-1602) | Structure is fine; values need checking against [[Checkpoint Inspection]] |

**Plan: keep the harness, replace `decode_yolo_grid()` with the correct maths from
[[Subsystem - Validation]].** That is a contained change to one function, and it converts 1800
lines of otherwise-stranded work into a usable host application.

> [!warning] Implication for the accuracy question
> Because the decode was wrong, **any accuracy figure the old project measured on the board is
> untrustworthy** — a bad mAP there does not mean the LeakyReLU swap was the problem. The open
> question about swap-without-finetune accuracy is genuinely still open, and cannot be answered
> from the old results.

> [!success] Closed 2026-09-11
> That question has now been answered properly: full COCO val2017 on the board, with the
> verified decode, gives **mAP@0.5 = 0.0591 against a 0.5275 baseline** — the swap without
> finetuning costs about 89% of accuracy. See
> [[Board mAP - LeakyReLU Without Finetune]]. So the July project's poor board results were
> plausibly *both* the decode bug and a genuinely broken model, and the LeakyReLU path
> definitely needs a real finetune.


## Recommended next action

**Read this folder before writing new code or training anything.** Specifically:

1. **Port the harness, not the maths.** Take `eval_yolo_vck190.py`'s VART plumbing, NMS and
   evaluation loop; replace `decode_yolo_grid()` with the correct YOLOv5 decode from
   [[Subsystem - Validation]]. One function, and the host app is most of the way done.
2. Check `LOGIT_OBJ_THRESH` and the `ANCHORS` values against [[Checkpoint Inspection]].
3. `runs/train/exp5/opt.yaml` and `exp6/opt.yaml` — are these comparable to the current exp2, and
   is exp6 a better finetune seed?
4. `quantize.py` — compare against `quantize_vitis_AB4.py` for anything it does better.
5. Ignore any board accuracy numbers from the old project — the decode bug invalidates them.

This is cheap (reading) and could remove days of duplicated effort.

---

Back to [[Code Map]] | [[Home]]
