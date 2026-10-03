---
tags: [findings, yolov8, board, hardware, results, headline]
date: 2026-09-14
result: all four YOLOv8 models run on the VCK190; every one beats the prior work on mAP@0.5
---

# YOLOv8 on Hardware

**All four stock YOLOv8 checkpoints now run on the VCK190.** Full val2017, 5000 images each,
HardSwish (arm A), letterbox preprocessing, host-side anchor-free DFL decode.

The YOLOv8 counterpart to [[Board mAP - best.pt on Hardware]]. Pipeline in
[[YOLOv8 Quantization and Compile]], decode in [[YOLOv8 Anchor-Free Decode]], and the numbers
these replace in [[YOLOv8 Prior Numbers Reconciled]].

---

## The numbers

| Model | mAP\@0.5 | mAP\@0.5:0.95 | mAP\@0.75 | AR\@100 | detections |
| --- | --- | --- | --- | --- | --- |
| `yolov8n` | **0.4453** | 0.2913 | 0.3072 | 0.4949 | 794,255 |
| `yolov8s` | **0.5423** | 0.3716 | 0.3931 | 0.5603 | 637,641 |
| `yolov8m` | **0.6001** | 0.4256 | 0.4565 | 0.6052 | 773,508 |
| `yolov8l` | **0.6233** | 0.4467 | 0.4807 | 0.6181 | 957,300 |

Against the float baselines, and against the only trustworthy prior generation (`_v3`):

| Model | float mAP\@0.5 | **board** | retained | prior `_v3` | **Δ vs prior** |
| --- | --- | --- | --- | --- | --- |
| `yolov8n` | 0.5187 | 0.4453 | 85.9 % | 0.4317 | **+0.0136** |
| `yolov8s` | 0.6106 | 0.5423 | 88.8 % | 0.5301 | **+0.0122** |
| `yolov8m` | 0.6654 | 0.6001 | 90.2 % | 0.5935 | **+0.0066** |
| `yolov8l` | 0.6916 | 0.6233 | 90.1 % | 0.6219 | **+0.0014** |

**Every model beats the prior work on mAP\@0.5**, and retention rises with model size exactly as
it did there. `yolov8n` at 11.6 FPS against the prior art's 7.0 — the single DPU subgraph and
batch-6 slot filling, with the host decode costing 4.8 ms/image and never becoming the limit.

## What is better, and why

| Change | Effect |
| --- | --- |
| **Letterbox** instead of a plain square resize | the largest share — worth +0.021 mAP in float, +0.047 after quantization ([[YOLOv8 Letterbox and the Activation Cost]]) |
| **Whole decode moved to the host** | 1 DPU subgraph instead of 2, `fix2float` the only CPU op, ~1.65× throughput |
| **Verified decode** | matches ultralytics to 6e-5 px, and the letterbox inverse to 3e-5 px across ten aspect ratios |

## The half-pixel bug, and what it was actually worth

The first board pass used a letterbox inverse that subtracted the **float** `dw/dh` rather than
the **integer** pad `cv2.copyMakeBorder` actually applied — a systematic sub-pixel shift on every
image whose padding lands on a half pixel, roughly half of COCO. Fixed to match ultralytics'
`scale_boxes`, and re-run:

| Model | mAP\@0.5:0.95 before | after | Δ |
| --- | --- | --- | --- |
| `yolov8n` | 0.291044 | 0.291348 | +0.0003 |
| `yolov8s` | 0.370781 | 0.371572 | +0.0008 |
| `yolov8m` | 0.424272 | 0.425569 | +0.0013 |
| `yolov8l` | 0.445045 | 0.446702 | +0.0017 |

Monotonic in model size, as predicted — sharper models lose more from a fixed localisation error.
But an order of magnitude smaller than predicted.

> [!warning] A wrong attribution, recorded so it is not repeated
> The bug was diagnosed *from aggregate metrics*: mAP\@0.5 improved on all four models while
> mAP\@0.5:0.95 fell, and a sub-pixel shift is exactly the kind of fault that shows up only at
> strict IoU. The reasoning was sound and the bug was real — but it accounts for only about a
> quarter of the deficit it was invoked to explain. **Diagnosing a cause from the shape of an
> aggregate is a hypothesis, not a finding.**

## The open question

After the fix, mAP\@0.5:0.95 is still slightly *below* the prior work, and the gap widens with
model size:

| Model | remaining Δ vs `_v3` |
| --- | --- |
| `yolov8n` | −0.0004 |
| `yolov8s` | −0.0008 |
| `yolov8m` | −0.0022 |
| `yolov8l` | −0.0061 |

So this pipeline **finds objects better** (mAP\@0.5 up on all four) and **localises marginally
worse** at strict IoU. No confirmed explanation. Candidates, none tested:

- board_eval writes `bbox` rounded to 2 decimals; the prior work wrote full float precision.
  0.01 px is far too small to matter at IoU 0.95, so this is unlikely but trivial to rule out.
- the prior build kept the DFL softmax and sigmoid **inside** the graph, so the two pipelines
  differ in where and at what precision the box distribution is reduced.
- the two letterbox implementations may differ in the rounding of `new_unpad`.

> [!tip] The right next diagnostic
> Match detections between the two prediction sets on the same images and compare **box
> coordinates directly**, rather than reasoning from aggregate metrics again — which is how the
> first attribution went wrong. Both prediction sets are saved (`board_results/` for the current
> runs, `board_results/v1_float_pad/` for the pre-fix pass, `Versal_AI/results_v3/` for the prior
> work), so this costs no board time.

## Plots

`board_results/plots_yolov8{n,s,m,l}/` — five per model, from `tools/board_curves.py`, the same
set `val.py` draws for a PC run so the board is comparable like-for-like: `PR_curve`, `P_curve`,
`R_curve`, `F1_curve`, `confusion_matrix`.

`board_results/plots_summary/` — the cross-model story no single run contains, from
`tools/plot_v8_summary.py`:

| File | What it shows |
| --- | --- |
| `v8_accuracy_by_model.png` | mAP\@0.5 and mAP\@0.5:0.95 per model — float ceiling, this deployment, prior work |
| `v8_activation_vs_quantization.png` | the 2×2 as a slope chart: the gap between lines is the activation swap, each line's slope is INT8 |
| `v8_summary.csv` | the same numbers as a table |

> [!tip] Why the decomposition is a slope chart and not bars
> The claim is a comparison of two distances — the vertical gap against the slope. Grouped bars
> forced that comparison into annotation arrows that had to cross unrelated bars to reach their
> targets. In a slope chart the geometry *is* the argument, with nothing to annotate around.

## Two independent scorers agree

The discipline [[yolov3_original on Hardware]] established: never trust one scorer.
`tools/board_curves.py` does its own single-threshold IoU matching rather than calling
pycocotools, so the two share no matching code.

| Model | pycocotools | `board_curves.py` | Δ |
| --- | --- | --- | --- |
| `yolov8n` | 0.4453 | 0.4430 | 0.0023 |
| `yolov8s` | 0.5423 | 0.5391 | 0.0032 |
| `yolov8m` | 0.6001 | 0.5976 | 0.0025 |
| `yolov8l` | 0.6233 | 0.6211 | 0.0022 |

Agreement within 0.003 on all four, in the same direction each time — consistent with
`board_curves` matching at a single IoU while pycocotools sweeps ten, not with a fault in either.

Best-F1 operating points, for demos rather than scoring:

| Model | best F1 | at confidence |
| --- | --- | --- |
| `yolov8n` | 0.468 | 0.269 |
| `yolov8s` | 0.548 | 0.377 |
| `yolov8m` | 0.598 | 0.320 |
| `yolov8l` | 0.613 | 0.377 |

## Reproducing

```bash
# host: build and gate
cd Yolo_v8_Versal_Implementation
./v8_run.sh python -u v8_quantize.py --weights yolov8n.pt --act hardswish --quant_mode calib
./v8_run.sh python -u v8_quantize.py --weights yolov8n.pt --act hardswish --quant_mode test
./v8_compile.sh yolov8n hardswish
./deploy_to_board.sh

# board
python3 board_eval_vck190.py --arch v8 \
    --model yolov8n_hardswish_vck190.xmodel \
    --images val2017 --annotations ann/instances_val2017.json \
    --out-json preds_yolov8n_hs.json --out-csv metrics_yolov8n_hs.csv
```

`--arch v8` is required and letterbox is the default for it. Then, on the host:

```bash
V=Versal_AI/venv/bin/python
A=Versal_AI/datasets/coco/annotations/instances_val2017.json
$V tools/board_curves.py "$A" board_results/preds_yolov8n_hs.json \
    board_results/plots_yolov8n "yolov8n INT8 HardSwish on VCK190"
$V tools/plot_v8_summary.py
```

---

Back to [[Implementation Plan - YOLOv8]] | [[YOLOv8 Letterbox and the Activation Cost]] | [[Home]]
