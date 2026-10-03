---
tags: [findings, accuracy, map, board, silu, resolved, important]
date: 2026-09-11
result: 0.4829 mAP@0.5 - 91.5% of the float baseline
---

# Board mAP - best.pt on Hardware

The headline deployment number: **`runs/train/exp2/weights/best.pt`, quantized to INT8 and
executed on the VCK190, scores mAP@0.5 = 0.4829** against a float baseline of 0.5275.

**91.5% of the float model's accuracy, running on the DPU.**

Related: [[SiLU Decomposition]], [[Board mAP - LeakyReLU Without Finetune]], [[Board Bring-Up]],
[[Training Run exp2]], [[Host Application]].

---

## The measurement

Full COCO **val2017, all 5000 images**, `compiled_silu_decomp/yolov3_vck190_silu_decomp.xmodel`
via `GraphRunner`:

```bash
python3 board_eval_vck190.py --model yolov3_vck190_silu_decomp.xmodel \
    --images val2017 --annotations ann/instances_val2017.json --conf-thres 0.001
```

```
mAP@0.50       : 0.4829
mAP@0.50:0.95  : 0.2639
mAP@0.75       : 0.2567
Recall (AR@100): 0.4516
detections kept: 1,007,510
```

Raw output in `yolov3_test/board_results/` (`silu_metrics.csv`, `silu_decomp_val2017.log`).

## What it cost to get here

| Model | mAP@0.5 | mAP@0.5:0.95 | Retained |
| --- | --- | --- | --- |
| PC float `best.pt` (exp2, 416, fp16) | 0.5275 | 0.3228 | — |
| **Board INT8, best.pt (SiLU decomposed)** | **0.4829** | **0.2639** | **91.5% / 81.8%** |
| Board INT8, LeakyReLU, no finetune | 0.0591 | 0.0188 | 11% / 6% |

### The split, at last

The two losses that were conflated in [[Board mAP - LeakyReLU Without Finetune]] separate
cleanly now, because this run is *the same INT8 pipeline with the correct activation*:

```
quantization + preprocessing cost = 0.5275 - 0.4829 = 0.0446   (8.5% relative)
activation-swap cost              = 0.4829 - 0.0591 = 0.4238   (88% relative)
```

> [!success] INT8 was never the problem
> Quantization plus the known preprocessing mismatch costs **8.5%**. Swapping SiLU for
> LeakyReLU without finetuning costs **88%**. The earlier worry that quantization might be
> carrying more of the blame than assumed is settled: it is not.

## The detections-kept number is a health check

| Build | Detections kept over 5000 images |
| --- | --- |
| LeakyReLU, no finetune | 1,500,000 = **300 × 5000** — the `max-det` cap, every image |
| **best.pt decomposed** | **1,007,510** — about 201/image, comfortably under the cap |

Saturating the cap on *every* image is what a network emitting near-uniform noise looks like.
Coming in under it means the model is discriminating. Worth checking on any future run before
looking at mAP at all — it is a one-line sanity test.

## A clue about the preprocessing bug

Accuracy is retained unevenly across IoU thresholds:

| Metric | Retained |
| --- | --- |
| mAP@0.50 (loose IoU) | **91.5%** |
| mAP@0.50:0.95 (averaged) | 81.8% |

Objects are being **found** but their boxes are **slightly misplaced** — loose-IoU matching
survives, tight-IoU matching degrades. That is the expected signature of the letterbox vs
plain-resize mismatch documented in [[Host Application]]: training used aspect-preserving
letterbox, calibration and inference use a plain square resize, so box geometry is distorted on
non-square images — which is most of COCO.

**This is the most promising remaining accuracy work, and it needs no GPU.** Switch calibration
*and* inference to letterbox together, add the inverse padding/scale correction when mapping
boxes back, requantize, recompile.

> [!warning] Two different rulers - the 8.5% figure is approximate
> The 0.5275 baseline comes from Ultralytics' own mAP implementation in `val.py`
> (`runs/val/exp14`, best.pt at 416 with `--half`). The board figure comes from **pycocotools**.
> These are not the same computation and can differ by a percentage point or two on the same
> predictions.
>
> For a like-for-like comparison, re-run the float model with pycocotools scoring:
>
> ```bash
> ./vitis_run.sh python val.py --weights runs/train/exp2/weights/best.pt \
>     --data data/train2017_yolo.yaml --img 416 --save-json
> ```
>
> Until that is done, treat "8.5% quantization cost" as the right order of magnitude rather
> than a precise figure.

## Cost of the run

| Phase | Time |
| --- | --- |
| Inference + decode + NMS, 5000 images | **1:26:38** (1.04 s/img, 0.96 FPS) |
| ...of which DPU + preprocess | 957 ms/img (92%) |
| ...of which decode + NMS | ~83 ms/img (8%) |
| `COCOeval` per-image evaluation | 346 s |
| `COCOeval` accumulate | 97 s |

Throughput is the open problem, not accuracy — see the comparison table in
[[SiLU Decomposition]]. This build runs at roughly **1 FPS**; the single-subgraph LeakyReLU
build runs at 56 FPS but is useless. Closing that gap is what the finetune is for.

## Plots

`val.py` draws PR / P / R / F1 curves and a confusion matrix automatically for a PC run. The
board evaluation only emits a COCO-format predictions JSON, so `tools/board_curves.py` rebuilds
the equivalent plots from it. Output in `yolov3_test/board_results/plots/`, directly comparable
with `runs/val/exp14/` (the float baseline - see [[Training Run exp2]]).

```bash
cd yolov3_test && ./vitis_run.sh python ../tools/board_curves.py \
    _tmp_eval/instances_val2017.json board_results/silu_preds.json \
    board_results/plots "best.pt INT8 on VCK190"
```

> [!tip] Do not call pycocotools just to draw curves
> The first attempt dumped COCOeval's internal precision/score arrays from the board, because
> pycocotools is not installed in the Vitis container. That takes **~8 minutes** - COCOeval
> re-matches every detection across 10 IoU thresholds x 80 classes x 4 area ranges, and the
> curves only need one IoU threshold.
>
> Doing the matching directly in numpy at IoU 0.50 takes **8.8 seconds** for the same 1,007,510
> detections. It is also closer to how Ultralytics draws its own curves, so the plots compare
> more fairly against `runs/val/exp14/`.

### The independent implementation agrees

`board_curves.py` does its own greedy highest-confidence-first matching and 101-point
interpolated AP, with no shared code with pycocotools:

| Implementation | mAP@0.5 |
| --- | --- |
| pycocotools (`board_eval_vck190.py`) | 0.4829 |
| `tools/board_curves.py` (own matching) | **0.4774** |

Agreement to 0.0055 (1.1% relative) from two independent implementations is a useful check that
neither the decode, the matching, nor the AP integration has a gross bug. The residual
difference is expected: the two use different interpolation and area-range handling.

### Operating threshold

```
best F1 0.486 at confidence 0.229
```

So **`--conf-thres 0.23`** is the threshold that maximises F1 for this build - the one to use for
demos and live inference. The `0.001` used for mAP is a scoring convention, not an operating
point, and it is also what made NMS expensive on the broken LeakyReLU build.

## Where this leaves the project

**There is now a working, accurate deployment of YOLOv3 on the VCK190**, obtained without any
GPU time. What remains is:

1. **Letterbox fix** — no GPU, probably recovers part of the 8.5%.
2. **Better calibration** — no GPU. Currently 100 coco128 images; should be a few hundred from
   the real training distribution.
3. **Like-for-like float baseline** — no GPU, one `val.py --save-json` run.
4. **Speed** — for this build the **DPU is 92% of frame time** (957 ms of 1040 ms); decode and
   NMS are only ~83 ms. Optimising the CPU post-processing would buy ~8%. The 50-subgraph
   fragmentation is the cost, so item 5 is the real speed lever ([[Board Bring-Up]]).
5. **The LeakyReLU finetune** — needs a real GPU, and it is a large job (see
   [[Board mAP - LeakyReLU Without Finetune]]). It buys throughput, not accuracy.
6. ~~**The actual research question**~~ — **answered 2026-09-12**, see
   [[yolov3_original on Hardware]]. On the same 79 classes: stock 0.559 float / 0.4681 board,
   merged23 0.525 float / 0.4829 board. The merged variant loses in float and wins on hardware.

---

Back to [[Code Map]] | [[Home]]
