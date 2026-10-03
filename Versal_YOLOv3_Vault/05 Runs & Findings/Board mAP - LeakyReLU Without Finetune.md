---
tags: [findings, accuracy, map, board, leakyrelu, resolved]
date: 2026-09-11
result: catastrophic - 89% of mAP@0.5 lost
---

# Board mAP - LeakyReLU Without Finetune

The open question from [[Prior Work in Yolo_v3_AB]] — *how much accuracy does swapping SiLU for
LeakyReLU cost if you do not finetune?* — measured on real hardware, with a decode that has been
verified against the model's own head.

**Answer: it destroys the model.**

Related: [[SiLU to LeakyReLU Experiment]], [[Board Bring-Up]], [[Training Run exp2]],
[[DPU Subgraph Fragmentation]], [[Host Application]].

---

## The measurement

Full COCO **val2017, all 5000 images**, on the VCK190, using
`compiled_lrelu/yolov3_vck190_lrelu.xmodel`:

```bash
python3 board_eval_vck190.py --model yolov3_vck190_lrelu.xmodel \
    --images val2017 --annotations ann/instances_val2017.json --conf-thres 0.001
```

```
mAP@0.50       : 0.0591
mAP@0.50:0.95  : 0.0188
mAP@0.75       : 0.0062
Recall (AR@100): 0.1003
```

Raw output kept in `yolov3_test/board_results/` (`lrelu_metrics.csv`, `lrelu_val2017.log`).

## Against the float baseline

| Metric | Board, LeakyReLU, no finetune | PC float best.pt | Fraction retained |
| --- | --- | --- | --- |
| mAP@0.50 | **0.0591** | 0.5275 | **11%** |
| mAP@0.50:0.95 | **0.0188** | 0.3228 | **6%** |

**Roughly 89% of mAP@0.5 is gone.** This is not a degradation to be tuned away at the margins;
the network is no longer doing its job.

### The detection count says the same thing

```
images processed: 5000 | detections kept: 1500000
```

`1500000 = 5000 × 300`, and 300 is the `--max-det` cap. **Every single image saturated the
cap.** A healthy detector at `conf 0.001` produces a long tail of low-confidence boxes, but not a
uniform 300 on all 5000 images. This is a model emitting noise at roughly uniform confidence.

## Why this matters

### 1. It closes the question the July project could not answer

[[Prior Work in Yolo_v3_AB]] found that the earlier attempt reached one DPU subgraph by the same
activation swap, and also never finetuned. But its host-side decode used Darknet YOLOv3 maths
against a YOLOv5-formulation model, so **every accuracy number it produced was meaningless** —
a bad score there could have been the decode, not the model.

This run used the decode verified to ~1e-4 against the model's own PyTorch head
([[Host Application]]). The number is trustworthy. Option C in
[[DPU Subgraph Fragmentation]] is now confirmed as diagnostic-only, with evidence.

### 2. It resizes the finetune job

The hope in [[SiLU to LeakyReLU Experiment]] was that because the weights are already trained and
`exp2` was itself a near-converged finetune, "a finetune should recover most of the accuracy."
Starting from 11% of baseline, that is optimistic. Recovering 0.059 → ~0.50 mAP@0.5 is closer to
a retrain than a touch-up.

Planning consequence: a 3-epoch run on a small subset will **not** settle whether the LeakyReLU
path works. It needs real GPU time on real data.

> [!warning] Two losses are still conflated here
> This figure contains **both** the INT8 quantization loss **and** the activation-swap loss, and
> they cannot be separated from this run alone. It is conceivable — though unlikely at this
> magnitude — that quantization is carrying more of the blame than assumed.
>
> **The separating experiment is to run `compiled/` (the SiLU build, from best.pt) on the board
> via `GraphRunner`.** That is the same INT8 quantization with the *correct* activation, so:
>
> ```
> activation-swap cost  =  mAP(compiled, SiLU)  -  mAP(compiled_lrelu, LeakyReLU)
> quantization cost     =  0.5275  -  mAP(compiled, SiLU)
> ```
>
> Until that runs, treat 89% as "swap + quantization combined", not "swap alone".

### 3. It strengthens the GraphRunner path

`compiled/` — built from `best.pt`, SiLU weights under SiLU activation — is the **only** build
currently known to be accurate. It is fragmented into 52 DPU subgraphs and so needs
`vitis_ai_library.GraphRunner`, and it will be slow, but slow and correct beats fast and blind.
See the revised table in [[DPU Subgraph Fragmentation]].

## Cost of the run

About 100 minutes end to end on the board:

| Phase | Time |
| --- | --- |
| Inference + decode + NMS, 5000 images | ~92 min (~1.1 s/image) |
| `COCOeval` per-image evaluation | 283 s |
| `COCOeval` accumulate | 78 s |

The inference phase is CPU-bound, not DPU-bound — see the bottleneck section in
[[Board Bring-Up]]. The scoring phase is slow specifically *because* the model is bad: 1.5M
detections is the worst case for `pycocotools`. A good model would score far faster.

---

Back to [[Code Map]] | [[Home]]
