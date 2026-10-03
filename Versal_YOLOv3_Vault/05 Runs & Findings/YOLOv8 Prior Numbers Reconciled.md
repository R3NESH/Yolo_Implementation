---
tags: [findings, yolov8, measurement, reconciliation]
date: 2026-09-14
result: all five generations explained; only _v3 is valid; no decode fault implicated
---

# YOLOv8 Prior Numbers Reconciled

[[YOLOv8 Prior Work in Versal_AI]] left five mutually contradictory generations of board mAP,
one scoring *above* float, and called rescoring the saved predictions "the first task, and it
costs nothing". That task is now done. **No board time was used, and nothing was re-quantized.**

Every generation reproduces exactly, and every contradiction turns out to be a **scoring
artifact, not a hardware or decode result**.

Harness: `tools/rescore_v8_prior.py`, run against
`Versal_AI/datasets/coco/annotations/instances_val2017.json` with `pycocotools` 2.5.0 from
`Versal_AI/venv`.

---

## The verdict

| Generation | Vault recorded (`yolov8n`) | Rescored | Verdict |
| --- | --- | --- | --- |
| `dpu_benchmark` | 0.0248 | **0.0248** | reproduces — genuinely broken, *and* pre-thresholded |
| `_v1` | 0.5495 | — | ❌ **void** — not `yolov8n`, and not the full set |
| `_v2` / `_fixed` | 0.3586 | **0.3586** | reproduces, but depressed by a pre-scoring threshold |
| `_v3` | 0.4317 | **0.4317** | ✅ **the only valid generation** |

## What each contradiction actually was

### 1. `_v1` scoring above float — two compounding errors

`results_v4/` is **not a separate run at all**. It is `results_v3/` truncated to the first
**1015 of 5000 images**. Proven at detection level, not inferred from scores:

```
v4 yolov8m vs v3 yolov8m, restricted to the shared 1015 images:
  images with byte-identical detection sets: 1015/1015
```

On top of that, `predictions_yolov8n_v4.json` and `predictions_yolov8s_v4.json` are the **same
file** (`md5 520c1dc762a0c9e72f4413bda34958b6`, both 25,009,157 bytes) — and that file is the
**`yolov8s`** predictions:

```
v4 "n" file vs v3 yolov8s, identical detection sets: 1015/1015
```

So the `_v1` row reported **`yolov8s`'s detections, on a fifth of the dataset, under the name
`yolov8n`**. Scoring 0.5495 against `yolov8n`'s 0.5187 float baseline was comparing a larger
model on an easier subset to a smaller model on the full set. Nothing beat its own float.

Restricted to those same 1015 images, every `_v1` figure is just a `_v3` figure:

| | `_v1` recorded | `_v3` on the same 1015 images |
| --- | --- | --- |
| `yolov8n` | 0.5495 | 0.5495 — *but this is `yolov8s`* (true n: 0.4548) |
| `yolov8s` | 0.5495 | 0.5495 |
| `yolov8m` | 0.6121 | 0.6121 |
| `yolov8l` | 0.6473 | 0.6473 |

### 2. `_v2` / `_fixed` and the original — a confidence threshold applied before scoring

Both carry a **minimum score of 0.2689** — that is `sigmoid(-1.0)`, so the predictions were
thresholded at a logit of −1 before being written. COCO AP integrates over the full
precision-recall curve, so discarding every detection below 0.269 truncates the curve and
caps AP structurally. `_v3` has a floor of 0.0012, the conventional 0.001.

| File | images | detections | min score |
| --- | --- | --- | --- |
| `results/yolov8n_json/predictions.json` | 4901 | 28,871 | 0.2689 |
| `results_fixed/predictions_yolov8n_fixed.json` | 4921 | 29,664 | 0.2689 |
| `results_v3/predictions_yolov8n_v3.json` | 5000 | **670,203** | **0.0012** |

The original generation is *additionally* broken — 0.0248 is far below what thresholding alone
explains, so that decode really was faulty. `_fixed` repaired the decode (0.0248 → 0.3586) but
kept the threshold; `_v3` removed the threshold too.

---

## The numbers that stand

`_v3`, full 5000 images, threshold 0.0012, HardSwish-for-SiLU, INT8 on DPUCVDX8G:

| Model | float mAP\@0.5 | **board mAP\@0.5** | retained | float mAP\@0.5:0.95 | board |
| --- | --- | --- | --- | --- | --- |
| `yolov8n` | 0.5187 | **0.4317** | 83.2 % | 0.3681 | 0.2917 |
| `yolov8s` | 0.6106 | **0.5301** | 86.8 % | 0.4440 | 0.3724 |
| `yolov8m` | 0.6654 | **0.5935** | 89.2 % | 0.4979 | 0.4278 |
| `yolov8l` | 0.6916 | **0.6219** | 89.9 % | 0.5244 | 0.4528 |

Retention rises monotonically with model size — the pattern you would predict if the loss is
INT8 quantization plus an activation approximation, since larger models carry more redundancy.
Compare [[Board mAP - best.pt on Hardware]], where YOLOv3 with an *exact* SiLU decomposition
retained 91.5 %. These sit just below that, which is the right side of plausible for an
activation that is approximate.

> [!note] The suspected decode fault is no longer implicated
> [[YOLOv8 Prior Work in Versal_AI]] flagged a precision collapse as "the signature of a decode
> or class-mapping fault". Both anomalies that motivated that suspicion — the above-float score
> and the byte-identical `n`/`s` rows — are now fully explained as scoring artifacts, and the
> older generations' precision figures were computed from runs thresholded at 0.269, which
> inflates precision and destroys recall by construction. The `_v3` gap of 10–17 % needs no
> decode fault to explain it. The class-agnostic diagnostic is no longer the priority it was.

## What this changes

- **Do not re-run the board to settle the old numbers.** They are settled.
- `_v3` is the baseline any new work must beat or match. A fresh HardSwish build that lands near
  0.4317 / 0.5301 / 0.5935 / 0.6219 has reproduced prior art correctly.
- `_v3` is still **arm A only** — HardSwish, no finetune. It does not price the approximation,
  because there is no exact-activation run to subtract. That remains the open measurement, and
  is what arm B of [[Implementation Plan - YOLOv8]] exists to produce.
- The `yolov8_leaky_relu_trained_e15` figure of 0.1936 shares the `_v1` coverage defect — it is a
  **1015-image subset score**, and `results_v3/` contains no leaky file, so no full-set run of
  that model exists. Treat 0.1936 as a subset figure, not a val2017 result. (Scoring it over all
  5000 images returns 0.0430, but that only measures the 3985 images it never covered.)

---

Back to [[Implementation Plan - YOLOv8]] | [[YOLOv8 Prior Work in Versal_AI]] | [[Home]]
