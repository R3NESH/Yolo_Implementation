---
tags: [findings, yolov8, preprocessing, quantization, activation, measurement]
date: 2026-09-14
result: INT8 sim reproduces the board to 0.004 mAP; letterbox worth +0.021; HardSwish costs 3x quantization
---

# YOLOv8 Letterbox and the Activation Cost

Two results from evaluating `yolov8n` off-board while the VCK190 was powered down, on a fixed
500-image stride sample of val2017. Both were found by comparing against the *real* board
predictions from [[YOLOv8 Prior Numbers Reconciled]], scored on the identical image subset.

Code: `Yolo_v8_Versal_Implementation/v8_eval_quantized.py`, scored with
`tools/rescore_v8_prior.py`.

---

## 1. The INT8 simulation reproduces the board

`vai_q_pytorch`'s `test` mode simulates INT8 on the CPU. Run end to end — wrapper, activation
swap, quantization, host decode, NMS, COCO mapping — it lands on the hardware result:

| | mAP\@0.5 | mAP\@0.5:0.95 |
| --- | --- | --- |
| **My INT8 simulation** (HardSwish, letterbox) | **0.4714** | 0.3197 |
| **Real board**, `_v3` generation | **0.4672** | 0.3234 |
| difference | 0.0042 | 0.0037 |

Agreement to under half a point, from an independently built pipeline against a board run made
months earlier by different code. This is the strongest available evidence that the wrapper, the
DFL decode ([[YOLOv8 Anchor-Free Decode]]) and the category mapping are all correct.

> [!tip] This makes board-free iteration possible
> The board is intermittently available; the simulation is not. Any change to the wrapper,
> calibration or decode can now be evaluated to within ~0.004 mAP without hardware. It is still
> a *prediction*: it does not exercise VART, the `fix_point` input scaling, or NHWC buffers —
> which is where [[Board Bring-Up]] found all three of its bugs. Use it to iterate, not to
> replace the final board run.

## 2. Letterbox, not plain resize

The plan inherited a rule from the YOLOv3 work — *"preprocessing must match calibration: the v3
host uses plain resize, not letterbox"*. The principle is right; the specific instruction cost
real accuracy here.

Float `yolov8n`, SiLU, same 500 images:

| Preprocessing | mAP\@0.5 | mAP\@0.5:0.95 |
| --- | --- | --- |
| plain resize to 640×640 | 0.5466 | 0.3782 |
| **letterbox** (aspect-preserving, 114-grey pad) | **0.5674** | **0.3980** |
| | **+0.0208** | **+0.0198** |

A plain resize squashes every image to a square and distorts each object's aspect ratio. The gain
is larger still after quantization: INT8 HardSwish went **0.4247 → 0.4714** with letterbox, a
gain of 0.047.

### How this was caught

Not by inspection — by a number that could not be explained. Scored on identical images, the
board (0.4672) beat my *float* HardSwish model (0.4355). Quantization cannot make a model better
than its own float, so something upstream differed. It was
`Versal_AI/src/runtime/dpu_runner.py`, which letterboxes.

> [!note] The prior work was internally inconsistent, and still ahead
> It calibrated with a plain resize (`build/calibrator.py`) but ran inference with letterbox —
> exactly the mismatch the vault warns against — and still beat a consistent plain-resize
> pipeline. **Preserving aspect ratio is worth more than the calibration mismatch costs.**

### Calibration preprocessing barely matters; inference preprocessing does

Worth stating separately, because it determines how much work a change implies. The 0.4714 run
used a model calibrated with **plain resize** and merely *inferred* with letterbox, and it still
matched the board. So switching the host to letterbox does **not** require re-calibrating or
re-compiling anything — it is a host-side change only.

`v8_quantize.py` now letterboxes by default anyway (`--no-letterbox` to revert), so calibration
and inference agree, satisfying the principle without paying for it.

## 3. What HardSwish actually costs

The measurement [[Implementation Plan - YOLOv8]] set out to obtain. Both arms were run, and
because each was also run in float, the result is a complete **2×2**: activation × precision, all
letterbox, all on the same 500 images. mAP\@0.5:

| | **float** | **INT8** | quantization costs |
| --- | --- | --- | --- |
| **SiLU** (exact) | **0.5674** | 0.5431 *(arm B)* | −0.0243 |
| **HardSwish** (approx) | 0.4809 | **0.4714** *(arm A)* | −0.0095 |
| **activation costs** | −0.0865 | −0.0717 | **total −0.0960** |

Both routes from 0.5674 to 0.4714 sum to the same −0.0960, which is the arithmetic check that
the four runs are mutually consistent.

The two factors interact, so the split depends on the order you attribute in:

| Cause | if attributed first | if attributed second | average | share |
| --- | --- | --- | --- | --- |
| **SiLU → HardSwish** | −0.0865 | −0.0717 | **−0.0791** | **82 %** |
| **INT8 quantization** | −0.0243 | −0.0095 | −0.0169 | 18 % |

**The activation approximation costs roughly five times what quantization costs** — between 3×
and 9× depending on attribution order, and dominant under every one of them.

> [!important] The DFL head is not the problem — the activation is
> [[YOLOv8 Prior Work in Versal_AI]] suggested the gap might be YOLOv8's "quantization-sensitive
> DFL head". It is not. With the activation held exact, INT8 costs 0.024 mAP — 4.3 % relative,
> entirely ordinary for INT8 on a detector. The activation swap costs several times that. Effort to recover
> accuracy belongs in a **HardSwish finetune**, not in quantization settings, calibration size,
> or per-channel schemes.
>
> This mirrors YOLOv3, where the activation swap cost 88 % against quantization's 8.5 %
> ([[Board mAP - LeakyReLU Without Finetune]]). Same lesson, a second model family: on this DPU
> the activation substitution dominates, and it is the only thing worth retraining for.

> [!warning] Quote the 2×2, not a single subtraction
> A single pair of runs can support almost any split: float-SiLU vs float-HardSwish at *plain
> resize* gives −0.1111 activation against −0.0108 quantization (91 %/9 %), which overstates the
> activation; arm B vs arm A alone gives 75 %/25 %. Neither is wrong, but neither is the whole
> picture, because the factors interact. The 2×2 above bounds the answer from both directions and
> is the defensible form. The qualitative conclusion is stable across every decomposition:
> **the activation dominates.**

> [!note] Arm B cannot be deployed on YOLOv8, only simulated
> `vai_c_xir` refuses the decomposed graph: `[XCOM_OPFACTORY_OP_UNSUPPORTED] Op_type 18 is
> invalid for xcompiler`. This is v8-specific — the YOLOv3 decomposition compiles to 50 DPU
> subgraphs with `sigmoid` on the CPU ([[SiLU Decomposition]]) — and most likely the elementwise
> multiply inside `C2f`. It does not block the measurement, because the INT8 simulation above
> reproduces the board to 0.004 mAP, but it does mean arm B will never run on this board. The
> plan's expectation of an on-board arm B at ~1 FPS should be struck.

---

Back to [[Implementation Plan - YOLOv8]] | [[YOLOv8 Prior Numbers Reconciled]] | [[Home]]
