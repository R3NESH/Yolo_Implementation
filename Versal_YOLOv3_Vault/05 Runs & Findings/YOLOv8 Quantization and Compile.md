---
tags: [findings, yolov8, quantization, compile, artefacts]
date: 2026-09-14
result: all four arm-A models compile to 1 DPU subgraph - better than the prior work's 2; arm B will not compile at all
---

# YOLOv8 Quantization and Compile

What each stage of the YOLOv8 pipeline produced, and the one thing that refused to build. The
YOLOv8 counterpart to [[Quantization and Compile Results]].

Scripts: `Yolo_v8_Versal_Implementation/{v8_run.sh, v8_dpu_wrapper.py, v8_quantize.py,
v8_compile.sh}`.

---

## The flow

```
yolov8*.pt  --v8_dpu_wrapper-->  backbone + neck + head convs, SiLU replaced
            --v8_quantize calib-->  quant_info.json  (200 images)
            --v8_quantize test -->  YOLOv8DPUWrapper_int.xmodel
            --v8_compile.sh   -->  <stem>_<act>_vck190.xmodel  + the inspect_xmodel gate
```

Each model and arm gets its own directory, `quantize_result/<stem>_<act>/` and
`compiled/<stem>_<act>/`, so arms never overwrite each other — the mistake that made
[[YOLOv8 Prior Numbers Reconciled]] necessary in the first place.

## Calibration

| | |
| --- | --- |
| Source | `/datasets/coco_split/images/train` — the prior work's 4000-image split of val2017 |
| Samples | **200**, taken at even stride through the sorted list, not the first 200 |
| Preprocessing | plain resize, 640, RGB, 0–1 *(see the note below)* |
| Disjointness | that split is disjoint from `coco_split/images/val`, so a clean read is available |

The prior work calibrated on the **whole** of val2017, putting the entire eval set in the
calibration pool. Using the disjoint half costs nothing and leaves the option of an
uncontaminated number.

> [!warning] These builds were calibrated with a plain resize, not letterbox
> `v8_quantize.py` now letterboxes by default, but every artefact recorded here predates that
> change. This is **fine, and does not require rebuilding**:
> [[YOLOv8 Letterbox and the Activation Cost]] shows that a plain-resize-*calibrated* model,
> merely *inferred* with letterbox, reproduces the board to 0.004 mAP. Calibration preprocessing
> barely moves the result; inference preprocessing moves it a lot. Recompiling for letterbox
> calibration is optional polish, not a correctness fix.

## Artefacts

| Model | arm | compiled xmodel | DPU subgraphs | CPU ops |
| --- | --- | --- | --- | --- |
| `yolov8n` | HardSwish | **3.4 MB** | **1** | `fix2float` ×3 |
| `yolov8s` | HardSwish | **11.1 MB** | **1** | `fix2float` ×3 |
| `yolov8m` | HardSwish | **25.4 MB** | **1** | `fix2float` ×3 |
| `yolov8l` | HardSwish | **42.7 MB** | **1** | `fix2float` ×3 |
| `yolov8n` | decompose | ❌ **will not compile** | — | — |

**All four arm-A models compile to a single DPU subgraph** with `fix2float` as the only CPU op —
so the result is a property of the wrapper design, not an artefact of the smallest model. Total
82.6 MB for the set, against the board's ~912 MB free ([[Disk and System Constraints]]).

Activations replaced per model: **57 / 57 / 77 / 97** for n / s / m / l — the `Conv` count, not
the `nn.SiLU` count, for the reason recorded in [[Implementation Plan - YOLOv8]].

## One DPU subgraph, beating the prior art

```
subgraphs: {'USER': 1, 'DPU': 1, 'CPU': 3}
CPU ops (each needs libvart_op_imp_<type>.so on the board):
  fix2float   3
DPU[0] in : [('YOLOv8DPUWrapper__input_0_fix', (1, 640, 640, 3))]
DPU[0] out: [(..., (1, 20, 20, 144)), (..., (1, 80, 80, 144)), (..., (1, 40, 40, 144))]
OK single DPU subgraph - vart.Runner is the fast path.
```

The prior work's compiled models report **2** DPU subgraphs with a softmax, a sigmoid and assorted
slices/reshapes on the CPU, because their wrapper left part of the DFL decode inside the graph.
This wrapper stops at the head convolutions and hands *everything* below them to the host, so
nothing is left to fragment on. The host pays for it in decode time, which
[[YOLOv8 Anchor-Free Decode]] measures at 4.8 ms/image — far below the DPU's own budget.

> [!tip] Only `fix2float` on the CPU, and the board ships it
> This is the check that matters, not the subgraph count. [[SiLU Decomposition]] records a build
> that compiled cleanly and then died on the board for want of
> `libvart_op_imp_aten__silu_.so`. `fix2float` is exercised by the existing YOLOv3 GraphRunner
> path, so it is known present. With one DPU subgraph, `board_eval_vck190.py --runner auto`
> selects `vart.Runner` and dequantizes the int8 outputs itself via `fix_point`.

> [!note] Output order is not sorted by scale
> The compiler emits the tensors as 20×20, 80×80, 40×40 — *not* in stride order. Both decode
> modules derive the stride from each tensor's grid size rather than its position, so this is
> harmless. It would silently corrupt any decode that assumed positional ordering.

## Arm B will not compile

`vai_c_xir` aborts on the SiLU-decomposed graph:

```
[UNILOG][FATAL][XCOM_OPFACTORY_OP_UNSUPPORTED][The operator has not been implemented by target DPU.]
Op_type 18 is invalid for xcompiler
```

This is **YOLOv8-specific**. The identical technique on YOLOv3 compiles and runs — 50 DPU
subgraphs, `sigmoid`/`fix2float`/`float2fix` on the CPU, deployed and measured at 0.4829 mAP\@0.5
([[SiLU Decomposition]]). The likely culprit is the elementwise multiply landing inside `C2f`'s
split/concat structure, where the YOLOv3 graph let the DPU absorb it.

Consequence for the plan: **arm B is a simulation-only arm on YOLOv8.** That costs nothing, since
the INT8 simulation reproduces the board to 0.004 mAP and the arm existed to produce a number, not
a deployment. The plan's expectation of running arm B on-board at ~1 FPS should be struck.

---

Back to [[Implementation Plan - YOLOv8]] | [[YOLOv8 Letterbox and the Activation Cost]] | [[Home]]
