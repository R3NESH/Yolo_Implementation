---
tags: [index, home]
date: 2026-09-11
---

# Home

Working notes for deploying **YOLOv3 onto an AMD Versal VCK190** via the Vitis AI DPU toolchain.

> [!tip] New here? Read these four, in order
> 1. [[Project Overview]] — what this repo is and what the task is
> 2. [[DPU Subgraph Fragmentation]] — **the blocking problem**, read this before planning any work
> 3. [[Implementation Plan]] — the nine steps, and where they currently stand
> 4. [[Code Map]] — the way into the codebase itself

---

## Current status at a glance

| Step | State |
| --- | --- |
| Inspect checkpoint | ✅ Done — [[Checkpoint Inspection]] |
| Verify DPU wrapper outputs | ✅ Done — 3 raw tensors, shapes confirmed |
| Quantize (calib) | ✅ Done |
| Export `.xmodel` | ✅ Done — 132 MB |
| Compile for VCK190 | ✅ Done — 35 MB xmodel |
| Single-subgraph check | ❌ **52 DPU subgraphs** — [[DPU Subgraph Fragmentation]] |
| *(added)* Prove the LeakyReLU fix | ✅ **1 DPU subgraph, 0 CPU ops** — [[SiLU to LeakyReLU Experiment]] |
| *(added)* Run it on real hardware | ✅ **50 imgs, 464 dets, 26.5 FPS** — [[Board Bring-Up]] |
| Finetune with LeakyReLU | ⬜ **Next action** — needs a GPU machine |
| Board bring-up | ✅ **Done** — VCK190 boots, DPU arch matches — [[Board Bring-Up]] |
| *(added)* Run **best.pt** on the board | ✅ **Done** — SiLU decomposed — [[SiLU Decomposition]] |
| VART host app | ✅ **Runs on hardware** — 3 bugs fixed — [[Board Bring-Up]] |
| Validate + benchmark | ✅ **best.pt on board: mAP@0.5 = 0.4829 (91.5% of float)** — [[Board mAP - best.pt on Hardware]] |
| *(added)* Deploy **yolov3_original.pt** | ✅ **Done — 0.4681 mAP@0.5 on board, but it has no `person` class** — [[yolov3_original on Hardware]] |

---

## Next up: YOLOv8

> [!tip] Starting the YOLOv8 work? Read these four first
> 1. [[Implementation Plan - YOLOv8]] — the plan and where each step stands
> 2. [[YOLOv8 on Hardware]] — **the results**: all four models, measured on the board
> 3. [[YOLOv8 Letterbox and the Activation Cost]] — what HardSwish costs, and why the host must
>    letterbox
> 4. [[YOLOv8 Prior Numbers Reconciled]] — why the five old generations disagreed, and which one
>    to trust

| Item | State |
| --- | --- |
| Checkpoints | ✅ `Yolo_v8_Versal_Implementation/` — stock `yolov8n/s/m/l`, nc 80, SiLU, **640** |
| Prior deployment | ✅ Already runs on this board — **2 DPU subgraphs, 6–8 FPS** (`Versal_AI` on the Desktop) |
| Prior accuracy numbers | ✅ **Reconciled — all five explained, only `_v3` is valid** — [[YOLOv8 Prior Numbers Reconciled]] |
| `ultralytics` in the container | ✅ **Unblocked** — real 8.4.5 vendored, loads all four under py3.8 — [[Compat Shims]] |
| Anchor-free DFL decode | ✅ **Written and verified to 6e-5 px on all four** — [[YOLOv8 Anchor-Free Decode]] |
| Compiled for VCK190 | ✅ **all four** — 3.4 / 11.1 / 25.4 / 42.7 MB, each **1 DPU subgraph** (prior art: 2) |
| What HardSwish costs | ✅ **~5× what quantization costs** — [[YOLOv8 Letterbox and the Activation Cost]] |
| Board run | ✅ **All four on hardware** — 0.4453 / 0.5423 / 0.6001 / 0.6233 mAP@0.5, every one beating prior art — [[YOLOv8 on Hardware]] |
| *(added)* LeakyReLU yolov8s | ✅ **On hardware, no activation swap** — 0.5104 mAP@0.5, **97.4% of its float score**, 1 DPU subgraph. Not stock v8s (lighter head). PDF report **not produced** — [[YOLOv8s LeakyReLU on Hardware]] |

> [!note] The surprise, and its answer
> YOLOv8 is *easier* on this DPU than YOLOv3. Swapping SiLU → HardSwish (which the DPU implements
> natively) collapses the graph against YOLOv3's 73, and it runs ~9× faster at a larger input.
> Stripping the whole decode to the host does better still: **1 DPU subgraph**, where the prior
> work got 2, with `fix2float` as the only CPU op.
>
> The open question — what that approximation costs — is now **answered**:
> [[YOLOv8 Letterbox and the Activation Cost]]. HardSwish costs roughly **five times** what INT8
> quantization does. The DFL head, long suspected, is not the problem. Accuracy recovery means a
> **HardSwish finetune**, and nothing else is worth tuning first.
>
> One correction to plan assumptions: arm B (exact `x · sigmoid(x)`) **cannot be compiled** for
> YOLOv8 — `vai_c_xir` rejects it with `Op_type 18 is invalid for xcompiler`, unlike YOLOv3 where
> it compiles to 50 subgraphs. Its number comes from INT8 simulation, which reproduces the board
> to 0.004 mAP.

> [!warning] The one thing to know
> The model uses **SiLU**, which XIR cannot map to the DPU — it shatters the graph into **52
> islands**. Swapping to **LeakyReLU(0.1015625)** collapses it to **1 subgraph with zero CPU
> fallback**, which has been *measured*, not guessed. Everything else in the pipeline works.
>
> **Worse than fragmentation: `compiled/` cannot execute at all.** There is no
> `libvart_op_imp_aten__silu_.so` on the board, so the SiLU build aborts on its first CPU
> subgraph. Compiling successfully proved nothing about running. Three paths:
>
> | Path | Accurate | Fast | GPU needed | State |
> | --- | --- | --- | --- | --- |
> | `compiled/` as-is | — | — | — | ❌ **cannot run** — no silu op library |
> | Decompose SiLU → `x*sigmoid(x)` | ✅ **0.4829 mAP@0.5** | ❌ ~1 FPS | **No** | ✅ **deployed and measured** |
> | Finetune LeakyReLU | ✅ after training | ✅ 56 FPS | Yes | ⬜ big job — 11% retained without it |
>
> **Next action: read [[Prior Work in Yolo_v3_AB]] first.** An earlier attempt (July 2026, in a
> third folder) already reached one DPU subgraph the same way, and left behind board-side eval and
> debug scripts plus 7 training runs — one of which scores *better* than the checkpoint currently
> being deployed. Reading it is free; retraining is not. **Then** finetune per
> [[Subsystem - Training]].

---

## The map

### Project & plan
- [[Project Overview]] — the codebase, the hardware target, the shape of the work
- [[Implementation Plan]] — nine steps, with live status
- [[Clone Provenance]] — where this working copy came from, and the containment rules

### Deployment
- [[Vitis AI DPU Concepts]] — the DPU, the toolchain, what the artefacts are
- [[Vitis AI Container]] — how to actually run anything (`./vitis_run.sh`)
- [[Host Application]] — the board-side code, in `host_vck190/`

### Findings
- [[YOLOv8 on Hardware]] — ⭐⭐ **the headline: all four YOLOv8 models on the VCK190**
- [[YOLOv8s LeakyReLU on Hardware]] — ⭐ a LeakyReLU-trained yolov8s on the board; keeps 97.4% of float, but below HardSwish s in absolute terms
- [[Files Not in Git]] — ⭐ **nine files over 100 MB are not in the repo; how to recreate them**
- [[YOLOv8 Prior Numbers Reconciled]] — ⭐ **the five contradictory v8 generations, settled offline**
- [[YOLOv8 Letterbox and the Activation Cost]] — ⭐⭐ **the 2×2: HardSwish costs ~5× quantization; INT8 sim matches the board to 0.004**
- [[YOLOv8 Anchor-Free Decode]] — ⭐ the DFL decode, verified against ultralytics before any board time
- [[YOLOv8 Quantization and Compile]] — the v8 artefacts: **1 DPU subgraph**, and the arm that would not build
- [[Board Bring-Up]] — ⭐ the board works; three VART bugs found and fixed
- [[Board mAP - best.pt on Hardware]] — ⭐⭐ **the headline number: 0.4829 mAP@0.5 on hardware**
- [[SiLU Decomposition]] — ⭐ how best.pt was made to run on the board without a GPU
- [[Board mAP - LeakyReLU Without Finetune]] — ⭐ the swap costs **89% of mAP**; measured on hardware
- [[DPU Subgraph Fragmentation]] — ⚠️ the blocker
- [[SiLU to LeakyReLU Experiment]] — measuring whether the proposed fix works
- [[Prior Work in Yolo_v3_AB]] — ⭐ **an earlier attempt got further; read before writing code**
- [[Quantization and Compile Results]] — what each stage produced
- [[Checkpoint Inspection]] — nc, anchors, strides, output shapes
- [[Training Run exp2]] — what was actually trained, and how well
- [[The numpy _core Segfault]] — a genuine bug found and fixed in the project's scripts

### Codebase
- [[Code Map]] — start here for code
- [[Subsystem - Models]] · [[Subsystem - Training]] · [[Subsystem - Validation]] · [[Subsystem - Export]]
- [[Subsystem - Utils Core]] · [[Subsystem - Utils Peripheral]] · [[Subsystem - Data and Datasets]]
- `02 Codebase/Modules/` — one note per Python module (58 of them), auto-generated with
  imports and importers as links

### Graphs
- [[Import Dependency Graph]] — the codebase as a graph
- [[Model Architecture Graph]] — the network, layer by layer

### Environment
- [[Disk and System Constraints]] — ⚠️ this machine is 99% full; read before installing anything
- [[Compat Shims]] — why `vitis_compat/` exists

### Changes
- [[Changes Log]] — every file created, modified, or deleted
- [[Bugs Found and Fixed]] — the debugging record: 8 problems, how each was diagnosed
- [[Code Improvements]] — how the codebase is better than it was

---

## Conventions in this vault

- **Graph View** is configured — module notes are colour-grouped separately from findings.
  Filter with `path:"02 Codebase/Modules"` to see the pure code graph.
- Module notes are **generated**, not hand-written: re-run
  `python3 tools/graphify_codebase.py` after code changes to refresh them.
- Anything marked *(unverified)* was not confirmed against the code or hardware. Treat it as a
  lead, not a fact.

---

Back to [[Code Map]] | [[Home]]
