---
tags: [plan, yolov8, deployment, next-session]
date: 2026-09-14
status: complete — all four models deployed and scored on the VCK190
---

# Implementation Plan — YOLOv8

Deploying **`yolov8n/s/m/l`** to the VCK190, the same task [[Implementation Plan]] covers for
YOLOv3. Checkpoints are in `Yolo_v8_Versal_Implementation/` at the repo root.

> [!important] The headline: this is not a green field
> YOLOv8 **already runs on this board**. `Versal_AI` on the Desktop has compiled xmodels for all
> four sizes, and full 5000-image board runs at 6–8 FPS. What it does not have is *trustworthy
> numbers* — see [[YOLOv8 Prior Work in Versal_AI]]. The work ahead is measurement discipline,
> not bring-up.

Related: [[Implementation Plan]], [[YOLOv8 Prior Work in Versal_AI]], [[SiLU Decomposition]],
[[DPU Subgraph Fragmentation]], [[Board mAP - best.pt on Hardware]], [[Host Application]].

---

## What the checkpoints are

Read out of the pickles directly (no torch needed — `zipfile` + `pickletools`):

| | value |
| --- | --- |
| Models | `yolov8n` 6.3 MB, `yolov8s` 22 MB, `yolov8m` 50 MB, `yolov8l` 84 MB |
| Provenance | stock Ultralytics release weights, dated **2022-12-30/31** |
| `nc` / task | 80 / `detect` |
| `imgsz` | **640** (not 416 — every v3 assumption about input size is wrong here) |
| Activation | **SiLU**, same as YOLOv3 |
| Module namespace | `ultralytics.nn.modules.*` (flat: `C2f`, `Conv`, `Concat`, `Detect`, `SPPF`) |

Sizes match `Versal_AI/benchmarks/cpu_benchmark.csv` (6.25 / 21.54 / 49.72 / 83.73 MB), so the
float baselines recorded there apply to exactly these weights.

## Why YOLOv8 is *easier* on this DPU than YOLOv3

Running `tools/inspect_xmodel.py` against the already-compiled `Versal_AI` models:

| | YOLOv3 (`compiled_orig`) | YOLOv8n/s/m/l |
| --- | --- | --- |
| DPU subgraphs | 73 | **2** |
| CPU ops | `sigmoid` ×72 | head only: 1 softmax, 1 sigmoid, slices/reshapes/concats |
| Input | 416 | 640 |
| Throughput | 0.83 FPS | **6–8 FPS** |

The whole backbone and neck compile into **one** DPU subgraph. The reason is
`Versal_AI/build/replace_activation.py`, which swaps every `nn.SiLU` for `nn.Hardswish` — an
activation the DPUCVDX8G implements natively. The fragmentation problem that dominated the entire
YOLOv3 effort simply does not arise.

DPU output shapes confirm the head: `(1, 80, 80, 144)`, `(1, 40, 40, 144)`, `(1, 20, 20, 144)`
for a 640 input — strides 8/16/32, and **144 = 64 + 80**, i.e. 4 sides × 16 DFL bins plus 80
class logits. Anchor-free. This is the one genuinely new piece of engineering.

---

## Status

| # | Step | Status |
| --- | --- | --- |
| 0 | Get a real `ultralytics` into the Vitis container | ✅ **done** — 8.4.5 vendored at `Yolo_v8_Versal_Implementation/vendor/`, all four load under py3.8.6 / torch 1.13.1 |
| 1 | Pre-flight: inspect all four checkpoints | ✅ **done** — `nc=80`, `reg_max=16`, `no=144`, stride 8/16/32, imgsz 640, COCO class order clean on all four |
| 2 | DPU wrapper: strip the Detect decode, expose 3 raw outputs | ✅ **done** — `v8_dpu_wrapper.py`; emits (1,144,80,80)/(1,144,40,40)/(1,144,20,20), matching prior art |
| 3 | Activation arm A — HardSwish (reproduce prior art) | ✅ **done — all four** |
| 3b | Activation arm B — exact SiLU decomposition | ⚠️ **measured, but will not compile** — `Op_type 18 is invalid for xcompiler`; the number comes from INT8 simulation instead — [[YOLOv8 Letterbox and the Activation Cost]] |
| 4 | Quantize — `calib` then `test` | ✅ **all four**, plus `yolov8n` arm B |
| 5 | Compile with `vai_c_xir` for VCK190 | ✅ **all four** — 3.4 / 11.1 / 25.4 / 42.7 MB |
| 6 | `inspect_xmodel.py` gate before any board time | ✅ **all four pass: 1 DPU subgraph, `fix2float` only** — better than prior art's 2 |
| 7 | **Host decode: anchor-free + DFL** — the real new work | ✅ **done and verified to 6e-5 px on all four** — [[YOLOv8 Anchor-Free Decode]] |
| 8 | Board evaluation, all 5000 images | ✅ **done — all four, 5000 images each** — [[YOLOv8 on Hardware]] |
| 9 | Reconcile against the `Versal_AI` numbers | ✅ **done first, as planned** — [[YOLOv8 Prior Numbers Reconciled]] |

---

## Step 0 — The environment blocker

`from ultralytics import YOLO` is required to load these checkpoints, and:

- the host has **no** `ultralytics` (`pip3 show ultralytics` → nothing);
- `yolov3_test/vitis_compat/ultralytics/` is a **deliberate stub** — its own docstring says it
  provides "only the symbols actually imported" by the YOLOv3 repo. It has no `nn.modules`, so it
  cannot reconstruct a `C2f` or a v8 `Detect`. Unpickling a YOLOv8 checkpoint against it will
  fail, and [[The numpy _core Segfault]] is the cautionary tale about what a *partial* shim does.

So a real `ultralytics` must go into the container. Two routes:

1. **Vendor the package** next to `vitis_compat/` and put it on `PYTHONPATH` ahead of the stub.
   Offline, reproducible, and matches how `seaborn` is already handled.
2. `pip install ultralytics` inside the image — simpler, but the container is rebuilt from
   `xilinx/vitis-ai-pytorch-cpu:latest` on every `vitis_run.sh` call, so it would not persist.

Route 1 is the one that fits this repo. Two version notes: the checkpoints predate the
`ultralytics.nn.modules.{conv,block,head}` split, but modern versions still re-export the flat
names from `ultralytics/nn/modules/__init__.py`, so a current release will *probably* unpickle
them — verify, do not assume. And ultralytics pulls in a large dependency tree; only
`ultralytics.nn` is needed for unpickling, so a trimmed vendor may be enough.

> [!tip] Definition of done for step 0
> `torch.load("yolov8n.pt")` succeeds inside `./vitis_run.sh` and prints the model summary.
> Nothing else can start until this passes.

## Step 1 — Pre-flight

Mirror `inspect_model_AB1.py`: confirm `nc`, `imgsz`, strides, and the 144-channel head split for
each of the four models before spending container time.

> [!warning] Run the class-sanity check on every checkpoint
> [[yolov3_original on Hardware]] cost a full day because a checkpoint silently carried a
> different class ordering — `nc: 80` with `person` removed and every index shifted by one. These
> are stock Ultralytics releases so they are almost certainly clean, but `tools/recover_class_mapping.py`
> answers it in minutes from a float run, and "almost certainly" is what made the last one expensive.

## Steps 2–6 — The pipeline

Structurally identical to the YOLOv3 flow, and `Versal_AI/build/` already has working versions of
most of it (`replace_activation.py`, `quantize_pt.py`, `export_xmodel.py`, `calibrator.py`,
`src/compile_models.py`). Reuse those rather than re-deriving; port them to `vitis_run.sh`
conventions so the flow is reproducible from this repo.

The **wrapper** (step 2) should expose the three raw `(H, W, 144)` tensors and nothing else —
same principle as `export_dpu_wrapper_AB3.py`: the DPU emits feature maps, the host decodes.

### Step 3 — The activation experiment, and why to run both arms

| Arm | Method | Maths | Expected graph | Expected speed |
| --- | --- | --- | --- | --- |
| **A** HardSwish | `nn.SiLU` → `nn.Hardswish` (prior art) | **approximate** | 2 DPU subgraphs | 6–8 FPS |
| **B** SiLU decomposition | `x · sigmoid(x)` ([[SiLU Decomposition]]) | **exact** | heavily fragmented | ~1 FPS or worse |

Arm A is the deployable one. Arm B is the *measurement*: it isolates INT8 quantization loss with
the activation held exactly, so subtracting the two finally prices the HardSwish approximation —
a number nobody currently has. This is the same separation that made the YOLOv3 result
interpretable (quantization 8.5 % vs activation swap 88 %).

Predict arm B's fragmentation before running it: [[DPU Subgraph Fragmentation]] establishes
**one DPU subgraph per activation, exactly** (49 SiLU → 50, 72 SiLU → 73).

> [!warning] Do not count `nn.SiLU` modules — the count is always 1
> Ultralytics declares `default_act = nn.SiLU()` as a **class attribute** on `Conv`
> (`nn/modules/conv.py:49`), so every convolution shares *one* instance. `model.modules()`
> de-duplicates by identity, so it reports **`{'SiLU': 1}`** for `yolov8n` through `yolov8l`
> alike. Counting it would predict 2 subgraphs for every model and be wrong every time.
>
> The activation is applied once per `Conv`, so the **`Conv` count** is the predictor:
>
> | | `yolov8n` | `yolov8s` | `yolov8m` | `yolov8l` |
> | --- | --- | --- | --- | --- |
> | `Conv` modules | 57 | 57 | 77 | 97 |
> | predicted arm-B DPU subgraphs | ~58 | ~58 | ~78 | ~98 |
>
> This does *not* affect `Versal_AI/build/replace_activation.py`, which is still correct: it
> walks `named_children()` and rebinds with `setattr` per parent, so each `Conv` receives its
> own fresh `Hardswish` regardless of the shared source instance.

At ~98 subgraphs `yolov8l` will be slow enough that a full 5000-image run is impractical; use a
fixed 500-image subset for arm B, applied identically to arm A so the comparison stays valid.

### Step 6 — The gate

`./vitis_run.sh python ../tools/inspect_xmodel.py <xmodel>` before **any** board time. It reports
subgraph counts and the CPU op types the runtime must supply. This is the check that would have
caught the `aten::silu_` problem before the board was ever booted.

## Step 7 — Host decode: the real new work

`host_vck190/yolo_decode.py` is **anchor-based** and cannot be reused. YOLOv8 is anchor-free with
a Distribution Focal Loss head. Per scale, for each of the 144 channels at each cell:

```
channels  0..63   ->  4 sides x 16 bins, the box distribution
channels 64..143  ->  80 class logits

softmax over each side's 16 bins, then expectation  ->  ltrb distances in cells
x1 = (col + 0.5 - l) * stride      y1 = (row + 0.5 - t) * stride
x2 = (col + 0.5 + r) * stride      y2 = (row + 0.5 + b) * stride
scores = sigmoid(class logits)     # no separate objectness - v8 has none
```

No anchors, no `exp()`, no objectness multiply — every one of which the v3 decode does.

> [!danger] Verify this off-board before running it on hardware
> [[Board Bring-Up]] records three host-application bugs that each crashed on frame one, in code
> that had been "verified" only in principle. `host_vck190/verify_decode.py` is the precedent:
> feed one image through float PyTorch and through the decode, and require the boxes to match.
> Board time is the expensive resource; a decode bug found on the board costs a run.

Preprocessing must match calibration: the v3 host uses **plain resize, not letterbox**, and at
**640** here. A mismatch between calibration and inference preprocessing is an invisible accuracy
tax ([[Board mAP - best.pt on Hardware]] discusses it).

## Step 8 — Board evaluation

`board_eval_vck190.py` carries over: `--runner auto` already counts DPU subgraphs and picks
`GraphRunner`, `--class-offset` exists, COCO scoring and CSV output are in place. Swap the decode
module and the input size; everything else is the same harness.

Cost is modest — at ~7 FPS, 5000 images is **~12 minutes per model**, against 1 h 40 m for
YOLOv3. All four models in arm A is roughly an hour of board time.

Dataset is already staged: `val2017` and `instances_val2017.json` are on the board at
`/home/root/yolo/`, and on the external HDD at `/media/aesicdab/One Touch/coco_val2017/` with YOLO
labels already converted ([[Disk and System Constraints]]).

## Step 9 — Reconcile, and do not inherit the old numbers

[[YOLOv8 Prior Work in Versal_AI]] has five generations of board results that disagree, one of
which scores *above* float. Treat all of them as unverified. The reconciliation is cheap because
every predictions JSON was saved — rescoring them offline needs no board time at all, and should
be done **before** re-running anything, because it may show that one generation was right all
along.

Apply the same three-way discipline that made the v3 numbers defensible:

1. a like-for-like float baseline, same scorer, same class space;
2. two independent scorers (`pycocotools` and `tools/board_curves.py`) agreeing;
3. the class-agnostic diagnostic if anything looks wrong (`tools/recover_class_mapping.py`).

---

## Risks

| Risk | Mitigation |
| --- | --- |
| `ultralytics` cannot be made to work offline in the container | Vendor only `ultralytics.nn`; fall back to exporting to ONNX on the host — but the host has no ultralytics either, so step 0 is unavoidable |
| DFL decode subtly wrong — plausible boxes, bad mAP | Verify against float PyTorch on single images before board time |
| Arm B fragmentation makes `yolov8l` impractical | Use a fixed 500-image subset for both arms |
| Board disk: 912 MB free | 4 xmodels ≈ 85 MB total; predictions JSONs are the real cost — delete between runs |
| HardSwish accuracy cost gets attributed to quantization | That is precisely what arm B is for |

## Definition of done

Four models, each with: a compiled xmodel, a verified decode, a board mAP from two independent
scorers, a like-for-like float baseline, and a vault note recording all of it — the same standard
[[yolov3_original on Hardware]] meets. Plus one number nobody has today: **what HardSwish costs.**

---

Back to [[Implementation Plan]] | [[Home]]
