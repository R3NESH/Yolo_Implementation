---
tags: [changes, improvements, quality]
date: 2026-09-11
---

# Code Improvements

How the codebase is better than it was — beyond the straight bug fixes in
[[Bugs Found and Fixed]]. Grouped by the kind of improvement, with the reasoning.

Related: [[Changes Log]], [[Host Application]], [[Vitis AI Container]].

---

## 1. The pipeline went from broken to working

The headline improvement. Before: `export_dpu_wrapper_AB3.py` segfaulted, and
`quantize_vitis_AB4.py` could not export an xmodel. Neither had ever produced a compiled model in
this project folder.

After: the full path runs end to end and produces a board-loadable artefact.

| Stage | Before | After |
| --- | --- | --- |
| Load checkpoint | SIGSEGV | ✅ works |
| Quantize (calib) | never reached | ✅ `quant_info.json` + `bias_corr.pth` |
| Export xmodel | `TypeError` | ✅ 132 MB `_int.xmodel` |
| Compile for VCK190 | never attempted | ✅ 35 MB xmodel |
| Host decode | wrong maths (old project) | ✅ verified to 9e-5 |

## 2. Reproducibility: one command instead of a remembered incantation

**Before:** running anything meant recalling a long `docker run` with the right mounts, the right
conda activation, and the right `PYTHONPATH`. Nothing recorded what that was.

**After:** `vitis_run.sh`.

```bash
./vitis_run.sh python inspect_model_AB1.py
./vitis_run.sh vai_c_xir --help
./vitis_run.sh bash
```

Each flag in it earns its place, and several fix real problems:

| Improvement | Why |
| --- | --- |
| `-u $(id -u):$(id -g)` | Without it the container writes **root-owned files into the repo** that you then cannot delete without sudo. |
| `-v CLONE_ROOT:/workspace` — clone only | The container **cannot see** the original folder or anything else on the machine. Containment enforced mechanically, not by discipline. |
| `--rm` | No container layers accumulate on a 99%-full disk. |
| `[ -t 0 ] && TTY_FLAGS=(-it)` | Interactive when there's a terminal, pipeline-safe when there isn't — so the same script works by hand and in scripts. |
| `-e HOME=/tmp` | Caches go to the container's throwaway `/tmp`, not the host. |

## 3. Shared shim instead of three divergent copies

**Before:** the NumPy compatibility hack was **copy-pasted inline** into each script that needed
it — and the copies were already subtly different. Each was the broken parent-only version.

**After:** one `vitis_compat/np2_pickle_compat.py`, imported by everything.

Beyond deduplication, the shared version is defensively written:

- `setdefault` rather than assignment — never clobbers a module that genuinely exists.
- Guarded on `hasattr(numpy, "_core")` — a complete **no-op on NumPy 2.x**, so the same file is
  safe inside and outside the container.
- `getattr(..., None)` per submodule — tolerates NumPy versions lacking one of them.
- A docstring that explains *why the obvious one-line version segfaults*, so nobody "simplifies"
  it back.

## 4. Testability: the decode can be tested without hardware

The most structurally significant improvement. The old project's decode was **welded into an
1823-line board-only script** that needed `vart`, `xir`, a COCO dataset and a physical VCK190 to
execute at all. It could not be tested, which is precisely why a wrong formula survived in it.

**Now split into three layers:**

| Layer | Dependencies | Testable where |
| --- | --- | --- |
| `yolo_decode.py` | numpy only | **any desktop** |
| `verify_decode.py` | + torch | the container |
| `board_eval_vck190.py` | + vart, xir, pycocotools | the board |

The decode — the part with the actual maths risk — now has **zero** hardware dependencies. That
made it possible to prove it correct before ever touching a board:

```
RESULT: MATCH - numpy decode is correct
worst: xy=3.052e-05 wh=9.155e-05 conf=5.960e-08
```

## 5. A regression test where there was none

`verify_decode.py` is not a one-off script — it is a **check that can be re-run**. It decodes the
same raw tensors two ways (NumPy via NHWC, PyTorch via the model's own `grid`/`anchor_grid`
buffers) and compares.

Critically it validates the **maths and the axis/channel ordering together**. Layout confusion —
NHWC vs NCHW, anchor-major vs class-major — is the classic way a hand-written decode goes silently
wrong, and a pure-maths review would not catch it.

It also prints what the **old wrong formula** would have produced, keeping the regression visible
rather than lost in a commit message.

## 6. Defensive guards in the board application

Added while porting, each targeting a specific way this deployment can silently misbehave:

| Guard | Failure it prevents |
| --- | --- |
| Warns when the xmodel has **>1 DPU subgraph** | The [[DPU Subgraph Fragmentation]] trap — otherwise reads as "the board is just slow" |
| Accepts **NHWC or NCHW** for input and output, detected from tensor dims | Silent transposition garbage |
| **Output order independent** — anchors keyed off grid size | Breaks if the runtime returns tensors in a different order |
| Proper `fix_point` **dequantization** per output tensor | Forgetting it yields meaningless scores |
| `clip_boxes` to image bounds | Out-of-frame boxes polluting mAP |
| Prints the **PC float baseline** beside measured numbers | Board results interpreted without a reference point |
| Warns on **0 predictions** and names the likely causes | Blaming the model for a preprocessing bug |

## 7. Scripts made parameterisable instead of hardcoded

**Before:** paths and modes were literals scattered through the files — `MODEL_PATH = "yolo_vck190.xmodel"`,
`VAL_DIR = "val2017"`, hardcoded checkpoint names in `inspect_model_AB1.py`.

**After:** `argparse` with sensible defaults. `board_eval_vck190.py` gained genuinely useful modes:

```bash
--limit 50 --no-eval      # fast smoke test: does it detect anything at all?
--benchmark 200           # throughput only, no scoring
--conf-thres / --iou-thres / --max-det
```

`--limit` and `--no-eval` matter practically: they turn "copy 5 GB of COCO to the board and wait"
into a 30-second check that the plumbing works.

## 8. A controlled experiment harness

`dpu_silu_experiment.py` exists to answer **one** question — does the activation actually cause the
fragmentation? — while changing nothing else.

Design choices worth keeping:

- Swaps activations **in memory**; writes no second 264 MB checkpoint, because the disk has no room.
- Writes into **separate** output directories, so the baseline and the control coexist for
  comparison rather than one overwriting the other.
- Reuses `get_calibration_dataloader` from the existing script, so calibration is provably identical
  between the two runs and the only variable is the activation.
- Its docstring states plainly that the result is **not shippable** — it measures mapping, not
  accuracy — so nobody deploys it by mistake.

## 9. Every run leaves a log

**Before:** no record of what was run or what it printed.

**After:** `vitis_out/` holds `inspect_best.log`, `quant_calib.log`, `quant_test.log`,
`compile.log`, `lrelu_calib.log`, `lrelu_test.log`, `compile_lrelu.log`.

This is what made the diagnosis possible at all. The SiLU finding came from comparing warning
counts across logs; the 52-vs-1 subgraph result is a one-line `grep` away:

```bash
grep "subgraph number" vitis_out/compile*.log
```

## 10. The codebase is now navigable

`tools/graphify_codebase.py` parses all 58 first-party modules with the stdlib `ast` module and
emits one Obsidian note per module, carrying imports, importers, classes, top-level functions and a
local dependency diagram.

Deliberately stdlib-only: no install, no network, no supply-chain risk — and re-runnable after any
code change. It is what turns 58 files and several 1500-line modules into something you can follow.

## 11. Comments that explain *why*

Where a line is non-obvious, the reason is now recorded next to it — the kind of comment that
prevents a future "cleanup" from reintroducing a bug:

- Why registering only `numpy._core` segfaults.
- That `export_xmodel` has no `deploy` kwarg, and what `deploy_check=True` would cost.
- That the DPU implements LeakyReLU at a specific slope, and that the quantizer rounds to it.
- That `yolo_decode` follows **calibration** preprocessing (plain resize), not **training**
  preprocessing (letterbox) — and that changing one without the other is wrong.
- That the live code in `train.py` starts around line 869, the first 59% being a dead copy.

---

## What was deliberately *not* changed

Restraint matters in someone else's codebase:

- **The dead commented-out blocks** in `train.py`, `export_dpu_wrapper_AB3.py` and
  `quantize_vitis_AB4.py` were left in place. They are a real readability hazard and are
  *documented* as such ([[Subsystem - Training]]), but deleting ~800 lines of someone else's
  history is their call, not ours.
- **`models/common.py` was not touched.** The SiLU→LeakyReLU change belongs in a training run, not
  as a silent edit to a shared model definition.
- **The original and old project folders** were never written to — [[Clone Provenance]].
- **`export.py`, the loggers, segmentation, AWS and Flask code** were left alone. They are
  irrelevant to this deployment and are simply marked as such in
  [[Subsystem - Utils Peripheral]].

---

Back to [[Code Map]] | [[Home]]
