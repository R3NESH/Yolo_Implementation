---
tags: [changes, changelog, session]
date: 2026-09-11
scope: Yolo_v3_AB_New_Clone, plus one approved deletion outside it
---

# Changes Log

Every change made during the 2026-09-10/11 session, in full.

> [!tip] Companion notes
> - [[Bugs Found and Fixed]] — the debugging record: what was broken, how it was diagnosed
> - [[Code Improvements]] — how the codebase is better, beyond the bug fixes

Related: [[Clone Provenance]], [[Compat Shims]], [[Disk and System Constraints]],
[[The numpy _core Segfault]], [[Host Application]].

> [!important] Containment rule, and where it was relaxed
> This machine belongs to someone else. **Every code and documentation change below lives inside
> `~/Documents/Yolo_v3_AB_New_Clone/`.** No project file anywhere else was modified.
>
> Three things happened outside the clone, all listed under [[#Outside the clone]]:
> the pip cache was purged, containers ran with `--rm`, and — **with explicit approval** — the
> original folder's unused 8.2 GB `.venv` was deleted. Nothing else outside the clone was written
> to, and a third folder (`~/Documents/Yolo_v3_AB/`) was read but never modified.

---

## Files created

### The Vitis AI compatibility layer

`yolov3_test/vitis_compat/` — a shim directory that is only active when it is on `PYTHONPATH`,
so it can never shadow a real install outside the container.

| File | Purpose |
| --- | --- |
| `vitis_compat/np2_pickle_compat.py` | Aliases `numpy._core.*` → `numpy.core.*` so NumPy-2.x-pickled checkpoints load under the image's NumPy 1.22. See [[The numpy _core Segfault]] — the naive version of this crashes. |
| `vitis_compat/ultralytics/__init__.py` | Package root for the offline `ultralytics` stand-in. |
| `vitis_compat/ultralytics/utils/__init__.py` | Namespace holder. |
| `vitis_compat/ultralytics/utils/plotting.py` | `Annotator`, `Colors`/`colors`, `save_one_box` — real cv2-backed implementations, small. |
| `vitis_compat/ultralytics/utils/checks.py` | `check_requirements` → no-op (the image pins its own versions). |
| `vitis_compat/ultralytics/utils/patches.py` | `torch_load` with `weights_only=False` defaulted. |
| `vitis_compat/ultralytics/utils/torch_utils.py` | `TORCH_2_4` flag + `autocast` shim for the torch 1.13 API. |

Why this exists: `models/common.py:24` and `utils/general.py:36-37` import from the `ultralytics`
package, which the Vitis AI image does not ship. Those imports run at **module import time**,
which `torch.load` triggers while unpickling a checkpoint — so without them the checkpoint cannot
be opened at all. Full reasoning in [[Compat Shims]].

### Third-party package, installed clone-locally

- `yolov3_test/vitis_compat/site/` — **seaborn 0.13.2**, installed with
  `pip install --no-cache-dir --no-deps --target vitis_compat/site seaborn` (2.2 MB).
  `--no-deps` was deliberate: numpy/pandas/matplotlib are already in the image, and pulling
  duplicates would have wasted scarce disk. Needed because `utils/plots.py:15` imports seaborn at
  module level and sits on the checkpoint-loading import path.

### Tooling

| File | Purpose |
| --- | --- |
| `yolov3_test/vitis_run.sh` | One-line runner for the Vitis AI container. Mounts **only** this clone, drops to the host UID/GID so nothing lands root-owned, sets `PYTHONPATH` to the compat dirs, activates the `vitis-ai-pytorch` conda env. See [[Vitis AI Container]]. |
| `tools/graphify_codebase.py` | The "graphify" codebase grapher — stdlib `ast` only, nothing installed. Emits the 58 per-module notes under `02 Codebase/Modules/` and [[Import Dependency Graph]]. |
| `Versal_YOLOv3_Vault/` | This Obsidian vault, including `.obsidian/` config with a tuned Graph View. |
| `yolov3_test/vitis_out/` | Run logs (`inspect_best.log`, `quant_calib.log`, `quant_test.log`, `compile.log`, `lrelu_calib.log`, `lrelu_test.log`, `compile_lrelu.log`) plus `probe_init.py`, the throwaway probe that located the segfault. |

### The board-side host application

`host_vck190/` — new directory, see [[Host Application]].

| File | Purpose |
| --- | --- |
| `host_vck190/yolo_decode.py` | Correct YOLOv5-form decode, per-class NMS, box clipping, preprocessing. **Pure NumPy** — no torch, no VART — so it is testable on a desktop and runs unchanged on the board. |
| `host_vck190/board_eval_vck190.py` | VART runner + COCO scoring + benchmark mode. Derived from the July project's harness, with its broken decode replaced. |
| `host_vck190/verify_decode.py` | Regression check: decodes the same raw tensors in NumPy and in PyTorch (using the model's own grid/anchor buffers) and compares. Worst disagreement 9e-5. |

---

## Files modified

Only **three existing project files** were touched, all minimally. (`dpu_silu_experiment.py` below
is a new file, listed here because it belongs to the same piece of work.)

### `inspect_model_AB1.py`

1. **Added the NumPy compat shim** at the top of the live code — without it `torch.load` raises
   `ModuleNotFoundError: No module named 'numpy._core'`. (This file kept an inline copy at first;
   it now shares `np2_pickle_compat`.)
2. **Retargeted the `__main__` block.** It hardcoded `yolov3_merged23_e75.pt` and
   `yolov3_original.pt`; it now takes paths from `sys.argv` and defaults to
   `runs/train/exp2/weights/best.pt` — the checkpoint that actually gets deployed.

```python
if __name__ == "__main__":
    import sys
    paths = sys.argv[1:] or ['runs/train/exp2/weights/best.pt']
    for path in paths:
        inspect_checkpoint(path)
```

### `export_dpu_wrapper_AB3.py`

Replaced its inline NumPy shim with the shared `np2_pickle_compat.apply()`.

> [!danger] This was a real bug fix, not cosmetics
> The original shim mapped only the `numpy._core` **parent** module. That is worse than doing
> nothing: the unpickler then resolves `numpy._core.multiarray` through the parent's `__path__`,
> re-executes NumPy's C extension under a second module name, and **segfaults** inside
> `torch.load` (exit 139, no traceback). Registering the submodules up front prevents the
> re-import. Full diagnosis in [[The numpy _core Segfault]].

### `quantize_vitis_AB4.py`

Two changes:

1. Same shim replacement as above, same reason. Its `sys` import was retained (still used by
   `sys.exit`).
2. **Fixed a broken `export_xmodel` call** — a real bug that blocked xmodel export entirely:

   ```python
   # before - TypeError: export_xmodel() got an unexpected keyword argument 'deploy'
   quantizer.export_xmodel(output_dir=output_dir, deploy=True)

   # after
   quantizer.export_xmodel(output_dir=output_dir, deploy_check=False)
   ```

   This Vitis AI build's signature is
   `export_xmodel(self, output_dir, deploy_check=False, dynamic_batch=False)` — there is no
   `deploy` kwarg. `deploy_check=True` would additionally dump golden per-layer tensors for
   on-board comparison; left off to conserve disk ([[Disk and System Constraints]]).

   > [!note] Edit targeting caution
   > The string appears **three times** in this file — the live call plus two commented-out
   > earlier drafts. Only the live one (in the `elif quant_mode == 'test':` block) was changed.

### `dpu_silu_experiment.py` *(new file)*

The controlled experiment that proved the activation fix — swaps all 49 SiLU modules for
`LeakyReLU(0.1015625)` **in memory** (no second checkpoint written) and re-runs the quantize
pipeline into separate output directories. Full write-up: [[SiLU to LeakyReLU Experiment]].

---

## Files deleted

- `yolov3_test/.venv` — **was a symlink pointing into the original folder**
  (`~/Documents/Yolo_v3_AB_New/yolov3_test/.venv`, 8.2 GB). Any `pip install`, or even a stray
  `.pyc` write, would have modified the original through it. Removed on discovery.
  - It was briefly replaced with a real empty venv, then removed entirely once the decision was
    made to run everything inside the Vitis AI container instead. **There is no `.venv` in the
    clone now, and that is intentional** — see [[Vitis AI Container]].

---

## Generated artefacts (tool output, not hand-written)

Two complete quantize+compile cycles now exist on disk — the SiLU baseline and the LeakyReLU
control. Details in [[Quantization and Compile Results]].

**Baseline (SiLU, as trained)** — `yolov3_test/quantize_result/` and `yolov3_test/compiled/`:

| Artefact | Size | What it is |
| --- | --- | --- |
| `quantize_result/YOLOv3DPUWrapper.py` | 40 KB | The quantizer's own rewrite of the model, showing which ops it wrapped as quantized vs left float. Worth reading. |
| `quantize_result/quant_info.json` | 12 KB | Per-tensor quantization scaling parameters from calibration. |
| `quantize_result/bias_corr.pth` | 199 KB | Bias-correction data. |
| `quantize_result/YOLOv3DPUWrapper_int.xmodel` | 132 MB | The quantized INT8 XIR graph. |
| `compiled/yolov3_vck190.xmodel` | 35.4 MB | Compiled for VCK190 — **52 DPU subgraphs**. |
| `compiled/meta.json` | 6 KB | Runtime metadata; lists all 52 kernels. |

**LeakyReLU control** — `yolov3_test/quantize_result_lrelu/` and `yolov3_test/compiled_lrelu/`:

| Artefact | Size | What it is |
| --- | --- | --- |
| `quantize_result_lrelu/*` | ~250 KB | Calibration output for the swapped model. |
| `quantize_result_lrelu/YOLOv3DPUWrapper_int.xmodel` | 132 MB | Quantized INT8 graph. |
| `compiled_lrelu/yolov3_vck190_lrelu.xmodel` | 34.9 MB | Compiled — **1 DPU subgraph, 0 CPU ops**. |
| `compiled_lrelu/meta.json` | 265 B | Lists a single kernel. |

> [!warning] ~340 MB of build output, and neither model is shippable
> The baseline has usable weights but unusable performance; the control has usable performance but
> weights that were never finetuned for the new activation. If disk gets tight, the baseline pair
> is the one to delete — the control is the reference result.

### The vault itself

`Versal_YOLOv3_Vault/` — 83 notes: 58 auto-generated module notes plus 25 hand-written ones
covering the project, pipeline, findings, environment and changes. All 844 rendered wikilinks
verified to resolve, with no orphaned notes.

---

## Outside the clone

Three things, all disclosed and all approved:

1. **The original folder's 8.2 GB `.venv` was deleted** —
   `~/Documents/Yolo_v3_AB_New/yolov3_test/.venv`. Done **only after explicit approval**, with the
   machine owner's consent confirmed. It was unused by any of this work (the pipeline runs entirely
   in Docker) and is fully regenerable from `requirements.txt` — see [[Regenerating the venv]].
   Verified beforehand that the target was a real directory (not a symlink) containing nothing but
   a standard venv layout. **Freed 8.2 GB: disk went 9.3 GB → 18 GB free.**

2. **`~/.cache/pip` was purged.** An aborted dependency install (see below) had added entries to
   the shared pip cache; the purge cleared it and recovered **~5.5 GB**, taking the disk from
   100% full to 99%. Only downloaded wheels were removed — no project or user data. It repopulates
   itself on the next install.
3. **Docker** — containers are all launched with `--rm`, so none persist. The
   `xilinx/vitis-ai-pytorch-cpu:latest` image (11.7 GB) was **already present** and was not
   pulled, modified, or removed.

### Abandoned: the host virtualenv

An install of `yolov3_test/requirements.txt` into a fresh clone-local venv was started and then
**deliberately aborted**. The file pins torch 2.13 + TensorFlow 2.21 + the full NVIDIA CUDA stack
(190 packages, 10–15 GB) and the disk had ~5 GB free at the time. The partial venv was deleted.
Nothing from `requirements.txt` is installed anywhere. Rationale in
[[Disk and System Constraints]].

---

## Net effect on the original folder

For most of this session: **nothing**. At the end, **one approved deletion**.

| Item | State |
| --- | --- |
| All code, configs, weights, `runs/` | ✅ **Untouched** — verified: zero files modified |
| `yolov3_test/.venv` (8.2 GB) | ❌ **Deleted**, with explicit approval — [[Regenerating the venv]] |

`~/Documents/Yolo_v3_AB_New/` went from 8.9 GB to 766 MB, entirely from that one venv removal.
Everything that constitutes the *project* is intact and byte-identical. Verified with:

```bash
find ~/Documents/Yolo_v3_AB_New -newermt "2026-09-10 13:00" -not -path "*/.venv/*"
# (empty - nothing else changed)
```

### A third folder was read, never written

`~/Documents/Yolo_v3_AB/` (43 GB) holds an earlier attempt at this deployment plus the 27 GB COCO
dataset the training yamls point at. It was **read only** — see [[Prior Work in Yolo_v3_AB]].


## Session 2026-09-12 — deploying `yolov3_original.pt`

| File | Change |
| --- | --- |
| `host_vck190/board_eval_vck190.py` | **`--class-offset`** (default 0). Added to every predicted class index before the COCO category lookup. `yolov3_original.pt` needs `1`; every existing run is unaffected — [[yolov3_original on Hardware]]. |
| `yolov3_test/utils/metrics.py` | `np.trapezoid` → `getattr(np, "trapezoid", None) or np.trapz`. The NumPy 2.x name crashed every container-side `val.py` after a full 10-minute inference pass — [[Bugs Found and Fixed]]. |
| `.gitignore` | `quantize_result_orig/YOLOv3DPUWrapper_int.xmodel` (237 MB) and `runs/val/*/*_predictions.json` (55 MB each) — both regenerable. |

### Files created

| File | Purpose |
| --- | --- |
| `tools/recover_class_mapping.py` | Recovers the class ordering a checkpoint was *actually* trained with, by matching confident detections to ground truth ignoring class. The tool that caught the missing `person` class. |
| `tools/coco_to_yolo_labels.py` | `instances_val2017.json` → Ultralytics `.txt` labels. The repo ships only coco128, so float baselines need this. |
| `host_vck190/rescore_shifted.py` | Remaps a saved board predictions JSON by a class offset and re-runs pycocotools — corrects a finished run without re-inferring. |


## Session 2026-09-14 — YOLOv8 deployment

All new code lives in `Yolo_v8_Versal_Implementation/`, except the decode and harness changes,
which belong with the rest of the board-side application in `host_vck190/`.

### Files created

| File | Purpose |
| --- | --- |
| `Yolo_v8_Versal_Implementation/v8_run.sh` | Container runner for the v8 work. Puts the vendored `ultralytics` on `PYTHONPATH` **ahead of** the YOLOv3 stub, and bind-mounts the COCO data read-only at `/datasets` rather than copying it — the disk is 99 % full. |
| `Yolo_v8_Versal_Implementation/v8_dpu_wrapper.py` | Backbone + neck + head convolutions, emitting three raw `(1, 144, H, W)` tensors. Carries all three activation modes (`silu`, `hardswish`, `decompose`). |
| `Yolo_v8_Versal_Implementation/v8_quantize.py` | `vai_q_pytorch` calib/test. Letterbox calibration by default; `--no-letterbox` reverts. |
| `Yolo_v8_Versal_Implementation/v8_compile.sh` | `vai_c_xir` for the VCK190, then runs `tools/inspect_xmodel.py` immediately as a gate. |
| `Yolo_v8_Versal_Implementation/verify_v8_decode.py` | Proves the host decode matches ultralytics' own, on identical features — [[YOLOv8 Anchor-Free Decode]]. |
| `Yolo_v8_Versal_Implementation/v8_eval_quantized.py` | Off-board INT8-simulated evaluation. Reproduces the board to 0.004 mAP, which is what made progress possible while the board was down. |
| `Yolo_v8_Versal_Implementation/deploy_to_board.sh` | Copies the host application and compiled xmodels to the VCK190, with the `scp -O` and U-Boot caveats baked into its error message. |
| `Yolo_v8_Versal_Implementation/vendor/ultralytics/` | Real ultralytics **8.4.5**, 4.2 MB, copied from `Versal_AI/models/yolov8_test/`. Clears the step-0 blocker — [[Compat Shims]]. |
| `host_vck190/yolov8_decode.py` | Anchor-free DFL decode, letterbox, and box un-mapping. Pure NumPy, shares NMS and clipping with the v3 module. |
| `tools/rescore_v8_prior.py` | Scores any predictions JSON over both the full set and its own covered subset, with coverage and score-floor diagnostics. The tool that settled [[YOLOv8 Prior Numbers Reconciled]]. |

### Files modified

| File | Change |
| --- | --- |
| `host_vck190/board_eval_vck190.py` | **`--arch {v3,v8}`** (default `v3`, so every existing invocation is unaffected). Late-binds the decode module; `output_to_nhwc` now reports the expected channel count on mismatch instead of just the shape. |

### Generated artefacts

| Path | Contents |
| --- | --- |
| `Yolo_v8_Versal_Implementation/quantize_result/<stem>_<act>/` | `quant_info.json`, `bias_corr.pth`, the traced module, and the INT8 xmodel |
| `Yolo_v8_Versal_Implementation/compiled/<stem>_<act>/` | Compiled xmodel, `meta.json`, `md5sum.txt` — `yolov8n` 3.5 MB, `yolov8s` 11.1 MB |
| `Yolo_v8_Versal_Implementation/sim_results/` | Predictions JSONs from the INT8 simulation, one per configuration |

> [!note] Per-arm directories, deliberately
> Every artefact path carries both the model stem and the activation arm. The prior work wrote
> arms and generations over each other, which is the direct cause of the five contradictory
> result sets in [[YOLOv8 Prior Numbers Reconciled]].

### Not yet in `.gitignore`

`quantize_result/*/YOLOv8DPUWrapper_int.xmodel` (12 MB for `yolov8n`, more for the larger models)
and `sim_results/*.json` are regenerable and should be ignored before any commit, matching how the
YOLOv3 artefacts are handled.

---

Back to [[Code Map]] | [[Home]]
