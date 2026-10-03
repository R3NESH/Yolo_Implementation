---
tags: [changes, bugs, debugging, record]
date: 2026-09-11
---

# Bugs Found and Fixed

The debugging record. Fourteen distinct problems across the YOLOv3 and YOLOv8 efforts: eleven were
fixed, two were diagnosed without a code fix being possible, one was found in prior work and
worked around by rewriting.

Related: [[Changes Log]], [[Code Improvements]], [[The numpy _core Segfault]],
[[DPU Subgraph Fragmentation]], [[Prior Work in Yolo_v3_AB]].

> [!summary] Why the pipeline was stuck
> Before this session, `export_dpu_wrapper_AB3.py` **segfaulted** and
> `quantize_vitis_AB4.py` **could not export an xmodel**. Both were hard blockers, and neither
> produced a usable error message. Four separate environment problems sat between the checkpoint
> and the quantizer. All are now cleared, and the pipeline runs end to end.

---

## 1. The clone was not actually isolated

| | |
| --- | --- |
| **Severity** | High — risk of corrupting the original project |
| **Symptom** | Original folder measured 8.9 GB, the clone 766 MB, yet `diff -rq` reported the trees identical |
| **Root cause** | `yolov3_test/.venv` in the clone was a **symlink into the original folder** (`Yolo_v3_AB_New/yolov3_test/.venv`, 8.2 GB). It had never been copied. |
| **Why it mattered** | Any `pip install` from the clone — or Python merely writing a `.pyc` — would have written **through the symlink into the original**, defeating the entire point of working in a clone. |
| **Fix** | Symlink removed. Briefly replaced with a real venv, then removed entirely once the Docker-only approach was settled ([[Vitis AI Container]]). |
| **How it was found** | Chasing the size discrepancy rather than trusting `diff -rq`, which compares content and says nothing about how a path is reached. |

## 2. `numpy._core` ModuleNotFoundError

| | |
| --- | --- |
| **Severity** | Blocker |
| **Symptom** | `torch.load` → `ModuleNotFoundError: No module named 'numpy._core'` |
| **Root cause** | Checkpoint pickled under NumPy 2.x (which renamed `numpy.core` → `numpy._core`); the container ships NumPy 1.22. |
| **Fix** | `vitis_compat/np2_pickle_compat.py` |

## 3. …and the shim that made it a segfault

| | |
| --- | --- |
| **Severity** | Blocker, and the hardest to diagnose |
| **Symptom** | **Exit code 139 (SIGSEGV)** inside `torch.load`. No traceback, no message. |
| **Root cause** | The project's existing shim registered only the **parent** module (`sys.modules['numpy._core'] = numpy.core`). The unpickler then resolved `numpy._core.multiarray` through that parent's `__path__`, **re-importing NumPy's C extension under a second name**. A twice-initialised C extension has two copies of its static state; NumPy's multiarray does not survive it. |
| **Fix** | Register the parent **and** every submodule a pickled array/scalar/dtype can reference, so the re-import never happens. Uses `setdefault` (never clobber a real module) and no-ops entirely on NumPy 2.x. |
| **How it was found** | `vitis_out/probe_init.py` — a bisection probe printing a marker after each step of the wrapper's `__init__` with `flush=True`, so output survived the crash. It stopped after `a:` and never reached `b:`, isolating the fault to `torch.load` itself rather than the model surgery after it. |

> [!danger] The lesson
> The partial shim was **worse than no shim**. With no shim you get a self-diagnosing
> `ModuleNotFoundError` naming the exact problem. With the partial shim you get a silent
> segmentation fault. A compatibility shim that half-works converts an easy bug into a hard one.

Full analysis: [[The numpy _core Segfault]].

## 4. Missing `ultralytics` package blocked checkpoint loading

| | |
| --- | --- |
| **Severity** | Blocker |
| **Symptom** | `ModuleNotFoundError: No module named 'ultralytics'` — while merely *inspecting* a checkpoint |
| **Root cause** | `torch.load` unpickles **model objects**, which imports the modules defining their classes. `models/common.py:24` and `utils/general.py:36-37` import from the `ultralytics` package, which the Vitis AI image does not ship. Those imports run at module-import time, so the checkpoint cannot be opened at all. |
| **Fix** | A minimal but *functional* offline shim at `vitis_compat/ultralytics/` — real cv2-backed `Annotator`, the actual 20-colour palette, working `save_one_box`, a no-op `check_requirements`, a `torch_load` wrapper, and version-aware `TORCH_2_4`/`autocast`. Activated only via `PYTHONPATH`, so it can never shadow a real install. |

## 5. Missing `seaborn` blocked checkpoint loading

| | |
| --- | --- |
| **Severity** | Blocker |
| **Symptom** | `ModuleNotFoundError: No module named 'seaborn'` — again while inspecting a checkpoint, with nothing being plotted |
| **Root cause** | The same import chain. `models/yolo.py:27` imports `feature_visualization` from `utils/plots.py`, which imports seaborn at module level (`utils/plots.py:15`). |
| **Fix** | `pip install --no-cache-dir --no-deps --target vitis_compat/site seaborn` — 2.2 MB. `--no-deps` deliberate: numpy/pandas/matplotlib are already in the image, and duplicates would have wasted scarce disk. |

> [!note] Genuinely counter-intuitive
> A **plotting library** prevented a **model from loading**. Worth internalising: in this codebase,
> `torch.load` drags in the entire `utils/` import graph. See the diagram in [[Compat Shims]].

## 6. `export_xmodel()` called with a non-existent kwarg

| | |
| --- | --- |
| **Severity** | Blocker — no xmodel could be produced |
| **Symptom** | `TypeError: export_xmodel() got an unexpected keyword argument 'deploy'` |
| **Root cause** | `quantize_vitis_AB4.py` called `export_xmodel(output_dir=..., deploy=True)`. This Vitis AI build's signature is `export_xmodel(self, output_dir, deploy_check=False, dynamic_batch=False)` — there is no `deploy` parameter. |
| **Fix** | Changed to `deploy_check=False`, with a comment recording the real signature and what `deploy_check=True` would add (golden per-layer tensor dumps for on-board comparison — omitted to conserve disk). |
| **Gotcha while fixing** | The offending line appears **three times** in the file — the live call plus two commented-out earlier drafts. Only the live one, inside `elif quant_mode == 'test':`, was changed. |

## 7. Inspection script pointed at the wrong checkpoints

| | |
| --- | --- |
| **Severity** | Low, but misleading |
| **Symptom** | `inspect_model_AB1.py` reported on models that are not the ones being deployed |
| **Root cause** | Its `__main__` block hardcoded `yolov3_merged23_e75.pt` and `yolov3_original.pt`. The actually-deployed checkpoint is `runs/train/exp2/weights/best.pt`. |
| **Fix** | Takes paths from `sys.argv`, defaulting to `runs/train/exp2/weights/best.pt`. |

## 8. The prior project's host-side decode was mathematically wrong

| | |
| --- | --- |
| **Severity** | High — silently wrong results |
| **Symptom** | Not a crash. Detections appear, but boxes are mispositioned and missized. Strongly suspected cause of the six debug/sweep scripts accumulated in the old project. |
| **Root cause** | `eval_yolo_vck190.py:1634-1642` implements **classic Darknet YOLOv3** decode (`cx = (σ(t)+cell)·stride`, `bw = exp(t)·anchor`). This model is YOLOv5-lineage and uses `cx = (2σ(t)−0.5+cell)·stride`, `bw = (2σ(t))²·anchor`. |
| **Measured impact** | Boxes **~1.4× too large** on average on the finest scale (mean w 46.4 px vs correct 32.8 px). |
| **Corroborating smell** | The old code carried `clip(pred, -20, 20)` and `clip(tw_th, -3, 3)` "to prevent overflow warnings" — band-aids for feeding raw logits to an unbounded `exp()`. The correct `(2σ)²` form is inherently bounded and needs neither. |
| **Fix** | Rewrote the decode in `host_vck190/yolo_decode.py` and **verified** it against the model's own PyTorch `Detect` head — max disagreement 9e-5. See [[Host Application]]. |

## 9. Diagnosed: SiLU prevents DPU mapping *(not a code bug)*

| | |
| --- | --- |
| **Severity** | The project's central blocker |
| **Symptom** | The model compiles, but into **52 DPU subgraphs** instead of 1, with 98 ops on the CPU |
| **Root cause** | `aten::silu_` has no XIR definition. Each SiLU lands on the CPU, and the compiler wraps each in a transpose pair for layout conversion — so every activation becomes a wall between DPU regions. |
| **Evidence trail** | Quantizer: `[VAIQ_WARN][QUANTIZER_TORCH_FLOAT_OP] ... aten::silu_ as a float operator`. Export: 49 × `not defined in XIR`. Compile: 98 × `transpose ... assigned to CPU`. Generated module: 49 × `py_nndct.nn.Module('aten::silu_')` — the generic fallback, where every other op got a purpose-built quantized class. |
| **Fix proven** | Swapping to `LeakyReLU` gives **1 DPU subgraph, 0 CPU ops, 0 warnings** — measured, not predicted ([[SiLU to LeakyReLU Experiment]]). |
| **Still outstanding** | A **finetune** so the weights suit the new activation. Not a code change; needs GPU time. |

---

## Corrections made to our own documentation

Honesty record — two things this session got wrong and then fixed:

1. **The LeakyReLU alpha.** An early note claimed a plain `0.1` was risky and only `0.1015625`
   (26/256) would map to the DPU. Evidence from the July project disproved it: it used `0.1` and
   the quantizer **snapped the slope to 26/256 itself**, compiling to one subgraph. Both values
   work. [[Vitis AI DPU Concepts]] §11 was rewritten.
2. **A bogus anchor-mismatch warning** in `verify_decode.py`. The check indexed a broadcast
   `anchor_grid` as `.reshape(-1,2)[:3]`, which returns anchor 0 three times rather than the three
   anchors. The anchors were always correct; the check was wrong. Now indexes `[0, :, 0, 0, :]`.


## 10. `np.trapezoid` crashed every container-side `val.py`

| | |
| --- | --- |
| **Severity** | High — silent until 10 minutes of inference had already been spent |
| **Symptom** | `AttributeError: module 'numpy' has no attribute 'trapezoid'` in `utils/metrics.py:116`, *after* a full 5000-image validation pass completed |
| **Root cause** | `compute_ap` called `np.trapezoid`, the **NumPy 2.x** spelling. The Vitis AI image ships NumPy 1.x, where the function is `np.trapz`. The repo's own code therefore worked on the host and could never work in the container. |
| **Why it mattered** | It fires in `ap_per_class`, which runs *after* inference and *before* `--save-json` writes anything — so the entire run was lost, predictions included. |
| **Fix** | `trapezoid = getattr(np, "trapezoid", None) or np.trapz` — works on both. |
| **How it was found** | Running `val.py` inside the container for the first time, to build a float baseline for `yolov3_original.pt`. Nothing had ever exercised that path in the container before. |

## 11. Docker's default `/dev/shm` killed the DataLoader

| | |
| --- | --- |
| **Severity** | Medium — clean failure, but non-obvious |
| **Symptom** | `RuntimeError: DataLoader worker (pid …) is killed by signal: Bus error` |
| **Root cause** | `docker run` defaults `/dev/shm` to 64 MB; `val.py` uses 8 workers sharing tensors through it. |
| **Fix** | `--shm-size=8g` on the `docker run` line. `vitis_run.sh` does not set it, so any future `val.py` invocation needs it — [[Compat Shims]]. |

## 12. The host preprocessed with a plain resize, costing real mAP

| | |
| --- | --- |
| **Severity** | High — a pure accuracy tax, invisible in every log |
| **Symptom** | Nothing failed. The INT8 model scored 0.4247 mAP\@0.5 where the board's own earlier run scored **0.4672 on the identical images** — and the board also beat our *float* HardSwish model at 0.4355. |
| **Root cause** | `yolov8_decode.preprocess` squashed each image to 640×640, distorting every object's aspect ratio. The prior work's board runner (`Versal_AI/src/runtime/dpu_runner.py`) **letterboxes** — aspect-preserving resize with 114-grey padding. |
| **Why it happened** | [[Implementation Plan - YOLOv8]] carried a rule inherited from the YOLOv3 work: *"preprocessing must match calibration: the v3 host uses plain resize, not letterbox."* The principle (calibration and inference must agree) is right; the specific instruction was wrong for this model and was followed literally. |
| **Fix** | `letterbox()` and `unletterbox_boxes()` in `host_vck190/yolov8_decode.py`; `--letterbox` on the eval harness; `v8_quantize.py` now letterboxes by default so both ends agree. |
| **Impact** | **+0.0208** mAP\@0.5 in float, **+0.047** after quantization — [[YOLOv8 Letterbox and the Activation Cost]]. |
| **How it was found** | A number that could not be true. An INT8 model cannot beat its own float, so when the board out-scored our float model on identical images, something upstream of quantization had to differ. |

> [!danger] The lesson
> The bug was not in the code; it was in a **documented rule applied out of its original
> context**. Inherited guidance deserves the same scepticism as inherited numbers — which is
> exactly what [[YOLOv8 Prior Numbers Reconciled]] had just established about the numbers.
>
> The second lesson is that the prior work was *internally inconsistent* — calibrating with a
> plain resize, inferring with letterbox — and still beat a consistent pipeline. Preserving
> aspect ratio is worth more than the mismatch costs. Consistency is a good default, not a
> justification for a worse transform.

## 13. Diagnosed: arm B will not compile on YOLOv8 *(not a code bug)*

| | |
| --- | --- |
| **Severity** | Medium — removes a planned deliverable, but not the measurement behind it |
| **Symptom** | `vai_c_xir` aborts: `[XCOM_OPFACTORY_OP_UNSUPPORTED] Op_type 18 is invalid for xcompiler`, then the gate fails to deserialize a file that was never written. |
| **Root cause** | Not established precisely. The exact `x·sigmoid(x)` decomposition compiles fine on **YOLOv3** (50 DPU subgraphs, `sigmoid` on the CPU — [[SiLU Decomposition]]), so it is specific to YOLOv8's graph; most likely the elementwise multiply inside `C2f`'s split/concat structure, which the YOLOv3 graph let the DPU absorb. |
| **Consequence** | Arm B is **simulation-only** on YOLOv8. Its number still exists, because the INT8 simulation reproduces the board to 0.004 mAP. |
| **Action** | The plan's expectation of an on-board arm B at ~1 FPS is struck — [[YOLOv8 Quantization and Compile]]. |

## 14. `np.trapz` — the mirror image of #10

| | |
| --- | --- |
| **Severity** | Low — immediate, clear failure |
| **Symptom** | `AttributeError: module 'numpy' has no attribute 'trapz'` from `tools/board_curves.py` |
| **Root cause** | NumPy **2.x removed** `trapz` (renamed `trapezoid`). Bug #10 was the same wall from the other side: `utils/metrics.py` called `np.trapezoid`, the 2.x name, and died on the container's NumPy 1.x. |
| **Why it appeared now** | `board_curves.py` had only ever run inside the container (NumPy 1.x). Running it on the host venv (NumPy 2.x) exercised the other branch for the first time. |
| **Fix** | `_trapezoid = getattr(np, "trapezoid", None) or np.trapz` — the same idiom as #10, so the file runs under both. |

> [!note] The general shape
> This repo straddles two NumPy generations — a 1.x container and a 2.x host — so any file that
> might run in both needs the `getattr` form. #10 and #14 are the same bug meeting from opposite
> directions.

---

Back to [[Code Map]] | [[Home]]
