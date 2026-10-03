---
tags: [codebase, utils, peripheral, triage]
date: 2026-09-11
---

# Subsystem - Utils Peripheral

Triage of everything in `utils/` that is **not** on the core train/val/deploy path. The purpose of
this note is to let you decide "ignore" vs "matters" quickly.

Related: [[Subsystem - Utils Core]], [[Compat Shims]], [[Import Dependency Graph]].

---

## Load-bearing despite looking peripheral

### `utils/plots.py` — **on the checkpoint-loading import path**

[[utils.plots]] looks like pure visualisation, and almost is. But `models/yolo.py:27` imports
`feature_visualization` from it, which means **`torch.load` of any checkpoint imports
`utils.plots`**, which imports `seaborn` at module level ([utils/plots.py:15](../../yolov3_test/utils/plots.py)).

> [!warning] This is why a missing plotting library broke model loading
> `ModuleNotFoundError: No module named 'seaborn'` appeared while trying to *inspect a
> checkpoint*. Nothing was being plotted. The import chain is the whole explanation — see
> [[Compat Shims]] for the diagram. seaborn is now installed clone-locally at
> `vitis_compat/site/` purely to satisfy this import.

It also imports `Annotator` from `ultralytics.utils.plotting` (line 19) — the other shimmed
dependency. Local definitions it *does* provide: `Colors`/`colors` (line 72) and `save_one_box`
(line 468).

Contents otherwise: `plot_labels`, `plot_results`, `plot_val_study`, `feature_visualization`,
`plot_images`, confusion-matrix and PR-curve plotting. Only `feature_visualization` is reachable
from the deploy path, and only as an import.

### `utils/activations.py` — **checked, and unhelpful**

[[utils.activations]] defines: `SiLU`, `Hardswish`, `Mish`, `MemoryEfficientMish`, `FReLU`,
`AconC`, `MetaAconC`.

> [!important] There is no LeakyReLU here, and nothing here helps DPU compatibility
> Every activation in this module is either the one causing the problem (SiLU) or *also*
> transcendental/unsupported (Mish, FReLU, Acon). `Hardswish` is piecewise-linear and might map
> *(unverified)*, but it is not YOLOv3's activation and carries no accuracy pedigree here.
>
> The fix in [[SiLU to LeakyReLU Experiment]] uses `torch.nn.LeakyReLU` directly. This file is a
> dead end for the activation question — worth knowing so nobody goes looking.

## Peripheral and genuinely irrelevant to Versal

| Module | What it is | Verdict |
| --- | --- | --- |
| [[utils.loggers]] + [[utils.loggers.wandb]], [[utils.loggers.clearml]], [[utils.loggers.comet]] | Cloud experiment-tracking integrations. `utils/loggers/__init__.py` dispatches to whichever is installed; each subpackage degrades gracefully when absent. | **Ignore.** None installed; none needed. Note `clearml_utils.py:10` also imports from `ultralytics.utils.plotting`. |
| [[utils.segment]] (`augmentations`, `dataloaders`, `general`, `loss`, `metrics`, `plots`) | Instance-segmentation variants of the core utils, for `Segment` models. | **Ignore.** This is a detection model; `Segment` is defined but unused. |
| [[utils.aws]] (`resume.py`) | Resume training on an AWS spot instance. | **Ignore.** |
| [[utils.flask_rest_api.restapi]], [[utils.flask_rest_api.example_request]] | A toy Flask server exposing the model over HTTP. | **Ignore.** Not a deployment path — and note `restapi.py:47` calls `torch.hub.load`, i.e. it downloads from the internet. |
| [[utils.triton]] | NVIDIA Triton Inference Server client. | **Ignore.** Wrong vendor entirely for this project. |
| [[utils.downloads]] | `attempt_download`, `url_getsize`, Google Drive / GitHub release fetching. | **Ignore, and be careful** — anything that triggers it will try to reach the network and write files. |
| [[utils.callbacks]] | A tiny callback registry (`Callbacks`) used by `train.py`. | Structural only; nothing to configure. |
| [[utils.autobatch]] | Estimates the largest batch size that fits in VRAM. | Irrelevant — CPU-only container, and exp2 used a fixed `--batch-size 8`. |
| [[models.common_orig]] | A pristine backup copy of `models/common.py`. | **Useful as a diff target** — see [[Subsystem - Models]] — but not imported by anything. |
| [[models.tf]] | TensorFlow reimplementation for TF-family exports. | **Ignore.** See [[Subsystem - Export]]. |

## The dependency-weight observation

Several of these modules are why the import chain is so wide. `utils/general.py` alone pulls
`ultralytics.utils.checks` and `ultralytics.utils.patches` (lines 36-37), and the plotting chain
pulls seaborn and `ultralytics.utils.plotting`. For a *deployment* environment, none of this is
wanted — it exists to serve training and experiment tracking.

> [!tip] A cleaner long-term deployment story
> If the DPU flow is ever productionised, the right move is to stop loading pickled model objects
> altogether: save a plain `state_dict`, rebuild the network from yaml in a minimal script, and
> the entire `utils/` import chain — plotting, loggers, ultralytics, seaborn — drops away. That
> removes the need for [[Compat Shims]] completely. See the recommendation at the end of that note.

## Already-handled missing dependencies

Two imports fail in the container but are **already guarded by the repo** and need no shim:

| Import | Guard |
| --- | --- |
| `thop` (FLOPs counting) | `utils/torch_utils.py:26-29` — `try/except ImportError: thop = None` |
| `git` (GitPython) | `utils/general.py:399` — imported lazily *inside* `check_git_status()` |

---

Back to [[Code Map]] | [[Home]]
