---
tags: [environment, docker, vitis-ai, howto]
date: 2026-09-11
---

# Vitis AI Container

**Everything runs in Docker. There is deliberately no host virtualenv.**

Related: [[Disk and System Constraints]], [[Compat Shims]], [[Clone Provenance]].

---

## The runner

`yolov3_test/vitis_run.sh` wraps the whole invocation:

```bash
./vitis_run.sh python inspect_model_AB1.py     # run a script
./vitis_run.sh vai_c_xir --help                # run a tool
./vitis_run.sh bash                            # interactive shell
```

What it does:

```bash
exec docker run --rm "${TTY_FLAGS[@]}" \
  -u "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e PYTHONPATH="${COMPAT}:${COMPAT}/site" \
  -v "${CLONE_ROOT}:/workspace" \
  -w /workspace/yolov3_test \
  "${IMAGE}" \
  bash -lc '
    source /opt/vitis_ai/conda/etc/profile.d/conda.sh
    conda activate vitis-ai-pytorch
    exec "$@"
  ' _ "$@"
```

Each flag earns its place:

| Flag | Why |
| --- | --- |
| `--rm` | No container persists. Nothing accumulates on a disk that has no room. |
| `-u $(id -u):$(id -g)` | Files created land owned by **you**, not root. Without this, the container writes root-owned files into the repo that you then cannot delete without sudo. |
| `-e HOME=/tmp` | The container user has no real home; pip/conda caches go to the container's throwaway `/tmp`. |
| `-e PYTHONPATH=...` | Activates [[Compat Shims]] — only inside the container, so they can never shadow a real install on the host. |
| `-v CLONE_ROOT:/workspace` | **Only the clone is mounted.** The container cannot see the original folder or anything else on the machine. |
| `-w /workspace/yolov3_test` | Scripts expect repo-root-relative paths. |
| `[ -t 0 ] && TTY_FLAGS=(-it)` | Interactive when there is a terminal; pipeline-safe when there isn't. |

Override the image with `VITIS_IMAGE=... ./vitis_run.sh ...`.

## What's in the image

`xilinx/vitis-ai-pytorch-cpu:latest`, 11.7 GB, **already present on this machine** — it was not
pulled, and must not be re-pulled ([[Disk and System Constraints]]).

| Component | Version |
| --- | --- |
| Python | 3.8.6 |
| torch | 1.13.1 |
| torchvision | present |
| numpy | 1.22.0 |
| `pytorch_nndct` | present (the quantizer) |
| `vai_c_xir` | present (the compiler) |

Conda environments: `base`, **`vitis-ai-pytorch`** (the one to use), `vitis-ai-wego-torch`,
`vitis-ai-wego-torch2`.

> [!warning] CPU-only image
> This is the `-cpu` variant, matching `quantize_vitis_AB4.py`'s hardcoded
> `device = torch.device("cpu")`. Calibration over 100 images takes a few minutes rather than
> seconds. Fine for quantization; useless for training. A LeakyReLU finetune
> ([[SiLU to LeakyReLU Experiment]]) needs a GPU environment elsewhere.

## Board architecture files

Inside the container:

```
/opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json      <- the VCK190 target
```

Others available: `DPUCV2DX8G/{VEK280,V70}`, `DPUCZDX8G/{ZCU102,ZCU104,KV260}`,
`DPUCAHX8H/*`, `DPUCVDX8H/*`, `DPUCADF8H/U200`.

## Pre-installed gaps

Three first-party imports are missing from the image; all are handled by [[Compat Shims]]:

| Missing | Handling |
| --- | --- |
| `ultralytics` | offline shim in `vitis_compat/ultralytics/` |
| `seaborn` | installed clone-locally into `vitis_compat/site/` (2.2 MB, `--no-deps`) |
| `thop`, `git` | already guarded by try/except in the repo — no action needed |

## Why not a host venv

One was started and abandoned. `requirements.txt` pins torch 2.13 + TensorFlow 2.21 + the full
NVIDIA CUDA stack — 190 packages, 10–15 GB — and the disk had ~5 GB free. The container already
contains a working, version-consistent quantization toolchain, so a host venv would have been
both impossible and redundant. Full reasoning in [[Disk and System Constraints]].

---

Back to [[Code Map]] | [[Home]]
