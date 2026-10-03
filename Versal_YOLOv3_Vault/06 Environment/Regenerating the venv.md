---
tags: [environment, venv, recovery, howto]
date: 2026-09-11
---

# Regenerating the venv

How to rebuild the Python virtual environment from scratch, and when you actually need to.

Related: [[Disk and System Constraints]], [[Vitis AI Container]], [[Compat Shims]],
[[Clone Provenance]].

---

## What the venv was

| | |
| --- | --- |
| Location | `Yolo_v3_AB_New/yolov3_test/.venv` (the **original** folder) |
| Size | **8.2 GB** |
| Python | 3.12.3, from `/usr/bin/python3.12` |
| Contents | The 190 packages pinned in `yolov3_test/requirements.txt` — torch 2.13, TensorFlow 2.21, the full NVIDIA CUDA stack, plus the usual scientific stack |

Its `pyvenv.cfg` recorded:

```
home = /usr/bin
include-system-site-packages = false
version = 3.12.3
executable = /usr/bin/python3.12
command = /usr/bin/python3 -m venv /home/aesicdab/Documents/Yolo_v3_AB_New/yolov3_test/.venv
```

> [!info] There was never a venv in the clone
> The clone's `.venv` was a **symlink pointing back into the original** — a hazard, since writing
> to it would have modified the original. It was removed early on. See [[Clone Provenance]].
> The whole Vitis AI pipeline runs in Docker and needs no venv at all.

## Do you actually need it?

Check before spending 15 GB and half an hour:

| Task | Needs the venv? |
| --- | --- |
| Quantize / export / compile for the DPU | ❌ No — use `./vitis_run.sh` ([[Vitis AI Container]]) |
| Inspect a checkpoint | ❌ No — `./vitis_run.sh python inspect_model_AB1.py` |
| Generate the vault module notes | ❌ No — stdlib only (`python3 tools/graphify_codebase.py`) |
| **Train / finetune** (the LeakyReLU work) | ✅ **Yes** — or any other torch environment with a GPU |
| Run `val.py` for mAP | ✅ Yes |
| `export.py` to ONNX/TensorRT/etc. | ✅ Yes |

So: the venv matters for **training and validation only**. Everything on the deployment path is
containerised.

## Rebuild — full

> [!danger] Check disk first
> The full `requirements.txt` install needs **10–15 GB**. Verify headroom before starting, or pip
> will fail partway and leave a broken half-installed tree:
> ```bash
> df -h /          # want 20 GB+ free for comfort
> ```

```bash
cd /home/aesicdab/Documents/Yolo_v3_AB_New_Clone/yolov3_test

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Then verify:

```bash
.venv/bin/python -c "import torch, torchvision; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

Expect `cuda True` if the NVIDIA driver is healthy. `cuda False` with a GPU present usually means
a driver/toolkit version mismatch rather than a broken venv.

### Things that can go wrong

- **Pinned versions disappear.** `requirements.txt` pins exact versions (`torch==2.13.0`,
  `tensorflow==2.21.0`, `nvidia-*==...`). If a wheel has been yanked from PyPI, that line fails.
  Relax the specific pin rather than the whole file.
- **Disk exhaustion mid-install.** pip downloads to a cache first, then installs — so peak usage is
  roughly double the final size. `--no-cache-dir` reduces the peak at the cost of re-downloading
  on retry.
- **This is not the environment the DPU flow uses.** Do not expect `pytorch_nndct` here; it only
  exists inside the Vitis AI image.

## Rebuild — minimal (recommended when disk is tight)

The full file installs TensorFlow *and* CUDA *and* torch — but nothing in this project needs
TensorFlow. TF is only pulled in for the TF-family exports in `export.py`, which this deployment
does not use ([[Subsystem - Export]]).

For training/validation only:

```bash
python3 -m venv .venv
.venv/bin/pip install --no-cache-dir \
  torch torchvision \
  numpy opencv-python pillow pyyaml requests scipy tqdm pandas matplotlib seaborn \
  thop tensorboard
```

That covers `train.py` and `val.py`. Saves roughly half the footprint by dropping TensorFlow and
the standalone CUDA toolkit packages (torch ships its own CUDA runtime).

Add `--index-url https://download.pytorch.org/whl/cu121` (or whichever CUDA build matches the
driver) if the default wheel does not match the installed driver.

> [!warning] 4 GB of VRAM
> This machine's GPU is an **NVIDIA RTX A400 with 4 GB**. exp2 trained at `--batch-size 8` and
> 416px; that may or may not fit. If you hit CUDA OOM, drop to `--batch-size 4` and raise
> gradient accumulation, or train elsewhere. See [[Subsystem - Training]].

## Alternative: skip the venv entirely

For a one-off finetune, a torch Docker image avoids the disk cost of a persistent venv:

```bash
docker run --rm -it --gpus all \
  -v /home/aesicdab/Documents/Yolo_v3_AB_New_Clone:/workspace \
  -w /workspace/yolov3_test \
  pytorch/pytorch:latest bash
```

Same containment benefit as [[Vitis AI Container]] — only the clone is mounted — and nothing
persists on disk afterwards. Requires the NVIDIA container toolkit for `--gpus`.

---

Back to [[Code Map]] | [[Home]]
