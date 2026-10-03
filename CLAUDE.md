# Orientation for an agent opening this repo

This is a working copy for deploying **YOLOv3 and YOLOv8 onto an AMD Versal VCK190** with the
Vitis AI DPU toolchain. The project's memory is an Obsidian-style vault, not this file.

**Start at `Versal_YOLOv3_Vault/00 Start Here/Home.md`.** It has the status table and the map.
Then, for the most recent work, `05 Runs & Findings/YOLOv8s LeakyReLU on Hardware.md`.

## Things you will otherwise trip over

- **Nine files over 100 MB are deliberately not in git** (INT8 quantizer intermediates and three
  YOLOv3 weights). `Versal_YOLOv3_Vault/06 Environment/Files Not in Git.md` lists each one, how to
  recreate it, and which cannot be recreated and must be asked for. Do not treat the missing paths as
  corruption. Every *compiled* `.xmodel` **is** in the repo, so running on the board needs none of them.
- `*.pt`, `*.xmodel` and `*.pth` go through **Git LFS** (`.gitattributes`). Install `git-lfs` before cloning.
- COCO data is **not** in the repo. Scripts expect `~/Desktop/Versal_AI/datasets/` (see `v8_run.sh`, `DATA_ROOT`).
- Anything that runs the toolchain goes through the Vitis AI PyTorch container: `yolov3_test/vitis_run.sh`
  for YOLOv3, `Yolo_v8_Versal_Implementation/v8_run.sh` for YOLOv8.
- The board is at `192.168.1.10` (host side `192.168.1.20/24` on the USB ethernet interface). Its address
  is lost on reboot and its disk was full; see the leaky note, section 6. Use `scp -O`.
- Do not trust a throughput number from before 2026-10-03 without re-measuring; one did not reproduce
  (leaky note, section 3).

## Unfinished

- The PDF report for the LeakyReLU yolov8s was not produced (LibreOffice failed to convert the HTML).
  `tools/build_report_leaky.py` regenerates the HTML. Details in the leaky note, section 5.
- A HardSwish **finetune** is the identified fix for the YOLOv8 accuracy gap; it needs a GPU and was not done.

## Git

The GitHub copy of this work is `R3NESH/Yolo_Implementation` (remote `yolo_impl`, branch `main`), pushed from a
single-commit branch so that the three oversized weights stay out of its history. A local `main` with the
older history exists only on the original machine.
