---
tags: [environment, git, recreate, important]
date: 2026-10-03
---

# Files Not in Git

> [!tip] Read this if a path in the notes does not exist on your machine
> Nine files over **100 MB** are deliberately **not pushed** to
> `github.com/R3NESH/Yolo_Implementation`. Everything else needed to run on the board **is**
> pushed, including every compiled `.xmodel` (`yolov3_test/compiled*/`,
> `Yolo_v8_Versal_Implementation/compiled/`). So **you can deploy and run without recreating any
> of these**. You only need them to *re-quantize or re-compile* a model, or to retrain.

Related: [[Disk and System Constraints]], [[Vitis AI Container]], [[Implementation Plan - YOLOv8]],
[[SiLU Decomposition]], [[yolov3_original on Hardware]], [[YOLOv8 on Hardware]].

How they are excluded: listed in `.gitignore` under *"Files over 100 MB"*. Anything the repo
sends through Git LFS (`*.pt`, `*.xmodel`, see `.gitattributes`) counts against the LFS quota too,
which is another reason these stay out.

---

## 1. What is missing, and how to get it back

Two kinds. **Regenerable** ones are produced by a command here from files that *are* in the repo.
**Not regenerable** ones are inputs that came from somewhere else; get them from the source named.

| # | Path (relative to repo root) | Size | Kind |
| --- | --- | --- | --- |
| 1 | `Yolo_v8_Versal_Implementation/quantize_result/yolov8l_hardswish/YOLOv8DPUWrapper_int.xmodel` | 167 MiB | regenerable |
| 2 | `Yolo_v8_Versal_Implementation/quantize_result/yolov8m_hardswish/YOLOv8DPUWrapper_int.xmodel` | 99 MiB | regenerable |
| 3 | `yolov3_test/quantize_result/YOLOv3DPUWrapper_int.xmodel` | 132 MiB | regenerable |
| 4 | `yolov3_test/quantize_result_lrelu/YOLOv3DPUWrapper_int.xmodel` | 132 MiB | regenerable |
| 5 | `yolov3_test/quantize_result_silu_decomp/YOLOv3DPUWrapper_int.xmodel` | 132 MiB | regenerable |
| 6 | `yolov3_test/quantize_result_orig/YOLOv3DPUWrapper_int.xmodel` | 237 MiB | regenerable |
| 7 | `yolov3_test/yolov3_original.pt` | 118 MiB | **not regenerable** |
| 8 | `yolov3_test/runs/train/exp2/weights/best.pt` | 264 MiB | **not regenerable** (training output) |
| 9 | `yolov3_test/runs/train/exp2/weights/last.pt` | 264 MiB | **not regenerable** (training output) |

Rows 1-6 are the INT8 intermediates that `vai_q_pytorch` writes. The small files beside them
(`quant_info.json`, `bias_corr.pth`, the generated `*DPUWrapper.py`) **are** in the repo.

> [!warning] Recreated files will not be byte-identical
> The hashes below are for the originals. A re-run uses the same calibration images and settings
> but is not guaranteed to reproduce the same bytes. Judge a recreation by compiling it and
> checking the gate (one DPU subgraph, only `fix2float` on the CPU), then by the board score, not
> by the hash.

---

## 2. Recreating the YOLOv8 intermediates (rows 1-2)

Inputs that **are** in the repo: `yolov8m.pt`, `yolov8l.pt`, `v8_dpu_wrapper.py`,
`v8_quantize.py`, `v8_compile.sh`, `v8_run.sh`.

Input that is **not** in the repo: the calibration images. `v8_quantize.py` reads
`/datasets/coco_split/images/train`, the 4000-image split of COCO val2017 that `v8_run.sh`
bind-mounts from `DATA_ROOT` (default `/home/aesicdab/Desktop/Versal_AI/datasets`). On another
machine, point `DATA_ROOT` at a directory holding `coco_split/images/train` and `coco/`. Without
it, nothing below runs. It takes the 200 images at even stride through that folder, so the choice
of images is deterministic.

```bash
cd Yolo_v8_Versal_Implementation
# row 1 (167 MiB); for row 2 replace yolov8l with yolov8m
./v8_run.sh python -u v8_quantize.py --weights yolov8l.pt --act hardswish --quant_mode calib
./v8_run.sh python -u v8_quantize.py --weights yolov8l.pt --act hardswish --quant_mode test
# writes quantize_result/yolov8l_hardswish/YOLOv8DPUWrapper_int.xmodel
./v8_compile.sh yolov8l hardswish      # optional: recompiles, then prints the DPU/CPU gate
```

Needs the Vitis AI PyTorch container (`xilinx/vitis-ai-pytorch-cpu:latest`, see
[[Vitis AI Container]]) and CPU time; the large model takes longest (not timed). The committed
compiled models were built from exactly these, so you do not need to redo this to run them.

| Row | sha256 of the original |
| --- | --- |
| 1 `yolov8l_hardswish` | `a9e759df1f466799f6a1f3050c22bcd6fa2826a84fa26f6585edda1f652494be` |
| 2 `yolov8m_hardswish` | `4c8a479c3c5c351cc7301d2dc76ceb8a4c0d9482107224fdcc524a43de21bc37` |

---

## 3. Recreating the YOLOv3 intermediates (rows 3-6)

Run from `yolov3_test/` with `./vitis_run.sh` (container runner). Each is a **calib** pass then a
**test** pass; the `test` pass exports the xmodel. Defaults: 416 input, calibration images in
`../datasets/coco128/images/train2017` (not in the repo either; COCO128).

| Row | Directory it fills | Produced by | Weights |
| --- | --- | --- | --- |
| 3 | `quantize_result/` | `quantize_vitis_AB4.py --data_dir <calib dir>` | `runs/train/exp2/weights/best.pt` |
| 4 | `quantize_result_lrelu/` | `dpu_silu_experiment.py` | same |
| 5 | `quantize_result_silu_decomp/` | `dpu_silu_decompose_AB5.py` | same |
| 6 | `quantize_result_orig/` | `dpu_silu_decompose_AB5.py --output_dir quantize_result_orig` | `yolov3_original.pt` |

```bash
cd yolov3_test
# row 5 as the worked example (from [[SiLU Decomposition]])
./vitis_run.sh python -u dpu_silu_decompose_AB5.py --quant_mode calib
./vitis_run.sh python -u dpu_silu_decompose_AB5.py --quant_mode test
./vitis_run.sh vai_c_xir \
    -x quantize_result_silu_decomp/YOLOv3DPUWrapper_int.xmodel \
    -a /opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json \
    -o compiled_silu_decomp -n yolov3_vck190_silu_decomp

# row 6 (from [[yolov3_original on Hardware]]), ~25 min
./vitis_run.sh python -u dpu_silu_decompose_AB5.py --quant_mode calib \
    --weights yolov3_original.pt --output_dir quantize_result_orig
./vitis_run.sh python -u dpu_silu_decompose_AB5.py --quant_mode test \
    --weights yolov3_original.pt --output_dir quantize_result_orig
```

Rows 3, 4 and 6 depend on rows 8 and 7, which are not regenerable (section 4). Row 3 is the
build that **compiles but cannot run on the board**, for lack of a SiLU runtime library; it exists
only as evidence. See [[SiLU Decomposition]].

| Row | sha256 of the original |
| --- | --- |
| 3 `quantize_result` | `db2fee80865a5cca2a60ed2f28c4de7080aa95ab75db16b1b99edce1c9beca42` |
| 4 `quantize_result_lrelu` | `af1ae9c349ec1ac3ea26e957cdae617c1b9ce4c8db9021fd310326df19b9f295` |
| 5 `quantize_result_silu_decomp` | `b691a6e208bb75304fa371dbbc8f91d7a50c52006f579c98b74f9218d501fdc3` |
| 6 `quantize_result_orig` | `f20eb7772a768fd65bf733d1c549856aba33c371ef78b68948f4c2ba699b8705` |

---

## 4. The three weights that cannot be rebuilt (rows 7-9)

| Row | What it is | Where it came from | sha256 |
| --- | --- | --- | --- |
| 7 `yolov3_original.pt` | The **stock, unmerged YOLOv3**, trained on 79 classes (no `person`). See [[yolov3_original on Hardware]]. | Supplied from outside; it arrived inside a `yolov3_test.zip` bundle. Ask the person who supplied that bundle. | `45815a4be528d095ebb96678ddcd01da785f7e5324b361c923b2fd0cf08e65ac` |
| 8 `exp2/weights/best.pt` | Best checkpoint of training run **exp2**, **with optimizer state** (hence 264 MiB vs 66 MiB). See [[Training Run exp2]]. | Training output; exp2 started from `yolov3_merged23_e75.pt`. Retraining needs a GPU and will not give the same weights. | `c1381c6f5f8322a7c60711a95e652f0460ed787efa1b5460632d4fa287c2f15b` |
| 9 `exp2/weights/last.pt` | Last checkpoint of exp2 (the run was truncated; `--resume` uses this). | Same. | `4a77f8f5a16bce16dd4b3d85fbe60d21f63d5a2273fecd94d31ed55f7aec8f76` |

`yolov3_test/yolov3_merged23_e75.pt` (66 MiB) **is** in the repo. It is exp2's *starting* weights,
not its result, so it is not a substitute for rows 8-9.

> [!note] What you lose without them
> Nothing for running. `compiled/`, `compiled_lrelu/`, `compiled_silu_decomp/` and
> `compiled_orig/` are all committed and deploy as they are. Without rows 7-9 you cannot
> **re-quantize** the YOLOv3 models (rows 3-6) or **resume** training, and
> `inspect_model_AB1.py`, which defaults to `exp2/best.pt`, will not find its default weights.

---

## 5. Other files that are ignored but not large

Regenerable, kept out of git to save space, and not needed to run. All are re-created by running
the board evaluation or `val.py --save-json` again.

- `yolov3_test/board_results/*.json` (`silu_preds.json`, `orig_preds_shifted.json`)
- `yolov3_test/runs/val/*/*_predictions.json`
- `__pycache__/`, `*.swp`

---

## 6. If you are an agent opening this on a new machine

1. **Do not** treat the missing paths in section 1 as corruption. They are intentional.
2. To *run* a model on the board you need nothing from this note. Use the compiled xmodels and
   [[Host Application]].
3. To *rebuild* a YOLOv8 model, section 2 is complete apart from the calibration data and the
   Vitis AI container, which you must provide.
4. To rebuild a YOLOv3 model you additionally need rows 7-9, which only the project owner can
   supply. Ask rather than retraining.
5. Verify any recreated file against its hash above; expect a mismatch, then use the gate and
   the board score as described.
