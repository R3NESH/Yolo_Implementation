---
tags: [findings, board, yolov8, leakyrelu, deployment]
date: 2026-10-03
result: runs on the VCK190, 1 DPU subgraph, mAP@0.5 0.5104 (97.4% of its float score); not stock yolov8s
---

# YOLOv8s LeakyReLU on Hardware

A **YOLOv8s trained with LeakyReLU** (checkpoint `yolov8_leaky_relu_trained_e15.pt`, now
`Yolo_v8_Versal_Implementation/yolov8s_leaky.pt`) was put through the same pipeline as the four
HardSwish models and run on the VCK190. It needed **no activation substitution**: LeakyReLU is
native to the DPU.

Related: [[YOLOv8 on Hardware]], [[YOLOv8 Quantization and Compile]],
[[YOLOv8 Letterbox and the Activation Cost]], [[Implementation Plan - YOLOv8]],
[[SiLU to LeakyReLU Experiment]], [[Files Not in Git]].

> [!warning] It is **not stock yolov8s**
> Each branch of its detection head has **two layers where stock yolov8s has three** — one fewer
> 3x3 conv per branch, six in all. Measured by loading both checkpoints:
>
> | | this model | stock yolov8s |
> | --- | --- | --- |
> | Conv2d layers | 58 | 64 |
> | Parameters | 10.612 M | 11.167 M |
> | Compute | 25.54 GOP/img | 28.64 GOP/img |
> | DPU operators | 226 | 244 |
>
> So any difference from the HardSwish yolov8s on the board is **not attributable to the
> activation alone** — the weights, the training and the head all differ. The recipe, dataset
> and the fork's changes were never seen; "trained for 15 epochs" is read from the file name.

---

## 1. Results (COCO val2017, 5000 images, board)

| | LeakyReLU yolov8s | HardSwish yolov8s |
| --- | --- | --- |
| mAP@0.5 | **0.5104** | 0.5423 |
| mAP@0.5:0.95 | 0.3286 | 0.3716 |
| mAP@0.75 | 0.3383 | 0.3931 |
| AR@100 | 0.5283 | **0.5603** |
| Precision / recall at best F1 | 0.5817 / 0.4854 | 0.6523 / 0.4906 |
| Best-F1 confidence | 0.183 | 0.377 |
| Float ceiling, 640 letterbox, own weights | 0.5241 / 0.3413 | 0.6106 / 0.4440 |
| **Retained vs own float @0.5** | **97.4 %** | 88.8 % |
| Second scorer, mAP@0.5 | 0.5060 | 0.5391 |

> [!warning] Correction
> A chat summary written during this work said the leaky model had the **higher recall**. That was
> wrong: it compared the leaky AR@100 (0.5283) with the HardSwish model's recall at its best-F1
> point (0.4906) instead of its AR@100 (0.5603). The HardSwish yolov8s is higher on every
> accuracy row, recall included. The only thing the leaky model leads on is **how much float
> accuracy it keeps** (97.4% vs 88.8%).

The leaky model starts from a lower float ceiling but **loses far less of it** to INT8. It does not
beat the HardSwish yolov8s outright. A float baseline from the training log (0.4928 mAP@0.5) was
measured at **416** without letterbox and is **not** comparable to the board's 640 run; the
640-letterbox float number above is the right one.

Compile gate: **1 DPU subgraph**, 3 CPU subgraphs whose only op is `fix2float`. xmodel md5
`9a0df69894f095da55a60a8de9d208c6`.

## 2. Speed, power, memory

Measured the same day for both models, same commands.

| | LeakyReLU | HardSwish |
| --- | --- | --- |
| DPU-only (`--benchmark 200`, batch 6) | 42.8 img/s | 42.4 img/s |
| End to end (`--limit 1000 --no-eval`) | 14.2 img/s | 16.7 img/s |
| Detections kept per image | 154.6 | 124.6 |
| Power idle / load / inference-only | 12.90 / 13.80 / 0.89 W | 13.27 / 14.00 / 0.73 W *(earlier session)* |
| Energy per image | 0.972 J *(using 14.2 img/s)* | 0.883 J |
| DDR footprint | 33.5 MiB | 34.4 MiB |
| Array utilisation | 95.1 % | 95.5 % |

The DPU is no faster despite ~11% less arithmetic. End to end it is slower, and it keeps ~24% more
detections per image, which fits the extra cost being host decode and NMS — **inferred, not
profiled.**

## 3. Two open discrepancies

> [!danger] The earlier DPU-only figure does not reproduce
> [[YOLOv8 on Hardware]] records the HardSwish yolov8s at **22.24 img/s DPU-only**. The same
> `--benchmark 200` on the same xmodel today gives **42.4**. `host_vck190/board_eval_vck190.py`
> has uncommitted-then-committed changes since, and the benchmark loop now decodes frames once up
> front; whether the old measurement did the same was **not checked**. Only same-day figures
> should be compared. Do not set the old number beside today's.

> [!warning] The full scored run was slower than the 1000-image run
> 5000 images took **7:37 (10.9 img/s)**, against 14.2 img/s for 1000 images without scoring.
> The board's disk was **full to the last block** at the time, which may be why. Not repeated.
> Energy per image above uses 14.2; using 10.9 it would be about 1.27 J.

Also: the independent scorer differs from pycocotools by 0.0044 here; on the earlier four it was
within 0.003. Not investigated.

## 4. How it was built (nothing special)

```bash
cd Yolo_v8_Versal_Implementation
./v8_run.sh python -u v8_quantize.py --weights yolov8s_leaky.pt --act silu \
    --output_dir quantize_result/yolov8s_leaky --quant_mode calib
./v8_run.sh python -u v8_quantize.py --weights yolov8s_leaky.pt --act silu \
    --output_dir quantize_result/yolov8s_leaky --quant_mode test
./v8_compile.sh yolov8s leaky
./deploy_to_board.sh yolov8s_leaky
```

`--act silu` is the wrapper's name for **"leave activations alone"** — this model has no SiLU, and
the wrapper reported **0 replacements**. A name like `none` would be clearer; not changed.

On the board (files in `/home/root/yolo/`; the COCO annotations are at
`ann/instances_val2017.json`, **not** the top level):

```bash
python3 board_eval_vck190.py --arch v8 --model yolov8s_leaky_vck190.xmodel --images val2017 \
    --annotations ann/instances_val2017.json \
    --out-json preds_yolov8s_leaky.json --out-csv metrics_yolov8s_leaky.csv
python3 measure_power.py --model yolov8s_leaky_vck190.xmodel --images val2017 \
    --benchmark 400 --reference-fps 14.2
```

Float baseline: `./v8_run.sh python -u v8_eval_quantized.py --weights yolov8s_leaky.pt --act silu
--float --letterbox --limit 0 --out sim_results/preds_yolov8s_leaky_float_lb.json` (~18 min), then
scored with pycocotools.

## 5. Not done

- **A PDF report was not produced.** `tools/build_report_leaky.py` writes
  `Yolo_v8_Versal_Implementation/report_leaky/report.html` plus figures and
  `board_results/metrics/yolov8s_leaky_metrics.csv`. LibreOffice then **failed to convert the HTML**:
  the first attempt hung ~14 minutes at 100% CPU with a 0-byte temp file, and a retry with a clean
  profile aborted at once with `Io Abort Code:27`. Cause not found. `report_leaky/` is **not in
  git**; regenerate with `~/Desktop/Versal_AI/venv/bin/python tools/build_report_leaky.py`
  (needs the COCO annotations at `~/Desktop/Versal_AI/datasets/coco/annotations/` and both
  predictions files, which *are* in git). Its prose was drafted from these numbers and not
  re-read after the build.
- No INT8 CPU simulation and no float run with the DPU's rounded LeakyReLU slope, so the 2.6%
  float-to-board loss is **not split** between 8-bit arithmetic and the slope.
- No power rail voltages/currents, no thermal data.

## 6. State of the board when this was written

Powered at `192.168.1.10` once `ip addr add 192.168.1.10/24 dev eth0` was run on its console (the
address is lost on reboot). Its disk was **100% full**; to fit the new files,
`/home/root/yolo/preds_yolov8l_hs.json` was deleted on the board (the host keeps an identical
copy under `board_results/`). `yolov8s_leaky_vck190.xmodel` and `preds_yolov8s_leaky.json` are on
the board.
