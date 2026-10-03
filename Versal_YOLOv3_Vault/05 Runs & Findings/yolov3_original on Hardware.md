---
tags: [findings, board, deployment, dataset, class-mapping, important]
date: 2026-09-12
result: deploys unchanged; the checkpoint has no 'person' class and a +1 index shift
---

# yolov3_original on Hardware

Deploying `yolov3_test/yolov3_original.pt` — the **unmerged, stock YOLOv3** — to the VCK190,
alongside the merged23 checkpoint already running there ([[Board mAP - best.pt on Hardware]]).

Two results, and the second was not the one being looked for:

1. **The pipeline carried over with no code changes.** Only paths differed.
2. **The checkpoint was trained on 79 classes, not 80.** Its class list is the COCO-80 list
   with **`person` removed**, so model index `i` means standard class `i + 1`. Scored against
   standard labels it looks broken (mAP\@0.5 = 0.0349); scored in its own class space it is
   *better* than merged23.

Related: [[SiLU Decomposition]], [[DPU Subgraph Fragmentation]], [[Board Bring-Up]],
[[Subsystem - Models]], [[Subsystem - Data and Datasets]].

---

## 1. The pipeline needed no changes

Everything that made merged23 run applies unchanged, because none of it is model-specific:

| Stage | Why it transfers |
| --- | --- |
| `export_dpu_wrapper_AB3.py` | Rebuilds routing from each layer's own `m.f` / `m.i` / `model.save`. Nothing merged23-specific. |
| `dpu_silu_decompose_AB5.py` | Recurses over `named_children()` swapping `nn.SiLU`. **Still mandatory** — the original is SiLU too. |
| `host_vck190/yolo_decode.py` | Already hardcodes 416, 80 classes and the stock anchor set — all three match. |

Pre-flight (`inspect_model_AB1.py`) confirmed the match before any board time was spent:
`Detect` at `model.28` with 3 heads, `no = 85`, strides 8/16/32, anchors × stride equal to the
pixel anchors in `yolo_decode.py`, and `imgsz: 416` in the saved training args.

### What came out bigger

The stock `Bottleneck` is two convolutions where `Bottleneck_merged` is one, across 23 backbone
blocks — so 72 Conv modules against merged23's 49. That propagates straight through:

| | merged23 (`compiled_silu_decomp`) | original (`compiled_orig`) |
| --- | --- | --- |
| Conv modules → SiLU | 49 | **72** |
| Graph ops (`vai_c_xir`) | 562 | **792** |
| DPU subgraphs | 50 | **73** |
| CPU ops | sigmoid ×49, fix2float ×52, float2fix ×49 | sigmoid ×72, fix2float ×75, float2fix ×72 |
| Compiled xmodel | 34 MB | **61 MB** |
| Throughput | 1.04 s/img | **1.30 s/img** |
| Full val2017 | 1:26:38 | **~1:40** |

`inspect_xmodel.py` reports zero `silu` ops and only `sigmoid` / `fix2float` / `float2fix` on the
CPU side — all three have runtime libraries on the board, so it runs.

> [!note] Fragmentation scales with activation count, exactly
> 49 SiLU → 50 DPU subgraphs; 72 SiLU → 73. One DPU subgraph per activation boundary, plus one.
> The slowdown is the extra 23 DPU↔CPU round trips per frame, nothing else.

## 2. The checkpoint has no `person` class

The first float baseline came back at **mAP\@0.5 = 0.0349**, which reads as a broken model.
It is not. Three measurements separate "bad model" from "bad labels":

| Measurement | Value | What it rules out |
| --- | --- | --- |
| Class-**aware** mAP\@0.5, standard labels | 0.0349 | — |
| Class-**agnostic** AP\@0.5 | **0.4091** | Not localization: the boxes are good |
| `exp2/best.pt` through the same harness | **0.528 / 0.323** vs exp14's 0.5275 / 0.3228 | Not the harness: it reproduces to 0.0005 |

So the boxes are right and the harness is right — only the class indices disagree. Matching
confident detections to ground truth *ignoring class*, then reading off which true class each
predicted index lands on, recovers the mapping:

```
model idx  std name at that idx -> actually detects    purity      n
        0  person             -> bicycle             92.3%    169
        1  bicycle            -> car                 92.3%   1281
        2  car                -> motorcycle          95.1%    265
        ...
       78  hair drier         -> toothbrush          88.5%     26
       79  toothbrush         -> (no matches)         0.0%      0
```

**79 of 79 matched classes are exactly `standard index + 1`**, at 90–100 % purity, and the
mapping is a clean permutation. The only standard class never produced is **`person`**, and
model index 79 never fires — the checkpoint has 80 outputs but 79 real classes.

Whoever prepared its training data (`coco_output/converted`, on `/SN02DATA/groupA/pratibha1`)
dropped `person` and shifted every remaining class down by one, while leaving `nc: 80`.

> [!warning] `person` is 10,777 of 36,335 val2017 boxes — 29.7 %
> This checkpoint cannot detect people at all. That is a property of the weights, not of the
> deployment, and no amount of re-quantizing will recover it.

## 3. Float baselines, like for like

All at 416, `val.py` defaults, on container CPU (no GPU used anywhere in this session).

| Float @ 416 | classes | instances | mAP\@0.5 | mAP\@0.5:0.95 |
| --- | --- | --- | --- | --- |
| `yolov3_original.pt`, standard labels | 80 | 36,335 | 0.0349 | 0.0221 |
| `yolov3_original.pt`, its own class space | 79 | 25,558 | **0.559** | **0.374** |
| `exp2/best.pt` (merged23), same 79 classes | 79 | 25,558 | 0.525 | 0.322 |
| `exp2/best.pt`, all classes (= exp14) | 80 | 36,335 | 0.528 | 0.323 |

### What the merged bottleneck costs

On identical images, identical labels and identical protocol — the cleanest comparison the
project has had:

```
mAP@0.50      0.559 - 0.525 = 0.034   (6.5% relative)
mAP@0.50:0.95 0.374 - 0.322 = 0.052   (16.1% relative)
```

Dropping the 1×1 reduction from 23 backbone bottlenecks costs **~3.4 points of mAP\@0.5** and
**~5.2 points of mAP\@0.5:0.95** at 416. [[Subsystem - Models]] describes the block; this is its
price. Note the caveat: the two checkpoints were trained separately, so this is the cost of the
*delivered pair*, not of the block in isolation.

## 4. On the board

Full COCO val2017, all 5000 images, `compiled_orig/yolov3_vck190_orig.xmodel` via `GraphRunner`
(73 DPU subgraphs, so `--runner auto` picks it):

```bash
python3 board_eval_vck190.py --model yolov3_vck190_orig.xmodel \
    --images val2017 --annotations ann/instances_val2017.json --conf-thres 0.001
```

Inference took **1:40:35** (1.21 s/img) and produced 362,551 detections — against merged23's
1,007,510 for the same images and the same `--conf-thres 0.001`. Some of that gap is the missing
`person` class, but a 2.8× difference is more than person's 30 % share of the labels, so this
model is also simply less trigger-happy at low confidence. That is consistent with its higher
precision (0.69 against 0.64 in float).

That first run scored **mAP\@0.5 = 0.0299**, because `board_eval_vck190.py` still used the
standard class table. Remapping the saved predictions (+1) and rescoring gives the real figures:

| VCK190, INT8 | mAP\@0.5 | mAP\@0.5:0.95 | mAP\@0.75 | AR\@100 |
| --- | --- | --- | --- | --- |
| 79 classes the model can predict | **0.4681** | 0.2869 | 0.3061 | 0.4151 |
| all 80, person necessarily 0 | 0.4622 | 0.2833 | 0.3023 | 0.4099 |

### Against float — same metric, same class space

[[Board mAP - best.pt on Hardware]] had to compare a pycocotools board figure against an
Ultralytics float one. That gap is closed here: the float predictions JSON was remapped and run
through **the same pycocotools call on the board**, so both rows below are the same measurement.

| `yolov3_original.pt`, 79 classes | mAP\@0.5 | mAP\@0.5:0.95 |
| --- | --- | --- |
| Float @ 416 (pycocotools) | 0.5610 | 0.3724 |
| Float @ 416 (`val.py`, for reference) | 0.559 | 0.374 |
| **VCK190 INT8** (pycocotools) | **0.4681** | **0.2869** |
| **retained** | **83.4 %** | **77.0 %** |

The two float implementations agree to 0.4 %, which is worth knowing on its own: for this
pipeline the metric choice is not what moves the number.

### The stock model quantizes worse

| | float mAP\@0.5 | board mAP\@0.5 | lost to INT8 |
| --- | --- | --- | --- |
| `yolov3_original.pt` (79 cls) | 0.5610 | 0.4681 | **16.6 % relative** |
| `exp2/best.pt` merged23 (80 cls) | 0.5275 | 0.4829 | 8.5 % relative |

The stock model is the better float model and the *worse* hardware model — it starts 3.4 points
ahead and finishes 1.5 points behind. Whatever the merged bottleneck costs in accuracy, it
appears to buy back in quantization robustness; one fewer convolution per block is also one
fewer set of activation ranges to fit into INT8. The merged23 row still mixes metrics slightly
(its float number is Ultralytics'), so treat 8.5 % as approximate and 16.6 % as exact.

> [!note] Cross-checked by a second implementation
> `tools/board_curves.py` does its own IoU matching in numpy rather than calling pycocotools,
> and computes **0.4690** over the same 79 classes — 0.2 % from pycocotools' 0.4681. Two
> implementations sharing no code agreeing that closely is good evidence the decode, the class
> remap and the AP integration are all sound. Plots in `board_results/plots_orig/`.
>
> Best F1 is **0.515 at confidence 0.241**, so `--conf-thres 0.24` is the demo operating point
> (merged23's was 0.229).

## 5. `--class-offset`

`board_eval_vck190.py` mapped class index → COCO category id through `yd.COCO_CAT_IDS`, which is
correct only for standard 80-class models. It now takes an explicit offset, defaulting to 0 so
every previous run is unaffected:

```bash
python3 board_eval_vck190.py --model yolov3_vck190_orig.xmodel \
    --images val2017 --annotations ann/instances_val2017.json \
    --conf-thres 0.001 --class-offset 1
```

Predictions already written with the wrong table do **not** need re-inference — `rescore_shifted.py`
remaps the saved JSON and re-runs pycocotools in about eight minutes.

## Reproducing it

```bash
# host: quantize + compile (~25 min container CPU, no GPU)
./vitis_run.sh python -u dpu_silu_decompose_AB5.py --quant_mode calib \
    --weights yolov3_original.pt --output_dir quantize_result_orig
./vitis_run.sh python -u dpu_silu_decompose_AB5.py --quant_mode test \
    --weights yolov3_original.pt --output_dir quantize_result_orig
./vitis_run.sh vai_c_xir -x quantize_result_orig/YOLOv3DPUWrapper_int.xmodel \
    -a /opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json \
    -o compiled_orig -n yolov3_vck190_orig
./vitis_run.sh python ../tools/inspect_xmodel.py compiled_orig/yolov3_vck190_orig.xmodel

# board
scp -O compiled_orig/yolov3_vck190_orig.xmodel root@192.168.1.10:/home/root/yolo/
```

The float baselines need COCO val2017 on the host. It was pulled off the board
(`tar cf - -C /home/root/yolo val2017 ann | tar xf -`, 1m48s) onto the external HDD, since the
root filesystem is at 99 % — see [[Disk and System Constraints]]. Labels were converted from
`instances_val2017.json` (36,335 non-crowd boxes, matching COCO's published count) and the
dataset bind-mounted into the container at `/data`.

> [!tip] Two container gotchas this run hit
> `val.py` needs `--shm-size` on `docker run` or the DataLoader workers die with a bus error,
> and `utils/metrics.py` called `np.trapezoid` — a NumPy 2.x name the container's NumPy 1.x does
> not have. Both are fixed; the second is now `getattr(np, "trapezoid", None) or np.trapz`.
> Loading `exp2/best.pt` in the container also needs `np2_pickle_compat.apply()`
> ([[The numpy _core Segfault]]) — `yolov3_original.pt` does not, being pickled under NumPy 1.x.

---

Back to [[Code Map]] | [[Home]]
