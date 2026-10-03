---
tags: [findings, yolov8, prior-work, measurement, caution]
date: 2026-09-14
result: the deployment works; the accuracy numbers do not agree with each other
---

# YOLOv8 Prior Work in Versal_AI

What already exists for YOLOv8 on this board, found at **`/home/aesicdab/Desktop/Versal_AI`** —
a separate git repo, 8 commits, last touched 2026-07-10. The companion to
[[Prior Work in Yolo_v3_AB]], and required reading before starting
[[Implementation Plan - YOLOv8]].

The short version: **bring-up is solved, measurement is not.** Four models compile, run on the
VCK190 and produce sensible throughput. Their reported accuracy exists in five mutually
contradictory versions, one of which beats the float model.

> [!success] Resolved — read [[YOLOv8 Prior Numbers Reconciled]]
> The contradictions below have since been settled offline, with no board time. Every generation
> reproduces, and every disagreement is a **scoring artifact**: `_v1` is `_v3` truncated to 1015
> images (with `yolov8s`'s detections filed under `yolov8n`), and the older generations were
> thresholded at `sigmoid(-1) = 0.2689` before scoring, which caps AP by construction. **`_v3` is
> the only valid generation.** The suspected decode / class-mapping fault is no longer implicated.
> The section below is kept as the record of what the problem looked like before it was solved.

---

## What works, and works well

`tools/inspect_xmodel.py` run against `Versal_AI/build/compiled_model/`:

| | YOLOv3 here | YOLOv8n/s/m/l there |
| --- | --- | --- |
| DPU subgraphs | 73 | **2** |
| CPU ops | `sigmoid` ×72 | 1 softmax, 1 sigmoid, slices / reshapes / concats — the head only |
| Input | 416 | 640 |
| Board throughput | 0.83 FPS | **6–8 FPS** |

`build/replace_activation.py` swaps every `nn.SiLU` for `nn.Hardswish`, which the DPUCVDX8G
implements natively. The entire backbone and neck become a single DPU subgraph, and the
fragmentation problem that shaped this whole project ([[DPU Subgraph Fragmentation]]) disappears.

Board runs are real: 5000 images each, timings recorded in four separate CSVs, predictions JSONs
saved in `results_v3/`, `results_v4/` and `results_fixed/`.

## Float baselines, which look trustworthy

`benchmarks/pytorch_results.csv` and `benchmarks/onnx_results.csv`, COCO val2017 at 640:

| Model | Params | mAP\@0.5 | mAP\@0.5:0.95 | P | R |
| --- | --- | --- | --- | --- | --- |
| yolov8n | 3.16 M | 0.5187 | 0.3681 | 0.6347 | 0.4739 |
| yolov8s | 11.17 M | 0.6106 | 0.4440 | 0.6834 | 0.5616 |
| yolov8m | 25.90 M | 0.6654 | 0.4979 | 0.7164 | 0.6102 |
| yolov8l | 43.69 M | 0.6916 | 0.5244 | 0.7411 | 0.6331 |
| yolov8_leaky_relu_trained_e15 | 10.61 M | 0.5076 | 0.3391 | 0.6078 | 0.4769 |

PyTorch and ONNX agree to within 0.002 on every model, which is a genuine cross-check and a good
sign. These are consistent with published Ultralytics figures, and the checkpoint sizes match the
four `.pt` files now in `Yolo_v8_Versal_Implementation/`, so they describe exactly those weights.

## The board numbers, which do not

Five generations of `dpu_benchmark*.csv`, same models, same board:

| mAP\@0.5 | float | `dpu_benchmark` | `_v1` | `_v2` / `_fixed` | `_v3` |
| --- | --- | --- | --- | --- | --- |
| yolov8n | 0.5187 | 0.0248 | **0.5495** | 0.3586 | 0.4317 |
| yolov8s | 0.6106 | 0.0304 | **0.5495** | 0.4584 | 0.5301 |
| yolov8m | 0.6654 | 0.0367 | 0.6121 | 0.5087 | 0.5935 |
| yolov8l | 0.6916 | 0.0423 | 0.6473 | 0.5302 | 0.6219 |

Three things are wrong on the face of it:

1. **`yolov8n` has four different answers** spanning 0.0248 to 0.5495 — a 22× range for one model
   on one board.
2. **`_v1` reports `yolov8n` at 0.5495 against a float baseline of 0.5187.** An INT8 model does
   not beat its own float. Whatever produced that row is measuring something else.
3. **In `_v1`, `yolov8n` and `yolov8s` are byte-identical** across mAP50, mAP50-95, P, R and F1
   (`0.5494833742421418` both). One predictions file was scored twice under two names. Their
   *timings* differ, so the inference ran; only the scoring is duplicated.

Nothing in the repo records what changed between generations, so there is no way to tell which is
current from the files alone. The `_fixed` naming suggests `_v2` is the intended answer, but
`_v3` is newer and higher.

> [!danger] Do not inherit these numbers
> Any of them could be quoted in a report and none can be defended. This is the same failure mode
> as [[yolov3_original on Hardware]] — a plausible-looking mAP that is measuring the wrong thing —
> except there it was one wrong number and here it is five.

### A signature worth diagnosing

Across *every* generation, precision collapses far harder than recall: float P 0.63 → DPU
0.22–0.36 for `yolov8n`, while recall roughly halves. Lots of confident boxes that are not
matching ground truth. That is the signature of a decode or class-mapping fault rather than plain
quantization noise — precisely what the class-agnostic diagnostic separates in one pass
(`tools/recover_class_mapping.py`; the method is written up in [[yolov3_original on Hardware]]).

It may equally be genuine: YOLOv8's DFL head is known to be quantization-sensitive, and
HardSwish-for-SiLU is an *approximation* applied with no finetune. But those two explanations
predict different diagnostics, and nobody has run them.

### The LeakyReLU variant repeats a known result

`yolov8_leaky_relu_trained_e15` scores 0.5076 mAP\@0.5 in float and **0.1936** on the DPU — a 62 %
loss, and the slowest model of the set at 5.1 FPS. Compare [[Board mAP - LeakyReLU Without Finetune]],
where the YOLOv3 LeakyReLU build scored 11 % of float. Different magnitude, same lesson: swapping
the activation without enough retraining does not survive quantization.

## What to reuse

| Path | What it does |
| --- | --- |
| `build/replace_activation.py` | SiLU → HardSwish, recursive over `named_children()` |
| `build/quantize_pt.py`, `build/calibrator.py` | `vai_q_pytorch` calib/test with COCO val2017 |
| `build/export_xmodel.py`, `src/compile_models.py` | export and `vai_c_xir` |
| `src/validate_models_board.py`, `src/convert_coco.py` | board-side evaluation and COCO conversion |
| `build/compiled_model/*.xmodel` | already compiled — usable for a decode check without recompiling |
| `results_*/predictions_*.json` | **the most valuable asset**: board predictions already saved, so the numbers can be reconciled offline with no board time |

## The first task, and it costs nothing

Rescore the saved predictions JSONs against `instances_val2017.json` with `pycocotools`, exactly
as `host_vck190/rescore_shifted.py` does. That settles which generation is correct — or shows all
of them are wrong — before a single model is re-quantized or a minute of board time is spent.

---

Back to [[Implementation Plan - YOLOv8]] | [[Home]]
