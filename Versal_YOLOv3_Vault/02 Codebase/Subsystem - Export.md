---
tags: [codebase, export, onnx, xmodel]
date: 2026-09-11
---

# Subsystem - Export

`export.py` — 1600+ lines supporting eleven export targets, **none of which is the one this
project uses.**

Modules: [[export]], [[models.tf]], [[models.experimental]], [[export_dpu_wrapper_AB3]],
[[quantize_vitis_AB4]]. Related: [[Vitis AI DPU Concepts]], [[Subsystem - Models]].

---

## What it supports

`export_formats()` ([export.py:166](../../yolov3_test/export.py)) declares the table; `run()`
([export.py:1329](../../yolov3_test/export.py)) dispatches. One function per target:

| Format | Function | Line | Suffix |
| --- | --- | --- | --- |
| TorchScript | `export_torchscript` | 244 | `.torchscript` |
| ONNX | `export_onnx` | 295 | `.onnx` |
| OpenVINO | `export_openvino` | 391 | `_openvino_model` |
| PaddlePaddle | `export_paddle` | 483 | `_paddle_model` |
| CoreML | `export_coreml` | 534 | `.mlmodel` |
| TensorRT | `export_engine` | 593 | `.engine` |
| TF SavedModel | `export_saved_model` | 691 | `_saved_model` |
| TF GraphDef | `export_pb` | 788 | `.pb` |
| TF Lite | `export_tflite` | 831 | `.tflite` |
| Edge TPU | `export_edgetpu` | 901 | `_edgetpu.tflite` |
| TensorFlow.js | `export_tfjs` | 971 | `_web_model` |

Each table row also records CPU/GPU compatibility, which `run()` uses to validate `--device`
against the requested format.

## Why this project bypasses it entirely

> [!important] `export.py` has no Vitis AI / DPU target
> There is no `export_xmodel`, no `vai_q_pytorch` integration, no XIR awareness. The DPU path is
> implemented separately in [[export_dpu_wrapper_AB3]] + [[quantize_vitis_AB4]].

The DPU flow needs three things `export.py` structurally cannot provide:

1. **A quantizer in the loop.** `torch_quantizer` must *observe real data* over many forward
   passes to derive INT8 scaling factors. Every `export.py` target is a one-shot graph
   conversion — there is no calibration concept anywhere in the file.
2. **A modified detection head.** The DPU cannot run anchor decode. The wrapper replaces
   `Detect.forward` with its three raw convolutions. `export.py` exports the model *as-is*,
   including the head.
3. **XIR output.** The artefact must be an `.xmodel`, which nothing here emits.

### Path comparison

| | `export.py` | The DPU path |
| --- | --- | --- |
| Entry point | `export.py --include onnx` | `export_dpu_wrapper_AB3.py` → `quantize_vitis_AB4.py` |
| Detection head | kept, decode included | **stripped** to 3 raw convs |
| Precision | fp32/fp16 (some int8 via TFLite/OpenVINO) | **INT8**, calibrated |
| Calibration data | none (except TFLite/OpenVINO int8) | 100 images, required |
| Output | `.onnx` / `.engine` / `.tflite` / … | `.xmodel`, then compiled `.xmodel` |
| Target hardware | CPU / GPU / TPU / NPU | **DPUCVDX8G on VCK190** |
| Post-processing | often bundled in the graph | **entirely on host CPU** |

## Is ONNX a viable alternative route to the DPU?

Worth asking, because `export_onnx` exists and works, and Vitis AI does have an ONNX front end.

**Arguments for:** Vitis AI ships a `vai_q_onnx` quantizer and an ONNX Runtime execution provider
for the DPU. Going via ONNX would decouple the flow from the torch 1.13 / NumPy 1.22 constraints
that forced [[Compat Shims]] into existence, and ONNX export from the training environment's
torch 2.x would sidestep the pickle-compatibility problem entirely.

**Arguments against:**

- The head still has to be stripped, so `export_dpu_wrapper_AB3.py` remains necessary either way —
  ONNX changes the serialisation, not the graph-shaping problem.
- The SiLU problem is **unchanged**. It is an XIR op-support issue, not a front-end issue; SiLU
  will fragment an ONNX-derived graph exactly as badly. See [[DPU Subgraph Fragmentation]].
- The PyTorch path is already working end-to-end here — a compiled xmodel exists. Switching
  front ends now would discard verified ground for unverified ground.

**Assessment:** not worth it for this deployment. The one scenario that would justify it is if the
torch/NumPy version straddling becomes unmanageable — but the cleaner fix there is to re-save the
checkpoint as a `state_dict` ([[Compat Shims]]), which is far less work than changing toolchains.
*(The specifics of `vai_q_onnx` support in this Vitis AI release were not verified — unverified.)*

## `models/tf.py`

A parallel TensorFlow/Keras reimplementation of the model layers, used by `export_saved_model` to
rebuild the network in TF before emitting SavedModel/TFLite/EdgeTPU/TFJS. Large, and completely
unused by the Versal path — the container has no TensorFlow. Safe to ignore.

## Practical note

`export.py` is not runnable on this machine (no torch on the host, CPU-only container, and most
targets need their own heavyweight SDK — TensorRT, CoreML, OpenVINO). If a TorchScript or ONNX
artefact is ever wanted for cross-checking, produce it on the GPU machine that runs the finetune.

---

Back to [[Code Map]] | [[Home]]
