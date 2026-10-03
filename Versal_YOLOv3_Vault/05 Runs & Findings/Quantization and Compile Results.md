---
tags: [findings, quantization, compile, artifacts]
date: 2026-09-11
---

# Quantization and Compile Results

A record of every pipeline stage that was actually executed, what it produced, and what went wrong
along the way. Related: [[Vitis AI DPU Concepts]], [[DPU Subgraph Fragmentation]],
[[SiLU to LeakyReLU Experiment]], [[Implementation Plan]].

All commands run through [[Vitis AI Container]]; all logs in `yolov3_test/vitis_out/`.

---

## Stage 1 — Calibration

```bash
./vitis_run.sh python -u quantize_vitis_AB4.py --quant_mode calib \
  --weights runs/train/exp2/weights/best.pt \
  --data_dir ../datasets/coco128/images/train2017 \
  --output_dir quantize_result --img_size 416
```

✅ Succeeded. 100 calibration images, CPU, 179 ops traced.

| Artefact | Size | Meaning |
| --- | --- | --- |
| `quantize_result/quant_info.json` | 12 KB | per-tensor INT8 scaling parameters |
| `quantize_result/bias_corr.pth` | 199 KB | bias-correction data |
| `quantize_result/YOLOv3DPUWrapper.py` | 40 KB | the quantizer's own rewrite of the model |

Op tally from the trace:

| Op | Count | DPU-mappable? |
| --- | --- | --- |
| `_convolution` | 52 | ✅ |
| `silu_` | 49 | ❌ **left float** |
| `batch_norm` | 49 | ✅ (folded into conv) |
| `add` | 23 | ✅ (bottleneck residuals) |
| `upsample_nearest2d` | 2 | ✅ |
| `cat` | 2 | ✅ (FPN concats) |

> [!warning] The warning that mattered
> ```
> [VAIQ_WARN][QUANTIZER_TORCH_FLOAT_OP]: The quantizer recognize new op
> `aten::silu_` as a float operator by default.
> ```
> Easy to scroll past. It was the whole story — see [[DPU Subgraph Fragmentation]].

## Stage 2 — xmodel export

First attempt **failed**:

```
TypeError: export_xmodel() got an unexpected keyword argument 'deploy'
```

`quantize_vitis_AB4.py` called `export_xmodel(output_dir=..., deploy=True)`, but this Vitis AI
build's signature is:

```python
export_xmodel(self, output_dir, deploy_check=False, dynamic_batch=False)
```

Fixed to `deploy_check=False` (see [[Changes Log]]). `deploy_check=True` additionally dumps golden
per-layer tensors for on-board comparison — genuinely useful for validation later, but it costs
disk, which this machine does not have ([[Disk and System Constraints]]).

✅ Second attempt succeeded: `quantize_result/YOLOv3DPUWrapper_int.xmodel`, **132 MB**.

Export also emitted, 49 times:
```
[UNILOG][WARNING] The operator named ...SiLU_act..., type: aten::silu_,
is not defined in XIR. XIR creates the definition of this operator automatically.
```

## Stage 3 — Compile for VCK190

```bash
./vitis_run.sh vai_c_xir \
  -x quantize_result/YOLOv3DPUWrapper_int.xmodel \
  -a /opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json \
  -o compiled -n yolov3_vck190
```

✅ Compiled — ⚠️ but badly fragmented.

```
[UNILOG][INFO] Total device subgraph number 105, DPU subgraph number 52
```

| Artefact | Size |
| --- | --- |
| `compiled/yolov3_vck190.xmodel` | 35.4 MB |
| `compiled/meta.json` | lists **52 kernels** |
| `compiled/md5sum.txt` | `067156a50e30d2d73f9a73956e38114e` |

98 ops assigned to CPU, **all of them `transpose`** — the layout conversions wrapped around each
un-mappable SiLU.

## Stage 4 — The LeakyReLU control

Same pipeline, activations swapped in memory. Full write-up:
[[SiLU to LeakyReLU Experiment]].

```
[UNILOG][INFO] Total device subgraph number 5, DPU subgraph number 1
```

| Artefact | Size |
| --- | --- |
| `compiled_lrelu/yolov3_vck190_lrelu.xmodel` | 34.9 MB |
| `compiled_lrelu/meta.json` | lists **1 kernel** |
| `compiled_lrelu/md5sum.txt` | `76fe34e58e93a3c066144bd40f4e32c8` |

## Side-by-side

| | Baseline (SiLU) | LeakyReLU |
| --- | --- | --- |
| DPU subgraphs | 52 | **1** |
| Total device subgraphs | 105 | 5 |
| CPU-assigned ops | 98 | **0** |
| Kernels in `meta.json` | 52 | **1** |
| Float-op warnings | 1 | 0 |
| XIR undefined-op warnings | 49 | 0 |
| Deployable accuracy | yes (mAP\@0.5 ≈ 0.53) | **no** (activation swapped without finetuning) |

> [!note] Neither build is shippable
> The baseline has usable weights but unusable performance. The LeakyReLU build has usable
> performance but unusable weights. The deliverable is the *intersection*: a model **finetuned**
> with LeakyReLU(0.1015625). Nothing in the toolchain blocks that — only GPU time.

## Target DPU configuration

Both `meta.json` files report:

```
"target": "DPUCVDX8G_ISA3_C32B6"
"lib": "libvart-dpu-runner.so"
```

> [!important] Verify this against the board before bring-up
> `DPUCVDX8G_ISA3_C32B6` comes from the container's stock VCK190 `arch.json`. If the board's
> loaded DPU bitstream is a different configuration (different batch `B` or channel-parallelism
> `C`), the xmodel will be **rejected at runtime**. Compile against the `arch.json` that matches
> the actual platform image, not the default one.

---

Back to [[Code Map]] | [[Home]]
