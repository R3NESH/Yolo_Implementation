---
tags: [findings, dpu, silu, xir, vart, resolved, important]
date: 2026-09-11
result: best.pt runs on the board, no GPU required
---

# SiLU Decomposition

How `runs/train/exp2/weights/best.pt` — the accurate, SiLU-trained checkpoint — was made to
execute on the VCK190 without retraining anything.

The short version: **`SiLU(x) ≡ x · sigmoid(x)`**, and writing that identity out explicitly
before quantization turns one unimplementable op into two implemented ones.

Related: [[DPU Subgraph Fragmentation]], [[Board Bring-Up]],
[[Board mAP - LeakyReLU Without Finetune]], [[SiLU to LeakyReLU Experiment]].

---

## The problem, in three layers

Each layer looked like the whole problem until the next one appeared.

### Layer 1 — `vart.Runner` runs one subgraph

`compiled/` (best.pt) has **52 DPU subgraphs**. `board_eval_vck190.py` did:

```python
vart.Runner.create_runner(dpu_subgraphs[0], "run")
```

That runs subgraph 0 — the *first convolution* — whose output is `(1,416,416,32)`, not the
three 255-channel detection maps. Garbage, or a crash.

### Layer 2 — `GraphRunner` exists, and still cannot run it

`vitis_ai_library.GraphRunner` executes whole graphs, every DPU and CPU subgraph in order. It
looked like the answer. It is not:

```
[UNILOG][FATAL][VAILIB_CPU_RUNNER_OPEN_LIB_ERROR][dlopen can not open lib!]
lib=libvart_op_imp_aten__silu_.so ... No such file or directory
op = xir::Op{... type = aten::silu_}
```

Every CPU subgraph needs a runtime **operator implementation** shared library. The board ships
68 of them; none is silu.

> [!danger] Compiling successfully proves nothing about running
> XIR has no definition for `aten::silu_`, so at compile time it **invents one**:
> *"XIR creates the definition of this operator automatically."* That satisfies `vai_c_xir`,
> which is why `compiled/` exists at all. It says nothing about whether any runtime can execute
> the op. `compiled/` is a 35 MB artefact that can never run on any board.

### Layer 3 — the ops we need are already there

```bash
ls /usr/lib/libvart_op_imp_*.so | wc -l          # 68
ls /usr/lib/libvart_op_imp_*.so | grep -i silu   # (nothing)
ls /usr/lib/libvart_op_imp_sigmoid.so            # exists
ls /usr/lib/libvart_op_imp_mul.so                # exists
```

And `SiLU` is *defined* as `x · sigmoid(x)`. The op is missing; the maths is not.

## The fix

`yolov3_test/dpu_silu_decompose_AB5.py`. It mirrors `dpu_silu_experiment.py`, but instead of
*replacing* the activation with a different function it *rewrites the same function*:

```python
class SiLUDecomposed(nn.Module):
    """nn.SiLU written out as x * sigmoid(x)."""
    def forward(self, x):
        return x * torch.sigmoid(x)
```

```python
def decompose_silu(module):
    """Recursively replaces every nn.SiLU with SiLUDecomposed. Returns the count."""
```

> [!important] This is not an approximation
> Unlike the LeakyReLU swap, **nothing about the computed function changes**. Same weights,
> same outputs, bit-for-bit the same maths in float. The script asserts the identity with
> `torch.allclose` before quantizing. The only thing that changes is how the graph is *traced*.

Then the identical quantize → export → compile pipeline:

```bash
./vitis_run.sh python -u dpu_silu_decompose_AB5.py --quant_mode calib
./vitis_run.sh python -u dpu_silu_decompose_AB5.py --quant_mode test
./vitis_run.sh vai_c_xir \
    -x quantize_result_silu_decomp/YOLOv3DPUWrapper_int.xmodel \
    -a /opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json \
    -o compiled_silu_decomp -n yolov3_vck190_silu_decomp
```

**No GPU. No training data. About 10 minutes of container CPU.**

## What changed in the graph

The tracer immediately shows the substitution taking effect:

```
YOLOv3DPUWrapper/Conv[layers]/ModuleList[0]/SiLUDecomposed[act]/ret.7,  type = sigmoid
YOLOv3DPUWrapper/Conv[layers]/ModuleList[0]/SiLUDecomposed[act]/ret.9,  type = mul
```

| Signal | `compiled/` (nn.SiLU) | `compiled_silu_decomp/` |
| --- | --- | --- |
| Quantizer float-op warnings | 1 (`aten::silu_`) | **0** |
| XIR "not defined in XIR" warnings | 49 | **0** |
| Total subgraphs | 105 | 103 |
| DPU subgraphs | 52 | 50 |
| CPU op types | `transpose` (+ the phantom silu) | `sigmoid` ×49, `fix2float` ×52, `float2fix` ×49 |
| Runs on the board? | ❌ **never** | ✅ **yes** |

Note `mul` does **not** appear in the CPU ops — the compiler absorbed it. Only the sigmoid and
the fixed/float conversions fall back to the CPU, and all three have runtime libraries.

## It runs

```
model has 50 DPU subgraph(s); using the 'graph' runner
input (6, 416, 416, 3) (NHWC)  outputs [(6,52,52,255), (6,26,26,255), (6,13,13,255)]
images processed: 50 | detections kept: 265
real  0m56.598s
```

### Early health signal

Same 50 images, same `--conf-thres 0.25`:

| Build | Detections kept |
| --- | --- |
| `compiled_lrelu` (no finetune) | 464 |
| `compiled_silu_decomp` (best.pt) | **265** |

**Fewer is better here.** The LeakyReLU build saturated the detection cap on every image
([[Board mAP - LeakyReLU Without Finetune]]) because it was emitting near-uniform noise. A
model producing 265 confident detections across 50 images is behaving like a detector.

## The cost: it is slow

| Build | Subgraphs | Throughput |
| --- | --- | --- |
| `compiled_lrelu` | 1 DPU | **56.3 FPS** (DPU+preprocess, batch 6) |
| `compiled_silu_decomp` | 50 DPU + 50 CPU | **~1.0-1.6 FPS** end to end |

Roughly 40-60x slower. Every frame pays 50 DPU↔CPU round trips, each a DMA transfer plus a CPU
kernel, plus 49 sigmoids evaluated on the ARM cores.

> [!note] So the finetune is still worth doing
> This path gives **accuracy without a GPU**. The LeakyReLU finetune gives **accuracy and
> speed**, at the cost of GPU time. They are complementary, not alternatives:
>
> | Path | Accurate | Fast | GPU |
> | --- | --- | --- | --- |
> | `compiled_silu_decomp` + `GraphRunner` | ✅ | ❌ ~1 FPS | No |
> | Finetuned LeakyReLU + `vart.Runner` | ✅ after training | ✅ 56 FPS | Yes |

## GraphRunner API notes

It is not a drop-in replacement for `vart.Runner`:

```python
from vitis_ai_library import GraphRunner
graph  = xir.Graph.deserialize(path)      # keep a reference! see [[Board Bring-Up]]
runner = GraphRunner.create_graph_runner(graph)   # takes the WHOLE graph, returns vart.RunnerExt
ins, outs = runner.get_inputs(), runner.get_outputs()   # List[vart.TensorBuffer]
arr = np.asarray(ins[0])                  # buffer protocol -> numpy view
job = runner.execute_async(ins, outs); runner.wait(job)
```

| | `vart.Runner` | `GraphRunner` |
| --- | --- | --- |
| Argument | one subgraph + `"run"` | the whole `xir.Graph` |
| Buffers | **you allocate** them | **it owns** them; `np.asarray()` for a view |
| Input dtype | int8, you scale by `2**fix_point` | int8, you scale by `2**fix_point` — *same* |
| Output dtype | int8, **you** apply `1/2**fix_point` | **float32, already dequantized** |

That last row matters: applying the output scale twice on the GraphRunner path would silently
divide every score by 8. `board_eval_vck190.py` sets `scales = [1.0]` in graph mode for exactly
this reason.

## Host app support

`board_eval_vck190.py` now takes `--runner {auto,dpu,graph}`, defaulting to **auto**: it counts
the DPU subgraphs and picks `graph` when there is more than one. It also warns if you force
`dpu` on a fragmented model.

```bash
# picks 'dpu' automatically - 1 subgraph
python3 board_eval_vck190.py --model yolov3_vck190_lrelu.xmodel ...

# picks 'graph' automatically - 50 subgraphs
python3 board_eval_vck190.py --model yolov3_vck190_silu_decomp.xmodel ...
```

Regression-checked: the LeakyReLU build still returns exactly 464 detections on the 50-image
set after the change.

---

Back to [[Code Map]] | [[Home]]
