---
tags: [findings, hardware, vart, board, bringup]
date: 2026-09-11
status: board verified, smoke test passed
---

# Board Bring-Up

First time anything ran on real hardware. The board arrived, it works, and the host
application needed three bug fixes before it would survive a single frame.

Related: [[Host Application]], [[Implementation Plan]], [[DPU Subgraph Fragmentation]],
[[SiLU to LeakyReLU Experiment]].

---

## The board

A VCK190 is connected over USB (FTDI FT4232H quad UART, so it appears as
`/dev/ttyUSB0..3`) and over USB gigabit ethernet.

| Item | Value |
| --- | --- |
| Board | Xilinx Versal **vck190 Eval board revA**, silicon v2, 8 GiB DRAM |
| Console | `/dev/ttyUSB1` @ 115200 (ttyUSB0/2 silent, ttyUSB3 is the system controller) |
| Boot | SD card, `BOOTMODE 0xE` (SD1_LS); two partitions, 1 GB boot + 13 GB rootfs |
| OS | **PetaLinux 2022.2**, kernel 5.15.36-xilinx-v2022.2, aarch64, root auto-login |
| Runtime | **VART 3.0.0**, xir 3.0, vaip 1.0.0 |
| Python | 3.9 with numpy 1.26.4, cv2 4.5.2, pycocotools |
| Network | `eth0`, no DHCP — set a static address by hand |

> [!warning] U-Boot will stop if you touch the console during autoboot
> The console sits in a 5-second `Hit any key to stop autoboot` window. Pressing Enter to
> "see if the board is alive" drops it to the `Versal>` prompt and it never boots. Type
> `boot` to continue. Also note U-Boot reports `No Valid Environment Area found` and
> `FEC: can't find phy-handle`, assigning random MACs — cosmetic, Linux brings `eth0` up fine.

### Networking it

There is no DHCP server on the USB ethernet link, so both ends are set by hand:

```bash
# host
ip -br addr show <usb-iface>          # 192.168.1.20/24
# board, over the serial console
ip addr add 192.168.1.10/24 dev eth0 && ip link set eth0 up
```

`scp` to the board needs `-O` (legacy protocol) — the PetaLinux image ships **no
`sftp-server`**, so modern OpenSSH clients fail with
`/usr/libexec/sftp-server: No such file or directory`.

## The DPU matches the compile target

This was the open risk flagged in [[Quantization and Compile Results]]. It is now closed:

```json
"DPU Arch":"DPUCVDX8G_ISA3_C32B6",
"DPU Batch Number":6,
"DPU Frequency (MHz)":333,
"fingerprint":"0x603000b56011861"
```

`xdputil query` on the board reports **`DPUCVDX8G_ISA3_C32B6`**, which is byte-identical to
the `target` string in both `compiled/meta.json` and `compiled_lrelu/meta.json`. No
recompilation against a different `arch.json` is needed.

> [!note] The DPU has a batch size of **6**
> This is a hardware property (`DPU Batch Number: 6`) and the input tensor is
> `(6, 416, 416, 3)`, not `(1, ...)`. [[Host Application]] fills only slot 0, so it is
> currently using one sixth of the available throughput.

## Three bugs in the host application

All three were latent in code that had been verified *off-board* — the decode maths were
correct, but nothing had ever driven VART itself. Each crashed the process on frame one.

### 1. The `xir.Graph` was garbage collected out from under the runner

The important one, and the one that looks like broken hardware.

```python
def build_runner(model_path):
    graph = xir.Graph.deserialize(model_path)     # local variable
    ...
    return vart.Runner.create_runner(dpu_subgraphs[0], "run")
    # graph refcount hits zero here; the runner still points into it
```

The VART runner holds raw pointers into the graph's memory. When the graph is collected the
runner is left dangling and `execute_async` dies with SIGSEGV or SIGBUS — **intermittently**,
because it depends on when the garbage collector happens to run.

This produced a genuinely misleading debugging session: an identical minimal script would
pass five times in a row when the graph was a module-level variable, and segfault instantly
when the same code lived inside a function. It is very easy to misread as a memory-alignment
problem, or as a flaky board.

**Fix:** anchor the graph for the process lifetime.

```python
_GRAPH_KEEPALIVE = []
...
graph = xir.Graph.deserialize(model_path)
_GRAPH_KEEPALIVE.append(graph)
```

`vart.Runner` is a pybind11 object and **rejects attribute assignment**, so the usual
`runner._graph = graph` trick does not work — hence the module-level list.

### 2. VART buffers were allocated `float32`; the DPU tensors are `xint8`

```python
in_buf = [np.empty(tuple(t.dims), dtype=np.float32, ...)]   # wrong
```

The tensors report `dtype=xint8`. A float32 buffer is four times the size VART expects, so
it writes well past the end. Buffers must be `np.int8`.

### 3. The input was never quantized

The input tensor carries `fix_point=6`, i.e. a scale of `2**6 = 64`. `preprocess()` returns
float in `[0, 1]`; cast straight to int8 that is almost entirely zeros, so the model would
have been looking at a black image.

```python
np.copyto(in_buf[0][0],
          np.clip(np.round(arr * in_scale), -128, 127).astype(np.int8))
```

Output dequantization was already correct (`scale = 1/2**fix_point`, here `fix_point=3` →
0.125), it was just being applied to a garbage buffer.

> [!tip] How to tell these apart quickly
> Write a ten-line script that makes a runner, allocates buffers and calls `execute_async`
> on zeros — no OpenCV, no decode, no tqdm. If that crashes, the problem is VART plumbing
> (bugs 1–2). If it runs and the numbers are wrong, the problem is quantization or decode.

## Smoke test result

```
50 images | 464 detections | no crashes
200 inferences in 7.55 s  ->  26.5 FPS  (37.7 ms/frame, DPU + preprocess)
```

The full chain executes: VART plumbing, the single DPU subgraph, NHWC handling, dequant,
the decode verified in [[Host Application]], and per-class NMS.

> [!danger] This says nothing about accuracy
> The model under test is `compiled_lrelu`, whose activations were swapped **without
> finetuning**. The detections prove the plumbing, not the model.

## Which runner to use, and why it matters

The board has three runner APIs. Picking the wrong one silently produces nonsense.

```python
>>> dir(vart)                  # ['Runner', 'RunnerExt', ...]
>>> dir(vitis_ai_library)      # ['GraphRunner', ...]
```

| API | What it executes | Use it for |
| --- | --- | --- |
| `vart.Runner` | **one** subgraph you hand it | Single-DPU-subgraph models (`compiled_lrelu/`) |
| `vart.RunnerExt` | same, with zero-copy tensor buffers | Performance work on single-subgraph models |
| `vitis_ai_library.GraphRunner` | **the whole graph** — every DPU and CPU subgraph, in order | Fragmented models (`compiled/`, 52 DPU + 52 CPU) |

> [!danger] `vart.Runner` on a fragmented model runs one convolution
> `board_eval_vck190.py` does `vart.Runner.create_runner(dpu_subgraphs[0], "run")`. On
> `compiled_lrelu/` that subgraph *is* the whole network, and its outputs are the three
> detection maps. On `compiled/` (the SiLU build) subgraph 0 is the **first convolution
> only** — input `(1,416,416,3)`, output `(1,416,416,32)`. The decode then receives a
> 32-channel feature blob instead of three 255-channel maps.
>
> Confirmed by deserialising both xmodels and listing the subgraphs:
>
> ```
> compiled/yolov3_vck190.xmodel        DPU 52 | CPU 52 | USER 1
>   execution order: USER,DPU,CPU,DPU,CPU,DPU,CPU,...
>   DPU[0] out (1,416,416,32)     <- one conv
>
> compiled_lrelu/..._lrelu.xmodel      DPU  1 | CPU  3 | USER 1
>   DPU[0] out (1,52,52,255) (1,26,26,255) (1,13,13,255)   <- whole model
> ```

### GraphRunner still cannot run `compiled/` — tested

It was tempting to conclude that `GraphRunner` makes the SiLU build deployable. It does not:

```
[UNILOG][FATAL][VAILIB_CPU_RUNNER_OPEN_LIB_ERROR][dlopen can not open lib!]
lib=libvart_op_imp_aten__silu_.so ... No such file or directory
op = xir::Op{... type = aten::silu_}
```

Every CPU subgraph needs a runtime **operator implementation** shared library. The board ships
68 of them:

```bash
ls /usr/lib/libvart_op_imp_*.so | wc -l     # 68
ls /usr/lib/libvart_op_imp_*.so | grep -i silu   # (nothing)
```

XIR auto-generating a definition for `aten::silu_` at *compile* time satisfies `vai_c_xir` and
says nothing about *runtime*. `compiled/` compiles and can never execute. This is the deeper
reason the SiLU build is unusable — fragmentation was only the visible symptom.

> [!tip] `sigmoid` and `mul` are both implemented
> ```bash
> /usr/lib/libvart_op_imp_sigmoid.so
> /usr/lib/libvart_op_imp_mul.so
> ```
> Since `SiLU(x) ≡ x * sigmoid(x)` exactly, rewriting the activation as an explicit
> `x * torch.sigmoid(x)` before quantization produces a graph made only of implemented ops —
> preserving best.pt's weights and its exact maths, with no GPU needed.
>
> ✅ **This was done and it works** — `compiled_silu_decomp/` executes on the board at ~1 FPS.
> Full write-up: [[SiLU Decomposition]].

`GraphRunner` also has a different API surface from `vart.Runner`:
`create_graph_runner(graph)` takes the **whole graph** and returns a `vart.RunnerExt`, whose
`get_inputs()` / `get_outputs()` hand back `TensorBuffer` objects rather than accepting numpy
arrays. It needs its own code path in the host app, not a drop-in swap.

## The real bottleneck is the CPU, not the DPU

> [!bug] Correction: this depends entirely on which model you measure
> An earlier revision of this note concluded that decode/NMS dominates by ~20x and the DPU is
> never the bottleneck. **That was measured on the un-finetuned LeakyReLU build, and it does
> not generalise.** That model saturates the 300-detection cap on every image, so NMS is
> handed an enormous candidate set. A model that detects properly produces far fewer
> candidates and NMS becomes cheap.

Measured at the mAP threshold (`--conf-thres 0.001`) over COCO val2017, for both builds:

| Build | DPU + preprocess | decode + NMS | End to end | Bottleneck |
| --- | --- | --- | --- | --- |
| `compiled_lrelu` (1 DPU subgraph, **broken accuracy**) | 18 ms | ~1042 ms | 1.06 s/img | CPU post-processing |
| `compiled_silu_decomp` (50 DPU subgraphs, **accurate**) | **957 ms** | **83 ms** | 1.04 s/img | **the DPU** |

The two land at almost identical end-to-end rates for completely opposite reasons.

Consequences worth acting on:

- **For the real model, the DPU is 92% of frame time.** Optimising the NumPy decode or NMS
  would buy at most ~8%. The 50-subgraph fragmentation is what costs the time, exactly as
  [[DPU Subgraph Fragmentation]] predicted — so collapsing it to one subgraph (the LeakyReLU
  finetune) is where a ~50x speedup lives.
- Benchmark FPS from `--benchmark` is **DPU+preprocess only**. For the accurate build that is
  now a good proxy for end-to-end; for the broken build it was ~60x optimistic.
- A demo threshold (`--conf-thres 0.25`) is cheaper than the mAP threshold, but the saving is
  small once the model is behaving.
- Filling all six batch slots helped the single-subgraph build (26.5 → 56.3 FPS). It does far
  less for the fragmented build, where per-subgraph round trips dominate.

---

Back to [[Code Map]] | [[Home]]
