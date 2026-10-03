---
tags: [finding, experiment, dpu, silu, resolved]
date: 2026-09-11
result: confirmed
---

# SiLU to LeakyReLU Experiment

> [!success] Result: confirmed, and the fix is total
> Replacing all 49 SiLU activations with `LeakyReLU(0.1015625)` collapses the compiled model from
> **52 DPU subgraphs to 1**, and from **98 CPU-assigned ops to zero**. SiLU was the sole cause of
> the fragmentation described in [[DPU Subgraph Fragmentation]].

## Why run it

[[DPU Subgraph Fragmentation]] established that SiLU fragments the graph, and that the likely fix
was finetuning with LeakyReLU. But "likely" is expensive to act on — a finetune costs GPU hours.
This experiment separates the **mapping** question from the **accuracy** question:

> If we swap the activation and nothing else, does the graph actually defragment?

If yes, a finetune is justified and its target is known. If no, something else is also wrong and
the finetune would have been wasted.

## Method

`yolov3_test/dpu_silu_experiment.py` — swaps activations **in memory**, so no second 264 MB
checkpoint is written to disk (the machine has no room for one, see
[[Disk and System Constraints]]):

```python
def swap_silu(module, alpha=DPU_LEAKY_ALPHA):
    """Recursively replaces every nn.SiLU with nn.LeakyReLU(alpha). Returns the count."""
    swapped = 0
    for name, child in module.named_children():
        if isinstance(child, nn.SiLU):
            setattr(module, name, nn.LeakyReLU(negative_slope=alpha, inplace=True))
            swapped += 1
        else:
            swapped += swap_silu(child, alpha)
    return swapped
```

Then the identical quantize → export → compile pipeline as the baseline, into separate output
directories so both builds coexist for comparison.

> [!warning] The alpha value is not arbitrary
> `DPU_LEAKY_ALPHA = 0.1015625` = 26/256. The DPU implements LeakyReLU **only** at this slope.
> A plain `0.1` — which is what Darknet YOLOv3 actually used — would fall back to the CPU and
> reintroduce the very fragmentation this is meant to fix. Anyone doing the real finetune must
> train with 0.1015625, not 0.1.

Commands run:

```bash
./vitis_run.sh python -u dpu_silu_experiment.py --quant_mode calib
./vitis_run.sh python -u dpu_silu_experiment.py --quant_mode test
./vitis_run.sh vai_c_xir \
    -x quantize_result_lrelu/YOLOv3DPUWrapper_int.xmodel \
    -a /opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json \
    -o compiled_lrelu -n yolov3_vck190_lrelu
```

## Results

| Metric | Baseline (SiLU) | LeakyReLU(0.1015625) | Change |
| --- | --- | --- | --- |
| **DPU subgraphs** | 52 | **1** | 52× fewer |
| Total device subgraphs | 105 | 5 | |
| CPU-assigned ops | 98 (`transpose`) | **0** | eliminated |
| Quantizer float-op warnings | 1 (`aten::silu_`) | **0** | eliminated |
| XIR "not defined in XIR" warnings | 49 | **0** | eliminated |
| Compiled xmodel size | 35.4 MB | 34.9 MB | ~equal |

Baseline log:
```
[UNILOG][INFO] Total device subgraph number 105, DPU subgraph number 52
```

LeakyReLU log:
```
[UNILOG][INFO] Total device subgraph number 5, DPU subgraph number 1
```

Nothing was assigned to the CPU at all in the LeakyReLU build — the `has been assigned to CPU`
warnings that appeared 98 times in the baseline appear **zero** times. The whole network is one
DPU kernel.

> [!info] Why "5 total" and not "1 total"
> The remaining 4 non-DPU subgraphs are the input/output plumbing the runtime always adds
> (data in, data out). They carry no computation. The single **DPU** subgraph is the number that
> matters for throughput.

## What this does and does not prove

**Proven:** SiLU alone caused the fragmentation. The model's topology, the `Bottleneck_merged`
blocks, the concat/upsample routing, and the stripped Detect head are all fully DPU-mappable.
The toolchain and the wrapper are correct.

**Not proven — and important:** this build's **accuracy is expected to be bad**. Swapping an
activation function without retraining changes the function the network computes; the
convolution weights were fit for SiLU. This xmodel is a mapping proof, **not a shippable model**.

> [!danger] Measured 2026-09-11: "bad" turned out to mean *89% of mAP@0.5 gone*
> Full COCO val2017 on the board gives **mAP@0.5 = 0.0591** against a float baseline of
> **0.5275** — 11% retained. Every one of the 5000 images saturated the 300-detection cap,
> which is the signature of a network emitting noise. Details in
> [[Board mAP - LeakyReLU Without Finetune]].

> [!danger] Do not deploy `compiled_lrelu/yolov3_vck190_lrelu.xmodel` as a product
> It will run fast and detect poorly. It exists to prove the graph maps.

## What to do next

The path is now unambiguous:

1. **Finetune** `exp2/best.pt` with `nn.LeakyReLU(0.1015625)` substituted throughout, on
   `data/train2017_yolo.yaml` at 416.

   > [!warning] Revised: this is a big job, not a touch-up
   > The original reasoning here — that the weights are already trained, so a finetune should
   > recover most of the accuracy — assumed the starting point was merely degraded. It is not:
   > the swapped model retains **11%** of baseline mAP@0.5
   > ([[Board mAP - LeakyReLU Without Finetune]]). Climbing 0.059 → ~0.50 is closer to a retrain
   > than a finetune. Budget real GPU time on real data; a few epochs on a subset will not
   > answer whether this path works.
2. Re-run this exact pipeline on the finetuned checkpoint. Expect 1 DPU subgraph again.
3. Validate mAP against the SiLU baseline (`val.py`) to quantify what the activation change cost.
4. Only then proceed to board bring-up ([[Implementation Plan]] steps 7-9).

Because a retrain is happening anyway, that is also the moment to fix the preprocessing mismatch
(letterbox vs plain resize) and to revisit whether `Bottleneck_merged` is earning its accuracy
cost — see [[Subsystem - Models]].

---

Back to [[Code Map]] | [[Home]]
