---
tags: [code, module]
module: dpu_silu_experiment
path: dpu_silu_experiment.py
loc: 118
---

# `dpu_silu_experiment`

> [!info] Source
> `yolov3_test/dpu_silu_experiment.py` - 118 lines

**Purpose:** Experiment: does replacing SiLU with LeakyReLU defragment the DPU subgraph?

## Imports (internal)

- [[export_dpu_wrapper_AB3]]
- [[quantize_vitis_AB4]]

## Imported by

_Nothing imports this (entry point or unused)._

## Third-party / stdlib

`argparse`, `np2_pickle_compat`, `os`, `pytorch_nndct`, `sys`, `torch`

## Top-level functions

`swap_silu()`, `run()`

## Local neighbourhood

```mermaid
graph LR
  subgraph (root)["(root)"]
    dpu_silu_experiment["dpu_silu_experiment"]
    export_dpu_wrapper_AB3["export_dpu_wrapper_AB3"]
    quantize_vitis_AB4["quantize_vitis_AB4"]
  end
  dpu_silu_experiment --> export_dpu_wrapper_AB3
  dpu_silu_experiment --> quantize_vitis_AB4
```

---

Back to [[Code Map]] | [[Home]]
