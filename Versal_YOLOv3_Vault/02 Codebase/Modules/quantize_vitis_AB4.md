---
tags: [code, module]
module: quantize_vitis_AB4
path: quantize_vitis_AB4.py
loc: 370
---

# `quantize_vitis_AB4`

> [!info] Source
> `yolov3_test/quantize_vitis_AB4.py` - 370 lines

## Imports (internal)

- [[export_dpu_wrapper_AB3]]

## Imported by

- [[dpu_silu_experiment]]

## Third-party / stdlib

`PIL`, `argparse`, `np2_pickle_compat`, `os`, `pytorch_nndct`, `sys`, `torch`, `torchvision`

## Top-level functions

`get_calibration_dataloader()`, `run_quantization()`

## Local neighbourhood

```mermaid
graph LR
  subgraph (root)["(root)"]
    dpu_silu_experiment["dpu_silu_experiment"]
    export_dpu_wrapper_AB3["export_dpu_wrapper_AB3"]
    quantize_vitis_AB4["quantize_vitis_AB4"]
  end
  dpu_silu_experiment --> quantize_vitis_AB4
  quantize_vitis_AB4 --> export_dpu_wrapper_AB3
```

---

Back to [[Code Map]] | [[Home]]
