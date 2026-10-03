---
tags: [code, module]
module: export_dpu_wrapper_AB3
path: export_dpu_wrapper_AB3.py
loc: 179
---

# `export_dpu_wrapper_AB3`

> [!info] Source
> `yolov3_test/export_dpu_wrapper_AB3.py` - 179 lines

## Imports (internal)

_None - leaf module._

## Imported by

- [[dpu_silu_experiment]]
- [[quantize_vitis_AB4]]
- [[vitis_out.probe_init]]

## Third-party / stdlib

`np2_pickle_compat`, `os`, `torch`

## Classes

- `YOLOv3DPUWrapper(nn.Module)`

## Local neighbourhood

```mermaid
graph LR
  subgraph (root)["(root)"]
    dpu_silu_experiment["dpu_silu_experiment"]
    export_dpu_wrapper_AB3["export_dpu_wrapper_AB3"]
    quantize_vitis_AB4["quantize_vitis_AB4"]
  end
  subgraph vitis_out["vitis_out"]
    vitis_out_probe_init["vitis_out.probe_init"]
  end
  dpu_silu_experiment --> export_dpu_wrapper_AB3
  quantize_vitis_AB4 --> export_dpu_wrapper_AB3
  vitis_out_probe_init --> export_dpu_wrapper_AB3
```

---

Back to [[Code Map]] | [[Home]]
