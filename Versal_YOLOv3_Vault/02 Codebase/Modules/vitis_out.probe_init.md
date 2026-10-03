---
tags: [code, module]
module: vitis_out.probe_init
path: vitis_out/probe_init.py
loc: 30
---

# `vitis_out.probe_init`

> [!info] Source
> `yolov3_test/vitis_out/probe_init.py` - 30 lines

> [!warning] Throwaway debug probe from the segfault hunt.

**Purpose:** Scratch probe: find which line of YOLOv3DPUWrapper.__init__ segfaults.

## Imports (internal)

- [[export_dpu_wrapper_AB3]]

## Imported by

_Nothing imports this (entry point or unused)._

## Third-party / stdlib

`sys`, `torch`

## Local neighbourhood

```mermaid
graph LR
  subgraph (root)["(root)"]
    export_dpu_wrapper_AB3["export_dpu_wrapper_AB3"]
  end
  subgraph vitis_out["vitis_out"]
    vitis_out_probe_init["vitis_out.probe_init"]
  end
  vitis_out_probe_init --> export_dpu_wrapper_AB3
```

---

Back to [[Code Map]] | [[Home]]
