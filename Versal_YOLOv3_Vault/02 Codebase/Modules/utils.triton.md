---
tags: [code, module]
module: utils.triton
path: utils/triton.py
loc: 93
---

# `utils.triton`

> [!info] Source
> `yolov3_test/utils/triton.py` - 93 lines

**Purpose:** Utils to interact with the Triton Inference Server.

## Imports (internal)

_None - leaf module._

## Imported by

- [[models.common]]
- [[models.common_orig]]

## Third-party / stdlib

`torch`, `tritonclient`, `typing`, `urllib`

## Classes

- `TritonRemoteModel`

## Local neighbourhood

```mermaid
graph LR
  subgraph models["models"]
    models_common["models.common"]
    models_common_orig["models.common_orig"]
  end
  subgraph utils["utils"]
    utils_triton["utils.triton"]
  end
  models_common --> utils_triton
  models_common_orig --> utils_triton
```

---

Back to [[Code Map]] | [[Home]]
