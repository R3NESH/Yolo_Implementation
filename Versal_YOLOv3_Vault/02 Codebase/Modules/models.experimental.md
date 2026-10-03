---
tags: [code, module]
module: models.experimental
path: models/experimental.py
loc: 134
---

# `models.experimental`

> [!info] Source
> `yolov3_test/models/experimental.py` - 134 lines

**Purpose:** Experimental modules.

## Imports (internal)

- [[models.yolo]]
- [[utils.downloads]]

## Imported by

- [[export]]
- [[models.common]]
- [[models.common_orig]]
- [[models.tf]]
- [[models.yolo]]
- [[train]]

## Third-party / stdlib

`math`, `numpy`, `torch`, `ultralytics`

## Classes

- `Sum(nn.Module)`
- `MixConv2d(nn.Module)`
- `Ensemble(nn.ModuleList)`

## Top-level functions

`attempt_load()`

## Local neighbourhood

```mermaid
graph LR
  subgraph (root)["(root)"]
    export["export"]
    train["train"]
  end
  subgraph models["models"]
    models_common["models.common"]
    models_common_orig["models.common_orig"]
    models_experimental["models.experimental"]
    models_tf["models.tf"]
    models_yolo["models.yolo"]
  end
  subgraph utils["utils"]
    utils_downloads["utils.downloads"]
  end
  export --> models_experimental
  models_common --> models_experimental
  models_common_orig --> models_experimental
  models_experimental --> models_yolo
  models_experimental --> utils_downloads
  models_tf --> models_experimental
  models_yolo --> models_experimental
  train --> models_experimental
```

---

Back to [[Code Map]] | [[Home]]
