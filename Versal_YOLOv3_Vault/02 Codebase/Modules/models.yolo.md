---
tags: [code, module]
module: models.yolo
path: models/yolo.py
loc: 460
---

# `models.yolo`

> [!info] Source
> `yolov3_test/models/yolo.py` - 460 lines

**Purpose:** YOLO-specific modules.

## Imports (internal)

- [[models.common]]
- [[models.experimental]]
- [[utils.autoanchor]]
- [[utils.general]]
- [[utils.plots]]
- [[utils.torch_utils]]

## Imported by

- [[export]]
- [[models.experimental]]
- [[models.tf]]
- [[train]]

## Third-party / stdlib

`argparse`, `copy`, `os`, `pathlib`, `platform`, `sys`, `thop`, `yaml`

## Classes

- `Detect(nn.Module)`
- `Segment(Detect)`
- `BaseModel(nn.Module)`
- `DetectionModel(BaseModel)`
- `SegmentationModel(DetectionModel)`
- `ClassificationModel(BaseModel)`

## Top-level functions

`parse_model()`

## Local neighbourhood

```mermaid
graph LR
  subgraph (root)["(root)"]
    export["export"]
    train["train"]
  end
  subgraph models["models"]
    models_common["models.common"]
    models_experimental["models.experimental"]
    models_tf["models.tf"]
    models_yolo["models.yolo"]
  end
  subgraph utils["utils"]
    utils_autoanchor["utils.autoanchor"]
    utils_general["utils.general"]
    utils_plots["utils.plots"]
    utils_torch_utils["utils.torch_utils"]
  end
  export --> models_yolo
  models_experimental --> models_yolo
  models_tf --> models_yolo
  models_yolo --> models_common
  models_yolo --> models_experimental
  models_yolo --> utils_autoanchor
  models_yolo --> utils_general
  models_yolo --> utils_plots
  models_yolo --> utils_torch_utils
  train --> models_yolo
```

---

Back to [[Code Map]] | [[Home]]
