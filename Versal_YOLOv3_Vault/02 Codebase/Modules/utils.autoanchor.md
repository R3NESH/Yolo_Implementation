---
tags: [code, module]
module: utils.autoanchor
path: utils/autoanchor.py
loc: 178
---

# `utils.autoanchor`

> [!info] Source
> `yolov3_test/utils/autoanchor.py` - 178 lines

**Purpose:** AutoAnchor utils.

## Imports (internal)

- [[utils]]
- [[utils.dataloaders]]
- [[utils.general]]

## Imported by

- [[models.yolo]]
- [[train]]

## Third-party / stdlib

`numpy`, `random`, `scipy`, `torch`, `tqdm`, `yaml`

## Top-level functions

`check_anchor_order()`, `check_anchors()`, `kmean_anchors()`

## Local neighbourhood

```mermaid
graph LR
  subgraph (root)["(root)"]
    train["train"]
    utils["utils"]
  end
  subgraph models["models"]
    models_yolo["models.yolo"]
  end
  subgraph utils["utils"]
    utils_autoanchor["utils.autoanchor"]
    utils_dataloaders["utils.dataloaders"]
    utils_general["utils.general"]
  end
  models_yolo --> utils_autoanchor
  train --> utils_autoanchor
  utils_autoanchor --> utils
  utils_autoanchor --> utils_dataloaders
  utils_autoanchor --> utils_general
```

---

Back to [[Code Map]] | [[Home]]
