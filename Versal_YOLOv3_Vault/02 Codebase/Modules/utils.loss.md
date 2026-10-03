---
tags: [code, module]
module: utils.loss
path: utils/loss.py
loc: 257
---

# `utils.loss`

> [!info] Source
> `yolov3_test/utils/loss.py` - 257 lines

**Purpose:** Loss functions.

## Imports (internal)

- [[utils.metrics]]
- [[utils.torch_utils]]

## Imported by

- [[train]]
- [[utils.segment.loss]]

## Third-party / stdlib

`torch`

## Classes

- `BCEBlurWithLogitsLoss(nn.Module)`
- `FocalLoss(nn.Module)`
- `QFocalLoss(nn.Module)`
- `ComputeLoss`

## Top-level functions

`smooth_BCE()`

## Local neighbourhood

```mermaid
graph LR
  subgraph (root)["(root)"]
    train["train"]
  end
  subgraph utils["utils"]
    utils_loss["utils.loss"]
    utils_metrics["utils.metrics"]
    utils_segment_loss["utils.segment.loss"]
    utils_torch_utils["utils.torch_utils"]
  end
  train --> utils_loss
  utils_loss --> utils_metrics
  utils_loss --> utils_torch_utils
  utils_segment_loss --> utils_loss
```

---

Back to [[Code Map]] | [[Home]]
