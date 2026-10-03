---
tags: [code, module]
module: utils.segment.loss
path: utils/segment/loss.py
loc: 202
---

# `utils.segment.loss`

> [!info] Source
> `yolov3_test/utils/segment/loss.py` - 202 lines

## Imports (internal)

- [[utils.general]]
- [[utils.loss]]
- [[utils.metrics]]
- [[utils.segment.general]]
- [[utils.torch_utils]]

## Imported by

_Nothing imports this (entry point or unused)._

## Third-party / stdlib

`torch`

## Classes

- `ComputeLoss`

## Local neighbourhood

```mermaid
graph LR
  subgraph utils["utils"]
    utils_general["utils.general"]
    utils_loss["utils.loss"]
    utils_metrics["utils.metrics"]
    utils_segment_general["utils.segment.general"]
    utils_segment_loss["utils.segment.loss"]
    utils_torch_utils["utils.torch_utils"]
  end
  utils_segment_loss --> utils_general
  utils_segment_loss --> utils_loss
  utils_segment_loss --> utils_metrics
  utils_segment_loss --> utils_segment_general
  utils_segment_loss --> utils_torch_utils
```

---

Back to [[Code Map]] | [[Home]]
