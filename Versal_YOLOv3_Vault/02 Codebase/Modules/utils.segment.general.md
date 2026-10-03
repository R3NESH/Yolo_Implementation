---
tags: [code, module]
module: utils.segment.general
path: utils/segment/general.py
loc: 161
---

# `utils.segment.general`

> [!info] Source
> `yolov3_test/utils/segment/general.py` - 161 lines

## Imports (internal)

_None - leaf module._

## Imported by

- [[utils.segment.loss]]

## Third-party / stdlib

`cv2`, `numpy`, `torch`

## Top-level functions

`crop_mask()`, `process_mask_upsample()`, `process_mask()`, `process_mask_native()`, `scale_image()`, `mask_iou()`, `masks_iou()`, `masks2segments()`

## Local neighbourhood

```mermaid
graph LR
  subgraph utils["utils"]
    utils_segment_general["utils.segment.general"]
    utils_segment_loss["utils.segment.loss"]
  end
  utils_segment_loss --> utils_segment_general
```

---

Back to [[Code Map]] | [[Home]]
