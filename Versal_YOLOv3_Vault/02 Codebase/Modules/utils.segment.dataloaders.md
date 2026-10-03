---
tags: [code, module]
module: utils.segment.dataloaders
path: utils/segment/dataloaders.py
loc: 363
---

# `utils.segment.dataloaders`

> [!info] Source
> `yolov3_test/utils/segment/dataloaders.py` - 363 lines

**Purpose:** Dataloaders.

## Imports (internal)

- [[utils.augmentations]]
- [[utils.dataloaders]]
- [[utils.general]]
- [[utils.segment.augmentations]]
- [[utils.torch_utils]]

## Imported by

_Nothing imports this (entry point or unused)._

## Third-party / stdlib

`cv2`, `numpy`, `os`, `random`, `torch`

## Classes

- `LoadImagesAndLabelsAndMasks(LoadImagesAndLabels)`

## Top-level functions

`create_dataloader()`, `polygon2mask()`, `polygons2masks()`, `polygons2masks_overlap()`

## Local neighbourhood

```mermaid
graph LR
  subgraph utils["utils"]
    utils_augmentations["utils.augmentations"]
    utils_dataloaders["utils.dataloaders"]
    utils_general["utils.general"]
    utils_segment_augmentations["utils.segment.augmentations"]
    utils_segment_dataloaders["utils.segment.dataloaders"]
    utils_torch_utils["utils.torch_utils"]
  end
  utils_segment_dataloaders --> utils_augmentations
  utils_segment_dataloaders --> utils_dataloaders
  utils_segment_dataloaders --> utils_general
  utils_segment_dataloaders --> utils_segment_augmentations
  utils_segment_dataloaders --> utils_torch_utils
```

---

Back to [[Code Map]] | [[Home]]
