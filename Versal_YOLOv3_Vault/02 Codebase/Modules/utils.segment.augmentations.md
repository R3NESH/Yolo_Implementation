---
tags: [code, module]
module: utils.segment.augmentations
path: utils/segment/augmentations.py
loc: 91
---

# `utils.segment.augmentations`

> [!info] Source
> `yolov3_test/utils/segment/augmentations.py` - 91 lines

**Purpose:** Image augmentation functions.

## Imports (internal)

- [[utils.augmentations]]
- [[utils.general]]

## Imported by

- [[utils.segment.dataloaders]]

## Third-party / stdlib

`cv2`, `math`, `numpy`, `random`

## Top-level functions

`mixup()`, `random_perspective()`

## Local neighbourhood

```mermaid
graph LR
  subgraph utils["utils"]
    utils_augmentations["utils.augmentations"]
    utils_general["utils.general"]
    utils_segment_augmentations["utils.segment.augmentations"]
    utils_segment_dataloaders["utils.segment.dataloaders"]
  end
  utils_segment_augmentations --> utils_augmentations
  utils_segment_augmentations --> utils_general
  utils_segment_dataloaders --> utils_segment_augmentations
```

---

Back to [[Code Map]] | [[Home]]
