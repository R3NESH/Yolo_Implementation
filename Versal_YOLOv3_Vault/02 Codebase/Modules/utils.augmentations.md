---
tags: [code, module]
module: utils.augmentations
path: utils/augmentations.py
loc: 419
---

# `utils.augmentations`

> [!info] Source
> `yolov3_test/utils/augmentations.py` - 419 lines

**Purpose:** Image augmentation functions.

## Imports (internal)

- [[utils.general]]
- [[utils.metrics]]

## Imported by

- [[utils.dataloaders]]
- [[utils.plots]]
- [[utils.segment.augmentations]]
- [[utils.segment.dataloaders]]

## Third-party / stdlib

`albumentations`, `cv2`, `math`, `numpy`, `random`, `torch`, `torchvision`

## Classes

- `Albumentations`
- `LetterBox`
- `CenterCrop`
- `ToTensor`

## Top-level functions

`normalize()`, `denormalize()`, `augment_hsv()`, `hist_equalize()`, `replicate()`, `letterbox()`, `random_perspective()`, `copy_paste()`, `cutout()`, `mixup()`, `box_candidates()`, `classify_albumentations()`, `classify_transforms()`

## Local neighbourhood

```mermaid
graph LR
  subgraph utils["utils"]
    utils_augmentations["utils.augmentations"]
    utils_dataloaders["utils.dataloaders"]
    utils_general["utils.general"]
    utils_metrics["utils.metrics"]
    utils_plots["utils.plots"]
    utils_segment_augmentations["utils.segment.augmentations"]
    utils_segment_dataloaders["utils.segment.dataloaders"]
  end
  utils_augmentations --> utils_general
  utils_augmentations --> utils_metrics
  utils_dataloaders --> utils_augmentations
  utils_plots --> utils_augmentations
  utils_segment_augmentations --> utils_augmentations
  utils_segment_dataloaders --> utils_augmentations
```

---

Back to [[Code Map]] | [[Home]]
