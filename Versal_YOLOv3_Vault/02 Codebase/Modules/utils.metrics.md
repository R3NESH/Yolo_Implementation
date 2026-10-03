---
tags: [code, module]
module: utils.metrics
path: utils/metrics.py
loc: 376
---

# `utils.metrics`

> [!info] Source
> `yolov3_test/utils/metrics.py` - 376 lines

**Purpose:** Model validation metrics.

## Imports (internal)

- [[utils]]

## Imported by

- [[train]]
- [[utils.augmentations]]
- [[utils.general]]
- [[utils.loggers.comet]]
- [[utils.loss]]
- [[utils.plots]]
- [[utils.segment.loss]]
- [[utils.segment.metrics]]
- [[val]]

## Third-party / stdlib

`math`, `matplotlib`, `numpy`, `pathlib`, `seaborn`, `torch`, `warnings`

## Classes

- `ConfusionMatrix`

## Top-level functions

`fitness()`, `smooth()`, `ap_per_class()`, `compute_ap()`, `bbox_iou()`, `box_iou()`, `bbox_ioa()`, `wh_iou()`, `plot_pr_curve()`, `plot_mc_curve()`

## Local neighbourhood

```mermaid
graph LR
  subgraph (root)["(root)"]
    train["train"]
    utils["utils"]
    val["val"]
  end
  subgraph utils["utils"]
    utils_augmentations["utils.augmentations"]
    utils_general["utils.general"]
    utils_loggers_comet["utils.loggers.comet"]
    utils_loss["utils.loss"]
    utils_metrics["utils.metrics"]
    utils_plots["utils.plots"]
    utils_segment_loss["utils.segment.loss"]
    utils_segment_metrics["utils.segment.metrics"]
  end
  train --> utils_metrics
  utils_augmentations --> utils_metrics
  utils_general --> utils_metrics
  utils_loggers_comet --> utils_metrics
  utils_loss --> utils_metrics
  utils_metrics --> utils
  utils_plots --> utils_metrics
  utils_segment_loss --> utils_metrics
  utils_segment_metrics --> utils_metrics
  val --> utils_metrics
```

---

Back to [[Code Map]] | [[Home]]
