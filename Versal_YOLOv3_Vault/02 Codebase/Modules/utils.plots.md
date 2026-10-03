---
tags: [code, module]
module: utils.plots
path: utils/plots.py
loc: 486
---

# `utils.plots`

> [!info] Source
> `yolov3_test/utils/plots.py` - 486 lines

**Purpose:** Plotting utils.

## Imports (internal)

- [[utils]]
- [[utils.augmentations]]
- [[utils.general]]
- [[utils.metrics]]

## Imported by

- [[models.yolo]]
- [[train]]
- [[utils.loggers]]
- [[utils.segment.plots]]
- [[val]]

## Third-party / stdlib

`PIL`, `contextlib`, `copy`, `cv2`, `math`, `matplotlib`, `numpy`, `os`, `pandas`, `pathlib`, `scipy`, `seaborn`, `torch`, `ultralytics`

## Classes

- `Colors`

## Top-level functions

`feature_visualization()`, `hist2d()`, `butter_lowpass_filtfilt()`, `output_to_target()`, `plot_images()`, `plot_lr_scheduler()`, `plot_val_txt()`, `plot_targets_txt()`, `plot_val_study()`, `plot_labels()`, `imshow_cls()`, `plot_evolve()`, `plot_results()`, `profile_idetection()`, `save_one_box()`

## Local neighbourhood

```mermaid
graph LR
  subgraph (root)["(root)"]
    train["train"]
    utils["utils"]
    val["val"]
  end
  subgraph models["models"]
    models_yolo["models.yolo"]
  end
  subgraph utils["utils"]
    utils_augmentations["utils.augmentations"]
    utils_general["utils.general"]
    utils_loggers["utils.loggers"]
    utils_metrics["utils.metrics"]
    utils_plots["utils.plots"]
    utils_segment_plots["utils.segment.plots"]
  end
  models_yolo --> utils_plots
  train --> utils_plots
  utils_loggers --> utils_plots
  utils_plots --> utils
  utils_plots --> utils_augmentations
  utils_plots --> utils_general
  utils_plots --> utils_metrics
  utils_segment_plots --> utils_plots
  val --> utils_plots
```

---

Back to [[Code Map]] | [[Home]]
