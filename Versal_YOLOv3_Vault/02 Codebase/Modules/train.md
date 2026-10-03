---
tags: [code, module]
module: train
path: train.py
loc: 1477
---

# `train`

> [!info] Source
> `yolov3_test/train.py` - 1477 lines

**Purpose:** Train a YOLOv3 model on a custom dataset. Models and datasets download automatically from the Ultralytics release.

## Imports (internal)

- [[models.experimental]]
- [[models.yolo]]
- [[utils.autoanchor]]
- [[utils.autobatch]]
- [[utils.callbacks]]
- [[utils.dataloaders]]
- [[utils.downloads]]
- [[utils.general]]
- [[utils.loggers]]
- [[utils.loggers.comet.comet_utils]]
- [[utils.loss]]
- [[utils.metrics]]
- [[utils.plots]]
- [[utils.torch_utils]]
- [[val]]

## Imported by

- [[utils.loggers.comet.hpo]]

## Third-party / stdlib

`argparse`, `comet_ml`, `copy`, `datetime`, `math`, `numpy`, `os`, `pathlib`, `random`, `subprocess`, `sys`, `time`, `torch`, `tqdm`, `ultralytics`, `yaml`

## Top-level functions

`train()`, `parse_opt()`, `main()`, `run()`

## Local neighbourhood

```mermaid
graph LR
  subgraph (root)["(root)"]
    train["train"]
    val["val"]
  end
  subgraph models["models"]
    models_experimental["models.experimental"]
    models_yolo["models.yolo"]
  end
  subgraph utils["utils"]
    utils_autoanchor["utils.autoanchor"]
    utils_autobatch["utils.autobatch"]
    utils_callbacks["utils.callbacks"]
    utils_dataloaders["utils.dataloaders"]
    utils_downloads["utils.downloads"]
    utils_general["utils.general"]
    utils_loggers["utils.loggers"]
    utils_loggers_comet_comet_utils["utils.loggers.comet.comet_utils"]
    utils_loggers_comet_hpo["utils.loggers.comet.hpo"]
    utils_loss["utils.loss"]
    utils_metrics["utils.metrics"]
    utils_plots["utils.plots"]
    utils_torch_utils["utils.torch_utils"]
  end
  train --> models_experimental
  train --> models_yolo
  train --> utils_autoanchor
  train --> utils_autobatch
  train --> utils_callbacks
  train --> utils_dataloaders
  train --> utils_downloads
  train --> utils_general
  train --> utils_loggers
  train --> utils_loggers_comet_comet_utils
  train --> utils_loss
  train --> utils_metrics
  train --> utils_plots
  train --> utils_torch_utils
  train --> val
  utils_loggers_comet_hpo --> train
```

---

Back to [[Code Map]] | [[Home]]
