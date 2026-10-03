---
tags: [code, module]
module: val
path: val.py
loc: 631
---

# `val`

> [!info] Source
> `yolov3_test/val.py` - 631 lines

**Purpose:** Validate a trained YOLOv3 detection model on a detection dataset.

## Imports (internal)

- [[models.common]]
- [[utils.callbacks]]
- [[utils.dataloaders]]
- [[utils.general]]
- [[utils.metrics]]
- [[utils.plots]]
- [[utils.torch_utils]]

## Imported by

- [[train]]

## Third-party / stdlib

`argparse`, `json`, `numpy`, `os`, `pathlib`, `pycocotools`, `subprocess`, `sys`, `torch`, `tqdm`

## Top-level functions

`save_one_txt()`, `save_one_json()`, `process_batch()`, `run()`, `parse_opt()`, `main()`

## Local neighbourhood

```mermaid
graph LR
  subgraph (root)["(root)"]
    train["train"]
    val["val"]
  end
  subgraph models["models"]
    models_common["models.common"]
  end
  subgraph utils["utils"]
    utils_callbacks["utils.callbacks"]
    utils_dataloaders["utils.dataloaders"]
    utils_general["utils.general"]
    utils_metrics["utils.metrics"]
    utils_plots["utils.plots"]
    utils_torch_utils["utils.torch_utils"]
  end
  train --> val
  val --> models_common
  val --> utils_callbacks
  val --> utils_dataloaders
  val --> utils_general
  val --> utils_metrics
  val --> utils_plots
  val --> utils_torch_utils
```

---

Back to [[Code Map]] | [[Home]]
