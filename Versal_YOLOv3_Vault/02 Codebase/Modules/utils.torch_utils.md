---
tags: [code, module]
module: utils.torch_utils
path: utils/torch_utils.py
loc: 485
---

# `utils.torch_utils`

> [!info] Source
> `yolov3_test/utils/torch_utils.py` - 485 lines

**Purpose:** PyTorch utils.

## Imports (internal)

- [[models.common]]
- [[utils.general]]

## Imported by

- [[export]]
- [[models.common]]
- [[models.common_orig]]
- [[models.yolo]]
- [[train]]
- [[utils]]
- [[utils.autobatch]]
- [[utils.dataloaders]]
- [[utils.loggers]]
- [[utils.loggers.comet.hpo]]
- [[utils.loss]]
- [[utils.segment.dataloaders]]
- [[utils.segment.loss]]
- [[val]]

## Third-party / stdlib

`contextlib`, `copy`, `math`, `os`, `pathlib`, `platform`, `subprocess`, `thop`, `time`, `torch`, `warnings`

## Classes

- `EarlyStopping`
- `ModelEMA`

## Top-level functions

`smart_inference_mode()`, `smartCrossEntropyLoss()`, `smart_DDP()`, `reshape_classifier_output()`, `torch_distributed_zero_first()`, `device_count()`, `select_device()`, `time_sync()`, `profile()`, `is_parallel()`, `de_parallel()`, `initialize_weights()`, `find_modules()`, `sparsity()`, `prune()`, `fuse_conv_and_bn()`, `model_info()`, `scale_img()`, `copy_attr()`, `smart_optimizer()`, `smart_hub_load()`, `smart_resume()`

## Local neighbourhood

```mermaid
graph LR
  subgraph (root)["(root)"]
    export["export"]
    train["train"]
    utils["utils"]
    val["val"]
  end
  subgraph models["models"]
    models_common["models.common"]
    models_common_orig["models.common_orig"]
    models_yolo["models.yolo"]
  end
  subgraph utils["utils"]
    utils_autobatch["utils.autobatch"]
    utils_dataloaders["utils.dataloaders"]
    utils_general["utils.general"]
    utils_loggers["utils.loggers"]
    utils_loggers_comet_hpo["utils.loggers.comet.hpo"]
    utils_loss["utils.loss"]
    utils_segment_dataloaders["utils.segment.dataloaders"]
    utils_segment_loss["utils.segment.loss"]
    utils_torch_utils["utils.torch_utils"]
  end
  export --> utils_torch_utils
  models_common --> utils_torch_utils
  models_common_orig --> utils_torch_utils
  models_yolo --> utils_torch_utils
  train --> utils_torch_utils
  utils --> utils_torch_utils
  utils_autobatch --> utils_torch_utils
  utils_dataloaders --> utils_torch_utils
  utils_loggers --> utils_torch_utils
  utils_loggers_comet_hpo --> utils_torch_utils
  utils_loss --> utils_torch_utils
  utils_segment_dataloaders --> utils_torch_utils
  utils_segment_loss --> utils_torch_utils
  utils_torch_utils --> models_common
  utils_torch_utils --> utils_general
  val --> utils_torch_utils
```

---

Back to [[Code Map]] | [[Home]]
