---
tags: [code, module]
module: utils.general
path: utils/general.py
loc: 1289
---

# `utils.general`

> [!info] Source
> `yolov3_test/utils/general.py` - 1289 lines

**Purpose:** General utils.

## Imports (internal)

- [[models.common]]
- [[utils]]
- [[utils.downloads]]
- [[utils.metrics]]

## Imported by

- [[export]]
- [[models.common]]
- [[models.common_orig]]
- [[models.tf]]
- [[models.yolo]]
- [[train]]
- [[utils]]
- [[utils.augmentations]]
- [[utils.autoanchor]]
- [[utils.autobatch]]
- [[utils.dataloaders]]
- [[utils.downloads]]
- [[utils.loggers]]
- [[utils.loggers.comet]]
- [[utils.loggers.comet.hpo]]
- [[utils.loggers.wandb.wandb_utils]]
- [[utils.plots]]
- [[utils.segment.augmentations]]
- [[utils.segment.dataloaders]]
- [[utils.segment.loss]]
- [[utils.segment.plots]]
- [[utils.torch_utils]]
- [[val]]

## Third-party / stdlib

`IPython`, `contextlib`, `copy`, `cv2`, `datetime`, `git`, `glob`, `inspect`, `itertools`, `logging`, `math`, `multiprocessing`, `numpy`, `os`, `pandas`, `pathlib`, `pkg_resources`, `platform`, `random`, `re`, `signal`, `socket`, `subprocess`, `sys`, `tarfile`, `time`, `torch`, `torchvision`, `typing`, `ultralytics`, `urllib`, `yaml`, `zipfile`

## Classes

- `Profile(contextlib.ContextDecorator)`
- `Timeout(contextlib.ContextDecorator)`
- `WorkingDirectory(contextlib.ContextDecorator)`

## Top-level functions

`is_ascii()`, `is_chinese()`, `is_colab()`, `is_jupyter()`, `is_kaggle()`, `is_docker()`, `is_writeable()`, `set_logging()`, `user_config_dir()`, `methods()`, `print_args()`, `init_seeds()`, `intersect_dicts()`, `get_default_args()`, `get_latest_run()`, `file_age()`, `file_date()`, `file_size()`, `check_online()`, `git_describe()`, `check_git_status()`, `check_git_info()`, `check_python()`, `check_version()`, `check_img_size()`, `check_imshow()`, `check_suffix()`, `check_yaml()`, `check_file()`, `check_font()`, `check_dataset()`, `check_amp()`, `yaml_load()`, `yaml_save()`, `unzip_file()`, `url2file()`, `download()`, `make_divisible()`, `clean_str()`, `one_cycle()`, `colorstr()`, `labels_to_class_weights()`, `labels_to_image_weights()`, `coco80_to_coco91_class()`, `xyxy2xywh()`, `xywh2xyxy()`, `xywhn2xyxy()`, `xyxy2xywhn()`, `xyn2xy()`, `segment2box()`, `segments2boxes()`, `resample_segments()`, `scale_boxes()`, `scale_segments()`, `clip_boxes()`, `clip_segments()`, `non_max_suppression()`, `strip_optimizer()`, `print_mutation()`, `apply_classifier()`, `increment_path()`, `imread()`, `imwrite()`, `imshow()`

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
    models_tf["models.tf"]
    models_yolo["models.yolo"]
  end
  subgraph utils["utils"]
    utils_augmentations["utils.augmentations"]
    utils_autoanchor["utils.autoanchor"]
    utils_autobatch["utils.autobatch"]
    utils_dataloaders["utils.dataloaders"]
    utils_downloads["utils.downloads"]
    utils_general["utils.general"]
    utils_loggers["utils.loggers"]
    utils_loggers_comet["utils.loggers.comet"]
    utils_loggers_comet_hpo["utils.loggers.comet.hpo"]
    utils_loggers_wandb_wandb_utils["utils.loggers.wandb.wandb_utils"]
    utils_metrics["utils.metrics"]
    utils_plots["utils.plots"]
    utils_segment_augmentations["utils.segment.augmentations"]
    utils_segment_dataloaders["utils.segment.dataloaders"]
    utils_segment_loss["utils.segment.loss"]
    utils_segment_plots["utils.segment.plots"]
    utils_torch_utils["utils.torch_utils"]
  end
  export --> utils_general
  models_common --> utils_general
  models_common_orig --> utils_general
  models_tf --> utils_general
  models_yolo --> utils_general
  train --> utils_general
  utils --> utils_general
  utils_augmentations --> utils_general
  utils_autoanchor --> utils_general
  utils_autobatch --> utils_general
  utils_dataloaders --> utils_general
  utils_downloads --> utils_general
  utils_general --> models_common
  utils_general --> utils
  utils_general --> utils_downloads
  utils_general --> utils_metrics
  utils_loggers --> utils_general
  utils_loggers_comet --> utils_general
  utils_loggers_comet_hpo --> utils_general
  utils_loggers_wandb_wandb_utils --> utils_general
  utils_plots --> utils_general
  utils_segment_augmentations --> utils_general
  utils_segment_dataloaders --> utils_general
  utils_segment_loss --> utils_general
  utils_segment_plots --> utils_general
  utils_torch_utils --> utils_general
  val --> utils_general
```

---

Back to [[Code Map]] | [[Home]]
