---
tags: [code, module]
module: utils.dataloaders
path: utils/dataloaders.py
loc: 1330
---

# `utils.dataloaders`

> [!info] Source
> `yolov3_test/utils/dataloaders.py` - 1330 lines

**Purpose:** Dataloaders and dataset utils.

## Imports (internal)

- [[utils.augmentations]]
- [[utils.general]]
- [[utils.torch_utils]]

## Imported by

- [[export]]
- [[models.common]]
- [[models.common_orig]]
- [[train]]
- [[utils.autoanchor]]
- [[utils.loggers.comet]]
- [[utils.segment.dataloaders]]
- [[val]]

## Third-party / stdlib

`PIL`, `contextlib`, `glob`, `hashlib`, `itertools`, `json`, `math`, `mss`, `multiprocessing`, `numpy`, `os`, `pafy`, `pathlib`, `psutil`, `random`, `shutil`, `threading`, `time`, `torch`, `torchvision`, `tqdm`, `urllib`, `yaml`

## Classes

- `InfiniteDataLoader(dataloader.DataLoader)`
- `_RepeatSampler`
- `LoadScreenshots`
- `LoadImages`
- `LoadStreams`
- `LoadImagesAndLabels(Dataset)`
- `HUBDatasetStats`
- `ClassificationDataset(torchvision.datasets.ImageFolder)`

## Top-level functions

`get_hash()`, `exif_size()`, `exif_transpose()`, `seed_worker()`, `create_dataloader()`, `img2label_paths()`, `flatten_recursive()`, `extract_boxes()`, `autosplit()`, `verify_image_label()`, `create_classification_dataloader()`

## Local neighbourhood

```mermaid
graph LR
  subgraph (root)["(root)"]
    export["export"]
    train["train"]
    val["val"]
  end
  subgraph models["models"]
    models_common["models.common"]
    models_common_orig["models.common_orig"]
  end
  subgraph utils["utils"]
    utils_augmentations["utils.augmentations"]
    utils_autoanchor["utils.autoanchor"]
    utils_dataloaders["utils.dataloaders"]
    utils_general["utils.general"]
    utils_loggers_comet["utils.loggers.comet"]
    utils_segment_dataloaders["utils.segment.dataloaders"]
    utils_torch_utils["utils.torch_utils"]
  end
  export --> utils_dataloaders
  models_common --> utils_dataloaders
  models_common_orig --> utils_dataloaders
  train --> utils_dataloaders
  utils_autoanchor --> utils_dataloaders
  utils_dataloaders --> utils_augmentations
  utils_dataloaders --> utils_general
  utils_dataloaders --> utils_torch_utils
  utils_loggers_comet --> utils_dataloaders
  utils_segment_dataloaders --> utils_dataloaders
  val --> utils_dataloaders
```

---

Back to [[Code Map]] | [[Home]]
