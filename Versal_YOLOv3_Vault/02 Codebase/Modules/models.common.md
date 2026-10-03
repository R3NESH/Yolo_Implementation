---
tags: [code, module]
module: models.common
path: models/common.py
loc: 1122
---

# `models.common`

> [!info] Source
> `yolov3_test/models/common.py` - 1122 lines

**Purpose:** Common modules.

## Imports (internal)

- [[export]]
- [[models.experimental]]
- [[utils]]
- [[utils.dataloaders]]
- [[utils.downloads]]
- [[utils.general]]
- [[utils.torch_utils]]
- [[utils.triton]]

## Imported by

- [[models.tf]]
- [[models.yolo]]
- [[utils.general]]
- [[utils.torch_utils]]
- [[val]]

## Third-party / stdlib

`IPython`, `PIL`, `ast`, `collections`, `contextlib`, `copy`, `coremltools`, `cv2`, `json`, `math`, `numpy`, `onnxruntime`, `openvino`, `paddle`, `pandas`, `pathlib`, `platform`, `requests`, `tensorflow`, `tensorrt`, `tflite_runtime`, `torch`, `ultralytics`, `urllib`, `warnings`, `zipfile`

## Classes

- `Conv(nn.Module)`
- `DWConv(Conv)`
- `DWConvTranspose2d(nn.ConvTranspose2d)`
- `TransformerLayer(nn.Module)`
- `TransformerBlock(nn.Module)`
- `Bottleneck(nn.Module)`
- `Bottleneck_merged(nn.Module)`
- `BottleneckCSP(nn.Module)`
- `CrossConv(nn.Module)`
- `C3(nn.Module)`
- `C3x(C3)`
- `C3TR(C3)`
- `C3SPP(C3)`
- `C3Ghost(C3)`
- `SPP(nn.Module)`
- `SPPF(nn.Module)`
- `Focus(nn.Module)`
- `GhostConv(nn.Module)`
- `GhostBottleneck(nn.Module)`
- `Contract(nn.Module)`
- `Expand(nn.Module)`
- `Concat(nn.Module)`
- `DetectMultiBackend(nn.Module)`
- `AutoShape(nn.Module)`
- `Detections`
- `Proto(nn.Module)`
- `Classify(nn.Module)`

## Top-level functions

`autopad()`

## Local neighbourhood

```mermaid
graph LR
  subgraph (root)["(root)"]
    export["export"]
    utils["utils"]
    val["val"]
  end
  subgraph models["models"]
    models_common["models.common"]
    models_experimental["models.experimental"]
    models_tf["models.tf"]
    models_yolo["models.yolo"]
  end
  subgraph utils["utils"]
    utils_dataloaders["utils.dataloaders"]
    utils_downloads["utils.downloads"]
    utils_general["utils.general"]
    utils_torch_utils["utils.torch_utils"]
    utils_triton["utils.triton"]
  end
  models_common --> export
  models_common --> models_experimental
  models_common --> utils
  models_common --> utils_dataloaders
  models_common --> utils_downloads
  models_common --> utils_general
  models_common --> utils_torch_utils
  models_common --> utils_triton
  models_tf --> models_common
  models_yolo --> models_common
  utils_general --> models_common
  utils_torch_utils --> models_common
  val --> models_common
```

---

Back to [[Code Map]] | [[Home]]
