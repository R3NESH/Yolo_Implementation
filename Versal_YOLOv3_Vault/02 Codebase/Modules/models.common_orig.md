---
tags: [code, module]
module: models.common_orig
path: models/common_orig.py
loc: 1078
---

# `models.common_orig`

> [!info] Source
> `yolov3_test/models/common_orig.py` - 1078 lines

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

_Nothing imports this (entry point or unused)._

## Third-party / stdlib

`IPython`, `PIL`, `ast`, `collections`, `contextlib`, `copy`, `coremltools`, `cv2`, `json`, `math`, `numpy`, `onnxruntime`, `openvino`, `paddle`, `pandas`, `pathlib`, `platform`, `requests`, `tensorflow`, `tensorrt`, `tflite_runtime`, `torch`, `ultralytics`, `urllib`, `warnings`, `zipfile`

## Classes

- `Conv(nn.Module)`
- `DWConv(Conv)`
- `DWConvTranspose2d(nn.ConvTranspose2d)`
- `TransformerLayer(nn.Module)`
- `TransformerBlock(nn.Module)`
- `Bottleneck(nn.Module)`
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
  end
  subgraph models["models"]
    models_common_orig["models.common_orig"]
    models_experimental["models.experimental"]
  end
  subgraph utils["utils"]
    utils_dataloaders["utils.dataloaders"]
    utils_downloads["utils.downloads"]
    utils_general["utils.general"]
    utils_torch_utils["utils.torch_utils"]
    utils_triton["utils.triton"]
  end
  models_common_orig --> export
  models_common_orig --> models_experimental
  models_common_orig --> utils
  models_common_orig --> utils_dataloaders
  models_common_orig --> utils_downloads
  models_common_orig --> utils_general
  models_common_orig --> utils_torch_utils
  models_common_orig --> utils_triton
```

---

Back to [[Code Map]] | [[Home]]
