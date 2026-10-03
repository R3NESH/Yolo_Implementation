---
tags: [code, module]
module: export
path: export.py
loc: 1575
---

# `export`

> [!info] Source
> `yolov3_test/export.py` - 1575 lines

**Purpose:** Export a YOLOv3 PyTorch model to other formats. TensorFlow exports authored by https://github.com/zldrobit.

## Imports (internal)

- [[models.experimental]]
- [[models.tf]]
- [[models.yolo]]
- [[utils.dataloaders]]
- [[utils.general]]
- [[utils.torch_utils]]

## Imported by

- [[models.common]]
- [[models.common_orig]]

## Third-party / stdlib

`PIL`, `argparse`, `contextlib`, `coremltools`, `json`, `nncf`, `numpy`, `onnx`, `onnxsim`, `openvino`, `os`, `pandas`, `pathlib`, `platform`, `re`, `subprocess`, `sys`, `tensorflow`, `tensorflowjs`, `tensorrt`, `tflite_support`, `time`, `torch`, `warnings`, `x2paddle`

## Classes

- `iOSModel(torch.nn.Module)`

## Top-level functions

`export_formats()`, `try_export()`, `export_torchscript()`, `export_onnx()`, `export_openvino()`, `export_paddle()`, `export_coreml()`, `export_engine()`, `export_saved_model()`, `export_pb()`, `export_tflite()`, `export_edgetpu()`, `export_tfjs()`, `add_tflite_metadata()`, `pipeline_coreml()`, `run()`, `parse_opt()`, `main()`

## Local neighbourhood

```mermaid
graph LR
  subgraph (root)["(root)"]
    export["export"]
  end
  subgraph models["models"]
    models_common["models.common"]
    models_common_orig["models.common_orig"]
    models_experimental["models.experimental"]
    models_tf["models.tf"]
    models_yolo["models.yolo"]
  end
  subgraph utils["utils"]
    utils_dataloaders["utils.dataloaders"]
    utils_general["utils.general"]
    utils_torch_utils["utils.torch_utils"]
  end
  export --> models_experimental
  export --> models_tf
  export --> models_yolo
  export --> utils_dataloaders
  export --> utils_general
  export --> utils_torch_utils
  models_common --> export
  models_common_orig --> export
```

---

Back to [[Code Map]] | [[Home]]
