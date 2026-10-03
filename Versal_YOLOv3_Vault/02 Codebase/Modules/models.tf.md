---
tags: [code, module]
module: models.tf
path: models/tf.py
loc: 751
---

# `models.tf`

> [!info] Source
> `yolov3_test/models/tf.py` - 751 lines

**Purpose:** TensorFlow, Keras and TFLite versions of YOLOv3

## Imports (internal)

- [[models.common]]
- [[models.experimental]]
- [[models.yolo]]
- [[utils.activations]]
- [[utils.general]]

## Imported by

- [[export]]

## Third-party / stdlib

`argparse`, `copy`, `numpy`, `pathlib`, `sys`, `tensorflow`, `torch`, `yaml`

## Classes

- `TFBN(keras.layers.Layer)`
- `TFPad(keras.layers.Layer)`
- `TFConv(keras.layers.Layer)`
- `TFDWConv(keras.layers.Layer)`
- `TFDWConvTranspose2d(keras.layers.Layer)`
- `TFFocus(keras.layers.Layer)`
- `TFBottleneck(keras.layers.Layer)`
- `TFCrossConv(keras.layers.Layer)`
- `TFConv2d(keras.layers.Layer)`
- `TFBottleneckCSP(keras.layers.Layer)`
- `TFC3(keras.layers.Layer)`
- `TFC3x(keras.layers.Layer)`
- `TFSPP(keras.layers.Layer)`
- `TFSPPF(keras.layers.Layer)`
- `TFDetect(keras.layers.Layer)`
- `TFSegment(TFDetect)`
- `TFProto(keras.layers.Layer)`
- `TFUpsample(keras.layers.Layer)`
- `TFConcat(keras.layers.Layer)`
- `TFModel`
- `AgnosticNMS(keras.layers.Layer)`

## Top-level functions

`parse_model()`, `activations()`, `representative_dataset_gen()`, `run()`, `parse_opt()`, `main()`

## Local neighbourhood

```mermaid
graph LR
  subgraph (root)["(root)"]
    export["export"]
  end
  subgraph models["models"]
    models_common["models.common"]
    models_experimental["models.experimental"]
    models_tf["models.tf"]
    models_yolo["models.yolo"]
  end
  subgraph utils["utils"]
    utils_activations["utils.activations"]
    utils_general["utils.general"]
  end
  export --> models_tf
  models_tf --> models_common
  models_tf --> models_experimental
  models_tf --> models_yolo
  models_tf --> utils_activations
  models_tf --> utils_general
```

---

Back to [[Code Map]] | [[Home]]
