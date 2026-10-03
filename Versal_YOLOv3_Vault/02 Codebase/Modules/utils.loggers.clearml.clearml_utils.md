---
tags: [code, module]
module: utils.loggers.clearml.clearml_utils
path: utils/loggers/clearml/clearml_utils.py
loc: 175
---

# `utils.loggers.clearml.clearml_utils`

> [!info] Source
> `yolov3_test/utils/loggers/clearml/clearml_utils.py` - 175 lines

**Purpose:** Main Logger class for ClearML experiment tracking.

## Imports (internal)

_None - leaf module._

## Imported by

- [[utils.loggers]]

## Third-party / stdlib

`clearml`, `glob`, `numpy`, `pathlib`, `re`, `ultralytics`, `yaml`

## Classes

- `ClearmlLogger`

## Top-level functions

`construct_dataset()`

## Local neighbourhood

```mermaid
graph LR
  subgraph utils["utils"]
    utils_loggers["utils.loggers"]
    utils_loggers_clearml_clearml_utils["utils.loggers.clearml.clearml_utils"]
  end
  utils_loggers --> utils_loggers_clearml_clearml_utils
```

---

Back to [[Code Map]] | [[Home]]
