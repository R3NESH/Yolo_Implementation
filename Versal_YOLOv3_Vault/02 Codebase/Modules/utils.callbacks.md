---
tags: [code, module]
module: utils.callbacks
path: utils/callbacks.py
loc: 73
---

# `utils.callbacks`

> [!info] Source
> `yolov3_test/utils/callbacks.py` - 73 lines

**Purpose:** Callback utils.

## Imports (internal)

_None - leaf module._

## Imported by

- [[train]]
- [[utils.loggers.comet.hpo]]
- [[val]]

## Third-party / stdlib

`threading`

## Classes

- `Callbacks`

## Local neighbourhood

```mermaid
graph LR
  subgraph (root)["(root)"]
    train["train"]
    val["val"]
  end
  subgraph utils["utils"]
    utils_callbacks["utils.callbacks"]
    utils_loggers_comet_hpo["utils.loggers.comet.hpo"]
  end
  train --> utils_callbacks
  utils_loggers_comet_hpo --> utils_callbacks
  val --> utils_callbacks
```

---

Back to [[Code Map]] | [[Home]]
