---
tags: [code, module]
module: utils.autobatch
path: utils/autobatch.py
loc: 73
---

# `utils.autobatch`

> [!info] Source
> `yolov3_test/utils/autobatch.py` - 73 lines

**Purpose:** Auto-batch utils.

## Imports (internal)

- [[utils.general]]
- [[utils.torch_utils]]

## Imported by

- [[train]]

## Third-party / stdlib

`copy`, `numpy`, `torch`

## Top-level functions

`check_train_batch_size()`, `autobatch()`

## Local neighbourhood

```mermaid
graph LR
  subgraph (root)["(root)"]
    train["train"]
  end
  subgraph utils["utils"]
    utils_autobatch["utils.autobatch"]
    utils_general["utils.general"]
    utils_torch_utils["utils.torch_utils"]
  end
  train --> utils_autobatch
  utils_autobatch --> utils_general
  utils_autobatch --> utils_torch_utils
```

---

Back to [[Code Map]] | [[Home]]
