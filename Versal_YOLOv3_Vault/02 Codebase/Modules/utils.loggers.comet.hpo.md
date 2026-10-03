---
tags: [code, module]
module: utils.loggers.comet.hpo
path: utils/loggers/comet/hpo.py
loc: 127
---

# `utils.loggers.comet.hpo`

> [!info] Source
> `yolov3_test/utils/loggers/comet/hpo.py` - 127 lines

## Imports (internal)

- [[train]]
- [[utils.callbacks]]
- [[utils.general]]
- [[utils.torch_utils]]

## Imported by

_Nothing imports this (entry point or unused)._

## Third-party / stdlib

`argparse`, `comet_ml`, `json`, `logging`, `os`, `pathlib`, `sys`

## Top-level functions

`get_args()`, `run()`

## Local neighbourhood

```mermaid
graph LR
  subgraph (root)["(root)"]
    train["train"]
  end
  subgraph utils["utils"]
    utils_callbacks["utils.callbacks"]
    utils_general["utils.general"]
    utils_loggers_comet_hpo["utils.loggers.comet.hpo"]
    utils_torch_utils["utils.torch_utils"]
  end
  utils_loggers_comet_hpo --> train
  utils_loggers_comet_hpo --> utils_callbacks
  utils_loggers_comet_hpo --> utils_general
  utils_loggers_comet_hpo --> utils_torch_utils
```

---

Back to [[Code Map]] | [[Home]]
