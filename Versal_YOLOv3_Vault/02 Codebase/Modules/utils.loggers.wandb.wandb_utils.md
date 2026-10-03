---
tags: [code, module]
module: utils.loggers.wandb.wandb_utils
path: utils/loggers/wandb/wandb_utils.py
loc: 211
---

# `utils.loggers.wandb.wandb_utils`

> [!info] Source
> `yolov3_test/utils/loggers/wandb/wandb_utils.py` - 211 lines

## Imports (internal)

- [[utils.general]]

## Imported by

- [[utils.loggers]]

## Third-party / stdlib

`contextlib`, `logging`, `os`, `pathlib`, `sys`, `wandb`

## Classes

- `WandbLogger`

## Top-level functions

`all_logging_disabled()`

## Local neighbourhood

```mermaid
graph LR
  subgraph utils["utils"]
    utils_general["utils.general"]
    utils_loggers["utils.loggers"]
    utils_loggers_wandb_wandb_utils["utils.loggers.wandb.wandb_utils"]
  end
  utils_loggers --> utils_loggers_wandb_wandb_utils
  utils_loggers_wandb_wandb_utils --> utils_general
```

---

Back to [[Code Map]] | [[Home]]
