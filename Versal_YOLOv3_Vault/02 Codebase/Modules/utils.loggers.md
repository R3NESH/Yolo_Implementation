---
tags: [code, module]
module: utils.loggers
path: utils/loggers/__init__.py
loc: 440
---

# `utils.loggers`

> [!info] Source
> `yolov3_test/utils/loggers/__init__.py` - 440 lines

**Purpose:** Logging utils.

## Imports (internal)

- [[utils.general]]
- [[utils.loggers.clearml.clearml_utils]]
- [[utils.loggers.comet]]
- [[utils.loggers.wandb.wandb_utils]]
- [[utils.plots]]
- [[utils.torch_utils]]

## Imported by

- [[train]]

## Third-party / stdlib

`clearml`, `comet_ml`, `os`, `pathlib`, `pkg_resources`, `torch`, `wandb`, `warnings`

## Classes

- `Loggers`
- `GenericLogger`

## Top-level functions

`log_tensorboard_graph()`, `web_project_name()`

## Local neighbourhood

```mermaid
graph LR
  subgraph (root)["(root)"]
    train["train"]
  end
  subgraph utils["utils"]
    utils_general["utils.general"]
    utils_loggers["utils.loggers"]
    utils_loggers_clearml_clearml_utils["utils.loggers.clearml.clearml_utils"]
    utils_loggers_comet["utils.loggers.comet"]
    utils_loggers_wandb_wandb_utils["utils.loggers.wandb.wandb_utils"]
    utils_plots["utils.plots"]
    utils_torch_utils["utils.torch_utils"]
  end
  train --> utils_loggers
  utils_loggers --> utils_general
  utils_loggers --> utils_loggers_clearml_clearml_utils
  utils_loggers --> utils_loggers_comet
  utils_loggers --> utils_loggers_wandb_wandb_utils
  utils_loggers --> utils_plots
  utils_loggers --> utils_torch_utils
```

---

Back to [[Code Map]] | [[Home]]
