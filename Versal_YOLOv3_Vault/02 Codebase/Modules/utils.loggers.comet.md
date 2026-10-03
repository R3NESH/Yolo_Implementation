---
tags: [code, module]
module: utils.loggers.comet
path: utils/loggers/comet/__init__.py
loc: 556
---

# `utils.loggers.comet`

> [!info] Source
> `yolov3_test/utils/loggers/comet/__init__.py` - 556 lines

## Imports (internal)

- [[utils.dataloaders]]
- [[utils.general]]
- [[utils.metrics]]

## Imported by

- [[utils.loggers]]

## Third-party / stdlib

`PIL`, `comet_ml`, `glob`, `json`, `logging`, `os`, `pathlib`, `sys`, `torch`, `torchvision`, `yaml`

## Classes

- `CometLogger`

## Local neighbourhood

```mermaid
graph LR
  subgraph utils["utils"]
    utils_dataloaders["utils.dataloaders"]
    utils_general["utils.general"]
    utils_loggers["utils.loggers"]
    utils_loggers_comet["utils.loggers.comet"]
    utils_metrics["utils.metrics"]
  end
  utils_loggers --> utils_loggers_comet
  utils_loggers_comet --> utils_dataloaders
  utils_loggers_comet --> utils_general
  utils_loggers_comet --> utils_metrics
```

---

Back to [[Code Map]] | [[Home]]
