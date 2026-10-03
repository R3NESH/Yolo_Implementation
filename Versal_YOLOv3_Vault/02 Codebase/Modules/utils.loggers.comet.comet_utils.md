---
tags: [code, module]
module: utils.loggers.comet.comet_utils
path: utils/loggers/comet/comet_utils.py
loc: 152
---

# `utils.loggers.comet.comet_utils`

> [!info] Source
> `yolov3_test/utils/loggers/comet/comet_utils.py` - 152 lines

## Imports (internal)

_None - leaf module._

## Imported by

- [[train]]

## Third-party / stdlib

`comet_ml`, `logging`, `os`, `urllib`, `yaml`

## Top-level functions

`download_model_checkpoint()`, `set_opt_parameters()`, `check_comet_weights()`, `check_comet_resume()`

## Local neighbourhood

```mermaid
graph LR
  subgraph (root)["(root)"]
    train["train"]
  end
  subgraph utils["utils"]
    utils_loggers_comet_comet_utils["utils.loggers.comet.comet_utils"]
  end
  train --> utils_loggers_comet_comet_utils
```

---

Back to [[Code Map]] | [[Home]]
