---
tags: [code, module]
module: utils
path: utils/__init__.py
loc: 106
---

# `utils`

> [!info] Source
> `yolov3_test/utils/__init__.py` - 106 lines

**Purpose:** utils/initialization.

## Imports (internal)

- [[utils.general]]
- [[utils.torch_utils]]

## Imported by

- [[models.common]]
- [[models.common_orig]]
- [[utils.autoanchor]]
- [[utils.general]]
- [[utils.metrics]]
- [[utils.plots]]
- [[utils.segment.plots]]

## Third-party / stdlib

`IPython`, `contextlib`, `os`, `platform`, `psutil`, `shutil`, `threading`, `ultralytics`

## Classes

- `TryExcept(contextlib.ContextDecorator)`

## Top-level functions

`emojis()`, `threaded()`, `join_threads()`, `notebook_init()`

## Local neighbourhood

```mermaid
graph LR
  subgraph (root)["(root)"]
    utils["utils"]
  end
  subgraph models["models"]
    models_common["models.common"]
    models_common_orig["models.common_orig"]
  end
  subgraph utils["utils"]
    utils_autoanchor["utils.autoanchor"]
    utils_general["utils.general"]
    utils_metrics["utils.metrics"]
    utils_plots["utils.plots"]
    utils_segment_plots["utils.segment.plots"]
    utils_torch_utils["utils.torch_utils"]
  end
  models_common --> utils
  models_common_orig --> utils
  utils --> utils_general
  utils --> utils_torch_utils
  utils_autoanchor --> utils
  utils_general --> utils
  utils_metrics --> utils
  utils_plots --> utils
  utils_segment_plots --> utils
```

---

Back to [[Code Map]] | [[Home]]
