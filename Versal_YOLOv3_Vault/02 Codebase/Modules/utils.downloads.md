---
tags: [code, module]
module: utils.downloads
path: utils/downloads.py
loc: 129
---

# `utils.downloads`

> [!info] Source
> `yolov3_test/utils/downloads.py` - 129 lines

**Purpose:** Download utils.

## Imports (internal)

- [[utils.general]]

## Imported by

- [[models.common]]
- [[models.common_orig]]
- [[models.experimental]]
- [[train]]
- [[utils.general]]

## Third-party / stdlib

`logging`, `pathlib`, `requests`, `subprocess`, `torch`, `urllib`

## Top-level functions

`is_url()`, `gsutil_getsize()`, `url_getsize()`, `curl_download()`, `safe_download()`, `attempt_download()`

## Local neighbourhood

```mermaid
graph LR
  subgraph (root)["(root)"]
    train["train"]
  end
  subgraph models["models"]
    models_common["models.common"]
    models_common_orig["models.common_orig"]
    models_experimental["models.experimental"]
  end
  subgraph utils["utils"]
    utils_downloads["utils.downloads"]
    utils_general["utils.general"]
  end
  models_common --> utils_downloads
  models_common_orig --> utils_downloads
  models_experimental --> utils_downloads
  train --> utils_downloads
  utils_downloads --> utils_general
  utils_general --> utils_downloads
```

---

Back to [[Code Map]] | [[Home]]
