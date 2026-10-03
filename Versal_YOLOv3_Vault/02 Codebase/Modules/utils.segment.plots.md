---
tags: [code, module]
module: utils.segment.plots
path: utils/segment/plots.py
loc: 151
---

# `utils.segment.plots`

> [!info] Source
> `yolov3_test/utils/segment/plots.py` - 151 lines

## Imports (internal)

- [[utils]]
- [[utils.general]]
- [[utils.plots]]

## Imported by

_Nothing imports this (entry point or unused)._

## Third-party / stdlib

`contextlib`, `cv2`, `math`, `matplotlib`, `numpy`, `pandas`, `pathlib`, `torch`

## Top-level functions

`plot_images_and_masks()`, `plot_results_with_masks()`

## Local neighbourhood

```mermaid
graph LR
  subgraph (root)["(root)"]
    utils["utils"]
  end
  subgraph utils["utils"]
    utils_general["utils.general"]
    utils_plots["utils.plots"]
    utils_segment_plots["utils.segment.plots"]
  end
  utils_segment_plots --> utils
  utils_segment_plots --> utils_general
  utils_segment_plots --> utils_plots
```

---

Back to [[Code Map]] | [[Home]]
