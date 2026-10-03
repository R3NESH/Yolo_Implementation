"""Minimal offline `ultralytics` shim for the Vitis AI Docker environment.

The vitis-ai-pytorch image ships torch 1.13 / numpy 1.22 and no `ultralytics`
package, but this repo's `models.common` and `utils.*` import a few helpers from
it at module import time - which `torch.load` triggers when unpickling a
checkpoint. Only the symbols actually imported are provided here; they cover
plotting and version-check paths that the DPU quantization flow never exercises.

Enable by putting this directory on PYTHONPATH, e.g.
    PYTHONPATH=/workspace/yolov3_test/vitis_compat
so it never shadows a real `ultralytics` install outside the container.
"""

__version__ = "0.0.0-vitis-shim"
