"""Let NumPy-2.x-pickled checkpoints load under the Vitis AI image's NumPy 1.x.

Checkpoints saved with NumPy >= 2.0 reference the `numpy._core.*` module paths.
NumPy 1.x only has `numpy.core.*`, so unpickling raises ModuleNotFoundError.

Mapping just the `numpy._core` parent is NOT enough and is actively dangerous:
the unpickler then resolves `numpy._core.multiarray` through the parent's
__path__ and re-executes numpy's C-extension module under a second name, which
double-initializes it and segfaults inside `torch.load`. Registering each
submodule up front avoids the re-import entirely.

Call `apply()` before `torch.load`.
"""

import sys

import numpy

# Submodules that pickled arrays/scalars/dtypes actually reference.
_SUBMODULES = (
    "multiarray",
    "umath",
    "numeric",
    "numerictypes",
    "_multiarray_umath",
    "einsumfunc",
    "fromnumeric",
)


def apply():
    """Aliases numpy._core[.sub] onto numpy.core[.sub]; no-op on NumPy 2.x."""
    if hasattr(numpy, "_core"):
        return False  # already NumPy 2.x, nothing to patch

    sys.modules.setdefault("numpy._core", numpy.core)
    for name in _SUBMODULES:
        sub = getattr(numpy.core, name, None)
        if sub is not None:
            sys.modules.setdefault("numpy._core." + name, sub)
    return True
