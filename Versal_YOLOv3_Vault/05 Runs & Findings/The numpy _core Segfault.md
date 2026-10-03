---
tags: [finding, bug, debugging, numpy, fixed]
date: 2026-09-11
status: fixed
---

# The numpy _core Segfault

A real bug in the project's deployment scripts: a compat shim that was **worse than no shim at
all**, turning a clean error message into a silent segmentation fault.

Related: [[Compat Shims]], [[Changes Log]], [[Checkpoint Inspection]].

---

## Symptom

Running `export_dpu_wrapper_AB3.py` inside the Vitis AI container: the process died with
**exit code 139** (SIGSEGV) inside `torch.load`. No Python traceback, no error message, nothing to
grep. A hard crash in C.

## The setup that causes it

| | Checkpoint was saved with | Container has |
| --- | --- | --- |
| torch | 2.x | 1.13.1 |
| NumPy | 2.x | 1.22.0 |

NumPy 2.0 renamed its internals from `numpy.core.*` to `numpy._core.*`. A pickle written under
NumPy 2.x embeds the **new** module paths. Unpickling it under NumPy 1.x raises:

```
ModuleNotFoundError: No module named 'numpy._core'
```

The project's scripts already anticipated this, with a shim at the top of
`export_dpu_wrapper_AB3.py` and `quantize_vitis_AB4.py`:

```python
if not hasattr(numpy, '_core'):
    try:
        sys.modules['numpy._core'] = numpy.core
    except AttributeError:
        pass
```

## Why that shim segfaults

It registers only the **parent** module. That is the trap.

When the unpickler then needs `numpy._core.multiarray`, Python finds `numpy._core` in
`sys.modules`, and resolves the submodule through that parent's `__path__`. Because the parent
*is* `numpy.core`, the path resolves — and Python **imports NumPy's C extension a second time**,
under a second module name.

A C extension module initialised twice in one process has two copies of its static state. NumPy's
multiarray module does not survive that. The result is a segfault deep inside the unpickler, with
no Python-level exception to catch.

> [!danger] Partial shims are worse than none
> With **no** shim: a clean `ModuleNotFoundError` naming the exact problem.
> With the **parent-only** shim: SIGSEGV, no traceback, no clue.
> The shim converted a self-diagnosing error into the hardest class of bug to debug.

## How it was located

A bisection probe (`vitis_out/probe_init.py`) that prints a marker after every step of the
wrapper's `__init__`, with `flush=True` so output survives the crash:

```python
print("a: imported", flush=True)
ckpt = torch.load("runs/train/exp2/weights/best.pt", map_location="cpu", weights_only=False)
print("b: loaded", flush=True)
```

Output stopped after `a:` and never reached `b:` — isolating the crash to `torch.load` itself
rather than to any of the model surgery that follows it.

## The fix

`vitis_compat/np2_pickle_compat.py` registers the parent **and** every submodule a pickled array,
scalar or dtype can reference, so the re-import never happens:

```python
_SUBMODULES = (
    "multiarray", "umath", "numeric", "numerictypes",
    "_multiarray_umath", "einsumfunc", "fromnumeric",
)

def apply():
    """Aliases numpy._core[.sub] onto numpy.core[.sub]; no-op on NumPy 2.x."""
    if hasattr(numpy, "_core"):
        return False                      # already NumPy 2.x, nothing to patch
    sys.modules.setdefault("numpy._core", numpy.core)
    for name in _SUBMODULES:
        sub = getattr(numpy.core, name, None)
        if sub is not None:
            sys.modules.setdefault("numpy._core." + name, sub)
    return True
```

Design choices that matter:

- **`setdefault`, not assignment** — never clobber a module that genuinely exists.
- **Guarded on `hasattr(numpy, "_core")`** — a complete no-op on NumPy 2.x, so the same file is
  safe inside and outside the container.
- **`getattr(..., None)` per submodule** — tolerates NumPy 1.x versions that lack one of them.
- **One shared module** rather than three copied inline blocks, so the next script to need it
  cannot get a subtly different version.

Both `export_dpu_wrapper_AB3.py` and `quantize_vitis_AB4.py` now call
`np2_pickle_compat.apply()` before `torch.load`.

## Takeaway

The proper long-term fix is to stop straddling NumPy major versions: re-save the checkpoint from
the training environment with NumPy 1.x pickling, or do the quantization in an environment whose
NumPy matches. The shim is a bridge, not a cure — but it is a **correct** bridge, which the
original was not.

---

Back to [[Code Map]] | [[Home]]
