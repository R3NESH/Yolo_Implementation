"""Minimal stand-in for `ultralytics.utils.torch_utils`."""

import torch

_ver = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
TORCH_2_4 = _ver >= (2, 4)  # False on the Vitis AI image (torch 1.13)


def autocast(enabled=True, device="cuda"):
    """Mixed-precision context; falls back to the torch 1.x API signature."""
    if TORCH_2_4:
        return torch.amp.autocast(device, enabled=enabled)
    return torch.cuda.amp.autocast(enabled)
