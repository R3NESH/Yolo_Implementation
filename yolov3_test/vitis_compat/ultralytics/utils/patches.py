"""Minimal stand-in for `ultralytics.utils.patches`."""

import torch


def torch_load(*args, **kwargs):
    """torch.load with `weights_only` defaulted off, so custom YOLO classes unpickle."""
    kwargs.setdefault("weights_only", False)
    try:
        return torch.load(*args, **kwargs)
    except TypeError:  # torch predates the weights_only kwarg
        kwargs.pop("weights_only", None)
        return torch.load(*args, **kwargs)
