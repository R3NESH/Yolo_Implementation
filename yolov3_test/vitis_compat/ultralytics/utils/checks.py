"""Minimal stand-in for `ultralytics.utils.checks`."""


def check_requirements(*args, **kwargs):
    """No-op: the Vitis AI image pins its own versions, so pip requirement checks are skipped."""
    return True
