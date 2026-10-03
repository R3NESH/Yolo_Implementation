"""Verify the NumPy host decode against the model's own PyTorch Detect decode.

Runs in the Vitis AI container (torch available):

    cd yolov3_test && ./vitis_run.sh python ../host_vck190/verify_decode.py

It takes the wrapper's raw conv outputs, decodes them two ways - once with
PyTorch using the model's own grid/anchor buffers and formula, once with
`yolo_decode`'s pure-NumPy path through the NHWC layout the DPU produces - and
reports the maximum disagreement. They should match to float precision.

This is what makes the port trustworthy: it checks the maths *and* the
channel/axis ordering, which is where a hand-written decode usually goes wrong.
"""

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(os.path.dirname(_HERE), "yolov3_test")
sys.path.insert(0, _REPO)                      # so `vitis_compat` etc. resolve
sys.path.insert(0, os.path.join(_REPO, "vitis_compat"))
sys.path.insert(0, _HERE)

from np2_pickle_compat import apply as _apply_np2_compat  # noqa: E402

_apply_np2_compat()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import yolo_decode as yd  # noqa: E402

WEIGHTS = os.path.join(_REPO, "runs/train/exp2/weights/best.pt")


def torch_reference(raw, grid, anchor_grid, stride):
    """The model's own decode, applied to one raw output tensor [1, 255, g, g]."""
    bs, _, ny, nx = raw.shape
    t = raw.view(bs, yd.NA, yd.NO, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
    xy, wh, conf = t.sigmoid().split((2, 2, yd.NUM_CLASSES + 1), 4)
    xy = (xy * 2 + grid) * stride
    wh = (wh * 2) ** 2 * anchor_grid
    return xy, wh, conf


def numpy_candidate(raw_np, g):
    """yolo_decode's maths, vectorised over the whole grid, from NHWC input."""
    nhwc = raw_np[0].transpose(1, 2, 0)              # [g, g, 255]
    pred = nhwc.reshape(g, g, yd.NA, yd.NO)          # [row, col, anchor, 85]
    stride = yd.INPUT_SIZE / float(g)

    sxy = yd.sigmoid(pred[..., 0:2])
    swh = yd.sigmoid(pred[..., 2:4])
    cxy = (sxy * 2.0 - 0.5 + yd._GRIDS[g]) * stride
    wh = (swh * 2.0) ** 2 * yd._ANCHOR_MAPS[g]
    conf = yd.sigmoid(pred[..., 4:])
    return cxy, wh, conf


def main():
    from export_dpu_wrapper_AB3 import YOLOv3DPUWrapper

    print("loading wrapper ...")
    wrapper = YOLOv3DPUWrapper(WEIGHTS, device="cpu")
    wrapper.eval()

    # The Detect module carries the authoritative grid/anchor buffers.
    ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    model = (ckpt.get("ema") or ckpt.get("model")).float().eval()
    detect = model.model[-1]

    torch.manual_seed(0)
    x = torch.rand(1, 3, yd.INPUT_SIZE, yd.INPUT_SIZE)

    with torch.no_grad():
        outs = wrapper(x)

    print("raw output shapes:", [tuple(o.shape) for o in outs])
    print()

    worst_xy = worst_wh = worst_conf = 0.0
    for i, raw in enumerate(outs):
        g = raw.shape[-1]
        stride = float(detect.stride[i])
        grid, anchor_grid = detect._make_grid(g, g, i)

        # sanity: the module's anchors must equal the hardcoded pixel anchors.
        # anchor_grid is (1, na, ny, nx, 2) broadcast, so take one spatial cell.
        ag = anchor_grid[0, :, 0, 0, :].numpy()
        expect = yd.ANCHORS[g]
        if np.allclose(ag, expect, atol=1e-3):
            print("  anchors g=%-3d ok: %s" % (g, ag.astype(int).tolist()))
        else:
            print("  !! anchor mismatch at g=%d: model=%s file=%s" % (g, ag.tolist(), expect.tolist()))

        t_xy, t_wh, t_conf = torch_reference(raw, grid, anchor_grid, stride)
        n_xy, n_wh, n_conf = numpy_candidate(raw.numpy(), g)

        # torch is [1, anchor, row, col, 2]; numpy is [row, col, anchor, 2]
        t_xy = t_xy[0].permute(1, 2, 0, 3).numpy()
        t_wh = t_wh[0].permute(1, 2, 0, 3).numpy()
        t_conf = t_conf[0].permute(1, 2, 0, 3).numpy()

        d_xy = np.abs(t_xy - n_xy).max()
        d_wh = np.abs(t_wh - n_wh).max()
        d_conf = np.abs(t_conf - n_conf).max()
        worst_xy, worst_wh, worst_conf = max(worst_xy, d_xy), max(worst_wh, d_wh), max(worst_conf, d_conf)

        print("scale g=%-3d stride=%-3.0f  max|dxy|=%.3e  max|dwh|=%.3e  max|dconf|=%.3e"
              % (g, stride, d_xy, d_wh, d_conf))

    print()
    tol = 2e-3  # generous: float32 across two different op orders
    ok = worst_xy < tol and worst_wh < tol and worst_conf < tol
    print("worst: xy=%.3e wh=%.3e conf=%.3e" % (worst_xy, worst_wh, worst_conf))
    print("RESULT:", "MATCH - numpy decode is correct" if ok else "MISMATCH - do not ship")

    # Also show what the WRONG (old) formula would have produced, for the record.
    raw0 = outs[0].numpy()
    g0 = raw0.shape[-1]
    pred = raw0[0].transpose(1, 2, 0).reshape(g0, g0, yd.NA, yd.NO)
    swh = yd.sigmoid(pred[..., 2:4])
    right = ((swh * 2.0) ** 2 * yd._ANCHOR_MAPS[g0])
    wrong = np.exp(np.clip(pred[..., 2:4], -3, 3)) * yd._ANCHOR_MAPS[g0]
    print()
    print("old Darknet formula vs correct, on the finest scale (pixels):")
    print("  correct  mean w=%.1f  h=%.1f" % (right[..., 0].mean(), right[..., 1].mean()))
    print("  old/exp  mean w=%.1f  h=%.1f" % (wrong[..., 0].mean(), wrong[..., 1].mean()))
    print("  ratio    w=%.2fx h=%.2fx" % (
        wrong[..., 0].mean() / right[..., 0].mean(),
        wrong[..., 1].mean() / right[..., 1].mean()))

    # candidate count check
    total = sum(3 * (o.shape[-1] ** 2) for o in outs)
    print("\ntotal candidates across scales: %d (expect 10647)" % total)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
