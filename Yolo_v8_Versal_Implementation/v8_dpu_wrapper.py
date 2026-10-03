"""DPU wrapper for YOLOv8: run backbone and neck, emit the three raw head tensors.

Same principle as yolov3_test/export_dpu_wrapper_AB3.py - the DPU cannot run the detection
head's decode (DFL expectation, sigmoid, NMS are not fixed-function convolutions), so the
wrapper stops at the head's output convolutions and the host does the rest.

YOLOv8's Detect head differs from YOLOv3's in a way that matters here. v3 has one conv per
scale; v8 has two parallel branches per scale - `cv2` (box) and `cv3` (class) - whose outputs
are concatenated:

    cv2[i](x) -> 64 channels   4 sides x 16 DFL bins
    cv3[i](x) -> 80 channels   class logits, no objectness
    concat    -> 144 channels

So this emits three tensors of 144 channels at strides 8/16/32. Concat is a DPU-native op, and
the already-compiled models in Versal_AI report exactly (1, 80, 80, 144) / (1, 40, 40, 144) /
(1, 20, 20, 144), so keeping the concat on-device matches the prior art.

Activations (see 'Implementation Plan - YOLOv8', step 3):

    --act silu      leave as-is. Compiles, but will NOT run on the board: there is no
                    libvart_op_imp_aten__silu_.so. Present only for float reference.
    --act hardswish arm A, the deployable path. DPUCVDX8G implements Hardswish natively;
                    prior art gets 2 subgraphs and 6-8 FPS. An approximation.
    --act decompose arm B, the measurement. SiLU rewritten as the exact identity
                    x * sigmoid(x), which traces to ops the board does implement. Heavily
                    fragmented and slow, but numerically exact - it isolates INT8 loss with
                    the activation held fixed, so A minus B prices the Hardswish swap.

Note ultralytics declares `default_act = nn.SiLU()` as a *class* attribute on Conv, so every
convolution shares one instance and `model.modules()` de-duplicates it to a count of 1. Walking
named_children() and rebinding with setattr gives each Conv its own module, which is what makes
the swap real.
"""

import os

# Prevent OpenMP / MKL thread allocation segfaults in the container.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse

import torch
import torch.nn as nn

torch.set_num_threads(1)

ACT_CHOICES = ("silu", "hardswish", "decompose")


class SiLUDecomposed(nn.Module):
    """nn.SiLU written out as x * sigmoid(x).

    Numerically identical to nn.SiLU. The difference is only in how the graph traces: nn.SiLU
    becomes a single `aten::silu_` that VART has no implementation library for, whereas this
    becomes `sigmoid` and `mul`, both of which the board ships.
    """

    def forward(self, x):
        return x * torch.sigmoid(x)


def replace_activation(module, kind):
    """Rebind every SiLU child to `kind`, returning the number replaced."""
    if kind == "silu":
        return 0
    factory = {"hardswish": nn.Hardswish, "decompose": SiLUDecomposed}[kind]
    count = 0
    for name, child in module.named_children():
        if isinstance(child, nn.SiLU):
            setattr(module, name, factory())
            count += 1
        else:
            count += replace_activation(child, kind)
    return count


def load_detection_model(weights_path):
    """Load a YOLOv8 checkpoint and return the float, eval-mode DetectionModel."""
    try:
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 1.13 has no weights_only kwarg
        ckpt = torch.load(weights_path, map_location="cpu")

    if isinstance(ckpt, dict):
        model = ckpt.get("ema") or ckpt.get("model")
    else:
        model = ckpt
    return model.float().eval()


class YOLOv8DPUWrapper(nn.Module):
    """Backbone + neck + the head's output convolutions. No decode, no NMS."""

    def __init__(self, weights_path, act="hardswish"):
        super().__init__()
        model = load_detection_model(weights_path)
        self.replaced = replace_activation(model, act)
        self.act_kind = act

        self.layers = nn.ModuleList(list(model.model[:-1]))
        self.save = set(model.save)
        self.routes = [[m.f] if isinstance(m.f, int) else list(m.f) for m in self.layers]

        detect = model.model[-1]
        self.detect_f = list(detect.f)
        self.cv2 = detect.cv2          # box branch, one per scale
        self.cv3 = detect.cv3          # class branch, one per scale
        self.nl = detect.nl
        self.nc = detect.nc
        self.reg_max = detect.reg_max
        self.strides = [int(s) for s in detect.stride.tolist()]

    def forward(self, x):
        y = []
        for m, route in zip(self.layers, self.routes):
            if route == [-1]:
                x = m(x)
            else:
                inputs = [x if j == -1 else y[j] for j in route]
                x = m(inputs[0]) if len(inputs) == 1 else m(inputs)
            y.append(x if m.i in self.save else None)

        feats = [y[j] for j in self.detect_f]
        return tuple(
            torch.cat((self.cv2[i](feats[i]), self.cv3[i](feats[i])), 1)
            for i in range(self.nl)
        )


def main():
    ap = argparse.ArgumentParser(description="Inspect the YOLOv8 DPU wrapper's raw outputs")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--act", default="hardswish", choices=ACT_CHOICES)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    wrapper = YOLOv8DPUWrapper(args.weights, act=args.act).eval()
    print(f"[INFO] {args.weights}  act={args.act}  replaced={wrapper.replaced} activations")
    print(f"[INFO] nc={wrapper.nc} reg_max={wrapper.reg_max} strides={wrapper.strides}")

    with torch.no_grad():
        outs = wrapper(torch.randn(1, 3, args.imgsz, args.imgsz))

    print("[INFO] DPU-safe raw output shapes:")
    for i, o in enumerate(outs):
        expected = args.imgsz // wrapper.strides[i]
        ok = o.shape[2] == expected and o.shape[1] == 4 * wrapper.reg_max + wrapper.nc
        print(f"   scale {i} stride {wrapper.strides[i]:>2}: {tuple(o.shape)}  {'OK' if ok else 'MISMATCH'}")


if __name__ == "__main__":
    main()
