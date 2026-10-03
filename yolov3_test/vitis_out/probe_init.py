"""Scratch probe: find which line of YOLOv3DPUWrapper.__init__ segfaults."""

import sys

sys.path.insert(0, ".")  # script lives in vitis_out/, repo root is the cwd

import export_dpu_wrapper_AB3 as w  # noqa: F401,E402  (numpy shim + thread env vars)
import torch
import torch.nn as nn

print("a: imported", flush=True)
ckpt = torch.load("runs/train/exp2/weights/best.pt", map_location="cpu", weights_only=False)
print("b: loaded", flush=True)
model = ckpt.get("ema") or ckpt.get("model")
print("c: picked", type(model).__name__, flush=True)
model = model.float().eval()
print("d: float+eval", flush=True)
sub = model.model[:-1]
print("e: sliced", type(sub).__name__, len(sub), flush=True)
layers = nn.ModuleList([m for m in sub])
print("f: ModuleList built", len(layers), flush=True)
save = set(model.save)
print("g: save set", sorted(save), flush=True)
routes = [[m.f] if isinstance(m.f, int) else list(m.f) for m in layers]
print("h: routes ok", flush=True)
d = model.model[-1]
print("i: detect", type(d).__name__, "f=", d.f, "nl=", getattr(d, "nl", None), flush=True)
m0, m1, m2 = d.m[0], d.m[1], d.m[2]
print("j: heads ok", flush=True)
