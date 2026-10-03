"""Experiment: does replacing SiLU with LeakyReLU defragment the DPU subgraph?

The stock exp2 checkpoint uses SiLU, which XIR has no definition for. The compiler
therefore islands every convolution between CPU-assigned transpose ops, yielding
52 DPU subgraphs (see the vault note "DPU Subgraph Fragmentation").

This script re-runs the quantize flow with every SiLU swapped for LeakyReLU
**in memory** - no second checkpoint is written to disk - so the resulting
subgraph count can be compared directly against the baseline.

    python dpu_silu_experiment.py --quant_mode calib
    python dpu_silu_experiment.py --quant_mode test

NOTE ON ACCURACY: swapping an activation without finetuning changes the function
the network computes, so detections from this build are NOT expected to be good.
The purpose here is purely to measure DPU mapping, not accuracy. A real fix
finetunes (or retrains) with LeakyReLU - see "Vitis AI DPU Concepts".
"""

import os

# Match quantize_vitis_AB4.py: single-threaded to avoid OpenMP crashes in Docker.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys

from np2_pickle_compat import apply as _apply_np2_compat

_apply_np2_compat()

import argparse

import torch
import torch.nn as nn

torch.set_num_threads(1)

try:
    from pytorch_nndct.apis import torch_quantizer
except ImportError:
    print("[ERROR] 'pytorch_nndct' missing. Run inside vitis-ai-pytorch (./vitis_run.sh).")
    sys.exit(1)

from export_dpu_wrapper_AB3 import YOLOv3DPUWrapper
from quantize_vitis_AB4 import get_calibration_dataloader

# The DPU implements leaky_relu only at alpha = 26/256; any other slope falls back
# to the CPU, which would defeat the entire point of the swap.
DPU_LEAKY_ALPHA = 0.1015625


def swap_silu(module, alpha=DPU_LEAKY_ALPHA):
    """Recursively replaces every nn.SiLU with nn.LeakyReLU(alpha). Returns the count."""
    swapped = 0
    for name, child in module.named_children():
        if isinstance(child, nn.SiLU):
            setattr(module, name, nn.LeakyReLU(negative_slope=alpha, inplace=True))
            swapped += 1
        else:
            swapped += swap_silu(child, alpha)
    return swapped


def run(quant_mode, weights, calib_dir, output_dir, img_size, alpha):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")

    model = YOLOv3DPUWrapper(weights, device=device).to(device)
    model.eval()

    n = swap_silu(model, alpha)
    print(f"[INFO] Replaced {n} SiLU activations with LeakyReLU(negative_slope={alpha}).")
    if n == 0:
        print("[WARN] No SiLU found - nothing to compare. Aborting.")
        return

    dummy = torch.randn(1, 3, img_size, img_size, dtype=torch.float32)

    quantizer = torch_quantizer(
        quant_mode=quant_mode,
        module=model,
        input_args=(dummy,),
        output_dir=output_dir,
    )
    quant_model = quantizer.quant_model.to(device)
    quant_model.eval()

    calib_images = get_calibration_dataloader(calib_dir, img_size=img_size)

    with torch.no_grad():
        for img in calib_images:
            _ = quant_model(img.unsqueeze(0).to(device))
            if quant_mode == "test":
                break

    if quant_mode == "calib":
        quantizer.export_quant_config()
        print(f"\n[SUCCESS] Calibration complete -> '{output_dir}'.")
    else:
        quantizer.export_xmodel(output_dir=output_dir, deploy_check=False)
        print(f"\n[SUCCESS] xmodel exported -> '{output_dir}'.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quant_mode", choices=["calib", "test"], default="calib")
    p.add_argument("--weights", default="runs/train/exp2/weights/best.pt")
    p.add_argument("--data_dir", default="../datasets/coco128/images/train2017")
    p.add_argument("--output_dir", default="quantize_result_lrelu")
    p.add_argument("--img_size", type=int, default=416)
    p.add_argument("--alpha", type=float, default=DPU_LEAKY_ALPHA)
    a = p.parse_args()
    run(a.quant_mode, a.weights, a.data_dir, a.output_dir, a.img_size, a.alpha)
