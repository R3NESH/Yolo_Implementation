"""Quantize the DPU wrapper with SiLU rewritten as an explicit x * sigmoid(x).

Why this exists
---------------
`compiled/` (the straight best.pt build) compiles but **cannot run on the board**:

    [UNILOG][FATAL][VAILIB_CPU_RUNNER_OPEN_LIB_ERROR][dlopen can not open lib!]
    lib=libvart_op_imp_aten__silu_.so ... No such file or directory

XIR has no definition for `aten::silu_`, so at compile time it invents one - which satisfies
`vai_c_xir` and says nothing about runtime. At runtime every CPU subgraph needs an operator
implementation library, and the board ships none for silu.

It does ship `libvart_op_imp_sigmoid.so` and `libvart_op_imp_mul.so`. And

    SiLU(x) == x * sigmoid(x)

is an exact identity, not an approximation. So tracing the model with the activation written
out explicitly should produce a graph built only from implemented ops, while computing exactly
what best.pt computes - no retraining, no accuracy loss, no GPU.

The graph will still be fragmented (sigmoid/mul run on the CPU, not the DPU), so this is the
*accurate but slow* path. The fast path is still a LeakyReLU finetune - see the vault notes
'DPU Subgraph Fragmentation' and 'Board mAP - LeakyReLU Without Finetune'.

Usage (inside the Vitis AI container):

    ./vitis_run.sh python -u dpu_silu_decompose_AB5.py --quant_mode calib
    ./vitis_run.sh python -u dpu_silu_decompose_AB5.py --quant_mode test
    ./vitis_run.sh vai_c_xir \
        -x quantize_result_silu_decomp/YOLOv3DPUWrapper_int.xmodel \
        -a /opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json \
        -o compiled_silu_decomp -n yolov3_vck190_silu_decomp
"""

import os
import sys
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


class SiLUDecomposed(nn.Module):
    """nn.SiLU written out as x * sigmoid(x).

    Numerically identical to nn.SiLU; the difference is purely in how the graph is traced.
    nn.SiLU traces to a single `aten::silu_` op that XIR cannot lower and VART cannot execute,
    whereas this traces to `sigmoid` and `mul`, which both have runtime implementations.
    """

    def forward(self, x):
        return x * torch.sigmoid(x)


def decompose_silu(module):
    """Recursively replaces every nn.SiLU with SiLUDecomposed. Returns the count."""
    swapped = 0
    for name, child in module.named_children():
        if isinstance(child, nn.SiLU):
            setattr(module, name, SiLUDecomposed())
            swapped += 1
        else:
            swapped += decompose_silu(child)
    return swapped


def run(quant_mode, weights, calib_dir, output_dir, img_size):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")

    model = YOLOv3DPUWrapper(weights, device=device).to(device)
    model.eval()

    n = decompose_silu(model)
    print(f"[INFO] Rewrote {n} SiLU activations as x * sigmoid(x).")
    if n == 0:
        print("[WARN] No SiLU found - nothing to do. Aborting.")
        return

    # Cheap sanity check: the rewrite must not change the numbers.
    with torch.no_grad():
        probe = torch.randn(8, 32, dtype=torch.float32)
        if not torch.allclose(SiLUDecomposed()(probe), nn.SiLU()(probe), atol=1e-6):
            print("[ERROR] x*sigmoid(x) disagrees with nn.SiLU. Aborting.")
            return
    print("[INFO] Identity check passed: x * sigmoid(x) == nn.SiLU(x).")

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
    p.add_argument("--output_dir", default="quantize_result_silu_decomp")
    p.add_argument("--img_size", type=int, default=416)
    a = p.parse_args()
    run(a.quant_mode, a.weights, a.data_dir, a.output_dir, a.img_size)
