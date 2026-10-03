"""Quantize the YOLOv8 DPU wrapper with vai_q_pytorch, calib then test.

Mirrors yolov3_test/quantize_vitis_AB4.py. Two runs are required per model and arm:

    ./v8_run.sh python -u v8_quantize.py --weights yolov8n.pt --act hardswish --quant_mode calib
    ./v8_run.sh python -u v8_quantize.py --weights yolov8n.pt --act hardswish --quant_mode test

`calib` collects activation ranges and writes the quant config; `test` re-runs one forward pass
and exports the INT8 .xmodel that vai_c_xir compiles.

CALIBRATION PREPROCESSING - must match the host
-----------------------------------------------
Plain resize to 640, BGR->RGB, /255, CHW. This is identical to
host_vck190/yolov8_decode.preprocess and to the prior work's build/calibrator.py, so
calibration, inference and the prior art all agree. A mismatch here is an invisible accuracy
tax that no later stage can recover.

CALIBRATION SET
---------------
Defaults to /datasets/coco_split/images/train - the prior work's 4000-image split of val2017,
which is disjoint from its 1000-image val half. The prior work itself calibrated on the *full*
val2017, so its numbers were produced with the eval set in the calibration pool. Using the
disjoint half costs nothing and lets the final report carry a clean read alongside the
like-for-like one.
"""

import os

# Prevent OpenMP / MKL thread allocation segfaults in the container.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/host_vck190")
import yolov8_decode as yd  # noqa: E402  - shared preprocessing, so calibration matches the host

torch.set_num_threads(1)

try:
    from pytorch_nndct.apis import torch_quantizer
except ImportError:
    print("[ERROR] 'pytorch_nndct' missing. Run inside vitis-ai-pytorch (./v8_run.sh).")
    sys.exit(1)

from v8_dpu_wrapper import ACT_CHOICES, YOLOv8DPUWrapper


def calibration_tensors(data_dir, img_size, max_samples, use_letterbox=True):
    """RGB / 0-1 CHW tensors, the same transform the host applies.

    Letterbox by default. The prior work calibrated with a plain resize but ran inference with
    letterbox - a mismatch - and still scored well, because preserving aspect ratio is worth
    more than the mismatch costs. Doing both with letterbox keeps calibration and inference
    consistent *and* keeps the aspect ratio, which is strictly better than either.
    """
    import cv2

    exts = (".jpg", ".jpeg", ".png", ".bmp")
    paths = []
    for root, _, files in os.walk(data_dir):
        for f in sorted(files):
            if f.lower().endswith(exts):
                paths.append(os.path.join(root, f))
    if not paths:
        raise FileNotFoundError(f"[ERROR] no calibration images under '{data_dir}'")

    # Even stride through the sorted list rather than the first N, so the sample spans the
    # whole set instead of whatever happens to sort first.
    if len(paths) > max_samples:
        idx = np.linspace(0, len(paths) - 1, max_samples).astype(int)
        paths = [paths[i] for i in idx]

    print(f"[INFO] {len(paths)} calibration images from '{data_dir}'")
    out = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"[WARNING] unreadable, skipping: {p}")
            continue
        if use_letterbox:
            arr, _, _ = yd.letterbox(img, img_size, channels_first=True)
        else:
            arr = yd.preprocess(img, img_size, channels_first=True)
        out.append(torch.from_numpy(arr))
    if not out:
        raise ValueError("[ERROR] no calibration images could be read")
    return out


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cpu")

    print(f"[INFO] wrapper: {args.weights}  act={args.act}")
    model = YOLOv8DPUWrapper(args.weights, act=args.act).to(device).eval()
    print(f"[INFO] replaced {model.replaced} activations")

    dummy = torch.randn(1, 3, args.imgsz, args.imgsz, dtype=torch.float32, device=device)

    print(f"[INFO] torch_quantizer mode='{args.quant_mode}' -> {args.output_dir}")
    quantizer = torch_quantizer(
        quant_mode=args.quant_mode,
        module=model,
        input_args=(dummy,),
        output_dir=args.output_dir,
    )
    quant_model = quantizer.quant_model.to(device).eval()

    images = calibration_tensors(args.data_dir, args.imgsz, args.max_samples,
                                 use_letterbox=not args.no_letterbox)

    print(f"[INFO] running forward passes ({args.quant_mode})...")
    with torch.no_grad():
        for i, t in enumerate(images):
            quant_model(t.unsqueeze(0).to(device))
            if args.quant_mode == "test":
                print("[INFO] single test forward pass complete")
                break
            if (i + 1) % 25 == 0:
                print(f"[INFO]   {i + 1}/{len(images)}")

    if args.quant_mode == "calib":
        quantizer.export_quant_config()
        print(f"\n[SUCCESS] calibration complete -> '{args.output_dir}'")
        print("[ACTION] now re-run with --quant_mode test to export the .xmodel")
    else:
        # This Vitis AI build's signature is export_xmodel(output_dir, deploy_check,
        # dynamic_batch) - there is no `deploy` kwarg. deploy_check=True additionally dumps
        # golden per-layer tensors; left off to keep the disk footprint small.
        quantizer.export_xmodel(output_dir=args.output_dir, deploy_check=False)
        print(f"\n[SUCCESS] INT8 .xmodel exported -> '{args.output_dir}'")
        print("[ACTION] compile it with vai_c_xir, then gate on tools/inspect_xmodel.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant_mode", default="calib", choices=["calib", "test"])
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--act", default="hardswish", choices=ACT_CHOICES)
    ap.add_argument("--data_dir", default="/datasets/coco_split/images/train")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max_samples", type=int, default=200)
    ap.add_argument("--no-letterbox", action="store_true",
                    help="calibrate with a plain square resize instead of letterbox")
    a = ap.parse_args()

    if a.output_dir is None:
        stem = os.path.splitext(os.path.basename(a.weights))[0]
        suffix = "" if not a.no_letterbox else "_plainresize"
        a.output_dir = f"quantize_result/{stem}_{a.act}{suffix}"
    run(a)
