"""Evaluate the INT8-simulated model off-board, and write COCO predictions.

The board is the scarce resource and is not always available. vai_q_pytorch's `test` mode returns
a model that *simulates* INT8 arithmetic on the CPU, so the accuracy cost of quantization can be
measured before any hardware is involved. That makes this the last gate before board time: it
exercises the whole chain - wrapper, activation swap, quantization, host decode, NMS, COCO
category mapping - and a number that lands near the float baseline means everything downstream of
the DPU is wired correctly.

It is a *prediction*, not a substitute. Simulated INT8 and the real DPU are not bit-identical, and
this path does not exercise VART, the fix_point input scaling, or NHWC buffer handling - which is
exactly where 'Board Bring-Up' found all three of its bugs.

Usage (inside the container):

    ./v8_run.sh python -u v8_eval_quantized.py --weights yolov8n.pt --act hardswish --limit 500

Writes predictions JSON; score it with the same scorer used for everything else:

    Versal_AI/venv/bin/python tools/rescore_v8_prior.py <annotations> <predictions.json>

Pass --float to run the unquantized model instead, which gives the like-for-like float baseline
on the identical image subset - the comparison that makes the INT8 number interpretable.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import glob
import json
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, "/workspace/host_vck190")

from v8_dpu_wrapper import ACT_CHOICES, YOLOv8DPUWrapper  # noqa: E402
import yolov8_decode as yd  # noqa: E402

torch.set_num_threads(1)


def build_model(args):
    """Returns the model to run: INT8-simulated by default, float with --float."""
    wrapper = YOLOv8DPUWrapper(args.weights, act=args.act).eval()
    if args.float:
        print(f"[INFO] float model, act={args.act} ({wrapper.replaced} replaced)")
        return wrapper

    from pytorch_nndct.apis import torch_quantizer

    dummy = torch.randn(1, 3, args.imgsz, args.imgsz)
    quantizer = torch_quantizer(
        quant_mode="test",
        module=wrapper,
        input_args=(dummy,),
        output_dir=args.quant_dir,
    )
    print(f"[INFO] INT8-simulated model from '{args.quant_dir}'")
    return quantizer.quant_model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--act", default="hardswish", choices=ACT_CHOICES)
    ap.add_argument("--quant_dir", default=None)
    ap.add_argument("--images", default="/datasets/coco/images/val2017")
    ap.add_argument("--out", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--limit", type=int, default=500, help="0 = all images")
    ap.add_argument("--conf-thres", type=float, default=0.001)
    ap.add_argument("--iou-thres", type=float, default=0.45)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--float", action="store_true", help="run float instead of INT8")
    ap.add_argument("--letterbox", action="store_true",
                    help="aspect-preserving resize with padding instead of a plain square "
                         "resize. What ultralytics validates with, and what the prior work's "
                         "board runner used.")
    args = ap.parse_args()

    stem = os.path.splitext(os.path.basename(args.weights))[0]
    if args.quant_dir is None:
        args.quant_dir = f"quantize_result/{stem}_{args.act}"
    if args.out is None:
        kind = "float" if args.float else "int8"
        box = "_lb" if args.letterbox else ""
        args.out = f"sim_results/preds_{stem}_{args.act}_{kind}{box}.json"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    model = build_model(args)

    paths = sorted(glob.glob(os.path.join(args.images, "*.jpg")))
    if args.limit:
        # Even stride, so a subset spans the whole set rather than its first N by filename.
        idx = np.linspace(0, len(paths) - 1, min(args.limit, len(paths))).astype(int)
        paths = [paths[i] for i in sorted(set(idx))]
    print(f"[INFO] {len(paths)} images from {args.images}")

    preds = []
    t0 = time.time()
    for n, p in enumerate(paths, 1):
        img = cv2.imread(p)
        if img is None:
            print(f"[WARNING] unreadable, skipping: {p}")
            continue
        h, w = img.shape[:2]
        if args.letterbox:
            arr, gain, pad = yd.letterbox(img, args.imgsz, channels_first=True)
        else:
            arr, gain, pad = yd.preprocess(img, args.imgsz, channels_first=True), None, None
        x = torch.from_numpy(arr).unsqueeze(0)

        with torch.no_grad():
            outs = model(x)

        # Emulate the DPU's NHWC layout so the decode runs exactly as it will on board.
        feats = [o[0].permute(1, 2, 0).cpu().numpy() for o in outs]
        # With letterbox, stay in input space through the decode and undo gain/pad after;
        # with a plain resize the per-axis scale can be folded into the decode directly.
        boxes, scores, cls = yd.decode_all(
            feats,
            conf_thresh=args.conf_thres,
            scale_x=1.0 if args.letterbox else w / float(args.imgsz),
            scale_y=1.0 if args.letterbox else h / float(args.imgsz),
            input_size=args.imgsz,
        )
        if len(boxes) and args.letterbox:
            yd.unletterbox_boxes(boxes, gain, pad)
        if len(boxes):
            yd.clip_boxes(boxes, w, h)
            keep = yd.nms_per_class(boxes, scores, cls, args.iou_thres, args.max_det)
            image_id = int(os.path.splitext(os.path.basename(p))[0])
            for k in keep:
                preds.append({
                    "image_id": image_id,
                    "category_id": yd.COCO_CAT_IDS[int(cls[k])],
                    "bbox": [round(float(v), 3) for v in boxes[k]],
                    "score": round(float(scores[k]), 5),
                })

        if n % 50 == 0:
            rate = n / (time.time() - t0)
            print(f"[INFO]   {n}/{len(paths)}  {rate:.2f} img/s  {len(preds):,} detections")

    json.dump(preds, open(args.out, "w"))
    print(f"\n[SUCCESS] {len(preds):,} detections over {len(paths)} images -> {args.out}")
    print("[ACTION] score it with tools/rescore_v8_prior.py")


if __name__ == "__main__":
    main()
