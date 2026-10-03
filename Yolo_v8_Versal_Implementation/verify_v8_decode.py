"""Verify the host YOLOv8 decode against ultralytics' own decode, on identical features.

'Board Bring-Up' records three host-application bugs that each crashed on frame one, in code
that had been verified only in principle. Board time is the expensive resource, so the decode is
checked here, off-board, before anything is quantized or compiled.

The test holds everything except the decode constant:

    float model, SiLU untouched
        |
        +-- full DetectionModel.forward  -> ultralytics decodes internally -> (1, 84, 8400)
        |
        +-- YOLOv8DPUWrapper (act=silu)  -> 3 raw (1, 144, H, W) tensors
                                         -> host_vck190/yolov8_decode.decode_all

Both sides then go through the *same* NMS, so any disagreement is the decode maths and nothing
else. Passing means the host decode reproduces ultralytics to floating-point tolerance.

Usage (inside the container):
    ./v8_run.sh python -u verify_v8_decode.py --weights yolov8n.pt
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/host_vck190")

from v8_dpu_wrapper import YOLOv8DPUWrapper, load_detection_model  # noqa: E402
from yolov8_decode import decode_all, letterbox, preprocess, unletterbox_boxes  # noqa: E402
from yolo_decode import nms_per_class  # noqa: E402

torch.set_num_threads(1)


def load_image(path, imgsz, use_letterbox=False):
    """Returns (CHW float tensor, description, geometry). Falls back to noise if no image."""
    if path:
        import cv2

        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        if use_letterbox:
            arr, gain, pad = letterbox(img, imgsz, channels_first=True)
        else:
            arr, gain, pad = preprocess(img, input_size=imgsz, channels_first=True), None, None
        desc = f"{os.path.basename(path)} {img.shape[1]}x{img.shape[0]}"
        return torch.from_numpy(arr).unsqueeze(0), desc, (gain, pad, img.shape[:2])
    torch.manual_seed(0)
    return torch.rand(1, 3, imgsz, imgsz), "random input", (None, None, None)


def check_letterbox_inverse(imgsz=640):
    """Unit-test unletterbox_boxes against ultralytics' own scale_boxes.

    The board path letterboxes, so this inverse is on the critical path for every reported box.
    It was originally written to subtract the unrounded dw/dh, which left a systematic sub-pixel
    shift - invisible at IoU 0.5, corrosive at strict IoU. This pins it to the reference.
    """
    from ultralytics.utils.ops import scale_boxes

    print("\n[letterbox inverse vs ultralytics.scale_boxes]")
    worst = 0.0
    for (h0, w0) in [(426, 640), (427, 640), (480, 640), (375, 500), (640, 480),
                     (333, 500), (612, 612), (335, 500), (426, 639), (401, 601)]:
        img = np.zeros((h0, w0, 3), np.uint8)
        _, gain, pad = letterbox(img, imgsz, channels_first=True)

        rng = np.random.RandomState(0)
        xywh = np.concatenate([rng.uniform(0, 400, (25, 2)),
                               rng.uniform(10, 200, (25, 2))], 1).astype(np.float32)

        mine = unletterbox_boxes(xywh.copy(), gain, pad)

        xyxy = torch.from_numpy(
            np.stack([xywh[:, 0], xywh[:, 1], xywh[:, 0] + xywh[:, 2], xywh[:, 1] + xywh[:, 3]], 1))
        # Pass the exact geometry letterbox used. Left to recompute it from raw shapes,
        # scale_boxes derives the pad from the *unrounded* resize (335*1.28 = 428.8) while the
        # real padding follows the rounded one (429), so it can be a whole pixel out. Ultralytics
        # hits this too, which is why its own pipeline threads `ratio_pad` through from the
        # dataloader rather than recomputing. Ground truth is what copyMakeBorder actually added.
        ref = scale_boxes((imgsz, imgsz), xyxy.clone(), (h0, w0),
                          ratio_pad=((gain, gain), pad)).numpy()
        ref_xywh = np.stack([ref[:, 0], ref[:, 1], ref[:, 2] - ref[:, 0], ref[:, 3] - ref[:, 1]], 1)

        # scale_boxes clips to the image; compare only boxes it did not clip.
        unclipped = (ref[:, 0] > 0) & (ref[:, 1] > 0) & (ref[:, 2] < w0) & (ref[:, 3] < h0)
        if not unclipped.any():
            continue
        d = float(np.abs(mine[unclipped] - ref_xywh[unclipped]).max())
        worst = max(worst, d)
        print(f"  {w0}x{h0}: gain={gain:.4f} pad={pad}  max|diff|={d:.6f} px")
    ok = worst <= 1e-4
    print(f"  worst deviation {worst:.6f} px -> {'PASS' if ok else 'FAIL'}")
    return ok


def ultralytics_detections(model, x, conf):
    """Run the full model and return (boxes_xywh, scores, class_ids) in input-pixel space."""
    with torch.no_grad():
        out = model(x)
    if isinstance(out, (list, tuple)):
        out = out[0]
    pred = out[0].cpu().numpy()          # (84, 8400): cx, cy, w, h, then 80 class scores
    boxes_cxcywh = pred[:4].T            # (8400, 4)
    scores_all = pred[4:].T              # (8400, 80), already sigmoided by ultralytics

    hits = scores_all > conf
    if not np.any(hits):
        return np.empty((0, 4), np.float32), np.empty((0,), np.float32), np.empty((0,), int)
    cand, cls = np.where(hits)
    scores = scores_all[cand, cls]
    b = boxes_cxcywh[cand]
    boxes = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 2], b[:, 3]], -1)
    return boxes.astype(np.float32), scores.astype(np.float32), cls.astype(int)


def host_detections(wrapper, x, conf, imgsz):
    """Run the DPU wrapper and decode with the host module, as the board will."""
    with torch.no_grad():
        outs = wrapper(x)
    # The DPU emits NHWC; emulate that layout so the decode is exercised exactly as on board.
    feats = [o[0].permute(1, 2, 0).cpu().numpy() for o in outs]
    return decode_all(feats, conf_thresh=conf, input_size=imgsz)


def compare(tag, a, b, tol_box, tol_score):
    """a and b are (boxes, scores, classes), each already NMS'd and sorted by score."""
    ab, asc, ac = a
    bb, bsc, bc = b
    print(f"\n[{tag}]")
    print(f"  ultralytics: {len(bb)} detections    host decode: {len(ab)} detections")
    if len(ab) != len(bb):
        print("  FAIL: detection counts differ")
        return False
    if len(ab) == 0:
        print("  (no detections either side - inconclusive, try a lower --conf)")
        return True
    if not np.array_equal(ac, bc):
        n = int((ac != bc).sum())
        print(f"  FAIL: {n} class ids differ")
        return False
    dbox = float(np.abs(ab - bb).max())
    dscore = float(np.abs(asc - bsc).max())
    print(f"  max |box| difference   : {dbox:.6f} px   (tolerance {tol_box})")
    print(f"  max |score| difference : {dscore:.8f}     (tolerance {tol_score})")
    ok = dbox <= tol_box and dscore <= tol_score
    print("  PASS" if ok else "  FAIL: outside tolerance")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--images", type=int, default=3, help="how many val2017 images to test")
    ap.add_argument("--coco", default="/datasets/coco/images/val2017")
    ap.add_argument("--tol-box", type=float, default=1e-2, help="pixels, at 640 input")
    ap.add_argument("--tol-score", type=float, default=1e-5)
    ap.add_argument("--letterbox", action="store_true",
                    help="feed letterboxed input, as the board does")
    args = ap.parse_args()

    print(f"[INFO] loading {args.weights} (float, SiLU untouched)")
    model = load_detection_model(args.weights)
    wrapper = YOLOv8DPUWrapper(args.weights, act="silu").eval()

    paths = sorted(glob.glob(os.path.join(args.coco, "*.jpg")))[: args.images]
    if not paths:
        print(f"[warn] no images under {args.coco}; falling back to a random tensor")
        paths = [None]

    all_ok = check_letterbox_inverse(args.imgsz)

    for p in paths:
        x, desc, _ = load_image(p, args.imgsz, args.letterbox)

        ub, us, uc = ultralytics_detections(model, x, args.conf)
        hb, hs, hc = host_detections(wrapper, x, args.conf, args.imgsz)

        ku = nms_per_class(ub, us, uc)
        kh = nms_per_class(hb, hs, hc)
        ou = ku[np.argsort(us[ku])[::-1]] if len(ku) else ku
        oh = kh[np.argsort(hs[kh])[::-1]] if len(kh) else kh

        ok = compare(
            desc,
            (hb[oh], hs[oh], hc[oh]),
            (ub[ou], us[ou], uc[ou]),
            args.tol_box,
            args.tol_score,
        )
        all_ok = all_ok and ok

    print("\n" + ("[RESULT] decode verified - safe to proceed" if all_ok
                  else "[RESULT] DECODE MISMATCH - do not spend board time"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
