"""Run a compiled YOLO xmodel on a VCK190 and score it against COCO.

Runs ON THE BOARD (needs `xir`, `vart`, and a Vitis AI Petalinux image).

Handles both model families; pick with `--arch`:

    --arch v3   (default) anchor-based YOLOv3 at 416, decoded by `yolo_decode.py`,
                verified against the model's PyTorch Detect head to ~1e-4 (`verify_decode.py`)
    --arch v8   anchor-free YOLOv8 with a DFL head at 640, decoded by `yolov8_decode.py`,
                verified against ultralytics' own decode to ~6e-5 px
                (`Yolo_v8_Versal_Implementation/verify_v8_decode.py`)

Everything below the decode - VART plumbing, NMS, COCO scoring, CSV output - is shared, so the
two families are measured by identical post-processing.

    python3 board_eval_vck190.py \
        --model yolov3_vck190_lrelu.xmodel \
        --images val2017 \
        --annotations instances_val2017.json

Quick smoke test on 50 images, no scoring:

    python3 board_eval_vck190.py --model m.xmodel --images val2017 --limit 50 --no-eval

Throughput only:

    python3 board_eval_vck190.py --model m.xmodel --images val2017 --benchmark 200

Derived from the July 2026 `eval_yolo_vck190.py`, whose VART plumbing and COCO
harness were sound but whose decode used the wrong (Darknet) box formulas.
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The decode module is chosen by --arch and bound in main(). Every use below is inside a
# function, so binding it as a global before those run is sufficient.
yd = None


def select_decode(arch):
    """Bind `yd` to the decode module for this model family."""
    global yd
    import importlib

    yd = importlib.import_module("yolov8_decode" if arch == "v8" else "yolo_decode")
    return yd


# The runner holds raw pointers into the xir.Graph. If the Graph is collected the
# runner is left dangling and execute_async segfaults - intermittently, because it
# depends on when the GC runs. Anchoring it here keeps it alive for the process.
_GRAPH_KEEPALIVE = []


def build_graph_runner(model_path):
    """Whole-graph runner, for models the compiler could not fit into one DPU subgraph.

    `vart.Runner` executes exactly one subgraph, so on a fragmented model it computes a
    fraction of the network and the decode gets nonsense. `GraphRunner` walks every subgraph -
    DPU and CPU alike - in dependency order.

    Two differences from the `vart.Runner` path:
      * it owns its buffers (`get_inputs()` / `get_outputs()`), which are exposed as numpy
        arrays through the buffer protocol rather than allocated by the caller;
      * its outputs arrive **already dequantized to float32**, because the graph carries the
        `fix2float` ops, so no output scaling is needed.

    The input is still int8 and still needs the `2**fix_point` scaling.
    """
    import xir
    from vitis_ai_library import GraphRunner

    graph = xir.Graph.deserialize(model_path)
    _GRAPH_KEEPALIVE.append(graph)
    return GraphRunner.create_graph_runner(graph)


def count_dpu_subgraphs(model_path):
    """How many DPU subgraphs the compiler produced - decides which runner to use."""
    import xir

    graph = xir.Graph.deserialize(model_path)
    root = graph.get_root_subgraph()
    return sum(1 for s in root.get_children()
               if s.has_attr("device") and s.get_attr("device").upper() == "DPU")


def build_runner(model_path):
    """Deserialises the xmodel and creates a VART runner over its DPU subgraph."""
    import vart
    import xir

    graph = xir.Graph.deserialize(model_path)
    _GRAPH_KEEPALIVE.append(graph)
    root = graph.get_root_subgraph()
    dpu_subgraphs = [
        s for s in root.get_children()
        if s.has_attr("device") and s.get_attr("device").upper() == "DPU"
    ]
    if not dpu_subgraphs:
        sys.exit("no DPU subgraph in %s - was it compiled with vai_c_xir?" % model_path)
    return vart.Runner.create_runner(dpu_subgraphs[0], "run")


def output_to_nhwc(arr):
    """Normalises one output buffer to (H, W, C), accepting NHWC or NCHW.

    C is 255 for YOLOv3 (3 anchors x 85) and 144 for YOLOv8 (4x16 DFL bins + 80 classes).
    """
    a = arr[0] if arr.ndim == 4 else arr
    channels = yd.NA * yd.NO
    if a.shape[-1] == channels:               # (H, W, C) - what the DPU emits
        return a
    if a.shape[0] == channels:                # (C, H, W)
        return a.transpose(1, 2, 0)
    raise ValueError("unrecognised output shape %s (expected %d channels)"
                     % (a.shape, channels))


def dequant_scale(tensor):
    """INT8 -> float scale from the tensor's fix_point attribute."""
    fixpos = tensor.get_attr("fix_point") if tensor.has_attr("fix_point") else 0
    return 1.0 / (2 ** fixpos)


def quant_scale(tensor):
    """float -> INT8 scale. The DPU input tensor is xint8, not float."""
    fixpos = tensor.get_attr("fix_point") if tensor.has_attr("fix_point") else 0
    return float(2 ** fixpos)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="compiled .xmodel")
    ap.add_argument("--images", required=True, help="directory of validation images")
    ap.add_argument("--annotations", help="COCO instances_val2017.json (required unless --no-eval)")
    ap.add_argument("--out-json", default="vck190_predictions.json")
    ap.add_argument("--out-csv", default="vck190_metrics.csv")
    ap.add_argument("--conf-thres", type=float, default=0.001,
                    help="0.001 for mAP (default), ~0.25 for demos")
    ap.add_argument("--iou-thres", type=float, default=0.45)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0, help="only process N images")
    ap.add_argument("--benchmark", type=int, default=0,
                    help="time N inferences and exit (no scoring)")
    ap.add_argument("--no-eval", action="store_true", help="run inference, skip COCO scoring")
    ap.add_argument("--class-offset", type=int, default=0,
                    help="added to every predicted class index before it is mapped to a COCO "
                         "category id. yolov3_original.pt was trained on the COCO-80 list with "
                         "'person' removed, so its index i means standard class i+1: use 1. "
                         "Leave at 0 for models trained on the standard 80 classes.")
    ap.add_argument("--arch", choices=["v3", "v8"], default="v3",
                    help="model family: v3 = anchor-based YOLOv3 at 416 (default); "
                         "v8 = anchor-free YOLOv8 with a DFL head at 640")
    ap.add_argument("--letterbox", dest="letterbox", action="store_true", default=None,
                    help="aspect-preserving resize with padding. Default ON for --arch v8, where "
                         "it is worth ~0.047 mAP; unavailable for v3, whose decode module has no "
                         "letterbox and whose models were calibrated without it.")
    ap.add_argument("--no-letterbox", dest="letterbox", action="store_false",
                    help="force the plain square resize")
    ap.add_argument("--runner", choices=["auto", "dpu", "graph"], default="auto",
                    help="auto (default) picks graph for fragmented models; dpu = vart.Runner "
                         "on one subgraph; graph = GraphRunner over the whole graph")
    args = ap.parse_args()
    select_decode(args.arch)
    if args.letterbox is None:
        args.letterbox = args.arch == "v8"
    if args.letterbox and not hasattr(yd, "letterbox"):
        sys.exit("--letterbox is not supported by %s" % yd.__name__)
    print("[INFO] arch %s, decode %s, input %d, preprocessing %s"
          % (args.arch, yd.__name__, yd.INPUT_SIZE,
             "letterbox" if args.letterbox else "plain resize"))

    import cv2

    n_dpu = count_dpu_subgraphs(args.model)
    mode = args.runner
    if mode == "auto":
        mode = "dpu" if n_dpu <= 1 else "graph"
    print("model has %d DPU subgraph(s); using the '%s' runner" % (n_dpu, mode))
    if mode == "dpu" and n_dpu > 1:
        print("WARNING: 'dpu' runner on a %d-subgraph model executes only the first "
              "subgraph. Results will be wrong. Use --runner graph." % n_dpu)

    if mode == "graph":
        # GraphRunner owns its buffers and hands them back as numpy views. Its outputs
        # are already float32 - the graph carries fix2float - so no output scaling.
        runner = build_graph_runner(args.model)
        in_buf = runner.get_inputs()
        out_buf = runner.get_outputs()
        in_tensors = [b.get_tensor() for b in in_buf]
        out_tensors = [b.get_tensor() for b in out_buf]
        in_view = np.asarray(in_buf[0])
        out_views = [np.asarray(b) for b in out_buf]
        scales = [1.0] * len(out_buf)
    else:
        runner = build_runner(args.model)
        in_tensors = runner.get_input_tensors()
        out_tensors = runner.get_output_tensors()
        # VART buffers must match the tensor data type. These are xint8 - allocating
        # float32 here makes VART write 4x past the end of the buffer and segfault.
        in_buf = [np.empty(tuple(t.dims), dtype=np.int8, order="C") for t in in_tensors]
        out_buf = [np.empty(tuple(t.dims), dtype=np.int8, order="C") for t in out_tensors]
        in_view = in_buf[0]
        out_views = out_buf
        scales = [dequant_scale(t) for t in out_tensors]

    in_scale = quant_scale(in_tensors[0])
    batch = in_tensors[0].dims[0]

    channels_first = in_tensors[0].dims[1] == 3
    use_letterbox = args.letterbox
    print("input %s (%s)  outputs %s"
          % (tuple(in_tensors[0].dims), "NCHW" if channels_first else "NHWC",
             [tuple(t.dims) for t in out_tensors]))
    print("input fix_point scale %g, output scales %s, batch %d (all slots used)"
          % (in_scale, scales, batch))

    def infer_batch(imgs):
        """Run up to `batch` frames in one call.

        The input tensor is (batch, 416, 416, 3) - this hardware reports batch 6.
        Filling one slot per call wastes five sixths of each invocation, so fill as
        many as we were given and slice the results back apart.
        """
        n = len(imgs)
        if n > batch:
            raise ValueError("%d images for a batch of %d" % (n, batch))
        geom = []
        for k, img_bgr in enumerate(imgs):
            if use_letterbox:
                arr, gain, pad = yd.letterbox(img_bgr, yd.INPUT_SIZE, channels_first=channels_first)
            else:
                arr, gain, pad = yd.preprocess(img_bgr, yd.INPUT_SIZE, channels_first), None, None
            geom.append((gain, pad))
            np.copyto(in_view[k],
                      np.clip(np.round(arr * in_scale), -128, 127).astype(np.int8))
        job = runner.execute_async(in_buf, out_buf)
        runner.wait(job)
        feats = [[output_to_nhwc(out_views[i][k]).astype(np.float32) * scales[i]
                  for i in range(len(out_views))]
                 for k in range(n)]
        return feats, geom

    def infer(img_bgr):
        """One frame, for benchmark mode and callers that want a single result."""
        return infer_batch([img_bgr])[0][0]

    # ---- benchmark mode -------------------------------------------------
    if args.benchmark:
        names = sorted(os.listdir(args.images))[: max(1, args.benchmark)]
        frames = [cv2.imread(os.path.join(args.images, n)) for n in names]
        frames = [f for f in frames if f is not None]
        if not frames:
            sys.exit("no readable images in %s" % args.images)
        infer(frames[0])                                  # warm up
        t0 = time.time()
        n = 0
        while n < args.benchmark:
            group = [frames[(n + j) % len(frames)]
                     for j in range(min(batch, args.benchmark - n))]
            infer_batch(group)
            n += len(group)
        dt = time.time() - t0
        print("\n%d inferences in %.2f s  ->  %.1f FPS  (%.2f ms/frame, DPU+preprocess, "
              "batch %d)" % (n, dt, n / dt, 1000 * dt / n, batch))
        print("NOTE: excludes decode/NMS. Profile those separately - on a fast DPU they "
              "can dominate end-to-end latency.")
        return 0

    # ---- evaluation mode ------------------------------------------------
    if args.no_eval:
        img_files = sorted(os.listdir(args.images))
        if args.limit:
            img_files = img_files[: args.limit]
        entries = [(None, f) for f in img_files]
        coco_gt = None
    else:
        if not args.annotations:
            sys.exit("--annotations is required unless --no-eval")
        from pycocotools.coco import COCO

        coco_gt = COCO(args.annotations)
        ids = coco_gt.getImgIds()
        if args.limit:
            ids = ids[: args.limit]
        entries = [(i, coco_gt.loadImgs(i)[0]["file_name"]) for i in ids]

    try:
        from tqdm import tqdm
    except ImportError:
        class tqdm(object):            # minimal stand-in: total/update/close only
            def __init__(self, total=0, **k):
                self.total = total

            def update(self, n=1):
                pass

            def close(self):
                pass

    results = []
    kept_total = 0
    pbar = tqdm(total=len(entries))
    for start in range(0, len(entries), batch):
        group = entries[start:start + batch]

        # Load the group first; the DPU call wants every slot ready at once.
        loaded = []
        for img_id, fname in group:
            img = cv2.imread(os.path.join(args.images, fname))
            if img is None:
                pbar.update(1)
                continue
            loaded.append((img_id, img))
        if not loaded:
            continue

        feats_per_image, geom_per_image = infer_batch([img for _, img in loaded])

        for (img_id, img), feats, (gain, pad) in zip(loaded, feats_per_image, geom_per_image):
            orig_h, orig_w = img.shape[:2]

            # Decode keys anchors off grid size, so output order does not matter.
            # With letterbox, stay in input space and undo gain/pad afterwards; with a plain
            # resize the per-axis scale folds into the decode directly.
            boxes, scores, class_ids = yd.decode_all(
                feats,
                conf_thresh=args.conf_thres,
                scale_x=1.0 if use_letterbox else orig_w / float(yd.INPUT_SIZE),
                scale_y=1.0 if use_letterbox else orig_h / float(yd.INPUT_SIZE),
            )
            pbar.update(1)
            if len(boxes) == 0:
                continue

            if use_letterbox:
                yd.unletterbox_boxes(boxes, gain, pad)
            yd.clip_boxes(boxes, orig_w, orig_h)
            keep = yd.nms_per_class(boxes, scores, class_ids, args.iou_thres, args.max_det)
            kept_total += len(keep)

            if coco_gt is not None:
                for k in keep:
                    c = int(class_ids[k]) + args.class_offset
                    if c < 0 or c >= len(yd.COCO_CAT_IDS):
                        continue
                    results.append({
                        "image_id": int(img_id),
                        "category_id": yd.COCO_CAT_IDS[c],
                        "bbox": [round(float(v), 2) for v in boxes[k]],
                        "score": round(float(scores[k]), 5),
                    })
    pbar.close()

    print("\nimages processed: %d | detections kept: %d" % (len(entries), kept_total))

    if coco_gt is None:
        print("--no-eval given; skipping scoring.")
        return 0

    if not results:
        print("WARNING: 0 predictions. Check preprocessing, dequant scales and thresholds "
              "before blaming the model.")
        return 1

    with open(args.out_json, "w") as f:
        json.dump(results, f)
    print("predictions -> %s" % args.out_json)

    from pycocotools.cocoeval import COCOeval

    coco_dt = coco_gt.loadRes(args.out_json)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    if args.limit:
        ev.params.imgIds = [e[0] for e in entries]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    map_50_95, map_50, map_75 = ev.stats[0], ev.stats[1], ev.stats[2]
    recall_100 = ev.stats[8]

    print("\n" + "-" * 44)
    print(" mAP@0.50       : %.4f" % map_50)
    print(" mAP@0.50:0.95  : %.4f" % map_50_95)
    print(" mAP@0.75       : %.4f" % map_75)
    print(" Recall (AR@100): %.4f" % recall_100)
    print("-" * 44)
    # Float baselines are per-model; printing the YOLOv3 one against a YOLOv8 run invites
    # exactly the kind of mismatched comparison the vault note 'YOLOv8 Prior Numbers
    # Reconciled' was written to stop.
    if args.arch == "v8":
        stem = os.path.basename(args.model).split("_")[0]
        v8_float = {"yolov8n": (0.5187, 0.3681), "yolov8s": (0.6106, 0.4440),
                    "yolov8m": (0.6654, 0.4979), "yolov8l": (0.6916, 0.5244)}
        if stem in v8_float:
            f50, f5095 = v8_float[stem]
            print(" float baseline (%s, val2017 @640): mAP@0.5 %.4f / mAP@0.5:0.95 %.4f"
                  % (stem, f50, f5095))
            print(" retained: %.1f%% / %.1f%%  (see 'YOLOv8 Letterbox and the Activation Cost')"
                  % (100 * ev.stats[1] / f50, 100 * ev.stats[0] / f5095))
        else:
            print(" no float baseline on record for '%s'" % stem)
    else:
        print(" PC float baseline for comparison: mAP@0.5 0.5275 / mAP@0.5:0.95 0.3228")
        print(" (from runs/train/exp2 - see the vault note 'Training Run exp2')")

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        w.writerow(["mAP@0.50", "%.6f" % map_50])
        w.writerow(["mAP@0.50:0.95", "%.6f" % map_50_95])
        w.writerow(["mAP@0.75", "%.6f" % map_75])
        w.writerow(["Recall (AR@100)", "%.6f" % recall_100])
        w.writerow(["Predictions", len(results)])
        w.writerow(["Images", len(entries)])
    print("metrics -> %s" % args.out_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
