"""Draw PR / P / R / F1 curves and a confusion matrix from board predictions.

`val.py` produces these plots automatically for a PC run. The board evaluation only
emits a COCO-format predictions JSON, so this rebuilds the equivalent plots from it -
for side-by-side comparison with `runs/val/exp14/`.

It does its own IoU matching at a single threshold rather than calling pycocotools,
which re-matches across 10 IoU thresholds x 80 classes x 4 area ranges and takes ~8
minutes. This is also closer to how Ultralytics draws its own curves, so the plots are
more directly comparable.

    ./vitis_run.sh python ../tools/board_curves.py \
        _tmp_eval/instances_val2017.json _tmp_eval/silu_preds.json \
        board_results/plots "best.pt INT8 on VCK190"
"""

import os
import sys
import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# NumPy renamed trapz -> trapezoid in 2.x and removed the old spelling. The container ships
# NumPy 1.x and the host venv 2.x, so this file has to run under both. Mirror image of the
# np.trapezoid crash in utils/metrics.py - see 'Bugs Found and Fixed' #10.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

IOU_PR = 0.50          # IoU for the precision/recall curves (matches mAP@0.5)
IOU_CM = 0.45          # IoU for the confusion matrix (Ultralytics' default)
CONF_CM = 0.25         # confidence for the confusion matrix (Ultralytics' default)
GRID = np.linspace(0, 1, 1000)


def xywh_to_xyxy(a):
    out = np.array(a, dtype=np.float64).reshape(-1, 4).copy()
    out[:, 2] += out[:, 0]
    out[:, 3] += out[:, 1]
    return out


def iou_one_to_many(box, boxes):
    """IoU of one xyxy box against an (N,4) array."""
    lt = np.maximum(box[:2], boxes[:, :2])
    rb = np.minimum(box[2:], boxes[:, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, 0] * wh[:, 1]
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (a + b - inter + 1e-12)


def average_precision(recall, precision):
    """COCO-style 101-point interpolated AP."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))       # monotonic envelope
    x = np.linspace(0, 1, 101)
    return float(_trapezoid(np.interp(x, mrec, mpre), x))


def load(gt_path, dt_path):
    gt = json.load(open(gt_path))
    cats = sorted(gt["categories"], key=lambda c: c["id"])
    cat_ids = [c["id"] for c in cats]
    names = [c["name"] for c in cats]
    cat_index = {c: i for i, c in enumerate(cat_ids)}

    gt_by_key = {}                       # (image_id, class) -> boxes
    n_gt = np.zeros(len(cat_ids), dtype=np.int64)
    for a in gt["annotations"]:
        if a.get("iscrowd", 0):
            continue
        k = (a["image_id"], cat_index[a["category_id"]])
        gt_by_key.setdefault(k, []).append(a["bbox"])
        n_gt[cat_index[a["category_id"]]] += 1
    for k in gt_by_key:
        gt_by_key[k] = xywh_to_xyxy(gt_by_key[k])

    dets = json.load(open(dt_path))
    return gt_by_key, n_gt, names, cat_index, dets


def match_per_class(gt_by_key, n_gt, cat_index, dets, n_classes):
    """Greedy highest-confidence-first matching, per class. Returns per-class (recall, precision)."""
    by_class = [[] for _ in range(n_classes)]
    for d in dets:
        by_class[cat_index[d["category_id"]]].append(d)

    curves = []
    for c in range(n_classes):
        dc = by_class[c]
        if not dc or n_gt[c] == 0:
            curves.append(None)
            continue
        dc.sort(key=lambda d: -d["score"])
        boxes = xywh_to_xyxy([d["bbox"] for d in dc])
        img_ids = [d["image_id"] for d in dc]

        used = {}                                  # image_id -> bool mask of consumed GT
        tp = np.zeros(len(dc), dtype=np.float64)
        for i in range(len(dc)):
            g = gt_by_key.get((img_ids[i], c))
            if g is None:
                continue
            mask = used.setdefault(img_ids[i], np.zeros(len(g), dtype=bool))
            ious = iou_one_to_many(boxes[i], g)
            ious[mask] = -1                        # already claimed by a higher-scoring det
            j = int(ious.argmax())
            if ious[j] >= IOU_PR:
                mask[j] = True
                tp[i] = 1.0

        tpc = np.cumsum(tp)
        fpc = np.cumsum(1.0 - tp)
        recall = tpc / n_gt[c]
        precision = tpc / np.maximum(tpc + fpc, 1e-12)
        conf = np.array([d["score"] for d in dc])
        curves.append((recall, precision, conf))
    return curves


def confusion_matrix(gt_by_key, cat_index, dets, n_classes):
    """Class-agnostic matching, so off-diagonal entries are real class confusions."""
    gt_by_img = {}
    for (img, cls), boxes in gt_by_key.items():
        gt_by_img.setdefault(img, [[], []])
        gt_by_img[img][0].append(boxes)
        gt_by_img[img][1].extend([cls] * len(boxes))

    dt_by_img = {}
    for d in dets:
        if d["score"] >= CONF_CM:
            dt_by_img.setdefault(d["image_id"], []).append(d)

    cm = np.zeros((n_classes + 1, n_classes + 1), dtype=np.int64)
    for img in set(list(gt_by_img.keys()) + list(dt_by_img.keys())):
        g = gt_by_img.get(img)
        gt_boxes = np.concatenate(g[0]) if g else np.zeros((0, 4))
        gt_cls = np.array(g[1], dtype=int) if g else np.zeros(0, dtype=int)

        dl = sorted(dt_by_img.get(img, []), key=lambda d: -d["score"])
        dt_boxes = xywh_to_xyxy([d["bbox"] for d in dl]) if dl else np.zeros((0, 4))
        dt_cls = np.array([cat_index[d["category_id"]] for d in dl], dtype=int)

        gt_used = np.zeros(len(gt_boxes), dtype=bool)
        for i in range(len(dt_boxes)):
            if len(gt_boxes) == 0:
                cm[dt_cls[i], n_classes] += 1
                continue
            ious = iou_one_to_many(dt_boxes[i], gt_boxes)
            ious[gt_used] = -1
            j = int(ious.argmax())
            if ious[j] >= IOU_CM:
                gt_used[j] = True
                cm[dt_cls[i], gt_cls[j]] += 1
            else:
                cm[dt_cls[i], n_classes] += 1        # false positive
        for j in np.nonzero(~gt_used)[0]:
            cm[n_classes, gt_cls[j]] += 1            # missed
    return cm


def line_plot(x, per_class, mean, xlabel, ylabel, legend, path, title):
    fig, ax = plt.subplots(figsize=(9, 6), tight_layout=True)
    for row in per_class:
        ax.plot(x, row, linewidth=0.5, color="grey")
    ax.plot(x, mean, linewidth=3, color="blue", label=legend)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(path, dpi=200); plt.close(fig)
    print("  wrote %s" % path)


def main(gt_path, dt_path, out_dir, title):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    print("loading...", flush=True)
    gt_by_key, n_gt, names, cat_index, dets = load(gt_path, dt_path)
    nc = len(names)
    print("  %d classes, %d ground-truth boxes, %d detections (%.1fs)"
          % (nc, n_gt.sum(), len(dets), time.time() - t0), flush=True)

    print("matching per class...", flush=True)
    curves = match_per_class(gt_by_key, n_gt, cat_index, dets, nc)
    print("  done (%.1fs)" % (time.time() - t0), flush=True)

    # --- interpolate every class onto common grids -----------------------
    rec_grid = np.linspace(0, 1, 101)
    P_rec, P_conf, R_conf, F_conf, aps = [], [], [], [], []
    for c, cur in enumerate(curves):
        if cur is None:
            continue
        recall, precision, conf = cur
        aps.append(average_precision(recall, precision))
        # precision as a function of recall (the PR curve)
        P_rec.append(np.interp(rec_grid, recall, precision, left=precision[0], right=0.0))
        # everything as a function of confidence: conf descends, so flip for np.interp
        cf, pr, rc = conf[::-1], precision[::-1], recall[::-1]
        p_i = np.interp(GRID, cf, pr, left=pr[0], right=pr[-1])
        r_i = np.interp(GRID, cf, rc, left=rc[0], right=0.0)
        P_conf.append(p_i)
        R_conf.append(r_i)
        F_conf.append(2 * p_i * r_i / (p_i + r_i + 1e-12))

    P_rec = np.array(P_rec); P_conf = np.array(P_conf)
    R_conf = np.array(R_conf); F_conf = np.array(F_conf)
    mAP = float(np.mean(aps))
    print("  mAP@%.2f = %.4f over %d classes" % (IOU_PR, mAP, len(aps)), flush=True)

    mF = F_conf.mean(0); best = int(mF.argmax())
    line_plot(rec_grid, P_rec, P_rec.mean(0), "Recall", "Precision",
              "all classes %.3f mAP@0.5" % mAP,
              os.path.join(out_dir, "PR_curve.png"), "Precision-Recall Curve - %s" % title)
    line_plot(GRID, F_conf, mF, "Confidence", "F1",
              "all classes %.2f at %.3f" % (mF[best], GRID[best]),
              os.path.join(out_dir, "F1_curve.png"), "F1-Confidence Curve - %s" % title)
    line_plot(GRID, P_conf, P_conf.mean(0), "Confidence", "Precision",
              "all classes %.2f at %.3f" % (P_conf.mean(0)[-1], GRID[-1]),
              os.path.join(out_dir, "P_curve.png"), "Precision-Confidence Curve - %s" % title)
    line_plot(GRID, R_conf, R_conf.mean(0), "Confidence", "Recall",
              "all classes %.2f at %.3f" % (R_conf.mean(0)[0], GRID[0]),
              os.path.join(out_dir, "R_curve.png"), "Recall-Confidence Curve - %s" % title)

    print("confusion matrix...", flush=True)
    cm = confusion_matrix(gt_by_key, cat_index, dets, nc)
    labels = names + ["background"]
    norm = cm.astype(np.float64) / (cm.sum(0, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(figsize=(12, 10), tight_layout=True)
    im = ax.imshow(norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xlabel("True"); ax.set_ylabel("Predicted")
    ax.set_title("Confusion Matrix (column-normalised) - %s" % title)
    ticks = list(range(0, len(labels), max(1, len(labels) // 40)))
    ax.set_xticks(ticks); ax.set_xticklabels([labels[i] for i in ticks], rotation=90, fontsize=6)
    ax.set_yticks(ticks); ax.set_yticklabels([labels[i] for i in ticks], fontsize=6)
    path = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=200); plt.close(fig)
    print("  wrote %s" % path)

    print("\nbest F1 %.3f at confidence %.3f  <- threshold to use for demos"
          % (mF[best], GRID[best]))
    print("total %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3],
         sys.argv[4] if len(sys.argv) > 4 else "board")
