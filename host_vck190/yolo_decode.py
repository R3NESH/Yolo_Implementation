"""Correct host-side YOLO decode + NMS for the VCK190 DPU outputs.

Pure NumPy - no torch, no VART - so it can be unit-tested on a desktop and then
copied to the board unchanged.

WHY THIS FILE EXISTS
--------------------
The earlier `eval_yolo_vck190.py` in ~/Documents/Yolo_v3_AB decoded with the
*classic Darknet YOLOv3* formulas:

    cx = (sigmoid(tx) + col) * stride
    bw = exp(tw) * anchor

This model is YOLOv5-lineage and uses a different parameterisation
(`models/yolo.py` Detect.forward, with grid = meshgrid - 0.5):

    cx = (2*sigmoid(tx) - 0.5 + col) * stride
    bw = (2*sigmoid(tw))**2 * anchor

`exp()` is unbounded while `(2*sigmoid())**2` is capped at 4x the anchor, so the
old formula produced systematically wrong box sizes - and needed arbitrary
`clip(-3, 3)` guards to stop `exp` overflowing. Those clips were a symptom, not
a fix. This module implements the correct maths.

TENSOR LAYOUT
-------------
The DPU emits **NHWC**, so each output arrives as (H, W, 255) after dropping the
batch axis. 255 = 3 anchors x 85 (4 box + 1 obj + 80 cls), anchor-major, which
matches PyTorch's `view(bs, na, no, ny, nx)` channel ordering.
"""

import numpy as np

INPUT_SIZE = 416  # square input the model was trained and quantized at

# Pixel-space anchors, keyed by grid size. Verified against the checkpoint:
# stored grid-unit anchors x stride == these values (the stock YOLOv3 set).
ANCHORS = {
    52: np.array([[10, 13], [16, 30], [33, 23]], dtype=np.float32),    # stride 8
    26: np.array([[30, 61], [62, 45], [59, 119]], dtype=np.float32),   # stride 16
    13: np.array([[116, 90], [156, 198], [373, 326]], dtype=np.float32),  # stride 32
}

NUM_CLASSES = 80
NO = NUM_CLASSES + 5  # 85 outputs per anchor
NA = 3                # anchors per scale

# COCO's category ids are not 0..79 - they skip numbers. Index by class id.
COCO_CAT_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]

# Pre-computed per-scale grids and anchor maps, shaped (g, g, NA, 2).
_GRIDS, _ANCHOR_MAPS = {}, {}
for _g in ANCHORS:
    _row, _col = np.mgrid[0:_g, 0:_g]
    # index [row, col] -> (x=col, y=row), matching torch.stack((xv, yv), 2)
    _grid = np.stack((_col, _row), axis=-1).reshape(_g, _g, 1, 2).astype(np.float32)
    _GRIDS[_g] = np.broadcast_to(_grid, (_g, _g, NA, 2)).copy()
    _ANCHOR_MAPS[_g] = np.broadcast_to(
        ANCHORS[_g].reshape(1, 1, NA, 2), (_g, _g, NA, 2)
    ).copy()


def sigmoid(x):
    """Numerically stable elementwise logistic."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def logit(p):
    """Inverse of sigmoid; used to pre-threshold objectness without sigmoiding everything."""
    return -np.log(1.0 / p - 1.0)


def decode_grid(feature_map, conf_thresh=0.001, scale_x=1.0, scale_y=1.0):
    """Decodes one DPU output tensor into boxes/scores/class ids.

    Args:
        feature_map: (H, W, 255) float array, already dequantized. H == W == grid size.
        conf_thresh: minimum obj*cls confidence to keep.
        scale_x/scale_y: multiply box coords by these to map from the 416 input
            space back to the original image. For a plain resize these are
            orig_w/416 and orig_h/416. Leave at 1.0 to stay in 416 space.

    Returns:
        (boxes_xywh, scores, class_ids) - boxes as [x1, y1, w, h] (COCO format).
    """
    g = feature_map.shape[0]
    if g not in ANCHORS:
        raise ValueError("unexpected grid size %d; expected one of %s" % (g, sorted(ANCHORS)))

    pred = feature_map.reshape(g, g, NA, NO)
    stride = INPUT_SIZE / float(g)

    # Cheap pre-filter in logit space: skip the sigmoid for cells that cannot pass.
    obj_mask = pred[..., 4] > logit(conf_thresh)
    if not np.any(obj_mask):
        return np.empty((0, 4), np.float32), np.empty((0,), np.float32), np.empty((0,), int)

    active = pred[obj_mask]                     # (N, 85)
    grid = _GRIDS[g][obj_mask]                  # (N, 2) -> (x, y) cell index
    anchors = _ANCHOR_MAPS[g][obj_mask]         # (N, 2) -> (w, h) in pixels

    obj = sigmoid(active[:, 4])
    cls = sigmoid(active[:, 5:])
    scores_all = obj[:, None] * cls

    keep = scores_all > conf_thresh
    if not np.any(keep):
        return np.empty((0, 4), np.float32), np.empty((0,), np.float32), np.empty((0,), int)

    cand_idx, class_ids = np.where(keep)
    scores = scores_all[cand_idx, class_ids]

    active = active[cand_idx]
    grid = grid[cand_idx]
    anchors = anchors[cand_idx]

    # --- the corrected YOLOv5 decode -------------------------------------
    sxy = sigmoid(active[:, 0:2])
    swh = sigmoid(active[:, 2:4])

    cxy = (sxy * 2.0 - 0.5 + grid) * stride     # centre, 416 space
    wh = (swh * 2.0) ** 2 * anchors             # size, 416 space
    # ---------------------------------------------------------------------

    cxy[:, 0] *= scale_x
    cxy[:, 1] *= scale_y
    wh[:, 0] *= scale_x
    wh[:, 1] *= scale_y

    x1 = cxy[:, 0] - wh[:, 0] / 2.0
    y1 = cxy[:, 1] - wh[:, 1] / 2.0
    boxes = np.stack([x1, y1, wh[:, 0], wh[:, 1]], axis=-1).astype(np.float32)

    return boxes, scores.astype(np.float32), class_ids.astype(int)


def decode_all(feature_maps, conf_thresh=0.001, scale_x=1.0, scale_y=1.0):
    """Decodes every scale and concatenates. `feature_maps` is a list of (H, W, 255)."""
    boxes, scores, classes = [], [], []
    for fm in feature_maps:
        b, s, c = decode_grid(fm, conf_thresh, scale_x, scale_y)
        if len(b):
            boxes.append(b)
            scores.append(s)
            classes.append(c)
    if not boxes:
        return np.empty((0, 4), np.float32), np.empty((0,), np.float32), np.empty((0,), int)
    return np.concatenate(boxes), np.concatenate(scores), np.concatenate(classes)


def clip_boxes(boxes, width, height):
    """Clips [x1, y1, w, h] boxes to the image, in place. Mirrors utils.general.clip_boxes."""
    if len(boxes) == 0:
        return boxes
    x2 = np.clip(boxes[:, 0] + boxes[:, 2], 0, width)
    y2 = np.clip(boxes[:, 1] + boxes[:, 3], 0, height)
    boxes[:, 0] = np.clip(boxes[:, 0], 0, width)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, height)
    boxes[:, 2] = np.maximum(x2 - boxes[:, 0], 0)
    boxes[:, 3] = np.maximum(y2 - boxes[:, 1], 0)
    return boxes


def nms_boxes(boxes, scores, iou_thresh=0.45):
    """Greedy IoU NMS over [x1, y1, w, h] boxes. Returns kept indices."""
    if len(boxes) == 0:
        return []
    x1, y1, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x2, y2 = x1 + w, y1 + h
    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(ovr <= iou_thresh)[0] + 1]
    return keep


def nms_per_class(boxes, scores, class_ids, iou_thresh=0.45, max_det=300):
    """Class-wise NMS, matching how val.py evaluates (not class-agnostic)."""
    keep_all = []
    for c in np.unique(class_ids):
        idx = np.where(class_ids == c)[0]
        kept = nms_boxes(boxes[idx], scores[idx], iou_thresh)
        keep_all.extend(idx[k] for k in kept)
    keep_all = np.array(keep_all, dtype=int)
    if len(keep_all) > max_det:
        keep_all = keep_all[np.argsort(scores[keep_all])[::-1][:max_det]]
    return keep_all


def preprocess(img_bgr, input_size=INPUT_SIZE, channels_first=False):
    """Plain resize + BGR->RGB + /255, matching how the model was calibrated.

    NOTE: training used `letterbox` (aspect-preserving with padding), while
    quantize_vitis_AB4.py calibrated with a plain resize. This follows the
    calibration, so inference matches quantization. If the model is ever
    recalibrated with letterbox, change this to match and rescale boxes
    accordingly.
    """
    import cv2

    resized = cv2.resize(img_bgr, (input_size, input_size))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    arr = np.ascontiguousarray(rgb.astype(np.float32) / 255.0)
    if channels_first:
        arr = np.transpose(arr, (2, 0, 1))
    return arr
