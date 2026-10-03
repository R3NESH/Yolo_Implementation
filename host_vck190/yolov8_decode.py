"""Host-side YOLOv8 decode + NMS for the VCK190 DPU outputs.

Pure NumPy - no torch, no VART - so it can be verified on a desktop against float PyTorch and
then copied to the board unchanged. Companion to yolo_decode.py, which handles YOLOv3.

WHY THIS IS A SEPARATE MODULE
-----------------------------
yolo_decode.py is anchor-based: it multiplies by anchor boxes, applies objectness, and uses the
YOLOv5 `(2*sigmoid(t))**2 * anchor` parameterisation. YOLOv8 shares none of that. It is
anchor-free with a Distribution Focal Loss head, so there are no anchors, no exp(), and no
objectness channel to multiply in. Reusing the v3 decode would silently produce plausible-looking
boxes with meaningless geometry.

TENSOR LAYOUT
-------------
The DPU emits NHWC, so each scale arrives as (H, W, 144) after dropping the batch axis, with
H == W == 640 / stride, i.e. 80, 40 and 20.

    channels   0..63   box distribution, laid out side-major: index = side * 16 + bin
                       sides are (left, top, right, bottom), 16 DFL bins each
    channels  64..143  class logits, one per COCO class. No objectness - v8 has none.

THE DECODE
----------
Mirrors ultralytics `DFL.forward` + `dist2bbox` + `make_anchors` exactly:

    softmax over each side's 16 bins, then expectation against bin indices 0..15
        -> ltrb, the distance from the cell centre to each box edge, in cell units
    anchor point is the cell centre, (col + 0.5, row + 0.5)     [make_anchors offset 0.5]

    x1 = (col + 0.5 - l) * stride      x2 = (col + 0.5 + r) * stride
    y1 = (row + 0.5 - t) * stride      y2 = (row + 0.5 + b) * stride
    score = sigmoid(class logit)

NMS, box clipping and preprocessing are shared with the v3 path and imported from yolo_decode,
so the two models are evaluated by identical post-processing below the decode itself.
"""

import numpy as np

from yolo_decode import (  # shared, model-agnostic post-processing
    COCO_CAT_IDS,
    clip_boxes,
    logit,
    nms_boxes,
    nms_per_class,
    sigmoid,
)

INPUT_SIZE = 640   # square input the v8 checkpoints were trained and quantized at
REG_MAX = 16       # DFL bins per box side
NUM_CLASSES = 80
NO = 4 * REG_MAX + NUM_CLASSES   # 144

# board_eval_vck190.py identifies a DPU output tensor by its channel count, which it computes
# as NA * NO - a YOLOv3 anchor-major layout. YOLOv8 is anchor-free, one prediction per cell, so
# NA is 1 and NO carries the full 144. Declared here so the shared harness works unchanged
# across both model families.
NA = 1

_BINS = np.arange(REG_MAX, dtype=np.float32)

# Per-grid cell-centre coordinates, shaped (g, g, 2) as (x, y) = (col + 0.5, row + 0.5).
_ANCHOR_POINTS = {}
for _g in (80, 40, 20):
    _row, _col = np.mgrid[0:_g, 0:_g]
    _ANCHOR_POINTS[_g] = np.stack((_col + 0.5, _row + 0.5), axis=-1).astype(np.float32)


def _anchor_points(g):
    """Cell centres for a g x g grid, (g, g, 2) as (x, y). Cached for the standard sizes."""
    if g not in _ANCHOR_POINTS:
        row, col = np.mgrid[0:g, 0:g]
        _ANCHOR_POINTS[g] = np.stack((col + 0.5, row + 0.5), axis=-1).astype(np.float32)
    return _ANCHOR_POINTS[g]


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def decode_grid(feature_map, conf_thresh=0.001, scale_x=1.0, scale_y=1.0, input_size=INPUT_SIZE):
    """Decodes one DPU output tensor into boxes/scores/class ids.

    Args:
        feature_map: (H, W, 144) float array, already dequantized. H == W == grid size.
        conf_thresh: minimum class confidence to keep.
        scale_x/scale_y: multiply box coords by these to map from the 640 input space back to
            the original image. For a plain resize these are orig_w/640 and orig_h/640.
        input_size: the square input the model was quantized at; sets the stride.

    Returns:
        (boxes_xywh, scores, class_ids) - boxes as [x1, y1, w, h] (COCO format).
    """
    g = feature_map.shape[0]
    if feature_map.shape[-1] != NO:
        raise ValueError("expected %d channels, got %d" % (NO, feature_map.shape[-1]))
    stride = input_size / float(g)

    cls_logits = feature_map[..., 4 * REG_MAX:]          # (g, g, 80)

    # Pre-filter in logit space. The DFL softmax is the expensive part of this decode, so only
    # cells with at least one class above threshold are worth reconstructing a box for.
    thresh = logit(conf_thresh)
    keep_cell = np.max(cls_logits, axis=-1) > thresh
    if not np.any(keep_cell):
        return np.empty((0, 4), np.float32), np.empty((0,), np.float32), np.empty((0,), int)

    active_cls = cls_logits[keep_cell]                   # (N, 80)
    active_box = feature_map[keep_cell][:, :4 * REG_MAX] # (N, 64)
    centres = _anchor_points(g)[keep_cell]               # (N, 2) as (x, y)

    scores_all = sigmoid(active_cls)
    hits = scores_all > conf_thresh
    if not np.any(hits):
        return np.empty((0, 4), np.float32), np.empty((0,), np.float32), np.empty((0,), int)

    cand_idx, class_ids = np.where(hits)
    scores = scores_all[cand_idx, class_ids]

    # --- DFL: softmax over each side's bins, then expectation --------------
    dist = softmax(active_box.reshape(-1, 4, REG_MAX), axis=-1) @ _BINS   # (N, 4) = l, t, r, b
    # -----------------------------------------------------------------------

    dist = dist[cand_idx]
    centres = centres[cand_idx]

    x1 = (centres[:, 0] - dist[:, 0]) * stride * scale_x
    y1 = (centres[:, 1] - dist[:, 1]) * stride * scale_y
    x2 = (centres[:, 0] + dist[:, 2]) * stride * scale_x
    y2 = (centres[:, 1] + dist[:, 3]) * stride * scale_y

    boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=-1).astype(np.float32)
    return boxes, scores.astype(np.float32), class_ids.astype(int)


def decode_all(feature_maps, conf_thresh=0.001, scale_x=1.0, scale_y=1.0, input_size=INPUT_SIZE):
    """Decodes every scale and concatenates. `feature_maps` is a list of (H, W, 144)."""
    boxes, scores, classes = [], [], []
    for fm in feature_maps:
        b, s, c = decode_grid(fm, conf_thresh, scale_x, scale_y, input_size)
        if len(b):
            boxes.append(b)
            scores.append(s)
            classes.append(c)
    if not boxes:
        return np.empty((0, 4), np.float32), np.empty((0,), np.float32), np.empty((0,), int)
    return np.concatenate(boxes), np.concatenate(scores), np.concatenate(classes)


def letterbox(img_bgr, input_size=INPUT_SIZE, color=(114, 114, 114), channels_first=False):
    """Aspect-preserving resize with centred padding. Returns (array, gain, (dw, dh)).

    The alternative, `preprocess()`, squashes the image to a square and distorts every object's
    aspect ratio. Letterbox is what ultralytics uses for validation, and what the prior work's
    board runner (`Versal_AI/src/runtime/dpu_runner.py`) used - measured to be worth several
    points of mAP over a plain resize, which is why it is the default here.

    Map boxes back with `unletterbox_boxes` using the returned gain and pad.
    """
    import cv2

    h0, w0 = img_bgr.shape[:2]
    gain = min(input_size / h0, input_size / w0)
    new_w, new_h = int(round(w0 * gain)), int(round(h0 * gain))
    dw, dh = (input_size - new_w) / 2.0, (input_size - new_h) / 2.0

    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=color)

    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    arr = np.ascontiguousarray(rgb.astype(np.float32) / 255.0)
    if channels_first:
        arr = np.transpose(arr, (2, 0, 1))
    # Return the INTEGER offsets actually added by copyMakeBorder, not the float dw/dh.
    # The content genuinely begins at pixel (left, top); subtracting the unrounded dw/dh in the
    # inverse leaves a systematic sub-pixel shift on every image whose padding lands on a half
    # pixel - about half of COCO. This matches ultralytics `scale_boxes`, which subtracts the
    # same `round(d - 0.1)`.
    return arr, gain, (left, top)


def unletterbox_boxes(boxes, gain, pad):
    """Map [x1, y1, w, h] boxes from letterboxed input space back to the original image.

    Undo the padding first, then the scale. Decode with scale_x=scale_y=1.0 so the boxes are
    still in input space when this is applied.

    `pad` must be the integer (left, top) that `letterbox` returns - the offsets actually added
    to the image. Using the unrounded dw/dh here shifts every box by up to half a pixel, which
    costs nothing at IoU 0.5 and measurably erodes AP at strict IoU.
    """
    if len(boxes) == 0:
        return boxes
    boxes[:, 0] = (boxes[:, 0] - pad[0]) / gain
    boxes[:, 1] = (boxes[:, 1] - pad[1]) / gain
    boxes[:, 2] /= gain
    boxes[:, 3] /= gain
    return boxes


def preprocess(img_bgr, input_size=INPUT_SIZE, channels_first=False):
    """Plain resize + BGR->RGB + /255, matching how the model is calibrated.

    Same caveat as the v3 path: this follows *calibration*, not training. v8_quantize.py
    calibrates with a plain resize, so inference must use a plain resize too. If calibration
    ever moves to letterbox, this and the box rescaling must move with it.
    """
    import cv2

    resized = cv2.resize(img_bgr, (input_size, input_size))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    arr = np.ascontiguousarray(rgb.astype(np.float32) / 255.0)
    if channels_first:
        arr = np.transpose(arr, (2, 0, 1))
    return arr
