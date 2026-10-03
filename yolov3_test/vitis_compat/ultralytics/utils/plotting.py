"""Minimal stand-in for `ultralytics.utils.plotting`.

Provides the three names this repo imports: Annotator, colors, save_one_box.
Visualization only - never called during quantization or DPU export.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


class Colors:
    """Ultralytics default palette; call with an index, `bgr=True` for cv2 order."""

    def __init__(self):
        hexs = (
            "FF3838", "FF9D97", "FF701F", "FFB21D", "CFD231", "48F90A", "92CC17",
            "3DDB86", "1A9334", "00D4BB", "2C99A8", "00C2FF", "344593", "6473FF",
            "0018EC", "8438FF", "520085", "CB38FF", "FF95C8", "FF37C7",
        )
        self.palette = [self.hex2rgb(h) for h in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        """Returns the i-th palette color (wrapping), as BGR if requested."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts a 6-digit hex string to an (R, G, B) tuple."""
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # instance, matching 'from ultralytics.utils.plotting import colors'


class Annotator:
    """cv2-backed subset of the upstream Annotator: box_label / rectangle / text / result."""

    def __init__(self, im, line_width=None, font_size=None, font=None, pil=False, example="abc"):
        """Wraps an image (PIL or ndarray) for in-place annotation."""
        if isinstance(im, Image.Image):
            im = np.asarray(im)
        self.im = np.ascontiguousarray(im)
        self.lw = line_width or max(round(sum(self.im.shape[:2]) / 2 * 0.003), 2)
        self.sf = self.lw / 3  # font scale
        self.tf = max(self.lw - 1, 1)  # font thickness

    def box_label(self, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255)):
        """Draws one xyxy box with an optional filled label above it."""
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(self.im, p1, p2, color, self.lw, cv2.LINE_AA)
        if label:
            w, h = cv2.getTextSize(label, 0, self.sf, self.tf)[0]
            outside = p1[1] - h >= 3
            bg = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(self.im, p1, bg, color, -1, cv2.LINE_AA)
            y = p1[1] - 2 if outside else p1[1] + h + 2
            cv2.putText(self.im, label, (p1[0], y), 0, self.sf, txt_color, self.tf, cv2.LINE_AA)

    def rectangle(self, xy, fill=None, outline=None, width=1):
        """Draws an unlabeled xyxy rectangle."""
        cv2.rectangle(
            self.im, (int(xy[0]), int(xy[1])), (int(xy[2]), int(xy[3])),
            outline or (255, 255, 255), width, cv2.LINE_AA
        )

    def text(self, xy, text, txt_color=(255, 255, 255), anchor="top", box_color=()):
        """Draws text at xy."""
        cv2.putText(
            self.im, str(text), (int(xy[0]), int(xy[1])), 0,
            self.sf, txt_color, self.tf, cv2.LINE_AA
        )

    def masks(self, *args, **kwargs):
        """Segmentation overlay - unused in the detection/DPU path."""

    def fromarray(self, im):
        """Replaces the wrapped image."""
        self.im = np.ascontiguousarray(np.asarray(im))

    def result(self):
        """Returns the annotated image as an ndarray."""
        return np.asarray(self.im)


def save_one_box(xyxy, im, file=Path("im.jpg"), gain=1.02, pad=10, square=False, BGR=False, save=True):
    """Crops the xyxy box out of `im` (with gain/pad margin), optionally writing it to disk."""
    b = torch.tensor(xyxy).view(-1, 4).float()
    if square:
        wh = b[:, 2:] - b[:, :2]
        b[:, 2:] = b[:, :2] + wh.max(1, keepdim=True)[0]
    center = (b[:, :2] + b[:, 2:]) / 2
    wh = (b[:, 2:] - b[:, :2]) * gain
    b = torch.cat((center - wh / 2, center + wh / 2), 1).long()
    x1, y1, x2, y2 = b[0].tolist()
    x1, y1 = max(x1 - pad, 0), max(y1 - pad, 0)
    crop = im[y1:y2 + pad, x1:x2 + pad, :: (1 if BGR else -1)]
    if save:
        f = Path(file).with_suffix(".jpg")
        f.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(f), crop)
    return crop
