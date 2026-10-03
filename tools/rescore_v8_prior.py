"""Rescore every saved YOLOv8 board-prediction file under identical conditions.

[[YOLOv8 Prior Work in Versal_AI]] records five generations of board mAP that disagree with
each other, one scoring above float. Every predictions JSON was kept, so the disagreement can
be settled offline with no board time.

The generations are not directly comparable as they stand: they differ in how many images they
cover and in whether a confidence threshold was applied before scoring. Both differences move
mAP on their own, independently of anything the DPU did. So for each file this reports

  mAP(all)    - scored over all 5000 val images; images the file never covered count as pure
                misses, which is the honest number if the run was meant to be a full sweep
  mAP(subset) - scored only over the images the file actually covers, which isolates detection
                quality from run coverage

together with the diagnostics that explain any gap: image coverage, detection count, and the
minimum score present (a floor well above ~0.001 means the predictions were thresholded before
scoring, which caps recall and therefore AP).

Usage:
    python tools/rescore_v8_prior.py <annotations.json> <predictions.json> [...]
"""
import contextlib
import io
import json
import sys

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def score(anno, pred_path, img_ids=None):
    with contextlib.redirect_stdout(io.StringIO()):
        pred = anno.loadRes(pred_path)
        e = COCOeval(anno, pred, "bbox")
        if img_ids is not None:
            e.params.imgIds = sorted(img_ids)
        e.evaluate()
        e.accumulate()
        e.summarize()
    return e.stats


def main():
    ann_path, pred_paths = sys.argv[1], sys.argv[2:]
    with contextlib.redirect_stdout(io.StringIO()):
        anno = COCO(ann_path)
    all_imgs = set(anno.getImgIds())

    hdr = f"{'file':<46} {'imgs':>6} {'dets':>9} {'minscore':>9} {'mAP50(all)':>11} {'mAP50(sub)':>11} {'mAP(all)':>9} {'mAP(sub)':>9}"
    print(hdr)
    print("-" * len(hdr))

    for p in pred_paths:
        dt = json.load(open(p))
        if not dt:
            print(f"{p.split('/')[-1]:<46} {'empty':>6}")
            continue
        imgs = {d["image_id"] for d in dt}
        lo = min(d["score"] for d in dt)

        s_all = score(anno, p)
        s_sub = score(anno, p, imgs) if imgs != all_imgs else s_all

        name = "/".join(p.split("/")[-2:])
        print(f"{name:<46} {len(imgs):>6} {len(dt):>9,} {lo:>9.4f} "
              f"{s_all[1]:>11.4f} {s_sub[1]:>11.4f} {s_all[0]:>9.4f} {s_sub[0]:>9.4f}")


if __name__ == "__main__":
    main()
