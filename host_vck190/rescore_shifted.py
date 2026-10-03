"""Rescore board predictions in yolov3_original.pt's own class space.

The checkpoint was trained on the COCO-80 list with 'person' removed, so model index i
corresponds to standard COCO-80 index i+1. board_eval_vck190.py writes category_id with the
standard table, which shifts every label by one class. This remaps the saved predictions and
rescores - no re-inference needed.

Reports two numbers:
  mAP80 - all 80 categories, person necessarily scoring AP 0 (comparable to exp2's figure)
  mAP79 - the 79 categories this checkpoint can actually predict
"""
import json, sys, contextlib, io

COCO_CAT_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]
c91_to_c80 = {c: i for i, c in enumerate(COCO_CAT_IDS)}

pred_in, pred_out, ann_path = sys.argv[1], sys.argv[2], sys.argv[3]

dt = json.load(open(pred_in))
out, dropped = [], 0
for d in dt:
    i = c91_to_c80[d["category_id"]]
    if i >= 79:                      # the model's dead 80th output
        dropped += 1
        continue
    d = dict(d)
    d["category_id"] = COCO_CAT_IDS[i + 1]
    out.append(d)
json.dump(out, open(pred_out, "w"))
print(f"remapped {len(out):,} detections (+1 class shift); dropped {dropped:,} from the dead index 79")

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

with contextlib.redirect_stdout(io.StringIO()):
    anno = COCO(ann_path)
    pred = anno.loadRes(pred_out)

for tag, cats in (("mAP80 (all categories, person unreachable)", None),
                  ("mAP79 (categories the model can predict)", COCO_CAT_IDS[1:])):
    e = COCOeval(anno, pred, "bbox")
    if cats is not None:
        e.params.catIds = cats
    with contextlib.redirect_stdout(io.StringIO()):
        e.evaluate(); e.accumulate(); e.summarize()
    print(f"\n{tag}")
    print(f"  mAP@0.50      : {e.stats[1]:.4f}")
    print(f"  mAP@0.50:0.95 : {e.stats[0]:.4f}")
    print(f"  mAP@0.75      : {e.stats[2]:.4f}")
    print(f"  AR@100        : {e.stats[8]:.4f}")
