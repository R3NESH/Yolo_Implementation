"""instances_val2017.json -> Ultralytics YOLO .txt labels (class cx cy w h, normalised).

The repo ships only coco128, so a val.py float baseline needs val2017 labels built from
the COCO annotations. Writes <root>/labels/val2017/*.txt beside <root>/images/val2017,
which is the layout Ultralytics derives label paths from.

    python3 tools/coco_to_yolo_labels.py /path/to/coco_val2017

Expect 36,335 boxes over 5000 files - COCO val2017's published non-crowd count.
"""
import json, re, sys, os
from collections import defaultdict

repo = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "yolov3_test/utils/general.py")
src = open(repo).read()
block = src.split("def coco80_to_coco91_class()", 1)[1].split("return [", 1)[1].split("]", 1)[0]
c80_to_c91 = [int(n) for n in re.findall(r"\d+", block)]
assert len(c80_to_c91) == 80, len(c80_to_c91)
c91_to_c80 = {c: i for i, c in enumerate(c80_to_c91)}

root = sys.argv[1]
ann = json.load(open(os.path.join(root, "annotations/instances_val2017.json")))
imgs = {im["id"]: im for im in ann["images"]}
per_img = defaultdict(list)
skipped_crowd = skipped_cls = 0

for a in ann["annotations"]:
    if a.get("iscrowd", 0):
        skipped_crowd += 1
        continue
    cls = c91_to_c80.get(a["category_id"])
    if cls is None:
        skipped_cls += 1
        continue
    im = imgs[a["image_id"]]
    w, h = im["width"], im["height"]
    x, y, bw, bh = a["bbox"]                      # COCO: top-left x,y + w,h in pixels
    cx, cy = (x + bw / 2) / w, (y + bh / 2) / h   # YOLO: centre, normalised
    bw, bh = bw / w, bh / h
    if bw <= 0 or bh <= 0:
        continue
    per_img[a["image_id"]].append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

out = os.path.join(root, "labels/val2017")
os.makedirs(out, exist_ok=True)
written = 0
for iid, im in imgs.items():                      # every image gets a file, empty if no objects
    stem = os.path.splitext(im["file_name"])[0]
    with open(os.path.join(out, stem + ".txt"), "w") as f:
        f.write("\n".join(per_img.get(iid, [])))
        if per_img.get(iid):
            f.write("\n")
    written += 1

print(f"images: {len(imgs)}  label files: {written}  "
      f"boxes: {sum(len(v) for v in per_img.values())}  "
      f"skipped iscrowd: {skipped_crowd}  unmapped class: {skipped_cls}")
