"""Recover the class ordering a checkpoint was trained with.

Matches confident detections to ground-truth boxes ignoring class, then reads off which
true class each predicted index actually lands on. A clean permutation means the model is
sound and only its class indexing differs from the standard COCO-80 order.

A model that scores near zero while its boxes are visibly correct is the symptom this
diagnoses. Written for yolov3_original.pt, whose class list turned out to be the COCO-80
list with 'person' removed - every remaining index shifted down by one.

    ./vitis_run.sh python ../tools/recover_class_mapping.py \
        /data/annotations/instances_val2017.json \
        runs/val/<run>/<weights>_predictions.json utils/general.py [out.json]

Note that val.py writes raw model indices as category_id unless it recognises the dataset
as COCO, which it does not for a custom yaml - that is what this script expects.
"""
import json, sys, re, numpy as np
from collections import defaultdict, Counter

gt_path, dt_path, general_py = sys.argv[1], sys.argv[2], sys.argv[3]

src = open(general_py).read()
block = src.split("def coco80_to_coco91_class()", 1)[1].split("return [", 1)[1].split("]", 1)[0]
c80_to_c91 = [int(n) for n in re.findall(r"\d+", block)]
c91_to_c80 = {c: i for i, c in enumerate(c80_to_c91)}

gt = json.load(open(gt_path))
names91 = {c["id"]: c["name"] for c in gt["categories"]}
std_names = [names91[c] for c in c80_to_c91]

gt_by_img = defaultdict(list)
for a in gt["annotations"]:
    if a.get("iscrowd", 0):
        continue
    b = a["bbox"]
    gt_by_img[a["image_id"]].append(([b[0], b[1], b[0]+b[2], b[1]+b[3]], c91_to_c80[a["category_id"]]))

dt = [d for d in json.load(open(dt_path)) if d["score"] >= 0.25]
ids = {d["category_id"] for d in dt}
print(f"predicted category_id range: {min(ids)}..{max(ids)} ({len(ids)} distinct)")
dt.sort(key=lambda d: -d["score"])
print(f"using {len(dt):,} detections with score >= 0.25")

def iou(box, boxes):
    boxes = np.asarray(boxes, dtype=np.float64)
    lt = np.maximum(box[:2], boxes[:, :2]); rb = np.minimum(box[2:], boxes[:, 2:])
    wh = np.clip(rb - lt, 0, None); inter = wh[:, 0] * wh[:, 1]
    a = (box[2]-box[0])*(box[3]-box[1])
    b = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    return inter / (a + b - inter + 1e-12)

conf = defaultdict(Counter)
used = defaultdict(set)
for d in dt:
    pool = gt_by_img.get(d["image_id"])
    if not pool:
        continue
    boxes = [p[0] for p in pool]
    b = d["bbox"]
    ious = iou(np.array([b[0], b[1], b[0]+b[2], b[1]+b[3]]), boxes)
    for j in np.argsort(-ious):
        if ious[j] < 0.5:
            break
        if j not in used[d["image_id"]]:
            used[d["image_id"]].add(j)
            conf[d["category_id"]][pool[j][1]] += 1  # category_id is the raw model index here
            break

print(f"matched {sum(sum(c.values()) for c in conf.values()):,} detections to GT boxes\n")
print(f"{'model idx':>9}  {'std name at that idx':<18} -> {'actually detects':<18} {'purity':>7}  {'n':>5}")
mapping = {}
rows = []
for i in range(80):
    c = conf.get(i)
    if not c:
        rows.append((i, std_names[i], "(no matches)", 0.0, 0)); continue
    true, n = c.most_common(1)[0]
    tot = sum(c.values())
    mapping[i] = true
    rows.append((i, std_names[i], std_names[true], n / tot, tot))
for r in rows:
    print(f"{r[0]:>9}  {r[1]:<18} -> {r[2]:<18} {r[3]:>6.1%}  {r[4]:>5}")
print("")

clean = [r for r in rows if r[4] >= 5 and r[3] >= 0.5]
print(f"\nrows with >=5 matches and >=50% purity: {len(clean)} / 80")
identity = sum(1 for i, t in mapping.items() if i == t)
print(f"model idx == standard idx for {identity} / {len(mapping)} matched classes")
print(f"mapping is a permutation: {len(set(mapping.values())) == len(mapping)}")

alpha = sorted(std_names)
alpha_hits = sum(1 for i, t in mapping.items() if i < len(alpha) and alpha[i] == std_names[t])
print(f"consistent with ALPHABETICAL ordering for {alpha_hits} / {len(mapping)} classes")

shift1 = sum(1 for i, t in mapping.items() if t == i + 1)
print(f"model idx == standard idx + 1 for {shift1} / {len(mapping)} matched classes")
never = [n for k, n in enumerate(std_names) if k not in set(mapping.values())]
print("standard classes NEVER produced by the model:", never)

out = sys.argv[4] if len(sys.argv) > 4 else "recovered_mapping.json"
json.dump({str(k): v for k, v in mapping.items()}, open(out, "w"), indent=1)
print(f"\nwrote {out}")
