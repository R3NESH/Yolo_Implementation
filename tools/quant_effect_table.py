"""Separate the cost of the activation swap from the cost of INT8, per model.

Three runs per model on one fixed image subset, changing one thing at a time:

    float SiLU  ->  float HardSwish  ->  INT8 HardSwish
                |                    |
                activation           quantization

Writes board_results/metrics/quantization_effect.csv and the figure the report uses.

    Versal_AI/venv/bin/python tools/quant_effect_table.py
"""

import contextlib
import csv
import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ANN = "/home/aesicdab/Desktop/Versal_AI/datasets/coco/annotations/instances_val2017.json"
SIM = "Yolo_v8_Versal_Implementation/sim_results"
OUTC = "Yolo_v8_Versal_Implementation/board_results/metrics/quantization_effect.csv"
OUTF = "Yolo_v8_Versal_Implementation/report/figures/quantization_effect.png"
MODELS = ["yolov8n", "yolov8s", "yolov8m", "yolov8l"]
BLUE, ORANGE, REF = "#2a78d6", "#eb6834", "#c9c7c1"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"

CONFIGS = {
    "float_silu": "preds_{m}_silu_float_lb.json",
    "float_hardswish": "preds_{m}_hardswish_float_lb.json",
    "int8_hardswish": "preds_{m}_hardswish_int8_lb.json",
}


def score(anno, path, img_ids):
    with contextlib.redirect_stdout(io.StringIO()):
        pred = anno.loadRes(path)
        e = COCOeval(anno, pred, "bbox")
        e.params.imgIds = sorted(img_ids)
        e.evaluate(); e.accumulate(); e.summarize()
    return float(e.stats[1])          # mAP@0.5


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        anno = COCO(ANN)

    have, missing = {}, []
    for m in MODELS:
        paths = {k: os.path.join(SIM, t.format(m=m)) for k, t in CONFIGS.items()}
        if all(os.path.exists(p) for p in paths.values()):
            have[m] = paths
        else:
            missing.append(m)
    if missing:
        print("missing runs for:", ", ".join(missing), file=sys.stderr)
    if not have:
        sys.exit("no complete model found")

    # One common subset across every file, so the columns are comparable.
    ids = None
    for paths in have.values():
        for p in paths.values():
            s = {d["image_id"] for d in json.load(open(p))}
            ids = s if ids is None else (ids & s)
    print(f"scoring on {len(ids)} images common to every run")

    rows = []
    for m in MODELS:
        if m not in have:
            continue
        r = {k: score(anno, p, ids) for k, p in have[m].items()}
        act = r["float_hardswish"] - r["float_silu"]
        qnt = r["int8_hardswish"] - r["float_hardswish"]
        tot = act + qnt
        rows.append({
            "model": m,
            "images": len(ids),
            "float_silu": f"{r['float_silu']:.4f}",
            "float_hardswish": f"{r['float_hardswish']:.4f}",
            "int8_hardswish": f"{r['int8_hardswish']:.4f}",
            "activation_delta": f"{act:+.4f}",
            "quant_delta": f"{qnt:+.4f}",
            "total_delta": f"{tot:+.4f}",
            "activation_share_pct": f"{100*abs(act)/max(abs(tot),1e-9):.0f}%",
        })
        print(f"  {m}: SiLU {r['float_silu']:.4f} -> HardSwish {r['float_hardswish']:.4f} "
              f"-> INT8 {r['int8_hardswish']:.4f}   act {act:+.4f}  quant {qnt:+.4f}")

    os.makedirs(os.path.dirname(OUTC), exist_ok=True)
    with open(OUTC, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote", OUTC)

    # ---- figure: stacked losses, activation vs quantization --------------------
    ms = [r["model"] for r in rows]
    act = [abs(float(r["activation_delta"])) for r in rows]
    qnt = [abs(float(r["quant_delta"])) for r in rows]
    x = np.arange(len(ms))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))

    axes[0].bar(x, act, 0.6, color=ORANGE, edgecolor="white", linewidth=1.2,
                label="SiLU to HardSwish")
    axes[0].bar(x, qnt, 0.6, bottom=act, color=BLUE, edgecolor="white", linewidth=1.2,
                label="INT8 quantization")
    for xi, a, q in zip(x, act, qnt):
        axes[0].text(xi, a / 2, f"{a:.3f}", ha="center", va="center", fontsize=8, color="white")
        axes[0].text(xi, a + q + 0.004, f"{a+q:.3f}", ha="center", va="bottom",
                     fontsize=8, color=INK2)
    axes[0].set_xticks(x); axes[0].set_xticklabels([m.replace("yolov8", "") for m in ms])
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper right")
    axes[0].set_ylabel("mAP@0.5 lost", color=INK2, fontsize=9)
    axes[0].set_title("Where the accuracy goes", color=INK, fontsize=11, loc="left", pad=8)

    share = [100 * a / (a + q) for a, q in zip(act, qnt)]
    axes[1].bar(x, share, 0.6, color=ORANGE, edgecolor="white", linewidth=1.2)
    for xi, v_ in zip(x, share):
        axes[1].text(xi, v_ + 1.2, f"{v_:.0f}%", ha="center", fontsize=8.5, color=INK2)
    axes[1].axhline(50, color=GRID, lw=1.2)
    axes[1].set_xticks(x); axes[1].set_xticklabels([m.replace("yolov8", "") for m in ms])
    axes[1].set_ylim(0, 108)
    axes[1].set_ylabel("%", color=INK2, fontsize=9)
    axes[1].set_title("Activation's share of the loss", color=INK, fontsize=11,
                      loc="left", pad=8)

    for ax in axes:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, length=0, labelsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUTF), exist_ok=True)
    fig.savefig(OUTF, dpi=200, facecolor="white", bbox_inches="tight")
    print("wrote", OUTF)


if __name__ == "__main__":
    main()
