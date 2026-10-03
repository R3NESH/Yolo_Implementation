"""Precision / recall / F1 at the operating point, for each board predictions JSON.

mAP integrates over every confidence threshold, so it does not tell you what the detector does
when you actually deploy it at one threshold. This reports P, R and F1 at the confidence that
maximises F1 - the number you would ship - reusing tools/board_curves.py's matcher so the
figures are consistent with the curves it draws.

    Versal_AI/venv/bin/python tools/extract_pr_f1.py <annotations.json> <preds.json> [...]
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from board_curves import GRID, load, match_per_class  # noqa: E402


def pr_f1(gt_path, dt_path):
    """P, R and F1 at the confidence that maximises mean F1.

    Mirrors tools/board_curves.py exactly - same matcher, same interpolation edges, and F1
    computed **per class then averaged** rather than as the harmonic mean of the averaged P and
    R. Those two are not the same number (F1 is nonlinear), and using the other one here would
    put a figure in the CSV that disagrees with the F1_curve.png sitting beside it.
    """
    gt_by_key, n_gt, names, cat_index, dets = load(gt_path, dt_path)
    curves = match_per_class(gt_by_key, n_gt, cat_index, dets, len(names))

    P_conf, R_conf, F_conf = [], [], []
    for cur in curves:
        if cur is None:
            continue
        recall, precision, conf = cur
        cf, pr, rc = conf[::-1], precision[::-1], recall[::-1]
        p_i = np.interp(GRID, cf, pr, left=pr[0], right=pr[-1])
        r_i = np.interp(GRID, cf, rc, left=rc[0], right=0.0)
        P_conf.append(p_i)
        R_conf.append(r_i)
        F_conf.append(2 * p_i * r_i / (p_i + r_i + 1e-12))

    P_conf = np.array(P_conf)
    R_conf = np.array(R_conf)
    F_conf = np.array(F_conf)

    mF = F_conf.mean(0)
    i = int(mF.argmax())
    return {"precision": P_conf.mean(0)[i], "recall": R_conf.mean(0)[i], "f1": mF[i],
            "confidence": GRID[i], "classes": len(F_conf)}


if __name__ == "__main__":
    ann, preds = sys.argv[1], sys.argv[2:]
    print(f"{'file':<28} {'conf':>6} {'precision':>10} {'recall':>8} {'F1':>8}")
    print("-" * 64)
    for p in preds:
        r = pr_f1(ann, p)
        print(f"{os.path.basename(p):<28} {r['confidence']:>6.3f} {r['precision']:>10.4f} "
              f"{r['recall']:>8.4f} {r['f1']:>8.4f}")
