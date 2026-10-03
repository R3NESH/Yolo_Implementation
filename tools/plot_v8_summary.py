"""Summary charts for the YOLOv8 VCK190 deployment.

Companion to tools/board_curves.py, which draws the per-model PR/P/R/F1 curves from a
predictions JSON. This draws the cross-model story that no single run contains:

  fig1  accuracy by model - float ceiling, this deployment, and the prior work
  fig2  the 2x2 that prices the DPU: activation x precision, on yolov8n

Colours are the reference categorical palette's slots 1-2 (blue, orange), with the float
baseline carried as a neutral reference rather than a third category - it is a ceiling, not a
peer. Values are direct-labelled so identity never rests on colour alone, and a CSV table view
is written alongside for the same reason.

    Versal_AI/venv/bin/python tools/plot_v8_summary.py [out_dir]
"""

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE = "#2a78d6", "#eb6834"      # categorical slots 1 and 2
REF = "#c9c7c1"                           # neutral: the float ceiling is a reference, not a series
INK, INK2 = "#0b0b0b", "#52514e"
GRID = "#e5e4e0"

MODELS = ["yolov8n", "yolov8s", "yolov8m", "yolov8l"]

# Full val2017, 5000 images. Float from Versal_AI/benchmarks; prior board from results_v3.
FLOAT_50 = [0.5187, 0.6106, 0.6654, 0.6916]
BOARD_50 = [0.4453, 0.5423, 0.6001, 0.6233]
PRIOR_50 = [0.4317, 0.5301, 0.5935, 0.6219]

FLOAT_5095 = [0.3681, 0.4440, 0.4979, 0.5244]
BOARD_5095 = [0.2913, 0.3716, 0.4256, 0.4467]
PRIOR_5095 = [0.2917, 0.3724, 0.4278, 0.4528]

# yolov8n, fixed 500-image subset, letterbox, INT8-simulated. mAP@0.5.
# Rows: activation. Columns: precision.
CELL = {("SiLU", "float"): 0.5674, ("SiLU", "INT8"): 0.5431,
        ("HardSwish", "float"): 0.4809, ("HardSwish", "INT8"): 0.4714}


def style(ax):
    """Recessive axes: no box, one faint horizontal grid, ink-coloured text."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, length=0, labelsize=9)


def bars(ax, series, title, ylabel):
    """Grouped bars: one group per model, `series` is [(label, values, colour), ...]."""
    x = np.arange(len(MODELS))
    n = len(series)
    w = 0.78 / n
    for i, (label, vals, colour) in enumerate(series):
        off = (i - (n - 1) / 2) * w
        ax.bar(x + off, vals, w * 0.92, label=label, color=colour,
               edgecolor="white", linewidth=1.2)          # 2px-equivalent surface gap
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=7.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("yolov8", "") for m in MODELS])
    ax.set_xlabel("model", color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, pad=10, loc="left")
    ax.set_ylim(0, max(max(v) for _, v, _ in series) * 1.16)
    style(ax)


def fig_accuracy(out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    bars(axes[0],
         [("float (PC ceiling)", FLOAT_50, REF),
          ("this deployment", BOARD_50, BLUE),
          ("prior work (_v3)", PRIOR_50, ORANGE)],
         "mAP@0.5", "mAP@0.5")
    bars(axes[1],
         [("float (PC ceiling)", FLOAT_5095, REF),
          ("this deployment", BOARD_5095, BLUE),
          ("prior work (_v3)", PRIOR_5095, ORANGE)],
         "mAP@0.5:0.95", "mAP@0.5:0.95")
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    fig.suptitle("YOLOv8 INT8 on VCK190 - COCO val2017, 5000 images",
                 color=INK, fontsize=13, x=0.078, ha="left", y=0.985)
    fig.text(0.078, 0.925,
             "HardSwish for SiLU, letterbox preprocessing, host-side DFL decode. "
             "Every model beats the prior work at mAP@0.5.",
             color=INK2, fontsize=9.5, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    p = os.path.join(out_dir, "v8_accuracy_by_model.png")
    fig.savefig(p, dpi=200, facecolor="white")
    plt.close(fig)
    print("  wrote", p)


def fig_decomposition(out_dir):
    """The 2x2: which costs more, the activation swap or INT8?

    Drawn as a slope chart rather than grouped bars, because the comparison IS the geometry:
    the vertical distance between the two lines is the activation cost, and each line's slope is
    the quantization cost. Bars forced the same comparison into arrows that had to cross
    unrelated bars to reach their targets.
    """
    fig, ax = plt.subplots(figsize=(9, 5.8))
    x = [0, 1]
    silu = [CELL[("SiLU", "float")], CELL[("SiLU", "INT8")]]
    hard = [CELL[("HardSwish", "float")], CELL[("HardSwish", "INT8")]]

    for vals, colour, label in ((silu, BLUE, "SiLU (exact)"),
                               (hard, ORANGE, "HardSwish (DPU-native)")):
        ax.plot(x, vals, color=colour, lw=2, marker="o", markersize=9,
                markeredgecolor="white", markeredgewidth=1.6, label=label, zorder=3)
        ax.annotate(f"{vals[0]:.4f}", (x[0], vals[0]), xytext=(-14, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=10, color=INK)
        ax.annotate(f"{vals[1]:.4f}", (x[1], vals[1]), xytext=(14, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=10, color=INK)
        # each line's slope is the quantization cost for that activation
        ax.annotate(f"INT8: {vals[1] - vals[0]:+.4f}", (0.5, (vals[0] + vals[1]) / 2),
                    xytext=(0, 9), textcoords="offset points", ha="center", va="bottom",
                    fontsize=9, color=colour)

    # the vertical gap is the activation cost - measured at both precisions
    for xi, a, b, side in ((0.0, silu[0], hard[0], 1), (1.0, silu[1], hard[1], -1)):
        ax.annotate("", xy=(xi + 0.055 * side, a), xytext=(xi + 0.055 * side, b),
                    arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.3))
        ax.text(xi + 0.075 * side, (a + b) / 2,
                f"activation\n{b - a:+.4f}", ha="left" if side > 0 else "right",
                va="center", fontsize=9.5, color=INK, linespacing=1.35)

    ax.set_xticks(x)
    ax.set_xticklabels(["float", "INT8 (quantized)"], fontsize=10)
    ax.set_xlim(-0.42, 1.42)
    ax.set_ylim(0.44, 0.60)
    ax.set_ylabel("mAP@0.5", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK2, loc="lower left")
    style(ax)
    ax.xaxis.grid(False)

    fig.suptitle("What the DPU actually costs - yolov8n", color=INK, fontsize=13,
                 x=0.055, ha="left", y=0.975)
    fig.text(0.055, 0.90,
             "The gap between the lines is the activation swap; each line's slope is INT8.\n"
             "500-image subset, letterbox, all four points from one pipeline.",
             color=INK2, fontsize=9.5, ha="left", va="top", linespacing=1.45)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    p = os.path.join(out_dir, "v8_activation_vs_quantization.png")
    fig.savefig(p, dpi=200, facecolor="white")
    plt.close(fig)
    print("  wrote", p)


def table_view(out_dir):
    """A chart is not the only way in - ship the numbers too."""
    p = os.path.join(out_dir, "v8_summary.csv")
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "float_mAP50", "board_mAP50", "prior_mAP50", "retained_pct_50",
                    "float_mAP50_95", "board_mAP50_95", "prior_mAP50_95", "retained_pct_50_95"])
        for i, m in enumerate(MODELS):
            w.writerow([m, FLOAT_50[i], BOARD_50[i], PRIOR_50[i],
                        round(100 * BOARD_50[i] / FLOAT_50[i], 1),
                        FLOAT_5095[i], BOARD_5095[i], PRIOR_5095[i],
                        round(100 * BOARD_5095[i] / FLOAT_5095[i], 1)])
    print("  wrote", p)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else \
        "Yolo_v8_Versal_Implementation/board_results/plots_summary"
    os.makedirs(out, exist_ok=True)
    fig_accuracy(out)
    fig_decomposition(out)
    table_view(out)
