"""Figures for the VCK190 hardware report.

Sized for an A4 text column. LibreOffice's HTML import places images by pixel size at 96 dpi
unless BOTH width and height attributes are given, in which case it honours them and keeps the
full resolution - so these are rendered large and scaled down in the document.

    Versal_AI/venv/bin/python tools/report_figures.py
"""

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
REF, INK, INK2, GRID = "#c9c7c1", "#0b0b0b", "#52514e", "#e5e4e0"
OUT = "Yolo_v8_Versal_Implementation/report/figures"
MODELS = ["yolov8n", "yolov8s", "yolov8m", "yolov8l"]
SHORT = ["n", "s", "m", "l"]


def load():
    p = "Yolo_v8_Versal_Implementation/board_results/metrics/all_models_metrics.csv"
    d = {}
    for r in csv.DictReader(open(p)):
        d[r["parameter"]] = {m: r[m] for m in MODELS}
    return d


def f(d, key):
    return [float(d[key][m]) for m in MODELS]


def style(ax, ylabel=None, title=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, length=0, labelsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("  wrote", p)


def bars(ax, vals, colour, fmt="{:.2f}", width=0.6):
    x = np.arange(len(MODELS))
    ax.bar(x, vals, width, color=colour, edgecolor="white", linewidth=1.2)
    for xi, v in zip(x, vals):
        ax.text(xi, v + max(vals) * 0.02, fmt.format(v), ha="center", va="bottom",
                fontsize=8.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT)
    ax.set_ylim(0, max(vals) * 1.18)


def fig_throughput(d):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    e2e, dpu = f(d, "fps"), f(d, "fps_dpu_only")
    x = np.arange(len(MODELS))
    w = 0.38
    axes[0].bar(x - w / 2, dpu, w * 0.92, label="DPU only", color=BLUE,
                edgecolor="white", linewidth=1.2)
    axes[0].bar(x + w / 2, e2e, w * 0.92, label="end to end", color=ORANGE,
                edgecolor="white", linewidth=1.2)
    for xi, v in zip(x - w / 2, dpu):
        axes[0].text(xi, v + 0.5, f"{v:.1f}", ha="center", fontsize=8, color=INK2)
    for xi, v in zip(x + w / 2, e2e):
        axes[0].text(xi, v + 0.5, f"{v:.1f}", ha="center", fontsize=8, color=INK2)
    axes[0].set_xticks(x); axes[0].set_xticklabels(SHORT); axes[0].set_ylim(0, 27)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper right")
    style(axes[0], "images / s", "Throughput")

    bars(axes[1], f(d, "latency_per_image"), AQUA, "{:.1f}")
    style(axes[1], "ms", "Latency per image (end to end)")
    fig.tight_layout()
    save(fig, "throughput.png")


def fig_power(d):
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9))
    idle, load = f(d, "power_idle"), f(d, "power_under_load")
    x = np.arange(len(MODELS))
    axes[0].bar(x, idle, 0.6, color=REF, edgecolor="white", linewidth=1.2, label="idle")
    axes[0].bar(x, [l - i for l, i in zip(load, idle)], 0.6, bottom=idle, color=ORANGE,
                edgecolor="white", linewidth=1.2, label="inference")
    for xi, l in zip(x, load):
        axes[0].text(xi, l + 0.18, f"{l:.2f}", ha="center", fontsize=8, color=INK2)
    axes[0].set_xticks(x); axes[0].set_xticklabels(SHORT); axes[0].set_ylim(0, 16.5)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    style(axes[0], "W", "Board power")

    bars(axes[1], f(d, "power_attributable"), ORANGE, "{:.2f}")
    style(axes[1], "W", "Power drawn by inference")

    bars(axes[2], f(d, "energy_per_image"), BLUE, "{:.3f}")
    style(axes[2], "J / image", "Energy per image")
    fig.tight_layout()
    save(fig, "power_energy.png")


def fig_memory(d):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    w_, a_ = f(d, "ddr_weights"), f(d, "ddr_activations")
    io = [float(d["ddr_input_buffer"][m]) + float(d["ddr_output_buffer"][m]) for m in MODELS]
    x = np.arange(len(MODELS))
    axes[0].bar(x, w_, 0.6, color=BLUE, edgecolor="white", linewidth=1.2, label="weights")
    axes[0].bar(x, a_, 0.6, bottom=w_, color=ORANGE, edgecolor="white", linewidth=1.2,
                label="activations")
    axes[0].bar(x, io, 0.6, bottom=[a + b for a, b in zip(w_, a_)], color=AQUA,
                edgecolor="white", linewidth=1.2, label="I/O buffers")
    for xi, t in zip(x, f(d, "ddr_total")):
        axes[0].text(xi, t + 1.5, f"{t:.1f}", ha="center", fontsize=8, color=INK2)
    axes[0].set_xticks(x); axes[0].set_xticklabels(SHORT); axes[0].set_ylim(0, 108)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    style(axes[0], "MiB", "DDR footprint")

    bars(axes[1], f(d, "xmodel_size"), BLUE, "{:.1f}")
    style(axes[1], "MiB", "Compiled xmodel size")
    fig.tight_layout()
    save(fig, "memory.png")


def fig_compute(d):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    wl, wla = f(d, "workload"), f(d, "workload_on_arch")
    x = np.arange(len(MODELS))
    w = 0.38
    axes[0].bar(x - w / 2, wl, w * 0.92, label="useful", color=BLUE,
                edgecolor="white", linewidth=1.2)
    axes[0].bar(x + w / 2, wla, w * 0.92, label="mapped onto array", color=REF,
                edgecolor="white", linewidth=1.2)
    axes[0].set_xticks(x); axes[0].set_xticklabels(SHORT)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")
    style(axes[0], "GOP / image", "Compute per image")

    eff = f(d, "dpu_array_efficiency")
    axes[1].plot(x, eff, color=ORANGE, lw=2, marker="o", markersize=8,
                 markeredgecolor="white", markeredgewidth=1.5)
    for xi, v in zip(x, eff):
        axes[1].annotate(f"{v:.1f}%", (xi, v), xytext=(0, 9), textcoords="offset points",
                         ha="center", fontsize=8.5, color=INK)
    axes[1].set_xticks(x); axes[1].set_xticklabels(SHORT); axes[1].set_ylim(82, 102)
    style(axes[1], "%", "DPU array utilisation")
    fig.tight_layout()
    save(fig, "compute.png")


def fig_tradeoff(d):
    """Accuracy against energy - the deployment decision in one picture."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    e, acc = f(d, "energy_per_image"), f(d, "mAP@0.5")
    # No connecting line: these are four independent design points, not a series. Joining them
    # in model order draws a zigzag that implies an ordering the axes do not have.
    for i, m in enumerate(MODELS):
        ax.scatter(e[i], acc[i], s=110, color=[BLUE, AQUA, ORANGE, "#4a3aa7"][i],
                   edgecolor="white", linewidth=1.6, zorder=3)
        ax.annotate(f"  {m}\n  {acc[i]:.3f} mAP, {e[i]:.3f} J",
                    (e[i], acc[i]), xytext=(8, -12), textcoords="offset points",
                    fontsize=8.5, color=INK, linespacing=1.3)
    ax.set_xlim(0.84, 1.22)
    ax.set_ylim(0.40, 0.68)
    style(ax, "mAP@0.5", "Accuracy against energy per image")
    ax.set_xlabel("energy per image (J)", color=INK2, fontsize=9)
    fig.tight_layout()
    save(fig, "tradeoff.png")


def fig_accuracy(d):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    for ax, k, fk, t in ((axes[0], "mAP@0.5", "float_mAP@0.5", "mAP@0.5"),
                         (axes[1], "mAP@0.5:0.95", "float_mAP@0.5:0.95", "mAP@0.5:0.95")):
        board, flt = f(d, k), f(d, fk)
        x = np.arange(len(MODELS))
        w = 0.38
        ax.bar(x - w / 2, flt, w * 0.92, label="float (PC)", color=REF,
               edgecolor="white", linewidth=1.2)
        ax.bar(x + w / 2, board, w * 0.92, label="INT8 on VCK190", color=BLUE,
               edgecolor="white", linewidth=1.2)
        for xi, v in zip(x - w / 2, flt):
            ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=7.5, color=INK2)
        for xi, v in zip(x + w / 2, board):
            ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=7.5, color=INK2)
        ax.set_xticks(x); ax.set_xticklabels(SHORT); ax.set_ylim(0, max(flt) * 1.2)
        style(ax, k, t)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    save(fig, "accuracy.png")


def fig_prf1(d):
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    x = np.arange(len(MODELS))
    w = 0.26
    for i, (k, c, lbl) in enumerate((("precision", BLUE, "precision"),
                                     ("recall", ORANGE, "recall"),
                                     ("f1_score", AQUA, "F1"))):
        v = f(d, k)
        ax.bar(x + (i - 1) * w, v, w * 0.9, color=c, label=lbl,
               edgecolor="white", linewidth=1.2)
        for xi, vv in zip(x + (i - 1) * w, v):
            ax.text(xi, vv + 0.012, f"{vv:.3f}", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(SHORT); ax.set_ylim(0, 0.88)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left", ncol=3)
    style(ax, "", "Precision, recall and F1 at the best-F1 operating point")
    fig.tight_layout()
    save(fig, "prf1.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    d = load()
    fig_accuracy(d)
    fig_prf1(d)
    fig_throughput(d)
    fig_power(d)
    fig_memory(d)
    fig_compute(d)
    fig_tradeoff(d)
