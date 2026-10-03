"""Build the VCK190 report for yolov8s with LeakyReLU, as HTML for LibreOffice -> PDF.

Companion to tools/build_report.py (the four-model HardSwish report). Same sections, one model,
with the HardSwish yolov8s from that report alongside as the reference.

Every board number below was measured on this model and board; each carries its source in the
metrics CSV this writes. The float baseline is scored here, from the predictions the float
model produced, with pycocotools - nothing is typed in for it.

    Versal_AI/venv/bin/python tools/build_report_leaky.py
    soffice --headless --convert-to pdf --outdir <report_leaky> <report_leaky>/report.html
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
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = os.path.abspath(".")
V8 = os.path.join(ROOT, "Yolo_v8_Versal_Implementation")
BOARD = os.path.join(V8, "board_results")
METRICS = os.path.join(BOARD, "metrics")
OUT = os.path.join(V8, "report_leaky")
FIGS = os.path.join(OUT, "figures")
ANN = os.path.expanduser("~/Desktop/Versal_AI/datasets/coco/annotations/instances_val2017.json")
FLOAT_PREDS = os.path.join(V8, "sim_results", "preds_yolov8s_leaky_float_lb.json")

FULL_W, MIB = 620, 1024 * 1024

# --------------------------------------------------------------------------------------
# LeakyReLU yolov8s - measured
# --------------------------------------------------------------------------------------
L = {
    # pycocotools over val2017, 5000 images, board run (metrics_yolov8s_leaky.csv)
    "map50": 0.510389, "map5095": 0.328551, "map75": 0.338326, "ar100": 0.528349,
    "detections": 787079,
    # tools/board_curves.py, independent matcher
    "map50_indep": 0.5060,
    # tools/extract_pr_f1.py, at the best-F1 confidence
    "precision": 0.5817, "recall": 0.4854, "f1": 0.5133, "conf": 0.183,
    # board_eval --benchmark 200, three runs (42.5, 42.8, 43.1); batch 6, preprocess + DPU
    "fps_dpu": 42.8,
    # board_eval --limit 1000 --no-eval, wall clock incl. process start-up, same day as the
    # HardSwish reference below
    "fps_e2e": 1000 / 70.22, "lat_ms": 70.22,
    # board_eval full scored run, tqdm loop only: 5000 images in 7:37
    "fps_scored_run": 5000 / 457,
    # host_vck190/measure_power.py, 17 INA226 rails, 400 back-to-back inferences
    "p_idle": 12.90, "p_load": 13.80, "p_attr": 0.89, "j_img": 0.972, "j_attr": 0.063,
    "eff": 1.03, "fps_for_energy": 14.20,
    # xir DPU-subgraph attributes (Yolo_v8_Versal_Implementation/dump_xmodel_stats.py)
    "gop": 25.5427136, "gop_arch": 26.8571802,
    "ddr_w": 10603968 / MIB, "ddr_a": 22092800 / MIB, "ddr_in": 1230784 / MIB,
    "ddr_out": 1209600 / MIB, "mc_kib": 192584 / 1024, "xmodel_mib": 11034685 / MIB,
    "dpu_ops": 226, "params_m": 10.612, "conv2d": 58,
    # training log shipped with the checkpoint (ultralytics val, 416 input, no letterbox)
    "log_map50": 0.4927818, "log_map5095": 0.3429585, "log_p": 0.6064, "log_r": 0.4565,
}
L["array_eff"] = 100 * L["gop"] / L["gop_arch"]
L["ddr_total"] = L["ddr_w"] + L["ddr_a"] + L["ddr_in"] + L["ddr_out"]

# --------------------------------------------------------------------------------------
# HardSwish yolov8s - from board_results/metrics/all_models_metrics.csv (earlier report),
# except the same-day throughput rows, re-measured so the comparison is like for like.
# --------------------------------------------------------------------------------------
def load_hs():
    with open(os.path.join(METRICS, "all_models_metrics.csv")) as fh:
        d = {r["parameter"]: r["yolov8s"] for r in csv.DictReader(fh)}
    f = lambda k: float(d[k])
    return {
        "map50": f("mAP@0.5"), "map5095": f("mAP@0.5:0.95"), "map75": f("mAP@0.75"),
        "ar100": f("AR@100"), "detections": int(d["detections_total"]),
        "map50_indep": f("mAP@0.5_independent_scorer"),
        "precision": f("precision"), "recall": f("recall"), "f1": f("f1_score"),
        "conf": f("operating_confidence"),
        "fps_dpu": 42.4,                          # same-day: 42.3, 42.5
        "fps_e2e": 1000 / 59.98, "lat_ms": 59.98,  # same-day, same command as the leaky row
        "fps_scored_run": f("fps"),
        "p_idle": f("power_idle"), "p_load": f("power_under_load"),
        "p_attr": f("power_attributable"), "j_img": f("energy_per_image"),
        "j_attr": f("energy_per_image_attributable"), "eff": f("efficiency"),
        "gop": f("workload"), "gop_arch": f("workload_on_arch"),
        "array_eff": f("dpu_array_efficiency"),
        "ddr_w": f("ddr_weights"), "ddr_a": f("ddr_activations"),
        "ddr_in": f("ddr_input_buffer"), "ddr_out": f("ddr_output_buffer"),
        "ddr_total": f("ddr_total"), "mc_kib": f("microcode_size"),
        "xmodel_mib": f("xmodel_size"), "dpu_ops": int(f("dpu_ops")),
        "params_m": f("parameters"), "conv2d": 64,
        "float_map50": f("float_mAP@0.5"), "float_map5095": f("float_mAP@0.5:0.95"),
    }


def score_float():
    """Float model, letterbox, 640, all 5000 images - same scorer as the board runs."""
    if not os.path.exists(FLOAT_PREDS):
        sys.exit("float predictions missing: %s" % FLOAT_PREDS)
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(ANN)
        dt = gt.loadRes(FLOAT_PREDS)
        e = COCOeval(gt, dt, "bbox")
        e.evaluate(); e.accumulate(); e.summarize()
    n = len({d["image_id"] for d in json.load(open(FLOAT_PREDS))})
    return float(e.stats[1]), float(e.stats[0]), n


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------
BLUE, ORANGE, REF, INK, INK2, GRID = "#2a78d6", "#eb6834", "#c9c7c1", "#0b0b0b", "#52514e", "#e5e4e0"


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


def grouped(ax, cats, series, fmt="{:.3f}", ymax=None):
    """series: [(label, colour, [values per category])]"""
    x = np.arange(len(cats))
    w = 0.8 / len(series)
    top = ymax or max(max(v) for _, _, v in series) * 1.2
    for i, (lab, col, vals) in enumerate(series):
        xs = x + (i - (len(series) - 1) / 2) * w
        ax.bar(xs, vals, w * 0.92, label=lab, color=col, edgecolor="white", linewidth=1.2)
        for xi, v in zip(xs, vals):
            ax.text(xi, v + top * 0.015, fmt.format(v), ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylim(0, top)


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    p = os.path.join(FIGS, name)
    fig.savefig(p, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return p


def make_figures(L, H, fl50, fl5095):
    figs = {}
    # accuracy: float ceiling against the board, both models
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    grouped(ax, ["mAP@0.5", "mAP@0.5:0.95"],
            [("LeakyReLU, float", REF, [fl50, fl5095]),
             ("LeakyReLU, board INT8", BLUE, [L["map50"], L["map5095"]]),
             ("HardSwish, float", "#e8c3b3", [H["float_map50"], H["float_map5095"]]),
             ("HardSwish, board INT8", ORANGE, [H["map50"], H["map5095"]])],
            ymax=0.8)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, ncol=2, loc="upper right")
    style(ax, "mAP", "Float ceiling and board result")
    figs["accuracy"] = save(fig, "accuracy.png")

    # throughput
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    grouped(axes[0], ["DPU only", "end to end"],
            [("LeakyReLU", BLUE, [L["fps_dpu"], L["fps_e2e"]]),
             ("HardSwish", ORANGE, [H["fps_dpu"], H["fps_e2e"]])], fmt="{:.1f}", ymax=55)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper right")
    style(axes[0], "images / s", "Throughput, same day")
    grouped(axes[1], ["LeakyReLU", "HardSwish"],
            [("latency", "#1baf7a", [L["lat_ms"], H["lat_ms"]])], fmt="{:.1f}", ymax=90)
    style(axes[1], "ms", "Wall clock per image, end to end")
    fig.tight_layout()
    figs["throughput"] = save(fig, "throughput.png")

    # power and memory
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9))
    grouped(axes[0], ["idle", "load"],
            [("LeakyReLU", BLUE, [L["p_idle"], L["p_load"]]),
             ("HardSwish", ORANGE, [H["p_idle"], H["p_load"]])], fmt="{:.2f}", ymax=17)
    style(axes[0], "W", "Board power")
    axes[0].legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="upper left")
    grouped(axes[1], ["J/img"],
            [("LeakyReLU", BLUE, [L["j_img"]]), ("HardSwish", ORANGE, [H["j_img"]])],
            fmt="{:.3f}", ymax=1.2)
    style(axes[1], "joules", "Energy per image")
    grouped(axes[2], ["DDR (MiB)"],
            [("LeakyReLU", BLUE, [L["ddr_total"]]), ("HardSwish", ORANGE, [H["ddr_total"]])],
            fmt="{:.1f}", ymax=42)
    style(axes[2], "MiB", "DDR footprint")
    fig.tight_layout()
    figs["power_memory"] = save(fig, "power_memory.png")
    return figs


# --------------------------------------------------------------------------------------
# HTML helpers (same look as build_report.py)
# --------------------------------------------------------------------------------------
HEAD = """<html><head><meta charset="utf-8"></head>
<body style="font-family:'Liberation Serif',serif;font-size:10.5pt;">
"""


def h1(t): return f'<h1 style="font-size:17pt;margin-bottom:2pt;">{t}</h1>'
def h2(t): return f'<h2 style="font-size:13pt;margin-top:18pt;margin-bottom:4pt;">{t}</h2>'
def h3(t): return f'<h3 style="font-size:11pt;margin-top:12pt;margin-bottom:3pt;">{t}</h3>'
def p(t): return f'<p style="text-align:justify;line-height:1.35;">{t}</p>'
def pagebreak(): return '<p style="page-break-before:always;"></p>'


def note(t):
    return (f'<p style="font-size:9pt;color:#333;border-left:3px solid #bbb;'
            f'padding-left:8pt;margin:8pt 0;">{t}</p>')


def img(path, target_w=FULL_W, caption=None):
    if not os.path.exists(path):
        return f'<p><i>[missing figure: {os.path.basename(path)}]</i></p>'
    w, h = Image.open(path).size
    th = int(round(target_w * h / w))
    cap = (f'<div style="font-size:8.5pt;color:#444;margin-top:2pt;">{caption}</div>'
           if caption else "")
    return (f'<div style="margin:10pt 0 12pt 0;">'
            f'<img src="file://{path}" width="{target_w}" height="{th}">{cap}</div>')


def table(headers, rows, aligns=None, width="100%", fs="9pt"):
    aligns = aligns or ["left"] + ["right"] * (len(headers) - 1)
    h = "".join(f'<th align="{a}">{x}</th>' for x, a in zip(headers, aligns))
    body = "".join("<tr>" + "".join(f'<td align="{a}">{x}</td>' for x, a in zip(r, aligns))
                   + "</tr>" for r in rows)
    return (f'<table border="1" cellspacing="0" cellpadding="4" width="{width}" '
            f'style="font-size:{fs};"><tr bgcolor="#e9e9e9">{h}</tr>{body}</table>')


def pct(a, b): return f"{100 * a / b:.1f}"


def write_metrics_csv(L, fl50, fl5095):
    """One row per metric, each with where it came from."""
    rows = [
        ("model", "yolov8s_leaky", "", "checkpoint yolov8_leaky_relu_trained_e15.pt"),
        ("board", "VCK190, DPUCVDX8G_ISA3_C32B6, batch 6, 333 MHz, VART 3.0.0", "", "fixed"),
        ("configuration", "INT8, LeakyReLU(0.1), letterbox, 640x640, conf 0.001, IoU 0.45, "
                          "max_det 300", "", "fixed"),
        ("mAP@0.5", f"{L['map50']:.6f}", "", "board_eval full run, pycocotools, 5000 images"),
        ("mAP@0.5:0.95", f"{L['map5095']:.6f}", "", "same"),
        ("mAP@0.75", f"{L['map75']:.6f}", "", "same"),
        ("AR@100", f"{L['ar100']:.6f}", "", "same"),
        ("mAP@0.5_independent_scorer", f"{L['map50_indep']:.4f}", "", "tools/board_curves.py"),
        ("float_mAP@0.5", f"{fl50:.4f}", "", "float model, letterbox 640, all 5000, pycocotools"),
        ("float_mAP@0.5:0.95", f"{fl5095:.4f}", "", "same"),
        ("accuracy_retained@0.5", pct(L["map50"], fl50), "%", "derived"),
        ("accuracy_retained@0.5:0.95", pct(L["map5095"], fl5095), "%", "derived"),
        ("precision", f"{L['precision']:.4f}", "", "tools/extract_pr_f1.py at best F1"),
        ("recall", f"{L['recall']:.4f}", "", "same"),
        ("f1_score", f"{L['f1']:.4f}", "", "same"),
        ("operating_confidence", f"{L['conf']:.3f}", "", "same"),
        ("detections_total", str(L["detections"]), "count", "board run"),
        ("fps_dpu_only", f"{L['fps_dpu']:.1f}", "images/s", "board_eval --benchmark 200, x3"),
        ("fps_end_to_end", f"{L['fps_e2e']:.2f}", "images/s", "--limit 1000 --no-eval wall clock"),
        ("fps_scored_run", f"{L['fps_scored_run']:.2f}", "images/s", "full 5000-image run, 7:37"),
        ("power_idle", f"{L['p_idle']:.2f}", "W", "measure_power.py, 17 INA226 rails"),
        ("power_under_load", f"{L['p_load']:.2f}", "W", "same, 400 back-to-back inferences"),
        ("power_attributable", f"{L['p_attr']:.2f}", "W", "load minus idle"),
        ("energy_per_image", f"{L['j_img']:.3f}", "J/image", "load power / 14.20 img/s"),
        ("parameters", f"{L['params_m']:.3f}", "M", "checkpoint"),
        ("workload", f"{L['gop']:.3f}", "GOP/image", "xir subgraph attribute"),
        ("workload_on_arch", f"{L['gop_arch']:.3f}", "GOP/image", "same"),
        ("dpu_array_efficiency", f"{L['array_eff']:.1f}", "%", "derived"),
        ("ddr_weights", f"{L['ddr_w']:.2f}", "MiB", "xir REG_0"),
        ("ddr_activations", f"{L['ddr_a']:.2f}", "MiB", "xir REG_1"),
        ("ddr_total", f"{L['ddr_total']:.2f}", "MiB", "REG_0..3"),
        ("xmodel_size", f"{L['xmodel_mib']:.2f}", "MiB", "file"),
        ("microcode_size", f"{L['mc_kib']:.1f}", "KiB", "xir mc_code"),
        ("dpu_ops", str(L["dpu_ops"]), "count", "xir"),
        ("dpu_subgraphs", "1", "count", "vai_c_xir gate"),
        ("cpu_ops", "fix2float x3", "", "vai_c_xir gate"),
    ]
    os.makedirs(METRICS, exist_ok=True)
    path = os.path.join(METRICS, "yolov8s_leaky_metrics.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["parameter", "value", "unit", "source"])
        w.writerows(rows)
    return path


def build():
    H = load_hs()
    fl50, fl5095, fl_n = score_float()
    if fl_n != 5000:
        sys.exit("float predictions cover %d images, expected 5000" % fl_n)
    write_metrics_csv(L, fl50, fl5095)
    figs = make_figures(L, H, fl50, fl5095)

    ret50, ret5095 = pct(L["map50"], fl50), pct(L["map5095"], fl5095)
    hret50, hret5095 = pct(H["map50"], H["float_map50"]), pct(H["map5095"], H["float_map5095"])
    plots = os.path.join(BOARD, "plots_yolov8s_leaky")

    s = [HEAD]
    A = s.append

    A(h1("YOLOv8s with LeakyReLU on the AMD Versal VCK190"))
    A('<p style="font-size:10pt;color:#444;margin-top:0;">'
      'Hardware implementation and measurement of one model, compared with the HardSwish '
      'yolov8s<br>COCO val2017, 5000 images &nbsp;&middot;&nbsp; October 2026</p><hr>')

    # 1 ------------------------------------------------------------------------------
    A(h2("1. What this covers"))
    A(p("A YOLOv8s detector trained with LeakyReLU activations was quantized to INT8, compiled "
        "for the DPU on a Versal VCK190 evaluation board, and run over the whole of COCO "
        "val2017. This report records how it was built, what it scored, how fast it ran, how "
        "much power it drew and how much memory it needed. It follows the structure of the "
        "four-model HardSwish report, and uses that report's yolov8s as the reference."))
    A(p("Three things are worth stating up front. First, LeakyReLU is something the DPU runs "
        "natively, so this model needed no activation substitution at all &mdash; the step that "
        "dominated the accuracy loss for the stock SiLU models does not exist here. Second, "
        "this is not stock yolov8s: its detection head is lighter (section 3), so the two "
        "models differ in more than the activation. Third, the board scored "
        f"<b>mAP@0.5 {L['map50']:.4f}</b>, which is {ret50}% of this model's own float score, "
        f"against {hret50}% for the HardSwish yolov8s."))
    A(table(["", "mAP@0.5", "mAP@0.5:0.95", "DPU-only img/s", "power (W)", "energy (J/img)"],
            [["yolov8s LeakyReLU", f"{L['map50']:.4f}", f"{L['map5095']:.4f}",
              f"{L['fps_dpu']:.1f}", f"{L['p_load']:.2f}", f"{L['j_img']:.3f}"],
             ["yolov8s HardSwish", f"{H['map50']:.4f}", f"{H['map5095']:.4f}",
              f"{H['fps_dpu']:.1f}", f"{H['p_load']:.2f}", f"{H['j_img']:.3f}"]]))
    A('<p style="font-size:8.5pt;color:#444;">Headline figures. Throughput for both rows was '
      'measured on the same day with the same command. The HardSwish power and energy come '
      'from the earlier report, taken in a different session.</p>')

    # 2 ------------------------------------------------------------------------------
    A(h2("2. Hardware and software"))
    A(p("The board and software are unchanged from the four-model report. The DPU is a "
        "DPUCVDX8G, configured as ISA3_C32B6: thirty-two channels wide with a batch of six, "
        "clocked at 333&nbsp;MHz. The compiled xmodel carries the same DPU fingerprint, "
        "433190037845252193, as the HardSwish models, so it is addressed to exactly the same "
        "hardware."))
    A(table(["Item", "Value"],
            [["Board", "Xilinx Versal VCK190 evaluation board, 8 GiB DRAM"],
             ["DPU", "DPUCVDX8G_ISA3_C32B6, batch 6, 333 MHz"],
             ["Operating system", "PetaLinux 2022.2, aarch64"],
             ["Runtime", "VART 3.0.0, XIR 3.0"],
             ["Toolchain", "Vitis AI 3.0 PyTorch container (vai_q_pytorch, vai_c_xir)"],
             ["Host link", "USB gigabit Ethernet, static 192.168.1.10/24 on the board"]],
            aligns=["left", "left"]))

    # 3 ------------------------------------------------------------------------------
    A(h2("3. The model"))
    A(p("The checkpoint is <tt>yolov8_leaky_relu_trained_e15.pt</tt>; by its name it was trained "
        "for fifteen epochs. It detects the standard 80 COCO classes at 640&times;640 input "
        "with an anchor-free head predicting box edges as a distribution over 16 bins per "
        "side. Its activation is LeakyReLU with a negative slope of 0.1, and the model "
        "contains no SiLU."))
    A(p("It is not the stock yolov8s architecture. Loading it next to the stock weights shows "
        "that each branch of its detection head has two layers where the stock model has three. "
        "That is one 3&times;3 convolution fewer per branch, six in all across the three "
        "scales."))
    A(table(["", "this model", "stock yolov8s"],
            [["Activation", "LeakyReLU (0.1)", "SiLU, replaced by HardSwish for the DPU"],
             ["Convolution layers (Conv2d)", str(L["conv2d"]), str(H["conv2d"])],
             ["Parameters (M)", f"{L['params_m']:.2f}", f"{H['params_m']:.2f}"],
             ["Compute (GOP/image)", f"{L['gop']:.2f}", f"{H['gop']:.2f}"],
             ["DPU operators", str(L["dpu_ops"]), str(H["dpu_ops"])]],
            aligns=["left", "right", "right"]))
    A(note("Because the architecture and the training both differ, any difference between this "
           "model and the HardSwish yolov8s on the board cannot be attributed to the activation "
           "alone. The comparison in section 9 is between two deployments, not an isolated "
           "experiment."))
    A(p("Two float references exist. The training log shipped with the checkpoint records "
        f"mAP@0.5 {L['log_map50']:.4f} and mAP@0.5:0.95 {L['log_map5095']:.4f}, measured by "
        "Ultralytics' validator at 416&times;416 input. The board runs at 640 with letterbox, so "
        "that is not the right ceiling for it. The reference used here is the same model run in "
        f"float on a PC at 640 with letterbox over all 5000 images: <b>mAP@0.5 {fl50:.4f}</b> "
        f"and <b>mAP@0.5:0.95 {fl5095:.4f}</b>."))

    # 4 ------------------------------------------------------------------------------
    A(pagebreak())
    A(h2("4. How the model was put on the board"))
    A(h3("4.1 Splitting the network between DPU and CPU"))
    A(p("The same cut as the other models. The DPU runs everything up to the head's output "
        "convolutions and produces three raw feature maps at strides 8, 16 and 32, each with "
        "144 channels: four box sides &times; 16 distribution bins, plus 80 class scores. The "
        "host does the softmax over the bins, the sigmoid on the scores, and NMS."))
    A(h3("4.2 The activation"))
    A(p("This is where the model differs most from the HardSwish work. SiLU has no DPU mapping, "
        "so the stock models had every SiLU rewritten as HardSwish before quantization, an "
        "approximation that cost far more accuracy than 8-bit arithmetic did. LeakyReLU is "
        "implemented by the DPU directly, so this model went through the toolchain exactly as "
        "trained. There was nothing to replace; the wrapper reported zero substitutions."))
    A(p("The DPU is understood to apply LeakyReLU with a fixed-point slope, so the 0.1 would be "
        "represented approximately rather than exactly; this was not checked against the "
        "compiler's documentation. How much it matters was not isolated. The "
        "whole difference between the float and board scores below therefore contains this "
        "rounding together with ordinary 8-bit quantization."))
    A(h3("4.3 Quantization"))
    A(p("vai_q_pytorch was run in two passes with the settings used for the other models: 200 "
        "calibration images at even stride through a 4000-image split of val2017 that is "
        "disjoint from the 1000 held back, letterboxed to match inference. The input tensor "
        "arrives as int8 with a fixed-point scale of 64 and the three outputs come back with a "
        "scale of 0.25."))
    A(h3("4.4 Compilation"))
    A(p("vai_c_xir compiled the INT8 model against the VCK190 architecture file and produced "
        "one DPU subgraph and three small CPU subgraphs whose only operator is the "
        "fixed-point-to-float conversion. That operator ships with the board. A single DPU "
        "subgraph means the runtime uses the plain VART runner and fills the DPU with six "
        "frames per call."))
    A(table(["Model", "xmodel (MiB)", "DPU subgraphs", "CPU operators", "DPU ops",
             "microcode (KiB)"],
            [["LeakyReLU", f"{L['xmodel_mib']:.2f}", "1", "fix2float &times;3",
              str(L["dpu_ops"]), f"{L['mc_kib']:.1f}"],
             ["HardSwish", f"{H['xmodel_mib']:.2f}", "1", "fix2float &times;3",
              str(H["dpu_ops"]), f"{H['mc_kib']:.1f}"]]))
    A(h3("4.5 Decoding and preprocessing"))
    A(p("Both are the shared host code. Images are letterboxed to 640&times;640 and boxes are "
        "mapped back with the integer padding. The decode was verified against Ultralytics' "
        "own on the stock models and is the same code here; the head's output shape and "
        "channel layout are identical, which the wrapper confirmed before quantization."))

    # 5 ------------------------------------------------------------------------------
    A(pagebreak())
    A(h2("5. How the model performed"))
    A(p("The model was run over all 5000 validation images with a confidence threshold of "
        "0.001, IoU 0.45 for NMS and at most 300 detections per image. Scoring is "
        "pycocotools. Precision, recall and F1 are quoted at the confidence that maximises F1, "
        "the threshold you would deploy at."))
    left = [["mAP@0.5", f"{L['map50']:.4f}"], ["mAP@0.5:0.95", f"{L['map5095']:.4f}"],
            ["mAP@0.75", f"{L['map75']:.4f}"], ["AR@100", f"{L['ar100']:.4f}"],
            ["Precision", f"{L['precision']:.4f}"], ["Recall", f"{L['recall']:.4f}"],
            ["F1", f"{L['f1']:.4f}"], ["Operating confidence", f"{L['conf']:.3f}"],
            ["Detections (5000 images)", f"{L['detections']:,}"],
            ["Second scorer, mAP@0.5", f"{L['map50_indep']:.4f}"],
            ["Retained vs float @0.5", ret50 + " %"],
            ["Retained vs float @0.5:0.95", ret5095 + " %"]]
    right = [["Throughput, DPU only", f"{L['fps_dpu']:.1f} img/s"],
             ["Throughput, end to end", f"{L['fps_e2e']:.1f} img/s"],
             ["Wall clock per image", f"{L['lat_ms']:.1f} ms"],
             ["Full scored run", f"{L['fps_scored_run']:.1f} img/s (7:37)"],
             ["Power, idle", f"{L['p_idle']:.2f} W"], ["Power, under load", f"{L['p_load']:.2f} W"],
             ["Power, inference only", f"{L['p_attr']:.2f} W"],
             ["Energy per image", f"{L['j_img']:.3f} J"],
             ["Efficiency", f"{L['eff']:.2f} img/s/W"],
             ["Compute", f"{L['gop']:.2f} GOP/image"],
             ["Array utilisation", f"{L['array_eff']:.1f} %"],
             ["DDR footprint", f"{L['ddr_total']:.1f} MiB"]]
    A(table(["Accuracy", "value", "Hardware", "value"],
            [[a[0], a[1], b[0], b[1]] for a, b in zip(left, right)],
            aligns=["left", "right", "left", "right"], fs="8.5pt"))
    A(img(os.path.join(plots, "PR_curve.png"), 470,
          "Precision against recall. Grey lines are the 80 individual classes, the bold line "
          "their mean."))
    A(img(os.path.join(plots, "F1_curve.png"), 470,
          "F1 against confidence. The peak marks the operating point quoted above."))
    A(p(f"The best-F1 confidence is {L['conf']:.3f}, well below the {H['conf']:.3f} of the "
        "HardSwish yolov8s. A low optimum means the model's scores are compressed toward "
        "the bottom of the range, so useful detections are only separated from noise at a low "
        f"threshold. Precision at that point is {L['precision']:.3f} and recall "
        f"{L['recall']:.3f}: the model finds fewer objects than it labels correctly, as every "
        "model in the earlier report did."))

    # 6 ------------------------------------------------------------------------------
    A(pagebreak())
    A(h2("6. What quantization costs"))
    A(p("For the HardSwish models the loss from float to board had two causes that had to be "
        "separated with extra runs. Here there is one fewer: the activation is unchanged, so "
        "the gap between the float PC score and the board score is the cost of moving to the "
        "DPU &mdash; 8-bit arithmetic and the fixed-point LeakyReLU slope &mdash; with no "
        "substitution on top."))
    A(table(["", "float", "board INT8", "lost", "retained"],
            [["mAP@0.5", f"{fl50:.4f}", f"{L['map50']:.4f}",
              f"{fl50 - L['map50']:.4f}", ret50 + " %"],
             ["mAP@0.5:0.95", f"{fl5095:.4f}", f"{L['map5095']:.4f}",
              f"{fl5095 - L['map5095']:.4f}", ret5095 + " %"]]))
    A(img(figs["accuracy"], FULL_W,
          "Float ceiling and board result for both models. The HardSwish float bars are the "
          "stock SiLU model's float score, the reference the board result was measured against."))
    A(p(f"The model keeps {ret50}% of its float mAP@0.5 on the board. The HardSwish yolov8s "
        f"keeps {hret50}%. Part of that gap is the HardSwish model's activation substitution, "
        "which this model does not pay, but the two also differ in weights and head, so how much "
        "of the difference comes from the activation was not measured."))
    A(note("What was not run: an INT8 CPU simulation of this model, and a float run with the "
           "slope rounded as the DPU does. Either would split the loss above into its parts. "
           "The board number itself does not depend on them."))

    # 7 ------------------------------------------------------------------------------
    A(pagebreak())
    A(h2("7. Power, current and energy"))
    A(p("The VCK190 carries seventeen INA226 monitors on its supply rails, readable through "
        "the kernel's hwmon interface. Power was sampled every half second, first with the "
        "board idle for ten seconds, then while the DPU was driven with 400 back-to-back "
        "inferences. Sustained load is used for the reason given in the earlier report: "
        "sampling during an ordinary walk through the dataset understates the load."))
    A(note("Energy per image is power under load divided by a throughput measured without the "
           "sensors being polled, because polling slows the run. The throughput used is 14.2 "
           "images/s, from the 1000-image end-to-end run. The full 5000-image run was slower "
           "(10.9 images/s, see section 10.3), and energy computed from that figure would be "
           f"about {L['p_load'] / L['fps_scored_run']:.2f} J rather than {L['j_img']:.3f} J."))
    A(table(["", "idle (W)", "load (W)", "inference (W)", "J/img total", "J/img inference",
             "img/s/W"],
            [["LeakyReLU", f"{L['p_idle']:.2f}", f"{L['p_load']:.2f}", f"{L['p_attr']:.2f}",
              f"{L['j_img']:.3f}", f"{L['j_attr']:.3f}", f"{L['eff']:.2f}"],
             ["HardSwish (earlier session)", f"{H['p_idle']:.2f}", f"{H['p_load']:.2f}",
              f"{H['p_attr']:.2f}", f"{H['j_img']:.3f}", f"{H['j_attr']:.3f}",
              f"{H['eff']:.2f}"]]))
    A(img(figs["power_memory"], FULL_W,
          "Power, energy per image and DDR footprint. The HardSwish power rows come from the "
          "earlier session, so idle differs by 0.4 W."))
    A(p(f"The pattern of the earlier report holds. The board draws about {L['p_idle']:.0f}&nbsp;W "
        f"doing nothing and inference adds {L['p_attr']:.2f}&nbsp;W, so roughly "
        f"{100 * (1 - L['p_attr'] / L['p_load']):.0f}% of the power is the board being on. "
        "Energy per image is therefore set mostly by how long an image takes, not by what the "
        "network is."))
    A(h3("7.1 Per-rail breakdown"))
    A(p("Four rails move under load; the other thirteen are flat to within a few milliwatts."))
    A(table(["Rail", "idle (W)", "load (W)", "change (W)"],
            [["hwmon1", "5.125", "5.421", "+0.296"], ["hwmon5", "0.151", "0.409", "+0.257"],
             ["hwmon12", "1.103", "1.336", "+0.233"], ["hwmon0", "3.606", "3.712", "+0.106"],
             ["other 13 rails", "2.915", "2.922", "+0.007"],
             ["total", f"{L['p_idle']:.2f}", f"{L['p_load']:.2f}", f"+{L['p_attr']:.2f}"]]))
    A(p("hwmon5 more than doubles, as it did in the earlier report, and is most likely the "
        "programmable-logic supply feeding the DPU. The rails are identified only by hwmon "
        "index; mapping them to named supplies needs the board's device tree, which was not "
        "consulted. Rail voltage and current were not recorded."))

    # 8 ------------------------------------------------------------------------------
    A(pagebreak())
    A(h2("8. Area and memory"))
    A(p("FPGA resource area is a property of the DPU bitstream, not of a model, and this model "
        "shares the bitstream of the others (same fingerprint). LUT, DSP and BRAM counts would "
        "repeat the figures for any other model and were not available here, because the board "
        "runs a prebuilt Vitis AI image and implementation was never run. What varies by model "
        "is memory and how well the DPU's compute array is filled."))
    A(table(["", "weights", "activations", "I/O buffers", "DDR total", "xmodel", "microcode"],
            [["LeakyReLU", f"{L['ddr_w']:.2f}", f"{L['ddr_a']:.2f}",
              f"{L['ddr_in'] + L['ddr_out']:.2f}", f"{L['ddr_total']:.2f}",
              f"{L['xmodel_mib']:.2f}", f"{L['mc_kib']:.1f} KiB"],
             ["HardSwish", f"{H['ddr_w']:.2f}", f"{H['ddr_a']:.2f}",
              f"{H['ddr_in'] + H['ddr_out']:.2f}", f"{H['ddr_total']:.2f}",
              f"{H['xmodel_mib']:.2f}", f"{H['mc_kib']:.1f} KiB"]]))
    A('<p style="font-size:8.5pt;color:#444;">All in MiB except microcode.</p>')
    A(p("The footprint is about 3% smaller than the HardSwish yolov8s, tracking the missing "
        "head convolutions. Against 8&nbsp;GiB of board DRAM it is negligible."))
    A(h3("8.1 How well the model fills the DPU"))
    A(table(["", "useful GOP/img", "mapped GOP/img", "utilisation (%)", "DPU operators"],
            [["LeakyReLU", f"{L['gop']:.3f}", f"{L['gop_arch']:.3f}", f"{L['array_eff']:.1f}",
              str(L["dpu_ops"])],
             ["HardSwish", f"{H['gop']:.3f}", f"{H['gop_arch']:.3f}", f"{H['array_eff']:.1f}",
              str(H["dpu_ops"])]]))
    A(p("Utilisation is essentially the same: the channel widths are those of yolov8s in both, "
        "so the padding to the array's 32-channel geometry is the same. The leaky model does "
        f"{100 * (1 - L['gop'] / H['gop']):.0f}% less useful work per image."))

    # 9 ------------------------------------------------------------------------------
    A(pagebreak())
    A(h2("9. The two compared"))
    A(table(["Metric", "LeakyReLU", "HardSwish"],
            [["mAP@0.5", f"{L['map50']:.4f}", f"{H['map50']:.4f}"],
             ["mAP@0.5:0.95", f"{L['map5095']:.4f}", f"{H['map5095']:.4f}"],
             ["mAP@0.75", f"{L['map75']:.4f}", f"{H['map75']:.4f}"],
             ["AR@100", f"{L['ar100']:.4f}", f"{H['ar100']:.4f}"],
             ["Precision at best F1", f"{L['precision']:.4f}", f"{H['precision']:.4f}"],
             ["Recall at best F1", f"{L['recall']:.4f}", f"{H['recall']:.4f}"],
             ["F1", f"{L['f1']:.4f}", f"{H['f1']:.4f}"],
             ["Best-F1 confidence", f"{L['conf']:.3f}", f"{H['conf']:.3f}"],
             ["Retained vs own float @0.5 (%)", ret50, hret50],
             ["Retained vs own float @0.5:0.95 (%)", ret5095, hret5095],
             ["DPU-only throughput, same day (img/s)", f"{L['fps_dpu']:.1f}", f"{H['fps_dpu']:.1f}"],
             ["End-to-end, 1000 images, same day (img/s)", f"{L['fps_e2e']:.1f}",
              f"{H['fps_e2e']:.1f}"],
             ["Detections kept per image (1000-image run)", "154.6", "124.6"],
             ["Power under load (W)", f"{L['p_load']:.2f}", f"{H['p_load']:.2f}"],
             ["Energy (J/img)", f"{L['j_img']:.3f}", f"{H['j_img']:.3f}"],
             ["Parameters (M)", f"{L['params_m']:.2f}", f"{H['params_m']:.2f}"],
             ["Compute (GOP/img)", f"{L['gop']:.2f}", f"{H['gop']:.2f}"],
             ["DDR total (MiB)", f"{L['ddr_total']:.1f}", f"{H['ddr_total']:.1f}"]]))
    A(img(figs["throughput"], FULL_W,
          "Both models measured the same day with the same commands."))
    A(h3("9.1 What the numbers say"))
    A(p(f"<b>The HardSwish yolov8s is the more accurate deployment.</b> It leads on mAP@0.5 by "
        f"{H['map50'] - L['map50']:.3f} and on mAP@0.5:0.95 by "
        f"{H['map5095'] - L['map5095']:.3f}. Against that, the leaky model is the better "
        f"quantization citizen: it keeps {ret50}% of its float score rather than {hret50}%. "
        "Those two facts are compatible. The leaky model started from a lower float ceiling "
        f"({fl50:.3f} against {H['float_map50']:.3f}), and it lost proportionally less of it."))
    A(p("<b>Recall is the leaky model's one lead.</b> AR@100 is "
        f"{L['ar100']:.3f} against {H['ar100']:.3f}. It reaches that with a much lower operating "
        f"confidence ({L['conf']:.2f} against {H['conf']:.2f}), so the two are not at "
        "comparable thresholds, and precision there is lower."))
    A(p("<b>Hardware cost is a wash on the DPU and worse on the host.</b> DPU-only throughput "
        f"is {L['fps_dpu']:.1f} against {H['fps_dpu']:.1f} images/s &mdash; the same within "
        "noise, despite 11% less arithmetic, because both fill the array equally and the DPU "
        "is not limited by arithmetic here. End to end the leaky model is slower, "
        f"{L['fps_e2e']:.1f} against {H['fps_e2e']:.1f}, and it keeps about 24% more "
        "detections per image, which fits the host's decode and NMS being the extra cost. That "
        "link was not profiled."))
    A(h3("9.2 Which to use"))
    A(p("On this board and on these numbers, the HardSwish yolov8s: more accurate, faster "
        "end to end and, on the earlier session's power figures, lower in energy per image. The "
        "leaky model is the one to prefer if the aim is to train with an activation the DPU "
        "supports, because then no substitution is needed and no accuracy is spent on one. A "
        "fairer test of that idea would train both activations from the same architecture and "
        "recipe; this comparison cannot supply it."))

    # 10 -----------------------------------------------------------------------------
    A(pagebreak())
    A(h2("10. Checks, and what is not measured"))
    A(h3("10.1 Checks that were run"))
    A(p("The wrapper was run before quantization and confirmed zero activation substitutions "
        "and three outputs of shape (1,&nbsp;144,&nbsp;80,&nbsp;80), (1,&nbsp;144,&nbsp;40,"
        "&nbsp;40) and (1,&nbsp;144,&nbsp;20,&nbsp;20). The compiled graph was inspected before "
        "deployment: one DPU subgraph and a single CPU operator type, which the board ships. "
        "The xmodel's md5 on the board matched the host. A 50-image smoke test ran before the "
        "full run."))
    A(p("The result was scored twice. pycocotools gives mAP@0.5 "
        f"{L['map50']:.4f}; an independent implementation that shares no matching code with "
        f"it gives {L['map50_indep']:.4f}, a difference of "
        f"{abs(L['map50'] - L['map50_indep']):.4f}. In the earlier report the two agreed within "
        "0.003 on every model, so this gap is somewhat larger, and the reason was not "
        "investigated."))
    A(h3("10.2 Not measured"))
    A(p("<b>Where the loss comes from.</b> No INT8 simulation and no slope-rounded float run, "
        "so quantization and the fixed-point slope are not separated (section 6)."))
    A(p("<b>Rail voltages and currents, FPGA utilisation and thermal behaviour.</b> As in the "
        "earlier report, for the same reasons."))
    A(p("<b>Training.</b> The training recipe, dataset and the fork's modifications to the "
        "model were not supplied or inspected. Everything about the model's origin above is "
        "read from the checkpoint and its file name."))
    A(p("<b>Host profiling.</b> The claim that decode and NMS explain the end-to-end gap is an "
        "inference from the detection counts."))
    A(h3("10.3 One unresolved discrepancy"))
    A(p("The earlier report records the HardSwish yolov8s at 22.24 images/s DPU-only. The same "
        "benchmark command run today on the same xmodel gives 42.4. The harness has been "
        "modified since (uncommitted changes to board_eval_vck190.py), and the benchmark loop "
        "now decodes its frames once up front; whether the earlier measurement did the same "
        "was not checked, and nothing was traced to the difference. The "
        "comparison in this report therefore uses only the same-day figures for both models, "
        "and the earlier DPU-only number should not be set beside today's."))
    A(p("The full 5000-image scored run of the leaky model took 7:37, 10.9 images/s, while a "
        "1000-image run of the same model with no scoring took 70 s, 14.2 images/s. The scored "
        "run was done with the board's disk full to the last block, which may have contributed. "
        "It was not repeated, so energy per image above uses the 14.2 figure and the alternative "
        "is shown in section 7."))

    # appendices ----------------------------------------------------------------------
    A(pagebreak())
    A(h2("Appendix A. Reproducing this"))
    A('<pre style="font-size:8.5pt;background:#f4f4f4;padding:6pt;border:1px solid #ddd;">'
      "cd Yolo_v8_Versal_Implementation\n"
      "./v8_run.sh python -u v8_quantize.py --weights yolov8s_leaky.pt --act silu \\\n"
      "    --output_dir quantize_result/yolov8s_leaky --quant_mode calib\n"
      "./v8_run.sh python -u v8_quantize.py --weights yolov8s_leaky.pt --act silu \\\n"
      "    --output_dir quantize_result/yolov8s_leaky --quant_mode test\n"
      "./v8_compile.sh yolov8s leaky\n"
      "./deploy_to_board.sh yolov8s_leaky"
      '</pre>')
    A(p("<tt>--act silu</tt> is the wrapper's name for &ldquo;leave activations alone&rdquo;; "
        "it replaces nothing in a model that has no SiLU. On the board:"))
    A('<pre style="font-size:8.5pt;background:#f4f4f4;padding:6pt;border:1px solid #ddd;">'
      "python3 board_eval_vck190.py --arch v8 --model yolov8s_leaky_vck190.xmodel \\\n"
      "    --images val2017 --annotations ann/instances_val2017.json \\\n"
      "    --out-json preds_yolov8s_leaky.json --out-csv metrics_yolov8s_leaky.csv\n\n"
      "python3 measure_power.py --model yolov8s_leaky_vck190.xmodel \\\n"
      "    --images val2017 --benchmark 400 --reference-fps 14.2"
      '</pre>')
    A(p("The float reference is "
        "<tt>v8_eval_quantized.py --weights yolov8s_leaky.pt --act silu --float --letterbox "
        "--limit 0</tt>, scored with pycocotools."))
    A(h2("Appendix B. Files"))
    A(table(["Path", "Contents"],
            [["compiled/yolov8s_leaky/yolov8s_leaky_vck190.xmodel", "the deployable model, "
              "md5 9a0df69894f095da55a60a8de9d208c6"],
             ["quantize_result/yolov8s_leaky/", "calibration output and the INT8 xmodel"],
             ["board_results/preds_yolov8s_leaky.json", "raw COCO predictions, all 5000 images"],
             ["board_results/metrics_yolov8s_leaky.csv", "the board run's own score file"],
             ["board_results/metrics/yolov8s_leaky_metrics.csv", "every figure here, with its source"],
             ["board_results/plots_yolov8s_leaky/", "PR, P, R, F1 curves and confusion matrix"],
             ["sim_results/preds_yolov8s_leaky_float_lb.json", "float reference predictions"]],
            aligns=["left", "left"], fs="8.5pt"))
    A("</body></html>")
    return s


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    html = "".join(build())
    out = os.path.join(OUT, "report.html")
    with open(out, "w") as fh:
        fh.write(html)
    print("wrote", out, f"({len(html):,} bytes)")
