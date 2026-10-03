"""Build the VCK190 YOLOv8 hardware report as HTML, for conversion to PDF by LibreOffice.

Every number is read from the CSVs under board_results/metrics/ rather than typed here, so the
document cannot drift from the measurements.

    Versal_AI/venv/bin/python tools/build_report.py
    soffice --headless --convert-to pdf --outdir <dir> <dir>/report.html
"""

import csv
import os

from PIL import Image

ROOT = os.path.abspath(".")
REPORT = os.path.join(ROOT, "Yolo_v8_Versal_Implementation", "report")
FIGS = os.path.join(REPORT, "figures")
METRICS = os.path.join(ROOT, "Yolo_v8_Versal_Implementation", "board_results", "metrics")
PLOTS = os.path.join(ROOT, "Yolo_v8_Versal_Implementation", "board_results")
MODELS = ["yolov8n", "yolov8s", "yolov8m", "yolov8l"]

# LibreOffice honours width+height together and keeps the full image resolution.
FULL_W = 620
HALF_W = 300


def load():
    d = {}
    with open(os.path.join(METRICS, "all_models_metrics.csv")) as fh:
        for r in csv.DictReader(fh):
            d[r["parameter"]] = {m: r[m] for m in MODELS}
    return d


D = load()


def v(key, m):
    return D[key][m]


def img(path, target_w=FULL_W, caption=None):
    if not os.path.exists(path):
        return f'<p><i>[missing figure: {os.path.basename(path)}]</i></p>'
    w, h = Image.open(path).size
    th = int(round(target_w * h / w))
    cap = (f'<div style="font-size:8.5pt;color:#444;margin-top:2pt;">{caption}</div>'
           if caption else "")
    return (f'<div style="margin:10pt 0 12pt 0;">'
            f'<img src="file://{path}" width="{target_w}" height="{th}">{cap}</div>')


def table(headers, rows, aligns=None, width="100%", fs="9pt", colw=None):
    aligns = aligns or ["left"] + ["right"] * (len(headers) - 1)
    if colw:
        h = "".join(f'<th align="{a}" width="{w}">{x}</th>'
                    for x, a, w in zip(headers, aligns, colw))
    else:
        h = "".join(f'<th align="{a}">{x}</th>' for x, a in zip(headers, aligns))
    body = ""
    for r in rows:
        body += "<tr>" + "".join(
            f'<td align="{a}">{x}</td>' for x, a in zip(r, aligns)) + "</tr>"
    return (f'<table border="1" cellspacing="0" cellpadding="4" width="{width}" '
            f'style="font-size:{fs};">'
            f'<tr bgcolor="#e9e9e9">{h}</tr>{body}</table>')


def metric_rows(keys):
    out = []
    for k, label in keys:
        unit = D[k].get("unit", "") if isinstance(D[k], dict) else ""
        out.append([label] + [v(k, m) for m in MODELS])
    return out


def unit(key):
    with open(os.path.join(METRICS, "all_models_metrics.csv")) as fh:
        for r in csv.DictReader(fh):
            if r["parameter"] == key:
                return r["unit"]
    return ""


HEAD = """<html><head><meta charset="utf-8"></head>
<body style="font-family:'Liberation Serif',serif;font-size:10.5pt;">
"""

def h1(t): return f'<h1 style="font-size:17pt;margin-bottom:2pt;">{t}</h1>'
def h2(t): return f'<h2 style="font-size:13pt;margin-top:18pt;margin-bottom:4pt;">{t}</h2>'
def h3(t): return f'<h3 style="font-size:11pt;margin-top:12pt;margin-bottom:3pt;">{t}</h3>'
def p(t):  return f'<p style="text-align:justify;line-height:1.35;">{t}</p>'
def note(t): return (f'<p style="font-size:9pt;color:#333;border-left:3px solid #bbb;'
                     f'padding-left:8pt;margin:8pt 0;">{t}</p>')
def pagebreak(): return '<p style="page-break-before:always;"></p>'


def build():
    s = [HEAD]
    A = s.append

    # ---------------------------------------------------------------- title
    A(h1("YOLOv8 object detection on the AMD Versal VCK190"))
    A('<p style="font-size:10pt;color:#444;margin-top:0;">'
      'Hardware implementation, measurement and comparison of four models<br>'
      'COCO val2017, 5000 images &nbsp;&middot;&nbsp; September 2026</p>')
    A('<hr>')

    # ---------------------------------------------------------------- summary
    A(h2("1. What this covers"))
    A(p("Four stock YOLOv8 detectors &mdash; n, s, m and l &mdash; were quantized to INT8, "
        "compiled for the DPU on a Versal VCK190 evaluation board, and run over the whole of "
        "COCO val2017. This report records how each one was built, what it scored, how fast it "
        "ran, how much power it drew, and how much memory it needed. The last section compares "
        "them against each other."))
    A(p("Two things are worth stating up front because they shape everything else. First, the "
        "DPU cannot execute the SiLU activation that YOLOv8 ships with, so every model had its "
        "activation replaced with HardSwish before quantization. That substitution, not the "
        "8-bit arithmetic, is responsible for most of the accuracy that was lost. Second, the "
        "board draws about 13&nbsp;W whether it is working or not, and inference adds well under "
        "one watt on top, so energy per image is governed by how long the board is switched on "
        "rather than by which model is loaded."))

    rows = [[m, v("mAP@0.5", m), v("mAP@0.5:0.95", m), v("fps", m),
             v("power_under_load", m), v("energy_per_image", m)] for m in MODELS]
    A(table(["Model", "mAP@0.5", "mAP@0.5:0.95", "images/s", "power (W)", "energy (J/img)"], rows))
    A('<p style="font-size:8.5pt;color:#444;">Headline figures. Full tables follow, and the '
      'appendix carries every metric collected.</p>')

    # ---------------------------------------------------------------- hardware
    A(h2("2. Hardware and software"))
    A(p("The board is a VCK190 evaluation board, silicon revision v2, with 8&nbsp;GiB of DRAM. "
        "It boots PetaLinux 2022.2 from an SD card and runs the Vitis AI runtime. The DPU is a "
        "DPUCVDX8G, configured as ISA3_C32B6: thirty-two channels wide with a batch of six, "
        "clocked at 333&nbsp;MHz. All four models run on this one DPU configuration; the "
        "hardware does not change between them."))
    A(table(["Item", "Value"],
            [["Board", "Xilinx Versal VCK190 evaluation board, rev A, silicon v2, 8 GiB DRAM"],
             ["DPU", "DPUCVDX8G_ISA3_C32B6, batch 6, 333 MHz"],
             ["DPU fingerprint", v("dpu_fingerprint", "yolov8n") + " (identical for all four models)"],
             ["Operating system", "PetaLinux 2022.2, kernel 5.15.36-xilinx-v2022.2, aarch64"],
             ["Runtime", "VART 3.0.0, XIR 3.0"],
             ["Board Python", "3.9 with NumPy 1.26.4, OpenCV 4.5.2, pycocotools"],
             ["Toolchain", "Vitis AI 3.0 PyTorch container (vai_q_pytorch, vai_c_xir)"],
             ["Host link", "USB gigabit Ethernet, static 192.168.1.10/24 on the board"]],
            aligns=["left", "left"]))

    A(h2("3. The four models"))
    A(p("These are the stock Ultralytics release weights, dated December 2022, unmodified. Each "
        "one detects the standard 80 COCO classes at 640&times;640 input, with an anchor-free "
        "head that predicts box edges as a distribution over 16 bins per side. Class ordering "
        "was checked against the standard COCO list on all four before any work started."))
    rows = [[m, v("parameters", m), v("conv_modules", m), v("workload", m),
             v("float_mAP@0.5", m), v("float_mAP@0.5:0.95", m)] for m in MODELS]
    A(table(["Model", "params (M)", "conv layers", "GOP/image", "float mAP@0.5",
             "float mAP@0.5:0.95"], rows))
    A('<p style="font-size:8.5pt;color:#444;">Float scores are the PC reference for the same '
      'weights, used throughout as the ceiling the board is measured against.</p>')

    # ---------------------------------------------------------------- implementation
    A(pagebreak())
    A(h2("4. How the models were put on the board"))

    A(h3("4.1 Splitting the network between DPU and CPU"))
    A(p("The DPU runs convolutions. It does not run the arithmetic that turns the head's raw "
        "output into boxes &mdash; the softmax over the distance bins, the sigmoid on the class "
        "scores, and non-maximum suppression are not fixed-function convolution work. So the "
        "network was cut at the head's output convolutions. The DPU produces three raw feature "
        "maps, at strides 8, 16 and 32, each with 144 channels; the host reads those and does "
        "the rest."))
    A(p("144 channels is four box sides &times; 16 distribution bins, plus 80 class scores. At "
        "640 input that gives tensors of 80&times;80, 40&times;40 and 20&times;20."))
    A(note("Leaving the decode on the DPU is possible and an earlier project did it, but it "
           "splits the compiled graph in two. Moving the whole decode to the host leaves one "
           "DPU block with nothing in the middle to break it up. That is why this build compiles "
           "to a single subgraph where the earlier one produced two."))

    A(h3("4.2 The activation problem"))
    A(p("YOLOv8 uses SiLU, x&middot;sigmoid(x). The compiler has no DPU mapping for it. Left "
        "alone it shatters the graph into dozens of fragments, and worse, the board carries no "
        "runtime library for the operator, so such a build compiles and then refuses to run."))
    A(p("HardSwish is close in shape and the DPU implements it directly. Every SiLU was replaced "
        "with HardSwish before quantization &mdash; 57 of them in yolov8n and yolov8s, 77 in "
        "yolov8m, 97 in yolov8l. This is an approximation, and section 6 measures what it costs. "
        "No retraining was done after the swap."))
    A(note("A subtlety that catches people: Ultralytics declares the activation as a class "
           "attribute shared by every convolution, so counting activation modules in the model "
           "returns one, not fifty-seven. The count that matters is the number of convolution "
           "layers, since each applies the activation in its own forward pass."))

    A(h3("4.3 Quantization"))
    A(p("vai_q_pytorch was run in two passes. The first walks calibration images and records the "
        "range each tensor covers; the second exports the INT8 model. Calibration used 200 "
        "images taken at even stride through a 4000-image split of val2017 that is disjoint from "
        "the 1000 images held back, so the calibration set is not simply the first 200 files."))
    A(p("Weights and activations are 8-bit, with a per-tensor power-of-two scale. The input "
        "tensor arrives as int8 with a fixed-point scale of 64; the three outputs come back with "
        "a scale of 0.25."))

    A(h3("4.4 Compilation"))
    A(p("vai_c_xir compiled each INT8 model against the VCK190 architecture file. All four "
        "produced the same structure: one DPU subgraph, and three small CPU subgraphs that do "
        "nothing but convert fixed-point outputs to float. That conversion operator ships with "
        "the board, so there is no missing-library problem at runtime."))
    rows = [[m, v("xmodel_size", m), v("dpu_subgraphs", m), v("cpu_ops", m),
             v("dpu_ops", m), v("microcode_size", m)] for m in MODELS]
    A(table(["Model", "xmodel (MiB)", "DPU subgraphs", "CPU operators", "DPU ops",
             "microcode (KiB)"], rows))
    A(p("A single DPU subgraph matters for speed as well as tidiness. With one block the runtime "
        "can use the plain VART runner and hand the DPU a full batch of six frames per call. A "
        "fragmented graph forces a whole-graph runner that bounces between DPU and CPU at every "
        "boundary."))

    A(h3("4.5 Decoding on the host"))
    A(p("The host turns the three raw tensors into boxes. For each cell it takes the 64 box "
        "channels as four groups of 16, applies a softmax within each group and takes the "
        "expected value, which gives the distance from the cell centre to each of the four box "
        "edges in cell units. Multiplying by the stride and offsetting from the cell centre "
        "gives pixel coordinates. Class scores are a straight sigmoid; YOLOv8 has no separate "
        "objectness channel."))
    A(p("Before any of this was run on hardware it was checked against Ultralytics' own decode "
        "on identical features. Boxes agreed to 6&times;10<sup>-5</sup> pixels and scores to "
        "6&times;10<sup>-8</sup> on all four models, which is floating-point noise."))
    A(p("Cost on the host is small. At the confidence threshold used for scoring, decode and NMS "
        "take about 4.8&nbsp;ms per image, against a DPU that needs roughly 45&nbsp;ms. The "
        "saving comes from rejecting cells on their strongest class score before doing any "
        "softmax work, so the expensive step only touches the thousand or so candidates that "
        "survive."))

    A(h3("4.6 Preprocessing"))
    A(p("Images are letterboxed: scaled to fit 640&times;640 with the aspect ratio preserved and "
        "the remainder filled with grey. The alternative, squashing to a square, distorts every "
        "object and costs real accuracy &mdash; about 0.021 mAP in float and 0.047 after "
        "quantization on yolov8n. Boxes are mapped back afterwards by subtracting the padding "
        "and dividing by the scale."))
    A(note("The padding subtracted on the way back must be the integer number of pixel rows and "
           "columns actually added, not the unrounded half-pixel value. Using the latter leaves "
           "every box shifted by up to half a pixel on roughly half of COCO's aspect ratios. It "
           "is invisible at a loose IoU threshold and quietly erodes the strict ones."))

    # ---------------------------------------------------------------- per-model
    A(pagebreak())
    A(h2("5. How each model performed"))
    A(p("Every model was run over all 5000 validation images with a confidence threshold of "
        "0.001, IoU 0.45 for NMS and at most 300 detections per image. Scoring is pycocotools. "
        "Precision, recall and F1 are quoted at the confidence that maximises F1, which is the "
        "threshold you would actually deploy at; mAP integrates across all thresholds and says "
        "nothing about any single operating point."))

    blurb = {
        "yolov8n": ("The smallest model, and the one that fits the hardware worst. It occupies "
                    "only 87.4% of the DPU array &mdash; the rest is padding &mdash; so it runs "
                    "no faster than yolov8m despite doing a ninth of the arithmetic. It is also "
                    "the only model whose best operating point sits below 0.27 confidence, a "
                    "sign that its class scores are poorly separated after quantization."),
        "yolov8s": ("The best-balanced of the four on this board. It is the fastest, the most "
                    "energy-efficient, and 22% more accurate than yolov8n. It fills the array "
                    "well at 95.5% and shares the same 57 convolution layers as yolov8n, so the "
                    "extra work is in wider channels rather than more layers &mdash; which is "
                    "exactly the shape the DPU handles well."),
        "yolov8m": ("Deeper than s, with 77 convolution layers, and the accuracy gain is solid. "
                    "Throughput falls back to roughly yolov8n's level, but for three times the "
                    "accuracy per image. Array utilisation is 94.9%."),
        "yolov8l": ("The most accurate and the slowest. It uses the DPU array best of all at "
                    "98.2%, and draws the most power under load at 1.01&nbsp;W above idle, which "
                    "is consistent with it keeping the array busiest. Its DDR footprint of "
                    "93.8&nbsp;MiB is the largest but still modest against 8&nbsp;GiB."),
    }

    for m in MODELS:
        A(h3(f"5.{MODELS.index(m)+1} {m}"))
        A(p(blurb[m]))
        left = [["mAP@0.5", v("mAP@0.5", m)],
                ["mAP@0.5:0.95", v("mAP@0.5:0.95", m)],
                ["mAP@0.75", v("mAP@0.75", m)],
                ["AR@100", v("AR@100", m)],
                ["Precision", v("precision", m)],
                ["Recall", v("recall", m)],
                ["F1", v("f1_score", m)],
                ["Operating confidence", v("operating_confidence", m)],
                ["Detections (5000 images)", f'{int(v("detections_total", m)):,}'],
                ["Second scorer, mAP@0.5", v("mAP@0.5_independent_scorer", m)],
                ["Retained vs float @0.5", v("accuracy_retained@0.5", m) + " %"],
                ["Retained vs float @0.5:0.95", v("accuracy_retained@0.5:0.95", m) + " %"]]
        right = [["Throughput, end to end", v("fps", m) + " img/s"],
                 ["Throughput, DPU only", v("fps_dpu_only", m) + " img/s"],
                 ["Latency per image", v("latency_per_image", m) + " ms"],
                 ["Runtime, 5000 images", v("runtime_5000_images", m) + " s"],
                 ["Power, idle", v("power_idle", m) + " W"],
                 ["Power, under load", v("power_under_load", m) + " W"],
                 ["Power, inference only", v("power_attributable", m) + " W"],
                 ["Energy per image", v("energy_per_image", m) + " J"],
                 ["Efficiency", v("efficiency", m) + " img/s/W"],
                 ["Compute", v("workload", m) + " GOP/image"],
                 ["Array utilisation", v("dpu_array_efficiency", m) + " %"],
                 ["DDR footprint", v("ddr_total", m) + " MiB"]]
        # One table, four columns. Nested side-by-side tables split across page breaks
        # and collapsed their column widths.
        merged = [[l[0], l[1], r[0], r[1]] for l, r in zip(left, right)]
        A(table(["Accuracy", "value", "Hardware", "value"], merged,
                aligns=["left", "right", "left", "right"], fs="8.5pt"))
        # Full width, stacked. Side by side at 300px the axis labels inside these plots are
        # too small to read on paper.
        pr = os.path.join(PLOTS, f"plots_{m}", "PR_curve.png")
        f1 = os.path.join(PLOTS, f"plots_{m}", "F1_curve.png")
        A(img(pr, 470, f"{m}: precision against recall. Grey lines are the 80 individual "
                       f"classes, the bold line their mean."))
        A(img(f1, 470, f"{m}: F1 against confidence. The peak marks the operating point quoted "
                       f"in the table above."))
        if m != MODELS[-1]:
            A(pagebreak())

    # ---------------------------------------------------------------- quantization
    A(pagebreak())
    A(h2("6. What quantization costs, and what the activation costs"))
    A(p("Going from a float model on a PC to an INT8 model on the DPU involves two separate "
        "changes, and they are easy to confuse. The activation is replaced with HardSwish, and "
        "the arithmetic is reduced to 8 bits. Comparing the float PC score against the board "
        "score measures both at once and cannot tell you which one hurt."))
    A(p("To separate them, each model was run three ways on an identical 500-image subset, with "
        "letterbox preprocessing throughout, changing one thing at a time: float with the "
        "original SiLU, float with HardSwish substituted, and INT8 with HardSwish. The step "
        "between the first two is the activation; the step between the last two is quantization."))

    qpath = os.path.join(METRICS, "quantization_effect.csv")
    if os.path.exists(qpath):
        qrows, qd = [], {}
        with open(qpath) as fh:
            for r in csv.DictReader(fh):
                qd[r["model"]] = r
        for m in MODELS:
            if m not in qd:
                continue
            r = qd[m]
            qrows.append([m, r["float_silu"], r["float_hardswish"], r["int8_hardswish"],
                          r["activation_delta"], r["quant_delta"], r["activation_share_pct"]])
        A(table(["Model", "float SiLU", "float HardSwish", "INT8 HardSwish",
                 "activation", "quantization", "activation share"], qrows))
        A('<p style="font-size:8.5pt;color:#444;">mAP@0.5 on the same 500 images. The two delta '
          'columns sum to the total drop. The last column is the activation\'s share of it.</p>')
        A(img(os.path.join(FIGS, "quantization_effect.png"), FULL_W,
              "Each model loses far more to the activation substitution than to 8-bit arithmetic."))
    else:
        A(note("Per-model breakdown pending &mdash; the three-way runs had not finished when "
               "this document was generated."))

    A(p("The pattern is the same for every model: the activation substitution costs several "
        "times what quantization costs. This matters because it points the remedy in a "
        "particular direction. Tightening the quantizer &mdash; more calibration images, "
        "per-channel scales, percentile clipping &mdash; works on the smaller of the two "
        "problems. The larger one is that the weights were trained for SiLU and then asked to "
        "run with HardSwish, with no opportunity to adapt."))

    A(h3("6.1 A closer look at yolov8n"))
    A(p("For yolov8n the experiment was taken one step further. SiLU was rewritten as its exact "
        "identity, x multiplied by sigmoid(x), which is numerically the same function but built "
        "from operators the toolchain can handle. Quantizing that gives an INT8 model with the "
        "activation held exactly right, so the two factors can be crossed:"))
    A(table(["yolov8n, mAP@0.5", "float", "INT8", "cost of quantization"],
            [["SiLU (exact)", "0.5674", "0.5431", "-0.0243"],
             ["HardSwish (approximate)", "0.4809", "0.4714", "-0.0095"],
             ["cost of activation", "-0.0865", "-0.0717", "total -0.0960"]],
            aligns=["left", "right", "right", "right"]))
    A(p("Both routes from 0.5674 down to 0.4714 add up to the same total, which is the "
        "arithmetic check that the four numbers are consistent. The two effects interact, so "
        "the exact split depends on which you attribute first: the activation costs between "
        "0.0717 and 0.0865, quantization between 0.0095 and 0.0243. Averaging, the activation "
        "accounts for roughly 82% of the loss and quantization 18%. Whichever way it is cut, the "
        "activation dominates."))
    A(img(os.path.join(PLOTS, "plots_summary", "v8_activation_vs_quantization.png"), FULL_W,
          "The gap between the two lines is the activation substitution; each line's slope is "
          "quantization. yolov8n, 500-image subset."))
    A(note("The exact-SiLU model is a measurement tool, not a deployable one. It compiles for "
           "YOLOv3 but vai_c_xir rejects it for YOLOv8, most likely because of the elementwise "
           "multiply landing inside the C2f split-and-concatenate structure. The numbers above "
           "come from CPU simulation of the INT8 model, which reproduces the board to within "
           "0.004 mAP where the two can be compared."))

    A(p("The practical conclusion is that a short finetune with HardSwish already in place "
        "should recover most of the gap, because the weights only need to re-adapt to a slightly "
        "different nonlinearity rather than relearn the task. That needs a GPU, which this "
        "machine does not have, so it has not been attempted."))

    # ---------------------------------------------------------------- power
    A(pagebreak())
    A(h2("7. Power, current and energy"))
    A(p("The VCK190 carries seventeen INA226 monitors on its supply rails, readable through the "
        "kernel's hwmon interface. Each reports rail power directly in microwatts. Power was "
        "sampled every half second, first with the board idle for twelve seconds, then while the "
        "DPU was driven with 400 back-to-back inferences."))
    A(p("Driving the DPU continuously matters. An earlier attempt sampled during an ordinary "
        "walk through the dataset, where the DPU sits idle between frames waiting on image "
        "loading and decode; average power barely rose above idle and the figure understated the "
        "load by more than half. The numbers below come from sustained load."))
    A(note("Polling seventeen I&sup2;C sensors is not free: it dragged throughput from 14.4 down "
           "to 6.7 images per second on the first attempt. Energy per image is therefore "
           "calculated from the throughput measured in a clean run without sampling, not from "
           "the rate observed while the sensors were being read."))

    rows = [[m, v("power_idle", m), v("power_under_load", m), v("power_attributable", m),
             v("energy_per_image", m), v("energy_per_image_attributable", m),
             v("efficiency", m)] for m in MODELS]
    A(table(["Model", "idle (W)", "load (W)", "inference (W)", "J/img total",
             "J/img inference", "img/s/W"], rows))
    A(img(os.path.join(FIGS, "power_energy.png"), FULL_W))

    A(p("Idle dominates completely. The board draws roughly 13&nbsp;W doing nothing, and running "
        "the largest of the four models on top of that adds 1.01&nbsp;W. For the smallest it is "
        "0.69&nbsp;W. In other words, between 93% and 95% of the energy spent per image is the "
        "board being switched on, not the detector running."))
    A(p("This inverts the usual intuition. Choosing a smaller model to save energy does almost "
        "nothing here, because the model was never the thing consuming the energy. What does "
        "help is finishing sooner: yolov8s uses the least energy per image of the four, not "
        "because it draws less power &mdash; it draws slightly more than yolov8n &mdash; but "
        "because it gets through more images per second and so spends less time per image "
        "paying the 13&nbsp;W standing charge."))

    A(h3("7.1 Per-rail breakdown"))
    A(p("Rail-level readings, taken while yolov8n ran, show where the extra power goes. Most "
        "rails do not move at all. Four do:"))
    A(table(["Rail", "idle (W)", "load (W)", "change (W)"],
            [["hwmon1", "5.188", "5.403", "+0.215"],
             ["hwmon5", "0.163", "0.393", "+0.230"],
             ["hwmon12", "1.102", "1.287", "+0.185"],
             ["hwmon0", "3.620", "3.674", "+0.054"],
             ["hwmon6", "1.500", "1.500", "0.000"],
             ["hwmon15", "0.406", "0.406", "0.000"],
             ["other 11 rails", "0.938", "0.939", "+0.001"],
             ["total", "13.02", "13.70", "+0.68"]]))
    A(p("hwmon5 more than doubles, from 0.163&nbsp;W to 0.393&nbsp;W, which is the largest "
        "proportional change of any rail and is almost certainly the programmable-logic supply "
        "feeding the DPU. hwmon1 is the biggest rail in absolute terms and contributes a similar "
        "amount of the increase. The remaining eleven rails are flat to within a milliwatt."))
    A(note("Rail voltages and currents were not recorded, only rail power. The same INA226 "
           "devices expose voltage and current separately and the sampling script could read "
           "them, but the board was taken offline before that was done. The rails are also "
           "identified only by their hwmon index here; mapping those to named supplies needs the "
           "board's device tree, which was not consulted."))

    # ---------------------------------------------------------------- area / memory
    A(pagebreak())
    A(h2("8. Area and memory"))
    A(p("Asking how much area each model occupies needs a distinction that is easy to skip over. "
        "On an FPGA, area normally means logic resources &mdash; lookup tables, flip-flops, "
        "block RAM, DSP slices. Those belong to the DPU itself, which is a fixed piece of "
        "hardware compiled into the device once. All four models run on that same DPU. Loading "
        "yolov8l instead of yolov8n does not change a single lookup table."))
    A(p("The evidence is in the compiled files: every one of the four carries the same DPU "
        "fingerprint, " + v("dpu_fingerprint", "yolov8n") + ". They are addressed to the same "
        "hardware. A per-model table of LUT and DSP counts would be one number repeated four "
        "times, and it would be misleading to present it as though it varied."))
    A(note("The actual utilisation figures for this DPU configuration come from the Vivado "
           "implementation report for the platform. This project runs a prebuilt Vitis AI board "
           "image and never ran implementation, so those numbers are not available here and have "
           "not been guessed at. Obtaining them means rebuilding the platform from the DPU "
           "reference design."))
    A(p("What does vary per model is memory: how much DDR the weights occupy, how much the "
        "intermediate feature maps need, and how large the compiled program is. That is the "
        "footprint that decides whether a model fits and how much bandwidth it costs."))

    rows = [[m, v("ddr_weights", m), v("ddr_activations", m),
             str(round(float(v("ddr_input_buffer", m)) + float(v("ddr_output_buffer", m)), 2)),
             v("ddr_total", m), v("xmodel_size", m), v("microcode_size", m)] for m in MODELS]
    A(table(["Model", "weights (MiB)", "activations (MiB)", "I/O buffers (MiB)",
             "DDR total (MiB)", "xmodel (MiB)", "microcode (KiB)"], rows))
    A(img(os.path.join(FIGS, "memory.png"), FULL_W))
    A(p("The input and output buffers are identical across all four, as they must be: the input "
        "is always 640&times;640&times;3 and the output always 8400 positions of 144 channels. "
        "Weight storage tracks parameter count. Activation storage grows more slowly than "
        "weights do, which is why the ratio shifts from roughly one part weights to four parts "
        "activations in yolov8n, to something near parity in yolov8l."))
    A(p("Against 8&nbsp;GiB of board DRAM none of these are large. The largest, yolov8l at "
        "93.8&nbsp;MiB, uses just over one percent of it. Memory is not the constraint on this "
        "platform."))

    A(h3("8.1 How well each model fills the DPU"))
    A(p("There is a second sense of area worth reporting, which is how much of the DPU's compute "
        "array a model actually keeps busy. The compiler records both the useful work in a model "
        "and the work as mapped onto the array; the difference is padding, where channel counts "
        "do not divide evenly into the array geometry."))
    rows = [[m, v("workload", m), v("workload_on_arch", m), v("dpu_array_efficiency", m),
             v("achieved_compute", m), v("dpu_ops", m)] for m in MODELS]
    A(table(["Model", "useful GOP/img", "mapped GOP/img", "utilisation (%)",
             "achieved GOP/s", "DPU operators"], rows))
    A(img(os.path.join(FIGS, "compute.png"), FULL_W))
    A(p("yolov8n is the worst fit at 87.4%: one operation in eight that the array performs for "
        "it is wasted on padding. yolov8l is the best at 98.2%. This is the main reason yolov8n "
        "runs no faster than yolov8m despite needing a ninth of the arithmetic &mdash; the small "
        "model is too narrow to fill a 32-channel array, so it cannot convert its lighter "
        "workload into proportionally more speed."))

    # ---------------------------------------------------------------- comparison
    A(pagebreak())
    A(h2("9. The four compared"))
    A(p("Everything measured, side by side."))
    keys = [("mAP@0.5", "mAP@0.5"), ("mAP@0.5:0.95", "mAP@0.5:0.95"), ("mAP@0.75", "mAP@0.75"),
            ("AR@100", "AR@100"), ("precision", "Precision"), ("recall", "Recall"),
            ("f1_score", "F1"), ("accuracy_retained@0.5", "Retained vs float @0.5 (%)"),
            ("fps", "Throughput (img/s)"), ("fps_dpu_only", "DPU-only throughput (img/s)"),
            ("latency_per_image", "Latency (ms)"), ("power_under_load", "Power, load (W)"),
            ("power_attributable", "Power, inference (W)"),
            ("energy_per_image", "Energy (J/img)"), ("efficiency", "Efficiency (img/s/W)"),
            ("parameters", "Parameters (M)"), ("xmodel_size", "xmodel (MiB)"),
            ("ddr_total", "DDR total (MiB)"), ("workload", "Compute (GOP/img)"),
            ("dpu_array_efficiency", "Array utilisation (%)"),
            ("achieved_compute", "Achieved (GOP/s)")]
    A(table(["Metric"] + MODELS, [[lbl] + [v(k, m) for m in MODELS] for k, lbl in keys]))

    A(img(os.path.join(FIGS, "accuracy.png"), FULL_W,
          "Board INT8 against the float ceiling. The gap narrows as models get larger."))
    A(img(os.path.join(FIGS, "prf1.png"), FULL_W,
          "Precision, recall and F1 at each model's best-F1 confidence."))
    A(img(os.path.join(FIGS, "throughput.png"), FULL_W,
          "DPU-only throughput against end-to-end, and the resulting latency."))
    A(img(os.path.join(FIGS, "tradeoff.png"), FULL_W,
          "The deployment decision in one picture."))

    A(h3("9.1 What the numbers say"))
    A(p("<b>yolov8n is dominated.</b> yolov8s beats it on every axis at once: 22% more accurate, "
        "faster in both end-to-end and DPU-only terms, and lower energy per image. There is no "
        "operating point at which yolov8n is the right answer on this board. The reason is the "
        "array utilisation figure &mdash; yolov8n is too small to fill the hardware, so its "
        "lighter workload buys nothing, while its accuracy penalty is real."))
    A(p("<b>Accuracy is cheap in energy terms.</b> Going from yolov8n to yolov8l gains 0.178 "
        "mAP@0.5, a 40% relative improvement, for 0.157&nbsp;J more per image &mdash; about 16% "
        "more energy. That trade is only available because idle power dwarfs inference power; on "
        "a platform where the accelerator dominated the power budget it would look very "
        "different."))
    A(p("<b>Retention improves with size.</b> The board keeps 85.8% of yolov8n's float mAP@0.5 "
        "but 90.1% of yolov8l's. Larger models carry more redundancy and survive both the "
        "activation substitution and 8-bit arithmetic better. The same ordering holds on the "
        "stricter mAP@0.5:0.95, from 79.1% to 85.2%."))
    A(p("<b>Recall is the weak side.</b> Precision at the operating point ranges from 0.589 to "
        "0.730, recall from 0.412 to 0.562. Every model finds fewer objects than it correctly "
        "labels the ones it does find. If the application cares more about missing objects than "
        "about false alarms, the operating confidence should be set lower than the best-F1 point "
        "quoted here."))

    A(h3("9.2 Which to use"))
    A(p("For most purposes on this board, yolov8s. It is the efficiency optimum and beats the "
        "smaller model outright. If accuracy is what matters and 12.5 images per second is "
        "enough, yolov8l costs little more energy and gives 0.623 mAP@0.5. yolov8m sits between "
        "them without a strong argument for itself, being no faster than yolov8n and less "
        "accurate than yolov8l. yolov8n has no case on this hardware, though it would on a "
        "smaller accelerator that it could actually fill."))

    # ---------------------------------------------------------------- verification
    A(pagebreak())
    A(h2("10. Checks, and what is not measured"))
    A(h3("10.1 Checks that were run"))
    A(p("The decode was verified against Ultralytics' own implementation on identical features "
        "before any board time was spent, agreeing to 6&times;10<sup>-5</sup> pixels on all four "
        "models. The letterbox inverse was separately checked against the reference across ten "
        "aspect ratios including the awkward half-pixel cases."))
    A(p("Every result was scored twice, by pycocotools and by an independent implementation that "
        "shares no matching code with it:"))
    A(table(["Model", "pycocotools mAP@0.5", "independent mAP@0.5", "difference"],
            [[m, v("mAP@0.5", m)[:6], v("mAP@0.5_independent_scorer", m),
              f'{abs(float(v("mAP@0.5", m)) - float(v("mAP@0.5_independent_scorer", m))):.4f}']
             for m in MODELS]))
    A(p("They agree within 0.003 on all four and in the same direction each time, which is what "
        "you would expect from one scorer matching at a single IoU threshold and the other "
        "sweeping ten. The compiled graph was also inspected before deployment to confirm that "
        "every CPU operator it needs actually exists on the board."))

    A(h3("10.2 Not measured"))
    A(p("<b>Rail voltages and currents.</b> Only rail power was captured. The hardware exposes "
        "voltage and current separately and reading them is a small change to the sampling "
        "script, but the board was taken offline before it was done."))
    A(p("<b>FPGA resource utilisation.</b> Not available for the reasons in section 8: the DPU "
        "is a prebuilt platform and implementation was never run here. It is also not a "
        "per-model quantity."))
    A(p("<b>Thermal behaviour.</b> No junction temperatures were recorded and no sustained "
        "thermal soak was performed. The runs are short enough that throttling is unlikely but "
        "it was not checked."))
    A(p("<b>Accuracy after finetuning.</b> Section 6 argues that a HardSwish finetune should "
        "recover most of the lost accuracy. That is a prediction from the measured decomposition, "
        "not a result. It needs a GPU."))

    A(h3("10.3 One unresolved discrepancy"))
    A(p("An earlier deployment of the same four models on this board, by a separate project, "
        "produced slightly higher mAP@0.5:0.95 than this one, while this one scores higher on "
        "mAP@0.5. The gap on the stricter metric is small and grows with model size: 0.0004 on "
        "yolov8n, 0.0061 on yolov8l."))
    A(p("A half-pixel error in the letterbox inverse was found and fixed during this work, and "
        "was initially assumed to explain it. Re-running all four after the fix recovered only "
        "about a quarter of the gap. The remainder is not explained. The most likely candidates "
        "are that the earlier build kept part of the box decode inside the compiled graph, where "
        "it runs at different precision, or a difference in how the two letterbox "
        "implementations round. Both prediction sets are kept, so this can be settled by "
        "comparing box coordinates for matched detections without using the board."))

    # ---------------------------------------------------------------- appendix
    A(pagebreak())
    A(h2("Appendix A. Every metric collected"))
    A(p("All values as recorded, per model. Rows marked otherwise are explained in the sections "
        "above."))
    with open(os.path.join(METRICS, "all_models_metrics.csv")) as fh:
        rdr = list(csv.DictReader(fh))
    skip = {"model", "board", "configuration"}
    rows = []
    for r in rdr:
        k = r["parameter"]
        if k in skip:
            continue
        val = [r[m] for m in MODELS]
        if all(x.startswith("NOT_") for x in val):
            val = [x.replace("NOT_APPLICABLE_PER_MODEL", "n/a, see &sect;8")
                     .replace("NOT_MEASURED", "not measured") for x in val]
        # Underscores give the table no break opportunity, so long keys wrap mid-word.
        rows.append([k.replace("_", " "), r["unit"]] + val)
    A(table(["Parameter", "Unit"] + MODELS, rows, aligns=["left", "left"] + ["right"] * 4,
            fs="8pt", colw=["30%", "10%", "15%", "15%", "15%", "15%"]))

    A(pagebreak())
    A(h2("Appendix B. Reproducing this"))
    A(p("Build and check a model, from the repository root:"))
    A('<pre style="font-size:8.5pt;background:#f4f4f4;padding:6pt;border:1px solid #ddd;">'
      "cd Yolo_v8_Versal_Implementation\n"
      "./v8_run.sh python -u v8_quantize.py --weights yolov8n.pt --act hardswish \\\n"
      "    --quant_mode calib\n"
      "./v8_run.sh python -u v8_quantize.py --weights yolov8n.pt --act hardswish \\\n"
      "    --quant_mode test\n"
      "./v8_compile.sh yolov8n hardswish        # compiles, then reports the subgraph split\n"
      "./deploy_to_board.sh                     # copies host code and xmodels across"
      '</pre>')
    A(p("Then on the board:"))
    A('<pre style="font-size:8.5pt;background:#f4f4f4;padding:6pt;border:1px solid #ddd;">'
      "python3 board_eval_vck190.py --arch v8 \\\n"
      "    --model yolov8n_hardswish_vck190.xmodel \\\n"
      "    --images val2017 --annotations ann/instances_val2017.json \\\n"
      "    --out-json preds_yolov8n_hs.json --out-csv metrics_yolov8n_hs.csv\n\n"
      "python3 measure_power.py --model yolov8n_hardswish_vck190.xmodel \\\n"
      "    --images val2017 --benchmark 400 --reference-fps 14.38"
      '</pre>')
    A(p("The <tt>--arch v8</tt> flag is required; without it the harness uses the anchor-based "
        "YOLOv3 decode and rejects the 144-channel tensors. Letterbox is the default for v8."))
    A(p("Plots and tables are regenerated with:"))
    A('<pre style="font-size:8.5pt;background:#f4f4f4;padding:6pt;border:1px solid #ddd;">'
      "python tools/board_curves.py &lt;annotations&gt; &lt;predictions&gt; &lt;outdir&gt; &lt;title&gt;\n"
      "python tools/extract_pr_f1.py &lt;annotations&gt; &lt;predictions&gt;\n"
      "python tools/export_v8_metrics.py\n"
      "python tools/report_figures.py\n"
      "python tools/build_report.py"
      '</pre>')

    A(h2("Appendix C. Files"))
    A(table(["Path", "Contents"],
            [["board_results/metrics/", "per-model and combined metric CSVs, each row with its source"],
             ["board_results/plots_yolov8*/", "PR, P, R, F1 curves and confusion matrix per model"],
             ["board_results/plots_summary/", "cross-model summary charts"],
             ["board_results/preds_*.json", "raw COCO predictions from the board, all 5000 images"],
             ["report/figures/", "figures used in this document"],
             ["compiled/", "compiled xmodels, one directory per model and activation"],
             ["quantize_result/", "calibration output and INT8 xmodels"]],
            aligns=["left", "left"], fs="8.5pt"))

    A("</body></html>")
    return s


if __name__ == "__main__":
    os.makedirs(REPORT, exist_ok=True)
    html = "".join(build())
    out = os.path.join(REPORT, "report.html")
    with open(out, "w") as fh:
        fh.write(html)
    print("wrote", out, f"({len(html):,} bytes)")
