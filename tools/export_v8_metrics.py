"""Emit a per-model metrics CSV for the YOLOv8 VCK190 deployment.

One CSV per model plus a combined sheet. Every row carries its own `source` column, because the
thing that made the prior YOLOv8 numbers unusable was not that they were wrong - it was that
nobody could tell where any of them came from ('YOLOv8 Prior Numbers Reconciled').

Rows that were never measured are emitted with value `NOT_MEASURED` and a reason, rather than
omitted or estimated. Two such rows matter here:

  power   the VCK190 carries INA226 rails readable under /sys/class/hwmon, so this IS
          measurable - but only with the board powered and running. Not captured during the
          runs, and the board has since been disconnected.

  area    FPGA resource utilisation (LUT/FF/BRAM/DSP/URAM) is a property of the DPU *bitstream*,
          not of a model. All four models share one DPUCVDX8G_ISA3_C32B6 build - the identical
          `dpu_fingerprint` in every compiled xmodel proves it - so area does not vary per model
          and a per-model column for it would be meaningless. The figure itself comes from the
          Vivado implementation report for the platform, which this project does not have: the
          board runs a prebuilt Vitis AI image.

    Versal_AI/venv/bin/python tools/export_v8_metrics.py [out_dir]
"""

import csv
import os
import sys

MODELS = ["yolov8n", "yolov8s", "yolov8m", "yolov8l"]

# --- measured: accuracy, pycocotools over COCO val2017, 5000 images -----------------------
COCO = {                       # mAP@0.5, mAP@0.5:0.95, mAP@0.75, AR@100, detections
    "yolov8n": (0.445271, 0.291348, 0.307229, 0.494916, 794255),
    "yolov8s": (0.542305, 0.371572, 0.393132, 0.560285, 637641),
    "yolov8m": (0.600114, 0.425569, 0.456461, 0.605190, 773508),
    "yolov8l": (0.623348, 0.446702, 0.480741, 0.618074, 957300),
}
# --- measured: accuracy, tools/board_curves.py (independent matcher) ----------------------
CURVES = {                     # mAP@0.5, best F1, confidence at best F1
    "yolov8n": (0.4430, 0.4682, 0.269),
    "yolov8s": (0.5391, 0.5482, 0.377),
    "yolov8m": (0.5976, 0.5981, 0.320),
    "yolov8l": (0.6211, 0.6128, 0.377),
}
# --- measured: precision / recall at the best-F1 operating point (tools/extract_pr_f1.py) --
PR = {                         # precision, recall
    "yolov8n": (0.5889, 0.4123),
    "yolov8s": (0.6523, 0.4906),
    "yolov8m": (0.6621, 0.5616),
    "yolov8l": (0.7295, 0.5559),
}
# --- measured: wall-clock throughput, end to end -----------------------------------------
THROUGHPUT = {                 # images/s, seconds for 5000 images
    "yolov8n": (14.38, 347),
    "yolov8s": (15.86, 315),
    "yolov8m": (14.34, 348),
    "yolov8l": (12.45, 401),
}
# --- read out of the compiled xmodel (xir subgraph attributes) ----------------------------
XMODEL = {   # workload, workload_on_arch, mc_code bytes, REG_0 weights, REG_1 activations,
             # REG_2 input, REG_3 output, xmodel bytes, ops
    "yolov8n": (8764000000, 10031488000, 148976, 3153280, 11635200, 1230784, 1209600, 3550373, 244),
    "yolov8s": (28640902400, 29984875520, 191372, 11157504, 22476800, 1230784, 1209600, 11597642, 244),
    "yolov8m": (79005369600, 83248071680, 403080, 25888320, 34560000, 1230784, 1209600, 26589968, 310),
    "yolov8l": (165254790400, 168269890560, 794004, 43669760, 52275200, 1230784, 1209600, 44766296, 376),
}
# --- from the checkpoints --------------------------------------------------------------
MODEL = {"yolov8n": (3.16, 57), "yolov8s": (11.17, 57),
         "yolov8m": (25.90, 77), "yolov8l": (43.69, 97)}
# --- float reference --------------------------------------------------------------------
FLOAT = {"yolov8n": (0.5187, 0.3681), "yolov8s": (0.6106, 0.4440),
         "yolov8m": (0.6654, 0.4979), "yolov8l": (0.6916, 0.5244)}

# --- measured: host_vck190/measure_power.py, 17 INA226 rails, sustained load -------------
# idle W, load W, attributable W, energy J/img (total board), efficiency img/s/W
POWER = {
    "yolov8n": (13.02, 13.70, 0.69, 0.953, 1.05),
    "yolov8s": (13.27, 14.00, 0.73, 0.883, 1.13),
    "yolov8m": (13.30, 14.01, 0.71, 0.977, 1.02),
    "yolov8l": (12.81, 13.81, 1.01, 1.110, 0.90),
}
# DPU-only throughput, back-to-back inferences, no host decode (from the same runs)
DPU_FPS = {"yolov8n": 22.16, "yolov8s": 22.24, "yolov8m": 20.69, "yolov8l": 15.04}

MB = 1024.0 * 1024.0
BOARD = "VCK190, DPUCVDX8G_ISA3_C32B6, batch 6, 333 MHz, VART 3.0.0"
CFG = "INT8, HardSwish, letterbox, 640x640, conf 0.001, IoU 0.45, max_det 300"


def rows_for(m):
    map50, map5095, map75, ar100, dets = COCO[m]
    c50, f1, f1conf = CURVES[m]
    prec, rec = PR[m]
    fps, secs = THROUGHPUT[m]
    wl, wla, mc, r0, r1, r2, r3, xm, ops = XMODEL[m]
    params, convs = MODEL[m]
    f50, f5095 = FLOAT[m]
    p_idle, p_load, p_attr, energy, eff = POWER[m]
    dpu_fps = DPU_FPS[m]
    ddr = r0 + r1 + r2 + r3

    S_COCO = "measured: board_eval_vck190.py + pycocotools, val2017 5000 images"
    S_CURV = "measured: tools/board_curves.py (own IoU matcher, no pycocotools)"
    S_PR = "measured: tools/extract_pr_f1.py at the best-F1 confidence, IoU 0.5"
    S_TIME = "measured: wall clock, end-to-end incl. preprocess+DPU+decode+NMS"
    S_XIR = "read from compiled xmodel: xir DPU subgraph attributes"
    S_CKPT = "read from the .pt checkpoint"
    S_REF = "Versal_AI/benchmarks/pytorch_results.csv (PC float baseline)"
    S_DERIV = "derived from the rows above"
    S_PWR = ("measured: host_vck190/measure_power.py, 17 INA226 rails via /sys/class/hwmon, "
             "sustained load (400 back-to-back inferences), 0.5 s sampling")
    S_ENER = ("derived: power_under_load / fps, using the CLEAN fps from the scored run - "
              "sampling 17 I2C rails perturbs throughput, so the rate observed during "
              "sampling is not the rate to divide by")
    S_AREA = ("Model footprint, not FPGA area. FPGA utilisation (LUT/FF/BRAM/URAM/DSP) belongs "
              "to the DPU bitstream, not the model: all four share one DPUCVDX8G_ISA3_C32B6 "
              "build (identical dpu_fingerprint), so it does not vary per model, and the "
              "figures live in a Vivado report this prebuilt platform does not ship.")

    return [
        # --- identity -------------------------------------------------------------------
        ("model", m, "", S_CKPT),
        ("board", BOARD, "", "fixed"),
        ("configuration", CFG, "", "fixed"),

        # --- performance ----------------------------------------------------------------
        ("fps", f"{fps:.2f}", "images/s", S_TIME),
        ("fps_dpu_only", f"{dpu_fps:.2f}", "images/s",
         "measured: board_eval --benchmark, back-to-back inferences, excludes host decode+NMS"),
        ("latency_per_image", f"{1000.0/fps:.2f}", "ms", S_DERIV),
        ("runtime_5000_images", secs, "s", S_TIME),

        # --- power and energy -----------------------------------------------------------
        ("power_idle", f"{p_idle:.2f}", "W", S_PWR),
        ("power_under_load", f"{p_load:.2f}", "W", S_PWR),
        ("power_attributable", f"{p_attr:.2f}", "W", "load minus idle; " + S_PWR),
        ("energy_per_image", f"{energy:.3f}", "J/image", S_ENER),
        ("energy_per_image_attributable", f"{p_attr/fps:.3f}", "J/image",
         "power_attributable / fps; excludes the board's static draw"),
        ("efficiency", f"{eff:.2f}", "images/s/W", "fps / power_under_load"),

        # --- accuracy -------------------------------------------------------------------
        ("mAP@0.5", f"{map50:.6f}", "", S_COCO),
        ("mAP@0.5:0.95", f"{map5095:.6f}", "", S_COCO),
        ("mAP@0.75", f"{map75:.6f}", "", S_COCO),
        ("AR@100", f"{ar100:.6f}", "", S_COCO),
        ("mAP@0.5_independent_scorer", f"{c50:.4f}", "", S_CURV),
        ("float_mAP@0.5", f"{f50:.4f}", "", S_REF),
        ("float_mAP@0.5:0.95", f"{f5095:.4f}", "", S_REF),
        ("accuracy_retained@0.5", f"{100*map50/f50:.1f}", "%", S_DERIV),
        ("accuracy_retained@0.5:0.95", f"{100*map5095/f5095:.1f}", "%", S_DERIV),

        # --- precision / recall / F1 at the deployed operating point ----------------------
        ("precision", f"{prec:.4f}", "", S_PR),
        ("recall", f"{rec:.4f}", "", S_PR),
        ("f1_score", f"{f1:.4f}", "", S_PR),
        ("operating_confidence", f"{f1conf:.3f}", "", S_PR),
        ("detections_total", dets, "count", S_COCO),

        # --- model size ------------------------------------------------------------------
        ("parameters", f"{params:.2f}", "M", S_CKPT),
        ("xmodel_size", f"{xm/MB:.2f}", "MiB", S_XIR),
        ("ddr_weights", f"{r0/MB:.2f}", "MiB", S_XIR),
        ("ddr_activations", f"{r1/MB:.2f}", "MiB", S_XIR),
        ("ddr_input_buffer", f"{r2/MB:.2f}", "MiB", S_XIR),
        ("ddr_output_buffer", f"{r3/MB:.2f}", "MiB", S_XIR),
        ("ddr_total", f"{ddr/MB:.2f}", "MiB", S_DERIV),
        ("microcode_size", f"{mc/1024.0:.1f}", "KiB", S_XIR),

        # --- compute ---------------------------------------------------------------------
        ("workload", f"{wl/1e9:.3f}", "GOP/image", S_XIR),
        ("workload_on_arch", f"{wla/1e9:.3f}", "GOP/image", S_XIR),
        ("dpu_array_efficiency", f"{100.0*wl/wla:.1f}", "%", S_DERIV),
        ("achieved_compute", f"{wl*fps/1e9:.1f}", "GOP/s", S_DERIV),
        ("conv_modules", convs, "count", S_CKPT),
        ("dpu_ops", ops, "count", S_XIR),
        ("dpu_subgraphs", 1, "count", S_XIR),
        ("cpu_ops", "fix2float x3", "", S_XIR),

        # --- FPGA area: not a per-model quantity -----------------------------------------
        ("fpga_area_LUT", "NOT_APPLICABLE_PER_MODEL", "count", S_AREA),
        ("fpga_area_FF", "NOT_APPLICABLE_PER_MODEL", "count", S_AREA),
        ("fpga_area_BRAM", "NOT_APPLICABLE_PER_MODEL", "count", S_AREA),
        ("fpga_area_URAM", "NOT_APPLICABLE_PER_MODEL", "count", S_AREA),
        ("fpga_area_DSP", "NOT_APPLICABLE_PER_MODEL", "count", S_AREA),
        ("dpu_fingerprint", 433190037845252193, "",
         "identical across all four models - one DPU build serves them all"),
    ]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else \
        "Yolo_v8_Versal_Implementation/board_results/metrics"
    os.makedirs(out, exist_ok=True)

    for m in MODELS:
        p = os.path.join(out, f"{m}_metrics.csv")
        with open(p, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["parameter", "value", "unit", "source"])
            w.writerows(rows_for(m))
        print("  wrote", p)

    # combined: one column per model, same row order
    keys = [r[0] for r in rows_for(MODELS[0])]
    table = {m: {r[0]: (r[1], r[2], r[3]) for r in rows_for(m)} for m in MODELS}
    p = os.path.join(out, "all_models_metrics.csv")
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["parameter", "unit"] + MODELS + ["source"])
        for k in keys:
            unit = table[MODELS[0]][k][1]
            src = table[MODELS[0]][k][2]
            w.writerow([k, unit] + [table[m][k][0] for m in MODELS] + [src])
    print("  wrote", p)


if __name__ == "__main__":
    main()
