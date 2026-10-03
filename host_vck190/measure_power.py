"""Measure VCK190 power while a model runs, and derive energy per image.

Runs ON THE BOARD. The VCK190 carries INA226 monitors on its supply rails, exposed through
/sys/class/hwmon. This samples them at idle, then during an inference run, and reports the
difference - idle is subtracted because the board's static draw (fans, DDR, PS) is not
attributable to the model.

    python3 measure_power.py --model yolov8n_hardswish_vck190.xmodel --images val2017 --limit 300

Reports total board power. Rail-level detail is printed too, so the PL/AIE rails that actually
carry the DPU can be separated from the PS rails if the platform names them.
"""

import argparse
import glob
import os
import subprocess
import sys
import threading
import time


def discover():
    """Return [(label, path, scale_to_watts)] for every readable power source."""
    out = []
    for h in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        try:
            name = open(os.path.join(h, "name")).read().strip()
        except OSError:
            name = os.path.basename(h)
        # direct power readings, microwatts
        for p in sorted(glob.glob(os.path.join(h, "power*_input"))):
            lbl = os.path.basename(p).replace("_input", "")
            lf = p.replace("_input", "_label")
            if os.path.exists(lf):
                try:
                    lbl = open(lf).read().strip()
                except OSError:
                    pass
            # The VCK190 exposes 17 INA226 rails that ALL report name="ina226" and
            # "power1", so a name-based key collapses them into one and the per-rail
            # breakdown silently becomes a single rail. Key on the hwmon id, which is unique.
            hid = os.path.basename(h)
            out.append((f"{hid}/{name}/{lbl}", p, 1e-6))
    return out


def read_all(sensors):
    total = 0.0
    per = {}
    for lbl, path, scale in sensors:
        try:
            v = float(open(path).read().strip()) * scale
        except (OSError, ValueError):
            continue
        per[lbl] = v
        total += v
    return total, per


class Sampler(threading.Thread):
    """Poll every rail in the background while something else runs."""

    def __init__(self, sensors, period=0.5):
        super().__init__(daemon=True)
        self.sensors, self.period = sensors, period
        self.samples, self.stop_flag = [], False

    def run(self):
        while not self.stop_flag:
            t, per = read_all(self.sensors)
            self.samples.append((t, per))
            time.sleep(self.period)

    def stop(self):
        self.stop_flag = True
        self.join(timeout=2)

    def mean(self):
        if not self.samples:
            return 0.0, {}
        tot = sum(s[0] for s in self.samples) / len(self.samples)
        keys = self.samples[0][1].keys()
        per = {k: sum(s[1].get(k, 0.0) for s in self.samples) / len(self.samples) for k in keys}
        return tot, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", default="val2017")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--idle-seconds", type=float, default=10.0)
    ap.add_argument("--arch", default="v8")
    ap.add_argument("--benchmark", type=int, default=0,
                    help="drive the DPU back-to-back for N inferences instead of walking the "
                         "dataset. Sustained load is what you want for a power figure: a "
                         "dataset walk leaves the DPU idle between images, so average power "
                         "collapses toward idle and understates the load.")
    ap.add_argument("--reference-fps", type=float, default=0.0,
                    help="throughput measured WITHOUT power sampling. Sampling 17 I2C rails "
                         "perturbs the run, so energy/image derived from the perturbed rate is "
                         "wrong; pass the clean figure and it is used for the energy maths.")
    args = ap.parse_args()

    sensors = discover()
    if not sensors:
        sys.exit("no readable power sensors under /sys/class/hwmon - "
                 "this platform may not expose INA226 rails")
    print(f"[INFO] {len(sensors)} power rails:")
    for lbl, path, _ in sensors:
        print(f"         {lbl:<28} {path}")

    print(f"\n[INFO] sampling idle for {args.idle_seconds:.0f}s (leave the board alone)...")
    idle = Sampler(sensors)
    idle.start()
    time.sleep(args.idle_seconds)
    idle.stop()
    p_idle, per_idle = idle.mean()
    print(f"[INFO] idle total {p_idle:.2f} W")

    if args.benchmark:
        cmd = [sys.executable, "board_eval_vck190.py", "--arch", args.arch,
               "--model", args.model, "--images", args.images,
               "--benchmark", str(args.benchmark)]
        n_img = args.benchmark
        print(f"\n[INFO] driving {args.benchmark} back-to-back inferences (sustained load)...")
    else:
        cmd = [sys.executable, "board_eval_vck190.py", "--arch", args.arch,
               "--model", args.model, "--images", args.images,
               "--limit", str(args.limit), "--no-eval"]
        n_img = args.limit
        print(f"\n[INFO] running {args.limit} images under load...")
    load = Sampler(sensors)
    load.start()
    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elapsed = time.time() - t0
    load.stop()
    if proc.returncode != 0:
        sys.exit(f"inference failed (exit {proc.returncode}) - run board_eval directly to see why")
    p_load, per_load = load.mean()

    fps_observed = n_img / elapsed
    fps = args.reference_fps if args.reference_fps > 0 else fps_observed
    delta = p_load - p_idle
    print("\n" + "-" * 58)
    print(f" model                 : {os.path.basename(args.model)}")
    print(f" images                : {n_img} in {elapsed:.1f}s  ({fps_observed:.2f} img/s observed)")
    if args.reference_fps > 0:
        print(f" fps used for energy   : {fps:.2f} img/s  (clean run, no power sampling)")
    print(f" power, idle           : {p_idle:.2f} W")
    print(f" power, under load     : {p_load:.2f} W")
    print(f" power, attributable   : {delta:.2f} W   (load - idle)")
    print(f" energy per image      : {p_load / fps:.3f} J   (total board)")
    print(f" energy per image      : {delta / fps:.3f} J   (attributable)")
    print(f" efficiency            : {fps / p_load:.2f} img/s/W  (total board)")
    print("-" * 58)
    print("\n per-rail mean (W):")
    for k in sorted(per_load):
        print(f"   {k:<28} idle {per_idle.get(k, 0):7.3f}   load {per_load[k]:7.3f}"
              f"   delta {per_load[k] - per_idle.get(k, 0):+7.3f}")


if __name__ == "__main__":
    main()
