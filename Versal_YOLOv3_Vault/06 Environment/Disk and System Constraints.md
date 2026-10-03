---
tags: [environment, constraints, disk, warning]
date: 2026-09-11
---

# Disk and System Constraints

> [!danger] Read this before installing, downloading, or copying anything
> This is **someone else's machine** and its disk is **99% full**. There is roughly **9 GB free
> out of 926 GB**. That was true before any of this work started — it is a pre-existing condition,
> not something this project caused.

Related: [[Clone Provenance]], [[Vitis AI Container]], [[Changes Log]].

---

## The numbers

```
/dev/nvme0n1p3  926G  869G  9.3G  99% /
```

For scale, things that will not fit:

| Thing | Size | Fits? |
| --- | --- | --- |
| `requirements.txt` install (torch 2.13 + TF 2.21 + CUDA stack, 190 pkgs) | 10–15 GB | ❌ |
| A second copy of `yolov3_test/` | 8.9 GB | ❌ barely |
| Another Vitis AI Docker image | ~12 GB | ❌ |
| `deploy_check=True` golden tensor dumps | 100s of MB | ⚠️ risky |
| A compiled xmodel | 35 MB | ✅ |
| The whole `vitis_compat/` tree | 2.2 MB | ✅ |
| This vault | < 1 MB | ✅ |

## Where the space went

The original project folder is itself the main consumer:

| Path | Size |
| --- | --- |
| `Yolo_v3_AB_New/yolov3_test/.venv` | **8.2 GB** |
| `Yolo_v3_AB_New/yolov3_test/runs` | 571 MB |
| `Yolo_v3_AB_New/yolov3_test/yolov3_original.pt` | 119 MB |
| `Yolo_v3_AB_New/yolov3_test/yolov3_merged23_e75.pt` | 67 MB |
| `Yolo_v3_AB_New/datasets` | 7.5 MB |

Plus the clone (766 MB at creation, since it skipped the venv), the 11.7 GB Docker image, and
several GB of unrelated archives in `~/Documents` (`SVD_*.zip` alone is ~320 MB across two files).

> [!note] The single biggest recoverable win
> `Yolo_v3_AB_New/yolov3_test/.venv` is **8.2 GB** and is not used by any of this work — the
> pipeline runs entirely in Docker. Deleting it roughly **doubles** the machine's free space,
> and it is fully regenerable from `requirements.txt` — see [[Regenerating the venv]].
>
> It lives in the **original** folder, so it is outside the containment boundary
> ([[Clone Provenance]]) and requires the machine owner's approval before removal.

## Rules followed in this work

1. **Nothing installed system-wide.** No apt, no global pip, no conda.
2. **One clone-local install only** — seaborn, 2.2 MB, `--no-deps`, into `vitis_compat/site/`.
3. **All containers `--rm`.** No container layers accumulate.
4. **The pre-existing Docker image was reused**, never re-pulled.
5. **`deploy_check=False`** on xmodel export, specifically to avoid golden-tensor dumps.
6. **Activations swapped in memory** for [[SiLU to LeakyReLU Experiment]] rather than writing a
   second 264 MB checkpoint.
7. Everything written stays inside the clone — [[Clone Provenance]].

## The one deliberate cleanup outside the clone

An aborted `requirements.txt` install had added entries to the shared pip cache. Purging
`~/.cache/pip` recovered **~5.5 GB**, taking the disk from 100% (4 GB free) to 99% (9.5 GB free).

Only downloaded wheels were removed — no project or user data. The cache regenerates itself on the
next install. This was disclosed at the time; it is the only modification made outside the clone.

## What to watch during future work

```bash
df -h /                    # before and after anything large
du -sh quantize_result*/   # quantizer output grows with deploy_check
docker ps -a               # should be empty; stray containers hold disk
```

Each quantize+compile cycle writes roughly **170 MB** (132 MB int xmodel + 35 MB compiled). Two
full cycles now exist on disk (`quantize_result/` + `compiled/` and `quantize_result_lrelu/` +
`compiled_lrelu/`). Delete a stale pair before running a third if space gets tight:

```bash
rm -rf quantize_result/ compiled/          # keep the lrelu pair as the reference
```

## Before the next phase

A LeakyReLU finetune needs a **GPU machine with room for the full training stack** — it cannot
happen here. Worth raising with the machine's owner:

- Is the 8.2 GB original `.venv` still needed?
- Is there a GPU box available for the finetune?
- Can the several GB of unrelated `~/Documents` archives be moved off?


## Update 2026-09-12: where COCO val2017 now lives

The root filesystem reached **99 % (16 GB free)**, so the dataset needed for float baselines was
staged elsewhere.

| Location | Contents |
| --- | --- |
| `/media/aesicdab/One Touch/coco_val2017/` (external HDD, **629 GB free**, exfat) | `images/val2017` (5000), `annotations/`, `labels/val2017` (36,335 boxes), plus `labels_shift/` and `labels_noperson/` for the 79-class runs |
| VCK190 `/home/root/yolo/` | `val2017`, `ann/`, the xmodels, predictions JSONs — **912 MB free**, so delete a predictions JSON before writing another |

The images were pulled *off the board* rather than downloaded — `tar cf - -C /home/root/yolo val2017 ann`
piped into `tar xf -`, 1 m 48 s over the USB-ethernet link.

The HDD is not visible inside the container: `vitis_run.sh` mounts only the clone. A `val.py` run
against this dataset needs its own `docker run` with `-v "/media/aesicdab/One Touch/coco_val2017:/data"`
added — the pattern is recorded in [[yolov3_original on Hardware]].

---

Back to [[Code Map]] | [[Home]]
