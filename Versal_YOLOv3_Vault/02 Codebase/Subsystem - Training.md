---
tags: [codebase, training, retrain, leakyrelu]
date: 2026-09-11
---

# Subsystem - Training

`train.py` — and, because of [[DPU Subgraph Fragmentation]], the file that now sits on the
project's critical path.

Modules: [[train]], [[utils.loss]], [[utils.dataloaders]], [[utils.autoanchor]], [[models.yolo]].
Related: [[Training Run exp2]], [[SiLU to LeakyReLU Experiment]], [[Subsystem - Utils Core]].

---

> [!warning] 59% of this file is dead code
> `train.py` is **1476 lines**, but there is **no executable statement before line ~869** — the
> entire first block is a commented-out earlier copy of the same script. Live definitions:
>
> | Symbol | Line |
> | --- | --- |
> | `train()` | 976 |
> | `parse_opt()` | 1362 |
> | `main()` | 1411 |
> | `run()` | 1466 |
> | `if __name__ == "__main__"` | 1474 |
>
> Editors will happily show you the dead half. Grep results and stack traces refer to the live
> half. Before editing anything here, confirm the line number is above 869. The same pattern
> affects `export_dpu_wrapper_AB3.py` and `quantize_vitis_AB4.py`, which each carry two superseded
> drafts in comments — this is a house style in the project, and a persistent footgun.

## `train()` flow

1. **Setup** — resolve `save_dir`, seed, device; load `hyp.yaml`; save `opt.yaml` + `hyp.yaml` into
   the run directory (this is why a crashed run still leaves those two files behind — see the
   `exp` post-mortem in [[Training Run exp2]]).
2. **Model** — load a checkpoint or build from `--cfg` via `parse_model`; apply `--freeze`.
3. **Optimizer** — parameter groups (weights / biases / BN), `--optimizer` SGD or Adam,
   `lr0` → `lrf` schedule (linear, or cosine with `--cos-lr`), warmup over `warmup_epochs`.
4. **EMA** — `ModelEMA` maintains a shadow model; **the EMA weights are what gets saved**.
5. **Dataloaders** — `create_dataloader` with mosaic/HSV/flip augmentation; autoanchor check
   unless `--noautoanchor`.
6. **Epoch loop** — AMP autocast forward, `ComputeLoss`, backward, optimizer step per
   accumulation window; end of epoch runs `val.run()` for mAP.
7. **Checkpointing** — appends a row to `results.csv`, then saves `last.pt` always and `best.pt`
   when `fitness` improves. Both happen **at end of epoch**, never mid-epoch.

> [!info] Why `best.pt` is 264 MB
> The checkpoint pickles the **whole model object** plus the EMA copy, the optimizer state, and
> `opt`/training metadata — not a bare `state_dict`. That is also why loading it re-imports
> `models.common` and drags in the whole plotting import chain ([[Compat Shims]]).

## The retrain that matters: SiLU → LeakyReLU

[[SiLU to LeakyReLU Experiment]] proved the fix collapses 52 DPU subgraphs to 1. Doing it
*properly* means finetuning so the weights adapt to the new activation.

### The one-line route (preferred)

`parse_model` supports an `activation:` key in the model yaml —
[models/yolo.py:348-351](../../yolov3_test/models/yolo.py):

```python
anchors, nc, gd, gw, act = d["anchors"], d["nc"], d["depth_multiple"], d["width_multiple"], d.get("activation")
if act:
    Conv.default_act = eval(act)  # redefine default activation
    LOGGER.info(f"{colorstr('activation:')} {act}")
```

So a copy of the model yaml with one added line changes every `Conv`'s activation globally:

```yaml
# models/yolov3_merged_23_lrelu.yaml
activation: nn.LeakyReLU(0.1015625)    # DPU-native slope - NOT 0.1
```

> [!danger] Use 0.1015625, not 0.1
> `0.1015625` (= 26/256) is the slope verified to compile to a single DPU subgraph. Darknet's
> historical `0.1` was **not** tested here and older DPU generations accept only 26/256. See §11 of
> [[Vitis AI DPU Concepts]].

Two caveats on this route:

- `Conv.default_act` only applies to layers built **from yaml**. Finetuning from `best.pt` loads a
  pickled model whose activation modules already exist, so the yaml key alone will **not** convert
  it *(this is the important subtlety)*.
- Therefore: either train from the yaml with `--weights ''` (from scratch, expensive), or load
  `best.pt` and swap the modules programmatically before training.

### The surgical route (recommended for a finetune)

Reuse the proven swap from `dpu_silu_experiment.py` — it is already written and verified to
produce a single-subgraph graph:

```python
def swap_silu(module, alpha=0.1015625):
    swapped = 0
    for name, child in module.named_children():
        if isinstance(child, nn.SiLU):
            setattr(module, name, nn.LeakyReLU(negative_slope=alpha, inplace=True))
            swapped += 1
        else:
            swapped += swap_silu(child, alpha)
    return swapped
```

Call it on the model right after it is built/loaded in `train()`, before the optimizer is
constructed. Expect it to report **49** swaps on this model.

### Suggested finetune command

```bash
python train.py \
  --weights runs/train/exp2/weights/best.pt \
  --data data/train2017_yolo.yaml \
  --imgsz 416 --epochs 40 --batch-size 8 \
  --hyp data/hyps/hyp.finetune.yaml \
  --name lrelu_finetune
```

Reasoning: `--epochs 40` because [[Training Run exp2]] showed gains flattening by epoch 34;
a finetune hyp (lower `lr0`) because the previous run's `lr0: 0.01` caused a visible 5-epoch
accuracy dip before recovering — a lower LR should avoid re-paying that cost while the network
adapts to the new activation. Verify the finetune hyp file exists in `data/hyps/` first
([[Subsystem - Data and Datasets]]).

Then re-run the pipeline and confirm `DPU subgraph number 1`
([[Implementation Plan]]), and compare mAP against the 0.5275 / 0.3228 baseline.

## `parse_opt()` flags

Live at line 1362. The ones that matter here:

| Flag | Note |
| --- | --- |
| `--weights` | finetune seed; `''` means from scratch |
| `--cfg` | model yaml; **empty in exp2** — architecture came from the checkpoint |
| `--data` | dataset yaml |
| `--hyp` | hyperparameter yaml |
| `--imgsz` / `--img` / `--img-size` | 416 throughout this project |
| `--epochs`, `--batch-size` | 75 / 8 in exp2 (only 35 epochs ran) |
| `--resume` | resume from `last.pt` — relevant, since exp2 was truncated |
| `--nosave` | suppresses checkpointing entirely |
| `--noval`, `--noplots`, `--noautoanchor` | skip eval / plots / anchor check |
| `--patience` | early-stop patience; 100 in exp2, i.e. effectively disabled |
| `--freeze` | freeze first N layers — a plausible finetune accelerator |
| `--device`, `--workers`, `--sync-bn` | hardware placement |
| `--project`, `--name`, `--exist-ok` | output dir; `name: exp` auto-increments to `exp2`, `exp3`… |

Full list: `--weights --cfg --data --hyp --epochs --batch-size --imgsz --rect --resume --nosave
--noval --noautoanchor --noplots --evolve --bucket --cache --image-weights --device --multi-scale
--single-cls --optimizer --sync-bn --workers --project --name --exist-ok --quad --cos-lr
--label-smoothing --patience --freeze --save-period --seed --entity`.

## Why `runs/train/exp/weights` is empty

Because `opt.yaml`/`hyp.yaml` are written at **startup** while `results.csv` and checkpoints are
written at **end of epoch** — a run that dies inside epoch 0 leaves exactly the residue seen in
`exp/`. Full evidence in [[Training Run exp2]].

## Not runnable here

`train.py` needs the full torch/CUDA stack, which this machine cannot host
([[Disk and System Constraints]]), and the Vitis AI container is CPU-only. **The finetune must
happen on a GPU machine elsewhere** — it is the one step in the plan that cannot be done on this
system.

---

Back to [[Code Map]] | [[Home]]
