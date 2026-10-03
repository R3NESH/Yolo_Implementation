---
tags: [project, provenance, safety]
date: 2026-09-11
---

# Clone Provenance

Where this working copy came from, and the containment rules that govern it.

Related: [[Disk and System Constraints]], [[Changes Log]], [[Vitis AI Container]].

---

## The two folders

| Path | Role |
| --- | --- |
| `~/Documents/Yolo_v3_AB_New/` | **ORIGINAL** — read-only, never modified |
| `~/Documents/Yolo_v3_AB_New_Clone/` | **WORKING COPY** — all work happens here |

> [!important] The rule
> This is a borrowed machine. Nothing outside the clone gets modified. The original must remain
> exactly as it was found, and it has been verified to be so.

## What the clone contains

```
Yolo_v3_AB_New_Clone/
├── .git/                    git repo, remote -> github.com/R3NESH/YOLOV8.git, NO COMMITS YET
├── .gitattributes           *.pt filter=lfs
├── .gitignore               .venv/, __pycache__/, build/, .vscode/ ...
├── datasets/coco128/        7.5 MB, 128-image COCO subset
├── yolov3_test/             the project (8.9 GB in the original; less here, see below)
├── tools/                   graphify_codebase.py            (added this session)
└── Versal_YOLOv3_Vault/     this vault                      (added this session)
```

The clone was created before this session — the file trees were verified identical to the original
apart from `.git/`, `.gitattributes` and `.gitignore`. File counts matched exactly (103 files in
`runs/`, for instance) and the `.pt` files are genuine independent copies, not hardlinks
(different inodes).

## The symlink that had to go

> [!danger] The clone was not actually isolated when found
> `yolov3_test/.venv` in the clone was a **symlink pointing back into the original**:
> `~/Documents/Yolo_v3_AB_New/yolov3_test/.venv` (8.2 GB).

This explained a puzzling size discrepancy — the original folder measured 8.9 GB while the clone
measured 766 MB, despite `diff -rq` reporting identical trees. The venv had never been copied, only
linked.

It was a live hazard: any `pip install` run from the clone, or even Python writing a stray `.pyc`
into the venv, would have **modified the original** through the link. Exactly the thing the clone
existed to prevent.

**Resolution:** the symlink was removed. A real isolated venv was created in its place, then
removed too once the decision was made to run everything in Docker instead
([[Vitis AI Container]]). **There is no `.venv` in the clone now, by design.**

## Git state

```
On branch main — No commits yet
remote: origin  https://github.com/R3NESH/YOLOV8.git
```

The repo is initialised with a remote configured but **nothing committed**. Everything is
untracked. `.gitattributes` sets up LFS for `*.pt`, which matters: the checkpoints are 67–277 MB
and would need LFS if they were ever pushed.

> [!warning] Before any `git add`
> `.gitignore` covers `.venv/` and `__pycache__/`, but **not** the large artefacts this session
> produced: `quantize_result*/`, `compiled*/`, `vitis_out/*.log`, and `vitis_compat/site/`.
> A naive `git add -A` would stage ~340 MB of build output and a vendored pip install.
> Add them to `.gitignore` first, or commit selectively.

## Verification

The original's integrity was confirmed by comparison at the start of the session and it has not
been written to since. The only interactions with it were:

1. Reading, to compare trees.
2. Severing the clone's symlink **into** it (a change to the clone, not the original).

Its 8.2 GB `.venv` remains intact and untouched.

## Containment in practice

The Docker runner enforces the rule mechanically rather than relying on discipline —
`vitis_run.sh` mounts **only** the clone:

```bash
-v "${CLONE_ROOT}:/workspace"
```

The container has no visibility of the original folder, the home directory, or anything else on
the machine. Even a buggy script inside it cannot reach out.

---

Back to [[Code Map]] | [[Home]]
