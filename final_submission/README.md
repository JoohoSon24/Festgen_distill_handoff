# Final submission (in progress): EDM teacher → Progressive Distillation + DSM

Status 2026-11-04: stages 32 and 16 fixed; stage 8 is still being explored (`../distill/PROGRESS.md`).
The 1-NFE and 4-NFE models are not ready yet. The submission `model.py` / `ModelOneNFE` /
`ModelFewNFE` wrapper is not written yet.

## Training pipeline (no FID inside)

```bash
cd final_submission
python -m training.train_pd_dsm --data-root /path/to/pokemon-generation-one-22k   # resumable
```

- Code: `training/train_pd_dsm.py`. Config: `configs/pd_dsm_fixed.json`.
- Stages: 32 → 16 → 8 → 4 → 2 → 1 Euler steps on nested Karras grids (64-step master grid, σ 80 → 0.002, ρ = 7).
- **Each stage trains for a fixed number of updates** (`stage_updates`). Its final EMA becomes the next stage's init and teacher.
- The pipeline computes no FID and reads no validation images. It reads only the 16,324-image development-train manifest (`data/development/seed42_holdout5/train.txt`).
- Loss per update = PD + 0.1·DSM, both EDM-weighted:
  - PD: one student Euler step must match two teacher steps.
  - DSM: EDM denoising loss on real images, σ ~ logN(−1.2, 1.2²).
- Stage 32 teacher: frozen EDM teacher, guided Heun, CFG 2.5 baked in. Inference uses no CFG, so 1 backbone call per step.
- Training settings: batch 64 (micro 32), Adam 1e-4 constant, EMA 0.999, grad clip 1, bf16, activation checkpointing.
- Limits:
  - Params: 89,577,673, under the 100M limit.
  - Peak VRAM: about 7 GB, under the 20 GB limit. The trainer asserts this.

## How the update counts were fixed

The counts come from an exploratory run. That run's code and logs are in `records/exploratory_run/`.

**Exploratory run decision rule:**
1. Train-loss plateau: PD and DSM window means each improve by < 3% for 3 consecutive 500-update windows. The first two stages used < 1%; it was relaxed for speed.
2. **and** the 755-image dev-holdout ("Valid") FID rises twice in a row.

At stage end, the dev-holdout-best EMA was handed on. The number of updates up to that EMA became `stage_updates`.

**Use of Official FID:** the exploratory run computed Official FID (validation split) for the log only. It was never used for any decision. This pipeline uses neither Official FID nor Valid FID.

| stage | fixed updates | dev-holdout FID at that update | exploratory run stopped at | stop reason |
|---:|---:|---:|---:|---|
| 32 | 1,900 | 42.28 | 16,500 | loss plateau + Valid FID rose twice |
| 16 | 15,500 | 42.59 | 21,001 | 10K cap (rule relaxed mid-run) |
| 8, 4, 2, 1 | TBD | | | |

`checkpoints/n32_ema.pt` and `checkpoints/n16_ema.pt` are the exploratory-run EMAs at those updates. Rerunning this pipeline gives the same recipe, but not bit-identical weights, for two reasons:
- The exploratory stage 32 was interrupted once and resumed from update 1,000, and the restart reshuffled the data order.
- GPU nondeterminism.

## Known issues to resolve before submitting

- **The teacher (`checkpoints/edm_100k_ema.pt`, sha `816638eab17241bc…`) came from a third party as an inference-only handoff.** Its training code is not in this folder, so the "trained from scratch" chain is not yet reproducible here.
- The teacher README says that **the official validation split was used for its experiment decisions** (sampling sweep, CFG 2.5). It also says that after 20K updates the teacher trained on all 17,079 training images, including the dev holdout, so dev-holdout FID is optimistically biased.

## Layout

- `model/`: EDM U-Net and `denoise` from Fastgen-EDM.
- `dataset.py`, `download_dataset.py`: TA files.
- `training/data.py`: train-split adapter.
- `data/development/seed42_holdout5/`: train/holdout manifests.
- `checkpoints/`: teacher (Git LFS). The stage EMAs fixed so far (`n32_ema.pt`, `n16_ema.pt`, 358 MB each) are kept locally only and are not in git.
- `records/exploratory_run/`: the exploratory trainer `pd_dsm.py`, its launcher, `PROGRESS.md`, `progress.json`, `metrics.jsonl` and `config.json`. This is a snapshot taken 2026-11-04.
