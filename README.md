# Festgen EDM Teacher

An independent inference handoff of the 89,577,673-parameter class-conditional EDM-preconditioned teacher trained for 100,000 updates. This is a teacher intended for sampling and distillation experiments, not a ready-made 1- or 4-NFE student. The model uses the existing Festgen U-Net with custom `log(sigma)/4 * 250` time embedding, not an exact reproduction of the EDM paper's architecture.

Checkpoint: `checkpoints/edm_100k_ema.pt` (Git LFS; SHA-256 `816638eab17241bcabf07eadcbf49f3ee2f0c03b8f76e8d83d811ee15cf437cc`). This is the unchanged evaluated EMA. Load only trusted PyTorch pickle files. The 100K EMA achieved Official FID **22.72855**, seed 42, 3,020 generated/3,020 validation images, BF16, batch 16, classifier-free guidance (CFG) 2.5 and EDM Heun 32 steps, 126 backbone evaluations. This was the best single-seed result in the explored sweep. It is not an independent test: the official validation was also used for experiment decisions. Check applicable benchmark restrictions before submission.

For lower teacher sampling cost, 12 steps/46 NFE scored 22.84814 at seed 42, but that setting has not been confirmed with other seeds. In the fixed CFG 2.5/18-step comparison, the 100K teacher scored 22.9262, 22.8632 and 23.6726 for seeds 42/1042/2042. Results and configurations: `results/sweep.json`, `configs/training.json`.

## Download and run

```bash
git clone https://github.com/TMRsungjea/Fastgen-EDM.git
cd Fastgen-EDM
git lfs pull
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python sample.py --device cuda --labels 0 --steps 32 --cfg 2.5 --output samples.pt
```

PyTorch 2.6.0 is required; install the appropriate PyTorch build for your CUDA device. CPU also works (`--device cpu`, float32) but is slow. `samples.pt` is a dictionary with float images `[B,3,64,64]` in `[-1,1]`, class IDs and the actual NFE. Category IDs 0–150 are in `configs/categories.json`.

```python
import torch
from model.edm import load_model, sample
model = load_model('cuda')
labels = torch.tensor([0], device='cuda', dtype=torch.long)
with torch.autocast('cuda', dtype=torch.bfloat16):
    images, nfe = sample(model, labels, seed=42, steps=32, cfg=2.5)
```

`model.edm.denoise(model, x, sigma, labels)` exposes the denoiser for distillation. Sigma is a positive tensor with one value per image; labels are int64. The null/unconditional label is 151. The sampler performs one unconditional and one conditional forward per guided denoiser evaluation. To reproduce the FID protocol, seed each batch with `42 + batch_index`, use batch 16, 20 images per class in class order, and use the original official reference/evaluation implementation. FID itself is not implemented in this inference-only handoff.

The first 20K updates used 16,324 training images; subsequent updates used all 17,079 official training images. The official 3,020-image validation split was not used for gradients. Neither dataset nor optimizer state is included. Configuration values in the checkpoint may include historical local paths; the loader does not use them. Numerical equivalence has been checked for the included inference adapter; independent full FID reproduction and final submission interface are not verified.
