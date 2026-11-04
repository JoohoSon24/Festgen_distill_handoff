"""Submission training pipeline: progressive distillation (32 -> 16 -> 8 -> 4 -> 2 -> 1 Euler steps)
of the EDM teacher with an auxiliary EDM denoising (score-matching) loss on real train images.

Each stage trains for a FIXED number of updates (configs/pd_dsm_fixed.json: stage_updates) and hands
its final EMA to the next stage as init + teacher. No FID is computed here and no validation data is read;
the update counts were fixed beforehand from an exploratory run (see records/).

  * Stage 32: target = 2 guided (CFG 2.5) Heun steps of the frozen teacher on the 64-step master grid
    (CFG is baked into the student; inference uses one backbone call per step, no CFG).
  * Later stages: target = 2 Euler steps of the previous stage's EMA.
  Loss = PD + dsm_weight * DSM, both EDM-weighted lambda(sigma) = (sigma^2 + sd^2) / (sigma * sd)^2.

Usage (from final_submission/):
  python -m training.train_pd_dsm --data-root /path/to/pokemon-generation-one-22k [--out runs/fixed]
Rerunning the same command resumes from the last saved stage/update.
"""
import argparse, copy, hashlib, json, os, random, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.unet import ClassConditionalUNet
from model.edm import denoise
from training.data import TrainImages

ARCH = dict(image_size=64, in_channels=3, num_classes=151, model_channels=130, channel_mult=(1, 2, 3, 4),
            num_res_blocks=2, attention_resolutions=(16,), dropout=0.0)
PARAMS = 89_577_673
NULL = 151


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def build(state_dict, train):
    net = ClassConditionalUNet(**ARCH, activation_checkpoint_resolutions=(64, 32, 16, 8) if train else ())
    net.load_state_dict(state_dict, strict=True)
    assert sum(p.numel() for p in net.parameters()) == PARAMS <= 100_000_000
    return net.cuda().train(train).requires_grad_(train)


def grid(c, n):
    """Nested Karras grids: the n-step grid is every (M/n)-th level of the M-step master grid, plus sigma=0."""
    M = c['master_steps']; i = torch.arange(M, dtype=torch.float64); r = c['rho']
    lv = (c['sigma_max'] ** (1 / r) + i / (M - 1) * (c['sigma_min'] ** (1 / r) - c['sigma_max'] ** (1 / r))) ** r
    assert M % n == 0
    return torch.cat([lv, torch.zeros(1, dtype=torch.float64)])[::M // n].float().cuda()


def bc(v):
    return v.reshape(-1, 1, 1, 1)


def euler(x, d, s, sn):
    return x + bc(sn - s) * (x - d) / bc(s)


def guided(net, x, s, y, cfg):
    u, c = denoise(net, torch.cat([x, x]), torch.cat([s, s]), torch.cat([torch.full_like(y, NULL), y])).chunk(2)
    return u + cfg * (c - u)


def edm_weight(s, sd):
    return (s.square() + sd ** 2) / (s * sd).square()


def batches(loader):
    while True:
        yield from loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(ROOT / 'configs/pd_dsm_fixed.json'))
    ap.add_argument('--data-root', required=True, help='folder containing PokemonData/ and the official split files')
    ap.add_argument('--out', default=str(ROOT / 'runs/fixed'))
    ap.add_argument('--stages', help='comma list to run a prefix, e.g. 32,16')
    a = ap.parse_args()
    c = json.loads(Path(a.config).read_text())
    stages = [int(s) for s in a.stages.split(',')] if a.stages else c['stages']
    todo = [n for n in stages if c['stage_updates'][str(n)] is None]
    assert not todo, f'stage_updates not fixed yet for {todo}'
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    teacher_path = (ROOT / c['teacher']).resolve() if not Path(c['teacher']).is_absolute() else Path(c['teacher'])
    assert sha(teacher_path) == c['teacher_sha256'], 'unexpected teacher checkpoint'

    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    random.seed(c['seed']); np.random.seed(c['seed']); torch.manual_seed(c['seed'])
    data = TrainImages(a.data_root, a.data_root, 64, 151, split_file=ROOT / c['train_manifest'], cache=True)
    assert len(data) == 16324
    loader = DataLoader(data, batch_size=c['micro'], shuffle=True, drop_last=True, num_workers=0,
                        generator=torch.Generator().manual_seed(c['seed']))
    it = batches(loader); sd = c['sigma_data']; accum = c['batch'] // c['micro']
    log = open(out / 'train_log.jsonl', 'a'); t0 = time.time()

    prev = torch.load(teacher_path, map_location='cpu', weights_only=False)['state_dict']
    for si, n in enumerate(stages):
        sdir = out / f'n{n}'; sdir.mkdir(exist_ok=True); final = sdir / 'final_ema.pt'
        if final.exists():
            prev = torch.load(final, map_location='cpu', weights_only=False)['state_dict']; continue
        total_updates = c['stage_updates'][str(n)]; first = si == 0
        teacher = build(prev, train=False); student = build(prev, train=True)
        ema = copy.deepcopy(student).eval().requires_grad_(False)
        opt = torch.optim.Adam(student.parameters(), lr=c['lr']); update = 0
        if (sdir / 'latest.pt').exists():
            s = torch.load(sdir / 'latest.pt', map_location='cpu', weights_only=False)
            student.load_state_dict(s['student']); ema.load_state_dict(s['ema']); opt.load_state_dict(s['opt'])
            update = s['update']; torch.set_rng_state(s['rng']); torch.cuda.set_rng_state(s['cuda_rng'])
        g_s, g_t = grid(c, n), grid(c, 2 * n)
        torch.cuda.reset_peak_memory_stats()
        while update < total_updates:
            update += 1; opt.zero_grad(set_to_none=True); pd_t = dsm_t = 0.0
            for _ in range(accum):
                x0, y = next(it); x0, y = x0.cuda(non_blocking=True), y.cuda(non_blocking=True); B = len(y)
                j = torch.randint(0, n, (B,), device='cuda')
                sa, sb, sc = g_s[j], g_t[2 * j + 1], g_s[j + 1]
                x = x0 + bc(sa) * torch.randn_like(x0)
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    if first:   # guided Heun steps of the EDM teacher
                        def step(z, s, s2):
                            d = (z - guided(teacher, z, s, y, c['cfg'])) / bc(s); zn = z + bc(s2 - s) * d
                            s2s = s2.clamp_min(1e-8)
                            d2 = (zn - guided(teacher, zn, s2s, y, c['cfg'])) / bc(s2s)
                            return torch.where(bc(s2) > 0, z + bc(s2 - s) * (d + d2) / 2, zn)
                    else:       # Euler steps of the previous student
                        def step(z, s, s2):
                            return euler(z, denoise(teacher, z, s, y), s, s2)
                    xc = step(step(x, sa, sb), sb, sc)
                    target = x - bc(sa) * (xc - x) / bc(sc - sa)   # one Euler step from sa lands on xc
                sig = torch.exp(torch.randn(B, device='cuda') * c['p_std'] + c['p_mean'])
                xn = x0 + bc(sig) * torch.randn_like(x0)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    d = denoise(student, torch.cat([x, xn]), torch.cat([sa, sig]), torch.cat([y, y]))
                dp, dd = d.chunk(2)
                pd = (bc(edm_weight(sa, sd)) * (dp - target).square()).mean()
                dsm = (bc(edm_weight(sig, sd)) * (dd - x0).square()).mean()
                ((pd + c['dsm_weight'] * dsm) / accum).backward()
                pd_t += float(pd) / accum; dsm_t += float(dsm) / accum
            gn = torch.nn.utils.clip_grad_norm_(student.parameters(), c['clip'], error_if_nonfinite=True); opt.step()
            with torch.no_grad():
                for e, p in zip(ema.parameters(), student.parameters()):
                    e.lerp_(p, 1 - c['ema'])
            peak = torch.cuda.max_memory_allocated(); assert peak < c['vram_limit'], peak
            if update % c['log_every'] == 0:
                log.write(json.dumps({'stage': n, 'update': update, 'pd': pd_t, 'dsm': dsm_t, 'grad_norm': float(gn),
                                      'peak_vram': peak, 'elapsed_s': time.time() - t0}) + '\n'); log.flush()
            if update % c['save_every'] == 0 and update < total_updates:
                torch.save({'student': student.state_dict(), 'ema': ema.state_dict(), 'opt': opt.state_dict(), 'update': update,
                            'rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state()}, sdir / 'latest.pt.tmp')
                os.replace(sdir / 'latest.pt.tmp', sdir / 'latest.pt')
        prev = {k: v.detach().cpu().clone() for k, v in ema.state_dict().items()}
        torch.save({'state_dict': prev, 'steps': n, 'update': update, 'sigma_grid': g_s.cpu().tolist(), 'arch': ARCH,
                    'parameterization': 'EDM denoiser D(x, sigma) via model.edm.denoise; Euler sampler, no CFG at inference'},
                   sdir / 'final_ema.pt.tmp')
        os.replace(sdir / 'final_ema.pt.tmp', final); (sdir / 'latest.pt').unlink(missing_ok=True)
        print(f'stage {n}: {update} updates done, peak VRAM {peak / 1e9:.2f} GB', flush=True)
        del teacher, student, ema, opt; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
