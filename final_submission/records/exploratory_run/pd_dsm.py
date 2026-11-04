"""Progressive distillation of the Fastgen EDM teacher with an auxiliary EDM denoising (score-matching) loss.

Stages: 32 -> 16 -> 8 -> 4 -> 2 -> 1 student steps (deterministic Euler on nested Karras grids).
  * Stage 32: target = 2 guided (CFG) Heun steps of the frozen EDM teacher on the 64-step master grid.
    CFG is baked into the student, so every student step costs exactly one backbone call.
  * Later stages: target = 2 Euler steps of the previous stage's final EMA student.
Loss per update = PD loss + dsm_weight * EDM denoising loss on real train images (log-normal sigma).
Both are EDM-weighted (lambda(sigma) = (sigma^2 + sd^2) / (sigma * sd)^2, i.e. MSE in F-space).

A stage ends when the training loss converges (window means stop improving), never by FID.
Every `eval_every` updates the EMA student generates 3,020 images (20/class, batch 16,
seed 42 + batch index, bf16) and Official FID (3,020 validation refs) and Valid FID
(755-image development holdout) are recorded to PROGRESS.md. FID is logged only.
"""
import argparse, copy, hashlib, json, math, os, random, shutil, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEAM = Path('/root/Diffusion 2/team_XX')        # protected TA dataset.py + train-split adapter
sys.path.insert(0, str(ROOT)); sys.path.insert(1, str(TEAM))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from cleanfid import fid, features

from model.unet import ClassConditionalUNet
from model.edm import denoise
from teacher.data import TrainImages

FID_DIR = Path('/root/Diffusion 2/FID/20261018_142153')
ARCH = dict(image_size=64, in_channels=3, num_classes=151, model_channels=130, channel_mult=(1, 2, 3, 4),
            num_res_blocks=2, attention_resolutions=(16,), dropout=0.0)
PARAMS = 89_577_673
NULL = 151
DEFAULTS = dict(
    stages=[32, 16, 8, 4, 2, 1], master_steps=64, sigma_min=0.002, sigma_max=80.0, rho=7.0, sigma_data=0.5,
    cfg=2.5, dsm_weight=0.1, p_mean=-1.2, p_std=1.2,
    batch=64, micro=32, lr=1e-4, ema=0.999, clip=1.0, seed=42,
    eval_every=100, window=500, rel_tol=0.03, patience=3, fid_rises=2, min_updates=2000, max_updates=10000,
    save_every=500, log_every=20, vram_limit=20e9,
    teacher=str(ROOT / 'checkpoints/edm_100k_ema.pt'),
    data_root='/root/Diffusion/code/workspace_distillation/data/pokemon-generation-one-22k',
    train_manifest=str(TEAM / 'data/development/seed42_holdout5/train.txt'),
)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write_json(p, v):
    t = Path(str(p) + '.tmp'); t.write_text(json.dumps(v, indent=1, ensure_ascii=False) + '\n', encoding='utf-8'); t.replace(p)


def build(state_dict, train):
    net = ClassConditionalUNet(**ARCH, activation_checkpoint_resolutions=(64, 32, 16, 8) if train else ())
    net.load_state_dict(state_dict, strict=True)
    assert sum(p.numel() for p in net.parameters()) == PARAMS
    return net.cuda().train(train).requires_grad_(train)


class Grids:
    """Nested Karras grids: the N-step grid is every (M/N)-th level of the M-step master grid (+ sigma=0)."""

    def __init__(self, c):
        M = c['master_steps']; i = torch.arange(M, dtype=torch.float64)
        lv = (c['sigma_max'] ** (1 / c['rho']) + i / (M - 1) * (c['sigma_min'] ** (1 / c['rho']) - c['sigma_max'] ** (1 / c['rho']))) ** c['rho']
        self.M, self.master = M, torch.cat([lv, torch.zeros(1, dtype=torch.float64)])

    def __call__(self, n):
        assert self.M % n == 0
        return self.master[::self.M // n].float().cuda()   # n + 1 levels, last is 0


def bc(v):
    return v.reshape(-1, 1, 1, 1)


def euler(x, d, s, sn):
    return x + bc(sn - s) * (x - d) / bc(s)


def guided(net, x, s, y, cfg):
    u, c = denoise(net, torch.cat([x, x]), torch.cat([s, s]), torch.cat([torch.full_like(y, NULL), y])).chunk(2)
    return u + cfg * (c - u)


def edm_weight(s, sd):
    return (s.square() + sd ** 2) / (s * sd).square()


class Evaluator:
    def __init__(self, group=8):
        self.group = group
        off = np.load(FID_DIR / 'reference_features.npy'); val = np.load(FID_DIR / 'valid_features.npy')
        assert off.shape == (3020, 2048) and val.shape == (755, 2048)
        self.ref = {'official': (off.mean(0), np.cov(off, rowvar=False)), 'valid': (val.mean(0), np.cov(val, rowvar=False))}
        self.fmap = features.build_feature_extractor('clean', device=torch.device('cuda'), use_dataparallel=False)

    @torch.no_grad()
    def __call__(self, net, grid, folder):
        folder = Path(folder); img = folder / 'generated'; img.mkdir(parents=True, exist_ok=True)
        labels = torch.arange(151).repeat_interleave(20); calls = [0]
        hook = net.register_forward_hook(lambda m, i, o: calls.__setitem__(0, calls[0] + i[0].shape[0]))
        net.eval(); n = len(grid) - 1; t0 = time.time()
        starts = list(range(0, 3020, 16))           # protocol batches: batch 16, noise seed 42 + batch index
        for gi in range(0, len(starts), self.group):  # several protocol batches share one forward (same noise)
            chunk = starts[gi:gi + self.group]
            x = torch.cat([torch.randn((min(16, 3020 - pos), 3, 64, 64), device='cuda',
                                       generator=torch.Generator(device='cuda').manual_seed(42 + gi + k)) for k, pos in enumerate(chunk)]) * grid[0]
            lo, hi = chunk[0], min(chunk[-1] + 16, 3020); y = labels[lo:hi].cuda(); B = len(y)
            for j in range(n):
                s = grid[j].expand(B)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    d = denoise(net, x, s, y)
                x = euler(x, d, s, grid[j + 1].expand(B))
            for i, im in enumerate(x.clamp(-1, 1).float().cpu()):
                k = lo + i
                to_pil_image(((im + 1) / 2).clamp(0, 1)).save(img / f'{int(labels[k]):04d}_{k % 20:02d}.png')
        hook.remove()
        assert calls[0] == 3020 * n, (calls[0], n)          # exactly n backbone calls per image
        gen_s = time.time() - t0
        g = fid.get_files_features(sorted(str(p) for p in img.glob('*.png')), model=self.fmap, num_workers=0,
                                   batch_size=64, device=torch.device('cuda'), mode='clean', verbose=False)
        mu, cov = g.mean(0), np.cov(g, rowvar=False)
        out = {k: float(fid.frechet_distance(mu, cov, *v)) for k, v in self.ref.items()}
        shutil.rmtree(img)
        out.update(nfe=n, eval_s=time.time() - t0, gen_s=gen_s)
        return out


class Converge:
    """Window-mean plateau detector. Converged when every loss component (PD and DSM) improves
    by less than rel_tol for `patience` consecutive windows (a rising component counts as flat)."""

    def __init__(self, c, state=None):
        self.c = c; s = state or {}
        self.buf, self.prev, self.flat, self.windows = s.get('buf', []), s.get('prev'), s.get('flat', 0), s.get('windows', [])

    def state(self):
        return dict(buf=self.buf, prev=self.prev, flat=self.flat, windows=self.windows)

    def push(self, update, pd, dsm):
        self.buf.append((pd, dsm))
        if update % self.c['window']:
            return False
        cur = [float(np.mean([b[i] for b in self.buf])) for i in range(2)]; self.buf = []
        rel = None if self.prev is None else [(p - q) / p for p, q in zip(self.prev, cur)]
        self.flat = self.flat + 1 if rel is not None and max(rel) < self.c['rel_tol'] else 0
        self.prev = cur
        self.windows.append(dict(update=update, pd=cur[0], dsm=cur[1], rel_pd=None if rel is None else rel[0],
                                 rel_dsm=None if rel is None else rel[1], flat=self.flat))
        return update >= self.c['min_updates'] and self.flat >= self.c['patience']


class Report:
    def __init__(self, run, c):
        self.run, self.c = run, c
        self.path = ROOT / 'distill' / ('PROGRESS.md' if run.name == 'main' else f'PROGRESS_{run.name}.md')
        p = run / 'progress.json'
        self.data = json.loads(p.read_text()) if p.exists() else {'stages': {}, 'started_utc': now()}

    def stage(self, n):
        return self.data['stages'].setdefault(str(n), {'status': 'pending', 'evals': [], 'windows': []})

    def save(self, status):
        self.data['status'] = status; self.data['updated_utc'] = now()
        write_json(self.run / 'progress.json', self.data); self.render()

    def render(self):
        c, d = self.c, self.data
        L = ['# EDM teacher → Progressive Distillation + Diffusion loss: 실시간 기록', '',
             f"갱신: {d['updated_utc']} UTC · 상태: **{d['status']}**", '',
             '## 설정', '',
             f"- Teacher: `checkpoints/edm_100k_ema.pt` (SHA `{d.get('teacher_sha', '')[:16]}…`, 89,577,673 params), CFG {c['cfg']} — 첫 단계 target에만 사용(학생에 CFG 내재화)",
             f"- 단계: {' → '.join(map(str, c['stages']))} step. Stage 32 target = teacher guided Heun 2 step(64-step master grid), 이후 = 직전 학생 EMA의 Euler 2 step",
             f"- Loss = PD + {c['dsm_weight']} × EDM denoising (σ ~ logN({c['p_mean']}, {c['p_std']}²), 실제 train 이미지), 둘 다 EDM λ(σ) 가중",
             f"- 데이터: dev-train 16,324장 (`seed42_holdout5/train.txt`), 증강 없음",
             f"- batch {c['batch']} (micro {c['micro']}), Adam lr {c['lr']} 고정, EMA {c['ema']}, grad clip {c['clip']}, bf16",
             f"- 단계 종료: (1) loss 수렴 — {c['window']} update 구간 평균의 PD·DSM 개선이 모두 < {c['rel_tol']*100:g}% 인 구간이 {c['patience']}회 연속, **그리고** (2) Valid FID {c['fid_rises']}회 연속 상승 (최소 {c['min_updates']}, 최대 {c['max_updates']} update)",
             "- 다음 단계로 넘기는 모델: 해당 단계 EMA 중 Valid FID 최저 시점",
             f"- 평가: {c['eval_every']} update마다 EMA, 3,020장(클래스당 20), batch 16, seed 42+batch index, bf16, clean-fid",
             '',
             '**규정 메모:** 단계 종료·체크포인트 선택은 train loss와 Valid FID(dev holdout)로만 한다. Official FID(validation 3,020장)는 기록용이며 결정에 쓰지 않는다. '
             'Valid FID는 train split 내부 dev holdout 755장 기준이다. 단, 이 EDM teacher는 20K update 이후 holdout을 포함한 17,079장으로 학습되어 Valid FID는 낙관적으로 편향될 수 있다.',
             '', '## 단계 요약', '',
             '| 단계 | 상태 | update | loss 수렴 | 종료 사유 | 시작 Official | 최근 Official | 시작 Valid | 최근 Valid | 선택(Valid 최저) | peak VRAM |',
             '|---:|---|---:|---|---|---:|---:|---:|---:|---|---:|']
        for n in c['stages']:
            s = d['stages'].get(str(n), {'status': 'pending', 'evals': []}); ev = s.get('evals', [])
            f = lambda e, k: f"{e[k]:.3f}" if e else '—'
            b = s.get('best'); sel = f"{b['update']}: Valid {b['valid']:.3f} / Off {b['official']:.3f}" if b else '—'
            lc = {True: '예', False: '아니오'}.get(s.get('loss_converged'), '—')
            L.append(f"| {n} | {s['status']} | {s.get('update', 0)} | {lc} | {s.get('stop_reason', '—')} | {f(ev[0] if ev else None, 'official')} | "
                     f"{f(ev[-1] if ev else None, 'official')} | {f(ev[0] if ev else None, 'valid')} | {f(ev[-1] if ev else None, 'valid')} | "
                     f"{sel} | {s.get('peak_vram_gb', '—')} |")
        for n in c['stages']:
            s = d['stages'].get(str(n))
            if not s or not s.get('evals'):
                continue
            L += ['', f'## {n}-step 학생', '', '| update | PD loss | DSM loss | total loss | Official FID ↓ | Valid FID ↓ | 경과(분) |',
                  '|---:|---:|---:|---:|---:|---:|---:|']
            for e in s['evals']:
                g = lambda k: '—' if e.get(k) is None else f"{e[k]:.5f}"
                L.append(f"| {e['update']} | {g('pd')} | {g('dsm')} | {g('total')} | {e['official']:.3f} | {e['valid']:.3f} | {e['elapsed_min']:.1f} |")
            if s.get('windows'):
                L += ['', f'<details><summary>{n}-step 수렴 판정 구간</summary>', '', '| update | PD 평균 | PD 개선 | DSM 평균 | DSM 개선 | 연속 정체 |', '|---:|---:|---:|---:|---:|---:|']
                for w in s['windows']:
                    r = lambda v: '—' if v is None else f"{v*100:+.2f}%"
                    L.append(f"| {w['update']} | {w['pd']:.5f} | {r(w['rel_pd'])} | {w['dsm']:.5f} | {r(w['rel_dsm'])} | {w['flat']} |")
                L += ['', '</details>']
        L += ['', f"원시 기록: `distill/runs/{self.run.name}/progress.json`, `metrics.jsonl`. 단계별 최종 EMA: `distill/runs/{self.run.name}/n<N>/final_ema.pt`.", '']
        t = self.path.with_suffix('.tmp'); t.write_text('\n'.join(L), encoding='utf-8'); t.replace(self.path)


def valid_rises(evals):
    """Number of consecutive Valid FID increases at the end of the eval history."""
    v = [e['valid'] for e in evals]; r = 0; i = len(v) - 1
    while i > 0 and v[i] > v[i - 1]:
        r += 1; i -= 1
    return r


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())


def batches(loader):
    while True:
        for b in loader:
            yield b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='main')
    ap.add_argument('--config', help='JSON overrides')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    c = dict(DEFAULTS)
    if a.config:
        c.update(json.loads(Path(a.config).read_text()))
    if a.smoke:
        c.update(stages=[32, 16], window=2, patience=1, min_updates=2, max_updates=4, eval_every=4, save_every=2, log_every=1)
    run = ROOT / 'distill/runs' / a.run; run.mkdir(parents=True, exist_ok=True)
    write_json(run / 'config.json', c)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    random.seed(c['seed']); np.random.seed(c['seed']); torch.manual_seed(c['seed'])

    rep = Report(run, c)
    rep.data.setdefault('teacher_sha', sha(c['teacher']))
    assert rep.data['teacher_sha'].startswith('816638eab17241bc'), 'unexpected teacher checkpoint'
    grids, ev = Grids(c), Evaluator()
    data = TrainImages(c['data_root'], c['data_root'], 64, 151, split_file=c['train_manifest'], cache=True)
    assert len(data) == 16324
    loader = DataLoader(data, batch_size=c['micro'], shuffle=True, drop_last=True, num_workers=0,
                        generator=torch.Generator().manual_seed(c['seed']))
    it = batches(loader); sd = c['sigma_data']; accum = c['batch'] // c['micro']
    log = open(run / 'metrics.jsonl', 'a')
    t_start = time.time()

    edm_sd = torch.load(c['teacher'], map_location='cpu', weights_only=False)['state_dict']
    prev_ema = edm_sd                                           # init + teacher for the first stage
    for si, n in enumerate(c['stages']):
        st = rep.stage(n); sdir = run / f'n{n}'; sdir.mkdir(exist_ok=True)
        if st['status'] == 'completed':
            prev_ema = torch.load(sdir / 'final_ema.pt', map_location='cpu', weights_only=False)['state_dict']; continue
        first = si == 0
        teacher = build(prev_ema, train=False)
        student = build(prev_ema, train=True)
        ema = copy.deepcopy(student).eval().requires_grad_(False)
        opt = torch.optim.Adam(student.parameters(), lr=c['lr'])
        conv = Converge(c); update = 0
        if (sdir / 'latest.pt').exists():                     # resume
            s = torch.load(sdir / 'latest.pt', map_location='cpu', weights_only=False)
            student.load_state_dict(s['student']); ema.load_state_dict(s['ema']); opt.load_state_dict(s['opt'])
            update, conv = s['update'], Converge(c, s['conv']); torch.set_rng_state(s['rng']); torch.cuda.set_rng_state(s['cuda_rng'])
            st['evals'] = [e for e in st['evals'] if e['update'] <= update]
            if st.get('best') and st['best']['update'] > update:
                st.pop('best'); (sdir / 'best_ema.pt').unlink(missing_ok=True)
        g_s, g_t = grids(n), grids(2 * n)
        st.update(status='running', teacher=('EDM 100K EMA, guided Heun' if first else f'n{c["stages"][si-1]} final EMA, Euler'))
        torch.cuda.reset_peak_memory_stats()
        recent = []

        def evaluate():
            r = ev(ema, g_s, sdir / 'eval_tmp')
            k = recent[-c['eval_every']:]
            row = dict(update=update, official=r['official'], valid=r['valid'], nfe=r['nfe'], eval_s=round(r['eval_s'], 1), gen_s=round(r['gen_s'], 1),
                       pd=float(np.mean([x[0] for x in k])) if k else None, dsm=float(np.mean([x[1] for x in k])) if k else None,
                       total=float(np.mean([x[2] for x in k])) if k else None, elapsed_min=(time.time() - t_start) / 60)
            st['evals'].append(row); st['update'] = update; st['windows'] = conv.windows
            if r['valid'] < st.get('best', {}).get('valid', float('inf')):   # dev-holdout selection only
                torch.save({k: v.detach().cpu() for k, v in ema.state_dict().items()}, sdir / 'best_ema.pt.tmp')
                os.replace(sdir / 'best_ema.pt.tmp', sdir / 'best_ema.pt')
                st['best'] = dict(update=update, valid=r['valid'], official=r['official'])
            log.write(json.dumps({'stage': n, 'eval': row}) + '\n'); log.flush()
            rep.save(f'{n}-step 학습 중 (update {update})')

        if update == 0 and not st['evals']:
            evaluate()
        stop = None
        while stop is None:
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
                    target = x - bc(sa) * (xc - x) / bc(sc - sa)
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
            total = pd_t + c['dsm_weight'] * dsm_t; recent.append((pd_t, dsm_t, total)); recent = recent[-max(c['eval_every'], 100):]
            peak = torch.cuda.max_memory_allocated(); assert peak < c['vram_limit'], peak
            st['peak_vram_gb'] = round(peak / 1e9, 2)
            if update % c['log_every'] == 0:
                log.write(json.dumps({'stage': n, 'update': update, 'pd': pd_t, 'dsm': dsm_t, 'total': total, 'grad_norm': float(gn),
                                      'peak_vram': peak, 'elapsed_s': time.time() - t_start}) + '\n'); log.flush()
            conv.push(update, pd_t, dsm_t)
            loss_ok = update >= c['min_updates'] and conv.flat >= c['patience']
            st['loss_converged'] = loss_ok
            if update % c['eval_every'] == 0 or update >= c['max_updates']:
                st['update'] = update; st['windows'] = conv.windows
                if not st['evals'] or st['evals'][-1]['update'] != update:
                    evaluate()
                if loss_ok and valid_rises(st['evals']) >= c['fid_rises']:
                    stop = f'loss 수렴 + Valid FID {c["fid_rises"]}회 연속 상승'
                elif update >= c['max_updates']:
                    stop = f'최대 {c["max_updates"]} update'
            if update % c['save_every'] == 0 and not stop:
                torch.save({'student': student.state_dict(), 'ema': ema.state_dict(), 'opt': opt.state_dict(), 'update': update,
                            'conv': conv.state(), 'rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state()},
                           sdir / 'latest.pt.tmp')
                os.replace(sdir / 'latest.pt.tmp', sdir / 'latest.pt')
        best = st.get('best')
        if best and (sdir / 'best_ema.pt').exists():
            prev_ema = torch.load(sdir / 'best_ema.pt', map_location='cpu', weights_only=False)
        else:
            prev_ema = {k: v.detach().cpu().clone() for k, v in ema.state_dict().items()}; best = dict(update=update)
        torch.save({'state_dict': prev_ema, 'steps': n, 'selected_update': best['update'],
                    'selected_by': 'lowest Valid FID (755-image dev holdout) among this stage\'s EMA evals', 'sigma_grid': g_s.cpu().tolist(), 'update': update, 'arch': ARCH,
                    'parameterization': 'EDM denoiser D(x, sigma) via model.edm.denoise; Euler sampler, no CFG at inference'},
                   sdir / 'final_ema.pt')
        (sdir / 'latest.pt').unlink(missing_ok=True); (sdir / 'best_ema.pt').unlink(missing_ok=True)
        st.update(status='completed', stop_reason=stop, update=update, final_sha=sha(sdir / 'final_ema.pt'))
        rep.save(f'{n}-step 완료 → 다음 단계')
        del teacher, student, ema, opt; torch.cuda.empty_cache()
    rep.save('전체 완료')


if __name__ == '__main__':
    main()
