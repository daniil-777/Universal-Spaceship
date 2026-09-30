"""vlm/train/pilot_eye/train.py — MPS (CUDA if present) training with class-balanced sampling (the jsonl `weight`), AdamW,
batch 32, warm-up then cosine decay. --smoke: 200 steps on 256 samples and the loss must fall. --fold k trains on the
seed-group half k (cross-fit). The val loss is checked every --eval-every steps and at the stop; <out>/model.pt always holds
the best checkpoint so far (lowest val loss, or the recent train loss when the val split is empty), so a --max-minutes
wall-clock stop keeps the best weights. Example (from the repo root):
  python -m vlm.train.pilot_eye.train --data $LACIE/datasets/apv-pilot --out $LACIE/models/pe_smoke --smoke --device cpu"""
import argparse, json, math, os, random, time
import numpy as np, torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from vlm.train.pilot_eye.model import PilotEye, loss_fn, BACKBONE
from vlm.train.pilot_eye.data import PilotEyeData

def pick_device(name):
    if name != 'auto': return name
    return 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def loss_fell(losses, k=20):
    """(fell, first, last): the mean of the first k losses against the last k (k shrinks to half the run when it is short)."""
    k = max(1, min(k, len(losses) // 2)); first, last = sum(losses[:k]) / k, sum(losses[-k:]) / k
    return last < first, first, last

def lr_factor(step, steps, warmup, sched):
    if sched == 'const': return 1.0
    w = min(1.0, (step + 1) / max(1, warmup))
    return w * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / max(1, steps)))))

def train_mode(m, freeze=False):
    """m.train(), but a frozen backbone stays in eval mode (its BatchNorm keeps the ImageNet statistics)."""
    m.train()
    if freeze: m.enc.backbone.eval()

def mean_loss(m, dl, dev, freeze=False):
    """The mean per-sample loss over a loader (eval mode, no augmentation); None when the loader is empty."""
    to = lambda d: {k: v.to(dev) for k, v in d.items()}
    m.eval(); tot, n = 0.0, 0
    with torch.no_grad():
        for f, dt, T, M, _ in dl:
            b = f.shape[0]; tot += loss_fn(m(f.to(dev), dt.to(dev)), to(T), to(M)).item() * b; n += b
    train_mode(m, freeze); return tot / n if n else None

def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', required=True); ap.add_argument('--out', required=True); ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--steps', type=int, default=None, help='default 6000 (200 with --smoke)'); ap.add_argument('--size', default='160x96'); ap.add_argument('--fold', type=int, default=None)
    ap.add_argument('--max-minutes', type=float, default=None, help='wall-clock stop; model.pt keeps the best checkpoint so far')
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'mps', 'cuda']); ap.add_argument('--batch', type=int, default=32); ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--eval-every', type=int, default=250); ap.add_argument('--val-limit', type=int, default=2048); ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--sched', default='cosine', choices=['cosine', 'const']); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-pretrained', action='store_true', help='random backbone init (tests)')
    ap.add_argument('--freeze-backbone', action='store_true', help='train only the 1x1 reduction and the heads (a CPU run: depthwise-conv backward on the CPU is ~10x the forward)')
    return ap.parse_args(argv)

def main(argv=None):
    a = parse(argv); W, H = map(int, a.size.split('x')); os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); dev = pick_device(a.device)
    ds = PilotEyeData(a.data, 'train', augment=True, limit=256 if a.smoke else a.limit, fold=a.fold, H=H, W=W)
    if not len(ds): raise SystemExit(f'no train rows in {a.data} (fold {a.fold})')
    bs, kw = min(a.batch, len(ds)), dict(num_workers=a.workers, persistent_workers=a.workers > 0)
    dl = DataLoader(ds, batch_size=bs, sampler=WeightedRandomSampler([r['weight'] for r in ds.rows], num_samples=len(ds), replacement=True), drop_last=True, **kw)
    val = PilotEyeData(a.data, 'val', limit=a.val_limit, H=H, W=W); vdl = DataLoader(val, batch_size=64) if len(val) else None
    m = PilotEye(pretrained=not a.no_pretrained).to(dev)
    if a.freeze_backbone:
        for p in m.enc.backbone.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW([{'params': m.enc.backbone.parameters(), 'lr': 3e-4}, {'params': list(m.enc.reduce.parameters()) + list(m.heads.parameters()), 'lr': 1e-3}], weight_decay=1e-4)
    base = [g['lr'] for g in opt.param_groups]
    steps = a.steps if a.steps is not None else (200 if a.smoke else 6000); warmup = min(200, max(1, steps // 20))
    losses, step, t0, stopped, last_eval, best = [], 0, time.time(), None, -1, {'crit': None, 'step': None}
    to = lambda d: {k: v.to(dev) for k, v in d.items()}
    log = open(f'{a.out}/train_log.jsonl', 'w')
    def checkpoint(step):
        vl = mean_loss(m, vdl, dev, a.freeze_backbone) if vdl is not None else None
        crit = vl if vl is not None else (sum(losses[-50:]) / len(losses[-50:]) if losses else float('inf'))
        better = best['crit'] is None or crit < best['crit']
        if better:
            best.update(crit=crit, step=step)
            torch.save({'state': m.state_dict(), 'size': a.size, 'fold': a.fold, 'step': step, 'best_step': step, 'criterion': crit, 'criterion_kind': 'val_loss' if vl is not None else 'train_loss_avg50', 'freeze_backbone': a.freeze_backbone,
                        'backbone': BACKBONE, 'data': os.path.basename(str(a.data).rstrip('/'))}, f'{a.out}/model.pt')
        log.write(json.dumps({'step': step, ('val_loss' if vl is not None else 'train_loss_avg50'): crit, 'best': better, 't': round(time.time() - t0, 1)}) + '\n'); log.flush()
    train_mode(m, a.freeze_backbone)
    while stopped is None:
        for f, dt, T, M, _ in dl:
            k = lr_factor(step, steps, warmup, a.sched)
            for g, b in zip(opt.param_groups, base): g['lr'] = b * k
            loss = loss_fn(m(f.to(dev), dt.to(dev)), to(T), to(M)); opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item()); log.write(json.dumps({'step': step, 'loss': losses[-1], 'lr': round(base[1] * k, 7), 't': round(time.time() - t0, 1)}) + '\n'); step += 1
            if step % 20 == 0: log.flush()
            if step % a.eval_every == 0: checkpoint(step); last_eval = step
            if step >= steps: stopped = 'steps'; break
            if a.max_minutes is not None and time.time() - t0 >= a.max_minutes * 60: stopped = 'time'; break
    if last_eval != step: checkpoint(step)
    fell, first, last = loss_fell(losses)
    log.write(json.dumps({'done': True, 'steps': step, 'stopped': stopped, 'best_step': best['step'], 'best_criterion': best['crit'], 'loss_first': first, 'loss_last': last, 't': round(time.time() - t0, 1)}) + '\n'); log.close()
    print(f'loss {first:.4f} -> {last:.4f} in {time.time() - t0:.0f} s ({step} steps, stopped by {stopped}; best step {best["step"]} at {best["crit"]:.4f}) on {dev}')
    if a.smoke and not fell: raise SystemExit('smoke loss did not decrease')

if __name__ == '__main__': main()
