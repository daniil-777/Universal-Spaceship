"""vlm/train/colab/eye.py — Pilot Eye on a Colab GPU: a full fine-tune of MobileNetV4-Conv-S + heads with AMP (bf16, or fp16
with a grad scaler), class-balanced sampling over every train clip each epoch, warm-up + cosine sized to the time budget,
the best val-loss checkpoint in train.py's model.pt format, and a resumable state on Drive written every few minutes (the
budget counts across sessions). Then evaluate.py (test split, gates vs the context-prior baseline), export_onnx.py (the
encoder/heads pair with parity) and a parity sample that install_models.mjs re-runs in Node."""
import json, math, os, random, time
import numpy as np, torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from vlm.train.pilot_eye.model import PilotEye, loss_fn, BACKBONE
from vlm.train.pilot_eye.data import PilotEyeData
from vlm.train.colab.env import save_json, load_json

def amp_ctx(dtype, device):
    if device != 'cuda' or dtype == 'fp32': return torch.autocast('cpu', enabled=False)
    return torch.autocast('cuda', dtype=torch.bfloat16 if dtype == 'bf16' else torch.float16)

def lr_at(step, total, warm, floor=0.05):
    w = min(1.0, (step + 1) / max(1, warm))
    return w * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / max(1, total)))))

def val_loss(m, dl, dev, dtype):
    to = lambda d: {k: v.to(dev) for k, v in d.items()}; m.eval(); tot = n = 0
    with torch.no_grad(), amp_ctx(dtype, dev):
        for f, dt, T, M, _ in dl: b = f.shape[0]; tot += float(loss_fn(m(f.to(dev), dt.to(dev)), to(T), to(M))) * b; n += b
    m.train(); return tot / n if n else None

def train_eye(data, out, plan, budget_min=30, max_epochs=40, ckpt_min=5, size='160x96', seed=0, max_steps=None, val_limit=4096, pretrained=True, log=print):
    """-> {'model': <out>/model.pt, ...}; a finished run (out/DONE.json) is not trained again."""
    done = load_json(f'{out}/DONE.json')
    if done: log(f'Pilot Eye already trained: {done}'); return done
    os.makedirs(out, exist_ok=True); W, H = map(int, size.split('x')); dev, dtype = plan['gpu']['device'], plan['dtype']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    ds = PilotEyeData(data, 'train', augment=True, H=H, W=W); val = PilotEyeData(data, 'val', limit=val_limit, H=H, W=W)
    bs = min(plan['eye']['batch'], len(ds)); kw = dict(num_workers=plan['workers'], persistent_workers=plan['workers'] > 0, pin_memory=dev == 'cuda')
    m = PilotEye(pretrained=pretrained).to(dev)
    k = math.sqrt(bs / 32); base = [min(1e-3, 3e-4 * k), min(3e-3, 1e-3 * k)]
    opt = torch.optim.AdamW([{'params': m.enc.backbone.parameters(), 'lr': base[0]}, {'params': list(m.enc.reduce.parameters()) + list(m.heads.parameters()), 'lr': base[1]}], weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=dev == 'cuda' and dtype == 'fp16')
    S = {'step': 0, 'epoch': 0, 'elapsed': 0.0, 'total': None, 'best': None, 'best_step': None, 'history': []}
    if os.path.exists(f'{out}/state.pt'):
        st = torch.load(f'{out}/state.pt', map_location='cpu', weights_only=False)
        m.load_state_dict(st['model']); opt.load_state_dict(st['opt']); scaler.load_state_dict(st['scaler']); S.update(st['S'])
        log(f'resumed Pilot Eye at step {S["step"]} (epoch {S["epoch"]}, {S["elapsed"] / 60:.1f} of {budget_min} min used)')
    per_epoch = max(1, len(ds) // bs); budget = budget_min * 60
    vdl = DataLoader(val, batch_size=min(256, max(1, len(val))), num_workers=kw['num_workers']) if len(val) else None
    def save_state():
        torch.save({'model': m.state_dict(), 'opt': opt.state_dict(), 'scaler': scaler.state_dict(), 'S': S}, f'{out}/state.tmp'); os.replace(f'{out}/state.tmp', f'{out}/state.pt')
    def checkpoint():
        vl = val_loss(m, vdl, dev, dtype) if vdl else None; crit = vl if vl is not None else S['history'][-1]['loss'] if S['history'] else float('inf')
        S['history'].append({'step': S['step'], 'epoch': S['epoch'], 'val_loss': vl, 't': round(S['elapsed'])})
        if S['best'] is None or crit < S['best']:
            S['best'], S['best_step'] = crit, S['step']
            torch.save({'state': m.state_dict(), 'size': size, 'fold': None, 'step': S['step'], 'best_step': S['step'], 'criterion': crit, 'criterion_kind': 'val_loss' if vl is not None else 'train_loss',
                        'freeze_backbone': False, 'backbone': BACKBONE, 'data': os.path.basename(str(data).rstrip('/'))}, f'{out}/model.pt')
        log(f'Pilot Eye epoch {S["epoch"]} step {S["step"]}: val loss {vl if vl is None else round(vl, 4)} (best {S["best"]:.4f} at {S["best_step"]}); {S["elapsed"] / 60:.1f} min')
    to = lambda d: {k: v.to(dev, non_blocking=True) for k, v in d.items()}; m.train(); t_ck = tick = time.time(); stop = None; losses = []; mark = None
    while stop is None:
        sampler = WeightedRandomSampler([r['weight'] for r in ds.rows], num_samples=len(ds), replacement=True, generator=torch.Generator().manual_seed(seed + S['epoch']))
        t_ep = time.time()
        for f, dt, T, M, _ in DataLoader(ds, batch_size=bs, sampler=sampler, drop_last=True, **kw):
            # size the schedule to the budget once the speed is known (steps 10-30 of this session, past the warm-up costs)
            if S['total'] is None and mark is None and S['step'] >= 10: mark = (S['step'], S['elapsed'])
            if S['total'] is None and mark and S['step'] - mark[0] >= 20:
                rate = (S['elapsed'] - mark[1]) / (S['step'] - mark[0]); S['total'] = int(min(max_epochs * per_epoch, S['step'] + 0.95 * (budget - S['elapsed']) / max(1e-6, rate)))
                log(f'Pilot Eye: {rate:.3f} s/step, planning {S["total"]} steps ({S["total"] / per_epoch:.1f} epochs of {per_epoch})')
            total = S['total'] or max_epochs * per_epoch
            for g, b in zip(opt.param_groups, base): g['lr'] = b * lr_at(S['step'], total, min(500, max(1, total // 20)))
            with amp_ctx(dtype, dev): loss = loss_fn(m(f.to(dev, non_blocking=True), dt.to(dev)), to(T), to(M))
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); scaler.step(opt); scaler.update()
            losses.append(loss.item()); S['step'] += 1; now = time.time(); S['elapsed'] += now - tick; tick = now
            if S['step'] % 50 == 0: log(f'Pilot Eye step {S["step"]}/{total} loss {np.mean(losses[-50:]):.4f} lr {opt.param_groups[1]["lr"]:.2e}')
            if max_steps and S['step'] >= max_steps: stop = 'max_steps'; break
            if S['step'] >= total: stop = 'steps'; break
            if S['elapsed'] >= budget: stop = 'time'; break
            if time.time() - t_ck >= ckpt_min * 60: save_state(); t_ck = time.time()
        S['elapsed'] += time.time() - tick; tick = time.time()
        S['epoch'] += 1
        if stop is None and S['epoch'] >= max_epochs: stop = 'epochs'
        S['history'].append({'epoch_s': round(time.time() - t_ep, 1), 'loss': float(np.mean(losses[-per_epoch:])) if losses else None})
        checkpoint(); save_state(); now = time.time(); S['elapsed'] += now - tick; tick = now
    res = {'model': f'{out}/model.pt', 'steps': S['step'], 'epochs': S['epoch'], 'stopped': stop, 'best_step': S['best_step'], 'best_val_loss': S['best'], 'minutes': round(S['elapsed'] / 60, 1),
           'batch': bs, 'dtype': dtype, 'gpu': plan['gpu']['name']}
    save_json(f'{out}/train_summary.json', {**res, 'history': S['history']}); save_json(f'{out}/DONE.json', res); log(f'Pilot Eye trained: {res}')
    return res

def node_input(n, frame):
    """The parity input install_models.mjs rebuilds in Node: byte k of frame f is (imul(k + f*n, 2654435761) >>> 24) / 255."""
    k = np.arange(n, dtype=np.uint64) + np.uint64(frame * n)
    return ((((k * np.uint64(2654435761)) & np.uint64(0xffffffff)) >> np.uint64(24)).astype(np.float32) / np.float32(255))

def node_parity_sample(export_dir, size='160x96'):
    """ORT (CPU) outputs of the exported pair on the Node-rebuildable input; install_models.mjs compares within 1e-3."""
    import onnxruntime as ort
    W, H = map(int, size.split('x')); n = 3 * H * W; so = ort.SessionOptions(); so.intra_op_num_threads = 1
    enc = ort.InferenceSession(f'{export_dir}/encoder.onnx', so, providers=['CPUExecutionProvider']); hd = ort.InferenceSession(f'{export_dir}/heads.onnx', so, providers=['CPUExecutionProvider'])
    maps = [enc.run(None, {'pixels': node_input(n, f).reshape(1, 3, H, W)})[0] for f in range(3)]
    outs = hd.run(None, {'m0': maps[0], 'm1': maps[1], 'm2': maps[2], 'dt': np.array([[1.0, 1.0]], np.float32)})
    sample = {'size': size, 'input': 'imul', 'dt': [1.0, 1.0], 'outputs': {o.name: v.reshape(-1).tolist() for o, v in zip(hd.get_outputs(), outs)}, 'tol': 1e-3}
    save_json(f'{export_dir}/parity_sample.json', sample); return sample

def evaluate_and_export(data, out, export_dir, device='cpu', log=print):
    """evaluate.py on test (metrics_test.json with the gates and the baseline, confusion.json; a failure is logged, since the
    export matters more) and export_onnx.py -> export_dir (its parity check must pass), plus the Node parity sample."""
    from vlm.train.pilot_eye import evaluate, export_onnx
    import shutil
    ck = f'{out}/model.pt'
    try: evaluate.main(['--ckpt', ck, '--data', data, '--split', 'test', '--out', out, '--device', device])
    except Exception as e: log(f'Pilot Eye evaluation failed ({e!r}); exporting anyway'); save_json(f'{out}/metrics_test.json', {'error': repr(e)})
    export_onnx.main(['--ckpt', ck, '--data', data, '--out', export_dir]); node_parity_sample(export_dir)
    for f in ('metrics_test.json', 'confusion.json'):
        if os.path.exists(f'{out}/{f}'): shutil.copy(f'{out}/{f}', f'{export_dir}/{f}')
    m = load_json(f'{out}/metrics_test.json', {}); p = load_json(f'{export_dir}/parity.json', {})
    summary = {k: m.get(k) for k in ('mean_family_auroc', 'pooled_auroc', 'baseline_auroc', 'pooled_macro_f1', 'baseline_macro_f1', 'gates_failed')}
    summary['parity_max_abs_diff'] = p.get('max_abs_diff'); log(f'Pilot Eye test: {json.dumps(summary)}')
    return summary
