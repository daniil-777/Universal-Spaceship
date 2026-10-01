"""vlm/train/colab/narrator.py — the Narrator on a Colab GPU. LoRA r 16 / alpha 32 on the 210 text projections of
SmolVLM-256M (lora_train.py's RX), vision_positions_fix, 512² square frames, the assistant-only loss mask (data.py), bf16 on
A100/L4 or fp16 + a grad scaler on T4. One epoch is a stratified sample of the train rows (pack_io.stratified_order, family
x task); steps and the cosine schedule are sized to the time budget from the measured speed; the adapter, optimizer and
position go to Drive every few minutes and a re-run resumes (the budget counts across sessions). Then merge (merge.py),
the deploy export in the narrator-v0 layout (export_decoder.py: builder q4f16 decoder + fix_io, our q8 vision, fp16
embed_tokens, configs, preprocessor with do_resize false), parity gates (c) and (e) (check_parity.py) and the slot
evaluation on the test views (generate.py + slot_eval.mjs)."""
import json, math, os, subprocess, sys, time
import numpy as np, torch
from torch.utils.data import DataLoader, Dataset
from vlm.train.colab.env import save_json, load_json
from vlm.train.colab.pack_io import stratified_order
from vlm.train.colab.eye import amp_ctx, lr_at

class Ordered(Dataset):
    """rows[order[i]] of a NarratorData, None for a row that fails its loss-mask check or image read (logged, skipped)."""
    def __init__(self, base, order): self.base, self.order = base, order
    def __len__(self): return len(self.order)
    def __getitem__(self, i):
        try: return self.base[self.order[i]]
        except (AssertionError, OSError, ValueError) as e: print(f'skipping row {self.order[i]}: {e}', flush=True); return None

def collate_skip(batch):
    from vlm.train.narrator.data import collate
    batch = [b for b in batch if b is not None]
    return collate(batch) if batch else None

def lora_model(device):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForVision2Seq
    from vlm.train.narrator.export_decoder import BASE, vision_positions_fix
    from vlm.train.narrator.lora_train import RX, lora_targets
    m = vision_positions_fix(AutoModelForVision2Seq.from_pretrained(BASE, dtype=torch.float32)); lora_targets(m)
    for p in m.parameters(): p.requires_grad = False
    m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=RX)); m.config.use_cache = False
    return m.to(device)

def batch_loss(m, b, dev, dtype, scale=1.0):
    with amp_ctx(dtype, dev): return m(**{k: v.to(dev, non_blocking=True) for k, v in b.items()}).loss / scale

def train_narrator(data, out, processor_dir, plan, budget_min=180, epoch_rows=400_000, ckpt_min=15, seed=0, max_steps=None, val_n=256, log=print):
    """-> {'adapter': <out>/adapter, ...}; a finished run (out/DONE.json) is not trained again."""
    done = load_json(f'{out}/DONE.json')
    if done: log(f'Narrator already trained: {done}'); return done
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from transformers import AutoProcessor
    from vlm.train.narrator.data import NarratorData
    os.makedirs(out, exist_ok=True); dev, dtype, P = plan['gpu']['device'], plan['dtype'], plan['narrator']
    torch.manual_seed(seed); torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    proc = AutoProcessor.from_pretrained(processor_dir); m = lora_model(dev); m.train()
    base = NarratorData(f'{data}/narrator/train.jsonl', proc); order = stratified_order(base.rows, epoch_rows, seed)
    vbase = NarratorData(f'{data}/narrator/val.jsonl', proc); vorder = stratified_order(vbase.rows, val_n, seed + 1, max_repeat=1)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=P['lr'], weight_decay=0.0)
    scaler = torch.amp.GradScaler('cuda', enabled=dev == 'cuda' and dtype == 'fp16')
    eff = P['batch'] * P['accum']; S = {'step': 0, 'pos': 0, 'elapsed': 0.0, 'total': None, 'history': [], 'skipped': 0}
    if os.path.exists(f'{out}/state.pt'):
        st = torch.load(f'{out}/state.pt', map_location='cpu', weights_only=False)
        set_peft_model_state_dict(m, st['adapter']); opt.load_state_dict(st['opt']); scaler.load_state_dict(st['scaler']); S.update(st['S'])
        log(f'resumed the Narrator at step {S["step"]} (row {S["pos"]} of {len(order)}, {S["elapsed"] / 60:.1f} of {budget_min} min used)')
    log(f'Narrator: {len(base.rows)} train rows -> an epoch of {len(order)} stratified rows; batch {P["batch"]} x accum {P["accum"]}, lr {P["lr"]}, {dtype} on {plan["gpu"]["name"]}')
    kw = dict(num_workers=plan['workers'], persistent_workers=plan['workers'] > 0, pin_memory=dev == 'cuda', **({'prefetch_factor': 4} if plan['workers'] else {}))
    vdl = DataLoader(Ordered(vbase, vorder), batch_size=P['batch'], collate_fn=collate_skip, num_workers=plan['workers']) if vorder else None
    def val_loss():
        if vdl is None: return None
        m.eval(); tot = n = 0
        with torch.no_grad():
            for b in vdl:
                if b is not None: k = int((b['labels'] != -100).any(1).sum()); tot += float(batch_loss(m, b, dev, dtype)) * k; n += k
        m.train(); return tot / n if n else None
    def save_state():
        torch.save({'adapter': get_peft_model_state_dict(m), 'opt': opt.state_dict(), 'scaler': scaler.state_dict(), 'S': S}, f'{out}/state.tmp'); os.replace(f'{out}/state.tmp', f'{out}/state.pt')
    budget, stop, losses, micro, tick, t_ck, mark, bad = budget_min * 60, None, [], 0, time.time(), time.time(), None, 0
    total_rows = len(order); left_steps = (total_rows - S['pos']) // eff
    dl = DataLoader(Ordered(base, order[S['pos']:]), batch_size=P['batch'], shuffle=False, collate_fn=collate_skip, **kw)
    opt.zero_grad(set_to_none=True)
    for b in dl:
        S['pos'] += P['batch']
        if b is None: S['skipped'] += P['batch']; continue
        loss = batch_loss(m, b, dev, dtype, P['accum'])
        if not torch.isfinite(loss):   # fp16 overflow: the scaler skips the step; a run of them falls back to fp32
            bad += 1; opt.zero_grad(set_to_none=True); micro = 0
            if bad >= 20 and dtype == 'fp16': dtype = 'fp32'; scaler = torch.amp.GradScaler('cuda', enabled=False); log('Narrator: 20 non-finite fp16 losses in a row, continuing in fp32')
            continue
        bad = 0; scaler.scale(loss).backward(); micro += 1; losses.append(loss.item() * P['accum'])
        if micro % P['accum']: continue
        if S['total'] is None and mark is None and S['step'] >= 5: mark = (S['step'], S['elapsed'])
        if S['total'] is None and mark and S['step'] - mark[0] >= 20:   # the schedule, sized to the budget from this session's speed
            rate = (S['elapsed'] - mark[1]) / (S['step'] - mark[0]); S['total'] = int(min(S['step'] + left_steps, S['step'] + 0.93 * (budget - S['elapsed']) / max(1e-6, rate)))
            log(f'Narrator: {rate:.3f} s/step ({eff / rate:.1f} rows/s), planning {S["total"]} steps = {S["total"] * eff} rows')
        total = S['total'] or (S['step'] + left_steps)
        for g in opt.param_groups: g['lr'] = P['lr'] * lr_at(S['step'], total, min(100, max(1, total // 20)), floor=0.1)
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        S['step'] += 1; now = time.time(); S['elapsed'] += now - tick; tick = now
        if S['step'] % 25 == 0: log(f'Narrator step {S["step"]}/{total} loss {np.mean(losses[-25 * P["accum"]:]):.4f} lr {opt.param_groups[0]["lr"]:.2e} {S["elapsed"] / 60:.1f} min')
        if max_steps and S['step'] >= max_steps: stop = 'max_steps'
        elif S['step'] >= total: stop = 'steps'
        elif S['elapsed'] >= budget: stop = 'time'
        if stop or time.time() - t_ck >= ckpt_min * 60:
            vl = val_loss(); S['history'].append({'step': S['step'], 'rows': S['pos'], 'train_loss': float(np.mean(losses[-100:])) if losses else None, 'val_loss': vl, 'min': round(S['elapsed'] / 60, 1)})
            log(f'Narrator checkpoint at step {S["step"]}: val loss {vl}'); save_state(); t_ck = time.time(); now = time.time(); S['elapsed'] += now - tick; tick = now
        if stop: break
    if stop is None: stop = 'epoch'; save_state()
    m.save_pretrained(f'{out}/adapter')
    res = {'adapter': f'{out}/adapter', 'steps': S['step'], 'rows_seen': S['pos'], 'epoch_rows': total_rows, 'stopped': stop, 'minutes': round(S['elapsed'] / 60, 1), 'batch': P['batch'], 'accum': P['accum'],
           'lr': P['lr'], 'dtype': dtype, 'gpu': plan['gpu']['name'], 'skipped_rows': S['skipped'], 'last_val_loss': S['history'][-1]['val_loss'] if S['history'] else None}
    save_json(f'{out}/train_log.json', {**res, 'history': S['history']}); save_json(f'{out}/DONE.json', res); log(f'Narrator trained: {res}')
    return res

def eval_rows(records, frames_root, out, n_parity=8):
    """The evaluation rows from the package's records (narrator_frame = '<shard>/<key>.jpg'): the Z test views with the
    detail prompt (slot evaluation and the stop set) and the first test frames with the one-sentence prompt (parity)."""
    from vlm.train.common import read_jsonl
    R = read_jsonl(records); img = lambda r: os.path.join(frames_root, r['narrator_frame'])
    test = [r for r in R if r.get('split') == 'test']; z = [r for r in test if r['family'] == 'Z'] or [r for r in R if r['family'] == 'Z'][:20]
    rows = {'zoom_test.jsonl': [{'key': r['key'], 'image': img(r), 'prompt': 'Describe the image in detail.'} for r in z],
            'stop_set.jsonl': [{'image': img(r), 'prompt': 'Describe the image in detail.'} for r in z[:40]],
            'parity_facts.jsonl': [{'image': img(r), 'prompt': 'Describe the image in one sentence.'} for r in (test or R)[:n_parity]]}
    os.makedirs(out, exist_ok=True)
    for f, xs in rows.items():
        with open(os.path.join(out, f), 'w') as fh: fh.write(''.join(json.dumps(x) + '\n' for x in xs))
    return {f: len(xs) for f, xs in rows.items()}

def merge_export(adapter, work, export_dir, baseline, evaldir, log=print):
    """merge.py -> <work>/merged; export_decoder.export_folder -> export_dir (q4f16 decoder, q8 vision, fp16 embed, configs);
    check_parity.py gates (c) vs the G2 published numbers and (e) the stop rate. Returns the parity summary (failures included)."""
    from vlm.train.narrator import merge, export_decoder, check_parity
    merged = f'{work}/merged'
    if not os.path.exists(f'{merged}/model.safetensors'): merge.main(['--adapter', adapter, '--out', merged])
    if not os.path.exists(f'{export_dir}/onnx/decoder_model_merged_q4f16.onnx'):
        export_decoder.export_folder(merged, export_dir, processor_src=str(export_decoder.SMOLVLM_DIR), dtypes=['q4f16'], vision=[])
    argv, sys.argv = sys.argv, ['check_parity', '--model-dir', export_dir, '--src', merged, '--facts', f'{evaldir}/parity_facts.jsonl', '--stop-set', f'{evaldir}/stop_set.jsonl', '--baseline', baseline]
    try: check_parity.main()
    except SystemExit as e: log(f'check_parity exit {e.code}')
    finally: sys.argv = argv
    p = load_json(f'{export_dir}/parity.json', {})
    summary = {'c': {d: {k: v for k, v in c.items() if not k.endswith('per_sample')} for d, c in (p.get('c') or {}).items()}, 'e': p.get('e'), 'vision': (p.get('b') or {}).get('candidate'), 'fails': p.get('fails')}
    log(f'Narrator parity: {json.dumps(summary)}'); return summary

def slot_eval(work, export_dir, records, evaldir, node, code_dir, device='cuda', log=print):
    """Greedy outputs of the merged model and of the zero-shot base on the Z test views, scored by slot_eval.mjs."""
    from vlm.train.narrator import generate
    from vlm.train.narrator.export_decoder import BASE
    outs = {}
    for name, model in (('merged', f'{work}/merged'), ('base', BASE)):
        outs[name] = f'{work}/gen_{name}.jsonl'
        if not os.path.exists(outs[name]): generate.main(['--model', model, '--processor-dir', export_dir, '--rows', f'{evaldir}/zoom_test.jsonl', '--out', outs[name], '--device', device])
    r = subprocess.run([node, f'{code_dir}/vlm/gen/text/slot_eval.mjs', '--outputs', outs['merged'], '--base', outs['base'], '--records', records, '--geo', 'none'], capture_output=True, text=True)
    try: res = json.loads(r.stdout)
    except ValueError: res = {'error': (r.stderr or r.stdout)[-2000:]}
    save_json(f'{work}/slot_eval_test.json', res)
    brief = {k: res.get(k) for k in ('gates',)} | {n: {k: (res.get(n) or {}).get(k) for k in ('n', 'recall_core', 'halluc')} for n in ('merged', 'zero_shot')}
    log(f'slot evaluation (test Z views): {json.dumps(brief)}'); return res
