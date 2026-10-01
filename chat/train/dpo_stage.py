"""chat/train/dpo_stage.py — S4 multi-turn-aware DPO (spec §6, R5/R6; CollabLLM-style 2-turn simulated futures).
Per round: N prompt contexts (prompts/train.jsonl), K = 4 policy samples each (temperature 0.8). For every sample the untuned teacher
plays the visitor (meta.persona + meta.goal + the chat so far incl. the sample -> one short message), the page's BM25 retrieves notes
for that message (kb.json min_score), chat.prompt.messages builds the next prompt and the policy answers greedily. Reward =
rewards.score(sample, row).total + FUTURE_W * rewards.score(next_reply, next_row).total (next_row: the simulated message and its notes,
must [], abstain False, lead True). chosen = best, rejected = worst sample, pairs kept when the margin >= 0.15. Then trl.DPOTrainer with
loss_type ['apo_zero'], beta 0.3, lr 1e-6, 2 epochs. Rounds chain (round r trains on round r-1's samples' winner); the last round's
output is S4/model, earlier rounds are S4_<r>/model. N and K scale with the tier; T4 runs one round. Generation is batched; a round's
pairs are cached in pairs.json (copied to Drive at once), so a disconnect during the DPO step does not repeat the rollouts.
  python -m chat.train.dpo_stage --pkg <package> --work <dir> [--drive <dir>] [--smoke] [--teacher <id>]"""
import json
import torch
from chat.prompt import messages
from chat.rewards import score
from chat.train.common import (DEFAULT_STATE, Notes, clean_visitor, cli, free, generate, infer_dtype, keep_model, load_model, load_tok,
                               log, prompt_rows, seed_all, sim_messages)
from chat.train.sft_stage import save_model

FUTURE_W = 0.5

def history(row, reply=None):
    """The chat so far without the system message (+ the candidate reply)."""
    h = [{'role': m['role'], 'content': m['content']} for m in row['prompt'] if m['role'] != 'system']
    return h + ([{'role': 'assistant', 'content': reply}] if reply is not None else [])

def rollouts(policy, ptok, sim, stok, rows, notes, h, seed=0):
    """[[{'text', 'score', 'now', 'future', 'visitor', 'next'} x K] per row]: K samples, their simulated visitor turn and next reply."""
    k = h['k']
    cands = generate(policy, ptok, [r['prompt'] for r in rows], n=k, bs=h['gen_bs'], max_new_tokens=h['max_new'], do_sample=True,
                     temperature=h['temperature'], repetition_penalty=1.0, seed=seed)
    cands = [c if isinstance(c, list) else [c] for c in cands]
    flat = [(i, j) for i in range(len(rows)) for j in range(k)]
    meta = [rows[i].get('meta') or {} for i, _ in flat]
    sp = [sim_messages(history(rows[i], cands[i][j]), m.get('persona'), m.get('goal')) for (i, j), m in zip(flat, meta)]
    visitors = [clean_visitor(v) for v in generate(sim, stok, sp, bs=h['gen_bs'], max_new_tokens=h['sim_tokens'], repetition_penalty=1.1)]
    nrows = []
    for (i, j), m, v in zip(flat, meta, visitors):
        hist = history(rows[i], cands[i][j]); ns = notes(v, hist)
        nrows.append({'prompt': messages(hist, v, ns, m.get('state') or DEFAULT_STATE), 'notes': ns, 'user': v, 'must': [], 'abstain': False, 'lead': True})
    nexts = generate(policy, ptok, [r['prompt'] for r in nrows], bs=h['gen_bs'], max_new_tokens=h['max_new'], repetition_penalty=1.1)
    groups = [[] for _ in rows]
    for f, (i, j) in enumerate(flat):
        now, fut = score(cands[i][j], rows[i])['total'], score(nexts[f], nrows[f])['total']
        groups[i].append({'text': cands[i][j], 'score': now + FUTURE_W * fut, 'now': now, 'future': fut, 'visitor': visitors[f], 'next': nexts[f]})
    return groups

def select_pairs(groups, margin=0.15):
    """[(i, chosen, rejected, margin)]: the best vs the worst distinct sample of each group when best - worst >= margin (> 0)."""
    out = []
    for i, g in enumerate(groups):
        if len(g) < 2: continue
        b, w = max(g, key=lambda c: c['score']), min(g, key=lambda c: c['score'])
        d = b['score'] - w['score']
        if d > 0 and d >= margin and b['text'].strip() and b['text'] != w['text']: out.append((i, b['text'], w['text'], round(d, 4)))
    return out

def make_pairs(run, stage, src, rows, h, seed):
    """(pairs [{'prompt', 'user', 'chosen', 'rejected', 'margin'}], info) from rollouts of the policy at src; cached as <stage>/pairs.json
    (copied to Drive at once) so a disconnect during the DPO step does not repeat the rollouts."""
    for p in (run.dir(stage) / 'pairs.json', run.ddir(stage) / 'pairs.json' if run.drive else None):
        if p and p.exists():
            d = json.loads(p.read_text())
            if d['info']['policy'] == str(src): log(f"{stage}: reusing {len(d['pairs'])} cached pairs"); return d['pairs'], d['info']
    ptok = load_tok(src, 'left'); policy = load_model(src, run.info, dtype=infer_dtype(run.info, student=True)).eval()
    stok = load_tok(run.sim_id, 'left'); sim = load_model(run.sim_id, run.info, dtype=infer_dtype(run.info)).eval()
    groups = rollouts(policy, ptok, sim, stok, rows, Notes(run.kb), h, seed)
    del policy, sim; free()
    sel, fallback = select_pairs(groups, h['margin']), False
    if not sel and run.smoke:  # smoke only: guarantee a pair so the DPO step is exercised
        fallback = True
        sel = [(i, rows[i]['reference'], min(g, key=lambda c: c['score'])['text'], 0.0) for i, g in enumerate(groups) if rows[i].get('reference')][:2]
    pairs = [{'prompt': rows[i]['prompt'], 'user': rows[i]['user'], 'chosen': c, 'rejected': r, 'margin': d} for i, c, r, d in sel]
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None
    info = {'policy': str(src), 'contexts': len(rows), 'samples': sum(len(g) for g in groups), 'pairs': len(pairs), 'smoke_fallback': fallback,
            'mean_score': mean([c['score'] for g in groups for c in g]), 'mean_future': mean([c['future'] for g in groups for c in g]),
            'mean_margin': mean([p['margin'] for p in pairs])}
    blob = json.dumps({'info': info, 'pairs': pairs, 'examples': [g[:2] for g in groups[:20]]}, ensure_ascii=False)
    (run.dir(stage) / 'pairs.json').write_text(blob)
    if run.drive: run.ddir(stage).mkdir(parents=True, exist_ok=True); (run.ddir(stage) / 'pairs.json').write_text(blob)
    return pairs, info

def dpo_round(run, stage, src, rows, h, seed):
    """One round -> <stage>/model: pairs from rollouts of the policy at src, then DPO (apo_zero) from src."""
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer
    out = run.dir(stage); seed_all(seed)
    pairs, info = make_pairs(run, stage, src, rows, h, seed)
    log(f'{stage}: {json.dumps(info)}')
    if not pairs: log(f'{stage}: no pairs above the margin, the policy is kept'); return info | {'trained': False}
    ds = Dataset.from_list([{'prompt': p['prompt'], 'chosen': [{'role': 'assistant', 'content': p['chosen']}],
                             'rejected': [{'role': 'assistant', 'content': p['rejected']}]} for p in pairs])
    tok = load_tok(src, 'right')
    model = load_model(src, run.info, dtype=torch.float32); ref = load_model(src, run.info, dtype=infer_dtype(run.info, student=True)).eval()
    cfg = run.targs(DPOConfig, stage, out, h, beta=h['beta'], loss_type=['apo_zero'], max_length=h['max_len'], warmup_steps=0.1,
                    precompute_ref_log_probs=False)
    tr = DPOTrainer(model=model, ref_model=ref, args=cfg, train_dataset=ds, processing_class=tok, callbacks=[run.mirror_callback(stage)])
    res = tr.train(resume_from_checkpoint=run.resume(stage, out))
    save_model(tr.model, tok, out / 'model')
    info |= {'trained': True, 'steps': res.global_step, 'train_loss': round(res.training_loss, 4)}
    del tr, model, ref; free()
    return info

def dpo(run):
    """S4: h['rounds'] rounds (S4_1 .. S4); starts from the best evaluated stage so far (S3). Skipped when DONE."""
    if run.done('S4'): return run.model('S4')
    h = run.hp('s4'); R = h['rounds']; src_stage, src = run.best('S4')
    log(f'S4: {R} round(s) from {src_stage} ({src}), simulator {run.sim_id}, hp {h}')
    for r in range(1, R + 1):
        stage = 'S4' if r == R else f'S4_{r}'
        if run.done(stage):
            src = run.model(stage) or src; continue
        rows = prompt_rows(run.pkg, 'train', h['n_ctx'], seed=run.seed + 100 * r)
        info = dpo_round(run, stage, src, rows, h, seed=run.seed + r)
        if not info['trained']: keep_model(src, run.dir(stage) / 'model', run.info)  # no pairs: the policy carries forward
        run.finish(stage, round=r, **info)
        src = run.model(stage)
    return run.model('S4')

if __name__ == '__main__':
    run, _ = cli(__doc__)
    print(dpo(run))
