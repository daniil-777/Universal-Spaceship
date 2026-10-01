"""chat/eval.py — CAPCOM's held-out evaluation (spec §8, R6). evaluate(): greedy replies (max 120 new tokens, repetition penalty 1.1,
left-padded batches) to prompt rows -> chat.rewards.score -> metrics: score (mean total x 100), recall, grounded rate (grounded == 1),
lead_ok rate (exactly one lead move), nagging rate (>= 2 questions), abstain accuracy (rows that expect an abstention), false-answer
rate (an abstention expected but not given — the ship gate on eval/unanswerable), mean words. Sets: eval/single, eval/unanswerable, a
fixed seeded slice of prompts/val, and simulate_dialogs(): the eval/dialog_scripts.jsonl visitors played by the teacher (persona +
style + hidden goal + the script's next intent) for `turns` CAPCOM turns, retrieval as on the page. The overall score is the mean of the
per-set scores; S6 keeps the best stage by it. Empty or missing sets are skipped.
  python -m chat.eval --model <dir|id> --pkg <package> [--sim <dir|id>] [--out eval.json] [--smoke]"""
import json, random
from pathlib import Path
from chat.prompt import messages
from chat.rewards import questions, score
from chat.train.common import (DEFAULT_STATE, Notes, clean_visitor, find_package, free, generate, hp, infer_dtype, load_model, load_tok,
                               log, probe, prompt_rows, read_jsonl, sim_messages)

SETS = ('single', 'unanswerable', 'val', 'dialogs')

def _mean(xs):
    return round(sum(xs) / len(xs), 4) if xs else None

def metrics(results):
    """results: [{'answer', 'row', 's' (rewards.score)}] -> the metric dict."""
    if not results: return {'n': 0, 'score': None}
    S, A, R = [r['s'] for r in results], [r['answer'] for r in results], [r['row'] for r in results]
    abst = [s['abstain'] for s, row in zip(S, R) if row.get('abstain')]
    return {'n': len(S), 'score': round(100 * _mean([s['total'] for s in S]), 2), 'recall': _mean([s['recall'] for s in S if s['recall'] is not None]),
            'grounded': _mean([float(s['grounded'] == 1) for s in S]), 'lead_ok': _mean([float(s['lead'] == 1) for s in S]),
            'nagging': _mean([float(questions(a) >= 2) for a in A]), 'abstain_acc': _mean(abst),
            'false_answer': _mean([float(x == 0) for x in abst]), 'words': _mean([len(a.split()) for a in A])}

def evaluate(model, tok, rows, bs=16, max_new_tokens=120):
    """Greedy replies to prompt rows, scored: (metrics, results)."""
    answers = generate(model, tok, [r['prompt'] for r in rows], bs=bs, max_new_tokens=max_new_tokens, repetition_penalty=1.1)
    res = [{'answer': a, 'row': r, 's': score(a, r)} for a, r in zip(answers, rows)]
    return metrics(res), res

def opening(s):
    """The script's first visitor message (tolerant: opening | turns[0] user | user_turns[0])."""
    first = next((t['content'] for t in s.get('turns', []) if isinstance(t, dict) and t.get('role') == 'user'), None)
    return s.get('opening') or first or (s.get('user_turns') or [None])[0] or 'hi, what is this?'

def simulate_dialogs(policy, ptok, sim, stok, scripts, notes, turns=4, bs=16, max_new_tokens=120, sim_tokens=48):
    """Play each script for `turns` CAPCOM turns: CAPCOM answers greedily from retrieved notes, the simulator writes the next visitor
    message (intents[t] steers turn t+1). Returns (metrics over all turns + per-turn scores, transcripts)."""
    st = [{'s': s, 'hist': [], 'user': opening(s), 'log': []} for s in scripts]
    for t in range(turns):
        rows = []
        for d in st:
            ns = notes(d['user'], d['hist'])
            rows.append({'prompt': messages(d['hist'], d['user'], ns, d['s'].get('state') or DEFAULT_STATE), 'notes': ns, 'user': d['user'],
                         'must': [], 'abstain': False, 'lead': True})
        answers = generate(policy, ptok, [r['prompt'] for r in rows], bs=bs, max_new_tokens=max_new_tokens, repetition_penalty=1.1)
        for d, r, a in zip(st, rows, answers):
            d['log'].append({'turn': t, 'answer': a, 'row': r, 's': score(a, r)})
            d['hist'] += [{'role': 'user', 'content': d['user']}, {'role': 'assistant', 'content': a}]
        if t == turns - 1: break
        intents = [(d['s'].get('intents') or [])[t] if t < len(d['s'].get('intents') or []) else '' for d in st]
        sp = [sim_messages(d['hist'], d['s'].get('persona'), d['s'].get('goal'), d['s'].get('style', ''), it) for d, it in zip(st, intents)]
        vis = generate(sim, stok, sp, bs=bs, max_new_tokens=sim_tokens, repetition_penalty=1.1)
        for d, v, it in zip(st, vis, intents): d['user'] = clean_visitor(v, it or 'ok, what else can I try?')
    flat = [x for d in st for x in d['log']]
    m = metrics(flat)
    m['per_turn'] = [round(100 * _mean([x['s']['total'] for x in flat if x['turn'] == t] or [0]), 2) for t in range(turns)]
    tr = [{'id': d['s'].get('id'), 'turns': [{'visitor': x['row']['user'], 'capcom': x['answer'], 'score': round(x['s']['total'], 3)}
                                           for x in d['log']]} for d in st]
    return m, tr

def eval_sets(pkg, h):
    """{'single', 'unanswerable', 'val': prompt rows, 'dialogs': scripts} cut per h (n_* None = all; val = a fixed seeded slice)."""
    pkg = Path(pkg); cut = lambda rows, n: rows[:n] if n else rows
    val = prompt_rows(pkg, 'val', h['n_val'], seed=0)
    return {'single': cut(read_jsonl(pkg / 'eval/single.jsonl'), h['n_single']),
            'unanswerable': cut(read_jsonl(pkg / 'eval/unanswerable.jsonl'), h['n_unans']), 'val': val,
            'dialogs': cut(read_jsonl(pkg / 'eval/dialog_scripts.jsonl'), h['n_scripts'])}

def run_eval(model_path, pkg, h, info=None, sim_path=None, kb=None, samples=8):
    """The full report of one model: {'model', 'score' (mean of set scores), 'sets': {name: metrics}, 'samples', 'dialogs'}."""
    info = info or probe(); pkg = Path(pkg); sets = eval_sets(pkg, h)
    kb = kb or json.loads((pkg / 'kb.json').read_text())
    tok = load_tok(model_path, 'left'); model = load_model(model_path, info, dtype=infer_dtype(info, student=True)).eval()
    rep = {'model': str(model_path), 'sets': {}, 'samples': {}}
    for name in ('single', 'unanswerable', 'val'):
        if not sets[name]: continue
        m, res = evaluate(model, tok, sets[name], h['bs'], h['max_new'])
        rep['sets'][name] = m
        rep['samples'][name] = [{'user': r['row']['user'], 'answer': r['answer'], 'reference': r['row'].get('reference'),
                                 'total': round(r['s']['total'], 3)} for r in res[:samples]]
    if sets['dialogs'] and sim_path:
        stok = load_tok(sim_path, 'left'); sim = load_model(sim_path, info, dtype=infer_dtype(info)).eval()
        m, tr = simulate_dialogs(model, tok, sim, stok, sets['dialogs'], Notes(kb), h['turns'], h['bs'], h['max_new'])
        rep['sets']['dialogs'], rep['dialogs'] = m, tr[:samples]
        del sim
    del model; free()
    scores = [m['score'] for m in rep['sets'].values() if m.get('score') is not None]
    rep['score'] = round(sum(scores) / len(scores), 2) if scores else None
    return rep

COLS = ('score', 'single', 'unanswerable', 'val', 'dialogs', 'recall', 'grounded', 'lead_ok', 'nagging', 'false_answer', 'words')

def table(evals):
    """A compact text table of {stage: report}: set columns are set scores, false_answer is eval/unanswerable's (the ship gate), the
    rest are pooled over the single-turn sets."""
    def pooled(rep, k):
        if k == 'false_answer' and (rep['sets'].get('unanswerable') or {}).get(k) is not None: return rep['sets']['unanswerable'][k]
        ms = [m for n, m in rep['sets'].items() if n != 'dialogs' and m.get(k) is not None]
        return sum(m[k] * m['n'] for m in ms) / sum(m['n'] for m in ms) if ms else None
    fmt = lambda v: '-' if v is None else (f'{v:.1f}' if abs(v) >= 2 else f'{v:.3f}')
    lines = ['stage  ' + ' '.join(f'{c[:9]:>9}' for c in COLS)]
    for st, rep in evals.items():
        vals = [rep.get('score')] + [rep['sets'].get(n, {}).get('score') for n in SETS] + [pooled(rep, k) for k in COLS[5:]]
        lines.append(f'{st:<6} ' + ' '.join(f'{fmt(v):>9}' for v in vals))
    return '\n'.join(lines)

def eval_stage(run, stage):
    """Evaluate a finished stage's model once (cached in evals/eval_<stage>.json; 'base' = the untuned student, a reference row); None
    when the stage has no model."""
    ev = run.evals()
    if stage in ev: return ev[stage]
    path = run.student_id if stage == 'base' else run.model(stage)
    if path is None: log(f'eval {stage}: no model, skipped'); return None
    rep = run_eval(path, run.pkg, run.hp('eval'), run.info, run.sim_id, run.kb)
    rep['stage'] = stage; run.save_eval(stage, rep); print(table({stage: rep}), flush=True)
    return rep

if __name__ == '__main__':
    import argparse, os
    os.environ.setdefault('HF_HOME', '/Volumes/LaCie/astro-pilot/chat/hf')
    ap = argparse.ArgumentParser(description='CAPCOM eval'); ap.add_argument('--model', required=True); ap.add_argument('--pkg', required=True)
    ap.add_argument('--sim'); ap.add_argument('--out'); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args(); info = probe()
    rep = run_eval(a.model, find_package(a.pkg), hp('eval', info['tier'], a.smoke), info, a.sim)
    if a.out: Path(a.out).write_text(json.dumps(rep, indent=1))
    print(table({'model': rep}))
