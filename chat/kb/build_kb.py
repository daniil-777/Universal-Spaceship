"""chat/kb/build_kb.py — merge the verified knowledge-base areas (kb/verified/<AREA>.jsonl + <AREA>.questions.jsonl, written by
the extract/verify workflow) into kb.json and the question splits, and calibrate the retriever's no-facts threshold.
- facts: schema-checked, ids unique, near-duplicates (token Jaccard >= DUP_J within an area) dropped (the later one);
- questions: references remapped/dropped with the facts, then split by a stable hash into eval (EVAL_PCT, never used to seed
  training dialogs) and train;
- retrieval.min_score: the largest threshold that keeps >= KEEP of the train questions' top-1 hits while rejecting the
  off-topic probes below as often as possible (both sides read it from kb.json).
  python -m chat.kb.build_kb --kb /Volumes/LaCie/astro-pilot/chat/kb"""
import argparse, json, re, zlib
from pathlib import Path
from chat.retrieval import BM25, tokens

ID = re.compile(r'^[A-Z]+-\d{3,4}$')
KINDS, LEVELS = {'fact', 'howto', 'number', 'flag', 'limit', 'why'}, {'basic', 'detail'}
DUP_J, EVAL_PCT, KEEP = 0.8, 15, 0.99
OFF_TOPIC = ['what is the weather in paris today', 'write me a python function to sort a list', 'who won the world cup', 'recommend a pizza recipe',
    'what is the capital of australia', 'tell me a joke about cats', 'how do i fix my wifi router', 'what stocks should i buy', 'translate hello into japanese',
    'who is the president of the united states', 'best laptop for gaming', 'how to lose weight fast', 'write a poem about love', 'what time is it',
    'can you book me a flight to london', 'explain quantum entanglement simply', 'what is the meaning of life', 'how do i bake sourdough bread',
    'what movies are playing tonight', 'help me with my math homework 2x+3=7', 'what is bitcoin', 'who painted the mona lisa', 'how tall is mount everest',
    'play some music', 'what is your favourite colour', 'do you like dogs', 'how do i learn guitar', 'what is the best programming language',
    'summarize the news', 'is coffee healthy']

def check_fact(f):
    need = {'id', 'area', 'title', 'text', 'keywords', 'source', 'kind', 'level'}
    miss = need - set(f)
    if miss: return f'missing {sorted(miss)}'
    if not ID.match(f['id']) or not f['id'].startswith(f['area'] + '-'): return f"bad id {f['id']}"
    if not isinstance(f['keywords'], list) or not f['keywords']: return 'keywords'
    if f['kind'] not in KINDS or f['level'] not in LEVELS: return f"kind/level {f['kind']}/{f['level']}"
    if not 3 <= len(f['text'].split()) <= 110: return f"text length {len(f['text'].split())}"
    return None

def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))

def read(path):
    return [json.loads(l) for l in Path(path).read_text().split('\n') if l.strip()] if Path(path).exists() else []

def merge(kb_dir):
    kb_dir, facts, qs, report = Path(kb_dir), [], [], {'dropped': [], 'dup': []}
    for p in sorted((kb_dir / 'verified').glob('*.jsonl')):
        if p.name.endswith('.questions.jsonl') or p.name.startswith('._'): continue
        area_facts, remap = [], {}
        for f in read(p):
            f = {k: f[k] for k in ('id', 'area', 'title', 'text', 'keywords', 'source', 'kind', 'level') if k in f}
            f['keywords'] = [str(k).lower() for k in f.get('keywords', [])]
            err = check_fact(f)
            if err: report['dropped'].append((f.get('id'), err)); continue
            dup = next((g for g in area_facts if jaccard(tokens(g['text']), tokens(f['text'])) >= DUP_J), None)
            if dup: remap[f['id']] = dup['id']; report['dup'].append((f['id'], dup['id'])); continue
            area_facts.append(f)
        facts += area_facts
        for q in read(p.with_name(p.stem + '.questions.jsonl')):
            q['facts'] = list(dict.fromkeys(remap.get(i, i) for i in q.get('facts', [])))
            qs.append(q)
    ids = [f['id'] for f in facts]; assert len(ids) == len(set(ids)), 'duplicate fact ids across areas'
    known = set(ids)
    qs = [q for q in qs if q.get('q') and q['facts'] and all(i in known for i in q['facts'])]
    seen, uq = set(), []
    for q in qs:
        k = ' '.join(tokens(q['q']))
        if k and k not in seen: seen.add(k); uq.append(q)
    return facts, uq, report

def split(qs):
    ev, tr = [], []
    for q in qs: (ev if zlib.crc32(q['q'].encode()) % 100 < EVAL_PCT else tr).append(q)
    return tr, ev

def calibrate(facts, train_q):
    b = BM25(facts)
    ins = sorted(max([s for s, _ in b.scores(q['q'])[:1]] or [0.0]) for q in train_q)
    off = sorted(max([s for s, _ in b.scores(q)[:1]] or [0.0]) for q in OFF_TOPIC)
    thr = ins[int((1 - KEEP) * len(ins))] if ins else 0.0
    return {'min_score': round(max(0.5, min(thr, 6.0)), 3), 'in_domain_kept': sum(s >= thr for s in ins) / max(1, len(ins)),
            'off_topic_rejected': sum(s < thr for s in off) / len(off)}

def build(kb_dir):
    facts, qs, report = merge(kb_dir); tr, ev = split(qs); ret = calibrate(facts, tr)
    areas = {}
    for f in facts: areas[f['area']] = areas.get(f['area'], 0) + 1
    kb = {'name': 'CAPCOM knowledge base (Astro Pilot)', 'version': 1, 'retrieval': ret, 'areas': areas, 'facts': facts}
    kb_dir = Path(kb_dir); (kb_dir / 'kb.json').write_text(json.dumps(kb, ensure_ascii=False, indent=0))
    for name, rows in (('questions.train.jsonl', tr), ('questions.eval.jsonl', ev)):
        (kb_dir / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    return {'facts': len(facts), 'areas': areas, 'questions_train': len(tr), 'questions_eval': len(ev), 'retrieval': ret,
            'dropped': report['dropped'][:20], 'n_dropped': len(report['dropped']), 'n_dup': len(report['dup'])}

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--kb', default='/Volumes/LaCie/astro-pilot/chat/kb')
    print(json.dumps(build(ap.parse_args().kb), indent=1))
