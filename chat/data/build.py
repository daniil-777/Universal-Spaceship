"""chat/data/build.py — build the CAPCOM dataset package (spec §5) from the verified KB, the generated dialogs, the held-out eval
material and the general replay data; zip it for Google Drive.
Rows (one per CAPCOM turn; history = the previous turns, cut to two exchanges by chat.prompt.messages):
- sft/capcom_{train,val}.jsonl  {"prompt":[system, history..., user], "completion":[{"role":"assistant","content":...}], "meta":{...}}
  (TRL prompt-completion format: the loss covers the completion only, never the history turns written under other notes).
  Notes (RAFT): a fact-citing turn shows its facts (<= 4) + BM25 distractors up to 3, shuffled; with p WITHHOLD_P a turn that has an
  "abstain" text shows only the distractors and teaches the abstain text instead. Turns without facts show the plain BM25 hits.
- sft/general.jsonl  smol-smoltalk / AstroChat conversations as prompt-completion rows (the notebook mixes them in).
- prompts/{train,val}.jsonl  the same prompts for distillation / DPO / GRPO, with the scorer's fields (chat.rewards.score):
  notes, user, must (retrieval tokens the reference grounded in the notes), abstain, lead, reference, meta (persona, goal, state).
- eval/single.jsonl, eval/unanswerable.jsonl (prompt rows with REAL retrieval, no oracle injection), eval/dialog_scripts.jsonl.
Dialogs failing chat/data/validate.py, exact duplicates, and dialogs whose visitor turns share an 8-gram with an eval prompt are dropped.
  python -m chat.data.build --root /Volumes/LaCie/astro-pilot/chat [--version v1]"""
import argparse, hashlib, json, random, re, sys, zipfile, zlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from chat.data.validate import check
from chat.prompt import MAX_NOTES, messages, prev_user
from chat.retrieval import BM25, tokens
from chat.rewards import ABSTAIN

WITHHOLD_P, VAL_PCT, MUST_N, NGRAM, MAX_GENERAL_CHARS = 0.2, 3, 5, 8, 6000
NO_FACT_KINDS = {'unknown', 'off_topic', 'chit_chat', 'adversarial'}
DEFAULT_STATE = {'scene': 'belt', 'seen': []}

EMOJI = re.compile('[\U0001F000-\U0001FAFF☀-➿️]')

def frac(key): return (zlib.crc32(key.encode()) % 10000) / 10000

def plain(text):
    """CAPCOM speaks plain text: emoji a writer slipped in are removed."""
    return re.sub(r'\s{2,}', ' ', EMOJI.sub('', text)).strip()

def read_jsonl(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text().split('\n') if l.strip()] if p.exists() else []

def write_jsonl(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))

def must_terms(target, oracle_texts, idf):
    both = set(tokens(target)) & set(tokens(' '.join(oracle_texts)))
    return sorted(both, key=lambda w: (-idf.get(w, 0.0), w))[:MUST_N]

def grams(text, n=NGRAM):
    w = re.findall(r'[a-z0-9]+', text.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}

def turn_rows(d, facts, bm, min_score):
    """(sft_row, prompt_row) for every CAPCOM turn of a dialog."""
    out, t = [], d['turns']
    for i, turn in enumerate(t):
        if turn['role'] != 'assistant' or i == 0: continue
        history = [{'role': x['role'], 'content': plain(x['content']) if x['role'] == 'assistant' else x['content']} for x in t[:i - 1]]
        user = t[i - 1]['content']
        key, oracle = f"{d['id']}:{i}", [f for f in turn.get('facts', []) if f in facts][:4]
        hits = [h for h, _ in bm.search(user, prev_user(history), k=MAX_NOTES, min_score=min_score)]
        if oracle and turn.get('abstain') and frac(key) < WITHHOLD_P:
            ids, target, kind = [h for h in hits if h not in oracle][:MAX_NOTES], plain(turn['abstain']), 'abstain'
        elif oracle:
            ids = (oracle + [h for h in hits if h not in oracle])[:max(MAX_NOTES, len(oracle))]
            random.Random(key).shuffle(ids); target, kind = plain(turn['content']), 'answer'
        else:
            ids, target = hits, plain(turn['content'])
            kind = d['archetype'] if d['archetype'] in NO_FACT_KINDS else 'chat'
        notes = [facts[x]['text'] for x in ids]
        abstain = kind == 'abstain' or (not oracle and bool(ABSTAIN.search(target)))
        prompt = messages(history, user, notes, d['state'])
        meta = {'dialog': d['id'], 'turn': i, 'kind': kind, 'archetype': d['archetype'], 'persona': d['persona'], 'goal': d.get('goal', ''),
                'state': d['state'], 'oracle': oracle, 'notes_ids': ids}
        must = [] if abstain else must_terms(target, [facts[x]['text'] for x in oracle], bm.idf)
        out.append(({'prompt': prompt, 'completion': [{'role': 'assistant', 'content': target}], 'meta': meta},
                    {'prompt': prompt, 'notes': notes, 'user': user, 'must': must, 'abstain': abstain, 'lead': turn.get('lead') != 'none',
                     'reference': target, 'meta': meta}))
    return out

def eval_rows(root, facts, bm, min_score):
    single, unans = [], []
    for r in read_jsonl(root / 'data/eval/single_raw.jsonl'):
        hits = [h for h, _ in bm.search(r['q'], '', k=MAX_NOTES, min_score=min_score)]; notes = [facts[h]['text'] for h in hits]
        got = [f for f in r.get('facts', []) if f in hits]
        single.append({'prompt': messages([], r['q'], notes, DEFAULT_STATE), 'notes': notes, 'user': r['q'], 'abstain': not got,
                       'must': must_terms(r['reference'], [facts[f]['text'] for f in got], bm.idf) if got else [], 'lead': True,
                       'reference': r['reference'], 'meta': {'id': r['id'], 'oracle': r.get('facts', []), 'retrieval_hit': bool(got),
                                                             'importance': r.get('importance', 2), 'kind': 'eval_single'}})
    for r in read_jsonl(root / 'data/eval/unanswerable_raw.jsonl'):
        hits = [h for h, _ in bm.search(r['q'], '', k=MAX_NOTES, min_score=min_score)]; notes = [facts[h]['text'] for h in hits]
        cat = r.get('category', 'unknown'); got = [f for f in r.get('facts', []) if f in hits]
        unans.append({'prompt': messages([], r['q'], notes, DEFAULT_STATE), 'notes': notes, 'user': r['q'], 'lead': True,
                      'abstain': cat in ('unknown', 'off_topic'), 'reference': r['reference'],
                      'must': must_terms(r['reference'], [facts[f]['text'] for f in got], bm.idf) if cat == 'false_premise' and got else [],
                      'meta': {'id': r['id'], 'category': cat, 'kind': f'eval_{cat}'}})
    return single, unans

def general_rows(root):
    out = []
    for name in ('smol_smoltalk_40k.jsonl', 'astrochat.jsonl'):
        for r in read_jsonl(root / 'data/general' / name):
            msgs = r.get('messages') or r.get('conversation') or r.get('conversations')
            if not isinstance(msgs, list) or len(msgs) < 2: continue
            msgs = [{'role': m.get('role') or {'human': 'user', 'gpt': 'assistant'}.get(m.get('from'), m.get('from')),
                     'content': m.get('content') or m.get('value') or ''} for m in msgs if isinstance(m, dict)]
            if not msgs or msgs[-1]['role'] != 'assistant' or any(m['role'] not in ('system', 'user', 'assistant') for m in msgs): continue
            if sum(len(m['content']) for m in msgs) > MAX_GENERAL_CHARS or not msgs[-1]['content'].strip(): continue
            out.append({'prompt': msgs[:-1], 'completion': msgs[-1:], 'meta': {'source': name.split('.')[0], 'kind': 'general'}})
    return out

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

DATACARD = """# CAPCOM dataset {version}

Training data for CAPCOM, the tiny in-browser chat guide of the Astro Pilot web demo (spec docs/superpowers/specs/2026-09-30-astro-pilot-capcom-design.md).

- kb.json — {n_facts} verified facts about Astro Pilot and basic spaceflight, extracted from the public repository and design specs and
  checked fact-by-fact by an independent verifier; retrieval.min_score is the BM25 no-facts threshold used in training and in the page.
- sft/capcom_train.jsonl, sft/capcom_val.jsonl — one prompt-completion row per CAPCOM turn of {n_dialogs} multi-turn dialogs
  ({n_train} / {n_val} rows). Dialogs were written by Claude subagents from per-dialog briefs (12 archetypes, 22 personas, messy
  visitor styles, twists), validated (grounding of every number/flag/key in the cited facts, shape, length, one lead move) and reviewed
  by an independent critic agent. RAFT notes: cited facts + BM25 distractors; {withhold}% of fact turns withhold the facts and teach an
  honest "not in my flight notes" + redirect.
- sft/general.jsonl — {n_general} general chat rows for replay: HuggingFaceTB/smol-smoltalk (Apache-2.0) sample{astro}.
- prompts/train.jsonl, prompts/val.jsonl — the same contexts with the verifiable-reward fields (notes, user, must, abstain, lead,
  reference) for on-policy distillation, multi-turn DPO and GRPO.
- eval/single.jsonl ({n_single}), eval/unanswerable.jsonl ({n_unans}), eval/dialog_scripts.jsonl ({n_scripts}) — held out: built from
  the 15 % eval split of the KB questions (never used to seed training dialogs), real retrieval (no oracle injection), 8-gram decontaminated.

Prompt contract: chat/prompt.py (system preamble + "Notes:" + "State:" line, two exchanges of history). Licences: see LICENSES.md.
Intended use: fine-tuning small models for the Astro Pilot guide (a narrow, non-competing domain assistant). Not for real flight decisions.
"""
LICENSES = """# Licences and credits

- Astro Pilot knowledge base and CAPCOM dialogs: derived from the Astro Pilot repository (github.com/daniil-777/Universal-Spaceship) and
  written for it; dialogs generated with Claude (Anthropic) for this narrow domain assistant.
- HuggingFaceTB/smol-smoltalk — Apache-2.0 (https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk).
- patrickfleith/AstroChat — CC-BY-4.0 (https://huggingface.co/datasets/patrickfleith/AstroChat), if present in sft/general.jsonl.
"""

def build(root, version='v1', include_parts=False):
    """include_parts: also read unfinished shards' part files (a dev build while generation runs)."""
    root = Path(root); pkg = root / 'data' / f'capcom-{version}'
    kb = json.loads((root / 'kb/kb.json').read_text()); facts = {f['id']: f for f in kb['facts']}
    min_score = kb['retrieval']['min_score']; bm = BM25(kb['facts'])
    single, unans = eval_rows(root, facts, bm, min_score)
    ev_grams, ev_exact = set(), set()
    for r in single + unans: ev_grams |= grams(r['user']); ev_exact.add(' '.join(tokens(r['user'])))
    stats = {'invalid': 0, 'dup': 0, 'contaminated': 0}
    dialogs, seen = [], set()
    sources = sorted((root / 'data/dialogs').glob('shard_*.jsonl'))
    if include_parts:
        done = {p.stem for p in sources}
        sources += sorted(p for p in (root / 'data/dialogs/parts').glob('shard_*/part_*.jsonl') if p.parent.name not in done and not p.name.startswith('._'))
    for p in sources:
        for d in read_jsonl(p):
            if check(d, facts): stats['invalid'] += 1; continue
            k = (d['turns'][0]['content'].strip().lower(), d['turns'][1]['content'].strip().lower())
            if k in seen: stats['dup'] += 1; continue
            users = [x['content'] for x in d['turns'] if x['role'] == 'user']
            if any(grams(u) & ev_grams or (' '.join(tokens(u)) in ev_exact and len(tokens(u)) > 3) for u in users):
                stats['contaminated'] += 1; continue
            seen.add(k); dialogs.append(d)
    split = {'train': ([], []), 'val': ([], [])}
    for d in dialogs:
        s = 'val' if zlib.crc32(d['id'].encode()) % 100 < VAL_PCT else 'train'
        for a, b in turn_rows(d, facts, bm, min_score): split[s][0].append(a); split[s][1].append(b)
    general = general_rows(root)
    write_jsonl(pkg / 'sft/capcom_train.jsonl', split['train'][0]); write_jsonl(pkg / 'sft/capcom_val.jsonl', split['val'][0])
    write_jsonl(pkg / 'prompts/train.jsonl', split['train'][1]); write_jsonl(pkg / 'prompts/val.jsonl', split['val'][1])
    write_jsonl(pkg / 'sft/general.jsonl', general)
    write_jsonl(pkg / 'eval/single.jsonl', single); write_jsonl(pkg / 'eval/unanswerable.jsonl', unans)
    write_jsonl(pkg / 'eval/dialog_scripts.jsonl', read_jsonl(root / 'data/eval/dialog_scripts.jsonl'))
    (pkg / 'kb.json').write_text(json.dumps(kb, ensure_ascii=False))
    kinds = {}
    for r in split['train'][1]: kinds[r['meta']['kind']] = kinds.get(r['meta']['kind'], 0) + 1
    n_scripts = len(read_jsonl(pkg / 'eval/dialog_scripts.jsonl'))
    (pkg / 'DATACARD.md').write_text(DATACARD.format(version=version, n_facts=len(facts), n_dialogs=len(dialogs), n_train=len(split['train'][0]),
        n_val=len(split['val'][0]), withhold=int(WITHHOLD_P * 100), n_general=len(general),
        astro=' + patrickfleith/AstroChat (CC-BY-4.0)' if any(g['meta']['source'] == 'astrochat' for g in general) else '',
        n_single=len(single), n_unans=len(unans), n_scripts=n_scripts))
    (pkg / 'LICENSES.md').write_text(LICENSES)
    files = sorted(p for p in pkg.rglob('*') if p.is_file() and not p.name.startswith('._') and p.name != 'MANIFEST.json')
    manifest = {'version': version, 'stats': {**stats, 'dialogs': len(dialogs), 'rows_by_kind': kinds},
                'files': {str(p.relative_to(pkg)): {'sha256': sha(p), 'bytes': p.stat().st_size,
                          'rows': sum(1 for _ in open(p)) if p.suffix == '.jsonl' else None} for p in files}}
    (pkg / 'MANIFEST.json').write_text(json.dumps(manifest, indent=1))
    z = root / 'data' / f'capcom-data-{version}.zip'
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in [*files, pkg / 'MANIFEST.json']: zf.write(p, f'capcom-{version}/{p.relative_to(pkg)}')
    return {'package': str(pkg), 'zip': str(z), 'zip_mb': round(z.stat().st_size / 2 ** 20, 1), **manifest['stats'],
            'rows': {k: v['rows'] for k, v in manifest['files'].items() if v['rows'] is not None}}

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--root', default='/Volumes/LaCie/astro-pilot/chat'); ap.add_argument('--version', default='v1')
    ap.add_argument('--include-parts', action='store_true', help='dev build: also read unfinished shards')
    a = ap.parse_args(); print(json.dumps(build(a.root, a.version, a.include_parts), indent=1))
