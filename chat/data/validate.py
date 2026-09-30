"""chat/data/validate.py — check generated CAPCOM dialogs (spec §5) against the KB and their briefs; stdlib only, so generator agents
run it on their own shard and repair what it reports.
  python3 chat/data/validate.py --kb kb.json --briefs briefs/shard_007.jsonl --dialogs dialogs/shard_007.jsonl
Exit 0 when every dialog is valid and every brief has exactly one dialog."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from chat.prompt import NAMES
from chat.rewards import ABSTAIN, LEAK, comp_grounded, questions

ARCHETYPES = {'qa_chain', 'tour', 'deep_dive', 'kid', 'recruiter', 'troubleshoot', 'space_general', 'unknown', 'off_topic', 'chit_chat',
              'adversarial', 'skeptic'}
LEADS = {'question', 'suggestion', 'none'}
MAX_A, MAX_A_DEEP, MAX_U, MAX_ABSTAIN, MAX_FACTS = 80, 110, 60, 40, 4

def words(s): return len(s.split())

def check(d, facts, brief=None):
    e = []
    for k in ('id', 'brief', 'persona', 'archetype', 'state', 'goal', 'turns'):
        if k not in d: e.append(f'missing {k}')
    if e: return e
    if d['archetype'] not in ARCHETYPES: e.append(f"archetype {d['archetype']}")
    st = d['state']
    if not isinstance(st, dict) or st.get('scene') not in NAMES or not all(s in NAMES for s in st.get('seen', [])): e.append(f'state {st}')
    if brief and (d['brief'] != brief['brief'] or d['archetype'] != brief['archetype']): e.append('brief/archetype mismatch')
    t = d['turns']
    if not isinstance(t, list) or len(t) < 2: return e + ['too few turns']
    roles = [x.get('role') for x in t]
    if any(roles[i] == roles[i + 1] for i in range(len(roles) - 1)) or set(roles) - {'user', 'assistant'}: e.append('roles do not alternate')
    if roles[0] != 'user': e.append('first turn must be the visitor')
    if roles[-1] != 'assistant': e.append('last turn must be the assistant')
    n_a = roles.count('assistant')
    if not 2 <= n_a <= 6 and not (d['archetype'] in ('chit_chat', 'off_topic', 'adversarial') and n_a >= 1): e.append(f'{n_a} assistant turns')
    cap = MAX_A_DEEP if d['archetype'] in ('deep_dive', 'skeptic') else MAX_A
    users = []
    for i, x in enumerate(t):
        c = (x.get('content') or '').strip()
        if not c: e.append(f'turn {i} empty'); continue
        if LEAK.search(c): e.append(f'turn {i} role/prompt leak')
        if x['role'] == 'user':
            users.append(c)
            if words(c) > MAX_U: e.append(f'turn {i} visitor too long ({words(c)} words)')
            continue
        if words(c) > cap: e.append(f'turn {i} too long ({words(c)} > {cap} words)')
        ids = x.get('facts', [])
        if not isinstance(ids, list) or len(ids) > MAX_FACTS or any(f not in facts for f in ids): e.append(f'turn {i} facts {ids}')
        lead, q = x.get('lead'), questions(c)
        if lead not in LEADS: e.append(f'turn {i} lead {lead}')
        elif lead == 'question' and q != 1: e.append(f'turn {i} lead question but {q} question marks')
        elif lead == 'suggestion' and q > 1: e.append(f'turn {i} suggestion with {q} question marks')
        elif lead == 'none' and q: e.append(f'turn {i} lead none but asks a question')
        notes = [facts[f]['text'] for f in ids if f in facts]
        if comp_grounded(c, notes, ' '.join(users)) < 1: e.append(f'turn {i} unsupported number/flag/key (cite the fact or drop it)')
        ab = x.get('abstain')
        if ab is not None:
            if not ids: e.append(f'turn {i} abstain without facts')
            if words(ab) > MAX_ABSTAIN: e.append(f'turn {i} abstain too long')
            if not ABSTAIN.search(ab): e.append(f"turn {i} abstain must say it is not in the notes (e.g. \"that's not in my flight notes\")")
            if comp_grounded(ab, [], ' '.join(users)) < 1: e.append(f'turn {i} abstain cites a number/flag/key')
            if questions(ab) > 1: e.append(f'turn {i} abstain asks > 1 question')
    return e

def load_jsonl(p):
    out = []
    for n, l in enumerate(Path(p).read_text().split('\n'), 1):
        if not l.strip(): continue
        try: out.append(json.loads(l))
        except json.JSONDecodeError as x: out.append({'_bad_json': f'line {n}: {x}'})
    return out

def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--kb', required=True); ap.add_argument('--dialogs', required=True)
    ap.add_argument('--briefs')
    a = ap.parse_args(argv)
    facts = {f['id']: f for f in json.loads(Path(a.kb).read_text())['facts']}
    briefs = {b['brief']: b for b in load_jsonl(a.briefs)} if a.briefs else {}
    ds, bad, seen = load_jsonl(a.dialogs), 0, {}
    for d in ds:
        if '_bad_json' in d: bad += 1; print('BAD JSON', d['_bad_json']); continue
        seen[d.get('brief')] = seen.get(d.get('brief'), 0) + 1
        e = check(d, facts, briefs.get(d.get('brief')))
        if e: bad += 1; print(d.get('id'), '|', '; '.join(e))
    missing = [b for b in briefs if b not in seen]; dup = [b for b, n in seen.items() if n > 1]
    if missing: print('MISSING briefs:', ' '.join(missing[:50]), f'(+{len(missing) - 50} more)' if len(missing) > 50 else '')
    if dup: print('DUPLICATE briefs:', ' '.join(dup[:50]))
    print(f'{len(ds)} dialogs, {bad} invalid, {len(missing)} briefs missing, {len(dup)} duplicated')
    return 0 if not bad and not missing and not dup else 1

if __name__ == '__main__':
    sys.exit(main())
