"""chat/rewards.py — CAPCOM's verifiable reply scorer (spec R6): one function used by eval, DPO pair ranking and GRPO rewards.
score(answer, row) -> {'total': 0..1, <component>: 0..1 or None (not applicable), ...}. row (a prompt row from chat/data/build.py):
  notes [str] shown to the model, user str, must [str] key terms (retrieval tokens) the reference answer grounded in the notes,
  abstain bool (the notes lack the answer: an honest "not in my notes" + redirect is right), lead bool (a lead move is expected).
Components: recall (share of must terms), grounded (numbers / ?flags / keys all appear in the notes or the visitor's message),
lead (exactly one lead move at the end; two or more questions = nagging), first (does not open with a question when it has more to
say), length (8-70 words), abstain (abstains iff it should), repetition (repeated 4-grams), leak (no role/prompt text)."""
import re
from chat.retrieval import tokens

W = {'recall': 0.30, 'grounded': 0.20, 'lead': 0.15, 'abstain': 0.10, 'length': 0.10, 'first': 0.05, 'repetition': 0.05, 'leak': 0.05}
MIN_W, MAX_W = 8, 70
NUM = re.compile(r'(?<![\w.])\d[\d,]*(?:\.\d+)?')
FLAG = re.compile(r'\?[a-z][a-z_]*(?:=[\w.,-]+)?')
KEY = re.compile(r"\b(?:press|key|hit|tap)\s+['\"]?([A-Za-z0-9+\-])['\"]?(?![\w-])", re.I)
ABSTAIN = re.compile(r"\b(not in my (?:flight )?notes|(?:i )?don'?t (?:know|have)|not sure|no (?:info|information|notes)|can'?t (?:tell|say|find)|"
                     r"isn'?t (?:covered|something i)|outside (?:what|my)|beyond (?:what|my)|i'?m (?:only|just) (?:the|a) guide|not (?:something )?i can help)", re.I)
CUE = re.compile(r"\b(try|open|switch|press|click|tap|want to|would you like|how about|check out|take a look|head (?:to|over)|flip|turn on|"
                 r"toggle|pick|jump|watch|ask me|curious|shall we|next)\b", re.I)
LEAK = re.compile(r"<\|im_|<\|endoftext|^(?:user|assistant|system|capcom)\s*:|\n(?:user|assistant|system)\s*:|^notes:|\nnotes:|^state:|\nstate:", re.I)
SMALL = {str(i) for i in range(11)}
QMARK = re.compile(r'\?(?![A-Za-z_])')  # a question mark, not the '?' that starts a URL flag such as ?scenario=landing

def questions(text): return len(QMARK.findall(text))

def norm_num(s):
    s = s.replace(',', '')
    return s[:-2] if s.endswith('.0') else s

def sentences(text):
    return [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s]

def comp_recall(answer, must):
    if not must: return None
    have = set(tokens(answer))
    return sum(t in have for t in must) / len(must)

def comp_grounded(answer, notes, user):
    src = ' '.join(notes) + ' ' + user
    nums = {norm_num(n) for n in NUM.findall(src)}
    bad = [n for n in (norm_num(x) for x in NUM.findall(answer)) if n not in nums and n not in SMALL]
    low = src.lower()
    bad += [f for f in FLAG.findall(answer.lower()) if f not in low]
    keys = {k.lower() for k in KEY.findall(src)}
    bad += [k for k in KEY.findall(answer) if k.lower() not in keys and f"'{k.lower()}'" not in low]
    return max(0.0, 1.0 - 0.5 * len(bad))

def comp_lead(answer, expected=True):
    q = questions(answer); s = sentences(answer); last = s[-1] if s else ''
    if not expected: return 1.0 if q == 0 else 0.5
    if q >= 2: return 0.0
    return 1.0 if QMARK.search(last) or CUE.search(last) else 0.0

def comp_first(answer):
    s = sentences(answer)
    return 0.0 if len(s) >= 2 and s[0].endswith('?') else 1.0

def comp_length(answer):
    n = len(answer.split())
    if n < MIN_W: return n / MIN_W
    return 1.0 if n <= MAX_W else max(0.0, 1.0 - (n - MAX_W) / 60)

def comp_abstain(answer, abstain, must):
    said = bool(ABSTAIN.search(answer))
    if abstain: return 1.0 if said else 0.0
    return 0.5 if said and must else None

def comp_repetition(answer):
    w = answer.lower().split(); grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    if not grams: return 1.0
    return max(0.0, 1.0 - 3 * (1 - len(set(grams)) / len(grams)))

def comp_leak(answer):
    return 0.0 if LEAK.search(answer.strip()) else 1.0

def score(answer, row):
    answer = (answer or '').strip(); must = row.get('must') or []
    c = {'recall': None if row.get('abstain') else comp_recall(answer, must),
         'grounded': comp_grounded(answer, row.get('notes', []), row.get('user', '')),
         'lead': comp_lead(answer, row.get('lead', True)), 'first': comp_first(answer), 'length': comp_length(answer),
         'abstain': comp_abstain(answer, bool(row.get('abstain')), must), 'repetition': comp_repetition(answer), 'leak': comp_leak(answer)}
    if not answer: c = {k: (0.0 if v is not None else None) for k, v in c.items()}
    w = [k for k, v in c.items() if v is not None]
    c['total'] = sum(W[k] * c[k] for k in w) / sum(W[k] for k in w)
    return c
