"""chat/retrieval.py — CAPCOM's fact retriever: BM25 over the knowledge-base facts, the same arithmetic as chat/web/retriever.js
(the browser side), so training rows see exactly the facts the page will show. Tokens: lowercase ASCII words and numbers,
stop words dropped, plurals and -ing/-ed stemmed (plural(), suffix()); a fact is indexed as title + text + keywords twice. The query is the visitor's
message plus the previous visitor message at weight PREV_W (follow-ups such as "and how fast is it?"). Ties break by id."""
import json, math, re
from pathlib import Path

K1, B, TOP_K, PREV_W, MIN_SCORE, ASKED_W = 1.2, 0.75, 4, 0.35, 2.0, 0.5
STOP = frozenset('''a an the is are was were be been being am of to in on for with and or but it its this that these those what which who whom
how why when where do does did done can could would should will shall may might must i you he she we they me my your our their them us as at
by from up about into than then so if not no yes please tell explain there here some any all just also very really much many more most too
hi hey hello thanks thank ok okay
try want see know show get make use work look like need thing go going way mean let kind sort something anything good cool nice actually wait lot bit one still even ever now else every guy guys pls plz could would whats wat u ur im'''.split())  # last line: conversational filler ('How do I *try* the landing?' must not outrank 'landing')
_WORD = re.compile(r"[a-z0-9]+")

def plural(w):
    """A light plural stemmer (the same rules as retriever.js): -ies -> y, -ches/-shes/-xes/-sses -> drop es, -s -> drop (not -ss/-us/-is)."""
    if len(w) > 4 and w.endswith('ies'): return w[:-3] + 'y'
    if len(w) > 4 and w.endswith(('ches', 'shes', 'xes', 'sses')): return w[:-2]
    if len(w) > 3 and w.endswith('s') and not w.endswith(('ss', 'us', 'is')): return w[:-1]
    return w

def suffix(w):
    """-ing / -ed (learned, learning -> learn; landing -> land), then one doubled final consonant (running -> run), not l/s/z."""
    if len(w) > 5 and w.endswith('ing'): w = w[:-3]
    elif len(w) > 4 and w.endswith('ed') and not w.endswith('eed'): w = w[:-2]
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in 'lsz': w = w[:-1]
    return w

def tokens(text):
    out = []
    for w in _WORD.findall(text.lower()):
        if w in STOP: continue
        w = suffix(plural(w))  # the stop list is checked before and after stemming ('does' must not become 'doe')
        if w not in STOP: out.append(w)
    return out

def doc_tokens(fact):
    kw = ' '.join(fact.get('keywords', []))
    return tokens(f"{fact.get('title', '')} {fact['text']} {kw} {kw}")

def field(docs):
    """(term counts per doc, doc lengths, average length, idf) of one BM25 field."""
    tf, ln, df = [], [], {}
    for t in docs:
        c = {}
        for w in t: c[w] = c.get(w, 0) + 1
        tf.append(c); ln.append(len(t))
        for w in c: df[w] = df.get(w, 0) + 1
    n = len(docs)
    return tf, ln, sum(ln) / max(n, 1), {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

def field_score(q, c, ln, avg, idf):
    s, norm = 0.0, K1 * (1 - B + B * ln / avg) if avg else K1
    for w, qw in q.items():
        f = c.get(w)
        if f: s += qw * idf[w] * f * (K1 + 1) / (f + norm)
    return s

class BM25:
    """Two fields: the fact (title + text + keywords) and, when the KB carries it, `asked` — the space-joined retrieval tokens visitors
    used when asking about the fact in the training dialogs (chat/kb/build_kb.py), added at weight ASKED_W."""
    def __init__(self, facts):
        self.facts = list(facts); self.ids = [f['id'] for f in self.facts]; self.by_id = {f['id']: f for f in self.facts}
        self.tf, self.len, self.avg, self.idf = field([doc_tokens(f) for f in self.facts])
        self.asked = any(f.get('asked') for f in self.facts)
        if self.asked: self.atf, self.alen, self.aavg, self.aidf = field([(f.get('asked') or '').split() for f in self.facts])

    def query(self, text, prev=''):
        """{token: weight}, summed in first-seen order (JS iterates the same Map order)."""
        q = {}
        for w in tokens(text): q[w] = q.get(w, 0.0) + 1.0
        for w in tokens(prev): q[w] = q.get(w, 0.0) + PREV_W
        return q

    def scores(self, text, prev=''):
        q, out = self.query(text, prev), []
        for i, c in enumerate(self.tf):
            s = field_score(q, c, self.len[i], self.avg, self.idf)
            if self.asked: s += ASKED_W * field_score(q, self.atf[i], self.alen[i], self.aavg, self.aidf)
            if s > 0: out.append((s, self.ids[i]))
        out.sort(key=lambda x: (-x[0], x[1]))
        return out

    def search(self, text, prev='', k=TOP_K, min_score=MIN_SCORE):
        """[(id, score)] of the top k facts scoring at least min_score."""
        return [(i, s) for s, i in self.scores(text, prev)[:k] if s >= min_score]

def load_facts(path):
    p = Path(path)
    if p.suffix == '.json': return json.loads(p.read_text())['facts']
    return [json.loads(l) for l in p.read_text().split('\n') if l.strip()]
