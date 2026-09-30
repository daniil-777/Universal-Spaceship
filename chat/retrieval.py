"""chat/retrieval.py — CAPCOM's fact retriever: BM25 over the knowledge-base facts, the same arithmetic as chat/web/retriever.js
(the browser side), so training rows see exactly the facts the page will show. Tokens: lowercase ASCII words and numbers,
stop words dropped, a plural 's' stripped; a fact is indexed as title + text + keywords twice. The query is the visitor's
message plus the previous visitor message at weight PREV_W (follow-ups such as "and how fast is it?"). Ties break by id."""
import json, math, re
from pathlib import Path

K1, B, TOP_K, PREV_W, MIN_SCORE = 1.2, 0.75, 3, 0.35, 2.0
STOP = frozenset('''a an the is are was were be been being am of to in on for with and or but it its this that these those what which who whom
how why when where do does did done can could would should will shall may might must i you he she we they me my your our their them us as at
by from up about into than then so if not no yes please tell explain there here some any all just also very really much many more most too
hi hey hello thanks thank ok okay'''.split())
_WORD = re.compile(r"[a-z0-9]+")

def tokens(text):
    out = []
    for w in _WORD.findall(text.lower()):
        if len(w) > 3 and w.endswith('s') and not w.endswith('ss'): w = w[:-1]
        if w not in STOP: out.append(w)
    return out

def doc_tokens(fact):
    kw = ' '.join(fact.get('keywords', []))
    return tokens(f"{fact.get('title', '')} {fact['text']} {kw} {kw}")

class BM25:
    def __init__(self, facts):
        self.facts = list(facts); self.ids = [f['id'] for f in self.facts]; self.by_id = {f['id']: f for f in self.facts}
        self.tf, self.len, df = [], [], {}
        for f in self.facts:
            t = doc_tokens(f); c = {}
            for w in t: c[w] = c.get(w, 0) + 1
            self.tf.append(c); self.len.append(len(t))
            for w in c: df[w] = df.get(w, 0) + 1
        n = len(self.facts); self.avg = sum(self.len) / max(n, 1)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def query(self, text, prev=''):
        """{token: weight}, summed in first-seen order (JS iterates the same Map order)."""
        q = {}
        for w in tokens(text): q[w] = q.get(w, 0.0) + 1.0
        for w in tokens(prev): q[w] = q.get(w, 0.0) + PREV_W
        return q

    def scores(self, text, prev=''):
        q, out = self.query(text, prev), []
        for i, c in enumerate(self.tf):
            s, norm = 0.0, K1 * (1 - B + B * self.len[i] / self.avg)
            for w, qw in q.items():
                f = c.get(w)
                if f: s += qw * self.idf[w] * f * (K1 + 1) / (f + norm)
            if s > 0: out.append((s, self.ids[i]))
        out.sort(key=lambda x: (-x[0], x[1]))
        return out

    def search(self, text, prev='', k=TOP_K, min_score=MIN_SCORE):
        """[(id, score)] of the top k facts scoring at least min_score."""
        return [(i, s) for s, i in self.scores(text, prev)[:k] if s >= min_score]

def load_facts(path):
    p = Path(path)
    if p.suffix == '.json': return json.loads(p.read_text())['facts']
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
