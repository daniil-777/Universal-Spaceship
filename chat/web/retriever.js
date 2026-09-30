// chat/web/retriever.js — CAPCOM's fact retriever in the page: the same BM25 as chat/retrieval.py (the training side), term for
// term and in the same summation order, so the page shows the model exactly the facts its training rows showed.
// Parity: chat/tests/retrieval_parity.test.mjs runs both on the same queries.
export const K1 = 1.2, B = 0.75, TOP_K = 3, PREV_W = 0.35, MIN_SCORE = 2.0;
export const STOP = new Set(`a an the is are was were be been being am of to in on for with and or but it its this that these those what which who whom
how why when where do does did done can could would should will shall may might must i you he she we they me my your our their them us as at
by from up about into than then so if not no yes please tell explain there here some any all just also very really much many more most too
hi hey hello thanks thank ok okay`.split(/\s+/));
const WORD = /[a-z0-9]+/g;

// a light plural stemmer (the same rules as chat/retrieval.py plural())
export function plural(w) {
  if (w.length > 4 && w.endsWith('ies')) return w.slice(0, -3) + 'y';
  if (w.length > 4 && ['ches', 'shes', 'xes', 'sses'].some((e) => w.endsWith(e))) return w.slice(0, -2);
  if (w.length > 3 && w.endsWith('s') && !['ss', 'us', 'is'].some((e) => w.endsWith(e))) return w.slice(0, -1);
  return w;
}

export function tokens(text) {
  const out = [];
  for (let w of String(text).toLowerCase().match(WORD) || []) {
    w = plural(w);
    if (!STOP.has(w)) out.push(w);
  }
  return out;
}

export function docTokens(f) {
  const kw = (f.keywords || []).join(' ');
  return tokens(`${f.title || ''} ${f.text} ${kw} ${kw}`);
}

export class BM25 {
  constructor(facts) {
    this.facts = facts; this.ids = facts.map((f) => f.id); this.byId = new Map(facts.map((f) => [f.id, f]));
    this.tf = []; this.len = []; const df = new Map();
    for (const f of facts) {
      const t = docTokens(f), c = new Map();
      for (const w of t) c.set(w, (c.get(w) || 0) + 1);
      this.tf.push(c); this.len.push(t.length);
      for (const w of c.keys()) df.set(w, (df.get(w) || 0) + 1);
    }
    const n = facts.length; this.avg = this.len.reduce((a, b) => a + b, 0) / Math.max(n, 1);
    this.idf = new Map([...df].map(([w, d]) => [w, Math.log(1 + (n - d + 0.5) / (d + 0.5))]));
  }
  query(text, prev = '') {
    const q = new Map();
    for (const w of tokens(text)) q.set(w, (q.get(w) || 0) + 1.0);
    for (const w of tokens(prev)) q.set(w, (q.get(w) || 0) + PREV_W);
    return q;
  }
  scores(text, prev = '') {
    const q = this.query(text, prev), out = [];
    for (let i = 0; i < this.tf.length; i++) {
      const c = this.tf[i], norm = K1 * (1 - B + B * this.len[i] / this.avg); let s = 0.0;
      for (const [w, qw] of q) { const f = c.get(w); if (f) s += qw * this.idf.get(w) * f * (K1 + 1) / (f + norm); }
      if (s > 0) out.push([s, this.ids[i]]);
    }
    out.sort((a, b) => b[0] - a[0] || (a[1] < b[1] ? -1 : a[1] > b[1] ? 1 : 0));
    return out;
  }
  // [[id, score]] of the top k facts scoring at least minScore
  search(text, prev = '', k = TOP_K, minScore = MIN_SCORE) {
    return this.scores(text, prev).slice(0, k).filter(([s]) => s >= minScore).map(([s, id]) => [id, s]);
  }
}
