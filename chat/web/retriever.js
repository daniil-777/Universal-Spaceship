// chat/web/retriever.js — CAPCOM's fact retriever in the page: the same BM25 as chat/retrieval.py (the training side), term for
// term and in the same summation order, so the page shows the model exactly the facts its training rows showed.
// Parity: chat/tests/retrieval_parity.test.mjs runs both on the same queries.
export const K1 = 1.2, B = 0.75, TOP_K = 4, PREV_W = 0.35, MIN_SCORE = 2.0, ASKED_W = 0.5;
export const STOP = new Set(`a an the is are was were be been being am of to in on for with and or but it its this that these those what which who whom
how why when where do does did done can could would should will shall may might must i you he she we they me my your our their them us as at
by from up about into than then so if not no yes please tell explain there here some any all just also very really much many more most too
hi hey hello thanks thank ok okay
try want see know show get make use work look like need thing go going way mean let kind sort something anything good cool nice actually wait lot bit one still even ever now else every guy guys pls plz could would whats wat u ur im`.split(/\s+/)); // last line: conversational filler (chat/retrieval.py)
const WORD = /[a-z0-9]+/g;

// a light plural stemmer (the same rules as chat/retrieval.py plural())
export function plural(w) {
  if (w.length > 4 && w.endsWith('ies')) return w.slice(0, -3) + 'y';
  if (w.length > 4 && ['ches', 'shes', 'xes', 'sses'].some((e) => w.endsWith(e))) return w.slice(0, -2);
  if (w.length > 3 && w.endsWith('s') && !['ss', 'us', 'is'].some((e) => w.endsWith(e))) return w.slice(0, -1);
  return w;
}

// -ing / -ed, then one doubled final consonant (not l/s/z) — chat/retrieval.py suffix()
export function suffix(w) {
  if (w.length > 5 && w.endsWith('ing')) w = w.slice(0, -3);
  else if (w.length > 4 && w.endsWith('ed') && !w.endsWith('eed')) w = w.slice(0, -2);
  if (w.length > 3 && w.at(-1) === w.at(-2) && !'lsz'.includes(w.at(-1))) w = w.slice(0, -1);
  return w;
}

export function tokens(text) {
  const out = [];
  for (let w of String(text).toLowerCase().match(WORD) || []) {
    if (STOP.has(w)) continue;
    w = suffix(plural(w)); // the stop list is checked before and after stemming ('does' must not become 'doe')
    if (!STOP.has(w)) out.push(w);
  }
  return out;
}

export function docTokens(f) {
  const kw = (f.keywords || []).join(' ');
  return tokens(`${f.title || ''} ${f.text} ${kw} ${kw}`);
}

// one BM25 field: term counts per doc, doc lengths, average length, idf (chat/retrieval.py field())
export function field(docs) {
  const tf = [], len = [], df = new Map();
  for (const t of docs) {
    const c = new Map();
    for (const w of t) c.set(w, (c.get(w) || 0) + 1);
    tf.push(c); len.push(t.length);
    for (const w of c.keys()) df.set(w, (df.get(w) || 0) + 1);
  }
  const n = docs.length;
  return { tf, len, avg: len.reduce((a, b) => a + b, 0) / Math.max(n, 1), idf: new Map([...df].map(([w, d]) => [w, Math.log(1 + (n - d + 0.5) / (d + 0.5))])) };
}

function fieldScore(q, c, len, avg, idf) {
  const norm = avg ? K1 * (1 - B + B * len / avg) : K1; let s = 0.0;
  for (const [w, qw] of q) { const f = c.get(w); if (f) s += qw * idf.get(w) * f * (K1 + 1) / (f + norm); }
  return s;
}

// two fields: the fact, and (when the KB carries it) `asked` = visitors' tokens from the training dialogs, at weight ASKED_W
export class BM25 {
  constructor(facts) {
    this.facts = facts; this.ids = facts.map((f) => f.id); this.byId = new Map(facts.map((f) => [f.id, f]));
    ({ tf: this.tf, len: this.len, avg: this.avg, idf: this.idf } = field(facts.map(docTokens)));
    this.asked = facts.some((f) => f.asked);
    if (this.asked) this.a = field(facts.map((f) => (f.asked || '').split(' ').filter(Boolean)));
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
      let s = fieldScore(q, this.tf[i], this.len[i], this.avg, this.idf);
      if (this.asked) s += ASKED_W * fieldScore(q, this.a.tf[i], this.a.len[i], this.a.avg, this.a.idf);
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
