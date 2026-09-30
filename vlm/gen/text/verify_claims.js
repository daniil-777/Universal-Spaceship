// vlm/gen/text/verify_claims.js — the §5.4 claim parser: numbers with units (digits or words, bound to their clause), ranges,
// counts, o'clock (image; "your N o'clock" = pilot bearing), 3x3 regions, compass words, verdict/action/reason words,
// outcomes, causes, hazard kinds, categories and relations, ungrounded descriptors, disagreements with the monitor, and
// gazetteer entities. Position claims carry the subject they are about; claims under "not" carry negated: true.
import { WORDS, UNIT_OF, UNITS, toNum, roundNice, RANGE_RES, NUMBER_RE, SPELLED_RE, spelledValue, isBelowCue, CLAUSE_CUT,
  esc } from './verify_numbers.js';
import { TERMS, STOP, COMMON_LOWER, isCommonOpener, PROPER_FRAME, NAME_FRAME, PRONOUNS, REASON_CTX, REASON_PHRASES, ACTION_PHRASES,
  CONTINUE_RE, OUTCOME_PHRASES, CAUSE_CUE, CAUSE_PHRASES, DISAGREE, UNGROUNDED, CATS, RELS, KIND_RE, kindOf, SUBJECTS, PRESENCE_NOUN,
  PRESENCE_CUE } from './verify_words.js';

export const norm = (s) => String(s).normalize('NFC').toLowerCase().replace(/[^\p{L}\p{N} ]+/gu, '').replace(/\s+/g, ' ').trim();
const TERM_KEYS = new Set(TERMS.map(norm));
// a name written without its diacritics ("Zurich") is the gazetteer name ("Zürich"); a diacritic the name lacks ("Zérmatt")
// is not
const fold = (s) => s.normalize('NFD').replace(/\p{M}+/gu, '').normalize('NFC');
export function makeGazetteer(names) {
  const m = new Map(), f = new Map();
  let maxWords = 1;
  for (const n of names) {
    const k = n ? norm(n) : '';
    if (!k) continue;
    if (!m.has(k)) m.set(k, String(n));
    if (!f.has(fold(k))) f.set(fold(k), String(n));
    maxWords = Math.max(maxWords, Math.min(8, k.split(' ').length));
  }
  const get = (s) => { const k = norm(s); return m.get(k) ?? (/^[\x20-\x7e]*$/.test(k) ? f.get(k) : undefined); };
  return { has: (s) => get(s) !== undefined, canonical: get, maxWords, names: [...m.values()] };
}
const isAcronym = (w) => /^[\p{Lu}\d_/-]+$/u.test(w) && /\p{Lu}/u.test(w);
const cleanWord = (w) => w.replace(/^[^\p{L}\p{N}]+/u, '').replace(/[’']s$/u, '').replace(/[^\p{L}\p{N}'’-]+$/u, '')
  .replace(/[’']s$/u, '');
// lower-casing that keeps every index (a character whose lower case changes length is left as it is)
const lowerSame = (s) => s.replace(/./gsu, (ch) => { const l = ch.toLowerCase(); return l.length === ch.length ? l : ch; });
const SENT_END = /[.!?:;]["')\]]*$/;
const SENTENCE_RE = /(?:[^.!?;]|\.(?=\d))+[.!?;]?/g;
const COUNT_WORDS = { no: 0, none: 0, zero: 0, 'a single': 1, single: 1, 'a pair of': 2, 'pair of': 2 };
const NW = `no|none|zero|a single|single|a pair of|pair of|${WORDS.slice(1).join('|')}|\\d+`;
const countOf = (k) => COUNT_WORDS[k] ?? (WORDS.indexOf(k) >= 0 ? WORDS.indexOf(k) : +k);
const COUNT_RE = new RegExp(`\\b(${NW})\\s+(rock|comet|satellite|airliner|flock|bird|hazard|jet|white|red)(?:e?s)?\\b`, 'g');
// PAPI phrasings the count pattern misses: "all four lights are red", "the lights are all white", "three of the lights show white"
const PAPI_RES = [
  [/\ball\s+(?:four\s+)?(?:of\s+the\s+)?(?:papi\s+)?(?:lights?\s+)?(?:are\s+|show\s+|read\s+|showing\s+)?(white|red)\b/g, (m) => [m[1], 4]],
  [/\blights?\s+(?:are|show|read)\s+all\s+(white|red)\b/g, (m) => [m[1], 4]],
  [new RegExp(`\\b(${NW})\\s+of\\s+the\\s+(?:four\\s+)?(?:papi\\s+)?lights?\\s+(?:show|are|read|showing|shows|is)\\s+(white|red)\\b`, 'g'),
    (m) => [m[2], countOf(m[1])]]];
const CLOCK_WORD = `(1[0-2]|[1-9]|${WORDS.slice(1, 13).join('|')})`, clockNum = (s) => (/\d/.test(s) ? +s : WORDS.indexOf(s));
const REGION_ALIAS = { 'upper center': 'upper centre', 'lower center': 'lower centre', 'top left': 'upper left', 'top right': 'upper right',
  'bottom left': 'lower left', 'bottom right': 'lower right', 'top centre': 'upper centre', 'bottom centre': 'lower centre',
  top: 'upper centre', bottom: 'lower centre', center: 'centre', middle: 'centre' };
const REGION_RE = new RegExp('\\b(upper left|upper centre|upper center|upper right|middle left|middle right|lower left|lower centre'
  + '|lower center|lower right|top left|top right|top centre|bottom left|bottom right|bottom centre)\\b', 'g');
const COMPASS_RE = /\b(north-east|north-west|south-east|south-west|northeast|northwest|southeast|southwest|north|south|east|west)\b/g;
const VERDICT_RE = /(?<![\p{L}])(?<!no )(unsafe|safe|caution)(?![\p{L}])(?! (?:action|breakout|option|distance|speed|margin|limit))/gu;
const phraseRe = (p) => new RegExp(`\\b${esc(p)}\\b`, 'g');

// ---- negation: only within a claim's clause (it ends at , ; : or a conjunction); two negations cancel ----
const NEG_CUT = /[,;:]|\b(?:and|but|while|whereas|because|so|although|though|yet|or)\b/g;
// "not" that modifies another word, not the claim ("not far from the rock", "not hard to spot", "not only a rock")
const NOT_IDIOM = /\b(?:not|n't)\s+(?:far|hard|difficult|easy|tough|only|just|merely|unlike|least|necessarily|surprising|unusual)\b/g;
// "must not exceed about 30 cm/s" states a limit, not a negated speed
const NOT_LIMIT = /\b(?:not|never|n't)\s+(?:be\s+)?(?:exceed|surpass|go above|go beyond|rise above)\b/;
// a negation of the whole clause after it; a hazard kind it reaches only as "... there is a comet"
const CLAUSE_NEG = new RegExp('\\b(?:it is not the case that|it is not true that|it isn\'t true that|not the case that|not true that'
  + '|nowhere|by no means)\\b');
// a negation right before a hazard kind's noun phrase ("no sign of a comet", "cannot see a comet")
const NP_NEG = /\b(?:no sign of|no trace of|without(?: any)?(?: sign of| trace of)?|cannot see|can't see|do not see|don't see)\s*$/;
const NEGATOR = /^(?:not|never|nor|neither|cannot)$|n't$/;
const countNeg = (ws) => ws.filter((x) => NEGATOR.test(x)).length - (ws.includes('neither') && ws.includes('nor') ? 1 : 0);
// is the claim at `at` negated? A predicate is negated by "not/never/nor/neither/n't" among the five words before it, by a
// clause negation before it, or by "no/without" right before it; a hazard kind (np = its determiner) only by a negation that
// stands right before its noun phrase
function negatedAt(low, sentAt, at, np = null) {
  const seg = low.slice(sentAt, np ?? at), cut = Math.max(-1, ...[...seg.matchAll(NEG_CUT)].map((m) => m.index + m[0].length - 1));
  let text = seg.slice(cut + 1).replace(NOT_IDIOM, ' '), n = 0;
  const cm = CLAUSE_NEG.exec(text);
  if (cm) {
    const rest = text.slice(cm.index + cm[0].length);
    if (np === null || /\bthere\b/.test(rest)) n++;
    text = rest;
  }
  const words = text.trim().split(/\s+/).filter(Boolean);
  if (NOT_LIMIT.test(words.slice(-6).join(' '))) return false;
  if (np !== null) {
    const near = countNeg(words.slice(-3));
    n += near || (NP_NEG.test(text) ? 1 : 0);
  } else {
    n += countNeg(words.slice(-5));
    // "no" and "without" negate the next word or the one after it ("no towering storm clouds"), never a phrase that opens
    // with an article ("with no intervention the flight stays clear")
    if (!/^(?:a|an|the)\s/.test(low.slice(at)) && words.slice(-2).some((x) => /^(?:no|without)$/.test(x))) n++;
  }
  return n % 2 === 1;
}
const POSITION = new Set(['clock', 'region', 'compass']);
const NEGATABLE = new Set(['number', 'range', 'count', 'clock', 'bearing_clock', 'region', 'compass', 'category', 'kind', 'cause',
  'outcome', 'entity', 'presence']);
// the object a position claim or a closing relation is about: the nearest subject before it in its sentence, else the
// nearest one after it
function subjectOf(low, s, cl, claims) {
  const text = low.slice(s.at, s.at + s.text.length), off = cl.at - s.at, found = [];
  for (const [re, get] of SUBJECTS) {
    for (const m of text.matchAll(re)) found.push({ at: m.index, end: m.index + m[0].length, ...get(m) });
  }
  for (const e of claims) {
    if (e.type === 'entity' && e.sent === cl.sent) found.push({ at: e.at - s.at, end: e.at - s.at + 1, kind: 'entity', value: e.value });
  }
  const before = found.filter((x) => x.end <= off).sort((a, b) => b.end - a.end);
  const pick = before[0] || found.filter((x) => x.at > off).sort((a, b) => a.at - b.at)[0];
  return pick ? { kind: pick.kind, value: pick.value ?? null } : null;
}

// entities first, so a place name's compass or number words are not read twice. A capitalised word that opens a sentence (or
// follows ":" / ";") is a common word, a gazetteer name, or an unknown name: a sentence-initial common word is a place only
// when a place frame follows it (a gazetteer name with any PROPER_FRAME, "Split lies on the coast"; any other word with a
// noun of NAME_FRAME, "Bright town")
function parseEntities(src, gaz, push, blank) {
  const words = [...src.matchAll(/\S+/g)], start = (i) => i === 0 || SENT_END.test(words[i - 1][0]);
  const after = (i) => src.slice(words[i].index + words[i][0].length).split(/[.!?;]/)[0];
  const noun = (i) => !PRONOUNS.has(cleanWord(words[i][0]).toLowerCase());
  const framed = (i) => noun(i) && PROPER_FRAME.test(after(i)), nameFramed = (i) => noun(i) && NAME_FRAME.test(after(i));
  for (let i = 0; i < words.length; i++) {
    const clean = cleanWord(words[i][0]), low = clean.toLowerCase(), cap = /^\p{Lu}/u.test(clean);
    let hit = null;
    for (let n = Math.min(gaz.maxWords, words.length - i); n >= 1 && !hit; n--) {
      if (words.slice(i, i + n - 1).some((x) => SENT_END.test(x[0]))) continue;
      const parts = words.slice(i, i + n).map((x) => cleanWord(x[0])), span = parts.join(' ');
      if (!gaz.has(span)) continue;
      const common1 = n === 1 && (STOP.has(clean) || (start(i) && COMMON_LOWER.has(low) && !framed(i)));
      if (cap ? !common1 : !/\d/.test(span) && parts.some((p) => !COMMON_LOWER.has(p.toLowerCase()))) hit = [span, n];
    }
    if (hit) {
      const a = words[i].index, e = words[i + hit[1] - 1];
      if (!TERM_KEYS.has(norm(hit[0]))) push('entity', gaz.canonical(hit[0]), a);
      blank(a, e.index + e[0].length - a);
      i += hit[1] - 1;
    } else if (cap && !STOP.has(clean) && !isAcronym(clean) && (!start(i) || nameFramed(i) || !isCommonOpener(low))) {
      push('unknown_entity', clean, words[i].index);
    }
  }
}

// presence claims: in a clause with a presence cue, every body or object noun ("the Earth is visible", "also in view are the
// Earth and the Moon"). The claim sits at the cue, so "the Moon is not in view" is negated; "no Moon" before the noun also is.
const PRESENCE_CUT = /[,;:]|\b(?:but|while|whereas|because|although|though)\b/g;
function parsePresence(low, sents, push) {
  for (const s of sents) {
    const text = low.slice(s.at, s.at + s.text.length);
    const cuts = [0, ...[...text.matchAll(PRESENCE_CUT)].map((m) => m.index + m[0].length), text.length];
    for (let i = 0; i + 1 < cuts.length; i++) {
      const part = text.slice(cuts[i], cuts[i + 1]), cue = PRESENCE_CUE.exec(part);
      if (!cue) continue;
      for (const m of part.matchAll(PRESENCE_NOUN)) {
        const noBefore = /\bno\s+(?:\S+\s+)?$/.test(part.slice(0, m.index));
        push('presence', m[1] === 'docking port' ? 'station' : m[1], s.at + cuts[i] + cue.index, { noBefore });
      }
    }
  }
}

export function parseClaims(text, gaz) {
  const src = String(text).replace(/[’‘]/g, "'"), low0 = lowerSame(src), c = [];
  const sents = [...src.matchAll(SENTENCE_RE)].map((m) => ({ at: m.index, text: m[0] }));
  const sentAt = (at) => { let k = 0; while (k + 1 < sents.length && sents[k + 1].at <= at) k++; return k; };
  const push = (type, value, at, extra = {}) => c.push({ type, value, at, sent: sentAt(at), ...extra });
  let w = src;
  const blank = (at, len) => { w = w.slice(0, at) + ' '.repeat(len) + w.slice(at + len); };
  const scan = (re, fn, lower = true) => {
    for (const m of [...(lower ? lowerSame(w) : w).matchAll(re)]) if (fn(m) !== false) blank(m.index, m[0].length);
  };
  parseEntities(src, gaz, push, blank);
  scan(DISAGREE, (m) => push('disagree', m[0], m.index));
  // a number carries its clause (and the clause before it) for the binding to a quantity noun
  const numAt = (m) => {
    const st = sents[sentAt(m.index)], lowS = lowerSame(st.text), off = m.index - st.at;
    const cuts = [0, ...[...lowS.matchAll(CLAUSE_CUT)].map((x) => x.index + x[0].length), lowS.length + 1];
    const a = Math.max(...cuts.filter((x) => x <= off)), b = Math.min(...cuts.filter((x) => x > off));
    const a0 = Math.max(0, ...cuts.filter((x) => x < a));
    return { clause: lowS.slice(a, b), off: off - a, len: m[0].length, prevClause: a > 0 ? lowS.slice(a0, a) : '' };
  };
  const unitAt = (uw, m) => { const unit = UNIT_OF[uw.toLowerCase()]; return { unit, dim: UNITS[unit][0], ...numAt(m) }; };
  for (const [re, get] of RANGE_RES) {
    scan(re, (m) => { const [lo, hi, uw] = get(m); push('range', null, m.index, { lo, hi, ...unitAt(uw, m) }); }, false);
  }
  scan(NUMBER_RE, (m) => {
    const v = toNum(m[1]), r = roundNice(v);
    push('number', v, m.index, { lo: r.lo, hi: r.hi, ...unitAt(m[2], m) });
  }, false);
  scan(SPELLED_RE, (m) => {
    const v = spelledValue(m[2]), at = m.index + m[0].indexOf(m[2]), r = roundNice(v), extra = unitAt(m[3], m);
    if (m[1]) push('range', null, at, isBelowCue(m[1]) ? { lo: 0, hi: v, ...extra } : { lo: v, hi: Infinity, ...extra });
    else push('number', v, at, { lo: r.lo, hi: r.hi, spelled: true, ...extra });
  });
  for (const [re, get] of PAPI_RES) scan(re, (m) => { const [noun, n] = get(m); push('count', n, m.index, { noun }); });
  scan(COUNT_RE, (m) => { push('count', countOf(m[1]), m.index, { noun: m[2] === 'flock' || m[2] === 'bird' ? 'birds' : m[2] }); });
  scan(new RegExp(`\\b(?:at\\s+)?your\\s+${CLOCK_WORD}\\s*o'clock`, 'g'), (m) => push('bearing_clock', clockNum(m[1]), m.index));
  scan(new RegExp(`\\b${CLOCK_WORD}\\s*o'clock`, 'g'), (m) => push('clock', clockNum(m[1]), m.index));
  scan(REGION_RE, (m) => push('region', REGION_ALIAS[m[1]] || m[1], m.index));
  scan(/\b(?:in|at) the (centre|center|middle|top|bottom)\b/g, (m) => push('region', REGION_ALIAS[m[1]] || m[1], m.index));
  scan(COMPASS_RE, (m) => push('compass', m[1].replace(/^(north|south)(east|west)$/, '$1-$2'), m.index));
  for (const [a, p] of ACTION_PHRASES) scan(phraseRe(p), (m) => push('action', a, m.index));
  scan(CONTINUE_RE, (m) => push('action', 'CONTINUE', m.index + m[0].lastIndexOf('continue')));
  scan(VERDICT_RE, (m) => push('verdict', m[1].toUpperCase(), m.index));
  // reason words count only in a sentence that states a verdict, a cause or the monitor ("a tailwind of 5 kt" is a wind
  // fact), and cause words only in a sentence about the threat
  const inSentence = (s, type, phrases) => {
    for (const [k, p] of phrases) {
      for (const m of [...lowerSame(w.slice(s.at, s.at + s.text.length)).matchAll(phraseRe(p))]) {
        push(type, k, s.at + m.index);
        blank(s.at + m.index, m[0].length);
      }
    }
  };
  for (const s of sents) {
    if (REASON_CTX.test(s.text)) inSentence(s, 'reason', REASON_PHRASES);
    if (CAUSE_CUE.test(s.text)) inSentence(s, 'cause', CAUSE_PHRASES);
  }
  for (const [o, p] of OUTCOME_PHRASES) scan(phraseRe(p), (m) => push('outcome', o, m.index));
  // a hazard kind: only the noun is used up, so the size or colour words before it stay claims of their own
  for (const m of [...lowerSame(w).matchAll(KIND_RE)]) {
    const at = m.index + m[0].lastIndexOf(m[1]), dets = [...m[0].matchAll(/\b(?:a|an|the|another|lone|that|this)\s/g)];
    push('kind', kindOf(m[1]), at, { np: m.index + dets[dets.length - 1].index });
    blank(at, m[0].length - (at - m.index));
  }
  // a relation's claim sits at its direction word, so "not" before that word negates it
  for (const [dim, re, get] of RELS) scan(re, (m) => push('category', get(m), m[1] ? m.index + m[0].lastIndexOf(m[1]) : m.index, { dim }));
  for (const [dim, value, re] of CATS) scan(re, (m) => push('category', value, m.index, { dim }));
  scan(UNGROUNDED, (m) => push('adjective', m[1], m.index));
  parsePresence(low0, sents, push);
  for (const cl of c) {
    const s = sents[cl.sent];
    if (NEGATABLE.has(cl.type) && negatedAt(low0, s.at, cl.at, cl.np ?? null) !== !!cl.noBefore) cl.negated = true;
    const bound = POSITION.has(cl.type) || (cl.type === 'category' && (cl.dim === 'side' || cl.dim === 'closing'));
    if (bound) cl.subject = subjectOf(low0, s, cl, c);
  }
  return c.sort((a, b) => a.at - b.at);
}
