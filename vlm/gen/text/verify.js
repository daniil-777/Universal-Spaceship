// vlm/gen/text/verify.js — the §5.4 verifier. One rounding table (nice steps 1, 2, 2.5, 5) for every number; slot formatters
// return the interval their wording stands for; the claim parser reads numbers+units, ranges ("between 95 and 285 m",
// "under 1 second"), number words with a counted noun, o'clock (image clock; "your N o'clock" = pilot bearing), 3x3 regions,
// compass words, verdict/action/reason words and gazetteer entities. Failures are rejected, never repaired.
import { REASON_TEXT, ACTION_TEXT, TTC_RANGE, CLR_RANGE_U, normContext } from './context.js';

const NICE = [1, 2, 2.5, 5];
export function niceStep(x) { if (!(x > 0)) return 1; const e = 10 ** Math.floor(Math.log10(x)); let b = e; for (const n of NICE) if (n * e <= x) b = n * e; return b; }
export function roundNice(v) { const st = niceStep(Math.abs(v) / 3), r = +(Math.round(v / st) * st).toPrecision(12); return { value: r, lo: +(r - st / 2).toPrecision(12), hi: +(r + st / 2).toPrecision(12) }; }
export const UNITS = Object.freeze({ m: ['len', 1], km: ['len', 1000], ft: ['len', 0.3048], mi: ['len', 1609.344], NM: ['len', 1852], u: ['len', 19], 'm/s': ['spd', 1], 'km/h': ['spd', 1 / 3.6],
  mph: ['spd', 0.44704], kt: ['spd', 0.514444], 'u/s': ['spd', 19], 'cm/s': ['spd', 0.01], fpm: ['spd', 0.00508], s: ['time', 1], deg: ['ang', 1], 'deg/s': ['angrate', 1], g: ['g', 1], dots: ['dots', 1],
  frac: ['frac', 1], '%': ['frac', 0.01] });
// unitless fractions in the label modules (0..1), read and written as per cent
const FRAC_IDS = new Set(['fuel_frac', 'geo.sea_frac', 'sun.lit', 'air.in_cloud', 'cfg.spoilers', 'thrust', 'image.white_frac', 'image.black_frac']);
export const unitOf = (id, f) => f.unit || (FRAC_IDS.has(id) ? 'frac' : null);
const DISPLAY = { metric: { len: (m) => (m < 1000 ? 'm' : 'km'), spd: () => 'm/s', frac: () => '%' }, imperial: { len: (m) => (m < 1609 ? 'ft' : 'mi'), spd: () => 'mph', frac: () => '%' },
  aviation: { len: (m) => (m < 1852 ? 'ft' : 'NM'), spd: () => 'kt', frac: () => '%' } };
const SHOW = { s: ['second', 'seconds'], deg: ['degree', 'degrees'], 'deg/s': ['degree per second', 'degrees per second'], dots: ['dot', 'dots'] };
const WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty'];
export const numText = (x) => { const s = Number.isInteger(x) && Math.abs(x) >= 1000 ? x.toLocaleString('en-US') : String(x); return /e/.test(s) ? x.toFixed(12).replace(/0+$/, '') : s; };
export const unitText = (unit, x) => (SHOW[unit] ? SHOW[unit][x === 1 ? 0 : 1] : unit);
export function fmtSlot(factId, fact, { system = 'metric', abs = false, unit = null } = {}) {
  const fu = unitOf(factId, fact), u = UNITS[fu], v = fact.v;
  if (fu === 'count') return { text: v <= 20 ? WORDS[v] : String(v), fact_id: factId, value: v, unit: 'count', lo: v, hi: v, type: 'count' };
  if (fu === 'clock') return { text: `${v} o'clock`, fact_id: factId, value: v, unit: 'clock', lo: v, hi: v, type: 'clock' };
  if (!u || typeof v !== 'number') return { text: String(v), fact_id: factId, value: v, unit: fu, lo: null, hi: null, type: 'category' };
  const x = abs ? Math.abs(v) : v, si = x * u[1], sys = DISPLAY[system] || DISPLAY.metric, dn = unit || (sys[u[0]] ? sys[u[0]](Math.abs(si)) : fu), k = UNITS[dn][1], r = roundNice(si / k);
  return { text: `about ${numText(r.value)} ${unitText(dn, r.value)}`, fact_id: factId, value: v, unit: fu, lo: r.lo * k / u[1], hi: r.hi * k / u[1], type: 'number', dim: u[0], shown: r.value, shownUnit: dn, ...(abs ? { derive: ['abs'] } : {}) };
}
// the other slot kinds; derive (below) maps the fact value to what the slot states
const opt = (derive) => (derive ? { derive } : {});
export const NOUNS = Object.freeze({ rock: ['rock', 'rocks'], comet: ['comet', 'comets'], satellite: ['satellite', 'satellites'], airliner: ['airliner', 'airliners'], birds: ['flock of birds', 'flocks of birds'],
  hazard: ['hazard', 'hazards'], jet: ['jet', 'jets'], white: ['white', 'white'], red: ['red', 'red'] });
export const countPhrase = (n, noun) => (n === 0 ? `no ${NOUNS[noun][1]}` : `${n <= 20 ? WORDS[n] : n} ${NOUNS[noun][n === 1 ? 0 : 1]}`);
export const catSlot = (id, value, text, derive) => ({ text, fact_id: id, value, unit: null, lo: null, hi: null, type: 'category', ...opt(derive) });
export const entSlot = (id, name, derive) => ({ text: name, fact_id: id, value: name, unit: null, lo: null, hi: null, type: 'entity', ...opt(derive) });
export const countSlot = (id, n, noun, derive) => ({ text: countPhrase(n, noun), fact_id: id, value: n, unit: 'count', lo: n, hi: n, type: 'count', noun, ...opt(derive) });
export const clockSlot = (id, v) => ({ text: `${v} o'clock`, fact_id: id, value: v, unit: 'clock', lo: v, hi: v, type: 'clock' });
export const regionSlot = (id, region, derive) => ({ text: `the ${region}`, fact_id: id, value: region, unit: null, lo: null, hi: null, type: 'region', ...opt(derive) });
export const compassSlot = (id, c, derive) => ({ text: c, fact_id: id, value: c, unit: null, lo: null, hi: null, type: 'compass', ...opt(derive) });
export const rangeText = (lo, hi, unit) => (lo <= 0 ? `under ${numText(hi)} ${unitText(unit, hi)}` : hi === Infinity ? `over ${numText(lo)} ${unitText(unit, lo)}` : `between ${numText(lo)} and ${numText(hi)} ${unitText(unit, hi)}`);
// a bin [lo, hi] stated in `unit`; the fact value (in its own unit) must lie inside it
export const rangeSlot = (id, fact, lo, hi, unit, derive) => ({ text: rangeText(lo, hi, unit), fact_id: id, value: fact.v, unit, lo, hi, type: 'range', shownLo: lo <= 0 ? 0 : lo, shownHi: hi, ...opt(derive) });
const DERIVE = { abs: (v) => Math.abs(v), has: (v, _, k) => v.includes(k), count_of: (v, _, k) => (v.includes(k) ? null : 0), len: (v) => v.length, papi_red: (v) => 4 - v,
  gt: (v, _, x) => v > x, lt: (v, _, x) => v < x, field: (v, _, k) => (v ? v[k] : null), idx: (v, _, i, k) => (v && v[i] ? v[i][k] : null), sign: (v) => Math.sign(v),
  all_true: (v) => Object.values(v).every(Boolean), false_keys: (v) => Object.keys(v).filter((k) => !v[k]).sort().join(','), sorted: (v) => [...v].sort().join(','),
  map: (v, _, m) => (Object.prototype.hasOwnProperty.call(m, v) ? m[v] : null), cmp_fact: (v, F, id) => (F[id] && typeof F[id].v === 'number' ? (v > F[id].v ? 'above' : 'below') : null),
  in_list: (v, _, list) => list.includes(v), relief: (v) => (v || []).find((t) => ['MOUNTAINS', 'HILLS', 'FLAT'].includes(t)) ?? null, in_band: (v, _, a, b) => v >= a && v <= b,
  cloud_code: (v) => (/\b(FEW|SCT|BKN|OVC|NSC|SKC|NONE)\b/.exec(String(v)) || [])[1] ?? null, count_type: (v, _, t) => v.filter((c) => c.type === t).length, min_dist_type: (v, _, t) => { const d = v.filter((c) => c.type === t).map((c) => c.dist_u); return d.length ? Math.min(...d) : null; } };
export const derive = (d, v, facts) => (!d ? v : v === null || v === undefined ? null : DERIVE[d[0]](v, facts, ...d.slice(1)));

// facts plus the pseudo-facts the text layer cites: the zoom block (visual) and the safety block (context-class)
export function factsOf(rec) {
  const F = { ...(rec.facts || {}) };
  if (rec.zoom) { F['zoom.tags'] = { v: rec.zoom.tags ?? null, unit: null, obs: 'visual' }; F['zoom.range_bin'] = { v: rec.zoom.range_bin ?? null, unit: null, obs: 'visual' }; }
  if (rec.safety) for (const k of ['verdict', 'severity', 'reasons', 'best_action', 'safe_actions', 'action_outcome', 'cause']) F[`safety.${k}`] = { v: rec.safety[k] ?? null, unit: null, obs: 'context' };
  return F;
}
export const ROUTE_NAMES = Object.freeze({ alps: 'Alps', china: 'China', newyork: 'New York', london: 'London', moscow: 'Moscow', dubai: 'Dubai', mega: 'Megacity' });
// the fixed part of the §5.4 gazetteer; the build adds every Natural Earth name it loads
// terms (acronyms, docking words, bodies, the airport) are known words, never place claims; route names are places
export const TERMS = Object.freeze(['Astro Pilot Spaceport', 'APX', 'IDSS', 'KOS', 'V-bar', 'Earth', 'Moon', 'Sun', 'PAPI', 'ILS', 'LOC', 'GS', 'IAS', 'METAR', 'AGL', 'UTC', 'TTC']);
export const BASE_NAMES = Object.freeze([...Object.values(ROUTE_NAMES), ...TERMS]);
const norm = (s) => String(s).normalize('NFC').toLowerCase().replace(/[^\p{L}\p{N} ]+/gu, '').replace(/\s+/g, ' ').trim();
const TERM_KEYS = new Set(TERMS.map(norm));
export function makeGazetteer(names) {
  const m = new Map(); let maxWords = 1;
  for (const n of names) { if (!n) continue; const k = norm(n); if (!k) continue; if (!m.has(k)) m.set(k, String(n)); maxWords = Math.max(maxWords, Math.min(8, k.split(' ').length)); }
  return { has: (s) => m.has(norm(s)), canonical: (s) => m.get(norm(s)), maxWords, names: [...m.values()] };
}
// words that never make an unknown entity mid-sentence, and sentence-initial words that are never read as a place
const STOP = new Set(['I', 'Context', 'Earth', 'Moon', 'Sun', 'V-bar']);
// sentence-initial words that are never read as a place even when a gazetteer holds them (Natural Earth has Split, Point,
// Orange, Wind...): every word that opens a sentence in the bank or in a slot value (tests/vlm_text.test.mjs checks this)
export const SENTENCE_WORDS = new Set(('a about above according across actions advice aerial air airspeed all along also am an and any are around as at because before behind ' +
  'below beside best between beyond both but by can chase chase-camera check clear clearance closest closing cloud cloudy colour-wise compare components conditions confirm contact correct ' +
  'could count counting crosswind current day daylight darkness describe descending direct distance do docking does down driven during dusk each estimate every everything expect explain ' +
  'failed failures fair falling flight flying fog following for from fuel g-load gates gear give given ground has have haze hazard hazards head headroom headwind here high highly hilly ' +
  'holding how however i identify if imagery in inside is it its judging landing large laterally left level light look looking low lower margin measured medium medium-sized minimum ' +
  'moderate monitor most mostly mountain mountainous name near nearest neither no none nose-down nose-up not note nothing of off on one open or out outside over overall overloading papi ' +
  'per phase pitch point problem rain range rate read reason receding recommended remaining resolution right rolling roughly route running sea seen severe shadow should skies slightly ' +
  'small smooth so space speed speed-wise split spoilers stabilized stall still stopped stormy sunlight tailwind taking tell that the there they this time tiny to tonally toward ' +
  'turbulence under unless up upper using vertically visibility visible was water we well what when where which whose why wind with within without yes yet you city climbing deployed ' +
  'textured transit overhead scattered broken part terrain zero two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty ' +
  'rock comet satellite airliner flock black white grey gray blue green brown tan red orange yellow purple pink beige cyan navy').split(' '));
const COMMON = SENTENCE_WORDS;
const isAcronym = (w) => /^[\p{Lu}\d_/-]+$/u.test(w) && /\p{Lu}/u.test(w);
const cleanWord = (w) => w.replace(/^[^\p{L}\p{N}]+/u, '').replace(/[’']s$/u, '').replace(/[^\p{L}\p{N}'’-]+$/u, '').replace(/[’']s$/u, '');
const UNIT_WORDS = [['degrees per second', 'deg/s'], ['degree per second', 'deg/s'], ['deg/s', 'deg/s'], ['km/h', 'km/h'], ['m/s', 'm/s'], ['cm/s', 'cm/s'], ['u/s', 'u/s'], ['feet per minute', 'fpm'], ['fpm', 'fpm'],
  ['mph', 'mph'], ['knots', 'kt'], ['knot', 'kt'], ['kt', 'kt'], ['nautical miles', 'NM'], ['nautical mile', 'NM'], ['nm', 'NM'], ['kilometres', 'km'], ['kilometers', 'km'], ['kilometre', 'km'], ['kilometer', 'km'], ['km', 'km'],
  ['miles', 'mi'], ['mile', 'mi'], ['mi', 'mi'], ['feet', 'ft'], ['foot', 'ft'], ['ft', 'ft'], ['metres', 'm'], ['meters', 'm'], ['metre', 'm'], ['meter', 'm'], ['m', 'm'], ['seconds', 's'], ['second', 's'], ['sec', 's'],
  ['s', 's'], ['degrees', 'deg'], ['degree', 'deg'], ['deg', 'deg'], ['°', 'deg'], ['g', 'g'], ['dots', 'dots'], ['dot', 'dots'], ['per cent', '%'], ['percent', '%'], ['%', '%'], ['u', 'u']];
const UNIT_OF = Object.fromEntries(UNIT_WORDS), esc = (s) => s.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&');
const UALT = UNIT_WORDS.map(([w]) => esc(w)).join('|'), UEND = '(?![\\p{L}\\d])', UN = `(${UALT})${UEND}`, UNOPT = `(?:(?:${UALT})${UEND})?`;
const NUM = '(\\d[\\d,]*(?:\\.\\d+)?)', toNum = (s) => +s.replace(/,/g, '');
const RANGE_RES = [[new RegExp(`\\bbetween\\s+(?:about\\s+)?${NUM}\\s*${UNOPT}\\s+and\\s+${NUM}\\s*${UN}`, 'giu'), (m) => [toNum(m[1]), toNum(m[2]), m[3]]],
  [new RegExp(`${NUM}\\s*(?:-|–|to)\\s*${NUM}\\s*${UN}`, 'giu'), (m) => [toNum(m[1]), toNum(m[2]), m[3]]],
  [new RegExp(`(?:\\b(?:under|below|less than|within|closer than|up to)\\s+(?:about\\s+)?|<\\s*)${NUM}\\s*${UN}`, 'giu'), (m) => [0, toNum(m[1]), m[2]]],
  [new RegExp(`(?:\\b(?:over|above|more than|beyond|farther than|further than)\\s+(?:about\\s+)?|>\\s*)${NUM}\\s*${UN}`, 'giu'), (m) => [toNum(m[1]), Infinity, m[2]]]];
const NUMBER_RE = new RegExp(`${NUM}\\s*${UN}`, 'giu');
const COUNT_WORDS = { no: 0, none: 0, zero: 0, 'a single': 1, single: 1, 'a pair of': 2, 'pair of': 2 };
const COUNT_RE = new RegExp(`\\b(no|none|zero|a single|single|a pair of|pair of|${WORDS.slice(1).join('|')}|\\d+)\\s+(rock|comet|satellite|airliner|flock|bird|hazard|jet|white|red)(?:e?s)?\\b`, 'g');
const CLOCK_WORD = `(1[0-2]|[1-9]|${WORDS.slice(1, 13).join('|')})`, clockNum = (s) => (/\d/.test(s) ? +s : WORDS.indexOf(s));
const REGION_ALIAS = { 'upper center': 'upper centre', 'lower center': 'lower centre', 'top left': 'upper left', 'top right': 'upper right', 'bottom left': 'lower left', 'bottom right': 'lower right',
  'top centre': 'upper centre', 'bottom centre': 'lower centre', top: 'upper centre', bottom: 'lower centre', center: 'centre', middle: 'centre' };
const COMPASS = ['north', 'north-east', 'east', 'south-east', 'south', 'south-west', 'west', 'north-west'];
const REASON_CTX = /\b(unsafe|safe|caution|monitor|because|due to|reasons?|cites?|cited|citing|flags?|flagged|owing to|caused by|driven by)\b/i;
const REASON_PHRASES = Object.entries(REASON_TEXT).map(([r, w]) => [r, w.replace(/^(a|an|the) /, '')]).sort((a, b) => b[1].length - a[1].length);
const ACTION_PHRASES = Object.entries(ACTION_TEXT).filter(([a]) => a !== 'CONTINUE').sort((a, b) => b[1].length - a[1].length);

// lower-casing that keeps every index (a character whose lower case changes length is left as it is)
const lowerSame = (s) => s.replace(/./gsu, (ch) => { const l = ch.toLowerCase(); return l.length === ch.length ? l : ch; });
const SENT_END = /[.!?:;]["')\]]*$/;
export function parseClaims(text, gaz) {
  const src = String(text).replace(/[’‘]/g, "'"), c = [], push = (type, value, at, extra = {}) => c.push({ type, value, at, ...extra });
  let w = src;
  const blank = (at, len) => { w = w.slice(0, at) + ' '.repeat(len) + w.slice(at + len); };
  const scan = (re, fn, lower = true) => { for (const m of [...(lower ? lowerSame(w) : w).matchAll(re)]) { fn(m); blank(m.index, m[0].length); } };
  // entities first, so a place name's compass or number words are not read twice
  const words = [...src.matchAll(/\S+/g)], start = (i) => i === 0 || SENT_END.test(words[i - 1][0]);
  for (let i = 0; i < words.length; i++) {
    const clean = cleanWord(words[i][0]); if (!/^\p{Lu}/u.test(clean)) continue;
    let hit = null;
    for (let n = Math.min(gaz.maxWords, words.length - i); n >= 1 && !hit; n--) {
      if (words.slice(i, i + n - 1).some((x) => SENT_END.test(x[0]))) continue;
      const span = words.slice(i, i + n).map((x) => cleanWord(x[0])).join(' ');
      if (gaz.has(span) && !(n === 1 && (STOP.has(clean) || (start(i) && COMMON.has(clean.toLowerCase()))))) hit = [span, n];
    }
    if (hit) { const a = words[i].index, e = words[i + hit[1] - 1]; if (!TERM_KEYS.has(norm(hit[0]))) push('entity', gaz.canonical(hit[0]), a); blank(a, e.index + e[0].length - a); i += hit[1] - 1; }
    else if (!start(i) && !STOP.has(clean) && !isAcronym(clean)) push('unknown_entity', clean, words[i].index);
  }
  for (const [re, get] of RANGE_RES) scan(re, (m) => { const [lo, hi, uw] = get(m), unit = UNIT_OF[uw.toLowerCase()]; push('range', null, m.index, { lo, hi, unit, dim: UNITS[unit][0] }); }, false);
  scan(NUMBER_RE, (m) => { const v = toNum(m[1]), unit = UNIT_OF[m[2].toLowerCase()], r = roundNice(v); push('number', v, m.index, { unit, dim: UNITS[unit][0], lo: r.lo, hi: r.hi }); }, false);
  scan(COUNT_RE, (m) => { const k = m[1]; push('count', COUNT_WORDS[k] ?? (WORDS.indexOf(k) >= 0 ? WORDS.indexOf(k) : +k), m.index, { noun: m[2] === 'flock' || m[2] === 'bird' ? 'birds' : m[2] }); });
  scan(new RegExp(`\\b(?:at\\s+)?your\\s+${CLOCK_WORD}\\s*o'clock`, 'g'), (m) => push('bearing_clock', clockNum(m[1]), m.index));
  scan(new RegExp(`\\b${CLOCK_WORD}\\s*o'clock`, 'g'), (m) => push('clock', clockNum(m[1]), m.index));
  scan(/\b(upper left|upper centre|upper center|upper right|middle left|middle right|lower left|lower centre|lower center|lower right|top left|top right|top centre|bottom left|bottom right|bottom centre)\b/g, (m) => push('region', REGION_ALIAS[m[1]] || m[1], m.index));
  scan(/\b(?:in|at) the (centre|center|middle|top|bottom)\b/g, (m) => push('region', REGION_ALIAS[m[1]] || m[1], m.index));
  scan(/\b(north-east|north-west|south-east|south-west|northeast|northwest|southeast|southwest|north|south|east|west)\b/g, (m) => push('compass', m[1].replace(/^(north|south)(east|west)$/, '$1-$2'), m.index));
  for (const [a, p] of ACTION_PHRASES) scan(new RegExp(`\\b${esc(p)}\\b`, 'g'), (m) => push('action', a, m.index));
  scan(/(?<![\p{L}])(?<!no )(unsafe|safe|caution)(?![\p{L}])(?! (?:action|breakout|option|distance|speed|margin|limit))/gu, (m) => push('verdict', m[1].toUpperCase(), m.index));
  // reason words count only in a sentence that states a verdict, a cause or the monitor ("a tailwind of 5 kt" is a wind fact)
  const low = lowerSame(w);
  for (const s of src.matchAll(/(?:[^.!?;]|\.(?=\d))+[.!?;]?/g)) {
    if (!REASON_CTX.test(s[0])) continue;
    const part = low.slice(s.index, s.index + s[0].length);
    for (const [r, p] of REASON_PHRASES) for (const m of part.matchAll(new RegExp(`\\b${esc(p)}\\b`, 'g'))) push('reason', r, s.index + m.index);
  }
  return c.sort((a, b) => a.at - b.at);
}

// numbers the facts hold: plain numbers with a unit, and km/dist fields of object facts (place.nearest.km, weather.cells)
const SUB_UNIT = { km: 'km', dist_u: 'u', dist_m: 'm' };
function numericEntries(F) {
  const out = [], sub = (id, o) => { for (const [k, u] of Object.entries(SUB_UNIT)) if (o && typeof o[k] === 'number') out.push([id, o[k], u]); };
  for (const [id, f] of Object.entries(F)) {
    const u = unitOf(id, f), v = f.v;
    if (typeof v === 'number' && UNITS[u]) out.push([id, v, u]);
    else if (Array.isArray(v)) v.forEach((x) => (x && typeof x === 'object' && !Array.isArray(x) ? sub(id, x) : null));
    else if (v && typeof v === 'object') sub(id, v);
  }
  return out;
}
// bin facts stored as text ('20-100 m', '<2 m', '>400 m'): the interval they name, in SI
const BIN_RE = /^(<|>)?(\d+(?:\.\d+)?)(?:-(\d+(?:\.\d+)?))? (m|km|s|u)$/;
function binEntries(F) {
  const out = [];
  for (const [id, f] of Object.entries(F)) {
    const m = typeof f.v === 'string' ? BIN_RE.exec(f.v) : null; if (!m) continue;
    const k = UNITS[m[4]][1], a = +m[2], b = m[3] === undefined ? null : +m[3];
    out.push([id, UNITS[m[4]][0], m[1] === '<' ? [0, a * k] : m[1] === '>' ? [a * k, Infinity] : [a * k, b * k]]);
  }
  return out;
}
function factNames(id, f) {
  const s = new Set(), add = (x) => { if (typeof x === 'string') s.add(norm(x)); }, v = f.v;
  if (Array.isArray(v)) v.forEach((x) => add(x && typeof x === 'object' ? x.name : x)); else if (v && typeof v === 'object') add(v.name); else add(v);
  if ((id === 'route' || id === 'world') && ROUTE_NAMES[v]) add(ROUTE_NAMES[v]);
  return s;
}
const inIv = (x, lo, hi) => x >= lo - 1e-9 * Math.max(1, Math.abs(lo)) && x <= hi + 1e-9 * Math.max(1, Math.abs(hi));
const regionOfBox = (b) => `${['upper', 'middle', 'lower'][Math.min(2, Math.floor(3 * ((b[1] + b[3]) / 2) / 504))]} ${['left', 'centre', 'right'][Math.min(2, Math.floor(3 * ((b[0] + b[2]) / 2) / 896))]}`.replace('middle centre', 'centre');
const sameBin = (lo, hi, [a, b]) => Math.abs(lo - a) <= 0.01 * Math.max(1, a) && (hi === Infinity ? b === Infinity : b !== Infinity && Math.abs(hi - b) <= 0.01 * Math.max(1, b));
// the fact ids (or 'monitor' / 'safety') that support a claim; [] when nothing does
function support(cl, F, rec, ctx, attributed) {
  const S = rec.safety, v = cl.value, ids = (pred) => Object.entries(F).filter(([id, f]) => pred(id, f)).map(([id]) => id);
  if (cl.type === 'number' || cl.type === 'range') {
    const k = UNITS[cl.unit][1], lo = cl.lo * k, hi = cl.hi * k;
    const out = numericEntries(F).filter(([, x, u]) => UNITS[u][0] === cl.dim && inIv(Math.abs(x * UNITS[u][1]), lo, hi)).map(([id]) => id);
    if (cl.type === 'number') return out;
    for (const [id, dim, iv] of binEntries(F)) if (dim === cl.dim && sameBin(lo, hi, iv)) out.push(id);
    if (attributed && ctx && ctx.monitor) {
      const t = TTC_RANGE[ctx.monitor.ttc_bin], r = CLR_RANGE_U[ctx.monitor.clr_bin];
      if ((cl.dim === 'time' && t && sameBin(lo, hi, t)) || (cl.dim === 'len' && r && sameBin(lo, hi, [r[0] * 19, r[1] * 19]))) out.push('monitor');
    }
    return out;
  }
  if (cl.type === 'count') {
    const n = F['hazards.count_in_frame'], kinds = F.kinds_in_frame, papi = F.papi_whites_cam;
    if (cl.noun === 'hazard') return n && n.v === v ? ['hazards.count_in_frame'] : [];
    if (cl.noun === 'white') return papi && papi.v === v ? ['papi_whites_cam'] : [];
    if (cl.noun === 'red') return papi && typeof papi.v === 'number' && 4 - papi.v === v ? ['papi_whites_cam'] : [];
    if (cl.noun === 'jet') return ids((id, f) => (id === 'jets_failed' || id === 'jets_firing') && Array.isArray(f.v) && f.v.length === v);
    if (!kinds || !Array.isArray(kinds.v)) return [];
    if (v === 0) return kinds.v.includes(cl.noun) ? [] : ['kinds_in_frame'];
    return kinds.v.length === 1 && kinds.v[0] === cl.noun && n && n.v === v ? ['kinds_in_frame', 'hazards.count_in_frame'] : [];
  }
  if (cl.type === 'clock') return ids((id, f) => id.endsWith('.clock') && f.v === v);
  if (cl.type === 'bearing_clock') return ids((id, f) => id.endsWith('.bearing_clock') && f.v === v);
  if (cl.type === 'region') return ids((id, f) => JSON.stringify(f.v ?? null).includes(`"region":"${v}"`) || (id.endsWith('.box_px') && Array.isArray(f.v) && regionOfBox(f.v) === v));
  if (cl.type === 'compass') return ids((id, f) => f.v && typeof f.v === 'object' && Number.isFinite(f.v.bearing) && Math.abs(((f.v.bearing - COMPASS.indexOf(v) * 45 + 540) % 360) - 180) <= 22.5);
  if (cl.type === 'entity') return ids((id, f) => !id.startsWith('safety.') && factNames(id, f).has(norm(v)));
  const mon = attributed && ctx && ctx.monitor ? { verdict: ctx.monitor.verdict, reasons: ctx.monitor.reasons, best_action: ctx.monitor.action } : null, s = mon || S, tag = mon ? 'monitor' : 'safety';
  if (!s) return [];
  if (cl.type === 'verdict') return s.verdict === v ? [tag] : [];
  if (cl.type === 'action') return s.best_action === v || (S && S.safe_actions && S.safe_actions.includes(v)) ? [tag] : [];
  if (cl.type === 'reason') return s.reasons.includes(v) ? [tag] : [];
  return [];
}
const MONITOR_TYPES = ['verdict', 'action', 'reason'];
// one parsed claim against the record (and the row's Context): true when some fact, the safety block or the monitor holds it
export const checkClaim = (cl, rec, { context = null, attributed = false, factIds = null } = {}) => {
  const all = factsOf(rec), F = factIds ? Object.fromEntries(Object.entries(all).filter(([id]) => factIds.includes(id))) : all;
  return support(cl, F, rec, normContext(context), attributed).length > 0;
};
function judge(claims, rec, F, ctx, { obsRule, attributed }) {
  const errors = [];
  for (const cl of claims) {
    cl.monitor = attributed;
    if (cl.type === 'unknown_entity') { errors.push(`unknown entity ${cl.value}`); continue; }
    let sup = support(cl, F, rec, ctx, attributed);
    if (obsRule) sup = sup.filter((id) => (id === 'monitor' || id === 'safety' ? !!(ctx && ctx.monitor) : (F[id] && F[id].obs === 'visual') || (!!ctx && (ctx.ids.has(id) || (!!ctx.monitor && id.startsWith('safety.'))))));
    if (!sup.length) errors.push(`${cl.type} ${JSON.stringify(cl.type === 'range' ? [cl.lo, cl.hi, cl.unit] : cl.type === 'number' ? [cl.value, cl.unit] : cl.value)} not supported`);
  }
  return errors;
}
// obsRule (Narrator rows, §5.7): a claim must rest on a visual fact, on a fact the row's Context supplies, or on its monitor
export function verifyFreeText(text, rec, { gaz, context = null, factIds = null, obsRule = false } = {}) {
  const ctx = normContext(context), all = factsOf(rec), F = factIds ? Object.fromEntries(Object.entries(all).filter(([id]) => factIds.includes(id))) : all;
  const claims = parseClaims(text, gaz), errors = judge(claims, rec, F, ctx, { obsRule, attributed: /monitor/i.test(text) });
  if (rec.family !== 'Z' && !ctx && claims.some((cl) => MONITOR_TYPES.includes(cl.type))) errors.push('a flight row without Context states a context-class fact');
  return { verified: errors.length === 0, errors, claims };
}
const deq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
function slotError(s, F) {
  if (MONITOR_TYPES.includes(s.type) || s.type === 'monitor_range') return null;
  const f = F[s.fact_id];
  if (!f || f.v === null || f.v === undefined) return `slot ${s.fact_id} has no fact`;
  const x = derive(s.derive, f.v, F);
  if (s.type === 'number') return typeof x === 'number' && inIv(x, s.lo, s.hi) ? null : `slot ${s.fact_id}: ${x} outside [${s.lo}, ${s.hi}]`;
  if (s.type === 'range') {
    if (typeof x === 'string') return x === s.value ? null : `slot ${s.fact_id}: ${x} is not ${s.value}`;
    const fu = UNITS[unitOf(s.fact_id, f)], y = typeof x === 'number' && fu ? Math.abs(x) * fu[1] / UNITS[s.unit][1] : NaN;
    return inIv(y, s.lo, s.hi) ? null : `slot ${s.fact_id}: ${x} outside ${s.text}`;
  }
  if (s.type === 'entity') return typeof x === 'string' && norm(x) === norm(s.value) ? null : `slot ${s.fact_id}: ${JSON.stringify(x)} is not ${s.value}`;
  return deq(x, s.value) ? null : `slot ${s.fact_id}: ${JSON.stringify(x)} is not ${JSON.stringify(s.value)}`;
}
const keyOfClaim = (c) => `${c.type}:${c.type === 'range' ? `${c.lo}-${c.hi === Infinity ? 'inf' : c.hi}` : c.value}`;
function slotKeys(s, gaz) {
  if (s.type === 'number') return [`number:${s.shown ?? +s.text.replace(/[^\d.]/g, '')}`];
  if (s.type === 'range' || s.type === 'monitor_range') return [`range:${s.shownLo}-${s.shownHi === Infinity ? 'inf' : s.shownHi}`];
  if (['count', 'clock', 'bearing_clock', 'region', 'compass'].includes(s.type)) return [`${s.type}:${s.value}`];
  if (s.type === 'entity') return [`entity:${gaz.canonical(String(s.value)) ?? s.value}`];
  if (s.type === 'category' && typeof s.value === 'string' && gaz.has(s.value) && /^\p{Lu}/u.test(String(s.text))) return [`entity:${gaz.canonical(s.value)}`];
  return [];
}
// template text: (a) every slot (and hidden check) holds for the facts; (b) the parser recovers exactly the slot set; plus
// the free-text check of the answer and of the prompt (verdict/action/reason words in a question are not claims; a
// negative's false premise, matched by `premise(claim)`, is exempt)
export function verifyTemplateItem(item, rec, { gaz, context = null, premise = null } = {}) {
  const F = factsOf(rec), errors = [];
  for (const s of [...(item.slots || []), ...(item.checks || [])]) { const e = slotError(s, F); if (e) errors.push(e); }
  const got = parseClaims(item.answer, gaz).filter((c) => !MONITOR_TYPES.includes(c.type)).map(keyOfClaim).sort();
  const want = (item.slots || []).flatMap((s) => slotKeys(s, gaz)).sort();
  const parserOk = JSON.stringify(got) === JSON.stringify(want);
  if (!parserOk) errors.push(`parser recovered ${got} but the slots are ${want}`);
  for (const e of verifyFreeText(item.answer, rec, { gaz, context }).errors) if (!errors.includes(e)) errors.push(e);
  if (item.prompt) {
    const claims = parseClaims(item.prompt, gaz).filter((c) => !MONITOR_TYPES.includes(c.type) && !(premise && premise(c)));
    for (const e of judge(claims, rec, F, normContext(context), { obsRule: false, attributed: /monitor/i.test(item.prompt) })) errors.push(`prompt: ${e}`);
  }
  return { verified: errors.length === 0, parserOk, errors };
}
