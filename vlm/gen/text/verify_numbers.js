// vlm/gen/text/verify_numbers.js — numbers for the §5.4 verifier: the one rounding table (nice steps 1, 2, 2.5, 5), units,
// slot formatting, the number and range patterns (digits and spelled-out words), the facts' numbers, and the binding of a
// stated number to the quantity noun of its clause.

const NICE = [1, 2, 2.5, 5];
export function niceStep(x) {
  if (!(x > 0)) return 1;
  const e = 10 ** Math.floor(Math.log10(x));
  let b = e;
  for (const n of NICE) if (n * e <= x) b = n * e;
  return b;
}
const p12 = (x) => +x.toPrecision(12);
export function roundNice(v) {
  // symmetric about zero: -2.5 steps rounds to -3 like 2.5 rounds to 3
  const st = niceStep(Math.abs(v) / 3), r = p12(Math.sign(v) * Math.round(Math.abs(v) / st) * st) || 0;
  return { value: r, lo: p12(r - st / 2), hi: p12(r + st / 2) };
}
// does a fact value x support a stated "about X"? Yes when the renderer rounds x to X, or when x lies within half of the
// largest nice step that divides X (a human rounding: "about 7 degrees" for 6.6, "about 140 kt" for 142.3)
const divides = (s, x) => Math.abs(x / s - Math.round(x / s)) < 1e-9;
export function humanStep(x) {
  const ax = Math.abs(x), top = niceStep(ax / 3) * 1.000001;
  let e = 10 ** Math.floor(Math.log10(top || 1)) * 10;
  for (let k = 0; k < 40; k++, e /= 10) for (const n of [5, 2.5, 2, 1]) if (n * e <= top && divides(n * e, ax)) return n * e;
  return niceStep(ax / 3);
}
export const aboutHolds = (x, X) => {
  const ax = Math.abs(x);
  return Math.abs(roundNice(ax).value - X) <= 1e-9 * Math.max(1, X) || Math.abs(ax - X) <= humanStep(X) / 2 + 1e-9 * Math.max(1, X);
};

export const UNITS = Object.freeze({ m: ['len', 1], km: ['len', 1000], ft: ['len', 0.3048], mi: ['len', 1609.344], NM: ['len', 1852],
  u: ['len', 19], 'm/s': ['spd', 1], 'km/h': ['spd', 1 / 3.6], mph: ['spd', 0.44704], kt: ['spd', 0.514444], 'u/s': ['spd', 19],
  'cm/s': ['spd', 0.01], fpm: ['spd', 0.00508], s: ['time', 1], deg: ['ang', 1], 'deg/s': ['angrate', 1], g: ['g', 1], dots: ['dots', 1],
  frac: ['frac', 1], '%': ['frac', 0.01] });
// unitless fractions in the label modules (0..1), read and written as per cent
const FRAC_IDS = new Set(['fuel_frac', 'geo.sea_frac', 'sun.lit', 'air.in_cloud', 'cfg.spoilers', 'thrust', 'image.white_frac',
  'image.black_frac']);
export const unitOf = (id, f) => f.unit || (FRAC_IDS.has(id) ? 'frac' : null);
const DISPLAY = {
  metric: { len: (m) => (m < 1000 ? 'm' : 'km'), spd: () => 'm/s', frac: () => '%' },
  imperial: { len: (m) => (m < 1609 ? 'ft' : 'mi'), spd: () => 'mph', frac: () => '%' },
  aviation: { len: (m) => (m < 1852 ? 'ft' : 'NM'), spd: () => 'kt', frac: () => '%' } };
const SHOW = { s: ['second', 'seconds'], deg: ['degree', 'degrees'], 'deg/s': ['degree per second', 'degrees per second'],
  dots: ['dot', 'dots'] };
export const WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen',
  'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty'];
export const numText = (x) => {
  const s = Number.isInteger(x) && Math.abs(x) >= 1000 ? x.toLocaleString('en-US') : String(x);
  return /e/.test(s) ? x.toFixed(12).replace(/0+$/, '') : s;
};
export const unitText = (unit, x) => (SHOW[unit] ? SHOW[unit][x === 1 ? 0 : 1] : unit);
export function fmtSlot(factId, fact, { system = 'metric', abs = false, unit = null } = {}) {
  const fu = unitOf(factId, fact), u = UNITS[fu], v = fact.v;
  if (fu === 'count') return { text: v <= 20 ? WORDS[v] : String(v), fact_id: factId, value: v, unit: 'count', lo: v, hi: v,
    type: 'count' };
  if (fu === 'clock') return { text: `${v} o'clock`, fact_id: factId, value: v, unit: 'clock', lo: v, hi: v, type: 'clock' };
  if (!u || typeof v !== 'number') return { text: String(v), fact_id: factId, value: v, unit: fu, lo: null, hi: null, type: 'category' };
  const x = abs ? Math.abs(v) : v, si = x * u[1], sys = DISPLAY[system] || DISPLAY.metric;
  const dn = unit || (sys[u[0]] ? sys[u[0]](Math.abs(si)) : fu), k = UNITS[dn][1], r = roundNice(si / k);
  return { text: `about ${numText(r.value)} ${unitText(dn, r.value)}`, fact_id: factId, value: v, unit: fu, lo: (r.lo * k) / u[1],
    hi: (r.hi * k) / u[1], type: 'number', dim: u[0], shown: r.value, shownUnit: dn, ...(abs ? { derive: ['abs'] } : {}) };
}
export const rangeText = (lo, hi, unit) => (lo <= 0 ? `under ${numText(hi)} ${unitText(unit, hi)}`
  : hi === Infinity ? `over ${numText(lo)} ${unitText(unit, lo)}` : `between ${numText(lo)} and ${numText(hi)} ${unitText(unit, hi)}`);

// ---- patterns ----
const UNIT_WORDS = [['degrees per second', 'deg/s'], ['degree per second', 'deg/s'], ['deg/s', 'deg/s'], ['kilometres per hour', 'km/h'],
  ['km/h', 'km/h'], ['metres per second', 'm/s'], ['meters per second', 'm/s'], ['m/s', 'm/s'], ['centimetres per second', 'cm/s'],
  ['centimeters per second', 'cm/s'], ['cm/s', 'cm/s'], ['u/s', 'u/s'], ['feet per minute', 'fpm'], ['fpm', 'fpm'], ['mph', 'mph'],
  ['knots', 'kt'], ['knot', 'kt'], ['kt', 'kt'], ['nautical miles', 'NM'], ['nautical mile', 'NM'], ['nm', 'NM'], ['kilometres', 'km'],
  ['kilometers', 'km'], ['kilometre', 'km'], ['kilometer', 'km'], ['km', 'km'], ['miles', 'mi'], ['mile', 'mi'], ['mi', 'mi'],
  ['feet', 'ft'], ['foot', 'ft'], ['ft', 'ft'], ['metres', 'm'], ['meters', 'm'], ['metre', 'm'], ['meter', 'm'], ['m', 'm'],
  ['seconds', 's'], ['second', 's'], ['sec', 's'], ['s', 's'], ['degrees', 'deg'], ['degree', 'deg'], ['deg', 'deg'], ['°', 'deg'],
  ['g', 'g'], ['dots', 'dots'], ['dot', 'dots'], ['per cent', '%'], ['percent', '%'], ['%', '%'], ['u', 'u']];
export const UNIT_OF = Object.fromEntries(UNIT_WORDS);
export const esc = (s) => s.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&');
// the same for patterns with the u flag, where "\-" outside a class is a syntax error
export const escU = (s) => s.replace(/[/\\^$*+?.()|[\]{}]/g, '\\$&');
const UALT = UNIT_WORDS.map(([w]) => esc(w)).join('|'), UEND = '(?![\\p{L}\\d])';
const UN = `(${UALT})${UEND}`, UNOPT = `(?:(?:${UALT})${UEND})?`, NUM = '(\\d[\\d,]*(?:\\.\\d+)?)';
export const toNum = (s) => +s.replace(/,/g, '');
const BELOW = 'under|below|less than|within|closer than|up to', ABOVE = 'over|above|more than|beyond|farther than|further than|outside';
export const RANGE_RES = [
  [new RegExp(`\\bbetween\\s+(?:about\\s+)?${NUM}\\s*${UNOPT}\\s+and\\s+${NUM}\\s*${UN}`, 'giu'), (m) => [toNum(m[1]), toNum(m[2]), m[3]]],
  [new RegExp(`${NUM}\\s*(?:-|–|to)\\s*${NUM}\\s*${UN}`, 'giu'), (m) => [toNum(m[1]), toNum(m[2]), m[3]]],
  [new RegExp(`(?:\\b(?:${BELOW})\\s+(?:about\\s+)?|<\\s*)${NUM}\\s*${UN}`, 'giu'), (m) => [0, toNum(m[1]), m[2]]],
  [new RegExp(`(?:\\b(?:${ABOVE})\\s+(?:about\\s+)?|>\\s*)${NUM}\\s*${UN}`, 'giu'), (m) => [toNum(m[1]), Infinity, m[2]]]];
export const NUMBER_RE = new RegExp(`${NUM}\\s*${UN}`, 'giu');
// spelled-out numbers ("about seven hundred and fifty metres", "twenty-one knots"), with an optional range cue before them
const ONES = WORDS.slice(0, 20), TENS = ['twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'];
const SW = `(?:${[...ONES, ...TENS, 'hundred', 'thousand'].join('|')})`;
export const SPELLED_RE = new RegExp(`\\b(?:(${BELOW}|${ABOVE})\\s+(?:about\\s+)?)?(${SW}(?:(?:\\s+and)?[\\s-]+${SW})*)\\s+${UN}`, 'giu');
export function spelledValue(s) {
  let total = 0, cur = 0;
  for (const t of s.toLowerCase().split(/[\s-]+/)) {
    if (ONES.includes(t)) cur += ONES.indexOf(t);
    else if (TENS.includes(t)) cur += 20 + 10 * TENS.indexOf(t);
    else if (t === 'hundred') cur = (cur || 1) * 100;
    else if (t === 'thousand') { total += (cur || 1) * 1000; cur = 0; }
  }
  return total + cur;
}
export const isBelowCue = (w) => new RegExp(`^(?:${BELOW})$`, 'i').test(w);

// ---- the numbers the facts hold: plain numbers with a unit, and km/dist fields of object facts ----
const SUB_UNIT = { km: 'km', dist_u: 'u', dist_m: 'm' };
export function numericEntries(F) {
  const out = [];
  const sub = (id, o) => { for (const [k, u] of Object.entries(SUB_UNIT)) if (o && typeof o[k] === 'number') out.push([id, o[k], u]); };
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
export function binEntries(F) {
  const out = [];
  for (const [id, f] of Object.entries(F)) {
    const m = typeof f.v === 'string' ? BIN_RE.exec(f.v) : null;
    if (!m) continue;
    const k = UNITS[m[4]][1], a = +m[2], b = m[3] === undefined ? null : +m[3];
    out.push([id, UNITS[m[4]][0], m[1] === '<' ? [0, a * k] : m[1] === '>' ? [a * k, Infinity] : [a * k, b * k]]);
  }
  return out;
}
export const inIv = (x, lo, hi) => x >= lo - 1e-9 * Math.max(1, Math.abs(lo)) && x <= hi + 1e-9 * Math.max(1, Math.abs(hi));
export const sameBin = (lo, hi, [a, b]) => Math.abs(lo - a) <= 0.01 * Math.max(1, a)
  && (hi === Infinity ? b === Infinity : b !== Infinity && Math.abs(hi - b) <= 0.01 * Math.max(1, b));

// ---- binding: the quantity nouns a number can belong to, with the facts each one names ----
const QTY = [[/\b(?:head|tail)wind/g, [/^wind\.head_kt$/]], [/\bcrosswind/g, [/^wind\.cross_kt$/]],
  [/\b(?:airspeed|ias|indicated)\b/g, [/^ias_kt$/, /^air\.V_u_s$/]],
  [/\b(?:(?<!closing )speed(?! limit)|moving(?! away)|moves|flying|travell?ing|travels)\b/g,
    [/^ship\.speed_[mu]_s$/, /^air\.V_u_s$/, /^ias_kt$/]],
  [/\b(?:clos(?:ing|es)|closing speed|moving away|recedes|receding|approaches|converge|separate|gap)\b/g,
    [/^closing_cms$/, /^inward_cms$/, /^hazard\.\d\.closing_u_s$/]],
  [/\b(?:limit|permitted|allowed|exceed)\b/g, [/_limit_cms$/]],
  [/\b(?:hazard|rock|comet|satellite|airliner|flock)\b/g, [/^hazard\.\d\.dist_u$/]],
  [/\b(?:closest approach|miss|pass|passes|separation)\b/g, [/^hazard\.\d\.cpa_u$/]], [/\bcontact\b/g, [/^hazard\.\d\.ttc_s$/, /^ttc_s$/]],
  [/\b(?:station|port)\b/g, [/^rho_m$/, /^r_m$/, /^station_distance_bin$/]],
  [/\b(?:clearance|ground|terrain|surface)\b/g, [/^clearance\./, /^air\.agl_m$/]],
  [/\b(?:ceiling|headroom|upper limit|top of the (?:flight )?band)\b/g, [/^edges\.ceiling_u$/]], [/\bstall/g, [/^air\.stall_margin_deg$/]],
  [/\b(?:g-load|load|pulling|pulls)\b/g, [/^air\.n_g$/]], [/\b(?:propellant|fuel|tanks?)\b/g, [/^fuel_frac$/]],
  [/\b(?:sea|water|ocean)\b/g, [/^geo\.sea_frac$/]], [/\b(?:range|camera|viewpoint|from)\b/g, [/^view\.range_km$/]],
  [/\b(?:pixel|imagery|resolution|sample distance)\b/g, [/^view\.gsd_m$/]], [/\btown\b/g, [/^place\.nearest$/]],
  [/\b(?:storm|cell|thunderstorm)\b/g, [/^weather\.cells$/]], [/\b(?:windsock|sock)\b/g, [/^windsock\.from_deg$/]],
  [/\b(?:attitude|pointing|aligned|alignment|port axis)\b/g, [/^att_err_deg$/]],
  [/\b(?:pitch|nose|pitched)\b/g, [/^ship\.pitch_deg$/, /^att\.pitch_deg$/]],
  [/\bglideslope\b/g, [/^ils\.gs_dots$/]], [/\blocalizer\b/g, [/^ils\.loc_dots$/]],
  [/\b(?:sink|vertical speed|descent rate)\b/g, [/^vs_fpm$/]],
  [/\b(?:altitude|height)\b/g, [/^alt_ft$/, /^ra_ft$/]]];
export const CLAUSE_CUT = /[;:]|\b(?:and|with|but|while|whereas|against|versus|than)\b/g;
const wordsIn = (s) => (s.match(/[\p{L}\d][\p{L}\d'-]*/gu) || []).length;
// the nearest quantity noun in `text` (before the number, or up to maxAfter words after it at half a word's extra cost)
function nearestNoun(text, off, len, dim, num, bins, { maxAfter, maxBefore = Infinity }) {
  const end = off + len;
  let best = null;
  for (const [re, pats] of QTY) {
    const ids = [...num.filter(([id, , u]) => UNITS[u][0] === dim && pats.some((p) => p.test(id))).map(([id]) => id),
      ...bins.filter(([id, d]) => d === dim && pats.some((p) => p.test(id))).map(([id]) => id)];
    if (!ids.length) continue;
    for (const m of text.matchAll(re)) {
      const before = m.index + m[0].length <= off, after = m.index >= end;
      const d = before ? wordsIn(text.slice(m.index + m[0].length, off)) : after ? wordsIn(text.slice(end, m.index)) + 0.5 : 0;
      if ((after && d > maxAfter) || (before && d > maxBefore)) continue;
      if (!best || d < best.d) best = { d, ids };
    }
  }
  return best;
}
// a number binds to a quantity noun of its clause; a clause without one looks back into the previous clause (before the
// number only, within eight words); otherwise the number is unbound and any fact of its dimension may hold it
export function boundIds(cl, F) {
  if (cl.clause === undefined) return null;
  const num = numericEntries(F), bins = binEntries(F);
  const best = nearestNoun(cl.clause, cl.off, cl.len, cl.dim, num, bins, { maxAfter: 4.5 })
    || (cl.prevClause
      ? nearestNoun(cl.prevClause, cl.prevClause.length + cl.off, cl.len, cl.dim, num, bins, { maxAfter: -1, maxBefore: 8 }) : null);
  return best ? new Set(best.ids) : null;
}
