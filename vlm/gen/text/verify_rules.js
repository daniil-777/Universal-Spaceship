// vlm/gen/text/verify_rules.js — what holds a claim: the record's facts (factsOf), the slot constructors and derivations
// that map a fact to what a slot states, the category checks, and support(): the fact ids (or 'monitor' / 'safety' /
// 'family') behind one parsed claim. A position claim is checked against the object it is about (its subject).
import { UNITS, WORDS, rangeText, aboutHolds, numericEntries, binEntries, inIv, binWithin, boundIds } from './verify_numbers.js';
import { W, NOUNS, ROUTE_NAMES, WORLD_TAGS, CAUSE_REASONS } from './verify_words.js';
import { norm } from './verify_claims.js';
import { TTC_RANGE, CLR_RANGE_U } from './context.js';
import { groundSupport, GROUND_TYPES } from './verify_ground.js';

export function factsOf(rec) {
  const F = { ...(rec.facts || {}) };
  if (rec.zoom) {
    F['zoom.tags'] = { v: rec.zoom.tags ?? null, unit: null, obs: 'visual' };
    F['zoom.range_bin'] = { v: rec.zoom.range_bin ?? null, unit: null, obs: 'visual' };
  }
  const keys = ['verdict', 'severity', 'reasons', 'best_action', 'safe_actions', 'action_outcome', 'cause', 'p_ref'];
  if (rec.safety) for (const k of keys) F[`safety.${k}`] = { v: rec.safety[k] ?? null, unit: null, obs: 'context' };
  return F;
}

// ---- slot constructors; derive (below) maps the fact value to what the slot states ----
const opt = (d) => (d ? { derive: d } : {});
const slot = (type, id, value, text, extra = {}) => ({ text, fact_id: id, value, unit: null, lo: null, hi: null, type, ...extra });
export const countPhrase = (n, noun) => (n === 0 ? `no ${NOUNS[noun][1]}` : `${n <= 20 ? WORDS[n] : n} ${NOUNS[noun][n === 1 ? 0 : 1]}`);
export const catSlot = (id, value, text, d) => slot('category', id, value, text, opt(d));
export const entSlot = (id, name, d) => slot('entity', id, name, name, opt(d));
export const countSlot = (id, n, noun, d) => slot('count', id, n, countPhrase(n, noun), { unit: 'count', lo: n, hi: n, noun, ...opt(d) });
export const clockSlot = (id, v) => slot('clock', id, v, `${v} o'clock`, { unit: 'clock', lo: v, hi: v });
export const regionSlot = (id, region, d) => slot('region', id, region, `the ${region}`, opt(d));
export const compassSlot = (id, c, d) => slot('compass', id, c, c, opt(d));
// a bin [lo, hi] stated in `unit`; the fact value (in its own unit) must lie inside it
export const rangeSlot = (id, fact, lo, hi, unit, d) => ({ text: rangeText(lo, hi, unit), fact_id: id, value: fact.v, unit, lo, hi,
  type: 'range', shownLo: lo <= 0 ? 0 : lo, shownHi: hi, ...opt(d) });
// an outcome: the safety block's CONTINUE outcome (L/D) or the monitor's p_ref reading (S/A, src 'monitor')
export const outcomeSlot = (code, src) => slot('outcome', src === 'monitor' ? 'safety.p_ref' : 'safety.action_outcome', code,
  W.outcome[code], { src, ...(src === 'monitor' ? {} : { derive: ['field', 'CONTINUE'] }) });
// S/A: the monitor's p_ref as an outcome (0 = clear, 1 = crash certain, otherwise crash possible)
export const prefOutcome = (p) => (p === null || p === undefined ? null : p <= 0 ? 'clear' : p >= 1 ? 'crash_certain' : 'crash_possible');
// an L/D outcome a text may state next to the monitor's verdict (controller ruling (b)): a benign outcome never stands next
// to an UNSAFE verdict, and a SAFE verdict admits only benign outcomes
const BENIGN = ['landed', 'capture', 'clear'];
const OUTCOME_OK = { SAFE: BENIGN, CAUTION: [...BENIGN, 'go_around', 'breakout', 'crash_possible'] };
export const outcomeAgrees = (outcome, verdict) => (verdict === 'UNSAFE' ? !!outcome && !BENIGN.includes(outcome)
  : (OUTCOME_OK[verdict] || []).includes(outcome));

// the cloud cover of a METAR group ("SCT030", "-RA BKN012", "NSC") or of a word (few, scattered, broken, overcast)
const CLOUD_WORD = { few: 'FEW', scattered: 'SCT', broken: 'BKN', overcast: 'OVC' };
export function cloudCode(v) {
  const s = String(v), m = /(?:^|[^A-Za-z])(FEW|SCT|BKN|OVC|NSC|SKC|NONE)(?=\d{3}|[^A-Za-z]|$)/.exec(s);
  if (m) return m[1];
  const w = /\b(few|scattered|broken|overcast)\b/i.exec(s);
  return w ? CLOUD_WORD[w[1].toLowerCase()] : null;
}
const minDist = (v, t) => { const d = v.filter((c) => c.type === t).map((c) => c.dist_u); return d.length ? Math.min(...d) : null; };
const has = (o, k) => Object.prototype.hasOwnProperty.call(o, k);
const cmpFact = (v, F, id) => (F[id] && typeof F[id].v === 'number' ? (v > F[id].v ? 'above' : 'below') : null);
const DERIVE = { abs: (v) => Math.abs(v), has: (v, _, k) => v.includes(k), count_of: (v, _, k) => (v.includes(k) ? null : 0),
  len: (v) => v.length, papi_red: (v) => 4 - v, gt: (v, _, x) => v > x, lt: (v, _, x) => v < x, field: (v, _, k) => (v ? v[k] : null),
  idx: (v, _, i, k) => (v && v[i] ? v[i][k] : null), sign: (v) => Math.sign(v), all_true: (v) => Object.values(v).every(Boolean),
  false_keys: (v) => Object.keys(v).filter((k) => !v[k]).sort().join(','), sorted: (v) => [...v].sort().join(','),
  map: (v, _, m) => (has(m, v) ? m[v] : null), cmp_fact: cmpFact,
  in_list: (v, _, list) => list.includes(v), relief: (v) => (v || []).find((t) => ['MOUNTAINS', 'HILLS', 'FLAT'].includes(t)) ?? null,
  in_band: (v, _, a, b) => v >= a && v <= b, cloud_code: (v) => cloudCode(v), count_type: (v, _, t) => v.filter((c) => c.type === t).length,
  min_dist_type: (v, _, t) => minDist(v, t) };
export const derive = (d, v, facts) => (!d ? v : v === null || v === undefined ? null : DERIVE[d[0]](v, facts, ...d.slice(1)));

// ---- category checks: (F, value, claim, rec) -> supporting fact ids ----
const val = (F, id) => (F[id] ? F[id].v : undefined);
const idsWhere = (F, pred) => Object.entries(F).filter(([id, f]) => f.v !== null && f.v !== undefined && pred(id, f.v)).map(([id]) => id);
const is = (id, v) => (F) => idsWhere(F, (i, x) => i === id && (Array.isArray(v) ? v.includes(x) : x === v));
const cloudIs = (codes) => (F) => idsWhere(F, (id, x) => id === 'scene.clouds' && codes.includes(cloudCode(x)));
const hazardIdx = (F) => [...new Set(Object.keys(F).map((id) => (/^hazard\.(\d+)\./.exec(id) || [])[1]).filter(Boolean))];
// the hazards, of one kind when the claim's subject names one (position facts are null for a hazard out of the frame)
function hazards(F, subj) {
  const k = subj && subj.kind === 'hazard' ? subj.value : null;
  return hazardIdx(F).filter((i) => !k || val(F, `hazard.${i}.kind`) === k);
}
export const regionOfBox = (b) => {
  const row = ['upper', 'middle', 'lower'][Math.min(2, Math.floor((3 * ((b[1] + b[3]) / 2)) / 504))];
  const col = ['left', 'centre', 'right'][Math.min(2, Math.floor((3 * ((b[0] + b[2]) / 2)) / 896))];
  return `${row} ${col}`.replace('middle centre', 'centre');
};
// a side word (left, right, centre, top, bottom) holds for a 3x3 region
const SIDE_OF_REGION = { left: (r) => r.endsWith('left'), right: (r) => r.endsWith('right'),
  centre: (r) => r === 'centre' || r.endsWith('centre'), top: (r) => r.startsWith('upper'), bottom: (r) => r.startsWith('lower') };
// the named features in view ([fact id, features]), of one name when given
const named = (name) => (f) => f && f.region && (!name || norm(f.name) === norm(name));
const features = (F, name) => idsWhere(F, (id, x) => Array.isArray(x) && x.some(named(name)))
  .map((id) => [id, F[id].v.filter(named(name))]);
const featuresWhere = (F, name, pred) => features(F, name).filter(([, fs]) => fs.some((f) => pred(f.region))).map(([id]) => id);
const coast = (F, side) => idsWhere(F, (id, x) => id === 'geo.coast_side' && x === side);
function sideCheck(F, v, cl) {
  const subj = cl.subject;
  if (subj && (subj.kind === 'body' || subj.kind === 'object')) return [];
  if (subj && subj.kind === 'sea') return coast(F, v);
  if (subj && subj.kind === 'entity') return featuresWhere(F, subj.value, SIDE_OF_REGION[v]);
  const out = [];
  for (const i of hazards(F, subj)) {
    const side = val(F, `hazard.${i}.side`), box = val(F, `hazard.${i}.box_px`);
    const upDown = Array.isArray(box) && (v === 'top' || v === 'bottom') && SIDE_OF_REGION[v](regionOfBox(box));
    if (side === v || upDown) out.push(`hazard.${i}.side`);
  }
  return subj ? out : [...out, ...coast(F, v)];
}
const APPROACH = ['TRANSFER', 'H1 ACQ', 'CORRIDOR', 'FINAL'], LEAVING = ['DEPART', 'BREAKOUT'], ON_GROUND = ['ROLLOUT', 'STOP'];
// closing in (+1) or moving away (-1): S/A a hazard's closing rate; D the closing rate, else the phase; L the aircraft
// approaching the runway (anything airborne but a go-around)
function closingCheck(F, v, cl, rec) {
  if (rec.family === 'D') {
    const c = val(F, 'closing_cms'), p = val(F, 'phase');
    if (typeof c === 'number') return Math.sign(c) === v ? ['closing_cms'] : [];
    return (v === 1 ? APPROACH : LEAVING).includes(p) ? ['phase'] : [];
  }
  if (rec.family === 'L') {
    const vm = val(F, 'vert_mode'), ground = val(F, 'wow') === true || ON_GROUND.includes(vm);
    if (v === 1) return ground || vm === 'GA' ? [] : ['family'];
    return vm === 'GA' ? ['vert_mode'] : [];
  }
  return hazards(F, cl.subject).filter((i) => Math.sign(val(F, `hazard.${i}.closing_u_s`)) === v)
    .map((i) => `hazard.${i}.closing_u_s`);
}
const tagCheck = (F, v) => idsWhere(F, (id, x) => (id === 'zoom.tags' && Array.isArray(x) && x.includes(v))
  || ((id === 'world' || id === 'route') && (WORLD_TAGS[x] || []).includes(v)));
const dots = (id, v, pos) => (F) => idsWhere(F, (i, x) => i === id && (v === 'on' ? Math.abs(x) <= 1 : v === pos ? x > 1 : x < -1));
const lightCheck = (F, vs) => idsWhere(F, (id, x) => ((id === 'sun.class' || id === 'scene.time') && vs.includes(x))
  || (id === 'sun.lit' && ((vs.includes('lit') && x >= 0.9) || (vs.includes('shadow') && x <= 0.1))));
function dclosingCheck(F, v) {
  const c = val(F, 'closing_cms'), l = val(F, 'corridor_limit_cms');
  return typeof c === 'number' && typeof l === 'number' && (c > l) === (v === 'above') ? ['closing_cms', 'corridor_limit_cms'] : [];
}
// rain: the rain flag or a METAR rain group; a storm: a storm-type weather cell (type 1); storm clouds: that or a stormy
// preset; snow: the ICE zoom tag
const rainCheck = (F) => idsWhere(F, (id, x) => (id === 'scene.rain' && x === true)
  || (id === 'scene.clouds' && /(?:^|\s)[-+]?RA\b/.test(x)));
const stormCheck = (F) => idsWhere(F, (id, x) => id === 'weather.cells' && Array.isArray(x) && x.some((c) => c && c.type === 1));
const stormCloudCheck = (F) => [...is('weather.preset', 'storm')(F), ...stormCheck(F)];
const signOf = (id, pos) => (F, v) => idsWhere(F, (i, x) => i === id && (v === pos ? x > 0 : x < 0));
const CAT = { size: (F, v) => idsWhere(F, (id, x) => /^hazard\.\d+\.size_bin$/.test(id) && x === v), side: sideCheck,
  sky: (F, v) => is('weather.preset', v)(F), skyclear: (F) => [...is('weather.preset', 'clear')(F), ...cloudIs(['NSC', 'SKC', 'NONE'])(F)],
  overcast: cloudIs(['BKN', 'OVC']), rain: rainCheck, storm: stormCheck, stormcloud: stormCloudCheck,
  snow: (F) => tagCheck(F, 'ICE').filter((id) => id === 'zoom.tags'),
  stars: () => [], tunnel: (F) => is('in_tunnel', true)(F),
  // T10-v: L's scene.in_cloud is already a boolean (the page's own closing-in-fog reading); A's air.in_cloud is a 0-1 density
  incloud: (F) => idsWhere(F, (id, x) => (id === 'air.in_cloud' && x > 0.25) || (id === 'scene.in_cloud' && x === true)),
  colour: (F, vs) => idsWhere(F, (id, x) => /^image\.palette_\d$/.test(id) && vs.includes(x)), light: lightCheck,
  bright: (F, v) => is('image.brightness_bin', v)(F), texture: (F, v) => is('image.edge_bin', v)(F), vis: (F, v) => is('scene.vis', v)(F),
  phase: (F, v) => is('phase', v)(F), lphase: (F, v) => is('vert_mode', v)(F), papi: (F, v) => is('papi_whites_cam', v)(F),
  band: (F, v) => is('air.speed_band', v)(F), world: (F, vs) => is('world', vs)(F), tag: tagCheck,
  clouds: (F, v) => cloudIs(Array.isArray(v) ? v : [v])(F),
  turb: (F, v) => idsWhere(F, (id, x) => id === 'air.turbulence' && W.turb[x] === v),
  stall: (F) => idsWhere(F, (id, x) => (id === 'air.stall_margin_deg' && x < 2) || (id === 'air.stalled' && x === true)),
  gear: (F, v) => is('cfg.gear', v)(F), gs: (F, v) => dots('ils.gs_dots', v, 'above')(F),
  loc: (F, v) => dots('ils.loc_dots', v, 'right')(F),
  cross: signOf('wind.cross_kt', 'right'), headtail: signOf('wind.head_kt', 'head'), closing: closingCheck, dclosing: dclosingCheck };

// ---- position claims, bound to their subject ----
const COMPASS = ['north', 'north-east', 'east', 'south-east', 'south', 'south-west', 'west', 'north-west'];
const nearBearing = (b, c) => Math.abs(((b - COMPASS.indexOf(c) * 45 + 540) % 360) - 180) <= 22.5;
function positionSupport(cl, F, rec = null) {
  const v = cl.value, subj = cl.subject, off = subj && (subj.kind === 'body' || subj.kind === 'object');
  if (cl.type === 'clock') {
    if (off || (subj && subj.kind !== 'hazard')) return [];
    return subj ? hazards(F, subj).filter((i) => val(F, `hazard.${i}.clock`) === v).map((i) => `hazard.${i}.clock`)
      : idsWhere(F, (id, x) => id.endsWith('.clock') && x === v);
  }
  if (cl.type === 'region') {
    if (off) return [];
    if (subj && subj.kind === 'sea') return idsWhere(F, (id, x) => id === 'geo.coast_side' && SIDE_OF_REGION[x] && SIDE_OF_REGION[x](v));
    if (subj && subj.kind === 'entity') return featuresWhere(F, subj.value, (r) => r === v);
    const inRegion = (i) => { const b = val(F, `hazard.${i}.box_px`); return Array.isArray(b) && regionOfBox(b) === v; };
    const boxes = hazards(F, subj).filter(inRegion).map((i) => `hazard.${i}.box_px`);
    return subj ? boxes : [...boxes, ...featuresWhere(F, null, (r) => r === v)];
  }
  // compass: a place's bearing from the view centre. A Z view has no docking port, runway or windsock, so an object noun
  // there is part of a place name ('Port Elizabeth', 'Port Moresby') and binds nothing (v1)
  if (subj && subj.kind !== 'entity' && !(subj.kind === 'object' && rec && rec.family === 'Z')) return [];
  const name = subj && subj.kind === 'entity' ? norm(subj.value) : null;
  const placed = (x) => x && typeof x === 'object' && Number.isFinite(x.bearing) && (!name || norm(x.name) === name);
  return idsWhere(F, (id, x) => placed(x) && nearBearing(x.bearing, v));
}
function factNames(id, f) {
  const s = new Set(), add = (x) => { if (typeof x === 'string') s.add(norm(x)); }, v = f.v;
  if (Array.isArray(v)) v.forEach((x) => add(x && typeof x === 'object' ? x.name : x));
  else if (v && typeof v === 'object') add(v.name);
  else add(v);
  if ((id === 'route' || id === 'world') && ROUTE_NAMES[v]) add(ROUTE_NAMES[v]);
  return s;
}
function numberSupport(cl, F, ctx, attributed) {
  const k = UNITS[cl.unit][1], bound = boundIds(cl, F), ok = (id) => !bound || bound.has(id);
  const dimOk = ([id, , u]) => ok(id) && UNITS[u][0] === cl.dim, si = ([, x, u]) => Math.abs(x * UNITS[u][1]);
  if (cl.type === 'number') return numericEntries(F).filter((e) => dimOk(e) && aboutHolds(si(e) / k, cl.value)).map(([id]) => id);
  const lo = cl.lo * k, hi = cl.hi * k, out = numericEntries(F).filter((e) => dimOk(e) && inIv(si(e), lo, hi)).map(([id]) => id);
  for (const [id, dim, iv] of binEntries(F)) if (ok(id) && dim === cl.dim && binWithin(lo, hi, iv)) out.push(id);
  if (attributed && ctx && ctx.monitor) {
    const t = TTC_RANGE[ctx.monitor.ttc_bin], r = CLR_RANGE_U[ctx.monitor.clr_bin];
    const inBin = (cl.dim === 'time' && t && binWithin(lo, hi, t)) || (cl.dim === 'len' && r && binWithin(lo, hi, [r[0] * 19, r[1] * 19]));
    if (inBin) out.push('monitor');
  }
  return out;
}
function countSupport(cl, F) {
  const v = cl.value, n = F['hazards.count_in_frame'], kinds = F.kinds_in_frame, papi = F.papi_whites_cam, g = F['ground.hazards'];
  if (g && Array.isArray(g.v) && F['ground.complete'] && F['ground.complete'].v === true && v > 0) {
    const mine = g.v.filter((h) => cl.noun === 'hazard' || h.kind === cl.noun);
    if (mine.length === v && mine.every((h) => h.alone) && (cl.noun === 'hazard' || g.v.every((h) => h.kind === cl.noun))) return ['ground.hazards'];
  }
  if (cl.noun === 'hazard') return n && n.v === v ? ['hazards.count_in_frame'] : [];
  if (cl.noun === 'white') return papi && papi.v === v ? ['papi_whites_cam'] : [];
  if (cl.noun === 'red') return papi && typeof papi.v === 'number' && 4 - papi.v === v ? ['papi_whites_cam'] : [];
  const jets = (id, x) => (id === 'jets_failed' || id === 'jets_firing') && Array.isArray(x) && x.length === v;
  if (cl.noun === 'jet') return idsWhere(F, jets);
  if (!kinds || !Array.isArray(kinds.v)) return [];
  if (v === 0) return kinds.v.includes(cl.noun) ? [] : ['kinds_in_frame'];
  return kinds.v.length === 1 && kinds.v[0] === cl.noun && n && n.v === v ? ['kinds_in_frame', 'hazards.count_in_frame'] : [];
}
// a body or object in the frame: the Earth and Moon by their in-frame flags (Z images are of the Earth), the PAPI and the
// windsock when their fact is recorded (it is null out of view), the station in D and the runway in L by the family
function presenceSupport(v, F, rec) {
  if (v === 'earth') return rec.family === 'Z' ? ['family'] : is('earth_in_frame', true)(F);
  if (v === 'moon') return is('moon_in_frame', true)(F);
  if (v === 'papi') return idsWhere(F, (id) => id === 'papi_whites_cam');
  if (v === 'windsock') return idsWhere(F, (id) => id === 'windsock.from_deg');
  return (v === 'station' && rec.family === 'D') || (v === 'runway' && rec.family === 'L') ? ['family'] : [];
}
// the fact ids (or 'monitor' / 'safety' / 'family') that support a claim's positive reading; [] when nothing does
export function support(cl, F, rec, ctx, attributed) {
  const S = rec.safety, v = cl.value, mon = attributed && ctx && ctx.monitor ? ctx.monitor : null;
  if (cl.type === 'number' || cl.type === 'range') return numberSupport(cl, F, ctx, attributed);
  if (cl.type === 'count') return countSupport(cl, F);
  if (cl.type === 'clock' || cl.type === 'region' || cl.type === 'compass') return positionSupport(cl, F, rec);
  if (GROUND_TYPES.includes(cl.type)) return groundSupport(cl, F);
  if (cl.type === 'bearing_clock') return idsWhere(F, (id, x) => id.endsWith('.bearing_clock') && x === v);
  if (cl.type === 'entity') return idsWhere(F, (id) => !id.startsWith('safety.') && factNames(id, F[id]).has(norm(v)));
  if (cl.type === 'category') return CAT[cl.dim] ? CAT[cl.dim](F, v, cl, rec) : [];
  if (cl.type === 'kind') {
    const k = val(F, 'kinds_in_frame'), out = Array.isArray(k) && k.includes(v) ? ['kinds_in_frame'] : [];
    const inFrame = hazards(F, { kind: 'hazard', value: v }).filter((i) => val(F, `hazard.${i}.in_frame`) === true);
    return [...out, ...inFrame.map((i) => `hazard.${i}.kind`)];
  }
  if (cl.type === 'presence') return presenceSupport(v, F, rec);
  if (cl.type === 'cause') {
    const m = ctx && ctx.monitor;
    return S && S.cause === v && (!m || (CAUSE_REASONS[v] || []).some((r) => m.reasons.includes(r))) ? ['safety.cause'] : [];
  }
  if (cl.type === 'adjective' || cl.type === 'disagree' || cl.type === 'unknown_entity') return [];
  const gtOut = S && S.action_outcome ? S.action_outcome.CONTINUE : null, LD = rec.family === 'L' || rec.family === 'D';
  if (cl.type === 'outcome') {
    const out = [];
    if (mon && (rec.family === 'S' || rec.family === 'A') && prefOutcome(mon.p_ref) === v) out.push('monitor');
    if (gtOut === v && !(ctx && ctx.monitor && LD && !outcomeAgrees(v, ctx.monitor.verdict))) out.push('safety.action_outcome');
    return out;
  }
  if (mon) {
    if (cl.type === 'verdict') return mon.verdict === v ? ['monitor'] : [];
    if (cl.type === 'action') return mon.action === v ? ['monitor'] : [];
    if (cl.type === 'reason') return mon.reasons.includes(v) ? ['monitor'] : [];
  }
  if (!S) return [];
  if (cl.type === 'verdict') return S.verdict === v ? ['safety'] : [];
  if (cl.type === 'action') return S.best_action === v || (S.safe_actions || []).includes(v) ? ['safety'] : [];
  if (cl.type === 'reason') return S.reasons.includes(v) ? ['safety'] : [];
  return [];
}
