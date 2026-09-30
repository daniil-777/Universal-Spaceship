// vlm/gen/text/context.js — the one Context line of spec §5.7 (telemetry exact, monitor = Pilot Eye's tuple), rendered by
// build.mjs and at runtime by vlm/web/narrator.js, parsed back by the verifier; gt+noise corruption for rows without oof.
// Browser-loadable. contextSupplies() / rowContext() are THE §5.7 export rule (export.js imports them): the telemetry
// supplies its own fact ids, and a monitor tuple supplies the safety.* ids text cites (SAFETY_TEXT_IDS).
import { SAFETY_TEXT_IDS } from '../schema.js';
export const REASON_TEXT = Object.freeze({ HAZARD_AHEAD: 'a hazard ahead', HAZARD_CLOSING_FAST: 'a hazard closing fast', TERRAIN_CLOSE: 'terrain close by', BUILDING_CLOSE: 'buildings close by',
  CORRIDOR_EDGE: 'the corridor edge', STALL: 'a stall', OVERSTRESS: 'overstress', SEVERE_TURBULENCE: 'severe turbulence', STORM_CELL: 'a storm cell', PULL_UP: 'ground proximity',
  UNSTABLE_APPROACH: 'an unstable approach', LOCALIZER_DEVIATION: 'a localizer deviation', GLIDESLOPE_DEVIATION: 'a glideslope deviation', SPEED_OUT_OF_BAND: 'speed outside the band',
  HIGH_SINK_RATE: 'a high sink rate', STRONG_CROSSWIND: 'a strong crosswind', TAILWIND: 'a tailwind', RUNWAY_EDGE: 'the runway edge', CANNOT_STOP: 'too little runway to stop',
  KOS_VIOLATION: 'a keep-out sphere violation', CLOSING_TOO_FAST: 'closing too fast', LATERAL_MISALIGNMENT: 'lateral misalignment', ATTITUDE_ERROR: 'an attitude error',
  JET_FAILURE: 'a failed jet', LOW_FUEL: 'low propellant', NO_BREAKOUT_AVAILABLE: 'no safe breakout' });
export const ACTION_TEXT = Object.freeze({ CONTINUE: 'continue', CLIMB: 'climb', DESCEND: 'descend', TURN_LEFT: 'turn left', TURN_RIGHT: 'turn right', SPEED_UP: 'speed up', SLOW_DOWN: 'slow down',
  GO_AROUND: 'go around', HOLD_POSITION: 'hold position', BREAKOUT: 'break out', NONE_SAFE: 'no safe action' });
export const ttcBin = (s) => (s === null || s === undefined ? 'none' : s < 1 ? '<1 s' : s < 3 ? '1-3 s' : s < 6 ? '3-6 s' : '>6 s');
export const clrBin = (u) => (u === null || u === undefined ? 'none' : u < 5 ? '<5 u' : u < 15 ? '5-15 u' : u < 40 ? '15-40 u' : '>40 u');
// the bins as intervals: TTC in seconds, clearance in corridor units (19 m each)
export const TTC_RANGE = Object.freeze({ '<1 s': [0, 1], '1-3 s': [1, 3], '3-6 s': [3, 6], '>6 s': [6, Infinity] });
export const CLR_RANGE_U = Object.freeze({ '<5 u': [0, 5], '5-15 u': [5, 15], '15-40 u': [15, 40], '>40 u': [40, Infinity] });
const p2 = (x) => (x === null || x === undefined ? null : +(+x).toFixed(2));
export function monitorOf(eye, facts, family) {
  const sa = family === 'S' || family === 'A';
  return { verdict: eye.verdict, severity: eye.severity, reasons: [...eye.reasons], action: eye.best_action, p_ref: sa ? p2(eye.p_ref) : null,
    ttc_bin: sa ? ttcBin(eye.ttc_s) : 'none', clr_bin: sa ? clrBin(facts['clearance.min_u'] ? facts['clearance.min_u'].v : null) : 'none' };
}
export const TELEMETRY = Object.freeze({ Z: [['place', 'place.nearest'], ['range', 'view.range_km'], ['sun', 'sun.class']],
  S: [['speed', 'ship.speed_m_s']], A: [['speed', 'ship.speed_m_s'], ['g-load', 'air.n_g'], ['stall margin', 'air.stall_margin_deg'], ['turbulence', 'air.turbulence'], ['wind', 'air.wind_u_s']],
  L: [['IAS', 'ias_kt'], ['LOC dots', 'ils.loc_dots'], ['GS dots', 'ils.gs_dots'], ['headwind', 'wind.head_kt'], ['crosswind', 'wind.cross_kt']],
  D: [['closing', 'closing_cms'], ['corridor limit', 'corridor_limit_cms'], ['speed limit', 'speed_limit_cms'], ['fuel', 'fuel_frac'], ['failed jets', 'jets_failed'], ['breakout available', 'breakout_available']] });
const LABEL_ID = Object.fromEntries(Object.values(TELEMETRY).flat());
const show = (v, unit) => (v === null || v === undefined ? 'n/a' : Array.isArray(v) ? (v.length ? v.join(' ') : 'none') : typeof v === 'object' ? v.name : `${typeof v === 'number' ? +v.toFixed(2) : v}${unit ? ' ' + unit : ''}`);
export const telemetryOf = (facts, family) => (TELEMETRY[family] || []).filter(([, id]) => facts[id]).map(([label, id]) => [label, show(facts[id].v, facts[id].unit), id]);
export function renderContext({ telemetry = [], monitor = null }) {
  const t = telemetry.map(([l, v]) => `${l} ${v}`).join(', ');
  const m = monitor ? `; monitor: ${monitor.verdict} (severity ${monitor.severity}), reasons: ${monitor.reasons.length ? monitor.reasons.join(' ') : 'none'}, action: ${monitor.action}${monitor.p_ref === null ? '' : `, p_ref ${monitor.p_ref}`}, TTC ${monitor.ttc_bin}, clearance ${monitor.clr_bin}` : '';
  return `Context: telemetry: ${t || 'none'}${m}.`;
}
// telemetry comes back as [[label, text, factId]] (the labels are a closed set, so a value may hold commas or spaces)
const LABEL_RE = new RegExp(`(?:^|, )(${Object.keys(LABEL_ID).sort((a, b) => b.length - a.length).map((l) => l.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&')).join('|')}) `, 'g');
export function parseContext(line) {
  const m = /monitor: (SAFE|CAUTION|UNSAFE) \(severity (\d)\), reasons: ([A-Z_ ]+|none), action: ([A-Z_]+)(?:, p_ref ([\d.]+))?, TTC ([^,]+), clearance ([^.]+)\.$/.exec(line);
  const raw = (/^Context: telemetry: (.*?)(?:; monitor:|\.$)/.exec(line) || [])[1] || '', telemetry = [];
  if (raw !== 'none') {
    const hits = [...raw.matchAll(LABEL_RE)];
    hits.forEach((h, i) => { const from = h.index + h[0].length, to = i + 1 < hits.length ? hits[i + 1].index : raw.length; telemetry.push([h[1], raw.slice(from, to), LABEL_ID[h[1]]]); });
  }
  return { telemetry, monitor: m ? { verdict: m[1], severity: +m[2], reasons: m[3] === 'none' ? [] : m[3].split(' '), action: m[4], p_ref: m[5] === undefined ? null : +m[5], ttc_bin: m[6], clr_bin: m[7] } : null };
}
// a Context given as a line, as {telemetry, monitor} or as null, reduced to {ids: Set of supplied telemetry fact ids, monitor}
export function normContext(ctx) {
  if (ctx === null || ctx === undefined) return null;
  const c = typeof ctx === 'string' ? parseContext(ctx) : ctx;
  return { monitor: c.monitor || null, ids: new Set((c.telemetry || []).map((t) => (Array.isArray(t) ? t[2] : null)).filter(Boolean)) };
}
// contextSupplies(factIds, context): every id is supplied (an id string works too; the older (context, factId) order is accepted)
export function contextSupplies(a, b) {
  const [ids, ctx] = typeof a === 'string' && !a.startsWith('Context:') ? [[a], b] : Array.isArray(a) ? [a, b] : [typeof b === 'string' ? [b] : b, a];
  const c = normContext(ctx);
  return (ids || []).every((id) => !!c && (c.ids.has(id) || (!!c.monitor && SAFETY_TEXT_IDS.includes(id))));
}
// the Narrator row of one text item: kept only when its Context supplies every context-class fact it cites, and it carries
// the Context line when it needs it, else with p = 0.5 (§5.7)
export const rowContext = (item, context, rng) => ({ keep: !item.needsContext || contextSupplies(item.context_facts || [], context), withCtx: !!item.needsContext || rng() < 0.5 });
const BINS = { ttc: ['<1 s', '1-3 s', '3-6 s', '>6 s'], clr: ['<5 u', '5-15 u', '15-40 u', '>40 u'] };
const shift = (list, v, rng) => { const i = list.indexOf(v); if (i < 0 || rng() >= 0.2) return v; return list[Math.max(0, Math.min(list.length - 1, i + (rng() < 0.5 ? -1 : 1)))]; };
// the pool holds tuples of the record's own family: given `family` (the build passes it), every pool entry must carry the
// same `family`, or corruptMonitor throws
export function corruptMonitor(m, { rng, confusion = null, pool, family = null }) {
  if (family) for (const p of pool) if (p.family !== family) throw new Error(`corruptMonitor: pool entry of family ${p.family} for a ${family} record`);
  const verdicts = ['SAFE', 'CAUTION', 'UNSAFE'], row = confusion ? confusion[m.verdict] : Object.fromEntries(verdicts.map((v) => [v, v === m.verdict ? 0.75 : 0.125]));
  let r = rng(), verdict = m.verdict; for (const v of verdicts) { r -= row[v]; if (r <= 0) { verdict = v; break; } }
  const same = pool.filter((p) => p.verdict === verdict), src = verdict === m.verdict || !same.length ? m : same[Math.floor(rng() * same.length)];
  // no same-family tuple with the drawn verdict: the tuple stays whole (a verdict without its severity, reasons and action is
  // not a monitor output), and only the bins shift
  return { verdict: src.verdict, severity: src.severity, reasons: [...src.reasons], action: src.action, p_ref: src.p_ref, ttc_bin: shift(BINS.ttc, m.ttc_bin, rng), clr_bin: shift(BINS.clr, m.clr_bin, rng) };
}
