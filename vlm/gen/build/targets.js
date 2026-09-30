// vlm/gen/build/targets.js — Pilot Eye targets from safety_eye and facts, and per-head masks from the family and the eye scope
// (spec §10.1): p_ref S/A only; regressions masked per family (closing only within EYE.closingRhoM, LOC dots within
// EYE.locNm). T.verdict is the index into VERDICTS (SAFE 0, CAUTION 1, UNSAFE 2).
import { VERDICTS, REASONS, ACTIONS, FAMILY_ACTIONS, ZOOM_TAGS } from '../schema.js';
import { EYE, EYE_REMOVED } from '../safety.js';

export const REG = Object.freeze(['log_ttc', 'log_clear', 'agl', 'vs', 'closing', 'loc_dots']);
// reason codes a family's safety_eye can never emit are masked (§10.1). L: TAILWIND (the eye drops it), GLIDESLOPE_DEVIATION
// (the eye never sees GS dots, EYE_REMOVED.L GS_DOTS, and GATE_MODES hides the vertical gate) and SPEED_OUT_OF_BAND (while
// EYE.hidden has SPEED_BAND). RUNWAY_EDGE, CANNOT_STOP and HIGH_SINK_RATE stay: the excursion/overrun/hard outcomes still
// emit them. D: LOW_FUEL, JET_FAILURE and NO_BREAKOUT_AVAILABLE (EYE_REMOVED.D). S/A: corridorSafety is the eye.
const L_EYE_NEVER = ['TAILWIND', 'GLIDESLOPE_DEVIATION', ...(EYE.hidden.includes('SPEED_BAND') ? ['SPEED_OUT_OF_BAND'] : [])];
const D_EYE_NEVER = ['LOW_FUEL', 'JET_FAILURE', 'NO_BREAKOUT_AVAILABLE'];
export const REASON_SETS = Object.freeze({ S: REASONS.slice(0, 10), A: REASONS.slice(0, 10), L: REASONS.slice(10, 19).filter((r) => !L_EYE_NEVER.includes(r)),
  D: REASONS.slice(19).filter((r) => !D_EYE_NEVER.includes(r)), Z: [] });
const v = (rec, id) => (rec.facts[id] && rec.facts[id].v !== null && rec.facts[id].v !== undefined ? rec.facts[id].v : null);
export function targetsOf(rec) {
  const f = rec.family, e = rec.safety_eye, reg = {};
  const T = { reasons: [], safe_actions: [], zoom_tags: [], range_bin: 0 };
  const M = { verdict: 0, severity: 0, reasons: REASONS.map(() => 0), actions: ACTIONS.map(() => 0), p_ref: 0, reg: REG.map(() => 0), zoom_tags: 0, range_bin: 0 };
  if (f === 'Z') { T.zoom_tags = ZOOM_TAGS.map((t) => ((rec.zoom.tags || []).includes(t) ? 1 : 0)); T.range_bin = rec.zoom.range_bin; M.zoom_tags = 1; M.range_bin = 1; } else {
    T.verdict = VERDICTS.indexOf(e.verdict); T.severity = e.severity; T.reasons = REASONS.map((r) => (e.reasons.includes(r) ? 1 : 0)); T.safe_actions = ACTIONS.map((a) => (e.safe_actions.includes(a) ? 1 : 0));
    M.verdict = 1; M.severity = 1; M.reasons = REASONS.map((r) => (REASON_SETS[f].includes(r) ? 1 : 0)); M.actions = ACTIONS.map((a) => (FAMILY_ACTIONS[f].includes(a) ? 1 : 0));
    if (f === 'S' || f === 'A') {
      T.p_ref = e.p_ref; M.p_ref = 1;
      if (e.ttc_s !== null && e.ttc_s !== undefined) reg.log_ttc = Math.log1p(Math.min(e.ttc_s, 30));
      const c = v(rec, 'clearance.min_u'); if (c !== null) reg.log_clear = Math.log1p(Math.min(Math.max(c, 0), 50));
    }
    if (f === 'A' && v(rec, 'air.agl_m') !== null) reg.agl = v(rec, 'air.agl_m') / 1000;
    if (f === 'L') {
      if (v(rec, 'ra_ft') !== null) reg.agl = (v(rec, 'ra_ft') * 0.3048) / 1000;
      if (v(rec, 'vs_fpm') !== null) reg.vs = (v(rec, 'vs_fpm') * 0.00508) / 10;
      if (v(rec, 'ils.loc_dots') !== null && (v(rec, 'thr_nm') ?? 99) <= EYE.locNm) reg.loc_dots = Math.max(-3, Math.min(3, v(rec, 'ils.loc_dots'))) / 3;
    }
    if (f === 'D' && v(rec, 'closing_cms') !== null && (v(rec, 'rho_m') ?? 1e9) <= EYE.closingRhoM) reg.closing = v(rec, 'closing_cms') / 10;
  }
  REG.forEach((k, i) => { if (reg[k] !== undefined) { M.reg[i] = 1; T[k] = reg[k]; } else T[k] = 0; });
  return { targets: T, masks: M, removed: EYE_REMOVED };
}
