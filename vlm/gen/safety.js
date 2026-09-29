// vlm/gen/safety.js — the one safety combinator of spec §4.3 for the corridor (S/A), landing (L) and docking (D), and
// safety_eye: the same rules with the now-criteria Pilot Eye cannot see at 160x96 removed. Pure and browser-loadable:
// the label modules reduce sim state to the inputs below. Rulings where §4.3 is silent (also written to labels.json):
// HAZARD_CLOSING_FAST and STORM_CELL only explain non-SAFE verdicts; CORRIDOR_EDGE is added at any verdict; the L dot
// CAUTION band is |dots| > 1 on GS at hRA >= 50 ft; a go-around outcome adds UNSTABLE_APPROACH; an UNSAFE chain ending
// at an unsafe CONTINUE becomes NONE_SAFE; the D NO-GO includes breakout availability (guidance.js goOk).
import { REASONS } from './schema.js';

export const C_NEAR_DEFAULT = 2.5;
export const CORRIDOR_ACTIONS = Object.freeze(['CONTINUE', 'CLIMB', 'DESCEND', 'TURN_LEFT', 'TURN_RIGHT', 'SPEED_UP', 'SLOW_DOWN']);
export const TIE_ORDER = Object.freeze(['CLIMB', 'TURN_LEFT', 'TURN_RIGHT', 'SLOW_DOWN', 'SPEED_UP', 'DESCEND']);
export const EYE_REMOVED = Object.freeze({ L: Object.freeze(['TAILWIND', 'GS_DOTS', 'LOC_DOTS_BEYOND_1NM']), D: Object.freeze(['LOW_FUEL', 'JET_FAILURE', 'NO_BREAKOUT_AVAILABLE', 'CLOSING_BEYOND_EYE_CLOSING_RHO']) });
// safety_eye scope boundaries and extra hidden criteria; Task 4's >= 1 px sensitivity test sets these (spec §4.3 last table)
export const EYE = Object.freeze({ locNm: 1, closingRhoM: 20, hidden: Object.freeze([]) });
const HAZARDS = ['rock', 'comet', 'satellite', 'airliner', 'birds'], L_BAD = ['crash', 'excursion', 'overrun', 'short', 'tailstrike', 'hard'];
const CAUSE_REASON = (c) => (HAZARDS.includes(c) ? 'HAZARD_AHEAD' : c === 'terrain' ? 'TERRAIN_CLOSE' : c === 'building' || c === 'roof' ? 'BUILDING_CLOSE' : c === 'overstress' ? 'OVERSTRESS' : null);
const CLR_REASON = { hazard: 'HAZARD_AHEAD', terrain: 'TERRAIN_CLOSE', building: 'BUILDING_CLOSE' };
const L_OUTCOME_REASON = { excursion: 'RUNWAY_EDGE', overrun: 'CANNOT_STOP', hard: 'HIGH_SINK_RATE', short: 'UNSTABLE_APPROACH', tailstrike: 'UNSTABLE_APPROACH', crash: 'UNSTABLE_APPROACH' };
const GATE_REASON = { lateral: 'LOCALIZER_DEVIATION', loc: 'LOCALIZER_DEVIATION', vertical: 'GLIDESLOPE_DEVIATION', gs: 'GLIDESLOPE_DEVIATION', speed: 'SPEED_OUT_OF_BAND', vs: 'HIGH_SINK_RATE', gear: 'UNSTABLE_APPROACH' };
export const outcomeRisk = (family, o) => (family === 'L' ? (L_BAD.includes(o) ? 1 : 0) : family === 'D' ? (o === 'fail' ? 1 : 0) : o);
const sorted = (set) => [...set].filter(Boolean).sort((a, b) => REASONS.indexOf(a) - REASONS.indexOf(b));
const mode = (xs) => { const n = {}; let best = null; for (const x of xs) { n[x] = (n[x] || 0) + 1; if (best === null || n[x] > n[best]) best = x; } return best; };

function assemble(verdict, reasons, risk, outcome, best, x) {
  const pRef = risk.CONTINUE, pBest = Math.min(...Object.values(risk));
  const severity = verdict === 'SAFE' ? 0 : verdict === 'CAUTION' ? 1 : pBest === 0 ? 2 : pBest < pRef ? 3 : 4;
  return { verdict, severity, escapable: verdict === 'UNSAFE' ? pBest < 0.5 : null, reasons: sorted(reasons), cause: x.cause ?? null,
    action_risk: risk, action_outcome: outcome, p_ref: pRef, p_best: pBest, regret_ref: pRef - pBest,
    safe_actions: x.safe, best_action: best, ref_policy: x.ref, horizon_s: x.horizon, draws: x.draws, p_pilot: x.p_pilot ?? null,
    ttc_s: x.ttc_s ?? null, cpa_m: x.cpa_m ?? null, macro: x.macro ?? null };
}

export function corridorSafety(inp, { cNear = C_NEAR_DEFAULT } = {}) {
  const B = inp.branches, n = inp.now, risk = {}, outcome = {}, R = new Set();
  for (const a of CORRIDOR_ACTIONS) { const b = B[a]; risk[a] = b.crashes / b.k; outcome[a] = b.crashes === 0 ? 'clear' : b.crashes === b.k ? 'crash_certain' : 'crash_possible'; }
  const pRef = risk.CONTINUE, pBest = Math.min(...Object.values(risk)), nearClr = pRef === 0 && B.CONTINUE.medClr <= cNear;
  const verdict = pRef >= 0.5 || n.overstressed || n.stalled || n.pullUp ? 'UNSAFE' : pRef > 0 || nearClr || n.turbSevere ? 'CAUTION' : 'SAFE';
  if (n.overstressed) R.add('OVERSTRESS');
  if (n.stalled) R.add('STALL');
  if (n.pullUp) R.add('PULL_UP');
  if (n.turbSevere) R.add('SEVERE_TURBULENCE');
  if (n.edge) R.add('CORRIDOR_EDGE');
  if (pRef > 0) for (const c of B.CONTINUE.causes) R.add(CAUSE_REASON(c));
  if (nearClr) R.add(CLR_REASON[n.clrSource]);
  if (verdict !== 'SAFE' && n.closingFast) R.add('HAZARD_CLOSING_FAST');
  if (verdict !== 'SAFE' && n.stormCell) R.add('STORM_CELL');
  const safe = pBest >= 0.5 ? [] : CORRIDOR_ACTIONS.filter((a) => risk[a] === pBest);
  const best = pBest >= 0.5 ? 'NONE_SAFE' : safe.includes('CONTINUE') ? 'CONTINUE' : [...safe].sort((a, b) => B[b].minClr - B[a].minClr || TIE_ORDER.indexOf(a) - TIE_ORDER.indexOf(b))[0];
  return assemble(verdict, R, risk, outcome, best, { cause: pRef > 0 ? mode(B.CONTINUE.causes) : null, safe, ref: 'HANDS_OFF', horizon: 3, draws: B.CONTINUE.k,
    p_pilot: inp.p_pilot, ttc_s: inp.ttc_s, cpa_m: inp.cpa_m, macro: { pulse_steps: inp.pulse, then: 'hands_off' } });
}

export function landingSafety(inp, { eye = false } = {}) {
  const n = inp.now, o = inp.outcome, R = new Set(), risk = { CONTINUE: outcomeRisk('L', o.CONTINUE) }, outcome = { CONTINUE: o.CONTINUE };
  if (o.GO_AROUND) { risk.GO_AROUND = outcomeRisk('L', o.GO_AROUND); outcome.GO_AROUND = o.GO_AROUND; }
  const approach = n.airborne && !['GA', 'FLARE', 'ROLLOUT', 'STOP'].includes(n.vert), locSeen = !eye || n.thrNm <= EYE.locNm, gsSeen = !eye, hid = (c) => eye && EYE.hidden.includes(c);
  const seen = (k) => (k !== 'gs' || gsSeen) && (k !== 'loc' || locSeen) && !((k === 'lateral' || k === 'vertical') && hid('GATE_MODES')) && !(k === 'speed' && hid('SPEED_BAND')) && !(k === 'vs' && hid('SINK_RATE'));
  const fails = Object.entries(n.gate).filter(([k, ok]) => !ok && seen(k)).map(([k]) => k);
  const gateReasons = () => { R.add('UNSTABLE_APPROACH'); for (const k of fails) R.add(GATE_REASON[k]); };
  let unsafe = false, caution = false, edgeNow = false, stopNow = false;
  if (L_BAD.includes(o.CONTINUE)) { unsafe = true; R.add(L_OUTCOME_REASON[o.CONTINUE]); }
  if (o.CONTINUE === 'go_around') { caution = true; R.add('UNSTABLE_APPROACH'); }
  if (approach && n.crossing500 && fails.length) { unsafe = true; gateReasons(); }
  if (approach && n.vert === 'GS' && n.hRAft > 30 && n.hRAft < 500) {
    if (locSeen && Math.abs(n.locDots) > 1.5) { unsafe = true; R.add('LOCALIZER_DEVIATION'); }
    if (gsSeen && Math.abs(n.gsDots) > 1.5) { unsafe = true; R.add('GLIDESLOPE_DEVIATION'); }
  }
  const ground = n.wow && (n.vert === 'ROLLOUT' || n.vert === 'STOP');
  if (ground && !hid('RUNWAY_EDGE') && Math.abs(n.mainGearZ) > n.halfWidthM - 8 && n.lat !== 'EXIT' && n.lat !== 'CLEAR') { unsafe = edgeNow = true; R.add('RUNWAY_EDGE'); }
  if (ground && n.stopNeedM > n.runwayLeftM && !hid('CANNOT_STOP')) { unsafe = stopNow = true; R.add('CANNOT_STOP'); }
  if (approach && n.hRAft >= 500 && n.hRAft <= 1000 && fails.length) { caution = true; gateReasons(); }
  if (Math.abs(n.crossKt) >= 15) { caution = true; R.add('STRONG_CROSSWIND'); }
  if (!eye && n.tailKt >= 10) { caution = true; R.add('TAILWIND'); }
  if (n.turbSevere) { caution = true; R.add('SEVERE_TURBULENCE'); }
  const dev = [];
  if (n.airborne && n.vert === 'GS' && n.hRAft >= 50) {
    if (locSeen && Math.abs(n.locDots) > 1) { caution = true; R.add('LOCALIZER_DEVIATION'); dev.push([Math.abs(n.locDots), n.locDots > 0 ? 'TURN_LEFT' : 'TURN_RIGHT']); }
    if (gsSeen && Math.abs(n.gsDots) > 1) { caution = true; R.add('GLIDESLOPE_DEVIATION'); dev.push([Math.abs(n.gsDots), n.gsDots > 0 ? 'DESCEND' : 'CLIMB']); }
    if (!n.gate.speed && !hid('SPEED_BAND')) { caution = true; R.add('SPEED_OUT_OF_BAND'); dev.push([Math.abs(n.speedErrKt) / 5, n.speedErrKt > 0 ? 'SLOW_DOWN' : 'SPEED_UP']); }
  }
  const verdict = unsafe ? 'UNSAFE' : caution ? 'CAUTION' : 'SAFE';
  let best = 'CONTINUE';
  if (unsafe && n.airborne && n.vert !== 'GA') best = risk.GO_AROUND === 0 ? 'GO_AROUND' : 'NONE_SAFE';
  else if (unsafe && !n.airborne) best = edgeNow ? (n.mainGearZ > 0 ? 'TURN_LEFT' : 'TURN_RIGHT') : stopNow ? 'SLOW_DOWN' : 'NONE_SAFE';
  else if (n.vert === 'GS' && dev.length) best = dev.sort((a, b) => b[0] - a[0])[0][1];
  if (unsafe && best === 'CONTINUE' && risk.CONTINUE === 1) best = 'NONE_SAFE';
  return assemble(verdict, R, risk, outcome, best, { safe: best === 'NONE_SAFE' ? [] : [best], ref: 'AUTOLAND', horizon: null, draws: 1, p_pilot: risk.CONTINUE });
}

export function dockingSafety(inp, { eye = false } = {}) {
  const n = inp.now, o = inp.outcome, R = new Set(), risk = { CONTINUE: outcomeRisk('D', o.CONTINUE) }, outcome = { CONTINUE: o.CONTINUE };
  if (o.BREAKOUT) { risk.BREAKOUT = outcomeRisk('D', o.BREAKOUT); outcome.BREAKOUT = o.BREAKOUT; }
  const closingSeen = !eye || n.rho <= EYE.closingRhoM, hid = (c) => eye && EYE.hidden.includes(c);
  let unsafe = false, caution = false;
  if (o.CONTINUE === 'fail') { unsafe = true; R.add(o.failReason); }
  if (n.kosViolation) { unsafe = true; R.add('KOS_VIOLATION'); }
  if (closingSeen && n.inCorridor && n.rho <= 20 && n.closing > 1.5 * n.limit) { unsafe = true; R.add('CLOSING_TOO_FAST'); }
  if (o.CONTINUE === 'breakout') caution = true;
  const noGo = [];
  if (n.attDeg > 2 || (n.rateDps > 0.1 && !hid('ATTITUDE_RATE'))) noGo.push('ATTITUDE_ERROR');
  if (!eye && n.fuelFrac < 0.4) noGo.push('LOW_FUEL');
  if (n.holdPhase && n.outsideHoldBox) noGo.push('LATERAL_MISALIGNMENT');
  if (!eye && n.breakoutAvailable === false) noGo.push('NO_BREAKOUT_AVAILABLE');
  for (const r of noGo) { caution = true; R.add(r); }
  const tooFast = closingSeen && n.inCorridor && n.closing > n.limit;
  if (tooFast) { caution = true; R.add('CLOSING_TOO_FAST'); }
  if (!eye && n.failedJets > 0) { caution = true; R.add('JET_FAILURE'); }
  const verdict = unsafe ? 'UNSAFE' : caution ? 'CAUTION' : 'SAFE';
  const best = unsafe ? (risk.BREAKOUT === 0 ? 'BREAKOUT' : 'NONE_SAFE') : n.holdPhase && noGo.length ? 'HOLD_POSITION' : tooFast ? 'SLOW_DOWN' : 'CONTINUE';
  return assemble(verdict, R, risk, outcome, best, { safe: best === 'NONE_SAFE' ? [] : [best], ref: 'GNC', horizon: null, draws: 1, p_pilot: risk.CONTINUE, ttc_s: n.ttc_s });
}

export function eyeView(family, inp, opts = {}) {
  if (family === 'L') return landingSafety(inp, { ...opts, eye: true });
  if (family === 'D') return dockingSafety(inp, { ...opts, eye: true });
  return corridorSafety(inp, opts);
}
