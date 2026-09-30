// vlm/gen/labels/docking.js — docking (D) labels (spec §4.2 D with the H1 fix, §4.3 docking, §7.3 D): dockingNow() for the
// combinator, dockingFacts(), kicks on sim.x/q/w/abort() (never R.failed or ctrl.jets.isFailed, §3.3), replays with
// identical constructor arguments that must match the live sim bit for bit. Tests reach the real sim only through here
// (tests/real_isolation.test.mjs forbids the path string in tests/).
import { createRealSim, drawRun } from '../../../src/real/sim.js';
import { JET_INDEX, JETS, KOS_R, IDSS, H1_POINT, H2_RHO, MASS0, PROP0, SHIP_PORT, STATION_PORT, CORRIDOR_HALF, CONE_FLOOR, portRel, inCone } from '../../../src/real/consts.js';
import { axialLimit, speedLimit } from '../../../src/real/safety.js';
import { passiveMin, planBreakout } from '../../../src/real/passive.js';
import { PH, CORRIDOR_PHASES, RULE_A_PHASES, GUID } from '../../../src/real/guidance.js';
import { attError, omegaRel } from '../../../src/real/rigid.js';
import { qMul, qFromAxisAngle } from '../../../src/mathx.js';
import { dockingSafety, eyeView } from '../safety.js';
import { fact, Discard } from '../schema.js';

export { createRealSim, drawRun, JET_INDEX, portRel, SHIP_PORT, STATION_PORT, axialLimit, GUID, H1_POINT, CORRIDOR_HALF, CONE_FLOOR };
const DEG = Math.PI / 180, r3 = (x) => (Number.isFinite(x) ? +x.toFixed(3) : null);
export const cycleOf = (sim) => Math.round(sim.t / 0.1);
const sameBits = (a, b) => a.every((v, i) => Object.is(v, b[i]));
// Rule A as the guidance evaluates it (guidance.js:54, 92): planBreakout with inside = a corridor phase or r <= KOS_R; null
// outside the Rule-A phases
export function breakoutAvailable(x, phase) {
  if (!RULE_A_PHASES.includes(phase)) return null;
  return !!planBreakout(Float64Array.from(x), { inside: CORRIDOR_PHASES.includes(phase) || Math.hypot(x[0], x[1], x[2]) <= KOS_R });
}
const distBin = (rho) => (rho < 2 ? '<2 m' : rho < 20 ? '2-20 m' : rho < 100 ? '20-100 m' : rho < 400 ? '100-400 m' : '>400 m');

export function dockingNow(sim) {
  const x = sim.x, q = sim.q, ph = sim.guid.st.phase, r = Math.hypot(x[0], x[1], x[2]), p = portRel(x, q), axial = -p[0], rho = Math.hypot(p[0], p[1], p[2]);
  const inCorridor = CORRIDOR_PHASES.includes(ph) && inCone(p), e = attError(q, [0, 0, 0, 1]), we = omegaRel(q, sim.w), inward = -(x[0] * x[3] + x[1] * x[4] + x[2] * x[5]) / r;
  const box = ph === PH.H1 ? Math.max(...[0, 1, 2].map((i) => Math.abs(x[i] - H1_POINT[i]))) > GUID.boxH1 : ph === PH.H2 ? Math.max(Math.abs(axial - H2_RHO), Math.abs(p[1]), Math.abs(p[2])) > GUID.boxH2 : false;
  return { phase: ph, rho, axial, closing: inCorridor ? x[3] : inward, limit: inCorridor ? axialLimit(Math.max(axial, 0)) : speedLimit(r), inCorridor,
    kosViolation: r < KOS_R && !inCorridor && ph !== PH.BREAKOUT && ph !== PH.DEPART, attDeg: Math.hypot(...e) / DEG, rateDps: Math.hypot(...we) / DEG,
    fuelFrac: (sim.mass - (MASS0 - PROP0)) / PROP0, failedJets: sim.R.failed.length, breakoutAvailable: breakoutAvailable(x, ph),
    holdPhase: ph === PH.H1 || ph === PH.H2, outsideHoldBox: box, ttc_s: inCorridor && x[3] > 1e-4 ? axial / x[3] : null };
}

export function dockingFacts(sim, { space = null } = {}) {
  const F = {}, put = (id, v, unit, obs) => { F[id] = fact(v, unit, obs); }, n = dockingNow(sim), x = sim.x, p = portRel(x, sim.q), st = sim.guid.st;
  put('phase', n.phase, null, 'visual'); put('station_distance_bin', distBin(n.rho), null, 'visual');
  put('r_m', r3(Math.hypot(x[0], x[1], x[2])), 'm', 'context'); put('rho_m', r3(n.rho), 'm', 'context'); put('axial_m', r3(n.axial), 'm', 'context'); put('lateral_m', r3(Math.hypot(p[1], p[2])), 'm', 'context');
  put('in_cone', inCone(p), null, 'context'); put('in_kos', Math.hypot(x[0], x[1], x[2]) < KOS_R, null, 'context');
  put('closing_cms', r3(x[3] * 100), 'cm/s', 'context'); put('lateral_cms', r3(Math.hypot(x[4], x[5]) * 100), 'cm/s', 'context'); put('inward_cms', r3(n.closing * 100), 'cm/s', 'context');
  put('ttc_s', n.ttc_s === null ? null : r3(n.ttc_s), 's', 'context');
  put('corridor_limit_cms', CORRIDOR_PHASES.includes(n.phase) ? r3(axialLimit(Math.max(n.axial, 0)) * 100) : null, 'cm/s', 'context'); put('speed_limit_cms', r3(speedLimit(Math.hypot(x[0], x[1], x[2])) * 100), 'cm/s', 'context');
  put('att_err_deg', r3(n.attDeg), 'deg', 'context'); put('att_rate_dps', r3(n.rateDps), 'deg/s', 'context');
  put('jets_firing', JETS.filter((_, i) => sim.onTimes[i] > 0).map((j) => j.name), null, 'context'); put('jets_failed', sim.R.failed.map((i) => JETS[i].name), null, 'context');
  put('fuel_frac', r3(n.fuelFrac), null, 'context'); put('dv_used_ms', r3(sim.rep.dv), 'm/s', 'context'); put('breakout_available', n.breakoutAvailable, null, 'context');
  put('drift24h_min_m', n.phase === PH.TRANSFER ? r3(passiveMin(Float64Array.from(x)).rMin) : null, 'm', 'context');
  put('burn', st.burn ? st.burn.kind : null, null, 'context');
  if (space) { put('sun.lit', r3(space.sunLit), null, 'visual'); put('earth_in_frame', space.earthInFrame ?? null, null, 'visual'); }
  return F;
}

// the judge's fail reasons (sim.js judge, contact, judgeEvents) as a closed set; safety.js D_CAUSE maps kos, collision,
// non_idss and keep_in to cause 'station' (controller ruling). "still in the KOS ... after the breakout" is a breakout
// that did not leave the KOS, not a KOS violation (the judge exempts DEPART), so it is breakout_kos.
const FAIL_PATTERNS = [[/^KOS violation/, 'kos'], [/^collision/, 'collision'], [/^non-IDSS contact/, 'non_idss'], [/^keep-in exit/, 'keep_in'], [/^propellant exhausted/, 'propellant'],
  [/^timeout/, 'timeout'], [/^Rule P/, 'rule_p'], [/^breakout not passively safe/, 'breakout_unsafe'], [/^still in the KOS/, 'breakout_kos']];
export function failKind(rep) {
  if (rep.result !== 'fail') return null;
  const why = rep.reason || '', hit = FAIL_PATTERNS.find(([re]) => re.test(why));
  return hit ? hit[1] : 'other';
}
const KIND_REASON = { kos: 'KOS_VIOLATION', collision: 'LATERAL_MISALIGNMENT', propellant: 'LOW_FUEL', rule_p: 'NO_BREAKOUT_AVAILABLE', breakout_unsafe: 'NO_BREAKOUT_AVAILABLE', breakout_kos: 'NO_BREAKOUT_AVAILABLE' };
export function failReason(rep) {
  const k = failKind(rep), c = rep.contact || {};
  if (k === 'non_idss') return c.close < IDSS.closeMin || c.close > IDSS.closeMax ? 'CLOSING_TOO_FAST' : c.mis > IDSS.mis || c.lat > IDSS.lat ? 'LATERAL_MISALIGNMENT' : 'ATTITUDE_ERROR';
  return KIND_REASON[k] ?? null;
}
export function applyDockingKick(sim, inj) {
  const x = sim.x, p = inj.params || {};
  if (inj.kind === 'closing_plus_0.2') x[3] += 0.2;
  else if (inj.kind === 'radial_plus_0.08') x[4] += 0.08;
  else if (inj.kind === 'lateral_drift') x[4] += p.dv;
  else if (inj.kind === 'inbound') { const r = Math.hypot(x[0], x[1], x[2]), u = [x[0] / r, x[1] / r, x[2] / r], vr = x[3] * u[0] + x[4] * u[1] + x[5] * u[2]; for (let i = 0; i < 3; i++) x[3 + i] += (-p.dv - vr) * u[i]; }
  else if (inj.kind === 'abort') sim.abort('injected');
  else if (inj.kind === 'attitude_kick') { sim.q.set(qMul(sim.q, qFromAxisAngle(p.axis[0], p.axis[1], p.axis[2], 10 * DEG))); for (let i = 0; i < 3; i++) sim.w[i] += p.axis[i] * 0.2 * DEG; }
  else throw new Error(`not a runtime D injection: ${inj.kind}`);
}
// replay to `step`; with the live x and q at that step (liveX, liveQ) it must match them bit for bit
export function replayDocking(args, { injection = null, step, liveX = null, liveQ = null }) {
  const sim = createRealSim(args);
  for (let n = 0; n < step && !sim.rep.done; n++) { if (injection && injection.step === n) applyDockingKick(sim, injection); sim.step(); }
  if (liveX && !(sameBits(sim.x, liveX) && sameBits(sim.q, liveQ))) throw new Error(`docking replay is not bit-exact at t_sample (cycle ${step})`);
  return sim;
}
// branches at the sample cycle `step`; a replayed branch that ends in timeout is a Discard. The live CONTINUE outcome is
// returned as is: isDroppedDockingOutcome() tells the capture to drop fail:timeout and fail:keep_in (controller ruling T4-d)
export function dockingBranches({ args, injection = null, step, liveRep, liveX = null, liveQ = null }) {
  const rt = injection && injection.step !== null && injection.step !== undefined ? injection : null;
  if (rt && rt.step === step) throw new Error(`injection ${rt.kind} at the sample step ${step}: the timing rule forbids it`);
  const at = replayDocking(args, { injection: rt && rt.step < step ? rt : null, step, liveX, liveQ });
  let cont = liveRep;
  if (rt && rt.step > step) { const s = replayDocking(args, { step, liveX, liveQ }); s.run(); cont = s.rep; if (failKind(cont) === 'timeout') throw new Discard(`D CONTINUE replay ends in timeout (cycle ${step})`); }
  let BREAKOUT = null;
  if (breakoutAvailable(at.x, at.guid.st.phase)) { at.abort('branch'); at.run(); if (failKind(at.rep) === 'timeout') throw new Discard(`D BREAKOUT replay ends in timeout (cycle ${step})`); BREAKOUT = at.rep.result; }
  return { CONTINUE: cont.result, BREAKOUT, failReason: cont.result === 'fail' ? failReason(cont) : null, failKind: failKind(cont) };
}
export const isDroppedDockingOutcome = (b) => b.CONTINUE === 'fail' && (b.failKind === 'timeout' || b.failKind === 'keep_in');
export function dockingLabel(sim, branches) {
  const input = { now: dockingNow(sim), outcome: branches };
  return { safety: dockingSafety(input), safety_eye: eyeView('D', input), input };
}
