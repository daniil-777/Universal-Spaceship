import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRealSim, drawRun, JET_INDEX, portRel, SHIP_PORT, STATION_PORT, axialLimit, dockingNow, dockingLabel, dockingBranches, applyDockingKick, replayDocking, cycleOf, failKind, failReason, dockingFacts } from '../vlm/gen/labels/docking.js';
import { P6_SEEDS, drawInjection, INJECT_RATE } from '../vlm/gen/inject.js';
import { EYE } from '../vlm/gen/safety.js';
import { lookAtCamera, project } from '../vlm/gen/labels/camera.js';
import { OBS } from '../vlm/gen/schema.js';
import { locMetres } from '../src/landing/nav.js';
import { AIRPORT } from '../src/landing/airport.js';
import { mulberry32 } from '../src/mathx.js';

const toAxial = (s, a) => { while (!s.rep.done && -portRel(s.x, s.q)[0] > a) s.step(); return s; };
test('the failed-P6 seed list is exactly the seeds 1..20000 whose drawRun fails P6, for every start', () => {
  const got = []; for (let s = 1; s <= 20000; s++) if (drawRun(s, { start: 'near' }).failed.includes(JET_INDEX.P6)) got.push(s);
  assert.deepEqual(got, P6_SEEDS); assert.equal(P6_SEEDS.length, 59);
  for (const s of P6_SEEDS.slice(0, 5)) for (const start of ['far', 'final']) assert.ok(drawRun(s, { start }).failed.includes(JET_INDEX.P6));
});
test('replays are bit-exact in x and q at t_sample, with a kick re-applied', () => {
  const args = { seed: 5, start: 'final', nav: 'noisy', filter: true }, live = createRealSim(args); toAxial(live, 2.0);
  const inj = { kind: 'closing_plus_0.2', params: {}, step: cycleOf(live), sim_t_s: live.t }; applyDockingKick(live, inj); for (let i = 0; i < 15; i++) live.step();
  const step = cycleOf(live), lx = Float64Array.from(live.x), lq = Float64Array.from(live.q), rep = replayDocking(args, { injection: inj, step });
  assert.ok(rep.x.every((v, i) => v === lx[i]) && rep.q.every((v, i) => v === lq[i]));
  live.run();
  const br = dockingBranches({ args, injection: inj, step, liveRep: live.rep, liveX: lx, liveQ: lq });
  assert.equal(br.CONTINUE, 'fail'); assert.equal(br.failReason, 'CLOSING_TOO_FAST');
  assert.equal(br.failKind, 'non_idss'); assert.equal(dockingLabel(rep, br).safety.cause, 'station');
});
test('injected faults: +0.2 m/s at 2 m ends non-IDSS (UNSAFE); the lateral drift is recovered (capture, never outcome-UNSAFE); KOS labels', () => {
  const s = createRealSim({ seed: 5, start: 'final' }); toAxial(s, 2.0); applyDockingKick(s, { kind: 'closing_plus_0.2', params: {} }); assert.ok(dockingNow(s).rho < 3); s.run();
  assert.equal(s.rep.result, 'fail'); assert.match(s.rep.reason, /non-IDSS/);
  assert.equal(dockingLabel(createRealSim({ seed: 5, start: 'final' }), { CONTINUE: 'fail', BREAKOUT: null, failReason: 'CLOSING_TOO_FAST' }).safety.verdict, 'UNSAFE');
  for (let seed = 1; seed <= 4; seed++) { const d = createRealSim({ seed, start: 'near' }); while (!d.rep.done && Math.hypot(...portRel(d.x, d.q)) > 30) d.step(); applyDockingKick(d, { kind: 'lateral_drift', params: { dv: 0.08 } }); d.run(); assert.equal(d.rep.result, 'capture', `seed ${seed}`); }
  const k = createRealSim({ seed: 5, start: 'final' }); while (!k.rep.done && Math.hypot(...portRel(k.x, k.q)) > 8) k.step(); k.x[1] += 3;
  const kl = dockingLabel(k, { CONTINUE: 'fail', BREAKOUT: null, failReason: 'KOS_VIOLATION' }); assert.equal(kl.safety.verdict, 'UNSAFE'); assert.ok(kl.safety.reasons.includes('KOS_VIOLATION'));
  k.run(); assert.equal(k.rep.result, 'fail'); assert.match(k.rep.reason, /KOS/);
  assert.equal(failKind(k.rep), 'kos'); assert.equal(failReason(k.rep), 'KOS_VIOLATION');
});
test('drawInjection respects the per-family rates', () => {
  const rng = mulberry32(7); for (const f of ['S', 'A', 'L', 'D']) { let n = 0; for (let i = 0; i < 4000; i++) if (drawInjection(f, rng)) n++; assert.ok(Math.abs(n / 4000 - INJECT_RATE[f]) < 0.03, `${f} ${n / 4000}`); }
});

// every judge fail reason (the sim's rep.reason strings) -> failKind, failReason and the label's cause (controller ruling:
// station only for a KOS violation, a collision, a non-IDSS contact or a keep-in exit; null for every other fail)
const JUDGE = [
  ['KOS violation in H2', 'kos', 'KOS_VIOLATION', 'station'], ['collision', 'collision', 'LATERAL_MISALIGNMENT', 'station'],
  ['non-IDSS contact', 'non_idss', 'CLOSING_TOO_FAST', 'station'], ['keep-in exit', 'keep_in', null, 'station'],
  ['propellant exhausted', 'propellant', 'LOW_FUEL', null], ['timeout', 'timeout', null, null], ['Rule P after DEPART: 212.4 m', 'rule_p', 'NO_BREAKOUT_AVAILABLE', null],
  ['breakout not passively safe', 'breakout_unsafe', 'NO_BREAKOUT_AVAILABLE', null], ['still in the KOS one orbit after the breakout', 'breakout_kos', 'NO_BREAKOUT_AVAILABLE', null]];
for (const [why, kind, reason, cause] of JUDGE) {
  test(`fail "${why}": failKind ${kind}, failReason ${reason}, cause ${cause}`, () => {
    const rep = { result: 'fail', reason: why, contact: kind === 'non_idss' ? { close: 0.101, lat: 0.01, mis: 0.02, rate: 0.01, ang: 0.5 } : null };
    assert.equal(failKind(rep), kind); assert.equal(failReason(rep), reason);
    const s = dockingLabel(createRealSim({ seed: 5, start: 'final' }), { CONTINUE: 'fail', BREAKOUT: null, failReason: failReason(rep), failKind: failKind(rep) });
    assert.equal(s.safety.verdict, 'UNSAFE'); assert.equal(s.safety.cause, cause); assert.equal(s.safety_eye.cause, cause);
  });
}
test('non-IDSS contact reasons split by the contact; capture and breakout have no failKind', () => {
  const c = (o) => failReason({ result: 'fail', reason: 'non-IDSS contact', contact: { close: 0.07, lat: 0.01, mis: 0.02, rate: 0.01, ang: 0.5, ...o } });
  assert.equal(c({ close: 0.04 }), 'CLOSING_TOO_FAST'); assert.equal(c({ mis: 0.11 }), 'LATERAL_MISALIGNMENT'); assert.equal(c({ lat: 0.05 }), 'LATERAL_MISALIGNMENT'); assert.equal(c({ ang: 5 }), 'ATTITUDE_ERROR');
  for (const result of ['capture', 'breakout']) assert.equal(failKind({ result, reason: null }), null);
});

// dockingFacts (review focus 1): every fact is {v, unit, obs} and every number is finite, in TRANSFER (passiveMin) and near the port
const numbersOk = (v) => (typeof v === 'number' ? Number.isFinite(v) : Array.isArray(v) ? v.every(numbersOk) : v && typeof v === 'object' ? Object.values(v).every(numbersOk) : true);
test('dockingFacts: {v, unit, obs} with finite numbers in TRANSFER and in FINAL; drift only in TRANSFER', () => {
  const far = createRealSim({ seed: 2, start: 'far' }), fin = toAxial(createRealSim({ seed: 5, start: 'final' }), 1.0);
  for (const [sim, phase] of [[far, 'TRANSFER'], [fin, 'FINAL']]) {
    const F = dockingFacts(sim, { space: { sunLit: 1, earthInFrame: true } });
    assert.equal(F.phase.v, phase); assert.equal(F.phase.obs, 'visual'); assert.equal(F['sun.lit'].obs, 'visual');
    for (const [id, f] of Object.entries(F)) { assert.ok(f && 'v' in f && 'unit' in f && OBS.includes(f.obs), id); assert.ok(numbersOk(f.v), `${id} = ${JSON.stringify(f.v)}`); }
    assert.equal(F['drift24h_min_m'].v === null, phase !== 'TRANSFER');
  }
  assert.equal(typeof dockingFacts(fin).ttc_s.v, 'number'); assert.equal(dockingFacts(far).ttc_s.v, null);
});

// safety_eye sensitivity (spec §4.3): every kept L/D criterion moves the 160x96 image by >= 1 px at its scope boundary
const W = 896, H = 504, SX = 160 / W, SY = 96 / H, DEG = Math.PI / 180, KT = 0.514444;
const dpx = (a, b) => Math.hypot((a.x - b.x) * SX, (a.y - b.y) * SY);
const lcam = (P) => lookAtCamera({ eye: [P[0] - 78, P[1] + 16, P[2]], target: [P[0] + 30, P[1], P[2]], fovDeg: 50, aspect: W / H, near: 0.5, far: 150000 });
const gsH = (x) => Math.hypot(x - AIRPORT.ils.gs.x, AIRPORT.ils.gs.z) * Math.tan(3 * DEG);
const dcam = (rho, yawDeg = 0, dy = 0) => { const e = [STATION_PORT[0] - rho, dy, 0]; return lookAtCamera({ eye: e, target: [e[0] + 100 * Math.cos(yawDeg * DEG), e[1], 100 * Math.sin(yawDeg * DEG)], fovDeg: 45, aspect: W / H, near: 0.05, far: 20000 }); };
const size5 = (cam) => dpx(project(cam, STATION_PORT, W, H), project(cam, [STATION_PORT[0], 5, 0], W, H));
export function sensitivities() {
  const s = {}, thr = [0, 0, 0], x1 = -1852 * EYE.locNm, x500 = -(500 * 0.3048) / Math.tan(3 * DEG);
  s.LOC_DOTS = dpx(project(lcam([x1, gsH(x1), 0]), thr, W, H), project(lcam([x1, gsH(x1), locMetres(1, x1)]), thr, W, H));
  s.GS_DOTS = dpx(project(lcam([x1, gsH(x1), 0]), thr, W, H), project(lcam([x1, gsH(x1) + Math.tan(AIRPORT.ils.gs.dot) * (AIRPORT.ils.gs.x - x1), 0]), thr, W, H));
  const P = [x500, gsH(x500), 0], crab = Math.asin(15 / 140);
  s.CROSSWIND = dpx(project(lcam(P), [P[0] + 13.5, P[1], 0], W, H), project(lcam(P), [P[0] + 13.5 * Math.cos(crab), P[1], 13.5 * Math.sin(crab)], W, H));
  s.SPEED_BAND = dpx(project(lcam(P), thr, W, H), project(lcam([P[0] + 5 * KT * 0.25, P[1], 0]), thr, W, H));
  s.SINK_RATE = dpx(project(lcam(P), thr, W, H), project(lcam([P[0], P[1] - 5.08 * 0.25, 0]), thr, W, H));
  const G = [2000, 5, 0], end = [4000, 0, 0];
  s.CANNOT_STOP = dpx(project(lcam(G), end, W, H), project(lcam([G[0] + 5 * KT * 0.25, 5, 0]), end, W, H));
  s.RUNWAY_EDGE = dpx(project(lcam(G), [G[0] + 60, 0, 30], W, H), project(lcam([G[0], 5, 1]), [G[0] + 60, 0, 30], W, H));
  const r = EYE.closingRhoM; s.D_CLOSING = Math.abs(size5(dcam(r)) - size5(dcam(r - 2 * axialLimit(r))));
  s.ATTITUDE_2DEG = dpx(project(dcam(r), STATION_PORT, W, H), project(dcam(r, 2), STATION_PORT, W, H));
  s.ATTITUDE_RATE = dpx(project(dcam(r), STATION_PORT, W, H), project(dcam(r, 0.2), STATION_PORT, W, H));
  s.HOLD_BOX_H2 = dpx(project(dcam(20), STATION_PORT, W, H), project(dcam(20, 0, 0.5), STATION_PORT, W, H));
  return s;
}
test('safety_eye: kept criteria >= 1 px at their scope boundary, hidden ones < 1 px (the result sets EYE in safety.js)', () => {
  const s = sensitivities(), hiddenByPx = Object.entries(s).filter(([, v]) => v < 1).map(([k]) => k).sort();
  console.log('eye sensitivity px', JSON.stringify(Object.fromEntries(Object.entries(s).map(([k, v]) => [k, +v.toFixed(2)]))));
  const expectHidden = ['GS_DOTS', ...EYE.hidden.filter((h) => h !== 'GATE_MODES')].sort();
  assert.deepEqual(hiddenByPx, expectHidden);
  assert.ok(EYE.hidden.includes('GATE_MODES'), 'autoland mode flags (LOC/GS captured) are never visible');
});
