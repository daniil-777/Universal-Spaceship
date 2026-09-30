import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRealSim, drawRun, JET_INDEX, portRel, SHIP_PORT, STATION_PORT, axialLimit, GUID, H1_POINT, CORRIDOR_HALF, CONE_FLOOR, dockingNow, dockingLabel, dockingBranches, applyDockingKick, replayDocking, cycleOf, failKind, failReason, dockingFacts, breakoutAvailable, isDroppedDockingOutcome } from '../vlm/gen/labels/docking.js';
import { P6_SEEDS, drawInjection, INJECT_RATE, INJECTIONS } from '../vlm/gen/inject.js';
import { EYE } from '../vlm/gen/safety.js';
import { lookAtCamera, project } from '../vlm/gen/labels/camera.js';
import { OBS } from '../vlm/gen/schema.js';
import { locMetres } from '../src/landing/nav.js';
import { AIRPORT } from '../src/landing/airport.js';
import { createLandingSim, drawConditions } from '../src/landing/sim.js';
import { bodyToWorld } from '../src/landing/flight.js';
import { FT, VEH } from '../src/landing/vehicle.js';
import { mulberry32 } from '../src/mathx.js';

const toAxial = (s, a) => { while (!s.rep.done && -portRel(s.x, s.q)[0] > a) s.step(); return s; };
const rho = (s) => Math.hypot(...portRel(s.x, s.q));
// kick a live run, step `lead` cycles (the D timing rule needs 2 x 20 + 1), keep x and q, run to the end
function kicked(args, until, inj, lead = 41) {
  const live = createRealSim(args); while (!live.rep.done && !until(live)) live.step();
  const injection = { ...inj, step: cycleOf(live), sim_t_s: live.t }; applyDockingKick(live, injection);
  for (let i = 0; i < lead; i++) live.step();
  const step = cycleOf(live), liveX = Float64Array.from(live.x), liveQ = Float64Array.from(live.q), now = dockingNow(live), t = live.t;
  return { args, live, injection, step, liveX, liveQ, now, t };
}
const labelOf = (k) => { const br = dockingBranches({ args: k.args, injection: k.injection, step: k.step, liveRep: k.live.rep, liveX: k.liveX, liveQ: k.liveQ }); return { br, ...dockingLabel(replayDocking(k.args, { injection: k.injection, step: k.step }), br) }; };

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
  const bad = Float64Array.from(lx); bad[1] += 1e-12;
  assert.throws(() => dockingBranches({ args, injection: inj, step, liveRep: live.rep, liveX: bad, liveQ: lq }), /bit-exact/);
  assert.throws(() => dockingBranches({ args, injection: inj, step: inj.step, liveRep: live.rep }), /sample step/);
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
test('catalogue rulings: attitude_kick at rho >= 20 m, two lateral_drift bands, inbound 1.2-1.6 m/s at 300-400 m, alt_plus_60 at 550-700 ft', () => {
  const rng = mulberry32(11), seen = {};
  for (let i = 0; i < 20000; i++) {
    const d = drawInjection('D', rng); if (!d) continue;
    const key = d.kind + (d.params.band ? `:${d.params.band}` : ''), v = d.trigger && d.trigger.value; seen[key] = (seen[key] || 0) + 1;
    if (d.kind === 'attitude_kick') assert.ok(v >= 20 && v <= 40, `attitude_kick at ${v}`);
    if (d.kind === 'inbound') assert.ok(v >= 300 && v <= 400 && d.params.dv >= 1.2 && d.params.dv <= 1.6 && d.trigger.phase === null);
    if (key === 'lateral_drift:recovered') assert.ok(v >= 20 && v <= 40 && d.params.dv >= 0.05 && d.params.dv <= 0.1 && d.trigger.phase === null);
    if (key === 'lateral_drift:kos') assert.ok(v >= 20 && v <= 25 && d.params.dv === 0.5 && d.trigger.phase === 'CORRIDOR');
  }
  assert.deepEqual(Object.keys(seen).sort(), ['abort', 'attitude_kick', 'closing_plus_0.2', 'failed_p6', 'inbound', 'lateral_drift:kos', 'lateral_drift:recovered', 'radial_plus_0.08']);
  assert.ok(!INJECTIONS.D.some((e) => e.kind === 'inbound_0.3'));
  for (let i = 0; i < 4000; i++) { const d = drawInjection('L', rng); if (d && d.kind === 'alt_plus_60') assert.ok(d.trigger.value >= 550 && d.trigger.value <= 700); }
});
test('T4-b: a 0.5 m/s drift at 22 m in CORRIDOR ends in a KOS fail with room for a sample (UNSAFE); a KOS now-violation ends the run in its own step', () => {
  const k = kicked({ seed: 1, start: 'near' }, (s) => s.guid.st.phase === 'CORRIDOR' && rho(s) <= 22, { kind: 'lateral_drift', params: { dv: 0.5, band: 'kos' } });
  while (!k.live.rep.done) { k.live.step(); if (dockingNow(k.live).kosViolation) assert.ok(k.live.rep.done && failKind(k.live.rep) === 'kos', 'the judge ends a run at a KOS violation'); }
  assert.equal(failKind(k.live.rep), 'kos'); assert.ok(k.live.rep.t - k.t >= 1, `sample ${(k.live.rep.t - k.t).toFixed(1)} s before the end`);
  const s = labelOf(k).safety; assert.equal(s.verdict, 'UNSAFE'); assert.equal(s.cause, 'station'); assert.ok(s.reasons.includes('KOS_VIOLATION'));
});
test('T4-c: inbound 1.4 m/s at 350 m exceeds the speed limit: the label is non-SAFE (CLOSING_TOO_FAST) and the GNC recovers', () => {
  const k = kicked({ seed: 1, start: 'far' }, (s) => rho(s) <= 350, { kind: 'inbound', params: { dv: 1.4 } });
  assert.ok(!k.now.inCorridor && k.now.closing > k.now.limit, `closing ${k.now.closing} limit ${k.now.limit}`);
  k.live.run(); assert.equal(k.live.rep.result, 'capture');
  const x = labelOf(k); assert.notEqual(x.safety.verdict, 'SAFE'); assert.ok(x.safety.reasons.includes('CLOSING_TOO_FAST')); assert.equal(x.safety.best_action, 'SLOW_DOWN');
  assert.ok(!x.safety_eye.reasons.includes('CLOSING_TOO_FAST'), 'beyond the eye closing scope');
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
test('T4-d: isDroppedDockingOutcome drops a continued run that ends fail:timeout or fail:keep_in, nothing else', () => {
  for (const [, kind] of JUDGE) assert.equal(isDroppedDockingOutcome({ CONTINUE: 'fail', BREAKOUT: null, failReason: null, failKind: kind }), kind === 'timeout' || kind === 'keep_in', kind);
  for (const r of ['capture', 'breakout']) assert.equal(isDroppedDockingOutcome({ CONTINUE: r, BREAKOUT: 'fail', failReason: null, failKind: null }), false);
});
test('a D replay branch that ends in timeout is a discard (BREAKOUT, and CONTINUE before a later injection)', () => {
  const args = { seed: 5, start: 'final', tMax: 60 }, live = createRealSim(args); for (let i = 0; i < 100; i++) live.step();
  const liveX = Float64Array.from(live.x), liveQ = Float64Array.from(live.q); live.run();
  assert.equal(failKind(live.rep), 'timeout'); assert.equal(isDroppedDockingOutcome({ CONTINUE: live.rep.result, failKind: failKind(live.rep) }), true);
  assert.throws(() => dockingBranches({ args, step: 100, liveRep: live.rep, liveX, liveQ }), (e) => e.discard === true && /BREAKOUT/.test(e.message));
  const later = { kind: 'abort', params: {}, step: 300, sim_t_s: 30 };
  assert.throws(() => dockingBranches({ args, injection: later, step: 100, liveRep: { result: 'breakout', reason: null }, liveX, liveQ }), (e) => e.discard === true && /CONTINUE/.test(e.message));
});
test('breakout availability mirrors the guidance Rule A (inside = a corridor phase or r <= KOS_R), step for step with truth nav', () => {
  const s = createRealSim({ seed: 3, start: 'near', nav: 'truth' }); let n = 0, outside = 0;
  while (!s.rep.done && s.t < 2500) {
    const x0 = Float64Array.from(s.x), ph0 = s.guid.st.phase, bo0 = s.guid.st.boT; s.step();
    if (s.guid.st.boT !== bo0 && s.guid.st.phase === ph0) { n++; if (Math.hypot(x0[0], x0[1], x0[2]) > 200 && !['CORRIDOR', 'H2', 'FINAL'].includes(ph0)) outside++; assert.equal(breakoutAvailable(x0, ph0), s.guid.st.boAvail, `t ${s.t}`); }
  }
  assert.ok(n > 50 && outside > 10, `${n} evaluations, ${outside} outside the KOS`); assert.equal(breakoutAvailable(s.x, 'TRANSFER'), null);
});

// dockingFacts (review focus 1): every fact is {v, unit, obs} and every number is finite, in TRANSFER (passiveMin) and near the port
const numbersOk = (v) => (typeof v === 'number' ? Number.isFinite(v) : Array.isArray(v) ? v.every(numbersOk) : v && typeof v === 'object' ? Object.values(v).every(numbersOk) : true);
test('dockingFacts: {v, unit, obs} with finite numbers in TRANSFER and in FINAL; drift only in TRANSFER; corridor and speed limits; unknown visuals stay null', () => {
  const far = createRealSim({ seed: 2, start: 'far' }), fin = toAxial(createRealSim({ seed: 5, start: 'final' }), 1.0);
  for (const [sim, phase] of [[far, 'TRANSFER'], [fin, 'FINAL']]) {
    const F = dockingFacts(sim, { space: { sunLit: 1, earthInFrame: true } });
    assert.equal(F.phase.v, phase); assert.equal(F.phase.obs, 'visual'); assert.equal(F['sun.lit'].obs, 'visual'); assert.equal(F.earth_in_frame.v, true);
    for (const [id, f] of Object.entries(F)) { assert.ok(f && 'v' in f && 'unit' in f && OBS.includes(f.obs), id); assert.ok(numbersOk(f.v), `${id} = ${JSON.stringify(f.v)}`); }
    assert.equal(F['drift24h_min_m'].v === null, phase !== 'TRANSFER'); assert.equal(F.corridor_limit_cms.v === null, phase !== 'FINAL'); assert.equal(typeof F.speed_limit_cms.v, 'number');
    assert.ok(!('limit_cms' in F));
  }
  assert.equal(dockingFacts(fin).corridor_limit_cms.v, +(axialLimit(Math.max(dockingNow(fin).axial, 0)) * 100).toFixed(3));
  assert.equal(typeof dockingFacts(fin).ttc_s.v, 'number'); assert.equal(dockingFacts(far).ttc_s.v, null);
  const u = dockingFacts(fin, { space: { sunLit: null, earthInFrame: null } }); assert.equal(u.earth_in_frame.v, null); assert.equal(u['sun.lit'].v, null);
  assert.equal(dockingFacts(fin, { space: { sunLit: 0, earthInFrame: false } }).earth_in_frame.v, false);
});

// safety_eye sensitivity (spec §4.3): every kept L/D criterion moves the 160x96 image by >= 1 px at its scope boundary
const W = 896, H = 504, SX = 160 / W, SY = 96 / H, DEG = Math.PI / 180, KT = 0.514444;
const dpx = (a, b) => Math.hypot((a.x - b.x) * SX, (a.y - b.y) * SY);
const lcam = (P) => lookAtCamera({ eye: [P[0] - 78, P[1] + 16, P[2]], target: [P[0] + 30, P[1], P[2]], fovDeg: 50, aspect: W / H, near: 0.5, far: 150000 });
const gsH = (x) => Math.hypot(x - AIRPORT.ils.gs.x, AIRPORT.ils.gs.z) * Math.tan(3 * DEG);
const dcam = (rho, yawDeg = 0, dy = 0) => { const e = [STATION_PORT[0] - rho, dy, 0]; return lookAtCamera({ eye: e, target: [e[0] + 100 * Math.cos(yawDeg * DEG), e[1], 100 * Math.sin(yawDeg * DEG)], fovDeg: 45, aspect: W / H, near: 0.05, far: 20000 }); };
const size5 = (cam) => dpx(project(cam, STATION_PORT, W, H), project(cam, [STATION_PORT[0], 5, 0], W, H));
const dClosing = (r) => Math.abs(size5(dcam(r)) - size5(dcam(r - 2 * axialLimit(r))));
// turbulence: RMS over 0.25 s frame pairs (1500 -> 500 ft) of the largest image motion of the wingtips, nose and tail, camera locked to the CoM
function jitterPx(seed, turb) {
  const c = drawConditions(seed, { final: true }); c.wind = { ...c.wind, turb };
  const s = createLandingSim(c), pts = [[0, 0, VEH.span / 2], [0, 0, -VEH.span / 2], [13.5, 0, 0], [-17.1, 0, 0]]; let prev = null, sum = 0, n = 0;
  while (!s.rep.done && s.flight.air.hRA / FT >= 500) {
    if (Math.round(s.flight.t * 120) % 30 === 0 && s.flight.air.hRA / FT <= 1500) {
      const cam = lcam(Array.from(s.flight.p)), px = pts.map((b) => project(cam, bodyToWorld(s.flight, b), W, H));
      if (prev) { const m = Math.max(...px.map((q, i) => dpx(q, prev[i]))); sum += m * m; n++; }
      prev = px;
    }
    s.step();
  }
  return Math.sqrt(sum / n);
}
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
  s.SEVERE_TURBULENCE = [1, 2].reduce((a, seed) => a + (jitterPx(seed, 'severe') - jitterPx(seed, 'moderate')) / 2, 0);
  const r = EYE.closingRhoM; s.D_CLOSING = dClosing(r);
  s.ATTITUDE_2DEG = dpx(project(dcam(r), STATION_PORT, W, H), project(dcam(r, 2), STATION_PORT, W, H));
  s.ATTITUDE_RATE = dpx(project(dcam(r), STATION_PORT, W, H), project(dcam(r, 0.2), STATION_PORT, W, H));
  s.HOLD_BOX_H2 = dpx(project(dcam(20), STATION_PORT, W, H), project(dcam(20, 0, 0.5), STATION_PORT, W, H));
  const rH1 = -H1_POINT[0] - SHIP_PORT[0] + STATION_PORT[0];
  s.HOLD_BOX_H1 = dpx(project(dcam(rH1), STATION_PORT, W, H), project(dcam(rH1, 0, GUID.boxH1), STATION_PORT, W, H));
  // the KOS now-rule leaves the cone (10 deg, 1 m floor) at the far end of the corridor inside the KOS (r = 200 m); its radial
  // KOS-sphere part (0.03 px) never reaches a kept sample: the judge ends the run in the step of the violation (T4-b test)
  const aK = 200 - SHIP_PORT[0] + STATION_PORT[0];
  s.KOS_CONE = dpx(project(dcam(aK), STATION_PORT, W, H), project(dcam(aK, 0, Math.max(Math.tan(CORRIDOR_HALF) * aK, CONE_FLOOR)), STATION_PORT, W, H));
  return s;
}
test('safety_eye: kept criteria >= 1 px at their scope boundary, hidden ones < 1 px (the result sets EYE in safety.js)', () => {
  const s = sensitivities(), hiddenByPx = Object.entries(s).filter(([, v]) => v < 1).map(([k]) => k).sort();
  console.log('eye sensitivity px', JSON.stringify(Object.fromEntries(Object.entries(s).map(([k, v]) => [k, +v.toFixed(2)]))));
  const expectHidden = ['GS_DOTS', ...EYE.hidden.filter((h) => h !== 'GATE_MODES')].sort();
  assert.deepEqual(hiddenByPx, expectHidden);
  assert.ok(EYE.hidden.includes('GATE_MODES'), 'autoland mode flags (LOC/GS captured) are never visible');
  assert.ok(dClosing(EYE.closingRhoM + 0.5) < 1, `closingRhoM is the largest passing 0.5 m step (${dClosing(EYE.closingRhoM + 0.5).toFixed(3)} px at +0.5 m)`);
  for (const k of ['SEVERE_TURBULENCE', 'KOS_CONE', 'HOLD_BOX_H1']) assert.ok(k in s, k);
});
