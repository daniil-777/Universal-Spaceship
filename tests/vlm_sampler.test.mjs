import { test } from 'node:test';
import assert from 'node:assert/strict';
import { planEpisode, aCells, SKIES, landingSchedule, dockingSchedule, rngOf } from '../vlm/gen/sampler.js';
import { frameStats, saneFrame, straddles, keepProb } from '../vlm/capture/probe/frame.js';
import { P6_SEEDS, drawInjection } from '../vlm/gen/inject.js';
import { pickCell, DEAD_TRIES } from '../vlm/capture/families.mjs';
import { armLanding, landingTriggerStep, landingTrig, landingOracle } from '../vlm/capture/probe/landing.js';
import { armDocking, dockingTriggerStep, dockingTrig, stepArmed, centrelinePose, dockingOracle } from '../vlm/capture/probe/docking.js';
import { createLandingSim, drawConditions } from '../src/landing/sim.js';
import { replayLanding, stepOf } from '../vlm/gen/labels/landing.js';
import { createRealSim, replayDocking, cycleOf, SHIP_PORT, portRel } from '../vlm/gen/labels/docking.js';

const q = (u) => new URLSearchParams(u.split('?')[1]);
test('L never uses seed 0, always view=chase and path=0; S uses lowpass=0; every URL carries rs', () => {
  for (let s = 0; s < 300; s++) { const e = planEpisode('L', s, { run: 'r', rs: 7 }), p = q(e.url); assert.ok(+p.get('seed') >= 1); assert.equal(p.get('view'), 'chase'); assert.equal(p.get('path'), '0'); assert.equal(p.get('rs'), '7'); }
  assert.equal(q(planEpisode('S', 3, { run: 'r', rs: 1 }).url).get('lowpass'), '0');
});
test('A URLs carry atmo=1, skyline=1 on city routes, and alps is flown by the search pilot only', () => {
  for (const c of aCells(1600)) { const e = planEpisode('A', 11, { run: 'r', rs: 1, cell: c }), p = q(e.url); assert.equal(p.get('atmo'), '1'); assert.equal(p.get('skyline') === '1', ['newyork', 'london', 'moscow', 'dubai', 'mega'].includes(c.route)); if (c.route === 'alps') assert.equal(e.policyId, 'search_v1'); }
  const cells = aCells(1600), share = (f) => cells.filter(f).reduce((a, c) => a + c.quota, 0) / 1600;
  assert.ok(Math.abs(share((c) => c.route === 'moscow') - 0.08) < 0.01 && Math.abs(share((c) => c.sky === 'cloudy' && c.route !== 'moscow') - 0.07) < 0.01);
  assert.equal(cells.length, 7 * SKIES.length);
});
test('D failed-P6 episodes use a list seed; D URLs are never begun (no autostart parameter)', () => {
  let n = 0; for (let s = 0; s < 3000; s++) { const e = planEpisode('D', s, { run: 'r', rs: 1 }); if (e.inject && e.inject.kind === 'failed_p6') { n++; assert.ok(P6_SEEDS.includes(+q(e.url).get('seed'))); } assert.equal(q(e.url).get('scenario'), 'real'); }
  assert.ok(n > 0);
});
test('frame sanity rejects a blank buffer (alpha 0 or no variance)', () => {
  const blank = new Uint8Array(4 * 100), grey = new Uint8Array(4 * 100).fill(128), noise = Uint8Array.from({ length: 400 }, (_, i) => (i % 4 === 3 ? 255 : (i * 37) & 255));
  assert.equal(saneFrame(frameStats(blank)), false); assert.equal(saneFrame(frameStats(grey)), false); assert.equal(saneFrame(frameStats(noise)), true);
});

// Task 10 additions: A behaviour mix, twins, schedules, and the in-page injection trigger that the probes arm
test('A: 50 % PPO / 50 % search off the alps; the policy is in params; S/A injections carry a candidate; twins keep the plan', () => {
  const city = aCells(1600).find((c) => c.route === 'london'), ids = Array.from({ length: 400 }, (_, s) => planEpisode('A', s, { rs: 1, cell: city }).policyId);
  const ppo = ids.filter((x) => x === 'atmo_ppo').length / ids.length; assert.ok(ppo > 0.4 && ppo < 0.6, `atmo_ppo share ${ppo}`);
  for (let s = 0; s < 400; s++) { const e = planEpisode('A', s, { rs: 1, cell: city }); assert.equal(e.params.policy, e.policyId); if (e.inject) { assert.equal(e.inject.kind, 'collision_course'); assert.ok(e.inject.candidate >= 1 && e.inject.candidate <= 8); } }
  assert.deepEqual(planEpisode('A', 5, { rs: 1, cell: city }), planEpisode('A', 5, { rs: 1, cell: city }), 'a plan is a pure function of the seed');
});
test('L twins: a wind injection moves to the twin URL without the wind; hflare is a param; D keeps scenario, start and nav', () => {
  let wind = 0, hflare = 0;
  for (let s = 0; s < 2000; s++) {
    const e = planEpisode('L', s, { rs: 3 });
    if (e.inject && e.inject.url) { wind++; assert.ok(q(e.url).get('wind')); assert.equal(q(e.twinUrl).get('wind'), null); assert.equal(q(e.twinUrl).get('seed'), q(e.url).get('seed')); }
    if (e.inject && e.inject.kind === 'hflare') { hflare++; assert.equal(e.params.hflare, true); }
    if (!e.inject) assert.equal(e.twinUrl, e.url);
  }
  assert.ok(wind > 0 && hflare > 0);
  const d = planEpisode('D', 9, { rs: 1 }), p = q(d.url); assert.equal(p.get('nav'), 'noisy'); assert.equal(p.get('filter'), '1'); assert.ok(['near', 'final', 'far'].includes(p.get('start')));
});
test('schedules: L spans a whole run (up to 900 s, 3-8 s apart); D spacing follows the start', () => {
  const l = landingSchedule(rngOf(4)); assert.ok(l.length > 100 && l.length <= 300 && l[l.length - 1] > 890, `${l.length} ${l[l.length - 1]}`); for (let i = 1; i < l.length; i++) assert.ok(l[i] - l[i - 1] >= 3 && l[i] - l[i - 1] <= 8 + 1e-9);
  for (const [start, lo, hi] of [['far', 240, 600], ['near', 40, 120], ['final', 8, 20]]) { const d = dockingSchedule(rngOf(8), start); assert.ok(d.length > 3); for (let i = 1; i < d.length; i++) assert.ok(d[i] - d[i - 1] >= lo - 0.1 && d[i] - d[i - 1] <= hi + 0.1, `${start} ${d[i] - d[i - 1]}`); }
});
test('L armed kick: fires before the first step at or below its trigger height, the oracle predicts that step, and the replay is bit-exact', () => {
  const cond = drawConditions(5, { final: true }), inj = { kind: 'alt_plus_60', at: 'runtime', trigger: { qty: 'hRAft', value: 640, phase: null }, params: {} };
  const oracle = landingTriggerStep(cond, {}, inj), sim = createLandingSim(cond), fired = [], ft = []; armLanding(sim, inj, (r) => fired.push(r));
  for (let n = 0; n < oracle + 400; n++) { ft.push(sim.flight.air.hRA / 0.3048); sim.step(); }
  assert.equal(fired.length, 1); assert.equal(fired[0].step, oracle); assert.equal(fired[0].kind, 'alt_plus_60'); assert.equal(fired[0].params.trigger.value, 640);
  assert.ok(ft[oracle] <= 640 && ft[oracle - 1] > 640, 'the kick lands before the first step whose height is at or below the trigger');
  assert.ok(ft[oracle + 1] > ft[oracle] + 150, 'the +60 m kick shows in the next step'); assert.equal(fired[0].params.trigger.at, +ft[oracle].toFixed(2));
  const r = replayLanding(cond, { injection: fired[0], step: stepOf(sim), liveP: Array.from(sim.flight.p), liveV: Array.from(sim.flight.v) });
  assert.equal(stepOf(r), stepOf(sim));
  assert.equal(landingTrig(sim, { value: 5000 }), true); assert.equal(landingTriggerStep(cond, {}, { ...inj, trigger: { qty: 'hRAft', value: -10 } }), null);
});
test('D armed kick: fires in cycles and skips at its trigger (and phase), the oracle predicts it, and the replay is bit-exact', () => {
  const args = { seed: 11, start: 'final', nav: 'noisy', filter: true }, inj = { kind: 'closing_plus_0.2', at: 'runtime', trigger: { qty: 'axial_m', value: 2.1, phase: null }, params: {} };
  const oracle = dockingTriggerStep(args, inj); assert.ok(oracle > 0);
  const sim = createRealSim(args), st = { pending: inj, injection: null }; armDocking(st, inj);
  stepArmed(st, sim, (oracle - 10) * 0.1); assert.equal(st.injection, null, 'a skip that stops short does not fire');
  stepArmed(st, sim, 30); assert.equal(st.injection.step, oracle, 'a skip across the trigger fires at the exact cycle'); assert.ok(-portRel(sim.x, sim.q)[0] < 2.1 + 1);
  const r = replayDocking(args, { injection: st.injection, step: cycleOf(sim), liveX: Array.from(sim.x), liveQ: Array.from(sim.q) }); assert.equal(cycleOf(r), cycleOf(sim));
  const kos = { kind: 'lateral_drift', trigger: { qty: 'rho_m', value: 25, phase: 'CORRIDOR' }, params: { dv: 0.5 } }, s2 = createRealSim({ seed: 11, start: 'near', nav: 'noisy', filter: true });
  while (!s2.rep.done && !dockingTrig(s2, kos.trigger)) s2.step();
  assert.equal(s2.rep.done, false, 'the KOS band triggers in the near run'); assert.equal(s2.guid.st.phase, 'CORRIDOR'); assert.ok(Math.hypot(...portRel(s2.x, s2.q)) <= 25);
});
test('D centreline camera: at SHIP_PORT + 0.5 m along body +X, looking 100 m down that axis, up = body +Y', () => {
  const calls = [], vec = () => ({ v: null, set(...a) { this.v = a; return this; } }), cam = { position: vec(), up: vec() };
  const half = Math.SQRT1_2, s = { x: [-40, 1, 2], q: [0, 0, half, half] };
  centrelinePose(cam, s, (x, y, z) => calls.push([x, y, z]));
  const port = [s.x[0] + 0.99, s.x[1] + 18.9, s.x[2]], near = (a, b) => a.every((v, i) => Math.abs(v - b[i]) < 1e-9);
  assert.ok(near(cam.position.v, [port[0], port[1] + 0.5, port[2]]), `${cam.position.v}`); assert.ok(near(cam.up.v, [-1, 0, 0]));
  assert.ok(near(calls[0], [port[0], port[1] + 100, port[2]])); assert.deepEqual(SHIP_PORT, [18.9, -0.99, 0]);
});
test('ring rule: a clip straddles an injection when f0 <= injection step <= f2', () => {
  const fr = [{ step: 10 }, { step: 30 }, { step: 50 }];
  assert.equal(straddles(null, fr), false); assert.equal(straddles({ step: 9 }, fr), false); assert.equal(straddles({ step: 51 }, fr), false);
  for (const s of [10, 30, 50]) assert.equal(straddles({ step: s }, fr), true);
});
test('injection draws stay inside the catalogue bands (attitude_kick rho >= 20 m, alt_plus_60 550-700 ft)', () => {
  const r = rngOf(99);
  for (let i = 0; i < 4000; i++) { const d = drawInjection('D', r.float), l = drawInjection('L', r.float);
    if (d && d.kind === 'attitude_kick') assert.ok(d.trigger.value >= 20); if (l && l.kind === 'alt_plus_60') assert.ok(l.trigger.value >= 550 && l.trigger.value <= 700); }
});
test('L/D slot selection: p = min(1, w * slots / m) keeps exactly the free slots over a run and spreads them; w thins the far phase', () => {
  let kept = 0, spread = 0, runs = 0;
  for (let seed = 1; seed <= 200; seed++) {
    const r = rngOf(seed), m = 40, got = []; let slots = 11;
    for (let i = 0; i < m && slots > 0; i++) if (r.float() < keepProb(1, slots, m - i)) { got.push(i); slots--; }
    assert.equal(got.length, 11, 'exactly the free slots (one kept back for a pending injection)'); kept += got.length; spread += got[got.length - 1] - got[0]; runs++;
  }
  assert.ok(spread / runs > 28, `kept samples span the run (mean span ${spread / runs} of 40)`);
  assert.ok(Math.abs(keepProb(0.35, 6, 3) - 0.7) < 1e-12); assert.equal(keepProb(1, 6, 3), 1); assert.equal(keepProb(1, 0, 3), 0); assert.equal(keepProb(0.4, 12, 0), 1);
});
test('oracles: one clean run gives the kick step and the run end (L steps at 120 Hz, D cycles)', () => {
  const cond = drawConditions(7, { final: true }), inj = { trigger: { qty: 'hRAft', value: 500 } }, o = landingOracle(cond, {}, inj), s = createLandingSim(cond); s.run();
  assert.equal(o.injAt, landingTriggerStep(cond, {}, inj)); assert.equal(o.endT, s.flight.t); assert.equal(landingOracle(cond, {}).injAt, null);
  const args = { seed: 5, start: 'final', nav: 'noisy', filter: true }, d = dockingOracle(args, { trigger: { qty: 'rho_m', value: 5 } }), r = createRealSim(args); r.run();
  assert.equal(d.endT, r.t); assert.ok(d.injAt > 0 && d.injAt < cycleOf(r));
});
test('A cell pick: the cell furthest from its quota first; a cell with no record after DEAD_TRIES tries is set aside', () => {
  assert.deepEqual(pickCell({}), aCells(1600)[0]);
  const c1 = pickCell({ 'alps|clear': 5 }); assert.notEqual(`${c1.route}|${c1.sky}`, 'alps|clear');
  const all = Object.fromEntries(aCells(1600).map((c) => [`${c.route}|${c.sky}`, 1])); all['moscow|storm'] = 0; assert.equal(`${pickCell(all).route}|${pickCell(all).sky}`, 'moscow|storm');
  assert.notEqual(`${pickCell(all, { 'moscow|storm': DEAD_TRIES }).route}`, 'moscow', 'the dead cell is skipped'); assert.equal(pickCell(all, { 'moscow|storm': DEAD_TRIES - 1 }).route, 'moscow');
  const dead = Object.fromEntries(aCells(1600).map((c) => [`${c.route}|${c.sky}`, DEAD_TRIES])); assert.equal(pickCell({}, dead), null);
});
