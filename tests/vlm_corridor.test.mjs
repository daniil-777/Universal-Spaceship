import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { SpaceEnv } from '../src/env.js';
import { qFromAxisAngle, qAxes, qIdentity } from '../src/mathx.js';
import { cloneEnv, macroAction, rollout, actionSearch, corridorLabel, crashCause, createSearchPilot, atanhClamp, PULSE } from '../vlm/gen/labels/corridor.js';
import { corridorFacts, clockOf } from '../vlm/gen/labels/corridor_facts.js';
import { lookAtCamera, projectSphere, cameraPosition, project } from '../vlm/gen/labels/camera.js';
import { pageWorld, registerLongGrids, ROUTES } from '../vlm/gen/labels/worlds.js';
import { validateRecord } from '../vlm/gen/schema.js';

const ang = (a, b) => Math.acos(Math.max(-1, Math.min(1, (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (Math.hypot(...a) * Math.hypot(...b))))) * 180 / Math.PI;
const emptySpace = () => { const e = new SpaceEnv(3, { level: 1, count: 0, speedScale: 1, comets: 0, cometSpeed: 1 }); e.setComets(0); e.setCount(0); e.spawnAsteroids(); e.reset(); return e; };
const chaseCam = (e) => { const s = e.ship; return lookAtCamera({ eye: [0, 1, 2].map((i) => s.p[i] - 13 * s.f[i] + 3.6 * s.u[i]), target: [0, 1, 2].map((i) => s.p[i] + 9 * s.f[i] - 1.5 * s.u[i]), up: [0, 1, 0], fovDeg: 50, aspect: 896 / 504, near: 0.1, far: 9000 }); };

test('cloneEnv tracks a noise-free world exactly and leaves the source untouched', () => {
  const e = new SpaceEnv(3, { level: 1, count: 0, comets: 0, planes: 6, mountains: true, atmosphere: true, weather: 0 }); e.setWeather(0, { wind: 0, cover: 0, turb: 0 }); e.reset();
  const act = (i) => [Math.sin(i * 0.1) * 0.6, Math.cos(i * 0.07) * 0.4, 0, 0.3];
  for (let i = 0; i < 60; i++) e.step(act(i));
  const c = cloneEnv(e, 99), p0 = [...e.ship.p], steps0 = e.steps; let dev = 0;
  for (let i = 0; i < 20; i++) c.step(act(i));
  assert.deepEqual([...e.ship.p], p0); assert.equal(e.steps, steps0);
  const d = cloneEnv(e, 99);
  for (let i = 60; i < 150 && !e.done; i++) { e.step(act(i)); d.step(act(i)); dev = Math.max(dev, Math.hypot(e.ship.p[0] - d.ship.p[0], e.ship.p[1] - d.ship.p[1], e.ship.p[2] - d.ship.p[2])); }
  assert.equal(dev, 0);
});
test('long worlds: registerLongGrid, hf.lap present, the clone gets its own long field', async () => {
  await registerLongGrids();
  for (const route of ['mega', 'china']) {
    const e = pageWorld({ route, sky: 'clear', seed: 5 });
    assert.ok(e.hf && e.hf.lap, `${route}: env.hf.lap exists`);
    const c = cloneEnv(e, 1);
    assert.notEqual(c.hf, e.hf); assert.equal(c.hf.lap.shift, e.hf.lap.shift);
    for (const [x, z] of [[0, 0], [17, -5], [-40, 9]]) assert.equal(c.hf.height(x, z), e.hf.height(x, z));
  }
});
test('HANDS_OFF from a state with |cmd| > 0.5 keeps the forward vector within 10 degrees; a held command reverses', () => {
  const e = emptySpace(); e.ship.p[1] = 0; e.ship.p[2] = 0;
  for (let i = 0; i < 6; i++) e.step([0.3, 0.3, 1.5, 1.0]);
  assert.ok(Math.max(...[...e.cmd].map(Math.abs)) > 0.5);
  const f0 = [...e.ship.f], c = cloneEnv(e, 1000), off = macroAction('CONTINUE', { air: false, uThr: atanhClamp(e.cmd[3]) });
  for (let i = 0; i < 45; i++) c.step(off(i));
  assert.ok(ang(c.ship.f, f0) < 10, `hands-off turned ${ang(c.ship.f, f0).toFixed(1)} deg`);
  const p = emptySpace(); p.ship.p[1] = 0; p.ship.p[2] = 0; for (let i = 0; i < 6; i++) p.step([1.5, 0, 0, 0]);
  const pf = [...p.ship.f], h = cloneEnv(p, 1000), held = cloneEnv(p, 1000), hold = Float32Array.from(p.cmd, atanhClamp); let maxHeld = 0;
  for (let i = 0; i < 45; i++) { h.step([0, 0, 0, 0]); held.step(hold); maxHeld = Math.max(maxHeld, ang(held.ship.f, pf)); }
  assert.ok(ang(h.ship.f, pf) < 20 && maxHeld > 150, `pure pitch 0.89: hands-off ${ang(h.ship.f, pf).toFixed(1)}, held ${maxHeld.toFixed(0)}`);
});
test('macro geometry in an empty space corridor (edge guard at the page default)', () => {
  for (const a of ['CLIMB', 'DESCEND', 'TURN_LEFT', 'TURN_RIGHT']) {
    for (const pulse of [PULSE.space, 45]) {
      const e = emptySpace(); for (let i = 0; i < 5; i++) e.step([0, 0, 0, 0]);
      const v0 = [...e.ship.v], fn = macroAction(a, { air: false, uThr: 0, pulse }); let maxDev = 0, minVx = Infinity;
      for (let i = 0; i < 45; i++) { e.step(fn(i)); maxDev = Math.max(maxDev, ang(e.ship.v, v0)); minVx = Math.min(minVx, e.ship.v[0]); }
      if (pulse === PULSE.space) { assert.ok(maxDev >= 25 && maxDev <= 60, `${a} pulse ${maxDev}`); assert.ok(minVx > 0, `${a} minVx ${minVx}`); }
      else assert.ok(maxDev > 150, `${a} held ${maxDev}`);
    }
  }
  for (const [a, sign] of [['SPEED_UP', 1], ['SLOW_DOWN', -1]]) {
    const e = emptySpace(), fn = macroAction(a, { air: false, uThr: 0 }); for (let i = 0; i < 45; i++) e.step(fn(i));
    assert.ok(sign * (Math.hypot(...e.ship.v) - 14) >= 5, `${a} ends at ${Math.hypot(...e.ship.v)}`);
  }
});
test('action search on a crafted head-on hazard: UNSAFE, CONTINUE not safe; crash cause is the rock', () => {
  const e = new SpaceEnv(4, { level: 1, count: 1, comets: 0 }); e.setComets(0); e.setCount(1); e.spawnAsteroids(); e.reset();
  const a = e.asteroids[0], s = e.ship; a.kind = 0; a.r = 2.5; for (let i = 0; i < 3; i++) { a.p[i] = s.p[i] + 25 * s.f[i]; a.v[i] = 0; }
  const { safety } = corridorLabel(e, { cNear: 2.5 });
  assert.equal(safety.verdict, 'UNSAFE'); assert.ok(!safety.safe_actions.includes('CONTINUE')); assert.equal(safety.cause, 'rock');
  const c = cloneEnv(e, 1000); let r; for (let i = 0; i < 45 && !(r && r.done); i++) r = c.step([0, 0, 0, atanhClamp(e.cmd[3])]);
  assert.equal(crashCause(c), 'rock');
});
test('search pilot is deterministic and returns 4 finite raw actions', () => {
  const a = pageWorld({ route: 'alps', sky: 'fair', seed: 9 }), b = pageWorld({ route: 'alps', sky: 'fair', seed: 9 }), pa = createSearchPilot(), pb = createSearchPilot();
  for (let i = 0; i < 20; i++) { const x = pa.act(a), y = pb.act(b); assert.deepEqual([...x], [...y]); assert.equal(x.length, 4); assert.ok([...x].every(Number.isFinite)); a.step(x); b.step(y); }
});
test('projection: a sphere dead ahead lands at the centre, one behind is out of frame', () => {
  const cam = lookAtCamera({ eye: [0, 0, 0], target: [1, 0, 0], up: [0, 1, 0], fovDeg: 50, aspect: 896 / 504 });
  const p = projectSphere(cam, [30, 0, 0], 2, 896, 504); assert.ok(p.inFrame); assert.ok(Math.abs(p.cx - 448) < 1e-6 && Math.abs(p.cy - 252) < 1e-6);
  assert.equal(projectSphere(cam, [-30, 0, 0], 2, 896, 504).inFrame, false);
  assert.deepEqual(cameraPosition(lookAtCamera({ eye: [3, -2, 7], target: [0, 0, 0], fovDeg: 45, aspect: 1 })).map((v) => +v.toFixed(9)), [3, -2, 7]);
  assert.equal(clockOf([1, 0, 0]), 12); assert.equal(clockOf([0, 0, 1]), 3); assert.equal(clockOf([0, 0, -1]), 9);
  assert.ok(project(cam, [10, 0, 0], 896, 504).front);
});
test('REVIEW FOCUS 1: an empty space corridor with a receding hazard gives null TTC and clearance, and the record validates', () => {
  const e = new SpaceEnv(4, { level: 1, count: 1, comets: 0 }); e.setComets(0); e.setCount(1); e.spawnAsteroids(); e.reset();
  const a = e.asteroids[0], s = e.ship; for (let i = 0; i < 3; i++) { a.p[i] = s.p[i] - 20 * s.f[i]; a.v[i] = -3 * s.f[i]; } e.step([0, 0, 0, 0]);
  const facts = corridorFacts(e, chaseCam(e), { sky: 'space', route: null });
  assert.equal(facts['hazard.0.ttc_s'].v, null); assert.equal(facts['clearance.ground_u'].v, null);
  const rec = JSON.parse(fs.readFileSync(new URL('./vlm_fixtures/record_S.json', import.meta.url), 'utf8'));
  const { safety, safety_eye } = corridorLabel(e, { cNear: 2.5 });
  Object.assign(rec, { facts, safety, safety_eye });
  assert.deepEqual(validateRecord(rec).errors, []);
});
test('REVIEW FOCUS 1 (zero relative velocity): a hazard exactly co-moving with the ship gives null TTC, tca_s 0, cpa equal to the current distance, and the record validates', () => {
  const e = new SpaceEnv(4, { level: 1, count: 1, comets: 0 }); e.setComets(0); e.setCount(1); e.spawnAsteroids(); e.reset();
  const a = e.asteroids[0], s = e.ship; for (let i = 0; i < 3; i++) { a.p[i] = s.p[i] + 20 * s.f[i]; a.v[i] = s.v[i]; }
  const facts = corridorFacts(e, chaseCam(e), { sky: 'space', route: null });
  assert.equal(facts['hazard.0.ttc_s'].v, null);
  assert.equal(facts['hazard.0.tca_s'].v, 0);
  assert.equal(facts['hazard.0.cpa_u'].v, facts['hazard.0.dist_u'].v);
  const rec = JSON.parse(fs.readFileSync(new URL('./vlm_fixtures/record_S.json', import.meta.url), 'utf8'));
  const { safety, safety_eye } = corridorLabel(e, { cNear: 2.5 });
  Object.assign(rec, { facts, safety, safety_eye });
  assert.deepEqual(validateRecord(rec).errors, []);
});
test('clock is screen-relative and bearing_clock is ship-body relative: a 90-degree banked ship sees them differ; a hazard behind the camera gives clock null', () => {
  const e = new SpaceEnv(3, { level: 1, count: 0, comets: 0 }); e.setComets(0); e.setCount(0); e.spawnAsteroids(); e.reset();
  const s = e.ship; s.p[0] = 0; s.p[1] = 0; s.p[2] = 0;
  qFromAxisAngle(1, 0, 0, Math.PI / 2, s.q); qAxes(s.q, s.f, s.u, s.r);
  e.asteroids.push({ p: Float64Array.from([0, 3, 0]), v: new Float64Array(3), q: qIdentity(), axis: Float64Array.from([0, 1, 0]), spin: 0, r: 0.5, m: 1, drift: 0, shape: 0, kind: 0, gen: 0 });
  e.sense();
  const cam = lookAtCamera({ eye: [-10, 0, 0], target: [10, 0, 0], up: [0, 1, 0], fovDeg: 50, aspect: 896 / 504 });
  const facts = corridorFacts(e, cam, { sky: 'space', route: null });
  assert.equal(facts['hazard.0.clock'].v, 12, 'a hazard offset along world up projects to 12 on screen');
  assert.equal(facts['hazard.0.bearing_clock'].v, 9, 'the same hazard sits at 9 in the rolled ship\'s own body frame');
  assert.notEqual(facts['hazard.0.clock'].v, facts['hazard.0.bearing_clock'].v);
  e.asteroids[0].p[0] = -20; e.asteroids[0].p[1] = 0; e.asteroids[0].p[2] = 0; e.sense();
  const behind = corridorFacts(e, cam, { sky: 'space', route: null });
  assert.equal(behind['hazard.0.clock'].v, null);
  assert.equal(behind['hazard.0.in_frame'].v, false);
});
test('space visual facts: a value the page does not provide stays null (never false); a known one is kept', () => {
  const e = emptySpace(), cam = chaseCam(e);
  const u = corridorFacts(e, cam, { sky: 'space', route: null, space: {} });
  for (const id of ['earth_in_frame', 'moon_in_frame', 'sun.lit', 'orbit.body']) { assert.ok(id in u, id); assert.equal(u[id].v, null, id); }
  const n = corridorFacts(e, cam, { sky: 'space', route: null, space: { earthInFrame: null, moonInFrame: null } });
  assert.equal(n.earth_in_frame.v, null); assert.equal(n.moon_in_frame.v, null);
  const k = corridorFacts(e, cam, { sky: 'space', route: null, space: { sunLit: 1, body: 'earth', lat: 10, lon: 20, earthInFrame: false, moonInFrame: true } });
  assert.equal(k.earth_in_frame.v, false); assert.equal(k.moon_in_frame.v, true); assert.equal(k['orbit.body'].v, 'earth'); assert.equal(k['sun.lit'].v, 1);
});
test('worlds.js ROUTES equals src/terrain.js ROUTES (parsed as text: terrain.js imports three)', () => {
  const src = fs.readFileSync(new URL('../src/terrain.js', import.meta.url), 'utf8'), body = /export const ROUTES = \{([\s\S]*?)\n\};/.exec(src)[1].replace(/\/\/[^\n]*/g, '');
  assert.deepEqual(JSON.parse(JSON.stringify(ROUTES)), new Function(`return {${body}};`)());
});
