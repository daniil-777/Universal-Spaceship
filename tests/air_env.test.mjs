import { test } from 'node:test';
import assert from 'node:assert/strict';
import { SpaceEnv, ENV, OBS_DIM, OBS_BASE, N_RAYS } from '../src/env.js';
import { qFromAxisAngle, qAxes } from '../src/mathx.js';

const airCorridor = (seed = 5, weather = 0) => new SpaceEnv(seed, { count: 0, comets: 0, atmosphere: true, weather });   // no terrain: the floor is the corridor's
const zero = [0, 0, 0, 0];
const stub = (o) => ({ windAt: (x, y, z, out) => { out[0] = 0; out[1] = 0; out[2] = 0; return out; }, turbAt: (x, y, z, o) => { o.sigma = 0.2; o.hA = y + 22; o.share = 0; o.Lc = 0; return o; }, cloudAt: () => 0, advance() {}, cellsNear: (x, r, out = []) => { out.length = 0; return out; }, floor: -22, ...o });

test('the air never reaches space: no weather, air inputs all zero', () => {
  const env = new SpaceEnv(3, {}); assert.equal(env.atmosphere, false); assert.equal(env.weather, null);
  for (let i = 0; i < 30; i++) env.step([Math.sin(i), 0.3, -0.2, 0.1]);
  for (let k = OBS_BASE; k < OBS_DIM; k++) assert.equal(env.obs[k], 0, 'obs ' + k);
});
test('calm air, neutral stick + auto-throttle: level flight for 20 s', () => {
  const env = airCorridor(); env.setWeather(0, { wind: 0 }); env.reset(); env.autoThrottle = true; const y0 = env.ship.p[1];   // truly calm: no mean wind to drift the ship into a wall
  for (let i = 0; i < 300; i++) { const r = env.step(zero); assert.ok(!r.done, 'crashed at ' + i); }
  assert.ok(Math.abs(env.ship.p[1] - y0) < 2.5, 'height ' + (env.ship.p[1] - y0)); for (const v of env.obs) assert.ok(Number.isFinite(v));
});
test('overstress ends the episode', () => {
  const env = airCorridor(); env.weather = stub({ windAt: (x, y, z, out) => { out[0] = 0; out[1] = 9; out[2] = 0; return out; } });
  for (let i = 0; i < 3; i++) env.ship.v[i] = 22 * env.ship.f[i];
  let done = false; for (let i = 0; i < 6 && !done; i++) done = env.step(zero).done;
  assert.ok(done); assert.equal(env.crashCause, 'overstress');
});
test('weather radar sees a towering cell ahead, not off to the side', () => {
  const env = airCorridor(), s = env.ship; s.q.set([0, 0, 0, 1]); qAxes(s.q, s.f, s.u, s.r);
  env.weather = stub({ cellsNear: (x, r, out = []) => { out.length = 0; out.push({ type: 1, x: s.p[0] + 25, z: s.p[2], R: 10, base: s.p[1] - 10, top: s.p[1] + 30, sig: 3.5, W: 5, seed: 0 }); return out; } });
  env.sense(); const centre = 2 * ENV.rays.nAz + 5, left = 2 * ENV.rays.nAz;
  assert.ok(env.obs[OBS_BASE + centre] > 0.3, 'centre ' + env.obs[OBS_BASE + centre]); assert.ok(env.obs[OBS_BASE + left] < 0.05, 'left ' + env.obs[OBS_BASE + left]);
});
test('air data: airspeed and load factor near trim in level flight', () => {
  const env = airCorridor(); env.autoThrottle = true; for (let i = 0; i < 45; i++) env.step(zero);
  const a = OBS_BASE + N_RAYS; assert.ok(Math.abs(env.obs[a]) < 0.2, 'V ' + env.obs[a]); assert.ok(Math.abs(env.obs[a + 3]) < 0.35, 'n−1 ' + env.obs[a + 3]);
});
test('the tunnel guide threads the bore under the new physics', () => {
  const env = new SpaceEnv(8, { count: 0, comets: 0, atmosphere: true }), s = env.ship, y0 = -26, inRock = (x) => x > 20 && x < 50, bore = (z) => Math.abs(z) < 7;
  env.hf = { y0, PERIOD: 960, lap: { shift: 0 }, peak: y0 + 40, tunnels: [{ x0: 20, x1: 50, z: 0, y: 25, hw: 7, hh: 9 }],
    height: (x, z) => (inRock(x) ? y0 + (bore(z) ? 16 : 40) : y0), ceiling: (x, z) => (inRock(x) && bore(z) ? y0 + 34 : Infinity) };
  env.weather = null;
  qFromAxisAngle(0, 1, 0, 0.1, s.q); qAxes(s.q, s.f, s.u, s.r); s.p.set([-50, 5, 10]); for (let i = 0; i < 3; i++) s.v[i] = ENV.ship.cruise * s.f[i];
  let crashed = false, maxX = s.p[0];
  for (let k = 0; k < 170 && env.laps === 0; k++) { const r = env.step(zero); if (r.done) { crashed = true; break; } if (env.laps === 0) maxX = Math.max(maxX, s.p[0]); }
  assert.ok(!crashed, 'no collision (x ' + s.p[0].toFixed(1) + ')'); assert.ok(maxX >= 55, 'out the other side ' + maxX.toFixed(1));
});
test('switching the atmosphere off returns to the space model', () => {
  const env = airCorridor(); for (let i = 0; i < 20; i++) env.step([0.2, 0, 0.1, 0.3]);
  env.setAtmosphere(false); env.reset(); assert.equal(env.weather, null);
  for (let i = 0; i < 10; i++) env.step(zero);
  assert.ok(Math.abs(Math.hypot(...env.ship.v) - ENV.ship.cruise) < 0.5); for (let k = OBS_BASE; k < OBS_DIM; k++) assert.equal(env.obs[k], 0);
});
test('a lap slide keeps the wind continuous', () => {
  const env = new SpaceEnv(12, { count: 0, comets: 0, atmosphere: true, weather: 0.8 });
  env.hf = { lap: { shift: 0 }, segments: 8, PERIOD: 960, peak: -40, height: () => -40 }; env.reset();
  const W = env.weather, a = W.windAt(ENV.xHalf - 0.05, -5, 2, new Float64Array(3)); env.hf.lap.shift += 2 * ENV.xHalf;
  const b = W.windAt(-ENV.xHalf + 0.05, -5, 2, new Float64Array(3));
  for (let i = 0; i < 3; i++) assert.ok(Math.abs(a[i] - b[i]) < 0.3, `axis ${i}: ${a[i]} vs ${b[i]}`);
});
test('weather changed mid-flight keeps the episode going', () => {
  const env = airCorridor(); env.autoThrottle = true; for (let i = 0; i < 50; i++) env.step(zero);
  const p = Array.from(env.ship.p), steps = env.steps; env.setWeather(1);
  assert.equal(env.weather.severity, 1); assert.equal(env.steps, steps); assert.deepEqual(Array.from(env.ship.p), p);
  const r = env.step(zero); assert.ok(Number.isFinite(r.reward)); for (const v of env.obs) assert.ok(Number.isFinite(v));
});

test('weather radar matches a fine brute-force sweep on random skies', async () => {
  const { cellShape } = await import('../src/weather.js'), { mulberry32 } = await import('../src/mathx.js');
  const env = airCorridor(21), s = env.ship, rng = mulberry32(4), range = ENV.rays.range; let lit = 0;
  for (let trial = 0; trial < 12; trial++) {
    const cells = Array.from({ length: 6 }, () => { const type = Math.floor(rng() * 3), base = s.p[1] - 15 + 20 * rng(); return { type, x: s.p[0] + (rng() * 2 - 1) * 60, z: s.p[2] + (rng() * 2 - 1) * 40, R: 6 + 18 * rng(), base, top: base + 8 + 30 * rng(), sig: 0.3 + 3.5 * rng(), W: 1, seed: 0 }; });
    env.weather = stub({ cellsNear: (x, r, out = []) => { out.length = 0; out.push(...cells); return out; } }); env.sense();
    for (let i = 0; i < N_RAYS; i++) {
      const d = env.rayDirWorld, tMax = env.rayHit[i]; let best = 0;
      for (let t = 0.125; t < tMax; t += 0.25) { const px = s.p[0] + d[3 * i] * t, py = s.p[1] + d[3 * i + 1] * t, pz = s.p[2] + d[3 * i + 2] * t;
        for (const c of cells) best = Math.max(best, cellShape(c.type, c.R, c.base, c.top, px - c.x, py, pz - c.z) * c.sig / 4 * (1 - 0.5 * t / range)); }
      best = Math.min(1, best); if (best > 0.2) lit++;
      assert.ok(Math.abs(env.obs[OBS_BASE + i] - best) < 0.08, `trial ${trial} beam ${i}: ${env.obs[OBS_BASE + i]} vs ${best}`);
    }
  }
  assert.ok(lit > 50, 'enough beams see cloud: ' + lit);
});

test('the radar sweeps a third of its beams per decision and a new sky at once', () => {
  const level = (e) => { e.ship.q.set([0, 0, 0, 1]); qAxes(e.ship.q, e.ship.f, e.ship.u, e.ship.r); return e; };
  const cell = { type: 1, x: 0, z: 0, R: 10, base: 0, top: 0, sig: 3.5, W: 5, seed: 0 }, sky = () => stub({ cellsNear: (x, r, out = []) => { out.length = 0; out.push(cell); return out; } });
  const env = level(airCorridor()), s = env.ship, radar = (e) => Array.from(e.obs.slice(OBS_BASE, OBS_BASE + N_RAYS));
  Object.assign(cell, { x: s.p[0] + 25, z: s.p[2], base: s.p[1] - 10, top: s.p[1] + 30 });
  env.weather = sky(); env.sense(); const before = radar(env);                 // a new sky: every beam at once
  cell.z += 12;                                                                  // the cell drifts beside the lane
  const ref = level(airCorridor()); ref.weather = sky(); ref.sense(); const after = radar(ref);
  const differ = (a, b) => a.filter((v, i) => Math.abs(v - b[i]) > 1e-6).length, moved = differ(before, after);
  assert.ok(moved > 6, 'the drift shows on a whole sweep: ' + moved);
  env.sense(); const stale = differ(radar(env), after);
  assert.ok(stale > 0 && stale < moved, `one decision refreshes a third of the sky: ${stale} of ${moved} beams still old`);
  env.sense(); env.sense(); assert.equal(differ(radar(env), after), 0, 'three decisions refresh all of it');
});

test('calm air over a mountain range and a skyline: no sudden drafts between decisions', () => {
  for (const o of [{ mountains: true }, { city: 'dubai' }]) {
    const env = new SpaceEnv(9, { count: 0, comets: 0, weather: 0, ...o }); env.autoThrottle = true; let prev = null, worst = 0;
    for (let i = 0; i < 1200; i++) { const r = env.step(zero), w = env.air.wind; if (prev && !r.done) worst = Math.max(worst, Math.hypot(w[0] - prev[0], w[1] - prev[1], w[2] - prev[2])); prev = r.done ? null : Array.from(w); if (r.done) env.reset(); }
    assert.ok(worst < 1, `${Object.keys(o)[0]}: a ${worst.toFixed(2)} u/s wind step between two decisions`);
  }
});
test('each episode flies its own sky, whatever the wing drops drew', () => {
  const mk = () => new SpaceEnv(6, { count: 0, comets: 0, mountains: true, weather: 0.7 }), a = mk(), b = mk(), cells = (e) => e.weather.cells.map((c) => [c.type, c.x, c.z, c.R]);
  b.rngAir(); b.rngAir(); a.reset(); b.reset(); assert.deepEqual(cells(b), cells(a));
});
test('the Playbox Weather and Wind sliders keep the sky mid-flight', () => {
  const env = new SpaceEnv(4, { count: 0, comets: 0, mountains: true, weather: 0.5 }); for (let i = 0; i < 30; i++) env.step(zero);
  const W = env.weather, xs = (w) => w.cells.slice(0, 2).map((c) => [c.x, c.z]);
  env.setWeather(0.5, { wind: 1.2 }); assert.deepEqual(xs(env.weather), xs(W)); assert.equal(env.weather.time, W.time); assert.equal(env.weather.drift, W.drift);
});
test('switching the atmosphere on mid-flight brings the sky and the air data at once', () => {
  const env = new SpaceEnv(3, { count: 0, comets: 0, mountains: true, atmosphere: false }); for (let i = 0; i < 5; i++) env.step(zero);
  env.setAtmosphere(true); assert.ok(env.weather, 'a sky'); env.step(zero); assert.notEqual(env.obs[OBS_BASE + N_RAYS], 0, 'airspeed input');
});
test('in the air the ceiling and the side walls are slid along: speed kept, no stall', () => {
  const env = new SpaceEnv(3, { count: 0, comets: 0, atmosphere: true, weather: 0, edgeGuard: false }), s = env.ship, top = ENV.yHalf + ENV.ship.wallClamp;   // the edge guard would turn away first
  qFromAxisAngle(0, 0, 1, 22 * Math.PI / 180, s.q); qAxes(s.q, s.f, s.u, s.r); s.p.set([0, top - 3, 0]); for (let i = 0; i < 3; i++) s.v[i] = 14 * s.f[i];
  env.autoThrottle = true; let worstA = 0, stalled = false;
  for (let i = 0; i < 40; i++) { env.step(zero); worstA = Math.max(worstA, Math.abs(env.air.alpha)); stalled = stalled || env.air.stalled; }
  assert.ok(!stalled && worstA < 16 * Math.PI / 180, `ceiling: α up to ${(worstA * 180 / Math.PI).toFixed(0)}°`); assert.ok(Math.hypot(...s.v) > 11, 'speed ' + Math.hypot(...s.v).toFixed(1));
  const w = new SpaceEnv(5, { count: 0, comets: 0, atmosphere: true, weather: 0, edgeGuard: false }), t = w.ship; w.autoThrottle = true;
  qFromAxisAngle(0, 1, 0, -30 * Math.PI / 180, t.q); qAxes(t.q, t.f, t.u, t.r); t.p.set([0, 0, ENV.zHalf - 4]); for (let i = 0; i < 3; i++) t.v[i] = 14 * t.f[i];
  let wStall = false; for (let i = 0; i < 40; i++) { w.step(zero); wStall = wStall || w.air.stalled; }
  assert.ok(!wStall && Math.hypot(...t.v) > 11, 'side wall: stalled ' + wStall + ', speed ' + Math.hypot(...t.v).toFixed(1));
});
test('smooth flying pays in the air: a flipped roll command costs more than the space-era jerk and rate terms', () => {
  const mk = () => { const e = airCorridor(7); e.setWeather(0, { wind: 0 }); e.reset(); e.autoThrottle = true; return e; }, a = mk(), b = mk();
  for (let i = 0; i < 10; i++) { a.step(zero); b.step(zero); }
  const ra = a.step(zero).reward, rb = b.step([0, 0, 0.55, 0]).reward;   // tanh(0.55) ≈ 0.5: |Δa|² = 0.25 → space-era jerk + rate ≈ 0.016
  assert.ok(ra - rb > 0.03, 'penalty for the jerk ' + (ra - rb).toFixed(4));
});
test('the mean wind changes evenly across a decision (sampled at its start and its end), not in one jump', () => {
  const env = airCorridor(3, 1); env.autoThrottle = true; const W = env.weather, adv = W.advance.bind(W), seen = [];
  W.advance = (h) => { seen.push(Array.from(env.air.wind)); adv(h); };
  for (let i = 0; i < 450 && !env.done; i++) env.step(zero);
  const d = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]); let sub = 0, dec = 0;
  for (let i = 1; i < seen.length; i++) { sub = Math.max(sub, d(seen[i], seen[i - 1])); if (i >= 4) dec = Math.max(dec, d(seen[i], seen[i - 4])); }
  assert.ok(dec > 0.05, 'the sky has some wind shear: ' + dec); assert.ok(sub < 0.45 * dec, `largest substep change ${sub.toFixed(3)} vs a decision's ${dec.toFixed(3)}`);
});
