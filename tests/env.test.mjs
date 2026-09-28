import { qFromAxisAngle } from '../src/mathx.js';
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { SpaceEnv, ENV, OBS_DIM, N_RAYS, RAY_DIRS_BODY } from '../src/env.js';
import { qIntegrate, qAxes, qRotate, qInvRotate, wrapX, qRandom, mulberry32 } from '../src/mathx.js';
import { createCityField } from '../src/cityfield.js';

const near = (a, b, eps, msg) => assert.ok(Math.abs(a - b) <= eps, `${msg || ''} ${a} vs ${b}`);

test('rate command: full pitch-up command settles at the commanded rate and pitches the nose up', () => {
  const env = new SpaceEnv(8, { count: 0, comets: 0 });
  env.ship.q.set([0, 0, 0, 1]); qAxes(env.ship.q, env.ship.f, env.ship.u, env.ship.r); env.ship.v.set([14, 0, 0]);
  for (let i = 0; i < 12; i++) env.step([3, 0, 0, 0]);      // tanh(3) ≈ 1 → pitch rate → 1.6 rad/s
  near(env.ship.w[2], ENV.ship.rateMax[2], 0.05, 'pitch rate reached'); assert.ok(env.ship.f[1] > 0.5, 'nose up ' + env.ship.f[1]); near(env.ship.w[0], 0, 1e-9); near(env.ship.w[1], 0, 1e-9);
});

test('quaternion integration: pitch rate about body z rotates the nose up', () => {
  const q = new Float64Array([0, 0, 0, 1]), f = new Float64Array(3), u = new Float64Array(3);
  for (let i = 0; i < 100; i++) qIntegrate(q, [0, 0, Math.PI / 2], 0.01);
  qAxes(q, f, u);
  near(f[0], 0, 1e-9, 'f.x'); near(f[1], 1, 1e-9, 'f.y'); near(u[1], 0, 1e-9, 'u.y'); near(u[0], -1, 1e-9, 'u.x');
  near(Math.hypot(...q), 1, 1e-12, 'unit norm');
});

test('qRotate / qInvRotate are inverses and match qAxes', () => {
  const rng = mulberry32(9), q = qRandom(rng), v = [0.3, -0.7, 0.2], w = qRotate(q, v), back = qInvRotate(q, w);
  for (let i = 0; i < 3; i++) near(back[i], v[i], 1e-12);
  const f = new Float64Array(3), u = new Float64Array(3), r = new Float64Array(3); qAxes(q, f, u, r);
  const fx = qRotate(q, [1, 0, 0]), rz = qRotate(q, [0, 0, 1]);
  for (let i = 0; i < 3; i++) { near(f[i], fx[i], 1e-12); near(r[i], rz[i], 1e-12); }
});

test('wrapX maps into [-half, half)', () => {
  assert.equal(wrapX(61, 60), -59); assert.equal(wrapX(-61, 60), 59); assert.equal(wrapX(60, 60), -60); assert.equal(wrapX(0, 60), 0); assert.equal(wrapX(-180, 60), -60);
});

test('centre ray hits an asteroid straight ahead at the right distance (through the wrap seam too)', () => {
  const env = new SpaceEnv(1, { count: 0, comets: 0 });
  env.ship.q.set([0, 0, 0, 1]); qAxes(env.ship.q, env.ship.f, env.ship.u, env.ship.r);
  env.ship.p.set([55, 0, 0]);
  env.asteroids = [{ p: new Float64Array([-45, 0, 0]), v: new Float64Array(3), q: new Float64Array([0, 0, 0, 1]), axis: new Float64Array([0, 1, 0]), spin: 0, r: 2, m: 8, drift: 3, shape: 0 }];   // 20 units ahead across the seam
  env.sense();
  const centre = Math.floor(ENV.rays.nEl / 2) * ENV.rays.nAz + Math.floor(ENV.rays.nAz / 2);
  near(env.rayHit[centre], 18, 1e-5, 'surface distance'); near(env.obs[centre], 1 - 18 / ENV.rays.range, 1e-6, 'centre beam value');
  assert.equal(env.obs[centre + 1], 0, 'a 2-unit rock 20 units out does not reach the 15°-off beam');
  { const d = env.rayDirWorld, tz = d[2] > 0 ? ENV.zHalf / d[2] : d[2] < 0 ? -ENV.zHalf / d[2] : Infinity, ty = d[1] < 0 ? -ENV.yHalf / d[1] : d[1] > 0 ? ENV.yHalf / d[1] : Infinity, t = Math.min(tz, ty);   // the corridor's edges are surfaces the beams see
    near(env.rayHit[0], t < ENV.rays.range ? t - ENV.ship.radius : ENV.rays.range, 1e-4, 'the far corner beam sees the corridor edge');
    assert.ok(env.rayEdge[0] === (t < ENV.rays.range ? 1 : 0) && env.obs[0] === 0, 'flagged as an edge; the policy input stays hazards only'); }
  near(env.obs[N_RAYS], 20 / ENV.rays.range, 1e-6, 'nearest rel x');   // nearest asteroid block starts after the rays
  near(env.obs[N_RAYS + 6], 2 / 5, 1e-6, 'nearest radius');
  env.asteroids[0].r = 4; env.sense();   // a bigger rock (angular radius 11.5°) spills into the neighbouring beam partially
  assert.ok(env.obs[centre + 1] > 0 && env.obs[centre + 1] < env.obs[centre], 'the neighbouring beam sees it partially (soft edge) ' + env.obs[centre + 1]);
});

test('collision terminates the episode with a large penalty', () => {
  const env = new SpaceEnv(2, { count: 0, comets: 0 });
  env.asteroids = [{ p: Float64Array.from([env.ship.p[0] + 1.5, env.ship.p[1], env.ship.p[2]]), v: new Float64Array(3), q: new Float64Array([0, 0, 0, 1]), axis: new Float64Array([0, 1, 0]), spin: 0, r: 1, m: 1, drift: 0, shape: 0 }];
  const res = env.step([0, 0, 0, 0]);
  assert.equal(res.done, true); assert.ok(res.reward < -9, 'penalty applied ' + res.reward);
});

test('zero action = smooth cruise east, wraps at the seam, positive progress reward', () => {
  const env = new SpaceEnv(3, { count: 0, comets: 0 });
  env.ship.q.set([0, 0, 0, 1]); qAxes(env.ship.q, env.ship.f, env.ship.u, env.ship.r); env.ship.v.set([14, 0, 0]);
  let ret = 0;
  for (let i = 0; i < 300; i++) ret += env.step([0, 0, 0, 0]).reward;
  assert.equal(env.laps, 2); near(env.ship.v[0], 14, 1e-6); near(env.ship.v[1], 0, 1e-9); near(ret, 300 * (0.1 + 0.02 + 0.02), 1e-6);   // 300 decisions = 20 s = 280 units east
  assert.ok(Math.abs(env.ship.p[0]) < 60);
});

test('asteroid size law: N(>2r_min) ≈ 2^-2.3 and sizes stay within bounds', () => {
  let n = 0, big = 0;
  for (let s = 0; s < 40; s++) { const env = new SpaceEnv(100 + s, { level: 1 }); for (const a of env.asteroids) { n++; if (a.r > 2) big++; assert.ok(a.r >= 1 && a.r <= 5.0001); } }
  const frac = big / n, expect = Math.pow(2, -2.3);
  assert.ok(Math.abs(frac - expect) < 0.05, `fraction > 2: ${frac.toFixed(3)} expected ${expect.toFixed(3)} (n=${n})`);
  assert.ok(n / 40 >= 34, 'spawn rejection keeps most asteroids ' + n / 40);
});

test('asteroids stay inside the corridor and keep finite velocities; observation stays bounded', () => {
  const env = new SpaceEnv(4, { level: 1 });
  for (let i = 0; i < 3000; i++) { const r = env.step([0.1, -0.1, 0.05, 0.3]); if (r.done || r.truncated) env.reset(); }
  for (const a of env.asteroids) { if (a.kind) continue; assert.ok(Math.abs(a.p[0]) <= 60 && Math.abs(a.p[1]) <= 22.01 && Math.abs(a.p[2]) <= 18.01, 'inside'); assert.ok(Math.hypot(...a.v) < 30, 'finite v'); near(Math.hypot(...a.q), 1, 1e-6); }
  for (let i = 0; i < OBS_DIM; i++) assert.ok(Number.isFinite(env.obs[i]) && Math.abs(env.obs[i]) <= 2.5, `obs[${i}] = ${env.obs[i]}`);
});

test('same seed → identical trajectories', () => {
  const a = new SpaceEnv(77, { level: 0.5 }), b = new SpaceEnv(77, { level: 0.5 });
  for (let i = 0; i < 200; i++) { const act = [Math.sin(i), Math.cos(i * 0.3), 0.1, 0.2]; a.step(act); b.step(act); }
  assert.deepEqual(Array.from(a.obs), Array.from(b.obs));
});

test('ray directions are unit vectors in the forward hemisphere', () => {
  for (let i = 0; i < N_RAYS; i++) { const x = RAY_DIRS_BODY[i * 3], y = RAY_DIRS_BODY[i * 3 + 1], z = RAY_DIRS_BODY[i * 3 + 2]; near(Math.hypot(x, y, z), 1, 1e-12); assert.ok(x > 0.1, "forward-ish " + x); }
});

test('comets cross the corridor fast, respawn after leaving, never sit on the ship, and are sensed like rocks', () => {
  const env = new SpaceEnv(21, { level: 1, count: 0, comets: 4 });
  assert.equal(env.asteroids.length, 4); assert.equal(env.cometCount, 4); assert.equal(env.rockCount, 0);
  let respawns = 0, maxSpeed = 0; const gens = env.asteroids.map((c) => c.gen);
  for (let i = 0; i < 600; i++) {
    const r = env.step([0, 0, 0, 0]); if (r.done) env.reset();
    env.asteroids.forEach((c, k) => { if (c.gen !== gens[k]) { respawns++; gens[k] = c.gen; } maxSpeed = Math.max(maxSpeed, Math.hypot(...c.v)); assert.ok(c.kind === 1); assert.ok(Math.abs(c.p[1]) <= 22 + c.r + 0.6 && Math.abs(c.p[2]) <= 18 + c.r + 0.6, 'comet inside or just leaving'); });
  }
  assert.ok(respawns >= 8, 'comets respawned ' + respawns); assert.ok(maxSpeed >= 10 && maxSpeed <= 20.01, 'comet speed ' + maxSpeed);
  const c = env.asteroids[0]; c.p.set([env.ship.p[0] + 15, env.ship.p[1], env.ship.p[2]]); c.v.set([-15, 0, 0]); env.ship.q.set([0, 0, 0, 1]); qAxes(env.ship.q, env.ship.f, env.ship.u, env.ship.r);
  env.sense(); assert.equal(env.nearest[0], 0, 'the approaching comet is the most urgent body'); assert.ok(env.obs[Math.floor(ENV.rays.nEl / 2) * ENV.rays.nAz + Math.floor(ENV.rays.nAz / 2)] > 0.6, 'centre beam sees it');
});

test('mountains: the beams see the ground ahead-below, ground clearance is reported, and flying into a peak is a crash', () => {
  const env = new SpaceEnv(31, { atmosphere: false, count: 0, comets: 0, mountains: true });
  assert.ok(env.hf && env.hf.peak > ENV.mountains.y0 + 10, 'peaks rise above the floor ' + env.hf.peak);
  env.ship.q.set([0, 0, 0, 1]); qAxes(env.ship.q, env.ship.f, env.ship.u, env.ship.r); env.ship.v.set([14, 0, 0]);
  // put the ship 3 units above the ground: the steep downward beams must see it, the level beams not
  env.ship.p[1] = env.hf.height(env.ship.p[0], env.ship.p[2]) + 3; env.sense();
  const down = env.obs[Math.floor(ENV.rays.nAz / 2)], level = env.obs[Math.floor(ENV.rays.nEl / 2) * ENV.rays.nAz + Math.floor(ENV.rays.nAz / 2)];
  assert.ok(down > 0.6, 'steep beam sees the ground ' + down); assert.ok(env.groundClearance() > 1 && env.groundClearance() < 2.5, 'clearance ' + env.groundClearance());
  assert.ok(level <= 1, 'beam values are bounded ' + level);
  // dive: pitch down hard until the ground is hit → terminal with the collision penalty
  let done = false, steps = 0, rew = 0; while (!done && steps < 300) { const r = env.step([-3, 0, 0, 0]); done = r.done; rew = r.reward; steps++; }
  assert.ok(done, 'crashed into the terrain'); assert.ok(rew < -15, 'collision penalty ' + rew);
  const high = new SpaceEnv(32, { atmosphere: false, count: 0, comets: 0, mountains: true }); high.ship.p[1] = 30; high.ship.q.set([0, 0, 0, 1]); qAxes(high.ship.q, high.ship.f, high.ship.u, high.ship.r);
  high.sense(); assert.equal(high.obs[Math.floor(ENV.rays.nEl / 2) * ENV.rays.nAz + Math.floor(ENV.rays.nAz / 2)], 0, 'level beams see no ground from high above the peaks');
  high.step([0, 0, 0, 0]); assert.ok(high.ship.p[1] <= ENV.mountains.ceiling, 'thin air holds the ship at the ceiling ' + high.ship.p[1]);
  assert.ok(high.hf.peak > ENV.mountains.ceiling - 2, 'the highest peaks reach the ceiling ' + high.hf.peak);
  // straight level flight at the spawn altitude runs into the range within a few laps: the peaks really do cross the flight band
  let crashes = 0; for (let seed = 40; seed < 46; seed++) { const e = new SpaceEnv(seed, { atmosphere: false, count: 0, comets: 0, mountains: true, mountainSeed: 5 + seed % 7 }); assert.ok(e.groundClearance() > 3, 'spawns clear of the ground ' + e.groundClearance()); for (let i = 0; i < 900; i++) { if (e.step([0, 0, 0, 0]).done) { crashes++; break; } } }
  assert.ok(crashes >= 3, 'straight flight hits the mountains in most worlds: ' + crashes + '/6');
});

test('skylines: towers are rasterised into the height field, walls stop the ship, beams see them, straight flight crashes', () => {
  for (const city of ['dubai', 'newyork', 'moscow']) {
    const env = new SpaceEnv(50, { atmosphere: false, count: 0, comets: 0, city }), hf = env.hf;
    assert.ok(hf.buildings.length > 60 || hf.chunks, city + ' has buildings or Meshy districts: ' + hf.buildings.length);
    let tall = null;                                        // the tallest wide ground-level box tower (spires are thinner than a cell); a chunk-only city has none
    if (hf.buildings.length) { tall = hf.buildings.filter((b) => b.w >= 2 && b.d >= 2 && b.y === 0).reduce((a, b) => (b.h > a.h ? b : a));
      const roofH = hf.height(tall.x, tall.z) - hf.y0; assert.ok(roofH >= tall.h - 0.01 && roofH <= tall.h + 6, city + ': the tallest roof is in the field (' + roofH.toFixed(1) + ' vs ' + tall.h + ', rooftop plant allowed)'); }
    else {                                                  // a Meshy tile city: the tallest wide roof (a 5×5-cell block all at least this high) stands in for the tower — a spire is thinner than a beam
      let best = -1, bi = 0, bj = 0;
      for (let j = 2; j < hf.NZ - 2; j++) for (let i = 0; i < hf.NX; i++) { let lo = Infinity; for (let dj = -2; dj <= 2; dj++) for (let di = -2; di <= 2; di++) lo = Math.min(lo, hf.grid[(j + dj) * hf.NX + ((i + di + hf.NX) % hf.NX)]); if (lo > best) { best = lo; bi = i; bj = j; } }
      tall = { x: bi * (hf.PERIOD / hf.NX), z: -hf.ZSPAN / 2 + bj * (hf.ZSPAN / (hf.NZ - 1)), w: 2, h: best, y: 0 }; }
    assert.ok(hf.peak > ENV.mountains.ceiling, city + ': the tallest tower reaches the thin air ' + hf.peak.toFixed(1));
    assert.ok(env.groundClearance() > 3, city + ': spawns clear of the roofs ' + env.groundClearance().toFixed(1));
    const same = createCityField({ city, seed: 1 }); assert.deepEqual(same.grid, hf.grid, city + ' is deterministic');
    // the ship's flank clips a tower wall → crash, even though the column under its centre is a street
    const t = hf.buildings.find((b) => b.h >= 26 && b.y === 0 && b.w >= 4 && !b.shape && hf.height(b.x - b.w / 2 - 0.5, b.z) < hf.y0 + b.h / 2 - 1.5);   // a tall box tower with a clear column just west of its wall
    if (t) { env.ship.q.set([0, 0, 0, 1]); qAxes(env.ship.q, env.ship.f, env.ship.u, env.ship.r); env.ship.v.set([0, 0, 0]);
      env.ship.p[0] = t.x - t.w / 2 - 0.5; env.ship.p[1] = hf.y0 + t.h / 2; env.ship.p[2] = t.z;
      assert.ok(env.step([0, 0, 0, 0]).done, city + ': a wing clipped the tower'); }
    else assert.ok(hf.chunks, city + ': no box tower to clip, so it must be a Meshy district city');
    // 12 units short of the tallest tower, near its top, flying level: the centre beam sees it
    env.reset(); env.ship.q.set([0, 0, 0, 1]); qAxes(env.ship.q, env.ship.f, env.ship.u, env.ship.r);
    env.ship.p[0] = tall.x - tall.w / 2 - 12; env.ship.p[1] = hf.y0 + tall.y + tall.h - 2; env.ship.p[2] = tall.z; env.sense();
    assert.ok(env.obs[Math.floor(ENV.rays.nEl / 2) * ENV.rays.nAz + Math.floor(ENV.rays.nAz / 2)] > 0.5, city + ': the centre beam sees the tower');
  }
  let crashes = 0; for (const city of ['dubai', 'newyork', 'moscow']) for (let seed = 60; seed < 64; seed++) { const e = new SpaceEnv(seed, { atmosphere: false, count: 0, comets: 0, city, citySeed: seed % 5 }); for (let i = 0; i < 900; i++) if (e.step([0, 0, 0, 0]).done) { crashes++; break; } }
  assert.ok(crashes >= 6, 'straight flight runs into the skylines: ' + crashes + '/12');
});

test('birds: flocks wander above whatever stands below, stay in the corridor, are sensed, and a bird strike ends the episode', () => {
  const env = new SpaceEnv(70, { atmosphere: false, count: 0, comets: 0, city: 'newyork', birds: 4 });
  assert.equal(env.asteroids.filter((a) => a.kind === 4).length, 4, 'four flocks');
  for (let i = 0; i < 300; i++) {
    const r = env.step([0, 0, 0, 0]);
    for (const f of env.asteroids) if (f.kind === 4) { assert.ok(f.p[1] >= env.hf.height(f.p[0], f.p[2]) + 2.9, 'a flock stays above the roofs'); assert.ok(Math.abs(f.p[2]) <= ENV.zHalf && f.p[1] <= Math.max(ENV.mountains.ceiling, env.hf.height(f.p[0], f.p[2]) + 3.5), 'inside the corridor (z ' + f.p[2].toFixed(1) + ', y ' + f.p[1].toFixed(1) + ')'); assert.ok(Math.hypot(f.v[0], f.v[2]) > 2, 'keeps flying'); }
    if (r.done) env.reset();
  }
  env.reset(); const f = env.asteroids.find((a) => a.kind === 4); env.ship.q.set([0, 0, 0, 1]); qAxes(env.ship.q, env.ship.f, env.ship.u, env.ship.r);
  env.ship.p[1] = ENV.mountains.ceiling - 2; f.p[0] = env.ship.p[0] + 10; f.p[1] = env.ship.p[1]; f.p[2] = env.ship.p[2]; f.v.set([0.5, 0, 0]); env.sense();
  assert.ok(env.obs[Math.floor(ENV.rays.nEl / 2) * ENV.rays.nAz + Math.floor(ENV.rays.nAz / 2)] > 0.5, 'the centre beam sees the flock');
  let done = false; for (let i = 0; i < 30 && !done; i++) done = env.step([0, 0, 0, 0]).done;
  assert.ok(done, 'flying straight into the flock ends the episode');
});

test('pillars: the Zhangjiajie field has its summit at the ceiling, spawns clear of the ground, and straight flight hits the pillars', () => {
  const env = new SpaceEnv(80, { atmosphere: false, count: 0, comets: 0, pillars: true });
  assert.equal(env.world, 'pillars'); assert.equal(env.hf.style, 'pillars');
  assert.ok(Math.abs(env.hf.peak - (ENV.mountains.y0 + ENV.mountains.amplitude)) < 0.01, 'summit at amplitude ' + env.hf.peak); assert.ok(env.groundClearance() > 3, 'spawns clear ' + env.groundClearance());
  let crashes = 0; for (let seed = 80; seed < 86; seed++) { const e = new SpaceEnv(seed, { atmosphere: false, count: 0, comets: 0, pillars: true, pillarSeed: 5 + seed % 5 }); for (let i = 0; i < 900; i++) if (e.step([0, 0, 0, 0]).done) { crashes++; break; } }
  assert.ok(crashes >= 3, 'straight flight hits the pillars: ' + crashes + '/6');
});

test('megacity: a long map slides under the corridor lap by lap, continuous across the seam, different each lap', async () => {
  const { registerLongGrid, createCityField } = await import('../src/cityfield.js');
  registerLongGrid('mega', await import('../src/mega_grid.js'));
  const env = new SpaceEnv(7, { atmosphere: false, count: 0, comets: 0, city: 'mega' }), hf = env.hf;
  assert.equal(hf.segments, 8, 'eight corridor laps per map'); assert.equal(hf.PERIOD, 960);
  assert.ok(hf.peak - hf.y0 > ENV.mountains.ceiling, 'towers reach the thin air');
  // continuity: the column just ahead of the seam in lap k is the column just behind it in lap k + 1
  let worst = 0;
  for (let k = 0; k < 8; k++) for (let z = -40; z <= 40; z += 4) {
    hf.lap.shift = k * 120; const a = hf.height(59.75, z); hf.lap.shift = ((k + 1) % 8) * 120; const b = hf.height(-60.25, z); worst = Math.max(worst, Math.abs(a - b));
  }
  assert.ok(worst < 1e-6, 'the map is continuous across the wrap: ' + worst);
  // the environment advances the shift when the ship wraps, and different laps are different city
  env.reset(); const s0 = hf.lap.shift; env.ship.p[0] = ENV.xHalf - 0.01; env.ship.p[1] = 20; env.ship.v.set([14, 0, 0]); env.step([0, 0, 0, 0]);
  assert.equal(hf.lap.shift, (s0 + 120) % 960, 'one lap on: ' + s0 + ' → ' + hf.lap.shift);
  let differ = 0; for (let z = -40; z <= 40; z += 2) for (let x = -60; x < 60; x += 2) { hf.lap.shift = 0; const a = hf.height(x, z); hf.lap.shift = 360; if (Math.abs(a - hf.height(x, z)) > 2) differ++; }
  assert.ok(differ > 200, 'lap 0 and lap 3 are different districts: ' + differ);
  const f2 = createCityField({ city: 'mega' }); assert.deepEqual(f2.grid, hf.grid, 'deterministic');
});

test('edge guard: heading for a side wall, the ship turns away on its own before the wall', () => {
  const env = new SpaceEnv(4, { count: 0, comets: 0 }), s = env.ship;
  qFromAxisAngle(0, 1, 0, -0.6, s.q); qAxes(s.q, s.f, s.u, s.r);   // 34° toward +z (right)
  s.p.set([0, 0, 9]); for (let i = 0; i < 3; i++) s.v[i] = ENV.ship.cruise * s.f[i];
  let maxZ = 0, engaged = 0; for (let k = 0; k < 45; k++) { env.step([0, 0, 0, 0]); maxZ = Math.max(maxZ, s.p[2]); engaged = Math.max(engaged, env.guard); }
  assert.ok(engaged > 0.5, 'the guard engaged ' + engaged);
  assert.ok(maxZ < ENV.zHalf + ENV.ship.wallClamp - 0.3, 'it never reached the wall clamp ' + maxZ.toFixed(2));
  assert.ok(s.f[2] < 0.05, 'and ended up heading back into the corridor ' + s.f[2].toFixed(2));
});

test('edge guard: idle in the middle of the corridor, the policy has full control', () => {
  const env = new SpaceEnv(5, { count: 0, comets: 0 }); env.ship.p.set([0, 0, 0]);
  for (let k = 0; k < 10; k++) { env.step([0, 0, 0, 0]); assert.equal(env.guard, 0); }
});

test('tunnel guide: a rock closes the lane but for its bore, and the ship threads it on its own', () => {
  const env = new SpaceEnv(8, { count: 0, comets: 0 }), s = env.ship, y0 = -26, inRock = (x) => x > 20 && x < 50, bore = (z) => Math.abs(z) < 7;
  env.hf = { y0, PERIOD: 960, lap: { shift: 0 }, peak: y0 + 40, tunnels: [{ x0: 20, x1: 50, z: 0, y: 25, hw: 7, hh: 9 }],
    height: (x, z) => (inRock(x) ? y0 + (bore(z) ? 16 : 40) : y0), ceiling: (x, z) => (inRock(x) && bore(z) ? y0 + 34 : Infinity) };
  qFromAxisAngle(0, 1, 0, 0.1, s.q); qAxes(s.q, s.f, s.u, s.r); s.p.set([-50, 5, 10]); for (let i = 0; i < 3; i++) s.v[i] = ENV.ship.cruise * s.f[i];
  let mid = null, crashed = false, maxX = s.p[0];
  for (let k = 0; k < 160 && env.laps === 0; k++) { const r = env.step([0, 0, 0, 0]); if (r.done) { crashed = true; break; } if (env.laps === 0) maxX = Math.max(maxX, s.p[0]); if (!mid && s.p[0] > 35) mid = [s.p[1], s.p[2]]; }
  assert.ok(!crashed, 'no collision on the way through (x ' + s.p[0].toFixed(1) + ')');
  assert.ok(mid && Math.abs(mid[1]) < 4 && Math.abs(mid[0] - (y0 + 25)) < 5, 'on the bore\'s centre line inside the rock ' + JSON.stringify(mid && mid.map((v) => +v.toFixed(2))));
  assert.ok(maxX >= 55, 'out the other side ' + maxX.toFixed(1));
});
