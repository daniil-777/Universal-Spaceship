import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseMetarWind, createWind, TURB_W20 } from '../src/landing/wind.js';

const KT = 0.514444;
test('METAR wind groups parse (direction, speed, gust; calm; variable)', () => {
  assert.deepEqual(parseMetarWind('23015G25'), { dir: 230, kt: 15, gust: 25 });
  assert.deepEqual(parseMetarWind('26008KT'), { dir: 260, kt: 8, gust: 0 });
  assert.deepEqual(parseMetarWind('00000'), { dir: 0, kt: 0, gust: 0 });
  const v = parseMetarWind('VRB03'); assert.equal(v.kt, 3); assert.equal(v.dir, null);
  assert.equal(parseMetarWind('nonsense'), null);
});

test('the mean wind: METAR speed at 10 m, a log profile (weaker near the ground, ~1.6× at 1000 ft), in runway components', () => {
  const w = createWind({ dir: 350, kt: 20, turb: 'none', rwyHdg: 260 }), o = new Float64Array(3);
  w.mean(10, o); assert.ok(Math.abs(Math.hypot(o[0], o[2]) - 20 * KT) < 0.05, 'at 10 m ' + Math.hypot(o[0], o[2]));
  assert.ok(Math.abs(o[0]) < 0.05 && o[2] < -10 * KT, 'from the right: blows toward −z ' + Array.from(o).map((v) => v.toFixed(2)));   // 350 is 90° right of 260
  w.mean(1, o); const low = Math.hypot(o[0], o[2]); w.mean(300, o); const high = Math.hypot(o[0], o[2]);
  assert.ok(low < 0.65 * 20 * KT && high > 1.45 * 20 * KT && high < 1.75 * 20 * KT, `1 m ${low.toFixed(2)}, 300 m ${high.toFixed(2)}`);
  const head = createWind({ dir: 260, kt: 10, turb: 'none', rwyHdg: 260 }); head.mean(10, o); assert.ok(o[0] < -5 && Math.abs(o[2]) < 1e-9, 'a headwind blows toward −x');
  const c = head.components(10); assert.ok(Math.abs(c.head - 10) < 1e-6 && Math.abs(c.cross) < 1e-6, 'components in kt');
});

test('turbulence: σ_w ≈ 0.1·W20 below 1000 ft for moderate (W20 30 kt), none for "none"; the same seed gives the same air', () => {
  const run = (turb, seed) => { const w = createWind({ dir: 260, kt: 10, turb, seed, rwyHdg: 260 }), o = new Float64Array(3), m = new Float64Array(3), p = [0, 150, 0]; let s2 = 0; const N = 200000;
    for (let i = 0; i < N; i++) { w.step(1 / 120, 80, 1, 0, 150); w.at(p, i / 120, o); w.mean(150, m); s2 += (o[1] - m[1]) ** 2; } return Math.sqrt(s2 / N); };
  const mod = run('moderate', 3); assert.ok(Math.abs(mod / (0.1 * TURB_W20.moderate * KT) - 1) < 0.12, 'σ_w ' + mod.toFixed(3));
  assert.equal(run('none', 3), 0); assert.equal(run('light', 5), run('light', 5));
});

test('gusts: a G group makes the wind speed swing up to about the gust value', () => {
  const w = createWind({ dir: 260, kt: 15, gust: 30, turb: 'none', seed: 2, rwyHdg: 260 }), o = new Float64Array(3); let top = 0, low = 1e9;
  for (let i = 0; i < 120 * 600; i++) { w.step(1 / 120, 80, 1, 0, 10); w.at([0, 10, 0], i / 120, o); const s = Math.hypot(o[0], o[2]) / KT; top = Math.max(top, s); low = Math.min(low, s); }
  assert.ok(top > 24 && top < 36 && low < 12, `speed ${low.toFixed(1)}–${top.toFixed(1)} kt`);
});
