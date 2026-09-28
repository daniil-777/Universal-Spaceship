import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Dryden, turbScales, SPAN } from '../src/turbulence.js';
import { mulberry32 } from '../src/mathx.js';

const H = 1 / 60, V = 14;
const T = (sigma, hA = 12, share = 0, Lc = 0) => ({ sigma, hA, share, Lc });
const run = (d, n, t, each, hx = 1, hz = 0) => { const out = new Float64Array(3); for (let i = 0; i < n; i++) { d.step(H, V, t, hx, hz, out); each(out, i); } };

test('length scales: MIL-F-8785C low altitude (L_w = h, L_u = L_v = h/(0.177+0.000823h)^1.2, h in feet), 1750 ft from 2000 ft up', () => {
  const s = turbScales(5, 0, 0, {}), k = 0.177 + 0.0513 * 5;
  assert.ok(Math.abs(s.Lw - 5) < 1e-9 && Math.abs(s.Lu - 5 / k ** 1.2) < 1e-9, `L at 5 u: ${s.Lw}, ${s.Lu}`);
  assert.ok(Math.abs(s.ru - k ** -0.4) < 1e-9 && s.ru > 1.3, 'σ_u/σ_w near the ground ' + s.ru);
  const hi = turbScales(40, 0, 0, {}); assert.ok(Math.abs(hi.Lw - 28) < 1e-9 && Math.abs(hi.Lu - 28) < 1e-9 && Math.abs(hi.ru - 1) < 1e-9, 'aloft: 28 u, isotropic');
  const mid = turbScales(24, 0, 0, {}); assert.ok(mid.Lw > 16 && mid.Lw < 28 && mid.Lu > 16 && mid.Lu < 28, 'blended between 1000 and 2000 ft');
  assert.ok(turbScales(0.2, 0, 0, {}).ru <= 1.7, 'σ_u/σ_w capped at the ground');
  const cloud = turbScales(30, 1, 6, {}); assert.ok(Math.abs(cloud.Lw - 6) < 1e-9 && Math.abs(cloud.Lu - 6) < 1e-9 && Math.abs(cloud.ru - 1) < 1e-9, 'inside a cloud: its own scale, isotropic');
});

test('Dryden: stationary RMS is σ_w vertically and r_u·σ_w along and across the track', () => {
  const d = new Dryden(mulberry32(3)), t = T(2, 5), ru = turbScales(5, 0, 0, {}).ru, s2 = [0, 0, 0], N = 400000;
  run(d, 3000, t, () => {});
  run(d, N, t, (o) => { for (let k = 0; k < 3; k++) s2[k] += o[k] * o[k]; });
  const want = [2 * ru, 2, 2 * ru]; for (let k = 0; k < 3; k++) { const rms = Math.sqrt(s2[k] / N); assert.ok(Math.abs(rms / want[k] - 1) < 0.08, `axis ${k}: RMS ${rms.toFixed(3)} vs ${want[k].toFixed(3)}`); }
});

test('Dryden: the vertical gust has Dryden\'s autocorrelation (1 − τV/2L)·e^(−τV/L), not a first-order one', () => {
  const d = new Dryden(mulberry32(8)), t = T(1, 12), L = turbScales(12, 0, 0, {}).Lw, N = 500000, w = new Float64Array(N);
  run(d, 2000, t, () => {}); run(d, N, t, (o, i) => { w[i] = o[1]; });
  let v = 0; for (let i = 0; i < N; i++) v += w[i] * w[i]; v /= N;
  for (const m of [0.5, 1, 2]) {
    const lag = Math.round(m * L / V / H), tau = lag * H; let c = 0; for (let i = 0; i + lag < N; i++) c += w[i] * w[i + lag]; c /= (N - lag) * v;
    const want = (1 - tau * V / (2 * L)) * Math.exp(-tau * V / L); assert.ok(Math.abs(c - want) < 0.05, `R(${m} L/V) = ${c.toFixed(3)}, Dryden ${want.toFixed(3)}`);
  }
});

test('Dryden: the intensity is applied at the output — a rough patch is felt as soon as it is entered and gone as soon as it is left', () => {
  const box = (x) => (x >= 0 && x < 30 ? 4 : 0.6), at = { in: [], out: [] }, t = T(0, 12);
  for (let seed = 1; seed <= 3000; seed++) {
    const d = new Dryden(mulberry32(seed)), out = new Float64Array(3); d.reset();
    for (let x = -20; x < 32; x += V * H) { t.sigma = box(x); d.step(H, V, t, 1, 0, out); if (x >= 1 && x < 1 + V * H) at.in.push(out[1]); if (x >= 31 && x < 31 + V * H) at.out.push(out[1]); }
  }
  const rms = (a) => Math.sqrt(a.reduce((s, v) => s + v * v, 0) / a.length);
  assert.ok(Math.abs(rms(at.in) / 4 - 1) < 0.1, '1 u inside: RMS ' + rms(at.in).toFixed(2)); assert.ok(Math.abs(rms(at.out) / 0.6 - 1) < 0.1, '1 u after: RMS ' + rms(at.out).toFixed(2));
});

test('Dryden: a reset draws stationary gusts (no calm first second), the same seed gives the same gusts, σ = 0 gives none', () => {
  let s2 = 0; const n = 4000; for (let seed = 1; seed <= n; seed++) { const d = new Dryden(mulberry32(seed)), o = new Float64Array(3); d.reset(); d.step(H, V, T(1, 12), 1, 0, o); s2 += o[1] * o[1]; }
  assert.ok(Math.abs(Math.sqrt(s2 / n) - 1) < 0.08, 'first-step RMS ' + Math.sqrt(s2 / n).toFixed(3));
  const a = new Dryden(mulberry32(9)), b = new Dryden(mulberry32(9)), oa = new Float64Array(3), ob = new Float64Array(3);
  for (let i = 0; i < 100; i++) { a.step(H, V, T(0), 1, 0, oa); b.step(H, V, T(0), 1, 0, ob); for (const v of oa) assert.ok(v === 0, 'gust ' + v); assert.ok(a.p === 0, 'rolling gust ' + a.p); }
  for (let i = 0; i < 500; i++) { a.step(H, V, T(1.5, 8), 0.6, 0.8, oa); b.step(H, V, T(1.5, 8), 0.6, 0.8, ob); }
  assert.deepEqual(Array.from(oa), Array.from(ob)); assert.equal(a.p, b.p);
});

test('Dryden: the along-track channel follows the heading it is given (the air-relative track)', () => {
  const d = new Dryden(mulberry32(4)), out = new Float64Array(3), t = T(1, 20), ru = turbScales(20, 0, 0, {}).ru;
  d.step(H, V, t, 0, 1, out);                                          // flying toward +z through the air: u → +z, v → −x
  assert.ok(Math.abs(out[2] - ru * d.x[0]) < 1e-12, 'u along +z'); assert.ok(out[0] < 0 === d.x[1] * Math.sqrt(3) + (1 - Math.sqrt(3)) * d.x[2] > 0, 'v along −x');
});

test('Dryden: the rolling gust (spanwise gradient of the vertical gust) has MIL-F-8785C\'s intensity', () => {
  const d = new Dryden(mulberry32(12)), t = T(1, 12), L = turbScales(12, 0, 0, {}).Lw, N = 300000; let s2 = 0;
  run(d, 2000, t, () => {}); run(d, N, t, () => { s2 += d.p * d.p; });
  const want = Math.sqrt(0.8 * Math.PI ** 2 * Math.cbrt(Math.PI / (4 * SPAN)) / (8 * SPAN * L ** (2 / 3)));
  assert.ok(Math.abs(Math.sqrt(s2 / N) / want - 1) < 0.08, `p_g RMS ${Math.sqrt(s2 / N).toFixed(4)} vs ${want.toFixed(4)}`);
});
