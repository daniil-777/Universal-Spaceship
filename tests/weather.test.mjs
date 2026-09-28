import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createWeather, cellShape, CELL } from '../src/weather.js';

const W0 = (o = {}) => createWeather({ seed: 11, severity: 0.5, period: 120, floor: -26, ceiling: 12, zHalf: 18, ...o });
const wind = (W, x, y, z) => W.windAt(x, y, z, new Float64Array(3));

test('weather: same seed → same cells and the same air', () => {
  const a = W0(), b = W0();
  assert.deepEqual(a.cells.map((c) => [c.type, c.x, c.z, c.R]), b.cells.map((c) => [c.type, c.x, c.z, c.R]));
  assert.deepEqual(Array.from(wind(a, 10, -5, 3)), Array.from(wind(b, 10, -5, 3)));
});
test('weather: air rises under a cumulus core and sinks in the ring around it', () => {
  const W = W0({ severity: 1 }), c = W.cells.find((k) => k.type === CELL.CUMULUS && k.W > 0);   // not a deck cell (no updraft)
  const core = wind(W, c.x, c.base - 3, c.z)[1], ring = wind(W, c.x + 1.4 * c.R, c.base - 3, c.z)[1];
  assert.ok(core > ring + 0.3 * c.W, `core ${core} ring ${ring} W ${c.W}`);
});
test('weather: severity brings more cells and more wind', () => {
  const calm = W0({ severity: 0 }), rough = W0({ severity: 1 });
  assert.ok(calm.cells.length < rough.cells.length); assert.equal(calm.U0, 2); assert.equal(rough.U0, 9);
});
test('weather: the wind grows with height above the ground', () => {
  const W = W0(), lo = wind(W, 0, -26 + 3, 150), hi = wind(W, 0, -26 + 30, 150);
  assert.ok(Math.hypot(hi[0], hi[2]) > Math.hypot(lo[0], lo[2]) * 1.3);
});
test('weather: air rises up a windward slope and sinks in the lee', () => {
  const hf = { height: (x) => -26 + 0.5 * x, lap: null };
  const up = createWeather({ seed: 3, severity: 0.5, period: 120, hf, floor: -26, ceiling: 12, zHalf: 18, windDir: 0 });
  const down = createWeather({ seed: 3, severity: 0.5, period: 120, hf, floor: -26, ceiling: 12, zHalf: 18, windDir: Math.PI });
  assert.ok(wind(up, 0, -20, 150)[1] > 0.5); assert.ok(wind(down, 0, -20, 150)[1] < -0.5);
});
test('weather: a towering core is violent, the air far from clouds is not', () => {
  const W = W0({ severity: 1 }), t = W.cells.find((k) => k.type === CELL.TOWERING);
  const core = W.sigmaAt(t.x, (t.base + Math.min(t.top, 40)) / 2, t.z), clear = W.sigmaAt(t.x, 60, 150);
  assert.ok(core > 2 && core > 2 * clear, `core ${core} clear ${clear}`);
});
test('weather: periodic in x (the loop has no seam)', () => {
  const W = W0({ severity: 1 }), a = wind(W, 17, -10, 4), b = wind(W, 17 + 120, -10, 4);
  for (let i = 0; i < 3; i++) assert.ok(Math.abs(a[i] - b[i]) < 1e-9);
});
test('weather: cellsNear returns every periodic copy within range, in render x', () => {
  const W = W0(), list = W.cellsNear(5, 200);
  assert.ok(list.length >= 3 * W.cells.length); for (const c of list) assert.ok(Math.abs(c.x - 5) <= 200 + 2 * c.R);
});
test('weather: clouds are dense inside a cell and absent far away', () => {
  const W = W0({ severity: 1 }), c = W.cells.find((k) => k.type === CELL.CUMULUS);
  assert.ok(W.cloudAt(c.x, (c.base + c.top) / 2, c.z) > 0.9); assert.equal(W.cloudAt(c.x, 200, c.z), 0);   // above every top (towering tops ≤ 82)
  assert.equal(cellShape(CELL.CUMULUS, 10, 0, 10, 0, 5, 0), 1);
});
test('weather: cells drift with the wind and the gust fronts stay bounded', () => {
  const W = W0({ windDir: 0 }); for (let i = 0; i < 20000; i++) W.advance(1 / 60);
  assert.ok(W.drift > 0 || W.drift < 0); assert.ok(W.front >= 0.7 && W.front <= 1.3); assert.ok(Math.abs(W.time - 20000 / 60) < 1e-6);
});
test('weather: cloudAt follows the lobes the renderer draws (edges pushed out and in by the noise)', () => {
  const W = W0({ severity: 0.3 }), c = W.cells.find((k) => k.type === CELL.CUMULUS && k.W > 0), y = c.base + 0.15 * (c.top - c.base);
  let out = 0, dent = 0;
  for (let i = 0; i < 90; i++) { const a = i / 90 * 2 * Math.PI, ca = Math.cos(a), sa = Math.sin(a);
    if (W.cloudAt(c.x + 1.12 * c.R * ca, y, c.z + 1.12 * c.R * sa) > 0.05) out++; if (W.cloudAt(c.x + 0.85 * c.R * ca, y, c.z + 0.85 * c.R * sa) < 0.05) dent++; }
  assert.ok(out > 0 && dent > 0, `lobes beyond the radius ${out}, dents inside it ${dent}`);
});
test('weather: a new severity or wind keeps the sky (same cells, same clock)', () => {
  const a = W0({ severity: 0.5 }); for (let i = 0; i < 600; i++) a.advance(1 / 60);
  const b = W0({ severity: 0.9, wind: 1.3, carry: a }), kind = (W, t, deck) => W.cells.filter((c) => c.type === t && (c.W === 0 && t === CELL.CUMULUS) === deck).map((c) => [c.x, c.z]);
  for (const [t, deck] of [[CELL.CUMULUS, false], [CELL.CUMULUS, true], [CELL.STRATUS, false], [CELL.TOWERING, false]]) {   // the rougher sky only adds cells
    const ka = kind(a, t, deck), kb = kind(b, t, deck), n = Math.min(ka.length, kb.length); assert.deepEqual(kb.slice(0, n), ka.slice(0, n)); }
  assert.equal(b.time, a.time); assert.equal(b.drift, a.drift); assert.equal(b.front, a.front); assert.equal(b.dir, a.dir);
});
test('weather: turbAt — σ, the cloud\'s share of σ² with its length scale (0.4 R), the height above the ground; sigmaAt is its σ', () => {
  const W = W0({ severity: 1 }), c = W.cells.find((k) => k.type === CELL.TOWERING), o = {}, y = c.base + 0.3 * (c.top - c.base);
  W.turbAt(c.x, y, c.z, o); assert.ok(o.share > 0.8 && Math.abs(o.Lc - 0.4 * c.R) < 1e-9, `share ${o.share}, L ${o.Lc}`);
  assert.ok(Math.abs(o.hA - (y + 26)) < 1e-9, 'hA ' + o.hA); assert.equal(W.sigmaAt(c.x, y, c.z), o.sigma);
  const far = (x) => W.cloudAt(x, 0, 0) === 0 && W.cells.every((k) => k.type !== CELL.TOWERING || Math.hypot(((x - k.x) % 120 + 180) % 120 - 60, k.z) > 2.5 * k.R);   // no cloud, no storm's shear
  let x = 0; while (!far(x) && x < 120) x += 1; assert.ok(far(x), 'a clear spot exists'); W.turbAt(x, 0, 0, o); assert.ok(o.share < 1e-6, 'clear-air share ' + o.share);
});
const tower = (o = {}) => { const W = W0({ severity: 1, ...o }); return [W, W.cells.find((k) => k.type === CELL.TOWERING)]; };
test('weather: a mature towering cell — air sinks in the rain shaft and rises in a ring around it at the base', () => {
  const [W, c] = tower(), core = wind(W, c.x, c.base - 4, c.z)[1], ring = wind(W, c.x + 0.8 * c.R, c.base, c.z)[1];
  assert.ok(core < -0.5 * c.W, `shaft ${core.toFixed(2)} (W ${c.W})`); assert.ok(ring > 0.5 * c.W, `ring ${ring.toFixed(2)}`);
  let hi = -1e9, lo = 1e9; for (let x = -2 * c.R; x <= 2 * c.R; x += 0.25) { const w = wind(W, c.x + x, 5, c.z)[1]; hi = Math.max(hi, w); lo = Math.min(lo, w); }
  assert.ok(hi - lo > 1.2 * c.W, `a pass through it at flight level swings ${(hi - lo).toFixed(2)} (W ${c.W})`);
});
test('weather: the downdraft spreads along the ground — an outflow ≥ 0.3 W 1–2 u up, mass conserved', () => {
  const [W, c] = tower(); let best = 0;
  for (let r = 0.1; r < 1.6; r += 0.05) for (const za of [1, 1.5, 2]) { const a = wind(W, c.x + r * c.R, -26 + za, c.z)[0], b = wind(W, c.x - r * c.R, -26 + za, c.z)[0]; best = Math.max(best, (a - b) / 2); }
  assert.ok(best > 0.3 * c.W, `outflow ${best.toFixed(2)} (W ${c.W})`);
  const e = 0.02, u = (x, y, z) => wind(W, x, y, z); let worst = 0;
  for (const r of [0, 0.1, 0.15]) for (const za of [0.5, 1, 2]) { const x = c.x + r * c.R, y = -26 + za, z = c.z + 0.3;
    const div = (u(x + e, y, z)[0] - u(x - e, y, z)[0] + u(x, y + e, z)[1] - u(x, y - e, z)[1] + u(x, y, z + e)[2] - u(x, y, z - e)[2]) / (2 * e); worst = Math.max(worst, Math.abs(div) * c.R / c.W); }
  assert.ok(worst < 0.05, 'divergence ' + worst.toFixed(3) + ' W/R');
});
test('weather: the shear around a towering cell\'s shaft is rough air', () => {
  const [W, c] = tower(), o = {}; W.turbAt(c.x + 0.6 * c.R, c.base - 4, c.z, o); const edge = o.sigma;
  let x = c.x + 3 * c.R; while (W.cloudAt(x, c.base - 4, c.z) > 0) x += 2; W.turbAt(x, c.base - 4, c.z, o);
  assert.ok(edge > 1.5 * o.sigma, `shaft edge ${edge.toFixed(2)} vs clear ${o.sigma.toFixed(2)}`);
});
test('weather: a cumulus\'s vigor sets its updraft, its roughness and its depth together (the radar can rank them)', () => {
  const Ws = [], S = [], D = []; for (let seed = 1; seed <= 300; seed++) for (const c of W0({ seed, severity: 0.7 }).cells) if (c.type === CELL.CUMULUS && c.W > 0) { Ws.push(c.W); S.push(c.sig); D.push(c.top - c.base); }
  const corr = (a, b) => { const n = a.length, ma = a.reduce((s, v) => s + v, 0) / n, mb = b.reduce((s, v) => s + v, 0) / n; let ab = 0, aa = 0, bb = 0; for (let i = 0; i < n; i++) { ab += (a[i] - ma) * (b[i] - mb); aa += (a[i] - ma) ** 2; bb += (b[i] - mb) ** 2; } return ab / Math.sqrt(aa * bb); };
  assert.ok(corr(Ws, S) > 0.95 && corr(Ws, D) > 0.95, `corr W–σ ${corr(Ws, S).toFixed(2)}, W–depth ${corr(Ws, D).toFixed(2)}`);
});
test('weather: inside a cumulus it is at least twice as rough as the clear air beside it (severity 1); the clear air calms above the cloud base', () => {
  const W = W0({ severity: 1 }), o = {}; let rin = 0, rout = 0, n = 0;
  for (const c of W.cells) { if (c.type !== CELL.CUMULUS || c.W <= 0 || c.base > 11) continue; const y = c.base + 0.3 * (c.top - c.base); W.turbAt(c.x, y, c.z, o); rin += o.sigma;
    let x = c.x + 2.2 * c.R; while (W.cloudAt(x, y, c.z) > 0) x += 2; W.turbAt(x, y, c.z, o); rout += o.sigma; n++; }
  assert.ok(n > 0 && rin > 2 * rout, `cumulus ${(rin / n).toFixed(2)} vs clear ${(rout / n).toFixed(2)}`);
  const still = W0({ severity: 1, wind: 0, cover: 0 }); still.turbAt(0, 4, 0, o); const low = o.sigma; still.turbAt(0, 11, 0, o);
  assert.ok(Math.abs(o.sigma / low - 0.5) < 0.02, `above the bases ${o.sigma.toFixed(3)} vs below ${low.toFixed(3)}`);
});
test('weather: the wind speeds up over a crest (2 u above a 30-u ridge vs the valley floor at the same height)', () => {
  const hf = { height: (x, z) => -26 + 30 * Math.exp(-(((x - 60) / 12) ** 2)) }, W = W0({ hf, cover: 0 });
  const h = (v) => Math.hypot(v[0], v[2]), crest = h(wind(W, 60, 6, 0)), valley = h(wind(W, 0, -24, 0));
  assert.ok(crest > 1.2 * valley, `crest ${crest.toFixed(2)} vs valley ${valley.toFixed(2)}`);
});
test('weather: gust fronts come and go over about a minute; no cover, no clouds', () => {
  const W = W0(), h = 1 / 15, N = 300000, f = new Float64Array(N); for (let i = 0; i < N; i++) { W.advance(h); f[i] = W.front; }
  let m = 0; for (const v of f) m += v; m /= N; let v0 = 0; for (const v of f) v0 += (v - m) ** 2;
  let lag = 0; for (; lag < 3000; lag += 15) { let c = 0; for (let i = 0; i + lag < N; i++) c += (f[i] - m) * (f[i + lag] - m); if (c / v0 < Math.exp(-1)) break; }
  assert.ok(lag * h > 40 && lag * h < 80, `1/e time ${(lag * h).toFixed(0)} s`);
  assert.equal(W0({ cover: 0 }).cells.length, 0);
});

test('weather: the turbulence knob scales σ (the Playbox and ?turb=): ×2 doubles it, 0 calms the air, 1 is today\'s sky', () => {
  const pts = [[10, -5, 3], [40, 2, -8], [77, 8, 12]], sig = (W) => pts.map(([x, y, z]) => W.sigmaAt(x, y, z));
  const base = sig(W0()), one = sig(W0({ turb: 1 })), two = sig(W0({ turb: 2 })), zero = sig(W0({ turb: 0 }));
  assert.deepEqual(one, base);
  two.forEach((v, i) => assert.ok(Math.abs(v - 2 * base[i]) < 1e-12, `×2 at point ${i}: ${v} vs ${base[i]}`));
  zero.forEach((v) => assert.equal(v, 0));
});
