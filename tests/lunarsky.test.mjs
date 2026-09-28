import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createLunarSky, MU_MOON, R_MOON } from '../src/lunarsky.js';
import { julianDay, moonEci } from '../src/ephem.js';

const DEG = Math.PI / 180, T0 = Date.UTC(2026, 8, 28, 9, 0, 0), len = (v) => Math.hypot(...v), dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

test('lunar orbit: circular speed, period and the horizon dip at 1500 km (μ☾ 4902.8 km³/s², R☾ 1737.4 km)', () => {
  const L = createLunarSky({ t0: T0, h: 1500 }), o = L.update(0);
  assert.ok(Math.abs(o.speed - Math.sqrt(MU_MOON / (R_MOON + 1500))) < 1e-9 && Math.abs(o.speed - 1.2306) < 1e-3, `v ${o.speed}`);
  assert.equal(o.alt, 1500); assert.equal(o.body, 'moon'); assert.equal(o.radiusKm, R_MOON);
  assert.ok(Math.abs(o.dip / DEG - 57.54) < 0.05, `dip ${o.dip / DEG}°`);
  assert.ok(Math.abs(L.period - 2 * Math.PI * Math.sqrt((R_MOON + 1500) ** 3 / MU_MOON)) < 1e-6, 'period');
});

test('lunar orbit: every direction is a unit vector in the ship frame, the Earth globe\'s axes are orthonormal', () => {
  const o = createLunarSky({ t0: T0, h: 1500 }).update(0);
  for (const v of [o.sun, o.earth, o.north, o.moonToEarth]) assert.ok(Math.abs(len(v) - 1) < 1e-9);
  const A = o.earthAxes; for (let i = 0; i < 3; i++) { assert.ok(Math.abs(len(A[i]) - 1) < 1e-9); for (let j = i + 1; j < 3; j++) assert.ok(Math.abs(dot(A[i], A[j])) < 1e-9); }
  assert.ok(o.earthFraction >= 0 && o.earthFraction <= 1);
});

// The chase camera's view of the backdrop, worked out here independently: world = R_z(pitch)·LVLH; the camera looks 13° down.
const world = (v, th) => [Math.cos(th) * v[0] - Math.sin(th) * v[1], Math.sin(th) * v[0] + Math.cos(th) * v[1], v[2]];
const groundSunAt = (o, elW) => {                       // the Sun's height over the ground seen straight ahead at world elevation elW
  const th = o.pitch, e = elW - th, d = [Math.cos(e), Math.sin(e), 0], r = R_MOON + o.alt, b = -r * d[1], disc = b * b - (r * r - R_MOON * R_MOON);
  if (disc < 0) return null; const t = b - Math.sqrt(disc), n = [t * d[0] / R_MOON, (t * d[1] + r) / R_MOON, 0]; return Math.asin(dot(n, o.sun));
};
test('lunar orbit: it starts over sunlit ground — the Sun low and raking where the chase camera looks, never in its frame — on any day of the month', () => {
  for (let k = 0; k < 10; k++) {
    const o = createLunarSky({ t0: T0 + k * 3 * 86400000, h: 1500 }).update(0), g = groundSunAt(o, -30 * DEG), sw = world(o.sun, o.pitch);
    assert.ok(Math.abs(o.pitch - Math.max(0, o.dip - Math.acos(6371 / 6791))) < 1e-12, 'the limb where the LEO Earth\'s is');
    assert.ok(g !== null && g > 8 * DEG && g < 55 * DEG, `day ${k * 3}: the Sun ${g && (g / DEG).toFixed(1)}° over the ground in view`);
    assert.ok(Math.acos(Math.cos(-13 * DEG) * sw[0] + Math.sin(-13 * DEG) * sw[1]) > 40 * DEG, `day ${k * 3}: the Sun out of the frame`);
  }
});

test('lunar orbit: the Earth is in the frame, above the lunar limb, whenever it stands well apart from the Sun', () => {
  let n = 0;
  for (let k = 0; k < 10; k++) {
    const t0 = T0 + k * 3 * 86400000, o = createLunarSky({ t0, h: 1500 }).update(0), sep = Math.acos(dot(o.sun, o.earth)); if (sep < 70 * DEG) continue; n++;
    const w = world(o.earth, o.pitch), el = Math.asin(w[1]), az = Math.atan2(w[2], w[0]);
    assert.ok(Math.abs(az) < 36 * DEG && el > -36 * DEG && el < 10 * DEG, `day ${k * 3}: the Earth at el ${(el / DEG).toFixed(1)}°, az ${(az / DEG).toFixed(1)}°`);
    assert.ok(Math.asin(o.earth[1]) > -o.dip, `day ${k * 3}: the Earth above the limb`);
  }
  assert.ok(n >= 3, `${n} days with the Earth ≥ 70° from the Sun`);
  const dEM = len(moonEci(julianDay(T0))), o = createLunarSky({ t0: T0, h: 1500 }).update(0);
  assert.ok(Math.abs(o.earthDist - dEM) <= R_MOON + 1500 + 1, `Earth ${o.earthDist.toFixed(0)} km vs ${dEM.toFixed(0)} km`);
});

test('lunar orbit: the phase advances at the mean motion (time warp included) and a new altitude keeps the ship where it is', () => {
  const L = createLunarSky({ t0: T0, h: 1500 }); const u0 = L.update(0).u; L.setWarp(100); const u1 = L.update(10).u;
  assert.ok(Math.abs((u1 - u0) - L.meanMotion * 1000) < 1e-9, 'Δu = n·Δt over 1000 s of sky time');
  const r0 = L.update(10).rHat.slice(); L.setAltitude(5000); const o = L.update(10);
  assert.ok(Math.abs(o.speed - 0.8531) < 1e-3 && o.alt === 5000, `v ${o.speed}`);
  assert.ok(dot(o.rHat, r0) > 1 - 1e-12, 'same place over the Moon');
});
