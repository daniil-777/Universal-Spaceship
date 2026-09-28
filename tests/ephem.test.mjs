import { test } from 'node:test';
import assert from 'node:assert/strict';
import { julianDay, gmst, sunEcliptic, sunEci, moonEcliptic, moonEci, moonPhase, createSkyClock } from '../src/ephem.js';
import { circularOrbit, R_EARTH, MU } from '../src/orbit.js';

const DEG = Math.PI / 180, dAng = (a, b) => Math.abs(((a - b + 540) % 360) - 180);

test('Julian day and GMST (Meeus examples 7.a, 12.a)', () => {
  assert.ok(Math.abs(julianDay(Date.UTC(1957, 9, 4, 19, 26, 24)) - 2436116.31) < 1e-6, 'Sputnik: JD 2436116.31');
  assert.ok(dAng(gmst(2446895.5) / DEG, 197.693195) < 1e-4, 'GMST 1987 Apr 10, 0h UT = 13h10m46.3668s');
});

test('the Sun (Meeus example 25.a): 1992 Oct 13.0 TD — apparent λ 199.909°, δ −7.785°', () => {
  const jd = 2448908.5, s = sunEcliptic(jd); assert.ok(dAng(s.lambda / DEG, 199.90895) < 0.01, 'λ ' + s.lambda / DEG);
  const v = sunEci(jd), dec = Math.asin(v[2]) / DEG; assert.ok(Math.abs(Math.hypot(...v) - 1) < 1e-12 && Math.abs(dec + 7.78507) < 0.01, 'δ ' + dec);
});

test('the Moon (Meeus example 47.a): 1992 Apr 12.0 TD — λ 133.163°, β −3.229°, Δ 368 409.7 km (within 0.05° / 50 km)', () => {
  const m = moonEcliptic(2448724.5);
  assert.ok(dAng(m.lambda / DEG, 133.162655) < 0.05, 'λ ' + m.lambda / DEG); assert.ok(Math.abs(m.beta / DEG + 3.229126) < 0.05, 'β ' + m.beta / DEG);
  assert.ok(Math.abs(m.dist - 368409.7) < 50, 'Δ ' + m.dist); assert.ok(Math.abs(Math.hypot(...moonEci(2448724.5)) - m.dist) < 1e-6);
});

test('Moon phase: new near 2024-01-11 11:57 UT, first quarter near 01-18 03:52, full near 01-25 17:54', () => {
  const at = (ms) => { const jd = julianDay(ms); return moonPhase(sunEci(jd), moonEci(jd)); };
  assert.ok(at(Date.UTC(2024, 0, 11, 12)).fraction < 0.01, 'new moon'); assert.ok(at(Date.UTC(2024, 0, 25, 18)).fraction > 0.99, 'full moon');
  const q = at(Date.UTC(2024, 0, 18, 4)).fraction; assert.ok(Math.abs(q - 0.5) < 0.08, 'first quarter ' + q);
});

test('sky clock: real time by default, time-warp keeps the sky continuous', () => {
  const c = createSkyClock(Date.UTC(2026, 8, 28, 3, 0, 0)); assert.equal(c.now(0), Date.UTC(2026, 8, 28, 3, 0, 0)); assert.equal(c.now(10) - c.now(0), 10000);
  const before = c.now(20); c.setWarp(60, 20); assert.equal(c.now(20), before); assert.equal(c.now(21) - c.now(20), 60000);
});

test('ISS-like orbit: 420 km, 51.6° — 92.8 min per orbit at 7.66 km/s, the ground track within ±51.6°, J2 turns the node ≈ −5°/day', () => {
  const o = circularOrbit({ h: 420, inc: 51.6 * DEG, raan: 0.3, u0: 0.1 });
  assert.ok(Math.abs(o.period / 60 - 92.8) < 0.1, 'period ' + o.period / 60);   // Kepler at 6791 km (the ISS's 92.7 min is its ~410 km mean) assert.ok(Math.abs(o.speed - 7.66) < 0.01, 'speed ' + o.speed);
  let maxLat = 0; for (let t = 0; t < o.period; t += 20) { const r = o.stateEci(t).r; maxLat = Math.max(maxLat, Math.abs(Math.asin(r[2] / Math.hypot(...r)))); }
  assert.ok(Math.abs(maxLat / DEG - 51.6) < 0.1, 'max latitude ' + maxLat / DEG); assert.ok(Math.abs(o.raanRate * 86400 / DEG + 5.0) < 0.2, 'nodal drift ' + o.raanRate * 86400 / DEG);
  assert.ok(Math.abs(MU - 398600.4418) < 1e-9 && R_EARTH === 6371);
});

test('LVLH: orthonormal, x along the track, y radial up, z = x × y', () => {
  const o = circularOrbit({ h: 420, inc: 51.6 * DEG, raan: 1, u0: 2 }), f = o.lvlh(1234);
  for (const a of [f.x, f.y, f.z]) assert.ok(Math.abs(Math.hypot(...a) - 1) < 1e-12);
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; assert.ok(Math.abs(dot(f.x, f.y)) < 1e-12 && Math.abs(dot(f.x, f.z)) < 1e-12 && Math.abs(dot(f.y, f.z)) < 1e-12);
  const s = o.stateEci(1234); assert.ok(dot(f.y, s.r) > 0.999 * Math.hypot(...s.r) && dot(f.x, s.v) > 0.999 * Math.hypot(...s.v));
  const cz = [f.x[1] * f.y[2] - f.x[2] * f.y[1], f.x[2] * f.y[0] - f.x[0] * f.y[2], f.x[0] * f.y[1] - f.x[1] * f.y[0]]; assert.ok(Math.abs(dot(cz, f.z) - 1) < 1e-12);
});

test('phaseOver: the orbit is re-phased so the ship passes over Moscow (55.75° N) now, inclined just enough; Dubai keeps 51.6°', () => {
  const utc = Date.UTC(2026, 8, 28, 3), o = circularOrbit({ h: 420, inc: 51.6 * DEG, raan: 0, u0: 0 }).phaseOver(55.75 * DEG, 37.62 * DEG, utc);
  const sp = o.subPoint(0, gmst(julianDay(utc)));
  assert.ok(Math.abs(sp.lat / DEG - 55.75) < 0.01 && dAng(sp.lon / DEG, 37.62) < 0.01, `sub-point ${sp.lat / DEG}, ${sp.lon / DEG}`); assert.ok(Math.abs(o.inc / DEG - 56.25) < 1e-9);
  const eq = circularOrbit({ h: 420, inc: 51.6 * DEG, raan: 0, u0: 0 }).phaseOver(25.2 * DEG, 55.3 * DEG, utc); assert.ok(Math.abs(eq.inc / DEG - 51.6) < 1e-9, 'Dubai keeps 51.6°');
});

test('sky from the ship: the point on the globe under the ship is straight below, the Sun starts ~10° up ahead at dawn, the orbit speed reads 27 600 km/h', async () => {
  const { createSkyOrbit } = await import('../src/skyorbit.js'), sky = createSkyOrbit({ t0: Date.UTC(2026, 8, 28, 3), start: 'dawn' }), o = sky.update(0), A = o.axes;
  const p = [Math.cos(o.lat) * Math.cos(o.lon), Math.sin(o.lat), -Math.cos(o.lat) * Math.sin(o.lon)];   // the sub-point in the sphere's frame
  const w = [0, 1, 2].map((k) => A[0][k] * p[0] + A[1][k] * p[1] + A[2][k] * p[2]);                      // → the ship's frame
  assert.ok(Math.abs(w[1] - 1) < 1e-9, 'sub-point direction ' + w.map((v) => v.toFixed(4)));
  const el = Math.asin(o.sun[1]) / DEG; assert.ok(el > -25 && el < 35 && o.sun[0] > 0, `the Sun at ${el.toFixed(1)}° elevation, ahead ${o.sun[0].toFixed(2)}`);
  assert.ok(Math.abs(o.speed * 3600 - 27580) < 100, 'orbital speed ' + (o.speed * 3600).toFixed(0) + ' km/h');
  const lat0 = o.lat, lon0 = o.lon, later = sky.update(60), dd = Math.hypot(later.lat - lat0, (later.lon - lon0) * Math.cos(lat0));   // update() refills one object
  assert.ok(dd > 3 * DEG && dd < 4.5 * DEG, 'the ground track moves ≈ 3.7° per minute: ' + (dd / DEG).toFixed(2));
});

// The Moon from the Earth's orbit: the chase camera looks 13° down, 50° × 79°; the ship's hull covers ±6° around the centre.
const moonView = (o) => { const dip = Math.acos(6371 / (6371 + o.alt)); return { above: Math.asin(o.moon[1]) + dip, az: Math.atan2(o.moon[2], o.moon[0]), sep: Math.acos(o.sun[0] * o.moon[0] + o.sun[1] * o.moon[1] + o.sun[2] * o.moon[2]) }; };
test('Earth orbit: whenever the Moon is lit enough to see, it starts a few degrees over the limb beside the ship and stays in the frame for minutes', async () => {
  const { createSkyOrbit } = await import('../src/skyorbit.js'); let moonDays = 0, dawnDays = 0;
  for (let k = 0; k < 15; k++) {
    const t0 = Date.UTC(2026, 8, 28, 3) + k * 2 * 86400000, sky = createSkyOrbit({ t0 }), o = sky.update(0), v = moonView(o);
    if (o.moonFraction < 0.15) { dawnDays++; const el = Math.asin(o.sun[1]) / DEG; assert.ok(el > -25 && el < 35 && o.sun[0] > 0, `day ${2 * k}: a thin Moon keeps the dawn pass (Sun ${el.toFixed(1)}°)`); continue; }
    moonDays++;
    assert.ok(v.above > 2 * DEG && v.above < 7 * DEG, `day ${2 * k}: the Moon ${(v.above / DEG).toFixed(1)}° over the limb`);
    assert.ok(Math.abs(v.az) > 8 * DEG && Math.abs(v.az) < 22 * DEG, `day ${2 * k}: the Moon ${(v.az / DEG).toFixed(1)}° beside the ship`);
    const w = moonView(sky.update(300)); assert.ok(w.above > v.above && Math.asin(sky.out.moon[1]) < 12 * DEG, `day ${2 * k}: five minutes on it has risen and is still in the frame`);
  }
  assert.ok(moonDays >= 10 && dawnDays >= 1, `${moonDays} Moon starts, ${dawnDays} dawn starts in a month`);
});

test('Earth orbit: start = "dawn" forces the classic dawn pass, "moon" the Moon view; a waxing half Moon rises over sunlit ground, a waning one in the Earth\'s shadow', async () => {
  const { createSkyOrbit } = await import('../src/skyorbit.js'), t0 = Date.UTC(2026, 8, 28, 3);
  const d = createSkyOrbit({ t0, start: 'dawn' }).update(0), m = createSkyOrbit({ t0, start: 'moon' }).update(0);
  assert.ok(d.sun[0] > 0 && Math.asin(d.sun[1]) > -25 * DEG, 'dawn: the Sun ahead');
  const mv = moonView(m); assert.ok(mv.above > 2 * DEG && mv.above < 7 * DEG && Math.abs(mv.az) < 22 * DEG && m.moon[0] > 0, `moon: the Moon ${(mv.above / DEG).toFixed(1)}° over the limb, ${(mv.az / DEG).toFixed(1)}° off the nose`);
  const at = (k) => createSkyOrbit({ t0: t0 + k * 86400000, start: 'moon' }).update(0), fr = (k) => at(k).moonFraction;
  let half = null, wane = null; for (let k = 0; k < 30; k++) { if (Math.abs(fr(k) - 0.5) >= 0.12) continue; if (fr(k + 1) > fr(k)) half = half || at(k); else wane = wane || at(k); }
  assert.ok(wane, 'a waning half Moon within a month'); const wd = Math.acos(6371 / (6371 + wane.alt)); assert.ok(Math.asin(wane.sun[1]) > -wd - 12 * DEG, `waning: the Sun ≤ 12° under the ship's horizon (${(Math.asin(wane.sun[1]) / DEG).toFixed(1)}°) — the Moon's week of pre-dawn starts`);
  assert.ok(moonView(wane).above > 1.5 * DEG && Math.abs(moonView(wane).az) < 22 * DEG, 'waning: the Moon in the frame');
  const hv = moonView(half); assert.ok(hv.above > 2 * DEG && hv.above < 7 * DEG && Math.abs(hv.az) < 22 * DEG && half.moon[0] > 0, 'half Moon: in the frame');
  assert.ok(half, 'a waxing half Moon within a month');
  const R = 6371, r = R + half.alt, lit = [-36, -30, -24].map((e) => { const d = [Math.cos(e * DEG), Math.sin(e * DEG), 0], b = -r * d[1], t = b - Math.sqrt(b * b - (r * r - R * R));
    return Math.asin((t * d[0] * half.sun[0] + (t * d[1] + r) * half.sun[1]) / R); });
  assert.ok(lit.some((e) => e > 0), 'some of the ground ahead is in daylight: ' + lit.map((e) => (e / DEG).toFixed(1)).join(', '));
});

test('Earth orbit: the Moon start fits the view it will be seen in — a low pass at load (250 km) and a tall phone screen', async () => {
  const { createSkyOrbit } = await import('../src/skyorbit.js'), dip250 = Math.acos(6371 / 6621), half = Math.atan(Math.tan(25 * DEG) * 0.45);   // a 412 × 915 phone: ±11.9°
  for (let k = 0; k < 12; k++) {
    const t0 = Date.UTC(2026, 8, 28, 3) + k * 2.5 * 86400000, low = createSkyOrbit({ t0, viewH: 250 }), o = low.update(0); if (o.moonFraction < 0.15) continue;
    low.setAltitude(250); const w = low.update(0), above = Math.asin(w.moon[1]) + dip250;
    assert.ok(above > 1.5 * DEG && above < 8 * DEG, `day ${2.5 * k}: the Moon ${(above / DEG).toFixed(1)}° over the 250-km limb`);
    const p = createSkyOrbit({ t0, halfFov: half }).update(0), az = Math.abs(Math.atan2(p.moon[2], p.moon[0]));
    assert.ok(az > 6 * DEG && az < half - 2 * DEG, `day ${2.5 * k}: the Moon ${(az / DEG).toFixed(1)}° aside on a ±${(half / DEG).toFixed(1)}° screen`);
  }
});
