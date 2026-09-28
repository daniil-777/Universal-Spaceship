import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createLandingSim, drawConditions } from '../src/landing/sim.js';
import { KT } from '../src/landing/vehicle.js';

const DEG = Math.PI / 180, fin = (wind, extra = {}) => ({ seed: 7, wind: { dir: 260, kt: 0, gust: 0, turb: 'none', ...wind }, start: { x: -14000, z: 0, h: 600, hdg: 0, V: 180 * KT, ...extra } });
const push = (a, m) => { if (a[a.length - 1] !== m) a.push(m); };   // the FMA's columns: combined, lateral, vertical
const land = (cond) => { const sim = createLandingSim(cond), modes = [], lats = [], verts = []; while (!sim.rep.done && sim.flight.t < 900) { sim.step(); const st = sim.gnc.st; push(modes, st.mode); push(lats, st.lat); push(verts, st.vert); } return { ...sim.rep, modes, lats, verts }; };

test('autoland, calm air, a 14-km final: lands in the touchdown zone on the centreline at 1–3.5 ft/s and stops on the runway', () => {
  const r = land(fin({})), t = r.td;
  assert.equal(r.result, 'landed'); assert.ok(t.x > 300 && t.x < 600, 'touchdown ' + t.x.toFixed(0) + ' m'); assert.ok(Math.abs(t.z) < 2, 'lateral ' + t.z.toFixed(1));
  assert.ok(t.sinkFps > 1 && t.sinkFps < 3.5, 'sink ' + t.sinkFps.toFixed(2)); assert.ok(t.pitch < 12.5 * DEG && Math.abs(t.bank) < 2 * DEG, 'attitude');
  assert.ok(r.stopX > t.x && r.stopX < 3000 && r.maxZ < 3, `stopped at ${r.stopX.toFixed(0)} m, rollout |z| ${r.maxZ.toFixed(1)}`);
});

test('autoland: the flight-mode sequence and the radio-altitude callouts come in order', () => {
  const r = land(fin({})), want = ['LOC', 'G/S', 'LAND', 'FLARE', 'RETARD', 'ROLLOUT', 'STOP'];
  let k = 0; for (const m of r.modes) if (m === want[k]) k++; assert.equal(k, want.length, 'modes ' + r.modes.join(' → '));
  const calls = r.calls.map((c) => c.label); for (const c of ['1000', '500', '100', 'MINIMUMS', '40', '30', '20', '10', 'RETARD']) assert.ok(calls.includes(c), 'call ' + c);
  assert.ok(calls.indexOf('1000') < calls.indexOf('500') && calls.indexOf('500') < calls.indexOf('MINIMUMS') && calls.indexOf('30') < calls.indexOf('10'), calls.join(','));
});

test('autoland: a 15-kt crosswind from the right is de-crabbed — small crab and bank, on the centreline, rolled out straight', () => {
  const r = land(fin({ dir: 350, kt: 15, turb: 'light' })), t = r.td;
  assert.equal(r.result, 'landed'); assert.ok(Math.abs(t.crab) < 5 * DEG && Math.abs(t.bank) < 7.5 * DEG, `crab ${(t.crab / DEG).toFixed(1)}°, bank ${(t.bank / DEG).toFixed(1)}°`);
  assert.ok(Math.abs(t.z) < 8.2 && r.maxZ < 10, `touchdown |z| ${Math.abs(t.z).toFixed(1)}, rollout ${r.maxZ.toFixed(1)}`);
});

test('autoland: gusts and moderate turbulence — stabilized at 1000 ft, no hard landing', () => {
  const r = land(fin({ dir: 230, kt: 18, gust: 28, turb: 'moderate' }));
  assert.equal(r.result, 'landed', r.result); assert.ok(r.gates[1000], 'stabilized'); assert.ok(r.td.sinkFps < 6, 'sink ' + r.td.sinkFps.toFixed(2));
});

test('autoland: from anywhere 16–30 km out it plans a Dubins path, captures the localizer and glideslope and lands; the same seed flies the same landing', () => {
  const a = land(drawConditions(12)), b = land(drawConditions(12));
  assert.equal(a.result, 'landed'); assert.deepEqual(a.lats.slice(0, 2), ['NAV', 'LOC'], a.lats.join(' → ')); assert.deepEqual(a.verts.slice(0, 3), ['ALT', 'GS', 'FLARE'], a.verts.join(' → '));
  assert.equal(a.td.x, b.td.x); assert.equal(a.td.sinkFps, b.td.sinkFps);
});

test('after the rollout the ship vacates by a rapid-exit taxiway (≤ 50 kt at the turn-off) and stops clear of the runway', () => {
  const r = land(fin({}));
  assert.equal(r.result, 'landed'); assert.ok(r.exit && ['A5', 'A6'].includes(r.exit.name), 'exit ' + JSON.stringify(r.exit));
  assert.ok(r.exit.kt <= 50 && r.exit.kt > 20, 'turn-off at ' + r.exit.kt.toFixed(0) + ' kt'); assert.ok(r.stopZ > 90 && r.stopZ < 210, 'stopped clear of the runway at z ' + r.stopZ.toFixed(0));
  assert.ok(r.lats.includes('EXIT'), r.lats.join(' → '));
});

test('a final start joins the localizer without a loop, as ATC vectors it, and the finals begin 8.4–10.3 NM out', () => {
  for (let seed = 1; seed <= 60; seed++) {
    const c = drawConditions(seed, { final: true }), d = createLandingSim(c).gnc.st.dub, st = c.start;
    const turn = d.word[1] === 'S' ? d.params[0] + d.params[2] : d.params[0] + d.params[1] + d.params[2];
    assert.ok(st.x <= -15500 && st.x >= -19000, `seed ${seed}: starts ${(-st.x / 1852).toFixed(1)} NM out`);
    assert.ok(turn < 150 * DEG, `seed ${seed}: ${d.word} turns ${(turn / DEG).toFixed(0)}° on the way to the localizer`);
  }
  const r = land(fin({})); assert.equal(r.result, 'landed'); assert.ok(r.td.t < 240, `touchdown ${r.td.t.toFixed(0)} s after starting 14 km out`);
});

test('a fast approach with a gusty tailwind needs ~1000 fpm on the glideslope — the V/S gate allows the ILS rate + 300 fpm (FSF: a briefed higher sink rate)', () => {
  const r = land(drawConditions(5284, { final: true }));   // 039/10G24: 10 kt of tailwind, V_app + 14 kt → ~960 fpm on a 3° path
  assert.equal(r.result, 'landed'); assert.ok(r.gates[500], 'stabilized at 500 ft');
});

test('a high start close in gets a path long enough to descend (the 3-to-1 rule; a longer final or a 360) and captures the glideslope at the beam, not from above', () => {
  for (const seed of [143, 183]) {                          // 5000 ft 16–17 km out: heading in (143), across the final (183)
    const r = land(drawConditions(seed));
    assert.equal(r.result, 'landed', `seed ${seed}`); assert.ok(r.gates[1000] && r.gates[500], `seed ${seed}: stabilized ${JSON.stringify(r.gates)}`);
  }
});
