import { test } from 'node:test';
import assert from 'node:assert/strict';
import { AIRPORT, RWY } from '../src/landing/airport.js';

const DEG = Math.PI / 180, marks = (kind, end = 26) => AIRPORT.markings.filter((m) => m.kind === kind && m.end === end);
const lights = (kind) => AIRPORT.lights.filter((l) => l.kind === kind);

test('runway 26: 4000 m × 60 m, threshold at x = 0, heading 260°', () => {
  assert.equal(RWY.length, 4000); assert.equal(RWY.width, 60); assert.equal(RWY.heading, 260);
});

test('markings (ICAO Annex 14, precision approach, 60 m wide): 16 threshold stripes, "26", centerline 30 m / 20 m gaps, aiming point at 400 m, TDZ 3-3-2-2-1-1', () => {
  for (const end of [26, 8]) {
    const th = marks('threshold', end); assert.equal(th.length, 16, 'threshold stripes ' + end);
    for (const m of th) { assert.ok(Math.abs(m.x1 - m.x0 - 30) < 1e-9 && Math.abs(m.z1 - m.z0 - 1.8) < 1e-9); }
    const zs = th.map((m) => (m.z0 + m.z1) / 2).sort((a, b) => a - b); assert.ok(Math.abs(zs[0] + zs[15]) < 1e-9 && zs[8] - zs[7] >= 3, 'symmetric, a central gap');
    const d = marks('designation', end); assert.equal(d.length, 1); assert.equal(d[0].text, end === 26 ? '26' : '08');
    const aim = marks('aiming', end); assert.equal(aim.length, 2); const a0 = end === 26 ? aim[0].x0 : 4000 - aim[0].x1; assert.ok(Math.abs(a0 - 400) < 1e-9 && Math.abs(aim[0].x1 - aim[0].x0 - 60) < 1e-9, 'aiming point at 400 m, 60 m long');
    const tdz = marks('tdz', end), at = (x) => tdz.filter((m) => Math.abs((end === 26 ? m.x0 : 4000 - m.x1) - x) < 1e-6).length;
    assert.deepEqual([150, 300, 600, 750, 900, 1050].map(at), [6, 6, 4, 4, 2, 2], 'TDZ stripes per pair');
  }
  const cl = marks('centerline').sort((a, b) => a.x0 - b.x0); assert.ok(cl.length > 70);
  for (let i = 1; i < cl.length; i++) { assert.ok(Math.abs(cl[i].x1 - cl[i].x0 - 30) < 1e-9); assert.ok(Math.abs(cl[i].x0 - cl[i - 1].x1 - 20) < 1e-9); }
});

test('PAPI: four units left of the runway, 9 m apart, the inner one 15 m from the edge, 300 m in, set 3°30′ … 2°30′ (inner highest)', () => {
  const p = AIRPORT.papi; assert.equal(p.length, 4);
  assert.deepEqual(p.map((u) => u.z), [-45, -54, -63, -72]); assert.ok(p.every((u) => u.x === 300));
  const want = [3.5, 3 + 10 / 60, 2 + 50 / 60, 2.5]; p.forEach((u, i) => assert.ok(Math.abs(u.angle / DEG - want[i]) < 1e-9, 'unit ' + i));
});

test('lights: edges every 60 m (yellow in the last 600 m), centerline every 15 m (red/white from 900 m to go, red over the last 300 m), TDZ barrettes over 900 m, green threshold, red end', () => {
  const edge = lights('edge'); assert.ok(edge.length >= 2 * 66); const yel = edge.filter((l) => l.color === 'yellow'); assert.ok(yel.every((l) => l.x >= 3400 - 1e-9) && yel.length >= 2 * 10);
  const cl = lights('centerline').sort((a, b) => a.x - b.x); for (let i = 1; i < cl.length; i++) assert.ok(Math.abs(cl[i].x - cl[i - 1].x - 15) < 1e-9);
  assert.ok(cl.filter((l) => l.x < 3100).every((l) => l.color === 'white')); assert.ok(cl.filter((l) => l.x > 3700).every((l) => l.color === 'red'));
  const mid = cl.filter((l) => l.x >= 3100 && l.x <= 3700); assert.ok(mid.some((l) => l.color === 'red') && mid.some((l) => l.color === 'white'));
  const tdz = lights('tdz'); assert.ok(tdz.length > 0 && tdz.every((l) => l.x > 0 && l.x <= 900 + 1e-9));
  assert.ok(lights('threshold').every((l) => l.color === 'green')); assert.ok(lights('end').every((l) => l.color === 'red'));
});

test('approach lights: 900 m of centerline barrettes every 30 m, the crossbar at 300 m, red side rows in the inner 270 m, sequenced flashers', () => {
  const c = lights('approach'); assert.ok(Math.min(...c.map((l) => l.x)) <= -900 + 1e-9);
  const xs = [...new Set(c.map((l) => l.x))].sort((a, b) => b - a); for (let i = 1; i < xs.length; i++) assert.ok(Math.abs(xs[i - 1] - xs[i] - 30) < 1e-9);
  assert.ok(lights('crossbar').every((l) => l.x === -300) && lights('crossbar').length >= 16);
  assert.ok(lights('siderow').every((l) => l.color === 'red' && l.x >= -270 - 1e-9 && l.x < 0));
  const fl = lights('flasher'); assert.ok(fl.length >= 15 && fl.every((l) => l.x <= -300 && l.x >= -900)); assert.ok(fl.every((l, i) => i === 0 || l.seq === fl[i - 1].seq + 1));
});

test('ILS: localizer 300 m past the far end, glideslope 3° from 300 m in (TCH ≈ 15.7 m), right of the runway', () => {
  const ils = AIRPORT.ils; assert.equal(ils.loc.x, 4300); assert.equal(ils.gs.angle, 3); assert.ok(ils.gs.z > 60);
  assert.ok(Math.abs(ils.gs.x * Math.tan(3 * DEG) - 15.72) < 0.1, 'TCH ' + ils.gs.x * Math.tan(3 * DEG));
});

test('airside: a parallel taxiway, two 30° rapid exits, an apron with stands, terminal, tower, hangars; nothing within 150 m of the centreline but taxiway links', () => {
  assert.ok(AIRPORT.taxiways.length >= 5); const exits = AIRPORT.taxiways.filter((t) => t.kind === 'rapid'); assert.equal(exits.length, 2);
  for (const e of exits) { const [a, b] = e.pts; assert.ok(Math.abs(Math.atan2(b[1] - a[1], b[0] - a[0]) / DEG - 30) < 1e-6, 'exit angle'); }
  assert.ok(AIRPORT.stands.length >= 8); for (const k of ['terminal', 'tower', 'hangar']) assert.ok(AIRPORT.buildings.some((b) => b.kind === k), k);
  for (const b of AIRPORT.buildings) assert.ok(Math.abs(b.z) - b.d / 2 > 150, `${b.kind} clear of the runway strip`);
});
