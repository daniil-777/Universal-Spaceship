// Real spacecraft S1: FINAL from H2 meets every IDSS gate (spec sections 3, 6 and 10).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { IDSS, DEG, SHIP_PORT, STATION_PORT, SHIP_SPHERES, STATION_CAPSULES, PORT_SUM, rFromRho, rhoFromR, portRel } from '../src/real/consts.js';
import { createRealSim, drawRun } from '../src/real/sim.js';
import { PH } from '../src/real/guidance.js';

test('ports and ranges: capture at r = 32.8 m, H2 at r = 52.8 m, H1 at rho = 217 m (one conversion, consts.js)', () => {
  assert.ok(Math.abs(PORT_SUM - 32.76) < 1e-9 && Math.abs(rFromRho(20) - 52.76) < 1e-9 && Math.abs(rhoFromR(250) - 217.24) < 1e-9);
  const x = [STATION_PORT[0] - SHIP_PORT[0], -SHIP_PORT[1], 0, 0, 0, 0];
  assert.ok(Math.hypot(...portRel(x, [0, 0, 0, 1])) < 1e-12, 'the ports meet at the capture pose');
  const seg = (c, a, b) => { const ab = b.map((v, i) => v - a[i]), L = ab.reduce((s, v) => s + v * v, 0); const s = Math.max(0, Math.min(1, c.reduce((t, v, i) => t + (v - a[i]) * ab[i], 0) / L)); return Math.hypot(...c.map((v, i) => v - a[i] - s * ab[i])); };
  let clear = Infinity;
  for (const [sx, sy, sz, r] of SHIP_SPHERES) for (const [a, b, rc] of STATION_CAPSULES) clear = Math.min(clear, seg([x[0] + sx, x[1] + sy, sz], a, b) - r - rc);
  assert.ok(clear > 1, `docked clearance ${clear} m: contact happens at the ports, not the hulls`);
});

test('50 dispersed FINALs from H2 capture inside every IDSS gate', () => {
  const lat = [];
  for (let s = 1; s <= 50; s++) {
    const sim = createRealSim({ seed: 500 + s, run: { ...drawRun(500 + s, { start: 'final' }), failed: [] } });
    let steady = 0;
    while (!sim.rep.done) {
      sim.step();
      if (sim.guid.st.phase === PH.FINAL) { const p = portRel(sim.x, sim.q); if (-p[0] < 10) steady = Math.max(steady, Math.hypot(p[1], p[2])); }
    }
    const c = sim.rep.contact;
    assert.equal(sim.rep.result, 'capture', `seed ${500 + s}: ${sim.rep.result} ${sim.rep.reason} ${JSON.stringify(c)}`);
    assert.ok(c.close >= IDSS.closeMin && c.close <= IDSS.closeMax && c.lat <= 0.010 && c.mis <= IDSS.mis && c.rate <= IDSS.rate / DEG && c.ang <= IDSS.ang / DEG, JSON.stringify(c));
    lat.push(steady);
  }
  // spec section 6: steady |e| <= 0.05 m by design (5 mm/s deadband x 10 s); nav noise adds ~1 cm at the port
  const sorted = [...lat].sort((a, b) => a - b);
  assert.ok(sorted[24] <= 0.05 && sorted[49] <= 0.08, `lateral error in the last 10 m: median ${sorted[24]}, max ${sorted[49]} m`);
});
