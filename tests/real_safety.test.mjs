// Real spacecraft S1: the safety filter, classes P1 (station/KOS, keep-in) and P4 (speed ball) (spec sections 6 and 10).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { N, KOS_R } from '../src/real/consts.js';
import { filterVelocity, constraints, barrier, speedLimit, axialLimit, project, BRAKE_SPEED } from '../src/real/safety.js';
import { mulberry32 } from '../src/mathx.js';

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const feasible = (v, halves, ball, tol = 1e-7) => halves.every((c) => dot(c.n, v) - c.b >= -tol) && (!ball || Math.hypot(...v) <= ball.R + tol);

test('P1: h = r - 200 barrier min(0.01 h, sqrt(0.04 h)); keep-in 3 km; P4: |v| <= min(2, 0.2 + 2n(r - 10))', () => {
  assert.equal(barrier(50), 0.5); assert.ok(Math.abs(barrier(10) - 0.1) < 1e-12); assert.ok(Math.abs(barrier(1e4) - 20) < 1e-12);
  assert.ok(Math.abs(speedLimit(250) - (0.2 + 2 * N * 240)) < 1e-12); assert.equal(speedLimit(3000), 2);
  const out = filterVelocity([2, 0, 0], [-250, 0, 0]);
  assert.ok(Math.abs(out.v[0] - 0.5) < 1e-7 && out.changed, `closing capped at 0.5 m/s at r = 250: ${out.v}`);
  const k = filterVelocity([-3, 0, 0], [-2990, 0, 0]);
  assert.ok(k.v[0] >= -barrier(10) - 1e-7, 'keep-in');
});

test('the projection (Dykstra, exact fallback) is the nearest feasible point (variational inequality)', () => {
  const rng = mulberry32(3);
  for (let k = 0; k < 200; k++) {
    const halves = [0, 1, 2].map(() => { const n = [rng() - 0.5, rng() - 0.5, rng() - 0.5]; return { n, b: -0.2 * rng(), cls: 'P1' }; });
    const ball = { R: 0.3 + rng() }, v0 = [3 * (rng() - 0.5), 3 * (rng() - 0.5), 3 * (rng() - 0.5)];
    const { v, residual } = project(v0, halves, ball);
    assert.ok(residual < 1e-7 && feasible(v, halves, ball));
    for (let s = 0; s < 50; s++) {
      const y = [2 * (rng() - 0.5), 2 * (rng() - 0.5), 2 * (rng() - 0.5)];
      if (!feasible(y, halves, ball, 0)) continue;
      assert.ok(dot([v0[0] - v[0], v0[1] - v[1], v0[2] - v[2]], [y[0] - v[0], y[1] - v[1], y[2] - v[2]]) <= 1e-6, 'not the nearest point');
    }
  }
});

test('10^4 random states: kept barriers hold to 1e-7 m/s, P1 is never dropped, every drop and brake is logged', () => {
  const rng = mulberry32(11), log = [];
  let brakes = 0, drops = 0;
  for (let k = 0; k < 10000; k++) {
    const r = 150 + 3000 * rng(), th = Math.acos(2 * rng() - 1), ph = 2 * Math.PI * rng();
    const x = [r * Math.sin(th) * Math.cos(ph), r * Math.cos(th), r * Math.sin(th) * Math.sin(ph)];
    const v0 = [4 * (rng() - 0.5), 4 * (rng() - 0.5), 4 * (rng() - 0.5)];
    const extra = rng() < 0.1 ? [{ n: [x[0] / r, x[1] / r, x[2] / r], b: 2.5, cls: 'P1' }] : [];
    const n0 = log.length, out = filterVelocity(v0, x, { extra }, log, k);
    if (out.brake) { brakes++; assert.equal(log.length, n0 + out.dropped.length + 1); assert.equal(log.at(-1).kind, 'brake'); assert.ok(Math.abs(Math.hypot(...out.v) - BRAKE_SPEED) < 1e-12); continue; }
    assert.ok(!out.dropped.includes('P1'));
    drops += out.dropped.length; assert.equal(log.length, n0 + out.dropped.length);
    const { list, ball } = constraints(x, { extra });
    assert.ok(feasible(out.v, list, out.dropped.includes('P4') ? null : ball), `state ${k}`);
  }
  assert.ok(brakes > 0 && drops > 0, `exercised: ${brakes} brakes, ${drops} drops`);
});

test('inside the KOS outside the corridor: brake away radially at 0.05 m/s', () => {
  const log = [], out = filterVelocity([0.1, 0, 0], [-150, 0, 0], {}, log, 42);
  assert.ok(out.brake && Math.abs(out.v[0] + 0.05) < 1e-12 && log[0].kind === 'brake' && log[0].t === 42);
});

test('the nominal CORRIDOR and FINAL commands pass the filter unchanged', () => {
  for (let rho = 217; rho >= 0; rho -= 1) {
    const x = [-(rho + 32.76), 0.99, 0], cmd = [rho >= 20 ? Math.min(0.2, Math.max(0.07, rho / 1000)) : 0.07, 0.004, -0.003];
    const out = filterVelocity(cmd, x, { inCorridor: true, axial: rho });
    assert.ok(!out.changed, `rho ${rho}: ${out.v}`);
  }
  assert.ok(axialLimit(0) >= 0.10 && axialLimit(0) < 0.12, 'hard-contact guard just above the IDSS 0.10 m/s');
  const h1 = filterVelocity([0.2, 0, 0], [-250, 0, 0], { inCorridor: true, axial: 217.2 });
  assert.ok(!h1.changed);
});
