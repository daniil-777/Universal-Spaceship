// Real spacecraft S1: the rigid body (spec sections 4 and 10).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { N, T_ORB, INERTIA } from '../src/real/consts.js';
import { rigidStep, makeInertia, energy, momentum, omegaRel, attError } from '../src/real/rigid.js';
import { qFromAxisAngle } from '../src/mathx.js';

test('torque-free: |H| and energy drift < 1e-9 over one orbit at 100 ms; body z is the intermediate axis', () => {
  const I = makeInertia(INERTIA), q = Float64Array.from([0, 0, 0, 1]), w = Float64Array.from([1e-6, 1e-6, 0.05]);
  const E0 = energy(I, w), H0 = momentum(I, w), steps = Math.round(T_ORB / 0.1), tr = [];
  let dE = 0, dH = 0;
  for (let i = 1; i <= steps; i++) {
    rigidStep(q, w, 0.1, { inertia: I, gg: false });
    dE = Math.max(dE, Math.abs(energy(I, w) / E0 - 1)); dH = Math.max(dH, Math.abs(momentum(I, w) / H0 - 1));
    if (i % 10 === 0 && i <= 3500) tr.push(Math.hypot(w[0], w[1]));
  }
  assert.ok(dE < 1e-9 && dH < 1e-9, `drift E ${dE} H ${dH}`);
  // growth rate: log-slope of the transverse rate over the linear window 50-300 s (it saturates near 500 s)
  const rate = Math.log(tr[299] / tr[49]) / 250, lam = 0.05 * Math.sqrt(((INERTIA[2] - INERTIA[0]) * (INERTIA[1] - INERTIA[2])) / (INERTIA[0] * INERTIA[1]));
  assert.ok(Math.abs(lam - 0.0216) < 1e-4, `theory ${lam}`);
  assert.ok(Math.abs(rate - 0.0216) <= 0.0216 * 0.05, `growth ${rate}`);
});

test('LVLH hold is an equilibrium with gravity gradient; its inertial rate is (0, 0, -n)', () => {
  const q = Float64Array.from([0, 0, 0, 1]), w = Float64Array.from([0, 0, -N]);
  for (let i = 0; i < Math.round(T_ORB / 0.1); i++) rigidStep(q, w, 0.1);
  const e = attError(q, [0, 0, 0, 1]);
  assert.ok(Math.hypot(...e) < 1e-6, `error ${Math.hypot(...e)}`);
  assert.ok(Math.hypot(...omegaRel(q, w)) < 1e-9);
  assert.ok(Math.abs(N * 180 / Math.PI - 0.0646) < 1e-4, 'LVLH hold 0.0646 deg/s inertial');
});

test('gravity gradient: a pitch offset from LVLH hold diverges as cosh(lambda t), lambda = n sqrt(3 (Iyy - Ixx) / Izz)', () => {
  const q = qFromAxisAngle(0, 0, 1, 0.001), w = Float64Array.from([0, 0, -N]);
  for (let i = 0; i < 6000; i++) rigidStep(q, w, 0.1);
  const lam = N * Math.sqrt((3 * (INERTIA[1] - INERTIA[0])) / INERTIA[2]), want = 0.001 * Math.cosh(lam * 600);
  const got = attError(q, [0, 0, 0, 1])[2];
  assert.ok(Math.abs(got - want) < 0.02 * want, `pitch ${got} vs ${want}`);
});

test('attError is the body-axis rotation vector and omegaRel removes the LVLH rate', () => {
  const q = qFromAxisAngle(0, 1, 0, 0.3); const e = attError(q, [0, 0, 0, 1]);
  assert.ok(Math.abs(e[1] - 0.3) < 1e-12 && Math.abs(e[0]) < 1e-12 && Math.abs(e[2]) < 1e-12);
  const r = omegaRel(Float64Array.from([0, 0, 0, 1]), Float64Array.from([0.001, 0, -N]));
  assert.ok(Math.abs(r[0] - 0.001) < 1e-15 && Math.abs(r[2]) < 1e-15);
});
