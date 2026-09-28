// Real spacecraft S1: attitude and translation control (spec sections 6 and 10).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { N, DEG, T_ORB, MASS0 } from '../src/real/consts.js';
import { createControl, MODES, BURN_TOL } from '../src/real/control.js';
import { createJets } from '../src/real/jets.js';
import { rigidStep, attError, omegaRel } from '../src/real/rigid.js';
import { qFromAxisAngle, qInvRotate, qRotate, qMul, mulberry32 } from '../src/mathx.js';
import { createNav } from '../src/real/nav.js';
import { propagate } from '../src/real/cw.js';

// closed attitude loop: nominal jets, gravity gradient on, no translation; returns per-cycle samples
function fly(q, w, seconds, cmd, onCycle = () => {}) {
  const jets = createJets(), ctrl = createControl({ jets: createJets() }), x = new Float64Array(6), F = new Float64Array(3), T = new Float64Array(3);
  let m = MASS0;
  for (let k = 0; k < Math.round(seconds / 0.1); k++) {
    const on = jets.schedule(ctrl.update(x, q, w, m, cmd));
    for (let s = 0; s < 10; s++) { const md = jets.active(on, s * 0.01, F, T); rigidStep(q, w, 0.01, { torque: T }); m -= md * 0.01; }
    onCycle((k + 1) * 0.1, attError(q, [0, 0, 0, 1]), omegaRel(q, w));
  }
  return { ctrl, prop: MASS0 - m };
}
const lvlhRate = (q) => { const w = new Float64Array(3); qInvRotate(q, [0, 0, -N], w); return w; };

test('mode table: primary 1 deg / 0.5 deg/s, vernier 0.3 deg (0.1 in FINAL) / 0.05 deg/s; rate deadbands > 1.25 x max-MIB pulse', () => {
  assert.equal(MODES.P.db, 1 * DEG); assert.equal(MODES.P.wmax, 0.5 * DEG); assert.equal(MODES.V.db, 0.3 * DEG); assert.equal(MODES.V.wmax, 0.05 * DEG);
  for (const k of ['P', 'V']) for (let i = 0; i < 3; i++) assert.ok(MODES[k].rdb[i] > 1.25 * MODES[k].alpha[i] * (k === 'P' ? 0.05 : 0.1) - 1e-12, `${k} axis ${i}`);
});

test('a perfect LVLH hold (inertial rate R(q)^T wL) needs no pulse for 600 s', () => {
  const q = Float64Array.from([0, 0, 0, 1]), w = lvlhRate(q);
  const { ctrl } = fly(q, w, 600, { mode: 'V' });
  assert.deepEqual(ctrl.stats.pulses, [0, 0, 0]);
});

test('30 deg acquisition in primary mode within 120 s (every axis within theta_db + 0.05 deg and staying there)', () => {
  const a = 1 / Math.sqrt(3), q = qFromAxisAngle(a, a, a, 30 * DEG), w = lvlhRate(q);
  let tIn = null;
  fly(q, w, 240, { mode: 'P' }, (t, e) => { const err = Math.max(...e.map(Math.abs)); if (err <= 1.05 * DEG) tIn ??= t; else tIn = null; });
  assert.ok(tIn !== null && tIn <= 120, `acquired at ${tIn}`);
});

test('one-orbit vernier hold: within theta_db + 0.05 deg, <= 6 pulses/min/axis, <= 2 kg', () => {
  const q = qMul(qFromAxisAngle(1, 0, 0, 0.1 * DEG), qFromAxisAngle(0, 0, 1, 0.2 * DEG)), w = lvlhRate(q); w[1] += 0.003 * DEG;
  const maxE = [0, 0, 0];
  const { ctrl, prop } = fly(q, w, T_ORB, { mode: 'V' }, (t, e) => { for (let i = 0; i < 3; i++) maxE[i] = Math.max(maxE[i], Math.abs(e[i])); });
  for (let i = 0; i < 3; i++) {
    assert.ok(maxE[i] <= 0.35 * DEG, `axis ${i}: ${maxE[i] / DEG} deg`);
    assert.ok(ctrl.stats.pulses[i] / (T_ORB / 60) <= 6, `axis ${i}: ${ctrl.stats.pulses[i]} pulses`);
  }
  assert.ok(prop <= 2, `${prop} kg`);
});

test('burn trims reach the target velocity within 1.2 mm/s per axis', () => {
  const jets = createJets(), ctrl = createControl({ jets: createJets() }), q = Float64Array.from([0, 0, 0, 1]), w = lvlhRate(q);
  const x = Float64Array.from([-250, 0, 0, 0, 0, 0]), F = new Float64Array(3), T = new Float64Array(3), a = new Float64Array(3), vDes = [0.02, 0.1, 0];
  for (let k = 0; k < 400; k++) {
    const on = jets.schedule(ctrl.update(x, q, w, MASS0, { vDes, burn: true, mode: 'P' }));
    for (let s = 0; s < 10; s++) { jets.active(on, s * 0.01, F, T); qRotate(q, F, a); for (let i = 0; i < 3; i++) x[3 + i] += (a[i] / MASS0) * 0.01; rigidStep(q, w, 0.01, { torque: T }); }
  }
  for (let i = 0; i < 3; i++) assert.ok(Math.abs(x[3 + i] - vDes[i]) <= BURN_TOL, `axis ${i}: ${x[3 + i]}`);
});

test('nav: truth passes through; noisy lidar at rho = 20 m tracks a coasting ship to < 3 cm and < 1 mm/s RMS', () => {
  const x0 = Float64Array.from([-52.76, 0.99, 0.3, 0.07, 0.001, -0.002]), t = createNav({ mode: 'truth' });
  assert.deepEqual(Array.from(t.update(x0, 20)), Array.from(x0));
  const nav = createNav({ mode: 'noisy', rng: mulberry32(5) }); let ep = 0, ev = 0, n = 0;
  for (let k = 0; k <= 6000; k++) {
    const x = propagate(x0, k * 0.1), xh = nav.update(x, 20);
    if (k < 600) continue;
    for (let i = 0; i < 3; i++) { ep += (xh[i] - x[i]) ** 2; ev += (xh[3 + i] - x[3 + i]) ** 2; }
    n++;
  }
  assert.ok(Math.sqrt(ep / (3 * n)) < 0.03 && Math.sqrt(ev / (3 * n)) < 0.001, `rms ${Math.sqrt(ep / (3 * n))} m, ${Math.sqrt(ev / (3 * n))} m/s`);
});
