// Real spacecraft S1: TRANSFER, Rule P / Rule A, GO and the near starts (spec sections 3, 6 and 10).
// Task 4 holds the pure planner tests; Task 7 replaces this file with the full one (adds the 6-DOF tests).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { N, T_ORB, DAY, DRAG_B, KOS_R, BREAKOUT_V } from '../src/real/consts.js';
import { propagate } from '../src/real/cw.js';
import { passiveMin, planTransfer, ruleP, breakoutOk, planBreakout } from '../src/real/passive.js';

const START = Float64Array.from([-2000, 0, 250, 0, 0, 0]);

test('nominal TRANSFER: half-orbit radial hop, 2 x 0.4936 m/s +- 1 %, a -Y burn and a 437.5 m dip', () => {
  const p = planTransfer(START);
  assert.ok(p.nominal && Math.abs(p.T - T_ORB / 2) < 1e-6);
  assert.ok(Math.abs(p.dv1 - 0.4936) <= 0.004936 && Math.abs(p.dv2 - 0.4936) <= 0.004936, `${p.dv1} ${p.dv2}`);
  assert.ok(p.v[1] < 0 && Math.abs(p.v[1] + (1750 * N) / 4) < 1e-6, 'the departure burn points down (-Y)');
  let dip = 0; for (let t = 0; t <= p.T; t += 5) dip = Math.min(dip, propagate(p.x, t)[1]);
  assert.ok(Math.abs(dip + 437.5) < 0.5, `dip ${dip}`);
});

test('Rule P after the hop: 24 h minimum r >= 240 m and 249.2 +- 0.5 m over a_d in [-b, b]; without cross-track < 20 m', () => {
  const p = planTransfer(START), pm = passiveMin(p.x, DAY, DRAG_B);
  assert.ok(pm.rMin >= 240 && Math.abs(pm.rMin - 249.2) <= 0.5, `rMin ${pm.rMin}`);
  const flat = Float64Array.from(p.x); flat[2] = 0;
  assert.ok(passiveMin(flat).rMin < 20, 'the cross-track phasing is what makes the hop safe');
});

test('T search: a start whose hop fails Rule P gets the least-dv T in [0.35, 0.6] orbit that passes', () => {
  const x = Float64Array.from([-2000, 0, 222, 0, 0, 0]);
  const hop = planTransfer(x, { rMin: 1e9 });
  assert.equal(hop, null, 'nothing passes an impossible bound');
  const p = planTransfer(x, { rMin: 250 });
  assert.ok(!p.nominal && p.T >= 0.35 * T_ORB - 1e-6 && p.T <= 0.6 * T_ORB + 1e-6, `T ${p.T / T_ORB}`);
  assert.ok(p.rMin >= 250 && ruleP(p.x, 250));
});

test('breakout criterion: from rho = 10 m on V-bar, v+ = (0.02, 0.10, 0) leaves the KOS within one orbit for good', () => {
  const x = Float64Array.from([-42.76, 0.99, 0, ...BREAKOUT_V]), ok = breakoutOk(x);
  assert.ok(ok.ok && ok.tExit > 600 && ok.tExit <= T_ORB, JSON.stringify(ok));
  assert.equal(breakoutOk(Float64Array.from([-42.76, 0.99, 0, 0, 0.1, 0])).ok, false, 'a radial-only push closes its ellipse and comes back');
  const h1 = Float64Array.from([-250, 0, 0, 0, 0, 0]);
  assert.ok(planBreakout(h1, { inside: false }), 'from H1 a breakout never enters the KOS');
});

test('Rule A: a hold at H1 is not passively safe (drag reaches X = 0 in ~8.0 h), so GO needs a Rule-P breakout', () => {
  const h1 = Float64Array.from([-250, 0, 0, 0, 0, 0]);
  let t = 0; while (propagate(h1, t, -DRAG_B)[0] < 0) t += 60;
  assert.ok(Math.abs(t / 3600 - 8.0) < 0.1, `${t / 3600} h`);
  assert.ok(passiveMin(h1).rMin < KOS_R);
});
