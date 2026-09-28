// Real spacecraft S1: TRANSFER, Rule P / Rule A, GO and the near starts (spec sections 3, 6 and 10).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { N, T_ORB, DAY, DRAG_B, KOS_R, JET_INDEX, BREAKOUT_V, PROP0 } from '../src/real/consts.js';
import { propagate } from '../src/real/cw.js';
import { passiveMin, planTransfer, ruleP, breakoutOk, planBreakout } from '../src/real/passive.js';
import { createRealSim, drawRun } from '../src/real/sim.js';
import { PH } from '../src/real/guidance.js';

const START = Float64Array.from([-2000, 0, 250, 0, 0, 0]);
const nominalRun = (x = START) => ({ ...drawRun(1, { start: 'far', disp: false }), x: Array.from(x), att: [0, 0, 0], rate: [0, 0, 0] });

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

test('6-DOF TRANSFER: arrival within 5 m of (-250, 0, -250); MCCs at T/3 and 2T/3; Rule P holds after every burn', () => {
  const sim = createRealSim({ seed: 1, run: nominalRun(), nav: 'truth', disp: false });
  while (!sim.rep.done && sim.guid.st.phase === PH.TRANSFER) sim.step();
  const x = sim.x, T = sim.guid.st.plan.T, t0 = sim.guid.st.tDep;
  assert.ok(Math.hypot(x[0] + 250, x[1], x[2] + 250) < 5, `arrival ${Array.from(x.subarray(0, 3))}`);
  const mcc = sim.events.filter((e) => e.kind === 'MCC' || e.kind === 'MCC skipped');
  assert.equal(mcc.length, 2);
  for (let k = 0; k < 2; k++) { const at = mcc[k].t0 ?? mcc[k].t, due = t0 + ((k + 1) * T) / 3; assert.ok(at >= due && at <= due + 1.01, `MCC ${k} at ${at}, due ${due}`); }
  for (const e of sim.events.filter((e) => e.kind === 'DEPART' || e.kind === 'MCC')) assert.ok(e.truthRMin >= 240, `${e.kind} ${e.truthRMin}`);
});

test('Rule A: a hold at H1 is not passively safe (drag reaches X = 0 in ~8.0 h), so GO needs a Rule-P breakout', () => {
  const h1 = Float64Array.from([-250, 0, 0, 0, 0, 0]);
  let t = 0; while (propagate(h1, t, -DRAG_B)[0] < 0) t += 60;
  assert.ok(Math.abs(t / 3600 - 8.0) < 0.1, `${t / 3600} h`);
  assert.ok(passiveMin(h1).rMin < KOS_R);
});

test('GO needs >= 40 % propellant: a near start with 30 % left goes NO-GO at H1 and breaks out safely', () => {
  const sim = createRealSim({ seed: 5, run: { ...drawRun(5, { start: 'near' }), propUsed: 420 } });
  sim.run();
  assert.equal(sim.rep.result, 'breakout', `${sim.rep.result} ${sim.rep.reason}`);
  assert.ok(sim.events.some((e) => e.kind === 'abort' && /NO-GO/.test(e.why)));
});

test('8 dispersed near starts all capture', () => {
  for (let s = 1; s <= 8; s++) {
    const sim = createRealSim({ seed: 100 + s, run: { ...drawRun(100 + s, { start: 'near' }), failed: [] } });
    sim.run();
    assert.equal(sim.rep.result, 'capture', `seed ${100 + s}: ${sim.rep.result} ${sim.rep.reason}`);
  }
});

test('review focus: a failed translation-critical primary (P1-P8) ends safe, never in a failure', () => {
  for (const name of ['P1', 'P7']) {
    const sim = createRealSim({ seed: 21, run: { ...drawRun(21, { start: 'near' }), failed: [JET_INDEX[name]] } });
    sim.run();
    assert.ok(sim.rep.result === 'capture' || sim.rep.result === 'breakout', `${name}: ${sim.rep.result} ${sim.rep.reason}`);
  }
});

test('review focus: a failed lateral primary ends safe — a failed DEPART jet breaks out at once, others break out at H1', () => {
  // forced P1 far starts: the six whose degraded -Y departures broke Rule P or ran out of propellant, and seed 181, whose
  // first-passing far breakout kept only a few metres of margin: break out at t = 0 on the widest-margin v+;
  // seed 64 (drawn with a failed P7), a forced P2 (seed 20, whose far-start breakout was not passively safe) and a
  // forced P3 (seed 127: the aft P9/P11 pair replaces it, so its DEPART is healthy): fly to H1
  const runs = [4, 7, 13, 20, 24, 25].map((s) => ({ seed: s, atOnce: true, run: { ...drawRun(s, { start: 'far' }), failed: [JET_INDEX.P1] } }));
  runs.push({ seed: 64, atOnce: false, run: drawRun(64, { start: 'far' }) }, { seed: 20, atOnce: false, run: { ...drawRun(20, { start: 'far' }), failed: [JET_INDEX.P2] } });
  runs.push({ seed: 181, atOnce: true, run: { ...drawRun(181, { start: 'far' }), failed: [JET_INDEX.P1] } }, { seed: 127, atOnce: false, run: { ...drawRun(127, { start: 'far' }), failed: [JET_INDEX.P3] } });
  assert.deepEqual(runs[6].run.failed, [JET_INDEX.P7], 'seed 64 draws a failed P7');
  for (const o of runs) {
    const sim = createRealSim(o), tag = `seed ${o.seed} (failed jet ${o.run.failed})`;
    sim.run();
    assert.equal(sim.rep.result, 'breakout', `${tag}: ${sim.rep.result} ${sim.rep.reason}`);
    assert.ok(sim.rep.prop < PROP0 && sim.events.every((e) => e.truthRMin === undefined || e.truthRMin >= 240), `${tag}: Rule P or propellant (${sim.rep.prop} kg)`);
    const abort = sim.events.find((e) => e.kind === 'abort'), bo = sim.events.find((e) => e.kind === 'BREAKOUT');
    assert.ok(abort && abort.verified && /NO-GO: a lateral translation jet has failed/.test(abort.why) && bo && bo.truth.ok, `${tag}: ${JSON.stringify(abort)}`);
    const flown = sim.events.filter((e) => ['plan', 'DEPART', 'MCC', 'MCC skipped', 'ARRIVE', 'ZNULL'].includes(e.kind)).map((e) => e.kind);
    const phaseAtAbort = sim.events.slice(0, sim.events.indexOf(abort)).filter((e) => e.kind === 'phase').map((e) => e.phase).at(-1);
    if (o.atOnce) assert.ok(abort.t === 0 && flown.length === 0, `${tag}: abort at ${abort.t} s after [${flown}]`);
    else assert.ok(flown.includes('DEPART') && flown.includes('ARRIVE') && phaseAtAbort === PH.H1, `${tag}: abort in ${phaseAtAbort} after [${flown}]`);
  }
});

test('review focus: the same seed flies the same mission (restart, MC reproducibility), step by step or in one run', () => {
  const a = createRealSim({ seed: 9, start: 'final' }), b = createRealSim({ seed: 9, start: 'final' });
  a.run(); while (!b.rep.done) b.step();
  assert.equal(a.rep.result, b.rep.result); assert.equal(a.rep.t, b.rep.t); assert.deepEqual(a.rep.contact, b.rep.contact);
});

test('review focus: nav=truth flies to capture as well', () => {
  const sim = createRealSim({ seed: 4, start: 'near', nav: 'truth' });
  sim.run();
  assert.equal(sim.rep.result, 'capture', `${sim.rep.result} ${sim.rep.reason}`);
});
