// Real spacecraft S1: BREAKOUT from the corridor and FINAL (spec sections 3, 6 and 10).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { DRAG_B, KOS_R, T_ORB, DAY, SHIP_PORT, STATION_PORT, portRel } from '../src/real/consts.js';
import { propagate } from '../src/real/cw.js';
import { breakoutOk } from '../src/real/passive.js';
import { createRealSim, drawRun } from '../src/real/sim.js';
import { PH } from '../src/real/guidance.js';

// a run placed at port range rho closing at vc on the docking axis, already in phase ph
function placed(seed, rho, vc, ph) {
  const R = drawRun(seed, { start: 'final' });
  return createRealSim({ seed, run: { ...R, failed: [], phase: ph, x: [STATION_PORT[0] - rho - SHIP_PORT[0], -SHIP_PORT[1], 0, vc, 0, 0] } });
}

for (const [rho, vc, ph] of [[10, 0.10, PH.FINAL], [20, 0.10, PH.FINAL], [30, 0.10, PH.CORRIDOR], [100, 0.2, PH.CORRIDOR]]) {
  test(`breakout from rho = ${rho} m closing at ${vc} m/s: <= 2 m forward, out of the KOS within one orbit, no 24 h re-entry`, () => {
    const sim = placed(7, rho, vc, ph);
    sim.step();
    const x0 = sim.x[0]; sim.abort('test');
    let fwd = 0, done = null;
    while (!sim.rep.done) { sim.step(); fwd = Math.max(fwd, sim.x[0] - x0); if (!done && sim.guid.st.phase === PH.DEPART) done = Float64Array.from(sim.x); }
    assert.equal(sim.rep.result, 'breakout', `${sim.rep.result} ${sim.rep.reason}`);
    assert.ok(fwd <= 2, `forward ${fwd} m`);
    const ev = sim.events.find((e) => e.kind === 'BREAKOUT');
    for (let i = 0; i < 3; i++) assert.ok(Math.abs(ev.executed[i] - ev.screened[i]) <= 0.002, `axis ${i}: executed ${ev.executed[i]} vs screened ${ev.screened[i]}`);
    assert.ok(sim.log.some((l) => l.kind === 'bypass' && l.burn === 'BREAKOUT'), 'the pre-validated burn bypassed the filter, logged');
    for (const ad of [-DRAG_B, 0, DRAG_B]) {
      let tOut = null;
      for (let t = 0; t <= DAY; t += 10) {
        const p = propagate(done, t, ad), r = Math.hypot(p[0], p[1], p[2]);
        if (r > KOS_R) tOut ??= t; else assert.equal(tOut, null, `re-entered at ${t} s for a_d ${ad}`);
      }
      assert.ok(tOut !== null && tOut <= T_ORB, `a_d ${ad}: out at ${tOut}`);
    }
    assert.ok(breakoutOk(done).ok);
  });
}
