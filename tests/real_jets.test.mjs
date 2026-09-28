// Real spacecraft S1: jets, MIB, propellant and jet select (spec sections 5 and 10).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { JETS, JET_INDEX, INERTIA, MASS0, PROP0, KIND, G0, DEG, TRANS_SETS, CYCLE } from '../src/real/consts.js';
import { createJets, transOnTimes, attSelect, torqueOf, authority } from '../src/real/jets.js';

const on = (map) => { const a = new Float64Array(JETS.length); for (const [k, v] of Object.entries(map)) a[JET_INDEX[k]] = v; return a; };
const degps2 = (tau, ax) => (tau / INERTIA[ax]) * 180 / Math.PI;

test('a 20 ms request fires nothing this cycle and carries over; a 40 ms pulse is 154.8 N s', () => {
  const j = createJets(), req = on({ P13: 0.02 });
  let out = j.schedule(req); assert.equal(out[JET_INDEX.P13], 0); assert.ok(Math.abs(j.carry[JET_INDEX.P13] - 0.02) < 1e-12);
  out = j.schedule(req); assert.ok(Math.abs(out[JET_INDEX.P13] - 0.04) < 1e-12, 'carried 20 + 20 ms fire as one 40 ms pulse');
  const imp = j.impulse(on({ P13: 0.04 }));
  assert.ok(Math.abs(Math.hypot(...imp.I) - 154.8) < 1e-9, `impulse ${Math.hypot(...imp.I)}`);
  assert.equal(j.schedule(on({}))[JET_INDEX.P13], 0, 'a dropped request drops its carry');
});

test('on-times are quantised to 10 ms and capped at 100 ms; failed jets never fire', () => {
  const j = createJets({ failed: [JET_INDEX.P2] });
  const out = j.schedule(on({ P1: 0.053, P2: 0.1, P3: 0.37 }));
  assert.ok(Math.abs(out[JET_INDEX.P1] - 0.05) < 1e-12 && out[JET_INDEX.P2] === 0 && Math.abs(out[JET_INDEX.P3] - CYCLE) < 1e-12);
  for (let k = 0; k < 50; k++) assert.equal(j.schedule(on({ P2: 0.1 }))[JET_INDEX.P2], 0);
  const r = transOnTimes(j, [0, 0.05, 0], MASS0, new Float64Array(22));
  assert.equal(r[JET_INDEX.P2], 0); assert.ok(r[JET_INDEX.P4] > 0, 'the set flies on its working jet');
});

test('600 kg of propellant gives 17.40 m/s +- 0.1 % (aft pair, Isp 280 s)', () => {
  const j = createJets(), set = on({ P13: CYCLE, P14: CYCLE }); let m = MASS0, v = 0;
  const F = new Float64Array(3), T = new Float64Array(3);
  while (MASS0 - m < PROP0) { const md = j.active(set, 0, F, T), h = Math.min(0.01, (PROP0 - (MASS0 - m)) / md); v += (F[0] / m) * h; m -= md * h; }
  assert.ok(Math.abs(v - 17.40) <= 17.40 * 1e-3, `dv ${v}`);
  assert.ok(Math.abs(j.mdot[JET_INDEX.P1] - 1.409) < 1e-3 && Math.abs(j.mdot[JET_INDEX.V1] - 0.0412) < 1e-4);
  assert.ok(Math.abs(KIND.P.isp * G0 - 2746) < 1);
});

test('authority: two-jet translation 0.0815 m/s^2; primary pitch/yaw/roll 0.52/0.55/3.72 deg/s^2; vernier 0.0063/0.0061/0.051', () => {
  const j = createJets();
  assert.ok(Math.abs((2 * 3870) / MASS0 - 0.0815) < 1e-4);
  const P = torqueOf(j, on({ P2: CYCLE, P3: CYCLE })), Y = torqueOf(j, on({ P5: CYCLE, P8: CYCLE })), R = torqueOf(j, on({ P9: CYCLE, P12: CYCLE }));
  assert.ok(Math.abs(degps2(P[2], 2) - 0.52) < 0.01 && Math.abs(degps2(Y[1], 1) - 0.55) < 0.01 && Math.abs(degps2(R[0], 0) - 3.72) < 0.02);
  const vp = torqueOf(j, on({ V2: CYCLE }))[2], vy = torqueOf(j, on({ V3: CYCLE }))[1], vr = torqueOf(j, on({ V6: CYCLE }))[0];
  assert.ok(Math.abs(degps2(vp, 2) - 0.0063) < 3e-4 && Math.abs(degps2(vy, 1) - 0.0061) < 3e-4 && Math.abs(degps2(vr, 0) - 0.051) < 2e-3, `vernier ${[vp, vy, vr]}`);
  // V5/V6 also pitch down 0.0073 deg/s^2 each: more than V1's own pitch authority (why the select is a least squares)
  assert.ok(Math.abs(degps2(torqueOf(j, on({ V5: CYCLE }))[2], 2) + 0.0073) < 3e-4);
  const a = authority(j, 'P'); assert.ok(a.every((t) => t > 0));
});

test('translation cross-rate <= 0.0035 deg/s per 40 ms pulse at nominal MIB, for every set', () => {
  const j = createJets();
  for (const set of TRANS_SETS) {
    const t = new Float64Array(22); for (const i of set) t[i] = 0.04;
    const L = j.impulse(t).L;
    for (let ax = 0; ax < 3; ax++) assert.ok(Math.abs(degps2(L[ax], ax)) <= 0.0035, `${set.map((i) => JETS[i].name)} axis ${ax}: ${degps2(L[ax], ax)}`);
  }
});

test('attitude select: least squares over the mode jets; a pure roll vernier request keeps pitch inside the deadband', () => {
  const j = createJets();
  const want = [0.5 * authority(j, 'P')[0], 0, 0], u = attSelect(j, 'P', want, new Float64Array(22)), got = torqueOf(j, u);
  assert.ok(Math.abs(got[0] - want[0]) < 0.02 * want[0] && Math.abs(got[1]) < 0.02 * want[0] && Math.abs(got[2]) < 0.02 * want[0], `P roll ${got}`);
  for (const i of u.keys()) if (u[i] > 0) assert.equal(JETS[i].kind, 'P');
  const roll = [1500, 0, 0], v = attSelect(j, 'V', roll, new Float64Array(22)), g = torqueOf(j, v);
  for (const i of v.keys()) if (v[i] > 0) assert.equal(JETS[i].kind, 'V');
  // one full cycle of that select changes the pitch rate by less than the vernier rate deadband (0.01 deg/s)
  assert.ok(Math.abs(degps2(g[2], 2) * CYCLE) < 0.01 && g[0] > 1000, `vernier roll ${g}`);
});
