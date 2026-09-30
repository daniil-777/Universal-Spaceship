import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createLandingSim, drawConditions } from '../src/landing/sim.js';
import { AIRPORT } from '../src/landing/airport.js';
import { FT } from '../src/landing/vehicle.js';
import { landingNow, landingLabel, landingBranches, stepOf, applyLandingKick, lOutcome, replayLanding, papiWhites, landingFacts, PAPI_BOX, SOCK_BOX } from '../vlm/gen/labels/landing.js';
import { landingSafety } from '../vlm/gen/safety.js';
import { lookAtCamera, project } from '../vlm/gen/labels/camera.js';
import { OBS, OUTCOMES } from '../vlm/gen/schema.js';

test('nominal vectored finals, seeds 1-20: all land; 0 UNSAFE frames, 0 GO_AROUND advice, 0 FLARE/ROLLOUT/STOP dot or speed CAUTION', () => {
  let frames = 0;
  for (let seed = 1; seed <= 20; seed++) {
    const cond = drawConditions(seed, { final: true }), sim = createLandingSim(cond), nows = [];
    while (!sim.rep.done && sim.flight.t < 900) { if (stepOf(sim) % 30 === 0) nows.push(landingNow(sim)); sim.step(); }
    assert.equal(sim.rep.result, 'landed', `seed ${seed}`);
    for (const now of nows) {
      const s = landingSafety({ now, outcome: { CONTINUE: 'landed', GO_AROUND: now.airborne ? 'go_around' : null } }); frames++;
      assert.notEqual(s.verdict, 'UNSAFE', `seed ${seed} ${now.vert} ${now.hRAft.toFixed(0)} ft ${s.reasons}`); assert.notEqual(s.best_action, 'GO_AROUND');
      if (['FLARE', 'ROLLOUT', 'STOP'].includes(now.vert)) assert.ok(!s.reasons.some((r) => ['LOCALIZER_DEVIATION', 'GLIDESLOPE_DEVIATION', 'SPEED_OUT_OF_BAND'].includes(r)), `${now.vert} ${s.reasons}`);
    }
  }
  assert.ok(frames > 1000);
});
// a live run kicked at `ft`; the flight's p and v are kept at the sample step inj.step + lead (the timing rule's lead is 61)
function runKicked(seed, kind, ft, lead = 70) {
  const cond = drawConditions(seed, { final: true }), sim = createLandingSim(cond); let inj = null, live = null;
  while (!sim.rep.done && sim.flight.t < 900) {
    if (!inj && !sim.flight.wow && sim.flight.air.hRA / FT <= ft) { inj = { kind, params: {}, step: stepOf(sim), sim_t_s: sim.flight.t }; applyLandingKick(sim, inj); }
    if (inj && !live && stepOf(sim) === inj.step + lead) live = { step: stepOf(sim), liveP: Float64Array.from(sim.flight.p), liveV: Float64Array.from(sim.flight.v) };
    sim.step();
  }
  return { cond, sim, inj, live };
}
const labelAt = (k) => {
  const at = replayLanding(k.cond, { injection: k.inj, step: k.live.step, liveP: k.live.liveP, liveV: k.live.liveV });
  const br = landingBranches({ cond: k.cond, injection: k.inj, step: k.live.step, liveResult: k.sim.rep.result, airborne: !at.flight.wow, retard: at.gnc.st.retard, liveP: k.live.liveP, liveV: k.live.liveV });
  return { br, label: landingLabel(at, br) };
};
test('injected faults: +60 m at 600 ft goes around (CAUTION); +25 m lateral at 40 ft ends in an excursion (UNSAFE)', () => {
  let ga = 0, exc = 0;
  for (let seed = 1; seed <= 10; seed++) {
    const a = runKicked(seed, 'alt_plus_60', 600); if (lOutcome(a.sim.rep) === 'go_around') ga++;
    const b = runKicked(seed, 'lateral_25', 40); if (lOutcome(b.sim.rep) === 'excursion') exc++;
  }
  assert.ok(ga >= 8, `go-arounds ${ga}/10`); assert.ok(exc >= 8, `excursions ${exc}/10`);
  let seed = 1, k = runKicked(1, 'lateral_25', 40); while (lOutcome(k.sim.rep) !== 'excursion') k = runKicked(++seed, 'lateral_25', 40);
  const x = labelAt(k); assert.equal(x.br.CONTINUE, 'excursion'); assert.equal(x.label.safety.verdict, 'UNSAFE');
  seed = 1; let a = runKicked(1, 'alt_plus_60', 600); while (lOutcome(a.sim.rep) !== 'go_around') a = runKicked(++seed, 'alt_plus_60', 600);
  const g = labelAt(a).label.safety;
  assert.equal(g.verdict, 'CAUTION', `${g.reasons}`); assert.equal(g.action_outcome.CONTINUE, 'go_around'); assert.ok(g.reasons.includes('UNSTABLE_APPROACH'));
  assert.ok(OUTCOMES.L.includes(g.action_outcome.GO_AROUND)); assert.equal(g.action_risk.GO_AROUND, 0); assert.equal(g.cause, null);
});
test('replay to a step is exact, forced GA only while airborne, PAPI reads 2 white on the glide path', () => {
  const cond = drawConditions(12, { final: true }), a = createLandingSim(cond); for (let i = 0; i < 20000; i++) a.step();
  const b = replayLanding(cond, { step: stepOf(a) }); assert.deepEqual([...b.flight.p], [...a.flight.p]); assert.deepEqual([...b.flight.v], [...a.flight.v]);
  const br = landingBranches({ cond, step: 5000, liveResult: 'landed', airborne: true, retard: false }); assert.equal(br.CONTINUE, 'landed'); assert.ok(OUTCOMES.L.includes(br.GO_AROUND), br.GO_AROUND);
  assert.equal(landingBranches({ cond, step: 5000, liveResult: 'landed', airborne: false, retard: false }).GO_AROUND, null);
});
test('PAPI reads two white on a 3 degree path', () => {
  const u = AIRPORT.papi[0], d = 1500; assert.equal(papiWhites([u.x - d, u.y + Math.tan(3 * Math.PI / 180) * d, u.z]), 2);
});
test('L replays match the live flight bit for bit after a kick; a mismatch throws; an injection at the sample step throws', () => {
  const k = runKicked(4, 'lateral_25', 100), { step, liveP, liveV } = k.live;
  const at = replayLanding(k.cond, { injection: k.inj, step, liveP, liveV });
  assert.ok(at.flight.p.every((v, i) => Object.is(v, liveP[i])) && at.flight.v.every((v, i) => Object.is(v, liveV[i])));
  assert.throws(() => replayLanding(k.cond, { step, liveP, liveV }), /bit-exact/, 'the replay without the kick is not the live flight');
  const bad = Float64Array.from(liveP); bad[2] += 1e-9;
  assert.throws(() => landingBranches({ cond: k.cond, injection: k.inj, step, liveResult: k.sim.rep.result, airborne: true, retard: false, liveP: bad, liveV }), /bit-exact/);
  assert.throws(() => landingBranches({ cond: k.cond, injection: k.inj, step: k.inj.step, liveResult: k.sim.rep.result, airborne: true, retard: false }), /sample step/);
});
test('a branch that ends in timeout is a discard, never a label (live CONTINUE and the GO_AROUND replay)', () => {
  const cond = drawConditions(12, { final: true });
  assert.throws(() => landingBranches({ cond, step: 5000, liveResult: 'timeout', airborne: true, retard: false }), (e) => e.discard === true && /CONTINUE/.test(e.message));
  const far = { ...cond, start: { ...cond.start, x: -90000 } }, s = replayLanding(far, { step: 885 * 120 });
  assert.ok(!s.rep.done && !s.flight.wow, 'still flying at 885 s (a forced GA needs 20 s, the run stops at 900 s)');
  assert.throws(() => landingBranches({ cond: far, step: 885 * 120, liveResult: 'landed', airborne: true, retard: false }), (e) => e.discard === true && /GO_AROUND/.test(e.message));
});

// landingFacts along one final (review focus 1): every fact is {v, unit, obs}, every number is finite (non-finite -> null)
const numbersOk = (v) => (typeof v === 'number' ? Number.isFinite(v) : Array.isArray(v) ? v.every(numbersOk) : v && typeof v === 'object' ? Object.values(v).every(numbersOk) : true);
const cam = (eye, target) => lookAtCamera({ eye, target, fovDeg: 50, aspect: 896 / 504, near: 0.5, far: 150000 });
test('landingFacts: {v, unit, obs} with finite numbers on every sampled frame; approach facts present', () => {
  const sim = createLandingSim(drawConditions(3, { final: true })); let airborne = 0, papi = 0;
  while (!sim.rep.done && sim.flight.t < 900) {
    if (stepOf(sim) % 60 === 0) {
      const P = sim.flight.p, F = landingFacts(sim, cam([P[0] - 78, P[1] + 16, P[2]], [P[0] + 30, P[1], P[2]]), { scene: { time: 'day', vis: 10, clouds: 'few', rain: 0 } });
      for (const [id, f] of Object.entries(F)) { assert.ok(f && 'v' in f && 'unit' in f && OBS.includes(f.obs), id); assert.ok(numbersOk(f.v), `${id} = ${JSON.stringify(f.v)}`); }
      if (F['papi_whites_cam'].v !== null) { papi++; assert.ok(F['papi_whites_cam'].v >= 0 && F['papi_whites_cam'].v <= 4); }
      if (!sim.flight.wow && sim.gnc.st.vert === 'GS') {
        airborne++;
        for (const id of ['ias_kt', 'gs_kt', 'vs_fpm', 'vapp_kt', 'ra_ft', 'thr_nm', 'att.heading_deg', 'ils.loc_dots', 'ils.gs_dots', 'wind.head_kt', 'thrust']) assert.equal(typeof F[id].v, 'number', id);
        assert.match(F['wind.metar'].v, /^\d{5}(G\d+)?KT$/); assert.ok(['down', 'transit', 'up'].includes(F['cfg.gear'].v));
      }
    }
    sim.step();
  }
  assert.equal(sim.rep.result, 'landed'); assert.ok(airborne > 10 && papi > 0, `airborne ${airborne} papi ${papi}`);
});
test('PAPI and windsock facts need their pixel box in frame and >= 1 px at 160x96 (the box, not the centre point)', () => {
  const sim = createLandingSim(drawConditions(3, { final: true })), W = 896, H = 504, DEG = Math.PI / 180;
  const pc = PAPI_BOX[0].map((v, i) => (v + PAPI_BOX[1][i]) / 2), sc = SOCK_BOX[0].map((v, i) => (v + SOCK_BOX[1][i]) / 2);
  const F = (c) => landingFacts(sim, c);
  assert.equal(typeof F(cam([pc[0] - 150, 8, pc[2]], pc))['papi_whites_cam'].v, 'number', '150 m out, looking at the bar');
  assert.equal(F(cam([pc[0] - 3000, 160, pc[2]], pc))['papi_whites_cam'].v, null, '3 km out the bar is < 1 px tall at 160x96');
  assert.equal(F(cam([pc[0] - 150, 8, pc[2]], [pc[0] - 300, 8, pc[2]]))['papi_whites_cam'].v, null, 'looking away');
  const e = [pc[0] - 120, 2, pc[2]], side = cam(e, [e[0] + 100 * Math.cos(42 * DEG), 2, e[2] - 100 * Math.sin(42 * DEG)]);
  assert.ok(project(side, pc, W, H).x > W, 'the PAPI centre is right of the frame');
  assert.equal(typeof F(side)['papi_whites_cam'].v, 'number', 'the near end of the bar is in frame and >= 1 px');
  assert.equal(F(cam([sc[0] - 120, 12, sc[2]], sc))['windsock.from_deg'].v, sim.rep.cond.wind.dir, '120 m from the sock');
  assert.equal(F(cam([sc[0] - 2000, 60, sc[2]], sc))['windsock.from_deg'].v, null, '2 km out the sock is < 1 px at 160x96');
});
