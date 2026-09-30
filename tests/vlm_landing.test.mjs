import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createLandingSim, drawConditions } from '../src/landing/sim.js';
import { AIRPORT } from '../src/landing/airport.js';
import { FT } from '../src/landing/vehicle.js';
import { landingNow, landingLabel, landingBranches, stepOf, applyLandingKick, lOutcome, replayLanding, papiWhites, landingFacts } from '../vlm/gen/labels/landing.js';
import { landingSafety } from '../vlm/gen/safety.js';
import { lookAtCamera } from '../vlm/gen/labels/camera.js';
import { OBS } from '../vlm/gen/schema.js';

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
function runKicked(seed, kind, ft) {
  const cond = drawConditions(seed, { final: true }), sim = createLandingSim(cond); let inj = null;
  while (!sim.rep.done && sim.flight.t < 900) { if (!inj && !sim.flight.wow && sim.flight.air.hRA / FT <= ft) { inj = { kind, params: {}, step: stepOf(sim), sim_t_s: sim.flight.t }; applyLandingKick(sim, inj); } sim.step(); }
  return { cond, sim, inj };
}
test('injected faults: +60 m at 600 ft goes around (CAUTION); +25 m lateral at 40 ft ends in an excursion (UNSAFE)', () => {
  let ga = 0, exc = 0;
  for (let seed = 1; seed <= 10; seed++) {
    const a = runKicked(seed, 'alt_plus_60', 600); if (lOutcome(a.sim.rep) === 'go_around') ga++;
    const b = runKicked(seed, 'lateral_25', 40); if (lOutcome(b.sim.rep) === 'excursion') exc++;
  }
  assert.ok(ga >= 8, `go-arounds ${ga}/10`); assert.ok(exc >= 8, `excursions ${exc}/10`);
  let seed = 1, k = runKicked(1, 'lateral_25', 40); while (lOutcome(k.sim.rep) !== 'excursion') k = runKicked(++seed, 'lateral_25', 40);
  const step = k.inj.step + 70, at = replayLanding(k.cond, { injection: k.inj, step });
  const br = landingBranches({ cond: k.cond, injection: k.inj, step, liveResult: k.sim.rep.result, airborne: !at.flight.wow, retard: at.gnc.st.retard });
  assert.equal(br.CONTINUE, 'excursion'); assert.equal(landingLabel(at, br).safety.verdict, 'UNSAFE');
});
test('replay to a step is exact, forced GA only while airborne, PAPI reads 2 white on the glide path', () => {
  const cond = drawConditions(12, { final: true }), a = createLandingSim(cond); for (let i = 0; i < 20000; i++) a.step();
  const b = replayLanding(cond, { step: stepOf(a) }); assert.deepEqual([...b.flight.p], [...a.flight.p]); assert.deepEqual([...b.flight.v], [...a.flight.v]);
  const br = landingBranches({ cond, step: 5000, liveResult: 'landed', airborne: true, retard: false }); assert.equal(br.CONTINUE, 'landed'); assert.ok(br.GO_AROUND);
  assert.equal(landingBranches({ cond, step: 5000, liveResult: 'landed', airborne: false, retard: false }).GO_AROUND, null);
});
test('PAPI reads two white on a 3 degree path', () => {
  const u = AIRPORT.papi[0], d = 1500; assert.equal(papiWhites([u.x - d, u.y + Math.tan(3 * Math.PI / 180) * d, u.z]), 2);
});

// landingFacts along one final (review focus 1): every fact is {v, unit, obs}, every number is finite (non-finite -> null)
const numbersOk = (v) => (typeof v === 'number' ? Number.isFinite(v) : Array.isArray(v) ? v.every(numbersOk) : v && typeof v === 'object' ? Object.values(v).every(numbersOk) : true);
test('landingFacts: {v, unit, obs} with finite numbers on every sampled frame; approach facts present; PAPI and windsock only in frame', () => {
  const sim = createLandingSim(drawConditions(3, { final: true })), seen = { papi: 0, sock: 0, airborne: 0 };
  while (!sim.rep.done && sim.flight.t < 900) {
    if (stepOf(sim) % 240 === 0) {
      const P = sim.flight.p, cam = lookAtCamera({ eye: [P[0] - 78, P[1] + 16, P[2]], target: [P[0] + 30, P[1], P[2]], fovDeg: 50, aspect: 896 / 504, near: 0.5, far: 150000 });
      const F = landingFacts(sim, cam, { scene: { time: 'day', vis: 10, clouds: 'few', rain: 0 } });
      for (const [id, f] of Object.entries(F)) { assert.ok(f && 'v' in f && 'unit' in f && OBS.includes(f.obs), id); assert.ok(numbersOk(f.v), `${id} = ${JSON.stringify(f.v)}`); }
      if (F['papi_whites_cam'].v !== null) { seen.papi++; assert.ok(F['papi_whites_cam'].v >= 0 && F['papi_whites_cam'].v <= 4); }
      if (F['windsock.from_deg'].v !== null) seen.sock++;
      if (!sim.flight.wow && sim.gnc.st.vert === 'GS') {
        seen.airborne++;
        for (const id of ['ias_kt', 'gs_kt', 'vs_fpm', 'vapp_kt', 'ra_ft', 'thr_nm', 'att.heading_deg', 'ils.loc_dots', 'ils.gs_dots', 'wind.head_kt', 'thrust']) assert.equal(typeof F[id].v, 'number', id);
        assert.match(F['wind.metar'].v, /^\d{5}(G\d+)?KT$/); assert.ok(['down', 'transit', 'up'].includes(F['cfg.gear'].v));
      }
    }
    sim.step();
  }
  assert.equal(sim.rep.result, 'landed'); assert.ok(seen.airborne > 10 && seen.papi > 0 && seen.sock > 0, JSON.stringify(seen));
});
