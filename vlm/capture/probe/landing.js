// vlm/capture/probe/landing.js — L in the page (spec §3.2 L, §3.3, §7.3 L, R8): the chase view asserted at setup and at every
// capture, hFlare applied before the first frame, f2 snapshots (facts, now, p, v), the live run finished headlessly after the
// last sample, and replay branches from createLandingSim(__landing.conditions(seed)), the same module instance the page uses.
// The one runtime kick is armed on the sim: sim.step is an own property of the sim object that main.js's rAF loop and
// __landing.advance both call, so the kick lands right before the first step at which its trigger holds (hRA at or below the
// drawn height, airborne), the very step at which replayLanding re-applies it. An oracle run of the same conditions gives that
// step (and the clean run's end, for the slot selection) at setup, so the drive can pre-roll >= 1 s of rendered frames up to
// the kick and take the injection clip right after it.
import { landingFacts, landingNow, landingBranches, stepOf, applyLandingKick } from '../../gen/labels/landing.js';
import { createLandingSim } from '../../../src/landing/sim.js';
import { landingState } from '../../gen/labels/rawstate.js';
const FT = 0.3048, r2 = (x) => +x.toFixed(2);
const camOf = (c) => ({ mode: 'chase', fov_deg: c.fov, aspect: c.aspect, near: c.near, far: c.far, matrixWorldInverse: Array.from(c.matrixWorldInverse.elements), projectionMatrix: Array.from(c.projectionMatrix.elements) });
export const landingTrig = (sim, trig) => !sim.flight.wow && sim.flight.air.hRA / FT <= trig.value;
// In cloud (ruling T10-b), from the page's own visibility state: before any frame the scene's fog density is the visibility's
// fogD (scene.js:42); each frame it is fogD + (0.012 - fogD) * k, k = how far the camera is inside a BKN/OVC deck (0 outside,
// 1 about 12 m in; scene.js:93). The camera is in cloud when k >= 0.5, i.e. between the deck's base and top.
export const IN_CLOUD_FOG = 0.012;
export const inCloudOf = (density, fogD) => fogD !== null && fogD < IN_CLOUD_FOG && (density - fogD) / (IN_CLOUD_FOG - fogD) >= 0.5;
// the live fog density the page is rendering with this frame (scene.js:93 closes it in inside a cloud deck); 0 when the
// scene carries no fog object at all
const liveFogDensity = (st) => { const f = st.L.scene.scene && st.L.scene.scene.fog; return f ? f.density : 0; };
const inCloud = (st) => inCloudOf(liveFogDensity(st), st.fogD);
// the injection record (provenance.injection): step = steps run when the kick lands, as replayLanding counts them
export function armLanding(sim, inj, onFire) {
  const step0 = sim.step; let fired = false;
  sim.step = function () {
    if (!fired && landingTrig(sim, inj.trigger)) {
      fired = true; const at = sim.flight.air.hRA / FT, rec = { kind: inj.kind, params: { ...(inj.params || {}), trigger: { qty: inj.trigger.qty, value: r2(inj.trigger.value), at: r2(at) } }, step: stepOf(sim), sim_t_s: +sim.flight.t.toFixed(4) };
      applyLandingKick(sim, inj); onFire(rec);
    }
    return step0();
  };
}
// one clean run of the page's conditions: the step at which an armed kick would land (null: never) and the run's end time
export function landingOracle(cond, params, inj = null) {
  const s = createLandingSim(cond, { params }); let injAt = null;
  while (!s.rep.done && s.flight.t < 900) { if (inj && injAt === null && landingTrig(s, inj.trigger)) injAt = stepOf(s); s.step(); }
  return { injAt, endT: s.flight.t };
}
export const landingTriggerStep = (cond, params, inj) => landingOracle(cond, params, inj).injAt;
export async function setup(family, p) {
  const L = window.__landing, sim = L.sim;
  if (L.scene.view !== 'chase') throw new Error(`landing view is ${L.scene.view}, not chase`);
  if (stepOf(sim) !== 0) throw new Error(`the landing sim ran ${stepOf(sim)} steps before setup`);
  const cond = L.conditions(sim.rep.cond.seed);
  if (JSON.stringify(cond) !== JSON.stringify(sim.rep.cond)) throw new Error('__landing.conditions(seed) differs from the page run');
  if (p.hflare) sim.gnc.params.hFlare = 0.3;
  const armed = p.inject && p.inject.at === 'runtime' ? p.inject : null, o = landingOracle(cond, p.hflare ? { hFlare: 0.3 } : {}, armed);
  const fog = L.scene.scene && L.scene.scene.fog, st = { L, sim, p, cond, params: p.hflare ? { hFlare: 0.3 } : {}, injection: null, injAt: o.injAt, endT: o.endT, snaps: [], fin: null, fogD: fog ? fog.density : null };
  if (armed) armLanding(sim, armed, (r) => { st.injection = r; });
  return st;
}
export const check = (st) => ({ ok: true, t: st.sim.flight.t, step: stepOf(st.sim), hRAft: st.sim.flight.air.hRA / FT, wow: st.sim.flight.wow, done: st.sim.rep.done, view: st.L.scene.view, injAt: st.injAt, endT: st.endT, injected: st.injection });
export function capture(st) {
  if (st.L.scene.view !== 'chase') throw new Error(`landing view is ${st.L.scene.view} at capture`);
  const cv = document.getElementById('view'), gl = cv.getContext('webgl2') || cv.getContext('webgl');
  if (!gl.getContextAttributes().preserveDrawingBuffer) throw new Error('preserveDrawingBuffer is off');
  // V1-2: the frame's raw state (aircraft pose, live fog density); the runway/PAPI geometry is static (rawstate.js sceneOf)
  return { png: cv.toDataURL('image/png'), cam: camOf(st.L.scene.camera), clock: { date_ms: Date.now(), perf_ms: performance.now() }, step: stepOf(st.sim), inCloud: inCloud(st), state: landingState(st.sim, liveFogDensity(st)) };
}
export function snap(st) {
  // scene.clouds is the page's METAR cloud group (conditions(): "-RA " + FEW|SCT|BKN|OVC with the base, or NSC), the form the
  // text layer's TEXT_FACTS 'clouds' shape reads; scene.in_cloud (visual, T10-b) says the f2 camera is inside the deck.
  // landingFacts gets the live fog density too (T10-v), so papi_whites_cam, windsock.from_deg and the aircraft's own
  // visual facts (cfg.gear, cfg.spoilers) are nulled once the fog or the in-cloud closing-in hides them.
  const sim = st.sim, f = sim.flight, cam = camOf(st.L.scene.camera);
  const sc = { time: st.p.time, vis: st.p.vis, clouds: st.cond.clouds ?? null, rain: st.p.rain, in_cloud: inCloud(st) };
  const facts = landingFacts(sim, cam, { scene: sc, fogDensity: liveFogDensity(st) });
  const s = { step: stepOf(sim), now: landingNow(sim), facts, airborne: !f.wow, retard: !!sim.gnc.st.retard, p: Array.from(f.p), v: Array.from(f.v) };
  st.snaps.push(s); return s.step;
}
// sim.run() steps the sim's own closure: an armed kick that has not fired by now stays unfired, as in the samples' labels
export function finish(st) { const rep = st.sim.run(); st.fin = { result: rep.result, t: st.sim.flight.t }; return st.fin; }
// a branch replay that ends in timeout is a sample-level discard (schema Discard, e.discard), returned instead of thrown
export function branches(st, { step }) {
  const s = st.snaps.find((x) => x.step === step);
  if (st.injection && st.injection.step === step) return { discard: `the ${st.injection.kind} kick landed at the sample step ${step}` };
  try { return { now: s.now, facts: s.facts, outcome: landingBranches({ cond: st.cond, params: st.params, injection: st.injection, step, liveResult: st.fin.result, airborne: s.airborne, retard: s.retard, liveP: s.p, liveV: s.v }) }; }
  catch (e) { if (e && e.discard) return { discard: e.message }; throw e; }
}
