// vlm/capture/probe/docking.js — D in the page (spec §3.2 D, §3.3, R8, §7.3 D). The page is never begun: its rAF only renders,
// and the probe steps the sim one CYCLE per rendered frame (sim.step(), exactly __real.advance(0.1)) and skips headlessly in
// whole cycles (exactly sim.run(sec)), checking the armed runtime kick's trigger (qty at or below the drawn value, in the drawn
// guidance phase if any) before every step, so the kick lands at the cycle replayDocking re-applies it at; an oracle run with
// the page's constructor arguments gives that cycle (and the clean run's end, for the slot selection) at setup for the drive. The centreline camera (R8) sits at
// SHIP_PORT + 0.5 m along body +X, looks along +X, up = body +Y; it overrides camera.lookAt on the scene's camera instance,
// which placeCamera calls every frame (delete restores the prototype method). No three import: Node tests load this module.
// Failed-P6 clean twins fly a probe-owned sim; scene.frame and setWarp are overridden on the scene instance so the page's own
// rAF renders that sim (dt, tReal, plume filter and sky warp as for a page sim) and a capture stays a read of the page frame.
import { dockingFacts, dockingNow, dockingBranches, cycleOf, applyDockingKick, createRealSim, drawRun, SHIP_PORT, portRel } from '../../gen/labels/docking.js';
import { qRotate } from '../../../src/mathx.js';
const camOf = (c, mode) => ({ mode, fov_deg: c.fov, aspect: c.aspect, near: c.near, far: c.far, matrixWorldInverse: Array.from(c.matrixWorldInverse.elements), projectionMatrix: Array.from(c.projectionMatrix.elements) });
const sim = (st) => st.own || st.R.sim, r3 = (x) => +x.toFixed(3);
const qtyOf = (s, qty) => { const p = portRel(s.x, s.q); return qty === 'rho_m' ? Math.hypot(p[0], p[1], p[2]) : -p[0]; };
export const dockingTrig = (s, trig) => (!trig.phase || s.guid.st.phase === trig.phase) && qtyOf(s, trig.qty) <= trig.value;
export function armDocking(st, inj) { st.pending = inj; }
// whole cycles over sec (the loop of sim.run(sec)), the armed kick applied before the first step at which its trigger holds
export function stepArmed(st, s, sec) {
  const end = s.t + sec;
  while (!s.rep.done && s.t < end - 1e-9) {
    const inj = st.pending;
    if (inj && dockingTrig(s, inj.trigger)) {
      st.pending = null; st.injection = { kind: inj.kind, params: { ...(inj.params || {}), trigger: { qty: inj.trigger.qty, value: r3(inj.trigger.value), phase: inj.trigger.phase ?? null, at: r3(qtyOf(s, inj.trigger.qty)) } }, step: cycleOf(s), sim_t_s: r3(s.t) };
      applyDockingKick(s, inj);
    }
    s.step();
  }
  return cycleOf(s);
}
// one clean run with the page's constructor arguments: the cycle at which an armed kick would land (null: never) and the end time
export function dockingOracle(args, inj = null) { const s = createRealSim(args); let injAt = null; while (!s.rep.done) { if (inj && injAt === null && dockingTrig(s, inj.trigger)) injAt = cycleOf(s); s.step(); } return { injAt, endT: s.t }; }
export const dockingTriggerStep = (args, inj) => dockingOracle(args, inj).injAt;
export function centrelinePose(cam, s, lookAt) {
  const x = s.x, pr = qRotate(s.q, SHIP_PORT, [0, 0, 0]), f = qRotate(s.q, [1, 0, 0], [0, 0, 0]), u = qRotate(s.q, [0, 1, 0], [0, 0, 0]), port = [x[0] + pr[0], x[1] + pr[1], x[2] + pr[2]];
  cam.position.set(port[0] + 0.5 * f[0], port[1] + 0.5 * f[1], port[2] + 0.5 * f[2]); cam.up.set(u[0], u[1], u[2]);
  lookAt(port[0] + 100 * f[0], port[1] + 100 * f[1], port[2] + 100 * f[2]);
}
export async function setup(family, p) {
  const R = window.__real, q = new URLSearchParams(location.search), args = { seed: +q.get('seed'), start: q.get('start'), nav: q.get('nav') || 'noisy', filter: q.get('filter') !== '0' };
  if (R.sim.t !== 0) throw new Error(`the docking sim ran to t = ${R.sim.t} before setup (the page must never be begun)`);
  if (JSON.stringify(R.sim.R) !== JSON.stringify(drawRun(args.seed, { start: args.start }))) throw new Error('the URL arguments do not reproduce the page run');
  const armed = p.inject && p.inject.at === 'runtime' ? p.inject : null, o = dockingOracle(args, armed);
  const st = { R, p, args, own: null, injection: null, pending: null, injAt: o.injAt, endT: o.endT, snaps: [], cl: false, liveRep: null };
  if (armed) armDocking(st, armed);
  return st;
}
export function centreline(st, on) {
  const cam = st.R.scene.camera;
  if (!on) { delete cam.lookAt; st.cl = false; return false; }
  const proto = Object.getPrototypeOf(cam).lookAt;
  cam.lookAt = function () { centrelinePose(this, sim(st), (x, y, z) => proto.call(this, x, y, z)); return this; };
  st.cl = true; return true;
}
export const check = (st) => { const s = sim(st), p = portRel(s.x, s.q); return { ok: true, t: s.t, cycle: cycleOf(s), rho: Math.hypot(p[0], p[1], p[2]), axial: -p[0], phase: s.guid.st.phase, done: s.rep.done, cl: st.cl, injAt: st.injAt, endT: st.endT, injected: st.injection }; };
export const skip = (st, sec) => stepArmed(st, sim(st), sec);
export function cycle(st, n = 1) { for (let i = 0; i < n; i++) stepArmed(st, sim(st), 0.1); return cycleOf(sim(st)); }
export function ownSim(st, { failed }) {
  const own = createRealSim({ ...st.args, run: { ...drawRun(st.args.seed, { start: st.args.start }), failed } }), sc = st.R.scene, frame0 = sc.frame, warp0 = sc.setWarp;
  st.own = own;
  sc.frame = (s, dt) => frame0(own, dt, own.t);
  sc.setWarp = () => { const r = Math.hypot(own.x[0], own.x[1], own.x[2]), w = Math.min(60, Math.max(2, r / 3)); warp0(own.cmd.burn ? Math.min(w, 5) : w); };
  return true;
}
export function capture(st, { mode }) {
  if ((mode === 'centreline') !== st.cl) throw new Error(`a ${mode} capture with the centreline override ${st.cl ? 'on' : 'off'}`);
  const cv = document.getElementById('view'), gl = cv.getContext('webgl2') || cv.getContext('webgl');
  if (!gl.getContextAttributes().preserveDrawingBuffer) throw new Error('preserveDrawingBuffer is off');
  return { png: cv.toDataURL('image/png'), cam: camOf(st.R.scene.camera, mode), clock: { date_ms: Date.now(), perf_ms: performance.now() }, step: cycleOf(sim(st)) };
}
export function snap(st) { const s = sim(st), sp = st.R.scene.space; st.snaps.push({ step: cycleOf(s), now: dockingNow(s), facts: dockingFacts(s, { space: sp ? { sunLit: sp.sunLit, earthInFrame: null } : null }), x: Array.from(s.x), q: Array.from(s.q) }); return cycleOf(s); }
// s.run() steps the sim's own closure: an armed kick that has not fired by now stays unfired, as in the samples' labels
export function finish(st) { const s = sim(st); s.run(); st.liveRep = s.rep; return { result: s.rep.result, reason: s.rep.reason, t: s.rep.t }; }
// a branch replay that ends in timeout is a sample-level discard (schema Discard, e.discard), returned instead of thrown
export function branches(st, { step }) {
  const s = st.snaps.find((x) => x.step === step), args = st.own ? { ...st.args, run: st.own.R } : st.args;
  try { return { now: s.now, facts: s.facts, outcome: dockingBranches({ args, injection: st.injection, step, liveRep: st.liveRep, liveX: s.x, liveQ: s.q }) }; }
  catch (e) { if (e && e.discard) return { discard: e.message }; throw e; }
}
