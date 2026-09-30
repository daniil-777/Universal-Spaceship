// vlm/capture/probe/corridor.js — S/A in the page: env.step wrapped as an own property (delete restores the prototype
// method), the prev=cur no-op that makes a drawn frame exact, the exactness check against the drawn ship, capture through
// __ap.capture() (composer + toDataURL, no HUD, no pip), and labels on the rendered env (R3).
import { corridorLabel, ppoPilot, cpaTrigger, clearance } from '../../gen/labels/corridor.js';
import { corridorFacts } from '../../gen/labels/corridor_facts.js';
import { projectSphere } from '../../gen/labels/camera.js';
// real host time for the throughput numbers (§7.6): page.clock replaces performance, but Playwright 1.63 keeps the real object
// in __pwClock.builtins (pinned internals; freeze mode has no fake clock); only durations are used, never stored in facts
const rt = () => (globalThis.__pwClock && globalThis.__pwClock.builtins ? globalThis.__pwClock.builtins.performance.now() : performance.now());
// measure (throughput runs only): a second toDataURL of the same drawing buffer in the same task times the PNG encode alone
export const camOf = (c) => ({ mode: 'chase', fov_deg: c.fov, aspect: c.aspect, near: c.near, far: c.far, matrixWorldInverse: Array.from(c.matrixWorldInverse.elements), projectionMatrix: Array.from(c.projectionMatrix.elements) });
// S flies the pretrained belt pilot (§3.2): a missing or fresh policy is a hard error (not a discard), so the drive stops
export function assertBeltPilot(ap, p) {
  const agent = ap.agent, why = [];
  if (!/^pretrained /.test(String(ap.policy)) || ap.policyError) why.push(`policy ${ap.policy}${ap.policyError ? ` (${ap.policyError})` : ''}`);
  if (ap.pilot !== agent) why.push('the flying pilot is not the belt agent');
  if (p.policySteps !== null && p.policySteps !== undefined && (!agent || agent.steps !== p.policySteps)) why.push(`agent steps ${agent && agent.steps} != model/policy.json ${p.policySteps}`);
  if (why.length) throw new Error(`S needs the pretrained belt pilot: ${why.join('; ')}`);
}
export async function setup(family, p) {
  const ap = window.__ap, env = ap.env, step = Object.getPrototypeOf(env).step;
  if (family === 'S') assertBeltPilot(ap, p);
  const st = { family, p, ap, env, noop: 0, noopDone: 0, crashed: false, override: null, lastRaw: null };
  env.step = function (a) {
    if (st.noop > 0) { st.noop--; st.noopDone++; return { reward: 0, done: false, truncated: false, progress: 0 }; }
    const act = st.override ? st.override(env) : a; st.lastRaw = Array.from(act);
    const r = step.call(env, act); if (r.done) st.crashed = true; return r;
  };
  return st;
}
export function check(st) {
  const ap = st.ap, t = ap.terrainStats ? ap.terrainStats() : null;
  const strips = !t || ['far', 'near', 'city'].every((k) => !t[k] || (t[k].loads === t[k].copies + t[k].fails && t[k].fails === 0));
  const meshy = [ap.mountains, ap.chunks].every((m) => !m || typeof m.stats !== 'string' || !m.stats.includes('L'));
  return { ok: strips && meshy, steps: st.env.steps, crashed: st.crashed || ap.state.crashTimer > 0, noopDone: st.noopDone, textures: !!ap.texturesReady,
    atmosphere: ap.space ? ap.space.atmosphere : 0, atmo: ap.state.atmo, route: ap.state.route, world: st.env.world, lap: !!(st.env.hf && st.env.hf.lap), policyAtmo: !!ap.policyAtmo, policyError: ap.policyError || null,
    renderScale: ap.renderScale ?? 1, pixelRatio: ap.debug().pixelRatio };
}
export function requestNoop(st) { st.noop = 1; return st.noopDone; }
export function capture(st) {
  const ap = st.ap, d = ap.scene.getObjectByName('ship').position, p = st.env.ship.p;
  // every captured frame is full resolution (§7.1): adaptQuality must not have lowered the render scale since setup
  const rs = ap.renderScale ?? 1, pr = ap.debug().pixelRatio;
  if (rs !== 1 || pr !== 1) return { exact: false, why: `renderScale ${rs} pixelRatio ${pr} at capture` };
  if (Math.hypot(d.x - p[0], d.y - p[1], d.z - p[2]) >= 1e-3) return { exact: false };
  const t0 = rt(), png = ap.capture(), t1 = rt(), enc = st.p.measure ? (document.getElementById('view').toDataURL('image/png'), rt() - t1) : null;
  return { exact: true, png, cam: camOf(ap.camera), clock: { date_ms: Date.now(), perf_ms: performance.now() }, step: st.env.steps, ms: { capture: t1 - t0, encode: enc } };
}
export const trigger = (st, { cNear }) => cpaTrigger(st.env) || clearance(st.env).value < cNear;
function spaceOf(ap, cam) {
  const s = ap.space; if (!s) return null; const o = s.orbitInfo; let moon = false;
  ap.scene.traverse((m) => { if (!moon && m.isMesh && m.name === 'moon' && m.visible) { m.geometry.computeBoundingSphere(); const c = m.geometry.boundingSphere.center.clone().applyMatrix4(m.matrixWorld); moon = projectSphere(cam, [c.x, c.y, c.z], m.geometry.boundingSphere.radius * m.matrixWorld.getMaxScaleOnAxis(), 896, 504).inFrame; } });
  return { sunLit: s.sunLit, body: o.body, lat: o.lat, lon: o.lon, moonInFrame: moon, earthInFrame: null };
}
export function label(st, { cNear }) {
  const ap = st.ap, cam = camOf(ap.camera), t0 = rt(), { safety, safety_eye } = corridorLabel(st.env, { cNear, pilot: st.family === 'S' ? ppoPilot(ap.agent) : null }), t1 = rt();
  const facts = corridorFacts(st.env, cam, { sky: st.p.sky ?? null, route: st.p.route ?? null, space: spaceOf(ap, cam) });
  return { facts, safety, safety_eye, cam, ms: { rollout: t1 - t0, facts: rt() - t1 } };
}
export function restore(st) { delete st.env.step; return true; }
export const camInCloud = (st) => { const c = st.ap.camera.position, w = st.env.weather; return w ? w.cloudAt(c.x, c.y, c.z) : 0; };
