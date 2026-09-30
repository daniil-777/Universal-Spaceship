// vlm/capture/probe/zoom.js — Z in the page: fresh view per sample (close, reopen, goTo), the Z gate of spec §3.4, capture
// through zoom.capture(), and the pose facts from zoom.info plus the camera the RenderPass actually rendered.
import { ringLevels } from '../../../src/earthtiles.js';
import { ZOOM } from '../../../src/earthzoom.js';
import { zoomFacts, lightingOf } from '../../gen/labels/zoom.js';
const camOf = (c) => ({ mode: 'zoom', fov_deg: c.fov, aspect: c.aspect, near: c.near, far: c.far, matrixWorldInverse: Array.from(c.matrixWorldInverse.elements), projectionMatrix: Array.from(c.projectionMatrix.elements) });
export async function setup(family, p) { return { p, page: window.__zoomPage, v: null, Lmax: p.licence === 'nc' ? 14 : 15 }; }
export const check = (st) => ({ ok: true, ready: !!st.page.ready });
// the zoom loader has no job running or queued: every tile it asked for has been fetched, decoded and handed to the rings
export const loaderIdle = (st) => { const i = st.page.zoom.info; return !i.active || (i.inFlight === 0 && i.queued === 0); };
export function goView(st, v) { const P = st.page; st.v = v; Object.assign(P.view, { lat: v.lat, lon: v.lon, altKm: v.rangeKm, utc: v.utcMs }); P.setAlwaysDay(v.alwaysDay); P.zoom.close(); P.zoom.open(); P.zoom.goTo({ lat: v.lat, lon: v.lon, rangeKm: v.rangeKm, tilt: v.tilt, heading: v.heading }); return true; }
export function gate(st) {
  const i = st.page.zoom.info, want = new Set(ringLevels(i.L0)), rings = (i.rings || []).filter((r) => want.has(r.level)), why = [];
  if (!i.settled) why.push('not settled'); if (i.failRate !== 0) why.push(`failRate ${i.failRate}`); if (rings.length !== want.size) why.push('rings not placed');
  for (const r of rings) { if (r.have !== r.valid || r.hHave !== r.hValid) why.push(`ring ${r.level} incomplete`); if (r.vis < 0.99 || r.height < 0.99) why.push(`ring ${r.level} easing`); }
  if (Math.max(...rings.map((r) => r.level)) > st.Lmax) why.push('ring above Lmax'); if (i.camAltKm > 2500) why.push('camAltKm > 2500');
  return { ok: why.length === 0, why, L0: i.L0 };
}
// groundKm (ruling T6-d): zoom.info.groundKm where the site exposes it; otherwise from the rendered camera. goView rebases
// the local frame on the view's target, so cameraPose put the camera at (0, g, 0) + rangeKm·(up·cos tilt − fwd·sin tilt)
// with up = +y and fwd horizontal: g = y − rangeKm·cos(tilt), the rings.heightAt(target) the page used. Unknown (null)
// when the 0.15 km clearance lift moved the camera (clearKm sits at that floor).
function groundOf(i, c) {
  if (Number.isFinite(i.groundKm)) return { km: i.groundKm, src: 'info' };
  if (Math.abs(i.clearKm - ZOOM.camClearKm) < 1e-9) return { km: null, src: null };
  return { km: c.position.y - i.rangeKm * Math.cos(i.tilt), src: 'pose' };
}
// The capture waits for a steady state (a determinism fix, not a tolerance): the rendered camera, its clip planes and each
// drawn ring's blend (uVis) and relief (uHeightK) ease per frame toward fixed points that do not depend on when the tiles
// arrived (x += (1 - x)·e with e = 1 - exp(-0.048) stops at 1 - 10 ulp from any start), so a view captured once this signature
// stops changing is the same state in a cold and a warm run. steady() counts the frames it has not changed.
function sigOf() {
  const c = window.__zoomCam, rings = [];
  if (window.__zoomScene) window.__zoomScene.traverse((m) => { const u = m.isMesh && m.material && m.material.uniforms; if (u && u.uHeightK && m.visible) rings.push([m.renderOrder, u.uVis.value, u.uHeightK.value, u.uInnerOn.value]); });
  return JSON.stringify([Array.from(c.matrixWorldInverse.elements), Array.from(c.projectionMatrix.elements), rings]);
}
export function steady(st) { const s = sigOf(); st.same = s === st.sig ? (st.same || 0) + 1 : 0; st.sig = s; return st.same; }
export function capture(st) { const png = st.page.zoom.capture(); return { exact: true, png, cam: camOf(window.__zoomCam), clock: { date_ms: Date.now(), perf_ms: performance.now() }, info: st.page.zoom.info }; }
export function label(st) {
  const i = st.page.zoom.info, c = window.__zoomCam, cam = camOf(c), g = groundOf(i, c), lighting = lightingOf({ mode: st.v.alwaysDay ? 'always_day' : 'utc', utcMs: st.v.utcMs, lat: st.v.lat, lon: st.v.lon });
  const facts = zoomFacts(i, cam, { origin: { lat: i.lat, lon: i.lon }, lighting, rings: ringLevels(i.L0), groundKm: g.km });
  return { facts, lighting, cam, ground: { km: g.km === null ? null : +g.km.toFixed(4), src: g.src }, info: { lat: i.lat, lon: i.lon, rangeKm: i.rangeKm, L0: i.L0 } };
}
