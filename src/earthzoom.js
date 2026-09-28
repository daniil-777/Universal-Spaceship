// earthzoom.js — the Earth zoom view: a free-pan telescope over the real Earth, opened from the ship (Playbox "Zoom in"
// or Z; closed with Z or Esc). Its own scene, camera and post chain in kilometres around the view's target (the local
// East-Up-South frame of src/earthtiles.js): the Blue Marble globe of src/earth.js (sharing the flight scene's textures)
// for the far view and the nested imagery rings of src/earthrings.js close up. The ship keeps flying underneath — the
// learned pilot flies while the view is open — and nothing in the simulation is touched.
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { createEarth } from './earth.js';
import { createEarthRings } from './earthrings.js';
import { R_KM, MAX_LAT, ZOOM_CREDIT, sourceForLevel, tileSizeKm, createLocalFrame, globeAxes, levelFloat, pickInnerLevel, cameraPose, panTarget, clipPlanes, sunLocal } from './earthtiles.js';

export const ZOOM = { fovDeg: 45, minClearKm: 0.3, camClearKm: 0.15, maxRangeKm: 20000, maxTilt: 70 * Math.PI / 180, ringsFullBelowKm: 2500, ringsGoneAboveKm: 4000, hazeK: 0.55, hazeKm: 40 };
const STEER_KEYS = new Set(['KeyW', 'KeyA', 'KeyS', 'KeyD', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Equal', 'Minus', 'NumpadAdd', 'NumpadSubtract', 'KeyQ', 'KeyE', 'KeyR', 'KeyF']);
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const fade = (from, to, x) => { const t = clamp((x - from) / (to - from), 0, 1); return t * t * (3 - 2 * t); };
const fmtDeg = (x, pos, neg) => `${Math.abs(x).toFixed(4)}° ${x >= 0 ? pos : neg}`;
const fmtKm = (km) => (km >= 10 ? `${Math.round(km).toLocaleString('en-US')} km` : km >= 1 ? `${km.toFixed(1)} km` : `${Math.round(km * 1000)} m`);

export function attachEarthZoom({ renderer, space, host, texturePath = 'textures/' }) {
  const canvas = renderer.domElement, btn = document.getElementById('btnZoom'), sunSwitch = document.getElementById('swZoomSun'), hud = document.getElementById('zoomHud');
  const v = { lat: 0, lon: 0, rangeKm: 420, tilt: 0, heading: 0, groundKm: 0 }, keys = new Set(), ptrs = new Map(), last = { clearKm: 0, camAltKm: 0 };
  const pose = { pos: [0, 0, 0], up: [0, 0, 0], look: [0, 0, 0], camLat: 0, camLon: 0, camAltKm: 0 }, sun = [0, 0, 0], up = [0, 0, 0], mk = [0, 0, 0], m4 = new THREE.Matrix4();
  let active = false, view = null, wasManual = false, prevCredit = '', dayLight = !!(sunSwitch && sunSwitch.checked), pinch = 0, hudT = 0;

  function build() {
    const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(ZOOM.fovDeg, 1, 0.1, 1000), sunDir = new THREE.Vector3(0, 1, 0);
    const globe = createEarth({ texturePath, R: R_KM, position: new THREE.Vector3(0, -R_KM, 0), axis: new THREE.Vector3(0, 1, 0), sunDir, sunCol: new THREE.Color(1, 0.98, 0.95), spin: 0, textures: space.earthTextures || null });
    // the globe never hides the rings (its facets sit up to 0.5 km under the true sphere); its atmosphere shell goes on top
    globe.group.traverse((o) => { if (o.isMesh) o.material.depthWrite = false; });
    for (const o of globe.group.children) if (o.isMesh) o.renderOrder = 30;
    const marker = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 12), new THREE.MeshBasicMaterial({ color: 0x8ec5ff })); marker.renderOrder = 40;
    scene.add(globe.group, marker);
    const composer = new EffectComposer(renderer); composer.addPass(new RenderPass(scene, camera)); composer.addPass(new OutputPass());
    return { scene, camera, sunDir, globe, marker, composer, rings: createEarthRings(scene, renderer), frame: null, L0: null, size: new THREE.Vector2(), w: 0, h: 0, pr: 0 };
  }
  function rebase() {
    view.frame = createLocalFrame(v.lat, v.lon);
    const [a, b, c] = globeAxes(view.frame).map((x) => new THREE.Vector3().fromArray(x));
    view.globe.group.quaternion.setFromRotationMatrix(m4.makeBasis(a, b, c)); view.rings.rebase(view.frame);
  }
  function steer(dt) {
    const k = (c) => keys.has(c), step = v.rangeKm * 0.8 * dt;
    if (k('KeyW') || k('ArrowUp')) panTarget(v, 0, step);
    if (k('KeyS') || k('ArrowDown')) panTarget(v, 0, -step);
    if (k('KeyD') || k('ArrowRight')) panTarget(v, step, 0);
    if (k('KeyA') || k('ArrowLeft')) panTarget(v, -step, 0);
    if (k('Equal') || k('NumpadAdd')) v.rangeKm *= Math.exp(-1.5 * dt);
    if (k('Minus') || k('NumpadSubtract')) v.rangeKm *= Math.exp(1.5 * dt);
    if (k('KeyQ')) v.heading -= dt * Math.PI / 3;
    if (k('KeyE')) v.heading += dt * Math.PI / 3;
    if (k('KeyR')) v.tilt = Math.min(ZOOM.maxTilt, v.tilt + dt * 0.7);
    if (k('KeyF')) v.tilt = Math.max(0, v.tilt - dt * 0.7);
  }
  function placeMarker() {
    const V = view, e = host.entry(), cam = V.camera.position;
    V.frame.toLocal(e.lat, e.lon, V.rings.heightAt(e.lat, e.lon) + 0.02, mk); V.frame.upAt(e.lat, e.lon, up); V.marker.position.fromArray(mk);
    V.marker.visible = (cam.x - mk[0]) * up[0] + (cam.y - mk[1]) * up[1] + (cam.z - mk[2]) * up[2] > 0;
    V.marker.scale.setScalar(Math.max(0.004, 0.006 * cam.distanceTo(V.marker.position)));
  }
  function writeHud() {
    if (!hud) return;
    const gone = view.rings.info.failRate > 0.5 ? ' · live imagery unavailable' : '';
    hud.textContent = `${fmtDeg(v.lat, 'N', 'S')}  ${fmtDeg(v.lon, 'E', 'W')} · ${fmtKm(last.clearKm)} above ground · ground ${Math.round(v.groundKm * 1000)} m · ${sourceForLevel(view.L0).name}, level ${view.L0}${gone}`;
  }
  function render(dt) {
    if (!active) return;
    const V = view, cam = V.camera;
    renderer.getSize(V.size); const pr = renderer.getPixelRatio();
    if (V.size.x !== V.w || V.size.y !== V.h || pr !== V.pr) { V.w = V.size.x; V.h = V.size.y; V.pr = pr; V.composer.setPixelRatio(pr); V.composer.setSize(V.w, V.h); cam.aspect = V.w / V.h; }
    steer(dt);
    v.rangeKm = clamp(v.rangeKm, ZOOM.minClearKm / Math.max(Math.cos(v.tilt), 0.3), ZOOM.maxRangeKm);
    const here = V.frame.toLocal(v.lat, v.lon, 0, mk);
    if (Math.hypot(here[0], here[2]) > tileSizeKm(V.L0 ?? 4, v.lat)) rebase();
    V.L0 = pickInnerLevel(levelFloat(v.rangeKm, v.lat, cam.fov, V.h * pr), V.L0);
    v.groundKm = V.rings.heightAt(v.lat, v.lon);
    cameraPose(V.frame, v, pose);
    const clear = pose.camAltKm - Math.max(V.rings.heightAt(pose.camLat, pose.camLon), v.groundKm), lift = Math.max(0, ZOOM.camClearKm - clear);
    if (lift > 0) { V.frame.upAt(pose.camLat, pose.camLon, up); for (let i = 0; i < 3; i++) pose.pos[i] += up[i] * lift; }
    last.clearKm = clear + lift; last.camAltKm = pose.camAltKm + lift;
    cam.position.fromArray(pose.pos); cam.up.fromArray(pose.up); cam.lookAt(pose.look[0], pose.look[1], pose.look[2]);
    const cp = clipPlanes(last.clearKm, last.camAltKm); cam.near = cp.near; cam.far = cp.far; cam.updateProjectionMatrix();
    if (dayLight) { V.frame.upAt(v.lat, v.lon, sun); V.sunDir.set(sun[0] + 0.35, sun[1], sun[2] - 0.25).normalize(); }
    else V.sunDir.fromArray(sunLocal(space.orbitInfo.utc, V.frame, sun));
    V.globe.setHaze(fade(400, 100, last.camAltKm)); V.globe.update(0);
    V.rings.update(dt, { lat: v.lat, lon: v.lon, L0: V.L0, sun: V.sunDir, vis: 1 - fade(ZOOM.ringsFullBelowKm, ZOOM.ringsGoneAboveKm, last.camAltKm), hazeK: ZOOM.hazeK, hazeL: ZOOM.hazeKm });
    placeMarker();
    hudT += dt; if (hudT >= 0.1) { hudT = 0; writeHud(); }
    V.composer.render(dt);
  }
  function open() {
    if (active) return;
    if (!view) view = build();
    const e = host.entry();
    Object.assign(v, { lat: clamp(e.lat, -MAX_LAT, MAX_LAT), lon: e.lon, rangeKm: clamp(e.altKm, 1, ZOOM.maxRangeKm), tilt: 0, heading: 0 });
    view.L0 = null; rebase();
    wasManual = !!host.manual;
    if (wasManual) { host.setManual(false); host.toast('the autopilot flies while you look around'); }
    prevCredit = host.credit(); host.setCredit(ZOOM_CREDIT); host.flightControls(false);
    active = true; document.body.classList.add('zooming');
    if (hud) hud.classList.remove('hidden');
    if (btn) btn.textContent = 'Back to flight';
  }
  function close() {
    if (!active) return;
    active = false; keys.clear(); ptrs.clear(); pinch = 0; document.body.classList.remove('zooming');
    if (hud) hud.classList.add('hidden');
    if (btn) btn.textContent = 'Zoom in';
    host.setCredit(prevCredit); host.flightControls(true);
    if (wasManual) { host.setManual(true); host.toast('you have the controls again'); }
    wasManual = false;
  }

  const typing = (e) => e.target && /INPUT|TEXTAREA|SELECT/.test(e.target.tagName);
  window.addEventListener('keydown', (e) => {
    if (typing(e)) return;
    if (e.code === 'KeyZ' && !e.metaKey && !e.ctrlKey) { e.preventDefault(); if (active) close(); else open(); return; }
    if (!active) return;
    if (e.code === 'Escape') { close(); return; }
    if (STEER_KEYS.has(e.code)) { keys.add(e.code); e.preventDefault(); }
  });
  window.addEventListener('keyup', (e) => keys.delete(e.code));
  window.addEventListener('blur', () => keys.clear());
  canvas.addEventListener('pointerdown', (e) => {
    if (!active) return;
    canvas.setPointerCapture(e.pointerId); ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY, turn: e.button === 2 || e.ctrlKey || e.shiftKey });
  });
  canvas.addEventListener('pointermove', (e) => {
    const p = ptrs.get(e.pointerId);
    if (!active || !p) return;
    const dx = e.clientX - p.x, dy = e.clientY - p.y; p.x = e.clientX; p.y = e.clientY;
    if (ptrs.size >= 2) {
      const [a, b] = [...ptrs.values()], d = Math.hypot(a.x - b.x, a.y - b.y);
      if (pinch > 0 && d > 0) v.rangeKm *= pinch / d;
      pinch = d; v.tilt = clamp(v.tilt - dy * 0.002, 0, ZOOM.maxTilt);
    } else if (p.turn) { v.heading += dx * 0.005; v.tilt = clamp(v.tilt - dy * 0.005, 0, ZOOM.maxTilt); }
    else { const kmPerPx = v.rangeKm * 2 * Math.tan(ZOOM.fovDeg * Math.PI / 360) / Math.max(1, canvas.clientHeight); panTarget(v, -dx * kmPerPx, dy * kmPerPx); }
  });
  const release = (e) => { ptrs.delete(e.pointerId); pinch = 0; };
  canvas.addEventListener('pointerup', release); canvas.addEventListener('pointercancel', release);
  canvas.addEventListener('wheel', (e) => { if (!active) return; e.preventDefault(); v.rangeKm *= Math.exp(e.deltaY * 0.0015); }, { passive: false });
  canvas.addEventListener('contextmenu', (e) => { if (active) e.preventDefault(); });
  if (btn) btn.addEventListener('click', () => { if (active) close(); else open(); });
  if (sunSwitch) sunSwitch.addEventListener('change', () => { dayLight = sunSwitch.checked; });

  return {
    get active() { return active; }, open, close, toggle: () => (active ? close() : open()), render,
    goTo({ lat, lon, rangeKm, tilt = 0, heading = 0 }) { Object.assign(v, { lat: clamp(lat, -MAX_LAT, MAX_LAT), lon, rangeKm, tilt, heading }); if (view) rebase(); },
    get info() {
      if (!view) return { active };
      return { active, lat: v.lat, lon: v.lon, rangeKm: v.rangeKm, tilt: v.tilt, heading: v.heading, L0: view.L0, clearKm: last.clearKm, camAltKm: last.camAltKm,
        aspect: view.camera.aspect, settled: view.rings.settled, ...view.rings.info };
    },
    capture() { render(0); return canvas.toDataURL('image/png'); },
  };
}
