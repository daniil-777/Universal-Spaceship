// feed.js — the Narrator's eyes on the page. The pure parts (Node-tested): the scenario family, the sim clock and episode
// key from the page's globals, verdict hysteresis and the centre 16:9 crop. createFeed() is the frame grab: installed only
// while the Narrator is on, it wraps EffectComposer.prototype.render (the same importmap module every page uses: the game
// S/A, the Earth zoom Z, the landing L and the docking D), and right after the original render, in the same task (no
// preserveDrawingBuffer anywhere), copies the centre 16:9 of the canvas at 896x504 (the training capture size). Frames go
// to Pilot Eye on the family's sim-time grid (gridIndex, frame_dt from labels.json) and not while the tab is hidden; a
// snapshot for the Narrator is taken on request. The wrap never throws into the render loop and is removed on close.
import { gridIndex } from '../../vlm/web/pilot-eye.js';

export const FRAME_W = 896, FRAME_H = 504;
const FRAME_DT = { S: 0.2, A: 0.2, L: 0.25, D: 2 }, M_PER_U = 19;

// Z if the zoom is open, L on the landing page, D on the docking page, A in the atmosphere, else S (null: nothing to see)
export function familyOf(g = globalThis) {
  if (g.__landing) return 'L';
  if (g.__real) return g.__real.scene ? 'D' : null;
  const ap = g.__ap;
  if (!ap || !ap.ready) return null;
  if (ap.zoom && ap.zoom.active) return 'Z';
  return ap.state && ap.state.atmo ? 'A' : 'S';
}
export const frameDtOf = (family, nominal = null) => (nominal && nominal[family]) || FRAME_DT[family] || 1;
// the sim time in seconds and a key whose change starts a new episode (a new landing or docking run, another route)
export function clockOf(family, g = globalThis) {
  if (family === 'L') { const s = g.__landing && g.__landing.sim; return { t: s && s.flight ? s.flight.t : 0, key: s }; }
  if (family === 'D') { const s = g.__real && g.__real.sim; return { t: s ? s.t : 0, key: s }; }
  const ap = g.__ap || {}, env = ap.env || {};
  return { t: env.t || 0, key: family === 'A' ? `A:${ap.state && ap.state.route}` : family };
}
export function createEpisodes() {
  let id = 0, fam = null, key, lastT = -Infinity;
  return { id(family, t, k) { if (family !== fam || t < lastT || k !== key) { id++; fam = family; key = k; } lastT = t; return id; } };
}
// the telemetry the page gives for free, as Context-line facts (vlm/gen/text/context.js TELEMETRY ids)
export function factsOf(family, g = globalThis) {
  const ap = g.__ap;
  if ((family === 'S' || family === 'A') && ap && ap.env && ap.env.ship) return { 'ship.speed_m_s': { v: +(Math.hypot(...ap.env.ship.v) * M_PER_U).toFixed(3), unit: 'm/s' } };
  if (family === 'Z' && ap && ap.zoom && ap.zoom.info && Number.isFinite(ap.zoom.info.rangeKm)) return { 'view.range_km': { v: +ap.zoom.info.rangeKm.toFixed(2), unit: 'km' } };
  return {};
}
// a new verdict shows once it holds for `hold` decisions in a row; UNSAFE shows at once; the first verdict shows at once
export function createHysteresis({ hold = 2 } = {}) {
  let shown = null, cand = null, n = 0;
  return {
    push(v) {
      if (v === shown) { cand = null; n = 0; return shown; }
      if (shown === null || v === 'UNSAFE') { shown = v; cand = null; n = 0; return shown; }
      if (v === cand) n++; else { cand = v; n = 1; }
      if (n >= hold) { shown = v; cand = null; n = 0; }
      return shown;
    },
    reset() { shown = null; cand = null; n = 0; },
    get value() { return shown; },
  };
}
// [sx, sy, sw, sh]: the largest centred 16:9 rectangle (three.js fov is vertical, so wide windows lose only the sides)
export function centerCrop(w, h, aspect = 16 / 9) {
  if (!(w > 0 && h > 0)) return [0, 0, 1, 1];
  if (w / h > aspect) { const sw = Math.round(h * aspect); return [Math.floor((w - sw) / 2), 0, sw, h]; }
  const sh = Math.round(w / aspect); return [0, Math.floor((h - sh) / 2), w, sh];
}

function grab(canvas) {
  const [sx, sy, sw, sh] = centerCrop(canvas.width, canvas.height);
  try { return createImageBitmap(canvas, sx, sy, sw, sh, { resizeWidth: FRAME_W, resizeHeight: FRAME_H, resizeQuality: 'high' }); } catch { /* fall through */ }
  const c = document.createElement('canvas'); c.width = FRAME_W; c.height = FRAME_H;
  const g = c.getContext('2d'); g.imageSmoothingQuality = 'high'; g.drawImage(canvas, sx, sy, sw, sh, 0, 0, FRAME_W, FRAME_H);
  return createImageBitmap(c);
}

// onFrame({bitmap, sim_t_s, family, episode_id}) receives each Pilot Eye frame (the receiver owns and closes the bitmap);
// nominal: labels.json nominal_frame_dt, or a function returning it (known once the model folder has been probed)
export function createFeed({ onFrame, nominal = null, g = globalThis, doc = globalThis.document, zoomEveryMs = 1000, maxInflight = 2 }) {
  const episodes = createEpisodes();
  let snap = null;
  let proto = null, orig = null, wrapped = null, busy = false, on = false, lastGrid = null, lastEp = null, lastZ = -Infinity, inflight = 0, errors = 0;
  function after(composer) {
    const cv = composer.renderer && composer.renderer.domElement;
    if (!cv || cv.id !== 'view' || composer.renderToScreen === false || !cv.width) return;
    if (snap) { const w = snap; snap = null; grab(cv).then(w.ok, w.no); }
    if (!onFrame || (doc && doc.hidden) || inflight >= maxInflight) return;
    const family = familyOf(g); if (!family) return;
    let sim_t_s, episode_id;
    if (family === 'Z') {
      const now = performance.now(); if (now - lastZ < zoomEveryMs) return;
      lastZ = now; sim_t_s = now / 1000; episode_id = episodes.id('Z', 0, 'Z');
    } else {
      const c = clockOf(family, g), gi = gridIndex(c.t, frameDtOf(family, typeof nominal === 'function' ? nominal() : nominal)); episode_id = episodes.id(family, c.t, c.key);
      if (gi === lastGrid && episode_id === lastEp) return;
      lastGrid = gi; lastEp = episode_id; sim_t_s = c.t;
    }
    inflight++;
    grab(cv).then((bitmap) => { inflight--; if (on) onFrame({ bitmap, sim_t_s, family, episode_id }); else bitmap.close(); }, () => { inflight--; errors++; });
  }
  return {
    async install() {
      if (on) return;
      on = true;
      if (!proto) {
        const { EffectComposer } = await import('three/addons/postprocessing/EffectComposer.js');
        proto = EffectComposer.prototype;
      }
      if (!on || wrapped) return;
      orig = proto.render;
      const base = orig;
      wrapped = function (deltaTime) {
        const r = base.call(this, deltaTime);
        if (on && !busy) { busy = true; try { after(this); } catch { errors++; } finally { busy = false; } }
        return r;
      };
      proto.render = wrapped;
    },
    uninstall() {
      on = false;
      if (proto && wrapped && proto.render === wrapped) proto.render = orig;
      wrapped = null; lastGrid = null; lastEp = null;
      if (snap) { const w = snap; snap = null; w.no(new Error('the Narrator was closed')); }
    },
    // the next rendered frame at 896x504 (an ImageBitmap the caller owns); a paused or hidden page times out
    snapshot(timeoutMs = 1500) {
      if (snap) return snap.promise;
      const w = {}; w.promise = new Promise((ok, no) => {
        w.t = setTimeout(() => { if (snap === w) snap = null; no(new Error('no frame was rendered')); }, timeoutMs);
        w.ok = (b) => { clearTimeout(w.t); ok(b); }; w.no = (e) => { clearTimeout(w.t); no(e); };
      });
      snap = w; return w.promise;
    },
    get installed() { return !!wrapped; },
    get errors() { return errors; },
  };
}
