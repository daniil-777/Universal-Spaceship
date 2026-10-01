// chat/web/launcher.js — CAPCOM on the Astro Pilot page (spec §7): a small glass "CAPCOM" button that opens the chat panel
// (demo.html?embed=1 in an iframe; on phones a bottom sheet). Lazy: until the first click only this module, prompt.js and a few CSS
// rules load — no worker, no kb, no model; the panel starts its worker (kb, then the model download) when it first opens. The state line
// follows the visitor: the scene comes from the URL flags (refined by the Playbox toggles in the main app) and a per-tab `seen` list
// (sessionStorage) remembers the scenes opened this session. The button parks in the free bottom-right spot clear of each scenario's HUD.
//   index.html?capcom=0 (no launcher) · ?capcom=<model folder URL | Hub id> · ?capcomkb=<kb.json URL> · ?capcomdevice=auto|webgpu|wasm|none
//   · ?capcomdtype=q4f16|q4 · ?capcomdebug=1 (TTFT / tok/s under each answer)
import { HIGHLIGHTS } from './prompt.js';

export const CONFIG = {
  model: '/__capcom/models/capcom-pilot-web-acc1',                                  // local dev: chat/web/serve.mjs serves LaCie's chat/models
  kb: [new URL('../kb/kb.json', import.meta.url).href, '/__capcom/kb/kb.json'],     // the repo's copy when one ships, else serve.mjs's
  // (on a loopback host — local dev with serve.mjs — LaCie's live kb is tried first: no 404 probe in the console, the freshest notes)
  demo: new URL('./demo.html', import.meta.url).href,
  storageKey: 'capcom.seen',
};
export const KEYS = Object.freeze(HIGHLIGHTS.map(([k]) => k));
const VALID = new Set(KEYS), CITIES = new Set(['newyork', 'london', 'moscow', 'dubai', 'mega']); // src/terrain.js ROUTES with a skyline

// the scene a page URL boots into (index.html → src/app.js, src/landing, src/real): the scenarios first, then atmospheric flight (as
// app.js decides it: ?atmo=1, or a city route whose skyline is not switched off), lunar orbit, live training; else the asteroid belt
export function sceneFromUrl(search) {
  const q = new URLSearchParams(search), route = q.get('route');
  if (q.get('scenario') === 'landing') return 'landing';
  if (q.get('scenario') === 'real' || q.get('real') === '1') return 'docking';
  const skyline = q.get('skyline') === '1' || (q.get('skyline') !== '0' && CITIES.has(route));
  if (skyline) return 'cities';
  if (q.get('atmo') === '1') return route === 'china' ? 'pillars' : q.get('sky') === 'storm' ? 'weather' : route === 'alps' ? 'alps' : 'atmo';
  if (q.get('orbit') === 'moon') return 'moon';
  if (q.get('train') === '1') return 'training';
  return 'belt';
}

// the main app's live state (its Playbox controls mirror it) → the scene; the scenarios and pages without the Playbox keep the URL's
export function sceneFromDom(d, urlScene = 'belt') {
  if (!d || urlScene === 'landing' || urlScene === 'docking') return urlScene;
  if (d.zooming) return 'zoom';
  if (d.atmo) return d.skyline ? 'cities' : d.route === 'china' ? 'pillars' : d.sky === 'storm' ? 'weather' : d.route === 'alps' && urlScene === 'alps' ? 'alps' : 'atmo';
  if (d.orbit === 'moon') return 'moon';
  return urlScene === 'training' ? 'training' : 'belt';
}
export function domSnapshot(doc) {
  const atmo = doc.getElementById('swAtmo');
  if (!atmo) return null;
  const on = (id, attr) => (doc.querySelector(`#${id} button.on`) || { dataset: {} }).dataset[attr] || null;
  return { zooming: doc.body.classList.contains('zooming'), atmo: atmo.checked, skyline: !!(doc.getElementById('swSkyline') || {}).checked,
    route: on('routeSeg', 'route'), sky: on('skySeg', 'sky'), orbit: on('orbitSeg', 'orbit') };
}

// the seen list: highlight keys only, in first-seen order, no repeats (input: an array, a JSON array or a comma list)
export function parseSeen(v) {
  let a = v;
  if (typeof v === 'string') { try { a = JSON.parse(v); } catch { a = v.split(','); } }
  if (!Array.isArray(a)) return [];
  return [...new Set(a.map((x) => String(x).trim()).filter((k) => VALID.has(k)))];
}
export const addSeen = (list, scene) => (VALID.has(scene) && !list.includes(scene) ? [...list, scene] : list);
export const seenFor = (list, scene) => list.filter((k) => k !== scene); // what the state line calls seen: everything but the scene

// ?capcom / ?capcomkb / ?capcomdevice / ?capcomdtype / ?capcomdebug → the panel's config; null = the launcher is off (?capcom=0)
const LOOPBACK = new Set(['localhost', '127.0.0.1', '[::1]', '::1']);
export function resolveConfig(search, cfg = CONFIG, host = globalThis.location ? location.hostname : '') {
  const q = new URLSearchParams(search), c = (q.get('capcom') || '').trim();
  if (['0', 'off', 'false', 'no'].includes(c.toLowerCase())) return null;
  const device = q.get('capcomdevice'), dtype = q.get('capcomdtype');
  return { model: c && !['1', 'on', 'true', 'yes'].includes(c.toLowerCase()) ? c : cfg.model, kbs: q.get('capcomkb') ? [q.get('capcomkb')]
    : LOOPBACK.has(host) ? [...cfg.kb].reverse() : [...cfg.kb],
    device: ['auto', 'webgpu', 'wasm', 'none'].includes(device) ? device : 'auto', dtype: ['q4f16', 'q4'].includes(dtype) ? dtype : null,
    debug: q.get('capcomdebug') === '1' };
}
// the first kb candidate that answers a HEAD request (the last one when none does: the panel then reports it cannot start)
export async function pickKb(kbs, fetchFn = globalThis.fetch) {
  for (const u of kbs.slice(0, -1)) { try { if ((await fetchFn(u, { method: 'HEAD', cache: 'no-store' })).ok) return u; } catch { /* next */ } }
  return kbs[kbs.length - 1];
}
export function frameUrl(cfg, kb, scene, seen, demo = CONFIG.demo) {
  const p = new URLSearchParams({ embed: '1', scene, seen: seen.join(','), model: cfg.model, kb, device: cfg.device });
  if (cfg.dtype) p.set('dtype', cfg.dtype);
  if (cfg.debug) p.set('debug', '1');
  return `${demo}?${p}`;
}

// a free spot for a w×h button: {right, bottom} offsets, preferring the lowest row, then the one nearest the right edge, clear of every
// obstacle rect (with a gap); null when the screen has none
export function freeSpot(w, h, obstacles, vw, vh, m = 22, gap = 10) {
  const hit = (x, y) => obstacles.some((o) => x < o.right + gap / 2 && x + w > o.left - gap / 2 && y < o.bottom + gap / 2 && y + h > o.top - gap / 2);
  const bottoms = [...new Set([m, ...obstacles.map((o) => Math.ceil(vh - o.top + gap))])].filter((b) => b >= m && b + h <= vh - m).sort((a, b) => a - b);
  const rights = [...new Set([m, ...obstacles.map((o) => Math.ceil(vw - o.left + gap))])].filter((r) => r >= m && r + w <= vw - m).sort((a, b) => a - b);
  for (const b of bottoms) for (const r of rights) if (!hit(vw - r - w, vh - b - h)) return { right: r, bottom: b };
  return null;
}
// a parked spot stays while it is on screen and clear (no jumping each time a HUD box changes size)
export function blocked(spot, w, h, obstacles, vw, vh) {
  const x = vw - spot.right - w, y = vh - spot.bottom - h;
  return x < 0 || y < 0 || obstacles.some((o) => x < o.right && x + w > o.left && y < o.bottom && y + h > o.top);
}

// the scenarios hide every other child of <body> (body.landing > *:not(#view):not(.lnd) { display: none !important }): the launcher lives
// in a display:contents root whose rule outranks theirs
const CSS = `
html body > #capcom-root.capcom-root.capcom-root{display:contents!important}
.capcom-btn{position:fixed;right:22px;bottom:22px;z-index:19;display:flex;align-items:center;gap:8px;height:40px;padding:0 15px 0 8px;border-radius:99px;cursor:pointer;
  font:600 12px/1 -apple-system,BlinkMacSystemFont,"SF Pro Text","Helvetica Neue",system-ui,sans-serif;letter-spacing:.09em;color:#f5f5f7;
  background:rgba(24,27,36,.62);border:1px solid rgba(255,255,255,.12);backdrop-filter:blur(24px) saturate(170%);-webkit-backdrop-filter:blur(24px) saturate(170%);
  box-shadow:0 10px 32px rgba(0,0,0,.4),inset 0 1px 0 rgba(255,255,255,.06);transition:transform .15s,background .2s,opacity .3s;animation:capcom-in .5s cubic-bezier(.2,.8,.2,1)}
.capcom-btn:hover{background:rgba(40,44,58,.7)} .capcom-btn:active{transform:scale(.96)} .capcom-btn:focus-visible{outline:2px solid #0a84ff;outline-offset:2px}
.capcom-btn i{width:24px;height:24px;border-radius:50%;flex:none;background:radial-gradient(circle at 32% 28%,#9fd0ff,#2a6df4 45%,#131a3a 100%);
  box-shadow:0 0 0 1px rgba(255,255,255,.14),0 4px 12px rgba(42,109,244,.4)}
.capcom-open .capcom-btn,body.nohud .capcom-btn,.capcom-wait .capcom-btn{opacity:0;pointer-events:none}
.capcom-panel{position:fixed;right:22px;bottom:22px;z-index:40;width:min(420px,calc(100vw - 44px));height:min(680px,calc(100dvh - 44px));border-radius:24px;overflow:hidden;
  background:rgba(24,27,36,.62);border:1px solid rgba(255,255,255,.08);backdrop-filter:blur(28px) saturate(170%);-webkit-backdrop-filter:blur(28px) saturate(170%);
  box-shadow:0 40px 100px rgba(0,0,0,.55),inset 0 1px 0 rgba(255,255,255,.06);transform-origin:bottom right;
  transition:opacity .28s cubic-bezier(.2,.8,.2,1),transform .28s cubic-bezier(.2,.8,.2,1),visibility 0s .28s;opacity:0;transform:translateY(14px) scale(.97);visibility:hidden}
.capcom-open .capcom-panel{opacity:1;transform:none;visibility:visible;transition-delay:0s}
.capcom-panel iframe{display:block;width:100%;height:100%;border:0;background:transparent;color-scheme:dark}
@media (max-width:600px){.capcom-btn{width:46px;height:46px;padding:0;justify-content:center}.capcom-btn span{display:none}.capcom-btn i{width:28px;height:28px}
  .capcom-panel{left:0;right:0;bottom:0;width:auto;height:min(82dvh,680px);padding-bottom:env(safe-area-inset-bottom);border-radius:22px 22px 0 0;border-bottom:0;transform-origin:bottom center;transform:translateY(40px)}}
@keyframes capcom-in{from{opacity:0;transform:translateY(10px)}}
@media (prefers-reduced-motion:reduce){.capcom-btn,.capcom-panel{animation:none;transition:none}}`;

// fixed boxes the button must not cover: the page's HUD (direct children of <body> and one level down), visible and taking clicks —
// toasts and other click-through overlays are skipped, except the landing's radio box, which comes and goes
function obstacles(doc, skip) {
  const out = [], vw = innerWidth, vh = innerHeight;
  for (const el of doc.body.querySelectorAll(':scope > *, :scope > * > *')) {
    if (skip.includes(el) || el.classList.contains('hidden')) continue;
    const cs = getComputedStyle(el);
    if (cs.position !== 'fixed' || cs.display === 'none' || cs.visibility === 'hidden') continue;
    if (!el.matches('.lnd-atc') && (+cs.opacity < 0.05 || cs.pointerEvents === 'none')) continue;
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2 || (r.width >= vw - 2 && r.height >= vh - 2)) continue;
    out.push({ left: r.left, top: r.top, right: r.right, bottom: r.bottom });
  }
  return out;
}

const session = () => { try { return globalThis.sessionStorage || null; } catch { return null; } }; // blocked storage throws on access

export function mount({ doc = document, cfg = resolveConfig(location.search), storage = session() } = {}) {
  if (!cfg) return null;
  const urlScene = sceneFromUrl(location.search), scene = () => sceneFromDom(domSnapshot(doc), urlScene);
  let seen = [];
  const remember = (s) => {
    try { seen = addSeen(parseSeen(storage.getItem(CONFIG.storageKey) || '[]'), s); storage.setItem(CONFIG.storageKey, JSON.stringify(seen)); } catch { seen = addSeen(seen, s); }
  };
  remember(urlScene);
  const style = doc.createElement('style'); style.textContent = CSS; doc.head.append(style);
  const root = doc.createElement('div'); root.id = 'capcom-root'; root.className = 'capcom-root';
  const btn = doc.createElement('button');
  Object.assign(btn, { type: 'button', className: 'capcom-btn', title: 'Ask CAPCOM, the Astro Pilot guide' });
  btn.setAttribute('aria-label', 'Open CAPCOM chat'); btn.setAttribute('aria-expanded', 'false'); btn.innerHTML = '<i></i><span>CAPCOM</span>';
  root.append(btn); doc.body.append(root);
  let panel = null, frame = null, spot = null, open = false;
  const loading = () => { const l = doc.getElementById('loading'); return !!l && !l.classList.contains('hidden') && getComputedStyle(l).display !== 'none'; };
  const place = (fresh = false) => {
    root.classList.toggle('capcom-wait', loading());
    if (open) return;
    const w = btn.offsetWidth, h = btn.offsetHeight, vw = innerWidth, vh = innerHeight, m = vw <= 600 ? 16 : 22;
    const obs = obstacles(doc, [btn, panel]);
    if (!fresh && spot && !blocked(spot, w, h, obs, vw, vh)) return;
    spot = freeSpot(w, h, obs, vw, vh, m) || { right: m, bottom: m };
    btn.style.right = `${spot.right}px`; btn.style.bottom = `${spot.bottom}px`;
  };
  const post = (m) => { if (frame && frame.dataset.loaded) frame.contentWindow.postMessage(m, location.origin); };
  const focusChat = () => { if (matchMedia('(pointer: fine)').matches) post({ type: 'capcom:focus' }); }; // phones: no keyboard pop-up
  const show = (on) => {
    open = on; root.classList.toggle('capcom-open', on); btn.setAttribute('aria-expanded', String(on));
    if (on) { const s = scene(); remember(s); post({ type: 'capcom:state', scene: s, seen: seenFor(seen, s) }); focusChat(); }
    else { place(true); btn.focus({ preventScroll: true }); }
  };
  async function first() {
    panel = doc.createElement('div'); panel.className = 'capcom-panel'; panel.setAttribute('role', 'dialog'); panel.setAttribute('aria-label', 'CAPCOM chat');
    root.append(panel); void panel.offsetWidth; show(true);
    const kb = await pickKb(cfg.kbs), s = scene(); remember(s);
    frame = doc.createElement('iframe'); frame.title = 'CAPCOM chat'; frame.src = frameUrl(cfg, kb, s, seenFor(seen, s));
    frame.addEventListener('load', () => { frame.dataset.loaded = '1'; if (open) focusChat(); });
    panel.append(frame);
  }
  btn.addEventListener('click', () => { if (!panel) first(); else show(!open); });
  addEventListener('message', (e) => {
    if (!frame || e.source !== frame.contentWindow || e.origin !== location.origin || !e.data) return;
    if (e.data.type === 'capcom:close') show(false);
  });
  addEventListener('keydown', (e) => { if (open && e.key === 'Escape') show(false); });
  addEventListener('resize', () => place(true));
  requestAnimationFrame(() => place(true));
  const timer = setInterval(() => { if (!doc.hidden) place(); }, 1000);
  return { root, button: btn, scene, seen: () => seen, open: () => (panel ? show(true) : first()), close: () => show(false), stop: () => clearInterval(timer) };
}

if (typeof document !== 'undefined' && !globalThis.__capcomLauncher) globalThis.__capcomLauncher = mount();
