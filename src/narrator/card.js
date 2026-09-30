// card.js — the Narrator in the site: a live verdict pill (Pilot Eye) at the top centre that morphs into a glass card
// (Narrator: Describe, Is it safe?, What now?, or a typed question), or a bottom sheet on phones. Loaded by boot.js on
// first use; while it is closed nothing of it runs (the frame grab is uninstalled and the Pilot Eye worker stopped).
// Both models are v0 and below their quality gates: the card says Beta and "Not for real flight decisions".
import { SPARKLE } from './boot.js';
import { createFeed, familyOf, factsOf, createHysteresis } from './feed.js';
import { createSentenceGate, CAPS } from './clean.js';
import { pillLabel, contextChip, humanTemplate, skyName } from './words.js';
import { placeBox } from './place.js';
import { createEngine, config, memory, hasWebGPU, TASKS, TOKENS, DOWNLOAD_MB } from './engine.js';
import { createMockEngine } from './mock.js';
import { createLifecycle } from './life.js';

const svg = (body, vb = '0 0 16 16') => `<svg viewBox="${vb}" aria-hidden="true" focusable="false">${body}</svg>`;
const st = 'fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"';
const ICON = {
  S: svg(`<circle cx="8" cy="8" r="3" ${st} stroke-width="1.4"/><ellipse cx="8" cy="8" rx="7" ry="2.4" transform="rotate(-24 8 8)" ${st} stroke-width="1.2"/>`),
  A: svg(`<path d="M1.5 13 6 5.2l2.7 4.3 1.8-2.6 4 6.1z" ${st} stroke-width="1.4"/>`),
  Z: svg(`<circle cx="8" cy="8" r="6.3" ${st} stroke-width="1.3"/><ellipse cx="8" cy="8" rx="2.7" ry="6.3" ${st} stroke-width="1.1"/><path d="M1.9 8h12.2" ${st} stroke-width="1.1"/>`),
  L: svg(`<path d="M5 14.5 7 1.8M11 14.5 9 1.8" ${st} stroke-width="1.35"/><path d="M8 4v1.4M8 7.4v1.8M8 11.3v2" ${st} stroke-width="1.15"/>`),
  D: svg(`<circle cx="8" cy="8" r="5.6" ${st} stroke-width="1.3"/><circle cx="8" cy="8" r="1.6" ${st} stroke-width="1.2"/><path d="M8 .9v2.4M8 12.7v2.4M.9 8h2.4M12.7 8h2.4" ${st} stroke-width="1.2"/>`),
};
const X = svg(`<path d="M1.5 1.5l7 7M8.5 1.5l-7 7" ${st} stroke-width="1.6"/>`, '0 0 10 10');
const SEND = svg(`<path d="M6 10.2V2.2M2.5 5.6 6 2.1l3.5 3.5" ${st} stroke-width="1.8"/>`, '0 0 12 12');
const HTML = `
<button class="nr-pill glass" type="button" aria-expanded="false"><span class="nr-pill-in"><span class="nr-st">${SPARKLE}</span><span class="nr-pl"></span><span class="nr-fam"></span><span class="nr-demo-p" hidden>Demo</span></span></button>
<section class="nr-card" role="dialog" aria-label="Narrator" tabindex="-1" data-view="idle">
  <div class="nr-glow" aria-hidden="true"><i></i></div>
  <div class="nr-clip glass"><div class="nr-in">
    <div class="nr-grab" aria-hidden="true"></div>
    <header class="nr-head">
      <div class="nr-thumb"><canvas width="192" height="108" aria-hidden="true"></canvas></div>
      <div class="nr-titles"><h2 class="nr-title">Narrator<span class="nr-beta">Beta</span><span class="nr-beta nr-demo" hidden title="Canned text, no models (?narratorMock=1)">Demo</span></h2><span class="nr-chip"><span class="nr-chip-i"></span><span class="nr-chip-t"></span></span></div>
      <button class="nr-x" type="button" aria-label="Close the Narrator">${X}</button>
    </header>
    <div class="nr-body" aria-live="polite">
      <div class="nr-view nr-v-text"><div class="nr-text"><div class="nr-tw"><p class="nr-tmpl"></p></div><p class="nr-out"></p></div></div>
      <div class="nr-view nr-v-consent"><p class="nr-lead">Narrator runs entirely on your device. It needs a one-time ${DOWNLOAD_MB} MB download.</p>
        <div class="nr-btns"><button class="btn" type="button" data-a="later">Not now</button><button class="btn primary" type="button" data-a="download">Download</button></div></div>
      <div class="nr-view nr-v-dl"><div class="nr-dl-row"><span class="nr-dl-l">Downloading Narrator</span><span class="nr-dl-n"></span></div><div class="nr-bar" role="progressbar" aria-label="Narrator download" aria-valuemin="0" aria-valuemax="100"><i></i></div></div>
      <div class="nr-view nr-v-missing"><p class="nr-lead">Narrator isn’t installed on this server.</p><p class="nr-sub">Serve the site with its on-device models:</p><code class="nr-code">node vlm/web/serve.mjs</code></div>
      <div class="nr-view nr-v-error"><p class="nr-lead">The Narrator couldn’t start.</p><p class="nr-sub nr-err"></p><div class="nr-btns"><button class="btn" type="button" data-a="retry">Try again</button></div></div>
    </div>
    <div class="nr-actions">
      <div class="seg nr-seg" role="group" aria-label="Ask the Narrator"><button type="button" data-task="describe">Describe</button><button type="button" data-task="safety">Is it safe?</button><button type="button" data-task="now">What now?</button></div>
      <form class="nr-ask" autocomplete="off"><input type="text" maxlength="120" placeholder="Ask about this view…" aria-label="Ask about this view" enterkeyhint="send"><button class="nr-send" type="submit" aria-label="Send" disabled>${SEND}</button></form>
    </div>
    <footer class="nr-foot"><span class="nr-dev"></span><span aria-hidden="true">·</span><span>Experimental on-device models. Not for real flight decisions.</span></footer>
  </div></div>
</section>`;
const UNIT = /^(km|m|kt|ft|s|min|o'clock|%|m\/s|cm\/s)[.,;:!?]*$/;
const OBSTACLES = ['#top .wordmark', '#top .tools', '#playbox', '#training', '.lnd-bar', '.lnd-wx', '.rl-chips', '.rl-read', '.rl-jets'];

const cfg = config(), engine = cfg.mock ? createMockEngine() : createEngine(cfg), hyst = createHysteresis(), labels = createHysteresis();
const sheet = matchMedia('(max-width: 640px)'), reduce = matchMedia('(prefers-reduced-motion: reduce)');
const S = { open: false, view: 'idle', family: null, heads: null, eye: 'idle', gen: false, task: null, dismissed: false, frame: null, last: null, pillKey: '' };
let el = null, feed = null, timer = 0, ticks = 0, genCtl = null, pending = null, rq = [], rt = 0, ending = false, committed = 0, collapseT = 0;
const place = { pill: { x: innerWidth / 2, y: 17 }, card: { x: innerWidth / 2, y: 17, w: 560 } };
const $ = (s) => el.root.querySelector(s);
const now = () => performance.now();

function build() {
  const link = document.createElement('link'); link.rel = 'stylesheet'; link.href = new URL('./card.css', import.meta.url).href;
  const css = new Promise((ok) => { link.onload = ok; link.onerror = ok; }); document.head.appendChild(link);
  const root = document.createElement('div'); root.id = 'nr'; root.className = 'nr-root'; root.innerHTML = HTML; document.body.appendChild(root);
  if (cfg.mock) for (const n of root.querySelectorAll('.nr-demo, .nr-demo-p')) n.hidden = false;
  el = { root };
  Object.assign(el, { pill: $('.nr-pill'), pillIn: $('.nr-pill-in'), pl: $('.nr-pl'), fam: $('.nr-fam'), card: $('.nr-card'), in: $('.nr-in'), thumb: $('.nr-thumb'), cv: $('.nr-thumb canvas'),
    chipI: $('.nr-chip-i'), chipT: $('.nr-chip-t'), x: $('.nr-x'), body: $('.nr-body'), text: $('.nr-text'), tw: $('.nr-tw'), tmpl: $('.nr-tmpl'), out: $('.nr-out'),
    seg: $('.nr-seg'), form: $('.nr-ask'), input: $('.nr-ask input'), send: $('.nr-send'), dev: $('.nr-dev'), dlL: $('.nr-dl-l'), dlN: $('.nr-dl-n'), bar: $('.nr-bar'), err: $('.nr-err'), grab: $('.nr-grab') });
  el.caret = document.createElement('span'); el.caret.className = 'nr-caret'; el.caret.setAttribute('aria-hidden', 'true');
  el.pill.addEventListener('click', () => expand());
  el.x.addEventListener('click', () => close());
  el.seg.addEventListener('click', (e) => { const b = e.target.closest('button[data-task]'); if (b && !b.disabled) ask(b.dataset.task); });
  el.input.addEventListener('input', () => { el.send.disabled = !el.input.value.trim(); });
  el.form.addEventListener('submit', (e) => { e.preventDefault(); const q = el.input.value.trim(); if (!q) return; el.input.value = ''; el.send.disabled = true; ask('ask', q); });
  el.card.addEventListener('click', (e) => { const a = e.target.closest('[data-a]'); if (a) action(a.dataset.a); });
  new ResizeObserver(() => { if (S.open) geom(cardGeom()); }).observe(el.in);
  el.text.addEventListener('scroll', edge);
  window.addEventListener('keydown', (e) => { if (S.open && e.key === 'Escape') { e.stopPropagation(); e.preventDefault(); collapse(); } }, true);
  document.addEventListener('pointerdown', (e) => { if (S.open && !el.root.contains(e.target) && !(e.target.closest && e.target.closest('#btnNarrator'))) collapse(); }, true);
  addEventListener('resize', () => layout());
  sheet.addEventListener('change', () => { if (S.open) { collapse(); } layout(); });
  dragToDismiss();
  return css;
}

// ---------- placement ----------
function obstacles(extra = []) {
  const out = [];
  for (const sel of [...OBSTACLES, ...extra]) for (const n of document.querySelectorAll(sel)) {
    const r = n.getBoundingClientRect(), cs = getComputedStyle(n);
    if (r.width > 0 && r.height > 0 && cs.visibility !== 'hidden' && !n.classList.contains('hidden')) out.push({ l: r.left, t: r.top, r: r.right, b: r.bottom });
  }
  return out;
}
// nothing while the Narrator is off; the pill's place while it is on; the card's only while the card is open
function layout() {
  if (!el || !life.on) return;
  const vw = innerWidth, vh = innerHeight, pw = pillWidth();
  const p = placeBox({ vw, vh, obstacles: obstacles(['#zoomHud']), h: 34, minW: pw, maxW: pw });
  place.pill = p; el.root.style.setProperty('--px', `${p.x}px`); el.root.style.setProperty('--py', `${p.y}px`);
  if (!S.open) return;
  const c = placeBox({ vw, vh, obstacles: obstacles(), h: Math.max(240, el.in.offsetHeight || 300), minW: 420, maxW: 560 });
  place.card = c; el.root.style.setProperty('--iw', `${c.w}px`);
  geom(cardGeom());
}
const pillWidth = () => Math.ceil(el.pillIn.offsetWidth) + 2;
const pillGeom = () => ({ x: place.pill.x, y: place.pill.y, w: pillWidth(), h: 34, r: 17 });
const cardGeom = () => ({ x: place.card.x, y: place.card.y, w: place.card.w, h: Math.min(el.in.offsetHeight, innerHeight - place.card.y - 16), r: 22 });
function geom(g) { const s = el.card.style; s.setProperty('--cx', `${g.x}px`); s.setProperty('--cy', `${g.y}px`); s.setProperty('--cw', `${g.w}px`); s.setProperty('--ch', `${g.h}px`); s.setProperty('--cr', `${g.r}px`); }

// ---------- the pill ----------
function zInfo() { const z = window.__ap && window.__ap.zoom; return { rangeKm: z && z.info ? z.info.rangeKm : NaN }; }
function chipText(f) {
  if (f === 'A') { const ap = window.__ap, env = ap && ap.env; return contextChip('A', { route: ap && ap.state && ap.state.route, sky: env ? skyName(env.weatherSeverity, env.weatherCover) : null }); }
  return contextChip(f, f === 'Z' ? zInfo() : {});
}
function renderPill() {
  const f = S.family, h = S.heads;
  const v = S.eye === 'missing' ? 'off' : f === 'Z' ? 'z' : h && h.status === 'ok' ? h.verdict.toLowerCase() : 'watch';
  const label = S.eye === 'missing' ? 'Not installed' : S.eye === 'error' ? 'Pilot Eye paused' : f === 'Z' ? chipText('Z') : pillLabel(h, f);
  const key = `${v}|${label}|${f}`;
  if (key === S.pillKey) return;
  const changed = S.pillKey.split('|')[1] !== label; S.pillKey = key;
  el.pill.dataset.v = v; el.fam.innerHTML = ICON[f] || '';
  const [head, ...rest] = label.split(' · '), b = document.createElement('b'); b.textContent = head; el.pl.replaceChildren(b);
  if (rest.length) { const s = document.createElement('span'); s.textContent = ` · ${rest.join(' · ')}`; el.pl.append(s); }
  el.pill.setAttribute('aria-label', `Narrator: ${label}. Show the Narrator`);
  if (changed && !reduce.matches && S.pillKey) el.pl.animate([{ opacity: 0, filter: 'blur(2px)' }, { opacity: 1, filter: 'blur(0)' }], { duration: 220, easing: 'ease-out' });
  el.pill.style.setProperty('--pw', `${pillWidth()}px`);
  if (!S.open) layout();
}
function renderChrome() {
  const f = S.family;
  el.chipI.innerHTML = ICON[f] || ''; el.chipT.textContent = chipText(f);
  for (const b of el.seg.querySelectorAll('button')) {
    const off = S.eye === 'missing' || (f === 'Z' && b.dataset.task !== 'describe');
    b.disabled = off; b.title = f === 'Z' && b.dataset.task !== 'describe' ? 'Available while flying' : '';
    b.classList.toggle('on', b.dataset.task === S.task); b.setAttribute('aria-pressed', String(b.dataset.task === S.task));
  }
  const dev = engine.ready ? engine.device : hasWebGPU() ? 'webgpu' : 'wasm';
  el.dev.textContent = dev === 'webgpu' ? 'On device · WebGPU' : 'On device · WASM · slower';
  if (S.view === 'idle') el.tmpl.textContent = template();
}
const template = () => (S.heads && S.heads.sentence ? humanTemplate(S.heads.sentence) : S.eye === 'error' ? 'Pilot Eye is paused.' : 'Watching the view…');

// ---------- Pilot Eye ----------
function onEye(m) {
  if (!life.on) return;
  if (m.status === 'ready') { S.eye = 'warming'; renderPill(); return; }
  if (m.status === 'error') { S.eye = 'error'; console.warn('pilot eye:', m.message); renderPill(); renderChrome(); return; }
  if (m.status !== 'ok') return;
  S.eye = 'ok';
  if (S.family === 'Z') S.heads = { ...m, at: now() };
  else {
    // the verdict holds by hysteresis, and so does the pill's detail (the reason or action): a new wording shows once it
    // comes twice in a row, so the pill does not flicker between "Caution · rock ahead" and "Caution · climb"
    const before = hyst.value, shown = hyst.push(m.verdict);
    if (shown !== before || !S.heads) { S.heads = { ...m, verdict: shown, at: now() }; labels.reset(); labels.push(pillLabel(S.heads, S.family)); }
    else if (m.verdict === shown) { const l = pillLabel(m, S.family); if (labels.push(l) === l) S.heads = { ...m, at: now() }; else S.heads.at = now(); }
    else S.heads.at = now();
  }
  renderPill(); if (S.view === 'idle') el.tmpl.textContent = template();
}
function tick() {
  const f = familyOf() || S.family;
  if (f !== S.family) { S.family = f; hyst.reset(); labels.reset(); S.heads = null; S.pillKey = ''; renderChrome(); }
  if (S.heads && now() - S.heads.at > 6000) { S.heads = null; hyst.reset(); labels.reset(); renderChrome(); }
  renderPill();
  if (++ticks % 4 === 0) { el.chipT.textContent = chipText(S.family); layout(); }
}

// ---------- open / close / morph ----------
// open and close go through life.js: a close or a reopen during the build, the grab install or the probe stops the
// stale open where it stands; the Narrator model is released after 5 minutes closed
const life = createLifecycle({
  build,
  show() {
    S.family = familyOf(); S.eye = 'warming'; S.pillKey = '';
    document.body.classList.add('nr-on'); headerButton(true);
    setView(engine.ready || memory.get().consent || S.dismissed ? 'idle' : 'consent');
    renderChrome(); renderPill();
    el.root.classList.add('on'); layout();
    expand();
    feed ||= createFeed({ onFrame: (f) => engine.postFrame(f), nominal: () => engine.nominal(), accepting: () => engine.accepting() });
  },
  install: async () => { await feed.install(); refresh(); },
  startTick: () => { clearInterval(timer); timer = setInterval(tick, 250); },
  probe: () => engine.probe(),
  ready(pr) { if (!pr.ok) { S.eye = 'missing'; setView('missing'); renderPill(); renderChrome(); return; } engine.startEye(onEye); },
  expand: () => expand(),
  teardown,
  idle: () => engine.release(),
});
export const open = () => life.open();
export const close = () => life.close();
function teardown() {
  const inside = el.root.contains(document.activeElement);
  cancel(); pending = null;
  collapse(true);
  clearInterval(timer); timer = 0; if (feed) feed.uninstall(); engine.stopEye(); hyst.reset(); labels.reset(); S.heads = null; S.eye = 'idle';
  el.root.classList.remove('on'); document.body.classList.remove('nr-on'); headerButton(false);
  if (S.frame) { S.frame.close(); S.frame = null; }
  const back = [document.getElementById('btnNarrator'), document.getElementById('nr-launch')].find((n) => n && n.offsetParent);
  if (inside && back) back.focus({ preventScroll: true });
}
export const toggle = () => (life.on ? close() : open());
function headerButton(on) { const b = document.getElementById('btnNarrator'); if (b) { b.classList.toggle('on', on); b.setAttribute('aria-pressed', String(on)); } }
export function expand() {
  if (!life.on || S.open || !el) return;
  S.open = true; clearTimeout(collapseT); layout();
  const c = el.card, morph = !sheet.matches && !reduce.matches;
  if (morph && !c.classList.contains('show')) { c.classList.add('snap'); geom(pillGeom()); c.classList.add('show'); void c.offsetWidth; c.classList.remove('snap'); }
  c.classList.add('show', 'open'); if (!sheet.matches) geom(cardGeom());
  el.pill.classList.toggle('away', !sheet.matches); el.pill.setAttribute('aria-expanded', 'true');
  document.body.classList.add('nr-card-open');
  requestAnimationFrame(() => focusIn());
  if (S.view === 'idle') refresh();
}
export function collapse(off = false) {
  if (!S.open) return;
  const inside = el.root.contains(document.activeElement);
  S.open = false; el.card.classList.remove('open'); document.body.classList.remove('nr-card-open');
  layout(); if (!sheet.matches) geom(pillGeom());
  el.pill.setAttribute('aria-expanded', 'false');
  clearTimeout(collapseT);
  collapseT = setTimeout(() => { if (!S.open) { el.card.classList.remove('show'); el.pill.classList.remove('away'); } }, reduce.matches ? 0 : sheet.matches ? 360 : 280);
  if (inside && !off) el.pill.focus({ preventScroll: true });
}
function focusIn() {
  if (!S.open) return;
  const t = { consent: '[data-a="download"]', error: '[data-a="retry"]', missing: '.nr-x', dl: null }[S.view];
  const n = t === undefined ? el.seg.querySelector('button:not(:disabled)') : t ? $(t) : null;
  (n || el.card).focus({ preventScroll: true });
}
function dragToDismiss() {
  let y0 = null, dy = 0;
  el.grab.addEventListener('pointerdown', (e) => { if (!sheet.matches) return; y0 = e.clientY; dy = 0; el.grab.setPointerCapture(e.pointerId); el.card.classList.add('drag'); });
  el.grab.addEventListener('pointermove', (e) => { if (y0 === null) return; dy = Math.max(0, e.clientY - y0); el.card.style.setProperty('--dy', `${dy}px`); });
  const end = () => { if (y0 === null) return; y0 = null; el.card.classList.remove('drag'); el.card.style.setProperty('--dy', '0px'); if (dy > 60) collapse(); };
  el.grab.addEventListener('pointerup', end); el.grab.addEventListener('pointercancel', end);
}

// ---------- views ----------
function setView(v, msg = '') {
  S.view = v; el.card.dataset.view = v === 'answer' ? 'idle' : v;
  if (v === 'error') el.err.textContent = msg;
  if (v === 'idle') { el.tw.classList.remove('gone'); el.tmpl.textContent = template(); el.out.replaceChildren(); S.task = null; }
  if (S.open && v !== 'answer') requestAnimationFrame(() => focusIn());
  renderChrome();
}
function action(a) {
  if (a === 'later') { S.dismissed = true; pending = null; setView('idle'); }
  else if (a === 'download') download();
  else if (a === 'retry') { setView('idle'); if (pending) ask(pending.kind, pending.question); }
}
function progress({ loaded, total }) {
  const p = total > 0 ? Math.min(1, loaded / total) : 0;
  // every byte is in: the models now compile for the GPU, which can take a while with nothing left to count
  const done = p >= 0.999; el.bar.classList.toggle('busy', done);
  if (done) { el.dlL.textContent = 'Preparing Narrator'; el.dlN.textContent = 'on this device'; }
  else el.dlN.textContent = `${Math.round(loaded / 1e6)} of ${Math.round(total / 1e6)} MB`;
  el.bar.style.setProperty('--p', p.toFixed(4)); el.bar.setAttribute('aria-valuenow', String(Math.round(p * 100)));
}
async function download() {
  memory.set({ consent: true });
  el.dlL.textContent = memory.get().downloaded ? 'Loading Narrator' : 'Downloading Narrator';
  progress({ loaded: 0, total: DOWNLOAD_MB * 1e6 }); setView('dl');
  const live = life.guard();
  try {
    await engine.loadNarrator((p) => { if (live()) progress(p); });
    memory.set({ downloaded: true });
    if (!live()) return;
    const p = pending || { kind: 'describe' }; pending = null; renderChrome(); run(p.kind, p.question);
  } catch (e) { console.error('narrator:', e); if (live()) setView('error', String(e && e.message ? e.message : e).slice(0, 160)); }
}

// ---------- asking ----------
async function refresh() {
  if (!feed || !feed.installed) return null;
  const live = life.guard();
  try {
    const b = await feed.snapshot();
    if (!live()) { if (b !== S.frame) b.close(); return null; }
    if (S.frame && S.frame !== b) S.frame.close();
    S.frame = b; el.cv.getContext('2d').drawImage(b, 0, 0, el.cv.width, el.cv.height); el.thumb.classList.add('has');
    return b;
  } catch { return S.frame; }
}
export function ask(kind, question = null) {
  if (!life.on || S.eye === 'missing') return;
  const family = familyOf() || S.family;
  if (family === 'Z' && (kind === 'safety' || kind === 'now')) return;
  S.task = kind; renderChrome();
  if (!engine.ready) { pending = { kind, question }; if (!memory.get().consent) setView('consent'); else if (S.view !== 'dl') download(); return; }
  run(kind, question, family);
}
function cancel() { if (genCtl) genCtl.abort(); genCtl = null; clearInterval(rt); rt = 0; rq = []; ending = false; setGen(false); }
function setGen(on) {
  S.gen = on; el.card.classList.toggle('gen', on); el.body.setAttribute('aria-busy', String(on));
  if (!on) { el.thumb.classList.remove('scan'); el.caret.remove(); }
}
async function run(kind, question = null, family = familyOf() || S.family) {
  cancel();
  const ctl = new AbortController(); genCtl = ctl; S.task = kind;
  setView('answer'); committed = 0; el.tw.classList.remove('gone'); el.tmpl.textContent = template(); el.tmpl.append(el.caret); el.out.replaceChildren();
  setGen(true); el.thumb.classList.add('scan');
  const k = question ? 'ask' : kind, gate = createSentenceGate({ max: CAPS[k], dropEcho: kind === 'safety' }), heads = family === 'Z' ? null : S.heads;
  const frame = await refresh();
  if (genCtl !== ctl) return;
  if (!frame) { setGen(false); setView('error', 'No frame to look at: is the view paused in a hidden tab?'); return; }
  const t0 = now(), stream = []; let ttft = null;
  try {
    const r = await engine.describe(await createImageBitmap(frame), { family, heads, facts: factsOf(family), task: question || TASKS[kind], kind: k, maxNewTokens: TOKENS[k], signal: ctl.signal,
      onToken: (t) => { if (ttft === null && t.trim()) { ttft = now() - t0; el.thumb.classList.remove('scan'); } stream.push([Math.round(now() - t0), t]); const g = gate.push(t); enqueue(g.add); if (g.stop) ctl.abort(); } });
    if (genCtl !== ctl) return;
    enqueue(gate.finish().add);
    S.last = { kind: k, family, task: question || TASKS[kind], ttftMs: ttft, ms: r.ms, raw: r.text, text: gate.text, stop: gate.reason, device: engine.device, loadMs: engine.loadMs, stream };
  } catch (e) {
    if (genCtl !== ctl || (e && e.name === 'AbortError')) return;
    console.error('narrator:', e); setGen(false); setView('error', String(e && e.message ? e.message : e).slice(0, 160)); return;
  }
  ending = true; if (!rt) finish();
}
function enqueue(sentences) {
  for (const s of sentences) {
    if (!committed++) el.tw.classList.add('gone');
    const ws = s.split(/\s+/).filter(Boolean);
    for (let i = 0; i < ws.length; i++) rq.push(UNIT.test(ws[i + 1] || '') && /^[\d.,]+$/.test(ws[i]) ? `${ws[i]}\u00a0${ws[++i]}` : ws[i]);
  }
  if (rq.length && !rt) rt = setInterval(step, 34);
}
function step() {
  if (el.caret.parentNode !== el.out && S.gen) el.out.append(el.caret);
  for (let i = rq.length > 14 ? 2 : 1; i > 0 && rq.length; i--) {
    if (el.out.querySelector('.nr-w')) el.out.insertBefore(document.createTextNode(' '), el.caret.parentNode === el.out ? el.caret : null);
    const w = document.createElement('span'); w.className = 'nr-w'; w.textContent = rq.shift(); el.out.insertBefore(w, el.caret.parentNode === el.out ? el.caret : null);
  }
  el.text.scrollTop = el.text.scrollHeight; edge();
  if (!rq.length) { clearInterval(rt); rt = 0; if (ending) finish(); }
}
function finish() { ending = false; genCtl = null; setGen(false); if (!committed) el.tw.classList.remove('gone'); edge(); }
function edge() { const t = el.text; t.classList.toggle('more', t.scrollHeight - t.scrollTop - t.clientHeight > 2); }

// visual QA and the harness: window.__narrator.card.debug
export const debug = {
  get state() { return { on: life.on, open: S.open, view: S.view, family: S.family, eye: S.eye, heads: S.heads, gen: S.gen, task: S.task, last: S.last, pill: S.pillKey, text: el ? el.out.textContent : '', device: engine.device, errors: feed ? feed.errors : 0 }; },
  get mock() { return engine.mock || null; }, engine, expand, collapse, ask, setView, layout, refresh,
};
