// boot.js — the Narrator's only page-load cost: the header button (index.html #btnNarrator), a top-centre launcher for the
// landing and docking pages (their CSS hides the header) and for phones (no room left in the header), the N key (skipped while typing, with modifiers or on repeat)
// and a lazy import of card.js on first use. It patches nothing, loads no model and runs nothing per frame; ?vlm=0
// turns it off. window.__narrator is the handle (open, close, toggle; the card adds its state once loaded).
export const SPARKLE = '<svg viewBox="0 0 16 16" aria-hidden="true" focusable="false"><path fill="currentColor" d="M8 0.6c.5 3.9 3 6.4 6.9 6.9v1c-3.9.5-6.4 3-6.9 6.9h-1C6.5 11.5 4 9 .1 8.5v-1C4 7 6.5 4.5 7 .6z"/></svg>';
const CSS = `#btnNarrator { display: inline-flex; align-items: center; gap: 6px; } #btnNarrator svg, .nr-launch svg { width: 12px; height: 12px; flex: none; }
  .nr-launch { display: none; }
  html body.landing > button#nr-launch.nr-launch, html body.real > button#nr-launch.nr-launch { display: inline-flex !important; }
  button.nr-launch { position: fixed; top: 17px; left: 50%; transform: translateX(-50%); z-index: 11; height: 34px; padding: 0 15px 0 13px; align-items: center; gap: 7px;
    border-radius: 17px; font: 500 13px var(--font); color: var(--text); cursor: pointer; transition: opacity .2s, transform .12s; }
  button.nr-launch:active { transform: translateX(-50%) scale(.97); } body.nr-on button.nr-launch { opacity: 0; pointer-events: none; }
  @media (max-width: 640px) { #btnNarrator { display: none; } html body > button#nr-launch.nr-launch { display: inline-flex; }
    button.nr-launch { top: 90px; } body.zooming button.nr-launch { top: 120px; } body.landing button.nr-launch, body.real button.nr-launch { top: 17px; } }
  body.nohud button.nr-launch { display: none !important; }`;
const typing = (e) => !!(e.target && /INPUT|TEXTAREA|SELECT/.test(e.target.tagName || '')) || !!(e.target && e.target.isContentEditable);

export function installBoot({ win = window, doc = document, load = () => import('./card.js') } = {}) {
  if (new URLSearchParams(win.location.search).get('vlm') === '0') return null;
  let mod = null, loading = null, last = Promise.resolve();
  const get = () => (loading ||= load().then((m) => (mod = m)));
  const run = (f) => (last = get().then(f).catch((e) => { loading = mod ? loading : null; console.error('narrator:', e); }));
  const api = {
    toggle: () => run((m) => m.toggle()), open: () => run((m) => m.open()), close: () => run((m) => m.close()),
    get loaded() { return !!mod; }, get ready() { return last; }, get card() { return mod; },
  };
  const style = doc.createElement('style'); style.textContent = CSS; doc.head.appendChild(style);
  const btn = doc.getElementById('btnNarrator');
  if (btn) btn.addEventListener('click', () => api.toggle());
  const launch = doc.createElement('button');
  launch.id = 'nr-launch'; launch.className = 'nr-launch glass'; launch.type = 'button';
  launch.setAttribute('aria-label', 'Open the Narrator'); launch.setAttribute('aria-keyshortcuts', 'N');
  launch.innerHTML = `${SPARKLE}<span>Narrator</span>`;
  launch.addEventListener('click', () => api.open());
  doc.body.appendChild(launch);
  win.addEventListener('keydown', (e) => {
    if (typing(e) || e.metaKey || e.ctrlKey || e.altKey || e.repeat || (e.key || '').toLowerCase() !== 'n') return;
    e.preventDefault(); api.toggle();
  });
  win.__narrator = api;
  return api;
}
if (typeof window !== 'undefined' && typeof document !== 'undefined' && document.body) installBoot();
