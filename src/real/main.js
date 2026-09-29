// The real-spacecraft page: index.html boots it for ?scenario=real or ?real=1 (the game, src/app.js, is not loaded).
// URL: start=far|near|final (r 2 km / rho 217 m / rho 20 m)  nav=noisy|truth  warp=auto|1|5|20|60  seed=N  headless=1
// filter=0 (safety filter off, for teaching). debris and inject are S2: any other value than 0 is ignored with a HUD
// note. warp auto = clamp(r / 3 m, 2, 60), capped at 5x during burns; the backdrop's sky clock runs at the same warp.
import { createRealSim } from './sim.js';
import { createRealScene, VIEWS } from './scene.js';
import { createHud } from './hud.js';
import { CYCLE } from './consts.js';

export const WARPS = Object.freeze(['auto', '1', '5', '20', '60']);
export const autoWarp = (r) => Math.min(60, Math.max(2, r / 3));
const pick = (v, list, d) => (list.includes(v) ? v : d);

export async function start() {
  document.body.classList.add('real');
  document.title = 'Astro Pilot — real spacecraft docking';
  const css = document.createElement('style');
  css.textContent = 'body.real > *:not(#view):not(.rl) { display: none !important; } body.real { background: #000; overflow: hidden; } body.real #view { position: fixed; inset: 0; width: 100vw; height: 100vh; }';
  document.head.appendChild(css);
  const qs = new URLSearchParams(location.search);
  const startMode = pick(qs.get('start'), ['far', 'near', 'final'], 'far'), nav = pick(qs.get('nav'), ['noisy', 'truth'], 'noisy'), filter = qs.get('filter') !== '0';
  let warpSel = pick(qs.get('warp'), WARPS, 'auto'), seed = qs.has('seed') && Number.isFinite(+qs.get('seed')) ? +qs.get('seed') : Date.now() % 100000;
  const ignored = ['debris', 'inject', 'pilot'].filter((k) => qs.has(k) && qs.get(k) !== '0' && !(k === 'pilot' && qs.get(k) === 'gnc'));
  const make = (s) => createRealSim({ seed: s, start: startMode, nav, filter });
  let sim = make(seed);
  if (qs.get('headless') === '1') {
    const t0 = performance.now();
    while (!sim.rep.done) { sim.run(600); await new Promise((r) => setTimeout(r, 0)); }
    window.__real = { get sim() { return sim; }, rep: sim.rep, wallMs: performance.now() - t0, ready: true, headless: true };
    return;
  }
  const scene = createRealScene(document.getElementById('view'), { seed });
  let paused = true, reported = false, acc = 0, warp = 1, last = performance.now();
  const perf = { simMs: [], frameMs: [] };
  const back = () => { location.search = ''; };
  const restart = (s = seed + 1) => { seed = s; sim = make(seed); reported = false; acc = 0; hud.closeCard(); paused = false; };
  const hud = createHud(document.body, {
    view: () => scene.setView(VIEWS[(VIEWS.indexOf(scene.view) + 1) % VIEWS.length]),
    warp: () => (warpSel = WARPS[(WARPS.indexOf(warpSel) + 1) % WARPS.length]),
    restart: () => restart(), back,
  });
  if (ignored.length) hud.note(`${ignored.join(', ')}: S2 (ignored)`);
  const resize = () => scene.resize(innerWidth, innerHeight);
  addEventListener('resize', resize); resize();
  hud.intro(() => { paused = false; });
  function frame(now) {
    const dt = Math.min(0.1, (now - last) / 1000); last = now;
    const r = Math.hypot(sim.x[0], sim.x[1], sim.x[2]);
    warp = warpSel === 'auto' ? autoWarp(r) : +warpSel;
    if (sim.cmd.burn) warp = Math.min(warp, 5);
    scene.setWarp(warp);
    const s0 = performance.now();
    if (!paused && !sim.rep.done) { acc += dt * warp; let n = 0; while (acc >= CYCLE && n < 600 && !sim.rep.done) { sim.step(); acc -= CYCLE; n++; } }
    const s1 = performance.now();
    scene.frame(sim, dt, sim.t);
    hud.update(sim, { warp, sel: warpSel });
    if (sim.rep.done && !reported) { reported = true; hud.report(sim); }
    perf.simMs.push(s1 - s0); perf.frameMs.push(performance.now() - now);
    if (perf.simMs.length > 600) { perf.simMs.shift(); perf.frameMs.shift(); }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  window.__real = {
    get sim() { return sim; }, scene, hud, perf, ready: true,
    begin() { hud.closeCard(); paused = false; },
    setView: (v) => scene.setView(v), setWarp: (w) => { warpSel = pick(String(w), WARPS, 'auto'); },
    restart, advance(sec) { sim.run(sec); },
  };
}
