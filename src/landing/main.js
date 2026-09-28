// The landing scenario's page: index.html boots it for ?scenario=landing (the corridor game, src/app.js, is not loaded).
// The autoland flies in real time (120 Hz physics, any frame rate) while the scene and the HUD follow; the report card
// ends each landing. URL: wind=23015G25  turb=none|light|moderate|severe  vis=cavok|haze|fog  time=day|dusk|night
// view=cinema|chase|cockpit|tower|runway|approach  seed=N  start=final|random  loop=1  speed=1…8  path=0
// clouds=FEW|SCT|BKN|OVC|NONE  ceiling=<ft> (default by vis: CAVOK FEW065, haze SCT030, fog OVC003)  rain=1 (a wet runway)
import { createLandingSim, drawConditions } from './sim.js';
import { createLandingScene } from './scene.js';
import { createHud } from './hud.js';
import { parseMetarWind } from './wind.js';
import { createLandingAudio } from './audio.js';
import { COVER } from './clouds.js';

const DECKS = { cavok: ['FEW', 6500, 400], haze: ['SCT', 3000, 350], fog: ['OVC', 300, 520] };   // each visibility's usual sky: layer, base (ft), thickness (m)
const H = 1 / 120, VIEWS = ['cinema', 'chase', 'cockpit', 'tower', 'runway', 'approach'];
const pick = (v, list, d) => (list.includes(v) ? v : d);
const WORD = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'niner'], say = (n, len = 1) => String(Math.round(n)).padStart(len, '0').split('').map((c) => WORD[+c]).join(' ');
function clearance(w) {                                   // ICAO phraseology: digits one by one, the wind from the tower's anemometer
  const calm = w.kt < 3, wt = calm ? 'WIND CALM' : `WIND ${String(w.dir).padStart(3, '0')}° ${w.kt} KT${w.gust ? ' GUSTING ' + w.gust : ''}`;
  const ws = calm ? 'wind calm' : `wind ${say(w.dir, 3)} degrees, ${say(w.kt)} knots${w.gust ? ', gusting ' + say(w.gust) : ''}`;
  return [`<b>APX TOWER</b> ASTRO ONE, RUNWAY 26, ${wt}, CLEARED TO LAND`, `Astro one, runway two six, ${ws}, cleared to land.`, 'Cleared to land runway two six, Astro one.'];
}

export async function start() {
  document.body.classList.add('landing'); document.title = 'Astro Pilot — autoland, runway 26';
  const css = document.createElement('style'); css.textContent = 'body.landing > *:not(#view):not(.lnd) { display: none !important; } body.landing { background: #000; overflow: hidden; } body.landing #view { position: fixed; inset: 0; width: 100vw; height: 100vh; }'; document.head.appendChild(css);
  const qs = new URLSearchParams(location.search);
  const time = pick(qs.get('time'), ['day', 'dusk', 'night'], 'day'), vis = pick(qs.get('vis'), ['cavok', 'haze', 'fog'], 'cavok'), startMode = qs.get('start') === 'random' ? 'random' : 'final';
  const metar = parseMetarWind(qs.get('wind')), turb = pick(qs.get('turb'), ['none', 'light', 'moderate', 'severe'], null), loop = qs.get('loop') === '1';
  const cq = (qs.get('clouds') || '').toUpperCase(), d0 = DECKS[vis], layer = cq === 'NONE' || cq === 'SKC' ? null : COVER[cq] !== undefined ? cq : d0[0], baseFt = +qs.get('ceiling') || d0[1];
  const clouds = layer ? { cover: COVER[layer], base: baseFt * 0.3048, top: baseFt * 0.3048 + d0[2], metar: layer + String(Math.round(baseFt / 100)).padStart(3, '0') } : null;
  const rain = qs.get('rain') === '1';
  let seed = +qs.get('seed') || (Date.now() % 100000), speed = Math.min(8, Math.max(0.25, +qs.get('speed') || 1)), view = pick(qs.get('view'), VIEWS, 'cinema');
  const scene = createLandingScene(document.getElementById('view'), { time, vis, clouds, rain }), root = document.createElement('div'); root.className = 'lnd'; document.body.appendChild(root);
  let sim = null, acc = 0, t = 0, doneAt = null, vacateSaid = false; const audio = createLandingAudio();
  function conditions(s) {
    const c = drawConditions(s, { final: startMode === 'final' });
    if (metar) c.wind = { dir: metar.dir ?? c.wind.dir, kt: metar.kt, gust: metar.gust, turb: turb ?? c.wind.turb }; else if (turb) c.wind.turb = turb;
    c.clouds = (rain ? '-RA ' : '') + (clouds ? clouds.metar : 'NSC'); return c;
  }
  const hud = createHud(root, {
    view: (v) => { view = v; scene.setView(v); },
    path: (b) => { const on = !b.classList.contains('on'); b.classList.toggle('on', on); scene.showPath(on); },
    voice: (b) => { const on = !b.classList.contains('on'); b.classList.toggle('on', on); hud.setVoice(on); if (on) audio.start(); else audio.stop(); },   // callouts and sound
    speed: (b) => { speed = speed >= 8 ? 1 : speed * 2; b.textContent = '×' + speed; },
    new: () => begin(seed + 1), again: () => begin(seed, sim.rep.cond), back: () => { location.search = ''; },
  });
  function begin(s, cond = conditions(s)) { seed = s; sim = createLandingSim(cond); scene.setPath(sim.gnc, cond.start.h); hud.reset(); doneAt = null; acc = 0; vacateSaid = false; hud.atc(...clearance(cond.wind)); }
  const showPath = qs.get('path') !== '0'; scene.showPath(showPath); root.querySelector('button[data-a="path"]').classList.toggle('on', showPath); root.querySelector('button[data-a="speed"]').textContent = '×' + speed;
  scene.setView(view); const resize = () => scene.resize(innerWidth, innerHeight); addEventListener('resize', resize); resize();
  begin(seed);
  let last = performance.now();
  function frame(now) {
    const dt = Math.min(0.1, (now - last) / 1000); last = now; t += dt;
    acc += dt * speed; let n = 0; while (acc >= H && n < 2400) { sim.step(); acc -= H; n++; }
    scene.frame(sim, dt, t); hud.update(sim, scene.camera, scene.view); audio.update(sim);
    const ex = sim.gnc.st.exit; if (ex && !vacateSaid) { vacateSaid = true; const n = ex.name.slice(1);   // the tower, as the rollout slows for the exit
      hud.atc(`<b>APX TOWER</b> ASTRO ONE, VACATE RIGHT VIA ${ex.name}, CONTACT GROUND 121.9`, `Astro one, vacate right via alpha ${say(n)}, contact ground one two one decimal niner.`, `Right via alpha ${say(n)}, ground one two one niner, Astro one.`); }
    if (sim.rep.done && doneAt === null) { doneAt = t; hud.report(sim.rep); }
    if (loop && doneAt !== null && t - doneAt > 8) begin(seed + 1);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  window.__landing = { get sim() { return sim; }, scene, ready: true, begin, conditions, setView: (v) => { view = v; scene.setView(v); }, advance(sec) { for (let i = 0; i < sec * 120 && !sim.rep.done; i++) sim.step(); } };
}
