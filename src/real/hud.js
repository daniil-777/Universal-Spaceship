// The real-spacecraft HUD (spec section 2, S1 subset): phase chips, r / rho / rho-dot / dv used of 17.4 m/s, the
// 22-jet map, the IDSS card at contact, an LVLH inset (KOS, corridor, holds, the planned path and the "if every jet
// failed now" 24 h drift, green only when it is passively safe) and the intro card (B3). S2 adds the conjunction chip
// with NEW TRACK tags and the TCA markers.
import { JETS, KOS_R, DRAG_B, DAY, T_ORB, H1_POINT, CORRIDOR_HALF, STATION_PORT, MASS0, PROP0, KIND, G0, IDSS, DEG, portRel, rhoFromR } from './consts.js';
import { propagate } from './cw.js';
import { passiveMin } from './passive.js';
import { PH } from './guidance.js';

const DV_TOTAL = KIND.P.isp * G0 * Math.log(MASS0 / (MASS0 - PROP0));
const CHIPS = [PH.TRANSFER, PH.H1ACQ, PH.H1, PH.CORRIDOR, PH.H2, PH.FINAL, 'CAPTURE', PH.BREAKOUT];
const CSS = `
  .rl { position: fixed; z-index: 20; font: 500 12px/1.35 ui-monospace, "SF Mono", Menlo, monospace; color: #e8eef6; background: rgba(8, 12, 20, 0.62); backdrop-filter: blur(6px); border: 1px solid rgba(255,255,255,0.12); border-radius: 12px; }
  .rl-chips { top: 14px; left: 16px; padding: 6px; display: flex; flex-wrap: wrap; gap: 4px; max-width: calc(100vw - 32px); }
  .rl-chips span { padding: 3px 8px; border-radius: 8px; opacity: 0.45; } .rl-chips span.on { opacity: 1; background: rgba(125,200,255,0.25); color: #fff; }
  .rl-read { top: 58px; left: 16px; padding: 10px 14px; min-width: 220px; } .rl-read b { color: #9fd0ff; font-weight: 600; }
  .rl-jets { top: 58px; right: 16px; padding: 8px; display: grid; grid-template-columns: repeat(8, 26px); gap: 3px; }
  .rl-jets i { font-style: normal; font-size: 9px; text-align: center; padding: 3px 0; border-radius: 4px; background: rgba(255,255,255,0.06); }
  .rl-jets i.on { background: rgba(160,215,255,0.85); color: #04121f; }
  .rl-bar { bottom: 16px; right: 16px; padding: 6px; display: flex; flex-wrap: wrap; gap: 6px; max-width: calc(100vw - 32px); }
  .rl-bar button, .rl-card button { font: 600 12px system-ui, sans-serif; color: #fff; background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.2); border-radius: 8px; padding: 6px 10px; cursor: pointer; }
  .rl-bar button.on { background: rgba(125,200,255,0.28); border-color: #9fd0ff; }
  .rl-inset { left: 16px; bottom: 16px; padding: 6px; } .rl-inset canvas { display: block; width: 240px; height: 240px; } .rl-inset p { margin: 4px 2px 0; font-size: 10px; opacity: 0.7; max-width: 240px; }
  .rl-card { left: 50%; top: 50%; transform: translate(-50%, -50%); padding: 18px 22px; width: min(460px, calc(100vw - 32px)); font: 400 14px/1.45 system-ui, sans-serif; }
  .rl-card h3 { margin: 0 0 8px; font: 700 18px/1.2 system-ui, sans-serif; } .rl-card td { padding: 2px 10px 2px 0; } .rl-card .ok { color: #7dffae; } .rl-card .no { color: #ff8a7a; }
  @media (max-width: 700px) { .rl-jets { display: none; } .rl-inset canvas { width: 160px; height: 160px; } }`;
// every value enters the DOM through textContent; only the fixed literal markup of INTRO and READ goes through innerHTML
const el = (tag, cls, parent, text = '') => { const e = document.createElement(tag); if (cls) e.className = cls; e.textContent = text; parent.appendChild(e); return e; };
const INTRO = '<h3>Real spacecraft</h3><p>Real reaction-control jets give a Shuttle-class ship about <b>0.08 m/s&sup2;</b>. Dodging the asteroid belt of the game would need about <b>35 g</b> with 1.5-3.6 s of warning, so real ships do not dodge: they rendezvous. Here a classical guidance, navigation and control autopilot flies a 95 t orbiter from 2 km behind a Mir-class station to a soft dock, as NASA plans it: Clohessy-Wiltshire relative motion with a drag bound, 22 jets with minimum impulse bits, a 600 kg propellant budget, and passive safety after every burn.</p>';
// the readouts' fixed markup: update() writes each data-k slot's textContent; the note row shows only when a note is set
const READ = '<b data-k="ph"></b> &middot; t <span data-k="t"></span> min<br>r <span data-k="r"></span> m &middot; rho <span data-k="rho"></span> m (<span data-k="rhoV"></span> on V-bar)<br>rho-dot <span data-k="rhoDot"></span> cm/s<br>dv <span data-k="dv"></span> / <span data-k="dvTotal"></span> m/s &middot; prop <span data-k="prop"></span> kg<br>warp <span data-k="warp"></span>x &middot; <span data-k="mode"></span> attitude<span data-k="noteRow" hidden><br><span data-k="note"></span></span>';

export function createHud(root, on) {
  el('style', '', document.head).textContent = CSS;
  const chips = el('div', 'rl rl-chips', root), read = el('div', 'rl rl-read', root), jets = el('div', 'rl rl-jets', root);
  const inset = el('div', 'rl rl-inset', root), bar = el('div', 'rl rl-bar', root);
  read.innerHTML = READ;
  const slot = Object.fromEntries([...read.querySelectorAll('[data-k]')].map((e) => [e.dataset.k, e]));
  slot.dvTotal.textContent = DV_TOTAL.toFixed(1);
  const cv = el('canvas', '', inset); cv.width = cv.height = 480; const g = cv.getContext('2d');
  el('p', '', inset, 'LVLH, X along-track, Y up; propagation is real (Clohessy-Wiltshire, drag bound 2e-7 m/s²)');
  const chipEl = CHIPS.map((c) => el('span', '', chips, c)), jetEl = JETS.map((j) => el('i', '', jets, j.name));
  const btn = (label, fn) => { const b = el('button', '', bar, label); b.addEventListener('click', () => fn(b)); return b; };
  btn('view: chase', (b) => { b.textContent = 'view: ' + on.view(); });
  btn('warp: auto', (b) => { b.textContent = 'warp: ' + on.warp(); });
  btn('new run', () => on.restart());
  btn('Back to the game', () => on.back());
  let card = null, drift = null, driftT = -1, note = '';
  const closeCard = () => { if (card) { card.remove(); card = null; } };
  // fill(card) builds the card's content; the button row follows it
  function showCard(fill, buttons) {
    closeCard(); card = el('div', 'rl rl-card', root); fill(card);
    const row = el('div', '', card); row.style.marginTop = '12px';
    for (const [label, fn] of buttons) { const b = el('button', '', row, label); b.style.marginRight = '8px'; b.addEventListener('click', fn); }
  }
  function drawInset(sim) {
    const x = sim.x, W = cv.width, r = Math.hypot(x[0], x[1], x[2]), half = Math.min(3000, Math.max(300, 1.4 * r));
    const P = (X, Y) => [W / 2 + (X / half) * (W / 2), W / 2 - (Y / half) * (W / 2)];
    g.clearRect(0, 0, W, W);
    g.strokeStyle = 'rgba(255,255,255,0.15)'; g.beginPath(); g.moveTo(0, W / 2); g.lineTo(W, W / 2); g.moveTo(W / 2, 0); g.lineTo(W / 2, W); g.stroke();
    g.strokeStyle = 'rgba(255,140,120,0.8)'; g.beginPath(); g.arc(W / 2, W / 2, (KOS_R / half) * (W / 2), 0, 2 * Math.PI); g.stroke();
    const [px, py] = P(STATION_PORT[0], 0), far = KOS_R + 50, t = Math.tan(CORRIDOR_HALF);
    g.strokeStyle = 'rgba(160,215,255,0.7)'; g.beginPath(); g.moveTo(px, py); g.lineTo(...P(STATION_PORT[0] - far, far * t)); g.moveTo(px, py); g.lineTo(...P(STATION_PORT[0] - far, -far * t)); g.stroke();
    g.fillStyle = '#9fd0ff'; for (const X of [H1_POINT[0], STATION_PORT[0] - 20 - 18.9]) { const [a, b] = P(X, 0); g.fillRect(a - 3, b - 3, 6, 6); }
    if (sim.t - driftT > 2 || !drift) {
      driftT = sim.t;
      const xs = Float64Array.from(x), safe = sim.guid.st.phase === PH.TRANSFER ? passiveMin(xs).rMin >= 240 : passiveMin(xs).rMin > KOS_R;
      drift = { safe, lines: [-DRAG_B, DRAG_B].map((ad) => Array.from({ length: 145 }, (_, k) => propagate(xs, (k * DAY) / 144, ad))) };
      drift.plan = Array.from({ length: 61 }, (_, k) => propagate(xs, (k * T_ORB) / 120, 0));
    }
    g.strokeStyle = drift.safe ? 'rgba(125,255,174,0.8)' : 'rgba(255,190,90,0.85)';
    for (const line of drift.lines) { g.beginPath(); line.forEach((p, k) => (k ? g.lineTo(...P(p[0], p[1])) : g.moveTo(...P(p[0], p[1])))); g.stroke(); }
    g.strokeStyle = 'rgba(255,255,255,0.55)'; g.setLineDash([6, 6]); g.beginPath(); drift.plan.forEach((p, k) => (k ? g.lineTo(...P(p[0], p[1])) : g.moveTo(...P(p[0], p[1])))); g.stroke(); g.setLineDash([]);
    const [sx, sy] = P(x[0], x[1]); g.fillStyle = '#fff'; g.beginPath(); g.arc(sx, sy, 5, 0, 2 * Math.PI); g.fill();
  }
  return {
    note(text) { note = text; },
    intro(onStart) {
      showCard((c) => { c.innerHTML = INTRO; },
        [['Start', () => { closeCard(); onStart(); }], ['Back to the game', () => on.back()]]);
    },
    update(sim, info) {
      const ph = sim.rep.result === 'capture' ? 'CAPTURE' : sim.guid.st.phase;
      chipEl.forEach((c, i) => c.classList.toggle('on', CHIPS[i] === ph || (ph === PH.DEPART && CHIPS[i] === PH.BREAKOUT)));
      const x = sim.x, r = Math.hypot(x[0], x[1], x[2]), p = portRel(x, sim.q), rho = Math.hypot(...p), rhoDot = (p[0] * x[3] + p[1] * x[4] + p[2] * x[5]) / Math.max(rho, 1e-9);
      const vals = { ph, t: (sim.t / 60).toFixed(1), r: r.toFixed(1), rho: rho.toFixed(2), rhoV: rhoFromR(r).toFixed(1), rhoDot: (rhoDot * 100).toFixed(1), dv: sim.rep.dv.toFixed(2), prop: (MASS0 - sim.mass).toFixed(1), warp: info.warp.toFixed(0), mode: sim.ctrl.stats.mode === 'P' ? 'primary' : 'vernier', note };
      for (const k in vals) slot[k].textContent = vals[k];
      slot.noteRow.hidden = !note;
      JETS.forEach((j, i) => jetEl[i].classList.toggle('on', sim.onTimes[i] > 0));
      drawInset(sim);
    },
    report(sim) {
      const c = sim.rep.contact, res = sim.rep.result;
      showCard((card) => {
        el('h3', '', card, res === 'capture' ? 'Soft capture' : res === 'breakout' ? 'Breakout' : 'Failure');
        if (c) {
          const body = el('tbody', '', el('table', '', card));
          const row = (name, v, ok) => { const tr = el('tr', '', body); el('td', '', tr, name); el('td', '', tr, v); el('td', ok ? 'ok' : 'no', tr, ok ? 'within' : 'outside'); };
          row('closing', (c.close * 100).toFixed(1) + ' cm/s', c.close >= IDSS.closeMin && c.close <= IDSS.closeMax);
          row('lateral', (c.lat * 100).toFixed(2) + ' cm/s', c.lat <= IDSS.lat);
          row('rates', c.rate.toFixed(3) + ' deg/s', c.rate <= IDSS.rate / DEG);
          row('misalignment', (c.mis * 100).toFixed(1) + ' cm', c.mis <= IDSS.mis);
          row('angle', c.ang.toFixed(2) + ' deg', c.ang <= IDSS.ang / DEG);
        } else el('p', '', card, sim.rep.reason || 'The breakout left the keep-out sphere on a passively safe drift.');
        el('p', '', card, `${(sim.t / 60).toFixed(1)} min · dv ${sim.rep.dv.toFixed(2)} m/s · ${(MASS0 - sim.mass).toFixed(1)} kg`);
      }, [['New run', () => { closeCard(); on.restart(); }], ['Back to the game', () => on.back()]]);
    },
    closeCard,
  };
}
