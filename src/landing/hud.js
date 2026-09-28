// What the crew sees: in the cockpit view a conformal green HUD — horizon and pitch ladder, the flight-path vector, the
// −3° glideslope reference, speed and altitude tapes with the speed bug, the radio altitude, localizer and glideslope
// deviation scales and the flight mode annunciator (thrust | vertical | lateral); in the other views a flight strip. Radio-
// altitude callouts appear (and are spoken, when voice is on), and the landing ends with a touchdown report card.
import * as THREE from 'three';
import { KT, FT } from './vehicle.js';
import { AIRPORT } from './airport.js';
import { parseMetarWind } from './wind.js';

const SAY = { '1000': 'one thousand', '500': 'five hundred', '100': 'one hundred', MINIMUMS: 'minimums', 40: 'forty', 30: 'thirty', 20: 'twenty', 10: 'ten', RETARD: 'retard', 'GO AROUND': 'go around' };
const CSS = `
  .lnd-strip, .lnd-card, .lnd-bar { position: fixed; z-index: 20; font: 500 13px/1.35 ui-monospace, "SF Mono", Menlo, monospace; color: #e8f4ec; background: rgba(8, 14, 20, 0.62); backdrop-filter: blur(6px); border: 1px solid rgba(255,255,255,0.12); border-radius: 10px; }
  .lnd-strip { left: 16px; bottom: 16px; padding: 10px 14px; min-width: 250px; max-width: calc(100vw - 32px); } .lnd-strip b { color: #7dffae; font-weight: 600; } .lnd-strip .fma { color: #7dffae; letter-spacing: 0.06em; margin-bottom: 4px; }
  .lnd-bar { top: 14px; right: 16px; padding: 8px; display: flex; flex-wrap: wrap; gap: 6px; max-width: min(560px, calc(100vw - 32px)); } .lnd-bar button { font: inherit; color: #e8f4ec; background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.18); border-radius: 7px; padding: 4px 8px; cursor: pointer; }
  .lnd-bar button.on { background: rgba(125,255,174,0.22); border-color: #7dffae; } .lnd-call { position: fixed; z-index: 21; left: 50%; top: 22%; transform: translateX(-50%); font: 700 34px/1 ui-monospace, Menlo, monospace; color: #7dffae; text-shadow: 0 0 12px rgba(0,0,0,0.7); pointer-events: none; transition: opacity 0.5s; }
  .lnd-card { left: 50%; top: 50%; transform: translate(-50%, -50%); padding: 18px 22px; min-width: 320px; max-width: calc(100vw - 32px); } .lnd-card h3 { margin: 0 0 8px; font: 700 18px/1.2 system-ui, sans-serif; color: #fff; } .lnd-card table { border-collapse: collapse; width: 100%; } .lnd-card td { padding: 2px 6px 2px 0; } .lnd-card td:last-child { text-align: right; color: #7dffae; }
  .lnd-card .row { display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap; } .lnd-card button { font: 600 13px system-ui, sans-serif; padding: 7px 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.2); background: rgba(125,255,174,0.16); color: #fff; cursor: pointer; }
  .lnd-atc { position: fixed; z-index: 20; right: 16px; bottom: 16px; max-width: min(430px, calc(100vw - 32px)); padding: 8px 12px; font: 500 12px/1.4 ui-monospace, "SF Mono", Menlo, monospace; color: #ffe9a8; background: rgba(8, 14, 20, 0.62); backdrop-filter: blur(6px); border: 1px solid rgba(255,255,255,0.12); border-radius: 10px; transition: opacity 0.6s; pointer-events: none; }
  @media (max-width: 700px) { .lnd-atc { left: 16px; bottom: auto; top: 110px; } }
  .lnd-wx { position: fixed; z-index: 22; top: 64px; right: 16px; width: min(330px, calc(100vw - 32px)); padding: 12px 14px; font: 500 12px/1.3 ui-monospace, "SF Mono", Menlo, monospace; color: #e8f4ec; background: rgba(8, 14, 20, 0.86); backdrop-filter: blur(6px); border: 1px solid rgba(255,255,255,0.14); border-radius: 10px; }
  .lnd-wx label { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin: 5px 0; } .lnd-wx select, .lnd-wx input { font: inherit; color: #e8f4ec; background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.2); border-radius: 6px; padding: 2px 4px; width: 118px; } .lnd-wx input[type=checkbox] { width: auto; }
  .lnd-wx .row { display: flex; gap: 8px; margin-top: 10px; } .lnd-wx button { font: 600 12px system-ui, sans-serif; padding: 6px 10px; border-radius: 7px; border: 1px solid rgba(255,255,255,0.2); background: rgba(125,255,174,0.16); color: #fff; cursor: pointer; }
  canvas.lnd-hud { position: fixed; inset: 0; width: 100vw; height: 100vh; z-index: 15; pointer-events: none; }`;

export function createHud(root, on) {
  const st = document.createElement('style'); st.textContent = CSS; document.head.appendChild(st);
  const cv = document.createElement('canvas'); cv.className = 'lnd-hud'; root.appendChild(cv); const g = cv.getContext('2d');
  const strip = document.createElement('div'); strip.className = 'lnd-strip'; root.appendChild(strip);
  const call = document.createElement('div'); call.className = 'lnd-call'; call.style.opacity = 0; root.appendChild(call);
  const bar = document.createElement('div'); bar.className = 'lnd-bar'; root.appendChild(bar);
  const radio = document.createElement('div'); radio.className = 'lnd-atc'; radio.style.opacity = 0; root.appendChild(radio);
  const views = ['cinema', 'chase', 'cockpit', 'tower', 'runway', 'approach'];
  bar.innerHTML = views.map((v) => `<button data-v="${v}">${v}</button>`).join('') + '<button data-a="path">path</button><button data-a="voice" title="callouts and sound">sound</button><button data-a="speed">×1</button><button data-a="wx">weather</button><button data-a="new">new weather</button><button data-a="back">back to flight</button>';
  bar.addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return; if (b.dataset.v) on.view(b.dataset.v); else if (b.dataset.a === 'wx') { wx.hidden = !wx.hidden; b.classList.toggle('on', !wx.hidden); } else on[b.dataset.a]?.(b); });
  // the weather panel: every condition the page takes by URL, set by hand; Fly reloads the landing with them (the scene's sky, light and clouds are built once)
  const q = new URLSearchParams(location.search), mw = parseMetarWind(q.get('wind')), ceil = +q.get('ceiling'), wx = document.createElement('div'); wx.className = 'lnd-wx'; wx.hidden = true;
  const sel = (id, opts, v) => `<select id="wx-${id}">${opts.map(([k, t]) => `<option value="${k}"${k === v ? ' selected' : ''}>${t}</option>`).join('')}</select>`, num = (id, v, max) => `<input id="wx-${id}" type="number" min="0" max="${max}" value="${v}">`;
  wx.innerHTML = `<b>Weather for the next landing</b>`
    + `<label>time ${sel('time', [['day', 'day'], ['dusk', 'dusk'], ['night', 'night']], q.get('time') || 'day')}</label>`
    + `<label>visibility ${sel('vis', [['cavok', 'CAVOK'], ['haze', 'haze (8 km)'], ['fog', 'fog (1 km)']], q.get('vis') || 'cavok')}</label>`
    + `<label>clouds ${sel('clouds', [['', 'by visibility'], ['NONE', 'none'], ['FEW', 'few'], ['SCT', 'scattered'], ['BKN', 'broken'], ['OVC', 'overcast']], (q.get('clouds') || '').toUpperCase())}</label>`
    + `<label>cloud base, ft ${num('ceiling', Number.isFinite(ceil) && ceil > 0 ? Math.round(ceil) : '', 20000)}</label>`
    + `<label>wind from, ° ${num('dir', mw && mw.dir != null ? mw.dir : '', 360)}</label><label>wind, kt ${num('kt', mw ? mw.kt : '', 60)}</label><label>gusts to, kt ${num('gust', mw && mw.gust ? mw.gust : '', 80)}</label>`
    + `<label>turbulence ${sel('turb', [['', 'as drawn'], ['none', 'none'], ['light', 'light'], ['moderate', 'moderate'], ['severe', 'severe']], q.get('turb') || '')}</label>`
    + `<label>rain <input id="wx-rain" type="checkbox"${q.get('rain') === '1' ? ' checked' : ''}></label>`
    + `<div class="row"><button data-w="fly">Fly</button><button data-w="random">Random</button></div><div style="opacity:.7;margin-top:6px">Empty wind = drawn at random. The sky is set when the page loads.</div>`;
  root.appendChild(wx);
  wx.addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return; const p = new URLSearchParams(location.search), g = (id) => wx.querySelector('#wx-' + id);
    for (const k of ['time', 'vis', 'clouds', 'ceiling', 'wind', 'turb', 'rain', 'seed']) p.delete(k);
    if (b.dataset.w === 'fly') { const val = (id) => g(id).value.trim(), kt = val('kt'); p.set('time', val('time')); p.set('vis', val('vis')); if (val('clouds')) p.set('clouds', val('clouds')); if (val('ceiling')) p.set('ceiling', val('ceiling'));
      if (kt !== '' && Number.isFinite(+kt)) { const k = Math.max(0, Math.round(+kt)), dv = val('dir'), dir = dv === '' || !Number.isFinite(+dv) ? 'VRB' : String(((Math.round(+dv) % 360) + 360) % 360 || 360).padStart(3, '0'), gu = Math.round(+val('gust')); p.set('wind', dir + String(k).padStart(2, '0') + (val('gust') !== '' && gu > k ? 'G' + String(gu).padStart(2, '0') : '')); }
      if (val('turb')) p.set('turb', val('turb')); if (g('rain').checked) p.set('rain', '1'); }
    location.search = p.toString(); });
  let voice = false, shown = 0, card = null;
  const _v = new THREE.Vector3(), W = () => cv.width, H = () => cv.height;
  function say(label) { call.textContent = label; call.style.opacity = 1; clearTimeout(call._t); call._t = setTimeout(() => { call.style.opacity = 0; }, 1300);
    if (voice && window.speechSynthesis) { const u = new SpeechSynthesisUtterance(SAY[label] || label); u.rate = 1.15; u.pitch = 0.8; window.speechSynthesis.speak(u); } }
  function project(cam, dir, out) { _v.copy(cam.position).addScaledVector(dir, 1000).project(cam); out.x = (_v.x + 1) / 2 * W(); out.y = (1 - _v.y) / 2 * H(); out.ok = _v.z < 1; return out; }
  const pt = { x: 0, y: 0, ok: false }, d3 = new THREE.Vector3(), D = Math.PI / 180;
  function drawHud(sim, cam) {                             // conformal: projected through the cockpit camera
    const s = sim.flight, a = s.air, gnc = sim.gnc.st, w = W(), h = H(), k = h / 900, C = '#7dff9e', hdg = a.psi;
    g.strokeStyle = C; g.fillStyle = C; g.lineWidth = 1.6 * k; g.font = `${Math.round(15 * k)}px ui-monospace, Menlo, monospace`; g.textAlign = 'center';
    for (let p = -20; p <= 20; p += 5) {                                        // pitch ladder about the heading
      d3.set(Math.cos(p * D) * Math.cos(hdg), Math.sin(p * D), Math.cos(p * D) * Math.sin(hdg)); project(cam, d3, pt); if (!pt.ok) continue;
      const L = (p === 0 ? 260 : 60) * k; g.save(); g.translate(pt.x, pt.y); g.rotate(-a.phi); g.setLineDash(p < 0 ? [8 * k, 6 * k] : []); g.beginPath(); g.moveTo(-L, 0); g.lineTo(-18 * k, 0); g.moveTo(18 * k, 0); g.lineTo(L, 0); g.stroke();
      if (p) g.fillText(String(p), -L - 16 * k, 5 * k); g.restore(); }
    d3.set(Math.cos(-3 * D) * Math.cos(hdg), Math.sin(-3 * D), Math.cos(-3 * D) * Math.sin(hdg)); project(cam, d3, pt);   // the −3° glidepath reference
    if (pt.ok) { g.setLineDash([3 * k, 5 * k]); g.beginPath(); g.moveTo(pt.x - 90 * k, pt.y); g.lineTo(pt.x + 90 * k, pt.y); g.stroke(); g.setLineDash([]); }
    d3.set(s.v[0], s.v[1], s.v[2]).normalize(); project(cam, d3, pt);          // the flight-path vector
    if (pt.ok) { g.beginPath(); g.arc(pt.x, pt.y, 9 * k, 0, Math.PI * 2); g.moveTo(pt.x - 26 * k, pt.y); g.lineTo(pt.x - 9 * k, pt.y); g.moveTo(pt.x + 9 * k, pt.y); g.lineTo(pt.x + 26 * k, pt.y); g.moveTo(pt.x, pt.y - 9 * k); g.lineTo(pt.x, pt.y - 20 * k); g.stroke(); }
    const ias = a.V / KT, alt = s.p[1] / FT, ra = Math.max(0, a.hRA / FT);                // tapes and readouts
    g.textAlign = 'left'; g.strokeRect(w * 0.22 - 60 * k, h * 0.5 - 16 * k, 60 * k, 30 * k); g.fillText(ias.toFixed(0), w * 0.22 - 52 * k, h * 0.5 + 6 * k);
    g.fillText('▶ ' + (gnc.vRef / KT).toFixed(0), w * 0.22 - 60 * k, h * 0.5 - 30 * k); g.textAlign = 'right'; g.strokeRect(w * 0.78, h * 0.5 - 16 * k, 76 * k, 30 * k); g.fillText(alt.toFixed(0), w * 0.78 + 70 * k, h * 0.5 + 6 * k);
    g.fillText(`${(s.v[1] / FT * 60).toFixed(0)} fpm`, w * 0.78 + 76 * k, h * 0.5 + 36 * k);
    if (ra < 2500) { g.textAlign = 'center'; g.font = `700 ${Math.round(24 * k)}px ui-monospace, Menlo, monospace`; g.fillText(ra.toFixed(0), w / 2, h * 0.72); g.font = `${Math.round(15 * k)}px ui-monospace, Menlo, monospace`; }
    const dev = gnc.dev || {}, dx = w / 2, dy = h * 0.8;                     // LOC (bottom) and G/S (right) scales, 2 dots each way
    for (let i = -2; i <= 2; i++) { g.beginPath(); g.arc(dx + i * 30 * k, dy, 3 * k, 0, 6.3); g.stroke(); g.beginPath(); g.arc(w * 0.74, h * 0.5 + i * 30 * k, 3 * k, 0, 6.3); g.stroke(); }
    if (dev.locValid) { const x = dx - Math.max(-2.4, Math.min(2.4, dev.loc)) * 30 * k; g.beginPath(); g.moveTo(x, dy - 9 * k); g.lineTo(x + 7 * k, dy); g.lineTo(x, dy + 9 * k); g.lineTo(x - 7 * k, dy); g.closePath(); g.fill(); }
    if (dev.gsValid) { const y = h * 0.5 + Math.max(-2.4, Math.min(2.4, dev.gs)) * 30 * k, x = w * 0.74; g.beginPath(); g.moveTo(x - 9 * k, y); g.lineTo(x, y + 7 * k); g.lineTo(x + 9 * k, y); g.lineTo(x, y - 7 * k); g.closePath(); g.fill(); }
    const thr = gnc.vert === 'GA' ? 'TOGA' : gnc.retard || gnc.vert === 'ROLLOUT' || gnc.vert === 'STOP' ? 'IDLE' : 'SPEED';
    const vert = { ALT: 'ALT', GS: 'G/S', FLARE: 'FLARE', ROLLOUT: 'ROLL OUT', STOP: 'STOP', GA: 'GO AROUND' }[gnc.vert] || gnc.vert;
    g.textAlign = 'center'; g.fillText(`${thr}  |  ${vert}  |  ${gnc.lat}      AUTOLAND`, w / 2, 40 * k); g.fillText(`${Math.round(((a.psi / D) + AIRPORT.runway.heading + 360) % 360).toString().padStart(3, '0')}°`, w / 2, 64 * k);
  }
  function stripHtml(sim) {
    const s = sim.flight, a = s.air, gn = sim.gnc.st, w = sim.rep.cond.wind, d = Math.max(0, -s.p[0]) / 1852, dev = gn.dev || {};
    return `<div class="fma">${gn.mode}${gn.vert === 'GA' ? ' — ' + gn.why : ''}</div>IAS <b>${(a.V / KT).toFixed(0)}</b> kt · GS ${(a.gs / KT).toFixed(0)} · target ${(gn.vRef / KT).toFixed(0)}<br>ALT <b>${(s.p[1] / FT).toFixed(0)}</b> ft · RA ${Math.max(0, a.hRA / FT).toFixed(0)} · V/S ${(s.v[1] / FT * 60).toFixed(0)} fpm<br>`
      + `LOC ${dev.locValid ? dev.loc.toFixed(2) : '—'} · G/S ${dev.gsValid ? dev.gs.toFixed(2) : '—'} dots · ${d.toFixed(1)} NM to go<br>wind ${String(w.dir).padStart(3, '0')}/${w.kt}${w.gust ? 'G' + w.gust : ''} kt · ${w.turb === 'none' ? 'no' : w.turb} turbulence${sim.rep.cond.clouds ? ' · ' + sim.rep.cond.clouds : ''} · gear ${s.gear > 0.99 ? 'DOWN' : s.gear > 0.01 ? 'transit' : 'up'}${s.chute === 1 ? ' · CHUTE' : ''}`;
  }
  return {
    update(sim, cam, view) {
      const pw = Math.round(innerWidth * devicePixelRatio); if (cv.width !== pw) { cv.width = pw; cv.height = Math.round(innerHeight * devicePixelRatio); }
      g.clearRect(0, 0, cv.width, cv.height); if (view === 'cockpit') drawHud(sim, cam);
      strip.innerHTML = stripHtml(sim); strip.style.display = view === 'cockpit' ? 'none' : '';
      for (const b of bar.querySelectorAll('button[data-v]')) b.classList.toggle('on', b.dataset.v === view);
      const calls = sim.gnc.st.calls; while (shown < calls.length) say(calls[shown++].label);
    },
    reset() { shown = 0; if (card) { card.remove(); card = null; } },
    setVoice(v) { voice = v; },
    atc(text, speech, readback) {                            // a radio call: shown for 10 s, spoken (tower, then the crew's readback) when voice is on
      radio.innerHTML = text; radio.style.opacity = 1; clearTimeout(radio._t); radio._t = setTimeout(() => { radio.style.opacity = 0; }, 10000);
      if (!voice || !window.speechSynthesis) return;
      for (const [words, pitch, rate] of [[speech, 1.0, 1.08], [readback, 0.75, 1.15]]) if (words) { const u = new SpeechSynthesisUtterance(words); u.pitch = pitch; u.rate = rate; window.speechSynthesis.speak(u); }
    },
    report(rep) {
      if (card) return; const t = rep.td, ok = rep.result === 'landed', f = (v, n = 1) => v.toFixed(n);
      card = document.createElement('div'); card.className = 'lnd-card';
      const rows = t ? [['touchdown', `${f(t.x, 0)} m past the threshold`], ['sink rate', `${f(t.sinkFps, 1)} ft/s (${f(t.sinkFps * 60, 0)} fpm)`], ['load factor', `${f(rep.tdG, 2)} g`], ['centreline', `${f(Math.abs(t.z), 1)} m ${t.z > 0 ? 'right' : 'left'}`],
        ['speed', `${f(t.V / KT, 0)} kt`], ['pitch / bank / crab', `${f(t.pitch / D)}° / ${f(t.bank / D)}° / ${f(t.crab / D)}°`],
        ['vacated', rep.exit ? `via ${rep.exit.name} at ${f(rep.exit.kt, 0)} kt` : rep.stopX ? `stopped on the runway at ${f(rep.stopX, 0)} m` : '—'],
        ['stabilized 1000 / 500 ft', `${rep.gates[1000] ? 'yes' : 'no'} / ${rep.gates[500] ? 'yes' : 'no'}`]] : [['result', rep.result]];
      card.innerHTML = `<h3>${ok ? (t.grade[0].toUpperCase() + t.grade.slice(1)) + ' landing' : rep.result.toUpperCase()}</h3><table>${rows.map(([a, b]) => `<tr><td>${a}</td><td>${b}</td></tr>`).join('')}</table><div class="row"><button data-a="again">fly it again</button><button data-a="new">new weather</button><button data-a="close">close</button></div>`;
      card.addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return; if (b.dataset.a === 'close') { card.remove(); card = null; } else on[b.dataset.a]?.(); });
      root.appendChild(card);
    },
  };
}
