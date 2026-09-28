// Flight board: a small glass instrument cluster in the lower right — an attitude indicator whose aircraft symbol is the
// ship seen from behind, a heading tape, speed (km/h; Mach in the air), altitude, radar altitude, vertical speed,
// proximity, throttle and a warning chip — beside the frame of the onboard camera (src/pipcam.js draws the picture into
// the canvas under it; click the frame to switch views). Scales: 19 m per corridor unit (the Burj Khalifa is 44 units
// tall) for speeds and heights in the air; in orbit the altitude comes from the globe (6371 km = 1400 units). The dial
// moves every frame; the text changes at ~12 Hz. All text goes through textContent.
const SVGNS = 'http://www.w3.org/2000/svg', M_PER_UNIT = 19, SOUND = 343;
const PITCH_K = 1.1, HDG_K = 1.3;                          // viewBox units per degree: pitch ladder, heading tape
const el = (tag, attrs, parent) => { const e = document.createElementNS(SVGNS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
const fmtInt = (v) => Math.round(v).toLocaleString('en-US');
const CARD = { 0: 'N', 90: 'E', 180: 'S', 270: 'W' };

export function createBoard(root, { onCamera } = {}) {
  const $ = (id) => document.getElementById(id);
  // pitch ladder: ±5° short, ±10…±40° long with numbers; below the horizon dashed (as on a real attitude indicator)
  const ladder = $('adiLadder');
  for (let d = -40; d <= 40; d += 5) {
    if (d === 0) continue;
    const y = -d * PITCH_K, w = d % 10 === 0 ? 13 : 6.5;
    el('line', { x1: -w, x2: w, y1: y, y2: y, class: d < 0 ? 'neg' : '' }, ladder);
    if (d % 10 === 0) for (const s of [-1, 1]) { const t = el('text', { x: s * (w + 5.5), y: y + 2, 'text-anchor': 'middle' }, ladder); t.textContent = String(Math.abs(d)); }
  }
  const bank = $('adiBank');                              // fixed bank scale on the case: 10, 20, 30, 45, 60°
  for (const a of [-60, -45, -30, -20, -10, 10, 20, 30, 45, 60]) {
    const r0 = 47.5, r1 = Math.abs(a) === 30 || Math.abs(a) === 60 ? 54 : 51, s = Math.sin(a * Math.PI / 180), c = -Math.cos(a * Math.PI / 180);
    el('line', { x1: r0 * s, y1: r0 * c, x2: r1 * s, y2: r1 * c }, bank);
  }
  el('path', { d: 'M0,-48 l-3.2,-5.5 h6.4 z', class: 'zero' }, bank);
  const tape = $('hdgTape');                              // heading tape −90…450° so it scrolls across north without a jump
  for (let d = -90; d <= 450; d += 5) {
    const x = d * HDG_K, long = d % 10 === 0;
    el('line', { x1: x, x2: x, y1: 18, y2: long ? 12.5 : 15 }, tape);
    if (d % 30 === 0) { const n = ((d % 360) + 360) % 360, t = el('text', { x, y: 10, 'text-anchor': 'middle', class: CARD[n] ? 'card' : '' }, tape); t.textContent = CARD[n] || String(n / 10).padStart(2, '0'); }
  }
  const hz = $('adiHz'), ptr = $('adiPtr'), hdgText = $('hdgVal');
  const out = { spd: $('bSpd'), sub: $('bSub'), bar: $('bSpdBar'), alt: $('bAlt'), ra: $('bRa'), vs: $('bVs'), prox: $('bProx'), thr: $('bThr'), thrS: $('bThrS'), warn: $('bWarn'), pilot: $('bPilot'), air: $('bAir'), ias: $('bIas'), aoa: $('bAoa'), g: $('bG'), gbar: $('bGbar'), wind: $('bWind'), warr: $('bWindArr'), turb: $('bTurb'), cam: $('camName'), world: $('camWorld') };
  const cam = $('boardCam');
  cam.addEventListener('click', () => { if (onCamera) { const m = onCamera(); if (m) out.cam.textContent = m.name; } });
  let visible = true, textT = 1, rectCb = null, lastWarn = '';

  const measure = () => {                                  // the camera frame's rect in CSS px (null when hidden), for pipcam.setRect
    if (!rectCb) return;
    const r = cam.getBoundingClientRect(), shown = visible && !document.body.classList.contains('nohud') && r.width > 8 && getComputedStyle(cam).display !== 'none';
    rectCb(shown ? { x: Math.round(r.left), y: Math.round(r.top), w: Math.round(r.width), h: Math.round(r.height) } : null);
  };
  new ResizeObserver(measure).observe(cam); window.addEventListener('resize', measure);
  new MutationObserver(measure).observe(document.body, { attributes: true, attributeFilter: ['class'] });   // the H key toggles body.nohud

  return {
    onRect(cb) { rectCb = cb; measure(); },
    setVisible(b) { visible = b; root.classList.toggle('hidden', !b); document.body.classList.toggle('noboard', !b); measure(); },
    get visible() { return visible; },
    setCamera(m) { out.cam.textContent = m.name; },
    // s: { q (render quaternion), f, u (THREE.Vector3), v (velocity array), y, ground (height under the ship or null), y0,
    //      atmo 0..1, orbitAlt 0..1 (low pass), throttle −1..1, prox (nearest beam hit, units; Infinity if none), crashed, manual, world,
    //      edge 0..1 (the edge guard's authority), proxEdge (the nearest beam hit is a corridor edge) }
    update(dt, s) {
      if (!visible) return;
      // attitude: pitch from the nose's climb, bank from the right wing's drop, heading with north = −z (east = +x = 090°)
      const fx = s.f.x, fy = s.f.y, fz = s.f.z, ux = s.u.x, uy = s.u.y, uz = s.u.z;
      const ry = fz * ux - fx * uz;                        // the right wing's climb (right = forward × up)
      const pitch = Math.asin(Math.max(-1, Math.min(1, fy))) * 180 / Math.PI, roll = Math.atan2(-ry, uy) * 180 / Math.PI;
      let hdg = Math.atan2(fx, -fz) * 180 / Math.PI; if (hdg < 0) hdg += 360;
      const pk = Math.max(-45, Math.min(45, pitch)) * PITCH_K;
      hz.setAttribute('transform', `rotate(${(-roll).toFixed(2)}) translate(0 ${pk.toFixed(2)})`);
      ptr.setAttribute('transform', `rotate(${(-roll).toFixed(2)})`);
      tape.setAttribute('transform', `translate(${(-hdg * HDG_K).toFixed(2)} 0)`);
      textT += dt; if (textT < 0.08) return; textT = 0;
      hdgText.textContent = String(Math.round(hdg) % 360).padStart(3, '0');
      const air = s.atmo > 0.5, speed = Math.hypot(s.v[0], s.v[1], s.v[2]), mps = speed * M_PER_UNIT;
      const orb = !air && s.orbit;                          // in orbit: the real orbit's speed (src/skyorbit.js), the belt is along for the ride
      out.spd.textContent = fmtInt(orb ? orb.speedKmh : mps * 3.6);
      out.sub.textContent = air ? 'MACH ' + (mps / SOUND).toFixed(2) : orb ? `${orb.body === 'moon' ? 'LUNAR ' : ''}ORBIT · GROUND ${fmtInt(orb.groundKmh)}${orb.warp !== 1 ? ' · ×' + orb.warp : ''}` : 'REL · BELT';
      out.bar.style.transform = `scaleX(${Math.min(1, speed / 22).toFixed(3)})`;
      if (air) {
        out.alt.textContent = fmtInt((s.y - s.y0) * M_PER_UNIT) + ' m';
        out.ra.textContent = s.ground == null ? '—' : fmtInt(Math.max(0, s.y - s.ground) * M_PER_UNIT) + ' m';
      } else {
        out.alt.textContent = fmtInt(orb && orb.body === 'moon' ? orb.altKm : 420 + (250 - 420) * s.orbitAlt) + ' km'; out.ra.textContent = '—';   // the orbit's height: 420 km, 250 on a low pass (space.js)
      }
      const vs = s.v[1] * M_PER_UNIT; out.vs.textContent = (vs >= 0 ? '+' : '−') + Math.abs(vs).toFixed(0) + ' m/s';
      out.prox.textContent = Number.isFinite(s.prox) ? fmtInt(s.prox * M_PER_UNIT) + ' m' : 'CLEAR';
      const th = 0.5 + 0.5 * s.throttle; out.thr.style.transform = `scaleX(${th.toFixed(3)})`;
      out.thrS.textContent = s.throttle > 0.25 ? 'BOOST' : s.throttle < -0.25 ? 'BRAKE' : 'CRUISE';
      const inAir = air && s.airOn; out.air.style.display = inAir ? '' : 'none';   // the air data: indicated airspeed, angle of attack, g, wind, turbulence
      if (inAir) { out.ias.textContent = fmtInt(s.ias * M_PER_UNIT * 3.6); out.aoa.textContent = (s.aoa * 180 / Math.PI).toFixed(0) + '°'; out.g.textContent = s.g.toFixed(1);
        const gf = Math.max(0, Math.min(1, (s.g + 1.5) / 5)); out.gbar.style.width = (gf * 100).toFixed(1) + '%'; out.gbar.parentNode.classList.toggle('hot', s.g > 2.8 || s.g < -0.8);   // between the −1.5 / +3.5 g limits
        const hx = s.f.x, hz = s.f.z; out.warr.style.transform = `rotate(${Math.atan2(s.wind[2] * hx - s.wind[0] * hz, s.wind[0] * hx + s.wind[2] * hz).toFixed(2)}rad)`;   // the wind relative to the nose
        out.wind.textContent = fmtInt(Math.hypot(s.wind[0], s.wind[2]) * M_PER_UNIT * 3.6); out.turb.textContent = s.turb < 0.8 ? 'LIGHT' : s.turb < 2 ? 'MOD' : 'SEVERE'; }
      out.pilot.textContent = s.manual ? 'MANUAL' : 'AUTOPILOT';
      out.world.textContent = s.world;
      const ra = s.ground == null ? Infinity : s.y - s.ground;
      const warn = s.crashed ? (s.over ? 'OVERSTRESS' : 'CRASH') : s.stall ? 'STALL' : air && ra < 3 && s.v[1] < -1.5 ? 'PULL UP' : s.edge > 0.25 || (s.prox < 4 && s.proxEdge) ? 'EDGE' : s.prox < 4 ? 'PROXIMITY' : '';   // EDGE: the edge guard is turning the ship away, or the nearest thing in sight is the corridor's edge
      if (warn !== lastWarn) { lastWarn = warn; out.warn.textContent = warn; out.warn.className = 'warn' + (warn ? ' on' : '') + (warn === 'PROXIMITY' || warn === 'EDGE' ? ' amber' : ''); }
    },
  };
}
