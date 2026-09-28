// The orbit's sky, worked out every frame for the backdrop (src/space.js draws it): the sky clock's UTC → the Earth's
// rotation (GMST), the Sun and the Moon (src/ephem.js) and the ship's orbit (src/orbit.js), all expressed in the ship's
// local frame (LVLH: x along the track — the corridor's +x —, y up, z right). For the textured globe (Greenwich at the
// sphere's +x, the north pole at +y, 90° E at −z) it returns the images of the sphere's axes — its orientation. It starts
// with the Moon in the chase camera's frame, rising over the Earth's limb beside the ship, whenever the Moon is lit enough
// to see (≥ 15 %; see moonStart), and otherwise on a dawn pass (the Sun ~10° above the ship's horizon, rising ahead);
// start: 'dawn' or 'moon' forces either.
import { julianDay, gmst, sunEci, moonEci, moonPhase, createSkyClock } from './ephem.js';
import { circularOrbit, R_EARTH } from './orbit.js';

const DEG = Math.PI / 180, dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2], cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const MOON_MIN = 0.15, RAYS = [];                                // the chase camera looks 13° down (src/space.js VIEW_PITCH): rays into the lower frame
for (const el of [-36, -30, -24]) for (const az of [-30, 0, 30]) RAYS.push([Math.cos(el * DEG) * Math.cos(az * DEG), Math.sin(el * DEG), Math.cos(el * DEG) * Math.sin(az * DEG)]);
// The Moon view: over the orbit's node Ω (2° steps) and phase u (1°) at t0, the Moon — seen from the ship, parallax and
// all — 1.5–8° over the limb of the height the view starts at (viewH: 250 km when the page opens on a low pass) and
// 8°…min(21°, halfFov − 2°) beside the ship (its hull covers ±6°; halfFov: the screen's horizontal half-angle, ±11.9° on a
// tall phone, the 2° keep the 4× disc inside it); among those, daylight first (the sunlit share of the ground in the frame,
// then the Sun's height: a waning Moon rises before the Sun, so for a week after full Moon the view starts in the Earth's
// shadow — near a half or gibbous Moon the sunrise can be 15–25 min away), the placement nearest 4° up, 13° aside (less on
// a narrow screen) breaking ties. It rises ~3.9°/min, so it crosses the frame in ~7 min.
function moonStart(h, inc, t0, viewH, halfFov) {
  const jd = julianDay(t0), s = sunEci(jd), m = moonEci(jd), a = R_EARTH + h, dip = Math.acos(R_EARTH / (R_EARTH + viewH)), ci = Math.cos(inc), si = Math.sin(inc);
  const azMax = Math.min(21 * DEG, halfFov - 2 * DEG), azT = Math.min(13, halfFov / DEG - 3); if (azMax < 8 * DEG) return null;   // too narrow a screen: the dawn pass
  let best = null;
  for (let O = 0; O < 360; O += 2) {
    const cO = Math.cos(O * DEG), sO = Math.sin(O * DEG);
    for (let u = 0; u < 360; u++) {
      const cu = Math.cos(u * DEG), su = Math.sin(u * DEG), y = [cO * cu - sO * su * ci, sO * cu + cO * su * ci, su * si], x = [-cO * su - sO * cu * ci, -sO * su + cO * cu * ci, cu * si];
      const d = [m[0] - a * y[0], m[1] - a * y[1], m[2] - a * y[2]], dl = Math.hypot(d[0], d[1], d[2]), mx = dot(x, d) / dl; if (mx <= 0) continue;
      const above = Math.asin(dot(y, d) / dl) + dip; if (above < 1.5 * DEG || above > 8 * DEG) continue;
      const z = cross(x, y), az = Math.atan2(dot(z, d) / dl, mx); if (Math.abs(az) < 8 * DEG || Math.abs(az) > azMax) continue;
      const ea = (above / DEG - 4) / 1.5, ez = (Math.abs(az) / DEG - azT) / 5, fit = Math.exp(-ea * ea - ez * ez), sx = dot(x, s), sy = dot(y, s), sz = dot(z, s);
      let lit = 0; for (const r of RAYS) { const b = -a * r[1], disc = b * b - (a * a - R_EARTH * R_EARTH); if (disc < 0) continue; const t = b - Math.sqrt(disc); if (t * r[0] * sx + (t * r[1] + a) * sy + t * r[2] * sz > 0) lit++; }
      const sunEl = Math.asin(sy), sunq = sunEl >= -dip ? 1 : Math.max(0, 1 - (-dip - sunEl) / (20 * DEG));   // the ship in sunlight, or the sunrise within ~5 min
      const score = (0.2 + 0.5 * lit / RAYS.length + 0.3 * sunq) * (0.5 + 0.5 * fit); if (!best || score > best.score) best = { score, raan: O * DEG, u0: u * DEG };
    }
  }
  return best ? circularOrbit({ h, inc, raan: best.raan, u0: best.u0 }) : null;
}
export function createSkyOrbit({ t0 = Date.now(), h = 420, start = 'auto', viewH = h, halfFov = 39.6 * DEG } = {}) {   // halfFov default: 16:9 at 50° vertical
  const clock = createSkyClock(t0), jd0 = julianDay(t0); let orbit = circularOrbit({ h }), epoch = t0, real = 0;
  const moonUp = start === 'moon' || (start !== 'dawn' && moonPhase(sunEci(jd0), moonEci(jd0)).fraction >= MOON_MIN), mo = moonUp ? moonStart(h, orbit.inc, t0, viewH, halfFov) : null;
  if (mo) orbit = mo;
  else { // dawn: the sub-satellite point 80° west of the sub-solar point (local time ≈ 6:40), at half the Sun's latitude
    const s = sunEci(jd0), g = gmst(jd0);
    orbit = orbit.phaseOver(Math.asin(s[2]) * 0.5, Math.atan2(s[1], s[0]) - g - 80 * DEG, t0);
  }
  const out = { axes: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], sun: [0, 1, 0], moon: [1, 0, 0], moonDist: 384400, moonFraction: 0.5, earthFraction: 0.5, north: [0, 1, 0], moonToEarth: [0, -1, 0],
    speed: orbit.speed, groundSpeed: 0, alt: h, lat: 0, lon: 0, utc: t0, warp: 1, radiusKm: R_EARTH };
  const toL = (L, v, o) => { o[0] = dot(L.x, v); o[1] = dot(L.y, v); o[2] = dot(L.z, v); return o; }, _v = [0, 0, 0];
  function update(realSec) {
    real = realSec; const utc = clock.now(realSec), jd = julianDay(utc), g = gmst(jd), t = (utc - epoch) / 1000, L = orbit.lvlh(t), cg = Math.cos(g), sg = Math.sin(g);
    toL(L, [cg, sg, 0], out.axes[0]); toL(L, [0, 0, 1], out.axes[1]); toL(L, [sg, -cg, 0], out.axes[2]);   // the sphere's +x (Greenwich), +y (north), +z (−90° E) seen from the ship
    const s = sunEci(jd); toL(L, s, out.sun);
    const m = moonEci(jd), r = orbit.stateEci(t).r; for (let i = 0; i < 3; i++) _v[i] = m[i] - r[i];
    const d = Math.hypot(..._v); for (let i = 0; i < 3; i++) _v[i] /= d; toL(L, _v, out.moon); out.moonDist = d;
    const ph = moonPhase(s, m); out.moonFraction = ph.fraction; out.earthFraction = (1 - Math.cos(ph.phaseAngle)) / 2;   // the Earth from the Moon: the complementary phase
    const md = Math.hypot(...m); toL(L, [-m[0] / md, -m[1] / md, -m[2] / md], out.moonToEarth); toL(L, [0, 0, 1], out.north);
    const sp = orbit.subPoint(t, g); out.lat = sp.lat; out.lon = sp.lon; out.speed = orbit.speed; out.groundSpeed = orbit.groundSpeed(t); out.alt = orbit.h; out.utc = utc; out.warp = clock.warp;
    return out;
  }
  return {
    out, update, clock, get orbit() { return orbit; },
    setWarp(w) { clock.setWarp(w, real); },
    setAltitude(hKm) {                                     // orbit ↔ low pass: the same plane and phase, a new height
      const t = (clock.now(real) - epoch) / 1000; orbit = circularOrbit({ h: hKm, inc: orbit.inc, raan: orbit.raan + orbit.raanRate * t, u0: orbit.u0 + orbit.n * t }); epoch = clock.now(real);
    },
    phaseOver(latDeg, lonDeg) { const now = clock.now(real); orbit = orbit.phaseOver(latDeg * DEG, lonDeg * DEG, now); epoch = now; },
  };
}
