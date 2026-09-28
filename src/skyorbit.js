// The orbit's sky, worked out every frame for the backdrop (src/space.js draws it): the sky clock's UTC → the Earth's
// rotation (GMST), the Sun and the Moon (src/ephem.js) and the ship's orbit (src/orbit.js), all expressed in the ship's
// local frame (LVLH: x along the track — the corridor's +x —, y up, z right). For the textured globe (Greenwich at the
// sphere's +x, the north pole at +y, 90° E at −z) it returns the images of the sphere's axes — its orientation — and it
// starts on a dawn pass (the Sun ~10° above the ship's horizon, rising ahead) unless told where to be.
import { julianDay, gmst, sunEci, moonEci, moonPhase, createSkyClock } from './ephem.js';
import { circularOrbit, R_EARTH } from './orbit.js';

const DEG = Math.PI / 180, dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export function createSkyOrbit({ t0 = Date.now(), h = 420 } = {}) {
  const clock = createSkyClock(t0); let orbit = circularOrbit({ h }), epoch = t0, real = 0;
  { // dawn: the sub-satellite point 80° west of the sub-solar point (local time ≈ 6:40), at half the Sun's latitude
    const jd = julianDay(t0), s = sunEci(jd), g = gmst(jd);
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
