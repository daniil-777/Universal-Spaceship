// The ship's orbit: a circular Kepler orbit (ISS-like by default: 420 km, 51.6°) with the J2 regression of its node
// (≈ −5°/day), in the equatorial frame of date (src/ephem.js). It gives the ship's state, its local frame (LVLH: x along
// the track — the corridor's +x —, y radial up, z = x × y), the point on the ground beneath it and its ground speed, and
// re-phases itself so that the ship passes over a chosen place at a chosen moment (inclined just enough to reach it).
import { gmst, julianDay } from './ephem.js';

export const MU = 398600.4418, R_EARTH = 6371, J2 = 1.08263e-3, OMEGA_E = 7.2921159e-5;   // km³/s², km, —, rad/s
export function circularOrbit({ h = 420, inc = 51.6 * Math.PI / 180, raan = 0, u0 = 0 } = {}) {
  const a = R_EARTH + h, n = Math.sqrt(MU / (a * a * a)), raanRate = -1.5 * n * J2 * (R_EARTH / a) ** 2 * Math.cos(inc), ci = Math.cos(inc), si = Math.sin(inc);
  const o = {
    h, inc, raan, u0, a, n, raanRate, period: 2 * Math.PI / n, speed: a * n,
    stateEci(t, out = { r: [0, 0, 0], v: [0, 0, 0] }) {     // t: seconds from the orbit's epoch
      const Om = raan + raanRate * t, u = u0 + n * t, cO = Math.cos(Om), sO = Math.sin(Om), cu = Math.cos(u), su = Math.sin(u), r = out.r, v = out.v;
      r[0] = a * (cO * cu - sO * su * ci); r[1] = a * (sO * cu + cO * su * ci); r[2] = a * su * si;
      v[0] = a * n * (-cO * su - sO * cu * ci); v[1] = a * n * (-sO * su + cO * cu * ci); v[2] = a * n * cu * si; return out;
    },
    lvlh(t) {
      const { r, v } = o.stateEci(t), rl = Math.hypot(...r), vl = Math.hypot(...v), y = r.map((c) => c / rl), x = v.map((c) => c / vl);
      return { x, y, z: [x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]] };
    },
    subPoint(t, g) {                                       // g: GMST (rad) at that moment → latitude, longitude (rad, east +)
      const r = o.stateEci(t).r, lon = Math.atan2(r[1], r[0]) - g;
      return { lat: Math.asin(r[2] / Math.hypot(...r)), lon: Math.atan2(Math.sin(lon), Math.cos(lon)) };
    },
    groundSpeed(t) {                                       // km/s over the turning Earth, at the surface
      const { r, v } = o.stateEci(t), w = [v[0] + OMEGA_E * r[1], v[1] - OMEGA_E * r[0], v[2]]; return Math.hypot(...w) * R_EARTH / a;
    },
    phaseOver(lat, lon, utc) {                             // the same height, inclined ≥ |lat| + 0.5°, over (lat, lon) at utc (the new epoch)
      const i2 = Math.max(inc, Math.abs(lat) + 0.5 * Math.PI / 180), u = Math.asin(Math.sin(lat) / Math.sin(i2)), dl = Math.atan2(Math.cos(i2) * Math.sin(u), Math.cos(u));
      return circularOrbit({ h, inc: i2, raan: lon + gmst(julianDay(utc)) - dl, u0: u });
    },
  };
  return o;
}
