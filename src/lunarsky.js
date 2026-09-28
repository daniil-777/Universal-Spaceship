// The sky of a lunar orbit (?orbit=moon; src/space.js draws it): a circular polar orbit round the Moon whose plane holds
// the Moon's pole (≈ the ecliptic pole, 1.5° off), its plane and starting point chosen for the view (see compose()): the
// ground ahead in daylight under a low, raking Sun that stays out of the chase camera's frame, and the Earth in that
// frame, above the limb, whenever the phase allows it. Everything is given
// in the ship's local frame as in src/skyorbit.js (LVLH: x along the track, y up — away from the Moon's centre —, z = x × y),
// with the textured Earth's axes (Greenwich at the sphere's +x, the north pole at +y, 90° E at −z) for the far globe.
// The Moon is tidally locked (no libration): selenographic longitude 0 faces the Earth.
import { julianDay, gmst, sunEci, moonEci, moonPhase, createSkyClock } from './ephem.js';

export const MU_MOON = 4902.8, R_MOON = 1737.4;                 // km³/s², km
const DEG = Math.PI / 180, AU = 149597870.7, EPS = 23.4393 * DEG, POLE = [0, -Math.sin(EPS), Math.cos(EPS)];   // the ecliptic pole in the Earth's equatorial frame
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2], cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = (v) => { const l = Math.hypot(v[0], v[1], v[2]); return [v[0] / l, v[1] / l, v[2] / l]; };
const toL = (L, v, o) => { o[0] = dot(L.x, v); o[1] = dot(L.y, v); o[2] = dot(L.z, v); return o; };
// The backdrop's pitch (src/space.js): up by the dip beyond the LEO Earth's (20.3°), so the lunar limb sits where the Earth's does
export const framePitch = (dip) => Math.max(0, dip - Math.acos(6371 / 6791));
const CAM = -13 * DEG, SAMPLES = [];                             // the chase camera: 13° down, 50° × 79° (src/space.js VIEW_PITCH, app.js CAM_FOV)
for (const el of [-36, -30, -24]) for (const az of [-30, 0, 30]) SAMPLES.push([Math.cos(el * DEG) * Math.cos(az * DEG), Math.sin(el * DEG), Math.cos(el * DEG) * Math.sin(az * DEG)]);
const litScore = (el) => (el <= 0 ? 0 : el < 12 * DEG ? el / (12 * DEG) : el <= 45 * DEG ? 1 : Math.max(0, 1 - (el - 45 * DEG) / (30 * DEG)));
// The start: over every polar plane (its node φ) and phase u, score the view — the Sun's height over the ground in the
// frame (9 rays), the Sun kept > 42° off the camera axis, ×1.6 if the Earth stands in the frame above the limb.
function compose(X, Y, P, sun, earth, r) {
  const dip = Math.acos(R_MOON / r), th = framePitch(dip), ct = Math.cos(th), st = Math.sin(th), cam = [Math.cos(CAM), Math.sin(CAM), 0];
  const rays = SAMPLES.map((w) => [ct * w[0] + st * w[1], -st * w[0] + ct * w[1], w[2]]), W = (v) => [ct * v[0] - st * v[1], st * v[0] + ct * v[1], v[2]];   // world → LVLH, LVLH → world
  let best = { score: -1, A: X, u: 0 }; const L = { x: 0, y: 0, z: 0 }, sL = [0, 0, 0], eL = [0, 0, 0];
  for (let f = 0; f < 360; f += 4) {
    const A = [0, 1, 2].map((i) => Math.cos(f * DEG) * X[i] + Math.sin(f * DEG) * Y[i]);
    for (let g = 0; g < 360; g += 2) {
      const cu = Math.cos(g * DEG), su = Math.sin(g * DEG); L.y = A.map((c, i) => cu * c + su * P[i]); L.x = A.map((c, i) => -su * c + cu * P[i]); L.z = cross(L.x, L.y);
      toL(L, sun, sL); if (dot(W(sL), cam) > Math.cos(42 * DEG)) continue;
      let lit = 0; for (const d of rays) { const b = -r * d[1], disc = b * b - (r * r - R_MOON * R_MOON); if (disc < 0) continue; const t = b - Math.sqrt(disc);
        lit += litScore(Math.asin((t * d[0] * sL[0] + (t * d[1] + r) * sL[1] + t * d[2] * sL[2]) / R_MOON)); }
      toL(L, earth, eL); const ew = W(eL), el = Math.asin(ew[1]), az = Math.atan2(ew[2], ew[0]);
      const score = lit / rays.length * (Math.abs(az) < 34 * DEG && el > CAM - 21 * DEG && el < CAM + 21 * DEG && Math.asin(eL[1]) > -dip + DEG ? 1.6 : 1);   // the Earth only with lit ground
      if (score > best.score + 1e-9) best = { score, A, u: g * DEG };
    }
  }
  return best;
}

export function createLunarSky({ t0 = Date.now(), h = 1500 } = {}) {
  const clock = createSkyClock(t0); let epoch = t0, real = 0, hKm = h, u0 = 0;
  const rad = () => R_MOON + hKm, mean = () => Math.sqrt(MU_MOON / rad() ** 3);
  const P = POLE; let A = [1, 0, 0];                              // the orbit: r̂(u) = cos u·A + sin u·P
  const place = (utc) => {                                         // compose the view for this moment and height: the plane (A) and the phase (u0) from utc on
    const jd = julianDay(utc), e = unit(moonEci(jd).map((c) => -c)), X = unit(e.map((c, i) => c - dot(e, P) * P[i])), Y = cross(P, X);   // the lunar equator: X toward the Earth
    const best = compose(X, Y, P, sunEci(jd), e, rad()); A = best.A; u0 = best.u; epoch = utc;
  };
  place(t0);
  const out = { body: 'moon', axes: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], earthAxes: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], sun: [0, 1, 0], earth: [1, 0, 0], earthDist: 384400,
    moon: [0, -1, 0], moonDist: rad(), north: [0, 1, 0], moonToEarth: [1, 0, 0], moonFraction: 0.5, earthFraction: 0.5, rHat: [1, 0, 0], u: u0, dip: 0, pitch: 0,
    speed: 0, groundSpeed: 0, alt: h, lat: 0, lon: 0, utc: t0, warp: 1, radiusKm: R_MOON };
  const L = { x: [1, 0, 0], y: [0, 1, 0], z: [0, 0, 1] };
  function update(realSec) {
    real = realSec; const utc = clock.now(realSec), jd = julianDay(utc), g = gmst(jd), cg = Math.cos(g), sg = Math.sin(g), r = rad(), n = mean();
    const u = u0 + n * (utc - epoch) / 1000, cu = Math.cos(u), su = Math.sin(u);
    for (let i = 0; i < 3; i++) { L.y[i] = cu * A[i] + su * P[i]; L.x[i] = -su * A[i] + cu * P[i]; }
    L.z = cross(L.x, L.y);
    const m = moonEci(jd), s = sunEci(jd), ship = L.y.map((c) => c * r);
    const eS = [-m[0] - ship[0], -m[1] - ship[1], -m[2] - ship[2]]; out.earthDist = Math.hypot(...eS); toL(L, unit(eS), out.earth);
    toL(L, unit([s[0] * AU - m[0] - ship[0], s[1] * AU - m[1] - ship[1], s[2] * AU - m[2] - ship[2]]), out.sun);
    const eM = unit(m.map((c) => -c)); toL(L, eM, out.moonToEarth); toL(L, P, out.north);
    toL(L, [cg, sg, 0], out.earthAxes[0]); toL(L, [0, 0, 1], out.earthAxes[1]); toL(L, [sg, -cg, 0], out.earthAxes[2]);
    const ph = moonPhase(s, m); out.moonFraction = ph.fraction; out.earthFraction = (1 - Math.cos(ph.phaseAngle)) / 2;   // the Earth from the Moon: the complementary phase
    const X = unit(eM.map((c, i) => c - dot(eM, P) * P[i])), Y = cross(P, X);                    // selenographic frame: longitude 0 at the Earth
    out.lat = Math.asin(dot(L.y, P)); out.lon = Math.atan2(dot(L.y, Y), dot(L.y, X));
    out.rHat = L.y.slice(); out.u = u; out.dip = Math.acos(R_MOON / r); out.pitch = framePitch(out.dip); out.moonDist = r;
    out.speed = Math.sqrt(MU_MOON / r); out.groundSpeed = out.speed * R_MOON / r; out.alt = hKm; out.utc = utc; out.warp = clock.warp;   // the Moon's own spin (4.6 m/s) left out
    return out;
  }
  return {
    out, update, clock,
    get period() { return 2 * Math.PI / mean(); }, get meanMotion() { return mean(); }, get altitude() { return hKm; },
    recompose() { place(clock.now(real)); },                        // a switch to the Moon later on: the composed view for now and the current height
    setWarp(w) { clock.setWarp(w, real); },
    setAltitude(km) { const now = clock.now(real); u0 += mean() * (now - epoch) / 1000; epoch = now; hKm = km; },   // the same plane and place, a new height
  };
}
