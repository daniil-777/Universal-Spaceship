// Navigation for the landing scenario: the ILS as the ship's receivers see it — localizer and glideslope deviations in dots
// (fly-to: + means the ship is right of the course / above the path), with the small beam bends and noise of a real
// installation — and the radio altimeter. The GS is a cone around its antenna (so its height at the threshold is a little
// above the nominal 15.7 m); the LOC fans out from its array beyond the far end.
import { AIRPORT } from './airport.js';

const LOC = AIRPORT.ils.loc, GS = AIRPORT.ils.gs, GSA = GS.angle * Math.PI / 180;
export function createIlsNoise(rng) {                     // slow bends (fixed per installation) + a little receiver noise
  const ph = [rng() * 6.28, rng() * 6.28, rng() * 6.28, rng() * 6.28];
  return { loc: (x, t) => 0.03 * Math.sin(x / 520 + ph[0]) + 0.015 * Math.sin(x / 170 + ph[1]) + 0.01 * Math.sin(t * 3.1 + ph[2]),
    gs: (x, t) => 0.04 * Math.sin(x / 610 + ph[3]) + 0.01 * Math.sin(t * 2.7 + ph[0]) };
}
export function ilsDeviation(p, noise, t, out) {          // p: the receiver's position (runway frame)
  const la = Math.atan2(p[2] - LOC.z, LOC.x - p[0]), d = Math.hypot(p[0] - GS.x, p[2] - GS.z), ga = Math.atan2(p[1], d);
  out.locAngle = la; out.gsAngle = ga;
  out.loc = la / LOC.dot + (noise ? noise.loc(p[0], t) : 0);
  out.gs = (ga - GSA) / GS.dot + (noise ? noise.gs(p[0], t) : 0);
  out.locValid = Math.abs(la) < 35 * Math.PI / 180 && p[0] < LOC.x - 50;
  out.gsValid = out.locValid && Math.abs(la) < 8 * Math.PI / 180 && p[0] < GS.x - 60 && ga > 0.3 * GSA && ga < 1.75 * GSA;
  out.dmeNm = Math.hypot(p[0] - LOC.x, p[2] - LOC.z) / 1852;
  return out;
}
export const glidePathHeight = (x, z) => Math.hypot(x - GS.x, z - GS.z) * Math.tan(GSA);   // the GS cone's height over (x, z)
export const locMetres = (dots, x) => Math.tan(dots * LOC.dot) * (LOC.x - x);             // a LOC deviation as metres off the centreline
