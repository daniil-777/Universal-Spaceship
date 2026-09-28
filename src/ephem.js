// Astronomy for the orbit's sky: the date and the Earth's rotation (Julian day and Greenwich mean sidereal time, Meeus
// ch. 7 and 12), the Sun (ch. 25, ~0.01°) and the Moon (ch. 47 with its main periodic terms, ~0.01° and ~10 km) in the
// equatorial frame of date (x → the equinox, z → the north pole — the frame GMST turns the Earth in), the Moon's phase,
// and a sky clock that runs in real time or warped. Pure; the backdrop places what it returns.
const DEG = Math.PI / 180, J2000 = 2451545.0, AU = 149597870.7;
const rev = (d) => ((d % 360) + 360) % 360;
export const julianDay = (utcMs) => utcMs / 86400000 + 2440587.5;
export function gmst(jd) {                                // Meeus 12.4, radians
  const T = (jd - J2000) / 36525;
  return rev(280.46061837 + 360.98564736629 * (jd - J2000) + 0.000387933 * T * T - T * T * T / 38710000) * DEG;
}
const obliquity = (T) => (23.439291111 - 0.013004167 * T - 1.639e-7 * T * T + 5.036e-7 * T * T * T) * DEG;   // mean, Meeus 22.2

export function sunEcliptic(jd) {                         // apparent longitude, distance (km), true obliquity
  const T = (jd - J2000) / 36525, L0 = 280.46646 + 36000.76983 * T + 0.0003032 * T * T, M = (357.52911 + 35999.05029 * T - 0.0001537 * T * T) * DEG;
  const C = (1.914602 - 0.004817 * T - 0.000014 * T * T) * Math.sin(M) + (0.019993 - 0.000101 * T) * Math.sin(2 * M) + 0.000289 * Math.sin(3 * M);
  const e = 0.016708634 - 0.000042037 * T - 0.0000001267 * T * T, R = 1.000001018 * (1 - e * e) / (1 + e * Math.cos(M + C * DEG)), Om = (125.04 - 1934.136 * T) * DEG;
  return { lambda: rev(L0 + C - 0.00569 - 0.00478 * Math.sin(Om)) * DEG, dist: R * AU, eps: obliquity(T) + 0.00256 * DEG * Math.cos(Om) };
}
export function sunEci(jd, out = [0, 0, 0]) {             // unit vector toward the Sun
  const s = sunEcliptic(jd), cl = Math.cos(s.lambda), sl = Math.sin(s.lambda);
  out[0] = cl; out[1] = Math.cos(s.eps) * sl; out[2] = Math.sin(s.eps) * sl; return out;
}

// Meeus table 47.A (D, M, M′, F; Σl in 1e-6°, Σr in 1e-3 km) and 47.B (Σb in 1e-6°): the terms above ~0.0003°
const LR = [[0, 0, 1, 0, 6288774, -20905355], [2, 0, -1, 0, 1274027, -3699111], [2, 0, 0, 0, 658314, -2955968], [0, 0, 2, 0, 213618, -569925], [0, 1, 0, 0, -185116, 48888],
  [0, 0, 0, 2, -114332, -3149], [2, 0, -2, 0, 58793, 246158], [2, -1, -1, 0, 57066, -152138], [2, 0, 1, 0, 53322, -170733], [2, -1, 0, 0, 45758, -204586], [0, 1, -1, 0, -40923, -129620],
  [1, 0, 0, 0, -34720, 108743], [0, 1, 1, 0, -30383, 104755], [2, 0, 0, -2, 15327, 10321], [0, 0, 1, 2, -12528, 0], [0, 0, 1, -2, 10980, 79661], [4, 0, -1, 0, 10675, -34782],
  [0, 0, 3, 0, 10034, -23210], [4, 0, -2, 0, 8548, -21636], [2, 1, -1, 0, -7888, 24208], [2, 1, 0, 0, -6766, 30824], [1, 0, -1, 0, -5163, -8379], [1, 1, 0, 0, 4987, -16675],
  [2, -1, 1, 0, 4036, -12831], [2, 0, 2, 0, 3994, -10445], [4, 0, 0, 0, 3861, -11650], [2, 0, -3, 0, 3665, 14403], [0, 1, -2, 0, -2689, -7003], [2, 0, -1, 2, -2602, 0],
  [2, -1, -2, 0, 2390, 10056], [1, 0, 1, 0, -2348, 6322], [2, -2, 0, 0, 2236, -9884], [0, 1, 2, 0, -2120, 5751], [0, 2, 0, 0, -2069, 0], [2, -2, -1, 0, 2048, -4950],
  [2, 0, 1, -2, -1773, 4130], [2, 0, 0, 2, -1595, 0], [4, -1, -1, 0, 1215, -3958], [0, 0, 2, 2, -1110, 0], [3, 0, -1, 0, -892, 3258], [2, 1, 1, 0, -810, 2616],
  [4, -1, -2, 0, 759, -1897], [0, 2, -1, 0, -713, -2117], [2, 2, -1, 0, -700, 2354], [2, 1, -2, 0, 691, 0], [2, -1, 0, -2, 596, 0], [4, 0, 1, 0, 549, -1423], [0, 0, 4, 0, 537, -1117],
  [4, -1, 0, 0, 520, -1571], [1, 0, -2, 0, -487, -1739], [2, 1, 0, -2, -399, 0], [0, 0, 2, -2, -381, -4421], [1, 1, 1, 0, 351, 0], [3, 0, -2, 0, -340, 0], [4, 0, -3, 0, 330, 0],
  [2, -1, 2, 0, 327, 0], [0, 2, 1, 0, -323, 1165], [1, 1, -1, 0, 299, 0], [2, 0, 3, 0, 294, 0], [2, 0, -1, -2, 0, 8752]];
const B = [[0, 0, 0, 1, 5128122], [0, 0, 1, 1, 280602], [0, 0, 1, -1, 277693], [2, 0, 0, -1, 173237], [2, 0, -1, 1, 55413], [2, 0, -1, -1, 46271], [2, 0, 0, 1, 32573],
  [0, 0, 2, 1, 17198], [2, 0, 1, -1, 9266], [0, 0, 2, -1, 8822], [2, -1, 0, -1, 8216], [2, 0, -2, -1, 4324], [2, 0, 1, 1, 4200], [2, 1, 0, -1, -3359], [2, -1, -1, 1, 2463],
  [2, -1, 0, 1, 2211], [2, -1, -1, -1, 2065], [0, 1, -1, -1, -1870], [4, 0, -1, -1, 1828], [0, 1, 0, 1, -1794], [0, 0, 0, 3, -1749], [0, 1, -1, 1, -1565], [1, 0, 0, 1, -1491],
  [0, 1, 1, 1, -1475], [0, 1, 1, -1, -1410], [0, 1, 0, -1, -1344], [1, 0, 0, -1, -1335], [0, 0, 3, 1, 1107], [4, 0, 0, -1, 1021], [4, 0, -1, 1, 833]];
export function moonEcliptic(jd) {                        // geometric longitude/latitude (equinox of date), distance (km)
  const T = (jd - J2000) / 36525, T2 = T * T, T3 = T2 * T, T4 = T3 * T;
  const Lp = rev(218.3164477 + 481267.88123421 * T - 0.0015786 * T2 + T3 / 538841 - T4 / 65194000);
  const D = rev(297.8501921 + 445267.1114034 * T - 0.0018819 * T2 + T3 / 545868 - T4 / 113065000) * DEG, M = rev(357.5291092 + 35999.0502909 * T - 0.0001536 * T2 + T3 / 24490000) * DEG;
  const Mp = rev(134.9633964 + 477198.8675055 * T + 0.0087414 * T2 + T3 / 69699 - T4 / 14712000) * DEG, F = rev(93.2720950 + 483202.0175233 * T - 0.0036539 * T2 - T3 / 3526000 + T4 / 863310000) * DEG;
  const A1 = rev(119.75 + 131.849 * T) * DEG, A2 = rev(53.09 + 479264.290 * T) * DEG, A3 = rev(313.45 + 481266.484 * T) * DEG, E = 1 - 0.002516 * T - 0.0000074 * T2, L = Lp * DEG;
  let sl = 0, sr = 0, sb = 0;
  for (const [d, m, mp, f, cl, cr] of LR) { const e = m === 0 ? 1 : Math.abs(m) === 1 ? E : E * E, arg = d * D + m * M + mp * Mp + f * F; sl += cl * e * Math.sin(arg); sr += cr * e * Math.cos(arg); }
  for (const [d, m, mp, f, cb] of B) { const e = m === 0 ? 1 : Math.abs(m) === 1 ? E : E * E; sb += cb * e * Math.sin(d * D + m * M + mp * Mp + f * F); }
  sl += 3958 * Math.sin(A1) + 1962 * Math.sin(L - F) + 318 * Math.sin(A2);
  sb += -2235 * Math.sin(L) + 382 * Math.sin(A3) + 175 * Math.sin(A1 - F) + 175 * Math.sin(A1 + F) + 127 * Math.sin(L - Mp) - 115 * Math.sin(L + Mp);
  return { lambda: rev(Lp + sl / 1e6) * DEG, beta: (sb / 1e6) * DEG, dist: 385000.56 + sr / 1000, eps: obliquity(T) };
}
export function moonEci(jd, out = [0, 0, 0]) {            // km, from the Earth's centre
  const m = moonEcliptic(jd), cb = Math.cos(m.beta), sb = Math.sin(m.beta), cl = Math.cos(m.lambda), sl = Math.sin(m.lambda), ce = Math.cos(m.eps), se = Math.sin(m.eps);
  out[0] = m.dist * cb * cl; out[1] = m.dist * (cb * sl * ce - sb * se); out[2] = m.dist * (cb * sl * se + sb * ce); return out;
}
export function moonPhase(sun, moon) {                    // sun: unit vector; moon: km → illuminated fraction, phase angle, elongation
  const d = Math.hypot(moon[0], moon[1], moon[2]), c = (sun[0] * moon[0] + sun[1] * moon[1] + sun[2] * moon[2]) / d, psi = Math.acos(Math.max(-1, Math.min(1, c)));
  const i = Math.atan2(AU * Math.sin(psi), d - AU * Math.cos(psi)); return { fraction: (1 + Math.cos(i)) / 2, phaseAngle: i, elongation: psi };
}
export function createSkyClock(t0Utc = Date.now()) {       // UTC (ms) as a function of the page's real seconds; warp re-bases so the sky never jumps
  let base = t0Utc, real0 = 0, warp = 1;
  return { now: (realSec) => base + (realSec - real0) * 1000 * warp, setWarp(w, realSec) { base = base + (realSec - real0) * 1000 * warp; real0 = realSec; warp = w; }, get warp() { return warp; } };
}
