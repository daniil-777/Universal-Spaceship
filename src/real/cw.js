// Clohessy-Wiltshire relative motion in LVLH (consts.js layout), exact for a circular orbit:
//   Xdd = -2n Yd + aX + ad,  Ydd = 3n^2 Y + 2n Xd + aY,  Zdd = -n^2 Z + aZ
// phi(h): 6x6 state transition (row-major); gamma(h): 6x3 zero-order-hold input matrix for a constant
// acceleration over h (column 0 = along-track, where the differential drag ad enters).
import { N } from './consts.js';

export function phi(h, out = new Float64Array(36)) {
  const n = N, s = Math.sin(n * h), c = Math.cos(n * h);
  out.fill(0);
  // X row
  out[0] = 1; out[1] = 6 * (s - n * h); out[3] = (4 * s - 3 * n * h) / n; out[4] = -2 * (1 - c) / n;
  // Y row
  out[7] = 4 - 3 * c; out[9] = 2 * (1 - c) / n; out[10] = s / n;
  // Z row
  out[14] = c; out[17] = s / n;
  // Xd row
  out[19] = 6 * n * (c - 1); out[21] = 4 * c - 3; out[22] = -2 * s;
  // Yd row
  out[25] = 3 * n * s; out[27] = 2 * s; out[28] = c;
  // Zd row
  out[32] = -n * s; out[35] = c;
  return out;
}

export function gamma(h, out = new Float64Array(18)) {
  const n = N, s = Math.sin(n * h), c = Math.cos(n * h), n2 = n * n;
  out.fill(0);
  // column 0: along-track acceleration
  out[0] = (4 / n2) * (1 - c) - 1.5 * h * h; out[3] = (2 / n2) * (n * h - s);
  out[9] = (4 / n) * s - 3 * h; out[12] = (2 / n) * (1 - c);
  // column 1: radial acceleration
  out[1] = -(2 / n2) * (n * h - s); out[4] = (1 - c) / n2;
  out[10] = -(2 / n) * (1 - c); out[13] = s / n;
  // column 2: cross-track acceleration
  out[8] = (1 - c) / n2; out[17] = s / n;
  return out;
}

const _t = new Float64Array(6);
// x+ = phi x + gamma (a + [ad, 0, 0]); P, G from phi(h), gamma(h) (cache them for a fixed step)
export function stepWith(P, G, x, a, ad = 0, out = x) {
  const a0 = (a ? a[0] : 0) + ad, a1 = a ? a[1] : 0, a2 = a ? a[2] : 0;
  for (let i = 0; i < 6; i++) {
    let v = 0;
    for (let j = 0; j < 6; j++) v += P[i * 6 + j] * x[j];
    _t[i] = v + G[i * 3] * a0 + G[i * 3 + 1] * a1 + G[i * 3 + 2] * a2;
  }
  out.set(_t);
  return out;
}

const _P = new Float64Array(36), _G = new Float64Array(18);
export function step(x, h, a = null, ad = 0, out = x) {
  return stepWith(phi(h, _P), gamma(h, _G), x, a, ad, out);
}

// free drift for t seconds under a constant differential drag ad (closed form)
export function propagate(x, t, ad = 0, out = new Float64Array(6)) {
  return stepWith(phi(t, _P), gamma(t, _G), x, null, ad, out);
}

// the natural CW acceleration at state x (what thrust must cancel to fly a straight line)
export function gCW(x, out = new Float64Array(3)) {
  out[0] = -2 * N * x[4];
  out[1] = 3 * N * N * x[1] + 2 * N * x[3];
  out[2] = -N * N * x[2];
  return out;
}

// time derivative (for RK4 cross-checks)
export function deriv(x, a = null, ad = 0, out = new Float64Array(6)) {
  gCW(x, _g);
  out[0] = x[3]; out[1] = x[4]; out[2] = x[5];
  out[3] = _g[0] + (a ? a[0] : 0) + ad; out[4] = _g[1] + (a ? a[1] : 0); out[5] = _g[2] + (a ? a[2] : 0);
  return out;
}
const _g = new Float64Array(3);

// Jacobi-like integral, conserved without drag or thrust
export const jacobi = (x) => x[3] * x[3] + x[4] * x[4] + x[5] * x[5] - 3 * N * N * x[1] * x[1] + N * N * x[2] * x[2];
