// Finite-torque rigid body in LVLH (spec section 4), RK4 with renormalisation:
//   J wd = -w x Jw + tau + 3n^2 o x Jo,  o = R(q)^T (0, -1, 0)   (gravity gradient; opts.gg = false turns it off)
//   qd = 1/2 q (x) (w - R(q)^T wL),  wL = (0, 0, -n)
// q: body -> LVLH [x, y, z, w]; w: body rate relative to inertial (rad/s, body axes). J: 3x3 row-major.
import { N, INERTIA, OMEGA_L } from './consts.js';
import { qInvRotate, qMul, qNormalize } from '../mathx.js';

export function makeInertia(diag = INERTIA, products = [0, 0, 0]) {
  const [a, b, c] = diag, [xy, xz, yz] = products;
  const J = Float64Array.from([a, -xy, -xz, -xy, b, -yz, -xz, -yz, c]);
  const det = J[0] * (J[4] * J[8] - J[5] * J[7]) - J[1] * (J[3] * J[8] - J[5] * J[6]) + J[2] * (J[3] * J[7] - J[4] * J[6]);
  const Jinv = Float64Array.from([
    (J[4] * J[8] - J[5] * J[7]) / det, (J[2] * J[7] - J[1] * J[8]) / det, (J[1] * J[5] - J[2] * J[4]) / det,
    (J[5] * J[6] - J[3] * J[8]) / det, (J[0] * J[8] - J[2] * J[6]) / det, (J[2] * J[3] - J[0] * J[5]) / det,
    (J[3] * J[7] - J[4] * J[6]) / det, (J[1] * J[6] - J[0] * J[7]) / det, (J[0] * J[4] - J[1] * J[3]) / det,
  ]);
  return { J, Jinv };
}
export const NOMINAL_INERTIA = makeInertia();

const mul3 = (M, v, out) => { out[0] = M[0] * v[0] + M[1] * v[1] + M[2] * v[2]; out[1] = M[3] * v[0] + M[4] * v[1] + M[5] * v[2]; out[2] = M[6] * v[0] + M[7] * v[1] + M[8] * v[2]; return out; };
const _wL = new Float64Array(3);
// body rate relative to LVLH: w - R(q)^T wL
export function omegaRel(q, w, out = new Float64Array(3)) {
  qInvRotate(q, OMEGA_L, _wL);
  out[0] = w[0] - _wL[0]; out[1] = w[1] - _wL[1]; out[2] = w[2] - _wL[2];
  return out;
}

const NADIR = Float64Array.from([0, -1, 0]);
const _Jw = new Float64Array(3), _o = new Float64Array(3), _Jo = new Float64Array(3), _r = new Float64Array(3), _wr = new Float64Array(4), _qd = new Float64Array(4);
function derivs(q, w, inertia, torque, gg, n, dq, dw) {
  const { J, Jinv } = inertia;
  mul3(J, w, _Jw);
  _r[0] = -(w[1] * _Jw[2] - w[2] * _Jw[1]); _r[1] = -(w[2] * _Jw[0] - w[0] * _Jw[2]); _r[2] = -(w[0] * _Jw[1] - w[1] * _Jw[0]);
  if (torque) { _r[0] += torque[0]; _r[1] += torque[1]; _r[2] += torque[2]; }
  if (gg) {
    qInvRotate(q, NADIR, _o); mul3(J, _o, _Jo);
    const k = 3 * n * n;
    _r[0] += k * (_o[1] * _Jo[2] - _o[2] * _Jo[1]); _r[1] += k * (_o[2] * _Jo[0] - _o[0] * _Jo[2]); _r[2] += k * (_o[0] * _Jo[1] - _o[1] * _Jo[0]);
  }
  mul3(Jinv, _r, dw);
  qInvRotate(q, [0, 0, -n], _wL);
  _wr[0] = 0.5 * (w[0] - _wL[0]); _wr[1] = 0.5 * (w[1] - _wL[1]); _wr[2] = 0.5 * (w[2] - _wL[2]); _wr[3] = 0;
  qMul(q, _wr, _qd); dq.set(_qd);
}

const K = Array.from({ length: 4 }, () => ({ q: new Float64Array(4), w: new Float64Array(3) }));
const _q = new Float64Array(4), _w = new Float64Array(3);
// one RK4 step of h seconds; mutates q and w. opts: { inertia, torque (body N m, constant over h), gg, n }
export function rigidStep(q, w, h, { inertia = NOMINAL_INERTIA, torque = null, gg = true, n = N } = {}) {
  derivs(q, w, inertia, torque, gg, n, K[0].q, K[0].w);
  for (let s = 1; s < 4; s++) {
    const f = s === 3 ? h : h / 2, p = K[s - 1];
    for (let i = 0; i < 4; i++) _q[i] = q[i] + f * p.q[i];
    for (let i = 0; i < 3; i++) _w[i] = w[i] + f * p.w[i];
    derivs(_q, _w, inertia, torque, gg, n, K[s].q, K[s].w);
  }
  for (let i = 0; i < 4; i++) q[i] += (h / 6) * (K[0].q[i] + 2 * K[1].q[i] + 2 * K[2].q[i] + K[3].q[i]);
  for (let i = 0; i < 3; i++) w[i] += (h / 6) * (K[0].w[i] + 2 * K[1].w[i] + 2 * K[2].w[i] + K[3].w[i]);
  qNormalize(q);
}

export const energy = (inertia, w) => 0.5 * (w[0] * mul3(inertia.J, w, _Jw)[0] + w[1] * _Jw[1] + w[2] * _Jw[2]);
export const momentum = (inertia, w) => Math.hypot(...mul3(inertia.J, w, _Jw));

// attitude error of q against the reference qRef as a rotation vector in body axes (rad)
const _qi = new Float64Array(4), _qe = new Float64Array(4);
export function attError(q, qRef, out = new Float64Array(3)) {
  _qi[0] = -qRef[0]; _qi[1] = -qRef[1]; _qi[2] = -qRef[2]; _qi[3] = qRef[3];
  qMul(_qi, q, _qe);
  const sgn = _qe[3] < 0 ? -1 : 1, v = Math.hypot(_qe[0], _qe[1], _qe[2]);
  const k = v < 1e-12 ? 2 * sgn : (2 * Math.atan2(v, Math.abs(_qe[3])) / v) * sgn;
  out[0] = k * _qe[0]; out[1] = k * _qe[1]; out[2] = k * _qe[2];
  return out;
}
