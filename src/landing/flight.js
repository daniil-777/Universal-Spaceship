// The landing scenario's flight dynamics (SI, runway frame: x along the runway, y up, z right): a rigid body under the
// air's forces (Polhamus lift with ground effect, drag build-up, side force), two engines with spool lag, gravity, three
// spring–damper landing-gear legs with tyres (rolling, braking, cornering), ground spoilers and a drag chute. Moments come
// from a fly-by-wire that tracks rate commands with the control power the dynamic pressure allows (C*-like pitch: a
// load-factor demand → α, protected at 20°; bank and sideslip holds; a pitch-rate mode for derotation), from the gear and
// the chute. Semi-implicit Euler at 120 Hz — the gear's heave and pitch modes are ~10 rad/s.
import { qIntegrate, qAxes, qInvRotate, qRotate, qFromAxisAngle, qMul, qIdentity, clamp } from '../mathx.js';
import { VEH, G, FT, liftCoeff, alphaForLift, inducedDrag, groundEffect, isaDensity } from './vehicle.js';

const K_ALPHA = 2.5, TAU_C = 0.25, K_PHI = 1.2, P_MAX = 0.26, K_BETA = 2.0, CN_BETA = 0.08;
export const steerLimit = (V) => VEH.steerMax + (60 * Math.PI / 180 - VEH.steerMax) * clamp((30 - V) / 25, 0, 1);   // the tiller's authority grows as the ship slows: 8° above 60 kt, 60° at taxi speed
export function newFlight() {
  return { p: new Float64Array(3), v: new Float64Array(3), q: qIdentity(), w: new Float64Array(3), f: new Float64Array(3), u: new Float64Array(3), r: new Float64Array(3),
    spool: VEH.idle, gear: 1, spoil: 0, chute: 0, chuteOpen: 0, brake: 0, t: 0,        // chute: 0 packed, 1 streaming, 2 jettisoned
    air: { V: 0, alpha: 0, beta: 0, rho: 1.225, qbar: 0, CL: 0, CDi: 0, n: 1, gamma: 0, phi: 0, theta: 0, psi: 0, hRA: 0, gs: 0 },
    legs: VEH.legs.map(() => ({ comp: 0, rate: 0, force: 0, onGround: false })), wow: false, noseOn: false,
    gearG: 1, touchdown: null, tailStrike: false, trimThr: 0, trimThrust: 0 };
}
const _b = new Float64Array(3), _o = new Float64Array(3);
export function bodyToWorld(s, b, out = new Float64Array(3)) { _b[0] = b[0]; _b[1] = b[1]; _b[2] = b[2]; qRotate(s.q, _b, out); out[0] += s.p[0]; out[1] += s.p[1]; out[2] += s.p[2]; return out; }
function airData(s, V, rho) {                              // what the instruments read before the first step
  const a = s.air; a.V = V; a.rho = rho; a.qbar = 0.5 * rho * V * V; a.gs = Math.hypot(s.v[0], s.v[2]); a.gamma = Math.asin(clamp(s.v[1] / Math.max(0.5, Math.hypot(s.v[0], s.v[1], s.v[2])), -1, 1));
  a.hRA = Math.min(bodyToWorld(s, VEH.legs[1].contact, _o)[1], bodyToWorld(s, VEH.legs[2].contact, _o)[1]); attitude(s);
}
function attitude(s) {                                    // pitch, bank (right wing down +), heading (toward +z = right, +)
  const a = s.air; a.theta = Math.asin(clamp(s.f[1], -1, 1)); a.phi = Math.atan2(-s.r[1], s.u[1]); a.psi = Math.atan2(s.f[2], s.f[0]);
}

const _w = new Float64Array(3), _va = new Float64Array(3), _vb = new Float64Array(3), _F = new Float64Array(3), _M = new Float64Array(3), _rc = new Float64Array(3), _pc = new Float64Array(3);
const _ww = new Float64Array(3), _tw = new Float64Array(3), _ob = new Float64Array(3), _mb = new Float64Array(3), _sf = new Float64Array(3), _wb = new Float64Array(3);
const cross = (a, b, o) => { const x = a[1] * b[2] - a[2] * b[1], y = a[2] * b[0] - a[0] * b[2], z = a[0] * b[1] - a[1] * b[0]; o[0] = x; o[1] = y; o[2] = z; return o; };

export function stepFlight(s, cmd, wind, h) {
  const A = s.air, m = VEH.mass, W = m * G;
  wind.at(s.p, s.t, _w);
  for (let i = 0; i < 3; i++) _va[i] = s.v[i] - _w[i];
  const Vraw = Math.hypot(_va[0], _va[1], _va[2]), V = Math.max(1, Vraw); qInvRotate(s.q, _va, _vb);
  const alpha = Vraw < 1 ? 0 : Math.atan2(-_vb[1], _vb[0]), beta = Vraw < 1 ? 0 : Math.asin(clamp(_vb[2] / V, -1, 1));
  const rho = isaDensity(Math.max(0, s.p[1])), qbar = 0.5 * rho * Vraw * Vraw; A.V = Vraw; A.alpha = alpha; A.beta = beta; A.rho = rho; A.qbar = qbar;
  const ge = groundEffect(bodyToWorld(s, VEH.wing, _pc)[1]), dump = 1 - VEH.liftDump * s.spoil, clB = liftCoeff(alpha);
  const CL = clB * ge[0] * dump, CDi = inducedDrag(clB, alpha) * ge[1], CD = VEH.cd0 + VEH.cdGear * s.gear + VEH.cdSpoil * s.spoil + CDi; A.CL = CL; A.CDi = CDi;
  const vx = _va[0] / V, vy = _va[1] / V, vz = _va[2] / V, r = s.r;
  let lx = r[1] * vz - r[2] * vy, ly = r[2] * vx - r[0] * vz, lz = r[0] * vy - r[1] * vx; const ll = Math.hypot(lx, ly, lz) || 1; lx /= ll; ly /= ll; lz /= ll;   // lift ⟂ the relative wind, in the plane of symmetry
  const sx = vy * lz - vz * ly, sy = vz * lx - vx * lz, sz = vx * ly - vy * lx;
  const T = VEH.thrMax * (rho / 1.225) ** 0.7 * s.spool, L = qbar * VEH.S * CL, D = qbar * VEH.S * CD, Y = -qbar * VEH.S * VEH.cyBeta * beta;
  const Dc = s.chute === 1 ? qbar * VEH.chuteCdS * s.chuteOpen : 0;
  _sf[0] = (T * s.f[0] + L * lx - D * vx + Y * sx) / m; _sf[1] = (T * s.f[1] + L * ly - D * vy + Y * sy) / m; _sf[2] = (T * s.f[2] + L * lz - D * vz + Y * sz) / m;   // air + thrust, per unit mass
  _F[0] = _sf[0] * m - Dc * vx; _F[1] = _sf[1] * m - W - Dc * vy; _F[2] = _sf[2] * m - Dc * vz; _M.fill(0);
  if (Dc > 0) { qRotate(s.q, VEH.tail, _rc); _tw[0] = -Dc * vx; _tw[1] = -Dc * vy; _tw[2] = -Dc * vz; cross(_rc, _tw, _o); _M[0] += _o[0]; _M[1] += _o[1]; _M[2] += _o[2]; }
  // the gear: each leg a spring–damper on the ground, tyres rolling, braking and cornering
  qRotate(s.q, s.w, _ww); let mainTouch = null, wow = false;
  const psi = Math.atan2(s.f[2], s.f[0]);
  for (let k = 0; k < 3; k++) {
    const G_ = VEH.legs[k], lg = s.legs[k]; lg.force = 0; lg.onGround = false; lg.comp = 0;
    if (s.gear < 0.99) continue;
    qRotate(s.q, G_.contact, _rc); const py = s.p[1] + _rc[1]; if (py >= 0) continue;
    cross(_ww, _rc, _pc); const cvx = s.v[0] + _pc[0], cvy = s.v[1] + _pc[1], cvz = s.v[2] + _pc[2], comp = -py, rate = -cvy;
    const N = Math.max(0, G_.k * Math.min(comp, G_.stroke) + (rate > 0 ? G_.c : G_.rebound * G_.c) * rate + (comp > G_.stroke ? 20 * G_.k * (comp - G_.stroke) : 0));
    const sl = steerLimit(Math.hypot(s.v[0], s.v[2])), hw = psi + (G_.steer ? clamp(cmd.steer || 0, -sl, sl) : 0), fx = Math.cos(hw), fz = Math.sin(hw);
    const vf = cvx * fx + cvz * fz, vl = -cvx * fz + cvz * fx, slip = Math.atan2(vl, Math.max(Math.abs(vf), 0.5));
    const Fl = -N * (VEH.muRoll + (G_.brake ? VEH.muBrake * s.brake : 0)) * Math.tanh(vf / 0.3), Fs = -N * VEH.muSide * Math.tanh(VEH.cornering * slip / VEH.muSide);
    _tw[0] = Fl * fx - Fs * fz; _tw[1] = N; _tw[2] = Fl * fz + Fs * fx; _F[0] += _tw[0]; _F[1] += _tw[1]; _F[2] += _tw[2];
    cross(_rc, _tw, _o); _M[0] += _o[0]; _M[1] += _o[1]; _M[2] += _o[2];
    lg.force = N; lg.onGround = true; lg.comp = comp; lg.rate = rate;
    if (G_.brake) { wow = true; if (!s.wow && !s.touchdown && mainTouch === null) mainTouch = rate; }
  }
  s.noseOn = s.legs[0].onGround;
  // fly-by-wire: rate commands → control moments, as much as the dynamic pressure allows
  attitude(s);
  const wPath = (_sf[0] * lx + _sf[1] * ly + _sf[2] * lz - G * ly) / V, wSide = (_sf[0] * sx + _sf[1] * sy + _sf[2] * sz - G * sy) / V;
  _tw[0] = wPath * sx - wSide * lx; _tw[1] = wPath * sy - wSide * ly; _tw[2] = wPath * sz - wSide * lz; qInvRotate(s.q, _tw, _ob);   // the relative wind's turn rate, body axes
  let qCmd;
  if (Number.isFinite(cmd.q)) qCmd = cmd.q;
  else {
    const fl = s.f[0] * lx + s.f[1] * ly + s.f[2] * lz, need = (cmd.nz * W - T * fl) / Math.max(qbar * VEH.S * ge[0] * dump, 1);
    const aCmd = clamp(alphaForLift(need), -5 * Math.PI / 180, VEH.alphaMax); qCmd = _ob[2] + K_ALPHA * (aCmd - alpha);
    if (alpha > VEH.alphaMax - 0.02) qCmd = Math.min(qCmd, _ob[2] + 6 * (VEH.alphaMax - alpha));   // the α-floor: never through the limit
  }
  const pCmd = clamp(_ob[0] + K_PHI * ((wow ? 0 : cmd.phi || 0) - A.phi), -P_MAX, P_MAX), rCmd = _ob[1] - K_BETA * (beta - (cmd.beta || 0));
  qInvRotate(s.q, _M, _mb);                               // gear + chute moments, body axes
  const I = VEH.I, qsb = qbar * VEH.S * VEH.span, qsc = qbar * VEH.S * VEH.chord, w = s.w;
  _mb[0] += clamp(I[0] * (pCmd - w[0]) / TAU_C, -qsb * VEH.ctrl[0], qsb * VEH.ctrl[0]);
  _mb[2] += clamp(I[2] * (qCmd - w[2]) / TAU_C, -qsc * VEH.ctrl[2], qsc * VEH.ctrl[2]);
  if (wow) _mb[1] += -qsb * (VEH.ctrl[1] * clamp((cmd.steer || 0) / VEH.steerMax, -1, 1) + CN_BETA * beta);   // on the ground: rudder with the steering, the fin weathervanes
  else _mb[1] += clamp(I[1] * (rCmd - w[1]) / TAU_C, -qsb * VEH.ctrl[1], qsb * VEH.ctrl[1]);
  _wb[0] = w[1] * I[2] * w[2] - w[2] * I[1] * w[1]; _wb[1] = w[2] * I[0] * w[0] - w[0] * I[2] * w[2]; _wb[2] = w[0] * I[1] * w[1] - w[1] * I[0] * w[0];   // ω × Iω
  for (let i = 0; i < 3; i++) w[i] += h * (_mb[i] - _wb[i]) / I[i];
  for (let i = 0; i < 3; i++) { s.v[i] += h * _F[i] / m; s.p[i] += h * s.v[i]; }
  qIntegrate(s.q, w, h); qAxes(s.q, s.f, s.u, s.r);
  // systems: engines spool, gear travels (8 s), ground spoilers deploy on weight-on-wheels when armed, chute
  const thr = clamp(cmd.thr ?? s.spool, VEH.idle, 1); s.spool += (thr - s.spool) * (1 - Math.exp(-h / (thr > s.spool ? VEH.spoolUp : VEH.spoolDown)));
  s.gear = clamp(s.gear + clamp((cmd.gear ?? 1) - s.gear, -h / 8, h / 8), 0, 1);
  s.spoil = clamp(s.spoil + (cmd.spoil && wow ? h : -h), 0, 1); s.brake = clamp(cmd.brake || 0, 0, 1);
  if (cmd.chute === 1 && s.chute === 0 && wow) s.chute = 1; else if (cmd.chute === 2 && s.chute === 1) s.chute = 2;
  if (s.chute === 1) s.chuteOpen = Math.min(1, s.chuteOpen + h / 1.5);
  if (mainTouch !== null) {                               // the first main-wheel contact: the landing's numbers
    const mid = bodyToWorld(s, [-2.5, -6.6, 0], _pc);
    s.touchdown = { t: s.t, sinkFps: mainTouch / FT, x: mid[0], z: mid[2], V: Vraw, gs: Math.hypot(s.v[0], s.v[2]), pitch: A.theta, bank: A.phi, crab: A.psi, alpha };
  }
  s.wow = wow; s.gearG = (_F[0] * s.u[0] + (_F[1] + W) * s.u[1] + _F[2] * s.u[2]) / W; A.n = s.gearG;
  A.gamma = Math.asin(clamp(s.v[1] / Math.max(0.5, Math.hypot(s.v[0], s.v[1], s.v[2])), -1, 1)); A.gs = Math.hypot(s.v[0], s.v[2]);
  A.hRA = Math.min(bodyToWorld(s, VEH.legs[1].contact, _pc)[1], bodyToWorld(s, VEH.legs[2].contact, _o)[1]);
  if (bodyToWorld(s, VEH.tail, _pc)[1] < 0) s.tailStrike = true;
  s.t += h; return s;
}

export function trimFlight(s, o) {                        // o: { x, z, h (main-gear height), V, gamma, gear, alpha?, heading, onGround }
  const W = VEH.mass * G, hd = o.heading || 0; s.t = 0; s.w.fill(0); s.touchdown = null; s.tailStrike = false; s.chute = 0; s.chuteOpen = 0; s.spoil = 0; s.gear = o.gear ?? 1;
  if (o.onGround) {
    const V = o.V || 0; qFromAxisAngle(0, 1, 0, -hd, s.q); qAxes(s.q, s.f, s.u, s.r);
    s.p[0] = o.x; s.p[1] = 6.6 - W * 0.844 / 2 / VEH.legs[1].k; s.p[2] = o.z || 0; s.v[0] = V * Math.cos(hd); s.v[1] = 0; s.v[2] = V * Math.sin(hd);
    s.spool = VEH.idle; s.gear = 1; s.wow = true; s.touchdown = { t: 0, sinkFps: 0, x: o.x, z: 0, V, gs: V, pitch: 0, bank: 0, crab: 0, alpha: 0 }; airData(s, V, 1.225); return s;
  }
  const V = o.V, gam = o.gamma || 0; let a = o.alpha ?? 10 * Math.PI / 180, T = 100e3, py = o.h + 6.6;
  for (let it = 0; it < 40; it++) {
    const th = gam + a; py = o.h + 2.5 * Math.sin(th) + 6.6 * Math.cos(th);   // main gear at o.h
    const rho = isaDensity(py), qbar = 0.5 * rho * V * V, ge = groundEffect(py - 1.0 * Math.cos(th) - 1.0 * Math.sin(th));
    if (o.alpha == null) a = alphaForLift((W * Math.cos(gam) - T * Math.sin(a)) / (qbar * VEH.S * ge[0]));
    const D = qbar * VEH.S * (VEH.cd0 + VEH.cdGear * s.gear + inducedDrag(liftCoeff(a), a) * ge[1]); T = (D + W * Math.sin(gam)) / Math.cos(a);
    s.trimThr = clamp(T / (VEH.thrMax * (rho / 1.225) ** 0.7), VEH.idle, 1); s.trimThrust = T;
  }
  qFromAxisAngle(0, 0, 1, gam + a, s.q); const qh = qFromAxisAngle(0, 1, 0, -hd, new Float64Array(4)); qMul(qh, s.q, s.q); qAxes(s.q, s.f, s.u, s.r);
  s.p[0] = o.x; s.p[1] = py; s.p[2] = o.z || 0; s.v[0] = V * Math.cos(gam) * Math.cos(hd); s.v[1] = V * Math.sin(gam); s.v[2] = V * Math.cos(gam) * Math.sin(hd);
  s.spool = s.trimThr; s.wow = false; s.air.alpha = a; airData(s, V, isaDensity(py)); return s;
}
