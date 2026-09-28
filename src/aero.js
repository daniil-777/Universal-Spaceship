// Flight in the air — atmospheric worlds only (env.atmosphere); the belt, Earth orbit and the low passes keep env.js's
// rate model. The ship moves through air that moves (src/weather.js + src/turbulence.js): lift, drag and side force
// come from the airspeed, the angle of attack α and the sideslip β; thrust from the throttle; gravity pulls down; the air
// thins with height (σ ≈ 0.55 at the ceiling). A fly-by-wire in the style of an airliner's normal law turns the pitch
// stick into a load-factor demand (neutral stick holds the flight path, also in a bank), the roll stick into a roll rate
// and the yaw stick into a sideslip; an α-protection keeps the stick from stalling the ship — a gust can still push α
// past the stall (16°): the lift falls and a wing drops. Beyond +3.5 / −1.5 g for 0.1 s the airframe is overstressed.
import { clamp, qIntegrate, qAxes, qInvRotate, qMul, qFromAxisAngle } from './mathx.js';
import { ENV } from './envconst.js';
import { SPAN } from './turbulence.js';

const DEG = Math.PI / 180;
export const AERO = Object.freeze({
  g: 9.8, K: 0.13, clAlpha: 4.0, alphaStall: 16 * DEG, clPost: 0.7, cd0: 0.05, kInd: 0.08, cdStall: 0.3, cyBeta: 1.0,
  tMax: 4.0, lapse: 0.7, yFloor: -26, rhoScale: 64,
  nMax: 3.5, nMin: -1.5, overTime: 0.1, nCmdMax: 2.5, nCmdMin: -1.0, nPerStick: 1.5, alphaProt: 0.9, bankMax: 67 * Math.PI / 180, betaMax: 8 * DEG,
  kAlpha: 6, kBeta: 4, vne: 24, vMax: 32, cruise: 14, atKp: 0.25, atKi: 0.08,
});
export const airDensity = (y) => Math.exp(-(y - AERO.yFloor) / AERO.rhoScale);   // σ = ρ/ρ₀: 1 on the valley floor
export function liftCoeff(alpha) {
  const A = AERO, a = Math.abs(alpha), sg = alpha < 0 ? -1 : 1;
  if (a <= A.alphaStall) return A.clAlpha * alpha;
  const past = Math.min(1, (a - A.alphaStall) / (8 * DEG)), cl = A.clAlpha * A.alphaStall * (1 - (1 - A.clPost) * past);   // down to 70 % over 8° past the stall …
  const fade = a > 45 * DEG ? Math.max(0, 1 - (a - 45 * DEG) / (45 * DEG)) : 1;                                        // … and to nothing at 90°
  return sg * cl * fade;
}
export const dragCoeff = (cl, stalled) => AERO.cd0 + AERO.kInd * cl * cl + (stalled ? AERO.cdStall : 0);
const field = () => ({ w: new Float64Array(3), sigma: 0, hA: 20, share: 0, Lc: 0, pg: 0 });   // the air at one point: mean wind, turbulence (weather.turbAt), mean-field rolling gust
export function newAirState() {
  return { V: 0, alpha: 0, beta: 0, n: 1, nSmooth: 1, var: 0, sigmaFelt: 0, stalled: false, stallSec: 0, overTime: 0, overstressed: false, rho: 1, sigma: 0, hAGL: 20,
    throttle: 0.42, climb: 0, wind: new Float64Array(3), gust: new Float64Array(3), atI: 0, commandN: 1, dev2: 0, phiHold: null,
    f0: field(), f1: field(), turb: { sigma: 0, hA: 20, share: 0, Lc: 0 }, pg: 0, wE: new Float64Array(3), gE: new Float64Array(3), lagOn: false };
}
export function autoThrottle(env, h, ff = 0.42) {                         // PI on airspeed around a feed-forward (the throttle that balances drag and climb); manual flight, the tunnel guide — never the learned pilot
  const air = env.air, e = (env.speedTarget || AERO.cruise) - air.V; air.atI = clamp(air.atI + e * h * AERO.atKi, -0.5, 0.5);
  return clamp(ff + AERO.atKp * e + air.atI, 0, 1);
}
export function rateToStick(env, frac) {                                 // a guard's pitch-rate wish (a fraction of rateMax) → the load-factor stick
  const V = Math.max(env.air.V, 5); return clamp(frac * ENV.ship.rateMax[2] * V / (AERO.g * AERO.nPerStick), -1, 1);
}
const _qa = new Float64Array(4), _tw = new Float64Array(3);
export const neutralLoad = (cb, cosG = 1) => (cb >= 0.4 ? cosG / Math.min(1, cb) : 2.5 * cosG * Math.max(0, cb) / 0.4);   // the neutral-stick load for bank cosine cb: 1/cos φ (level turns) to 66°, then fading to 0 g at 90° and upside down
export function trimAttitude(env) {                                      // the nose α_trim above the velocity: the neutral-stick load (1 g wings level, 1/cos φ in a bank)
  const s = env.ship, V = Math.hypot(s.v[0], s.v[1], s.v[2]); env.air.V = V; if (V < 1) return;
  const cb = s.u[1] / Math.max(0.2, Math.sqrt(Math.max(0, 1 - s.f[1] * s.f[1]))), cosPhi = clamp(cb, 0.4, 1);
  const q = AERO.K * airDensity(s.p[1]) * V * V, a = clamp(AERO.g * neutralLoad(cb) / Math.max(q * AERO.clAlpha, 1e-3), 0, AERO.alphaProt * AERO.alphaStall);
  qMul(s.q, qFromAxisAngle(0, 0, 1, a, _qa), s.q); qAxes(s.q, s.f, s.u, s.r); env.air.alpha = a;
  const uh = Math.hypot(s.u[0], s.u[2]), ac = AERO.g * Math.sqrt(Math.max(0, 1 / (cosPhi * cosPhi) - 1));   // in a bank the path already turns (g·tan φ toward the low wing):
  if (uh > 1e-3 && ac > 1e-6) { const ax = ac * s.u[0] / uh, az = ac * s.u[2] / uh, V2 = V * V;               // the nose turns with it (ω = v × a / V²), in body axes
    _tw[0] = (s.v[1] * az) / V2; _tw[1] = (s.v[2] * ax - s.v[0] * az) / V2; _tw[2] = (-s.v[1] * ax) / V2; qInvRotate(s.q, _tw, s.w); }
}

const TAU = 2 * Math.PI, _smooth = { h: NaN, n: 0, v: 0 };   // the nSmooth / var filter factors for the last substep length
const _va = new Float64Array(3), _vb = new Float64Array(3), _sf = new Float64Array(3), _ob = new Float64Array(3), _wa = new Float64Array(3), _wb = new Float64Array(3);
const QC = AERO.K * airDensity(5) * 14 * 14, TAU_R = 0.5, CHORD = 1.8;   // the dynamic pressure the gust rates were set at (cruise, y 5); the roll damping's time; the chord a gust builds up over
function sampleField(env, x, y, z, o) {                                 // the mean wind, the turbulence and the mean-field rolling gust at a point
  const W = env.weather;
  if (!W) { o.w.fill(0); o.sigma = 0; o.share = 0; o.Lc = 0; o.pg = 0; o.hA = env.hf ? y - env.hf.height(x, z) : y + ENV.yHalf; return o; }
  W.windAt(x, y, z, o.w); W.turbAt(x, y, z, o);
  const r = env.ship.r, u = env.ship.u, b = SPAN / 2; W.windAt(x - r[0] * b, y - r[1] * b, z - r[2] * b, _wa); W.windAt(x + r[0] * b, y + r[1] * b, z + r[2] * b, _wb);
  o.pg = ((_wa[0] - _wb[0]) * u[0] + (_wa[1] - _wb[1]) * u[1] + (_wa[2] - _wb[2]) * u[2]) / SPAN;   // air rising faster under the left wing rolls the right wing down (+p)
  return o;
}
export function stepAir(env, h, k = -1) {                                // k: the decision's substep — at 0 the air is sampled where the ship is and where it will be when the decision ends, and the substeps fly through the blend (k < 0: sampled at every call)
  const s = env.ship, air = env.air, A = AERO, c = env.cmd, S = ENV.ship, Tb = air.turb, f0 = air.f0;
  if (k <= 0) {
    sampleField(env, s.p[0], s.p[1], s.p[2], f0);
    if (k === 0) { const d = ENV.dt, W = env.weather, ahead = W && W.lookAhead; if (ahead) W.lookAhead(d); sampleField(env, s.p[0] + s.v[0] * d, s.p[1] + s.v[1] * d, s.p[2] + s.v[2] * d, air.f1); if (ahead) W.lookAhead(0); }   // the end: where the ship will be, in the sky as it will be
  }
  const f1 = k < 0 ? f0 : air.f1, t = k < 0 ? 0 : (k + 0.5) / ENV.substeps;
  for (let i = 0; i < 3; i++) air.wind[i] = f0.w[i] + t * (f1.w[i] - f0.w[i]);
  const sh = f0.share + t * (f1.share - f0.share), shL = f0.share * f0.Lc + t * (f1.share * f1.Lc - f0.share * f0.Lc);
  Tb.sigma = air.sigma = f0.sigma + t * (f1.sigma - f0.sigma); Tb.hA = air.hAGL = f0.hA + t * (f1.hA - f0.hA); Tb.share = sh; Tb.Lc = sh > 1e-9 ? shL / sh : 0; air.pg = f0.pg + t * (f1.pg - f0.pg);
  let hx = s.v[0] - air.wind[0], hz = s.v[2] - air.wind[2], hl = Math.sqrt(hx * hx + hz * hz);   // the gust axes: the ship's level track through the air
  if (hl < 1e-6) { hx = s.f[0]; hz = s.f[2]; hl = Math.sqrt(hx * hx + hz * hz); if (hl < 1e-6) { hx = 1; hz = 0; hl = 1; } }
  const Vg = Math.max(air.V, 5); env.dryden.step(h, Vg, Tb, hx / hl, hz / hl, air.gust);   // the gusts evolve every substep
  if (!air.lagOn) { air.wE.set(air.wind); air.gE.set(air.gust); air.lagOn = true; }   // the wing feels a change of the air as it spreads over the chord (penetration)
  else { const kp = 1 - Math.exp(-h * Vg / CHORD); for (let i = 0; i < 3; i++) { air.wE[i] += kp * (air.wind[i] - air.wE[i]); air.gE[i] += kp * (air.gust[i] - air.gE[i]); } }
  for (let i = 0; i < 3; i++) _va[i] = s.v[i] - air.wE[i] - air.gE[i];
  const Vraw = Math.sqrt(_va[0] * _va[0] + _va[1] * _va[1] + _va[2] * _va[2]), V = Math.max(0.5, Vraw); air.V = Vraw; air.climb = _va[1];
  qInvRotate(s.q, _va, _vb);                                            // the relative wind in body axes (x forward, y up, z right)
  const alpha = Vraw < 0.5 ? 0 : Math.atan2(-_vb[1], _vb[0]), beta = Vraw < 0.5 ? 0 : Math.asin(clamp(_vb[2] / V, -1, 1)); air.alpha = alpha; air.beta = beta;
  if (!air.stalled && Math.abs(alpha) > A.alphaStall) { air.stalled = true; s.w[0] += (env.rngAir() < 0.5 ? -1 : 1) * 0.8; }   // a wing drops
  else if (air.stalled && Math.abs(alpha) < 0.9 * A.alphaStall) air.stalled = false;
  if (air.stalled) air.stallSec += h;
  const rho = airDensity(s.p[1]), q = A.K * rho * V * V, cl = liftCoeff(alpha), cd = dragCoeff(cl, air.stalled); air.rho = rho;
  const vx = _va[0] / V, vy = _va[1] / V, vz = _va[2] / V, r = s.r;     // drag against the relative wind; lift ⟂ to it in the plane of symmetry (r × v̂); side force along v̂ × l̂
  let lx = r[1] * vz - r[2] * vy, ly = r[2] * vx - r[0] * vz, lz = r[0] * vy - r[1] * vx; const ll = Math.sqrt(lx * lx + ly * ly + lz * lz) || 1; lx /= ll; ly /= ll; lz /= ll;
  const sx = vy * lz - vz * ly, sy = vz * lx - vx * lz, sz = vx * ly - vy * lx;
  const tAvail = A.tMax * Math.exp(-A.lapse * (s.p[1] - A.yFloor) / A.rhoScale), D = q * cd;   // T_max σ^0.7
  air.throttle = env.autoThrottle ? autoThrottle(env, h, (D + A.g * vy) / Math.max(1e-3, tAvail)) : clamp(0.5 + 0.5 * c[3], 0, 1);
  const T = tAvail * air.throttle, L = q * cl, Y = -q * A.cyBeta * beta;
  _sf[0] = T * s.f[0] + L * lx - D * vx + Y * sx; _sf[1] = T * s.f[1] + L * ly - D * vy + Y * sy; _sf[2] = T * s.f[2] + L * lz - D * vz + Y * sz;   // specific force: what the ship feels
  const n = (_sf[0] * s.u[0] + _sf[1] * s.u[1] + _sf[2] * s.u[2]) / A.g; air.n = n;
  // fly-by-wire (normal law): stick → n demand → the α that delivers it (protected) → the nose rate that holds that α on a turning path
  const cosG = Math.sqrt(Math.max(0, 1 - vy * vy)), cb = s.u[1] / Math.max(0.2, Math.sqrt(Math.max(0, 1 - s.f[1] * s.f[1])));   // cb: cos of the bank
  const nCmd = clamp(neutralLoad(cb, cosG) + A.nPerStick * c[0], A.nCmdMin, A.nCmdMax); air.commandN = nCmd;   // level turns held up to 66° of bank (2.5 g); never pulling into the ground upside down
  const fl = s.f[0] * lx + s.f[1] * ly + s.f[2] * lz;                  // the thrust's share of the path-normal force: the wing supplies the rest
  const aProt = A.alphaProt * A.alphaStall, alphaCmd = clamp((nCmd * A.g - T * fl) / Math.max(q * A.clAlpha, 1e-3), -aProt, aProt);
  const wPath = (_sf[0] * lx + _sf[1] * ly + _sf[2] * lz - A.g * ly) / V, wSide = (_sf[0] * sx + _sf[1] * sy + _sf[2] * sz - A.g * sy) / V;   // how fast the relative wind turns (gravity included)
  _tw[0] = wPath * sx - wSide * lx; _tw[1] = wPath * sy - wSide * ly; _tw[2] = wPath * sz - wSide * lz; qInvRotate(s.q, _tw, _ob);   // the relative wind's turn (Ω = v̂ × a⊥ / V) in body axes: the rates that keep α and β
  const qCmd = clamp(_ob[2] + A.kAlpha * (alphaCmd - alpha), -S.rateMax[2], S.rateMax[2]);
  const rCmd = clamp(_ob[1] - A.kBeta * (beta - c[1] * A.betaMax), -S.rateMax[1], S.rateMax[1]);   // positive yaw = nose left, as in the rate model
  const phi = Math.atan2(-s.r[1], s.u[1]);                                     // bank (positive: right wing down)
  if (Math.abs(c[2]) > 0.05 || air.phiHold === null) air.phiHold = clamp(phi, -A.bankMax, A.bankMax);   // the roll stick commands a roll rate; at neutral the bank is held — never beyond 67°
  let pCmd = Math.abs(c[2]) > 0.05 ? c[2] * S.rateMax[0] : _ob[0] + 2 * (air.phiHold - phi - TAU * Math.round((air.phiHold - phi) / TAU));
  const room = A.bankMax - Math.abs(phi); if (pCmd * Math.sign(phi) > 2 * room) pCmd = 2 * room * Math.sign(phi);   // bank protection: rolling toward 67° slows, beyond it the ship rolls back
  pCmd = clamp(pCmd, -S.rateMax[0], S.rateMax[0]);
  const kRate = h / S.rateTau * (air.stalled ? 0.35 : 1);                      // a stalled wing answers sluggishly
  s.w[0] += kRate * (pCmd - s.w[0]); s.w[1] += kRate * (rCmd - s.w[1]); s.w[2] += kRate * (qCmd - s.w[2]);
  const qr = q / QC, gLat = air.gE[0] * r[0] + air.gE[1] * r[1] + air.gE[2] * r[2];   // turbulence rocks the wings: a sideways gust through the dihedral, the rolling gust (the air's spanwise gradient,
  s.w[0] += h * qr * (6 * gLat / V + (air.pg + env.dryden.p) / TAU_R); s.w[1] += h * qr * 2 * gLat / V;   // mean and turbulent) through the roll damping — both grow with the dynamic pressure
  qIntegrate(s.q, s.w, h); qAxes(s.q, s.f, s.u, s.r);
  s.v[0] += h * _sf[0]; s.v[1] += h * (_sf[1] - A.g); s.v[2] += h * _sf[2];
  const sp = Math.sqrt(s.v[0] * s.v[0] + s.v[1] * s.v[1] + s.v[2] * s.v[2]); if (sp > A.vMax) for (let i = 0; i < 3; i++) s.v[i] *= A.vMax / sp;
  if (n > A.nMax || n < A.nMin) air.overTime += h; else air.overTime = 0; if (air.overTime >= A.overTime) air.overstressed = true;   // latched: the airframe stays broken (newAirState at the next episode)
  if (_smooth.h !== h) { _smooth.h = h; _smooth.n = 1 - Math.exp(-h / 0.8); _smooth.v = 1 - Math.exp(-h); }
  air.nSmooth += (n - air.nSmooth) * _smooth.n; const dev = n - air.nSmooth; air.var += (dev * dev - air.var) * _smooth.v; air.sigmaFelt = Math.sqrt(air.var);
  air.dev2 += (n - 1) * (n - 1);
}
