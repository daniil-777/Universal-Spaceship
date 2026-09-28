// The 22 RCS jets (spec section 5): per-jet force, torque and mass flow (with MC dispersions), the 10 ms timing
// quantum with minimum-impulse-bit carry-over, the fixed translation sets and the attitude jet select (bounded least
// squares over the mode's jets, the L1 term weighted by mass flow).
import { JETS, KIND, G0, QUANTUM, CYCLE, TRANS_SETS, ATT_JETS } from './consts.js';

// opts: thrustScale[22], dirTilt[22] (small rotation vectors, rad), com [3] (m), failed [indices], mib { P, V } (s)
export function createJets({ thrustScale = null, dirTilt = null, com = [0, 0, 0], failed = [], mib = null } = {}) {
  const n = JETS.length, F = new Float64Array(n * 3), Tq = new Float64Array(n * 3), mdot = new Float64Array(n);
  const mibOf = new Float64Array(n), isFailed = new Uint8Array(n), carry = new Float64Array(n);
  JETS.forEach((j, i) => {
    const k = KIND[j.kind], T = k.thrust * (thrustScale ? thrustScale[i] : 1);
    let d = j.d;
    if (dirTilt) {
      const t = dirTilt[i];
      d = [d[0] + t[1] * d[2] - t[2] * d[1], d[1] + t[2] * d[0] - t[0] * d[2], d[2] + t[0] * d[1] - t[1] * d[0]];
      const l = Math.hypot(...d); d = d.map((v) => v / l);
    }
    const f = [-T * d[0], -T * d[1], -T * d[2]], r = [j.pos[0] - com[0], j.pos[1] - com[1], j.pos[2] - com[2]];
    F.set(f, i * 3);
    Tq.set([r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]], i * 3);
    mdot[i] = T / (k.isp * G0);
    mibOf[i] = mib ? mib[j.kind] : k.mib;
  });
  for (const i of failed) isFailed[i] = 1;

  // requested on-times (s) -> fired on-times: 10 ms quantum, >= MIB or carried to the next cycle, <= CYCLE, failed = 0
  function schedule(req, out = new Float64Array(n)) {
    for (let i = 0; i < n; i++) {
      if (isFailed[i] || !(req[i] > 0)) { out[i] = 0; carry[i] = 0; continue; }
      const total = req[i] + carry[i], on = Math.min(CYCLE, Math.round(total / QUANTUM) * QUANTUM);
      if (on + 1e-12 < mibOf[i]) { out[i] = 0; carry[i] = total; continue; }
      out[i] = on; carry[i] = Math.min(CYCLE, Math.max(0, total - on));
    }
    return out;
  }
  // body force (N), torque (N m) and mass flow (kg/s) of the jets on at time tc (s) into the cycle
  function active(onTimes, tc, outF, outT) {
    outF.fill(0); outT.fill(0); let md = 0;
    for (let i = 0; i < n; i++) {
      if (!(onTimes[i] > tc + 1e-9)) continue;
      for (let a = 0; a < 3; a++) { outF[a] += F[i * 3 + a]; outT[a] += Tq[i * 3 + a]; }
      md += mdot[i];
    }
    return md;
  }
  // total impulse (N s), angular impulse (N m s) and propellant (kg) of a set of on-times
  function impulse(onTimes) {
    const I = [0, 0, 0], L = [0, 0, 0]; let dm = 0;
    for (let i = 0; i < n; i++) {
      const t = onTimes[i]; if (!(t > 0)) continue;
      for (let a = 0; a < 3; a++) { I[a] += F[i * 3 + a] * t; L[a] += Tq[i * 3 + a] * t; }
      dm += mdot[i] * t;
    }
    return { I, L, dm };
  }
  return { n, F, Tq, mdot, mibOf, isFailed, carry, schedule, active, impulse, reset() { carry.fill(0); } };
}

// Translation: desired body acceleration aBody (m/s^2, average over the cycle) -> requested on-times per jet
export function transOnTimes(jets, aBody, mass, out) {
  for (let ax = 0; ax < 3; ax++) {
    const a = aBody[ax];
    if (a === 0) continue;
    const set = TRANS_SETS[2 * ax + (a > 0 ? 0 : 1)].filter((i) => !jets.isFailed[i]);
    if (!set.length) continue;
    let aFull = 0;
    for (const i of set) aFull += Math.abs(jets.F[i * 3 + ax]) / mass;
    const t = Math.min(CYCLE, (Math.abs(a) / aFull) * CYCLE);
    for (const i of set) out[i] += t;
  }
  return out;
}

// the average torque (N m) over a cycle of a set of requested on-times
export function torqueOf(jets, onTimes, out = new Float64Array(3)) {
  out.fill(0);
  for (let i = 0; i < jets.n; i++) {
    const t = onTimes[i]; if (!(t > 0)) continue;
    for (let a = 0; a < 3; a++) out[a] += (jets.Tq[i * 3 + a] * t) / CYCLE;
  }
  return out;
}

// per-axis torque authority of a mode (N m): the largest achievable |torque| on each axis with u in [0, 1]
export function authority(jets, mode) {
  const out = [0, 0, 0];
  for (let a = 0; a < 3; a++) {
    let pos = 0, neg = 0;
    for (const i of ATT_JETS[mode]) { const t = jets.Tq[i * 3 + a]; if (t > 0) pos += t; else neg -= t; }
    out[a] = Math.min(pos, neg);
  }
  return out;
}

// Attitude jet select: min |W (B u - tau)|^2 + lambda sum mdot_j u_j over 0 <= u <= 1 (projected coordinate descent).
// tau: average torque wanted over the cycle (N m). Adds u * CYCLE to out for the mode's jets.
export function attSelect(jets, mode, tau, out, { lambda = 1e-3, sweeps = 40 } = {}) {
  const J = ATT_JETS[mode].filter((i) => !jets.isFailed[i]), k = J.length, auth = authority(jets, mode);
  const w2 = auth.map((t) => 1 / Math.max(t, 1e-9) ** 2), u = new Float64Array(k), r = [-tau[0], -tau[1], -tau[2]];
  const H = J.map((i) => 2 * (w2[0] * jets.Tq[i * 3] ** 2 + w2[1] * jets.Tq[i * 3 + 1] ** 2 + w2[2] * jets.Tq[i * 3 + 2] ** 2));
  const mdMax = Math.max(...J.map((i) => jets.mdot[i]));
  for (let s = 0; s < sweeps; s++) {
    for (let c = 0; c < k; c++) {
      const i = J[c], b0 = jets.Tq[i * 3], b1 = jets.Tq[i * 3 + 1], b2 = jets.Tq[i * 3 + 2];
      const g = 2 * (w2[0] * b0 * r[0] + w2[1] * b1 * r[1] + w2[2] * b2 * r[2]) + lambda * (jets.mdot[i] / mdMax);
      const nu = Math.min(1, Math.max(0, u[c] - g / H[c])), du = nu - u[c];
      if (du === 0) continue;
      u[c] = nu; r[0] += b0 * du; r[1] += b1 * du; r[2] += b2 * du;
    }
  }
  for (let c = 0; c < k; c++) if (u[c] > 1e-6) out[J[c]] += u[c] * CYCLE;
  return out;
}

// Failed-jet fallback: one joint 6-DOF select (the fixed sets lose their torque balance when a jet of a pair is out),
// over the working attitude jets (P1-P12, V1-V6) plus the translation sets in the requested directions only.
// min |(F u - f)/F_REF|^2 + |W_T (T u - tau)|^2 + lambda mdot u, 0 <= u <= 1; torque first (W_T).
export const F_REF = 7740;
export function select6(jets, f, tau, out, { wT = 10, lambda = 0.01, sweeps = 60 } = {}) {
  const auth = authority(jets, 'P').map((t) => wT / Math.max(t, 1)), pool = new Set([...ATT_JETS.P, ...ATT_JETS.V]);
  for (let ax = 0; ax < 3; ax++) if (f[ax] !== 0) for (const i of TRANS_SETS[2 * ax + (f[ax] > 0 ? 0 : 1)]) pool.add(i);
  const J = [...pool].filter((i) => !jets.isFailed[i]).sort((a, b) => a - b);
  const W = [1 / F_REF, 1 / F_REF, 1 / F_REF, auth[0], auth[1], auth[2]], w2 = W.map((v) => v * v);
  const col = (i) => [jets.F[i * 3], jets.F[i * 3 + 1], jets.F[i * 3 + 2], jets.Tq[i * 3], jets.Tq[i * 3 + 1], jets.Tq[i * 3 + 2]];
  const B = J.map(col), H = B.map((b) => 2 * b.reduce((s, v, k) => s + w2[k] * v * v, 0)), u = new Float64Array(J.length);
  const r = [-f[0], -f[1], -f[2], -tau[0], -tau[1], -tau[2]], mdMax = Math.max(...J.map((i) => jets.mdot[i]));
  for (let s = 0; s < sweeps; s++) {
    for (let c = 0; c < J.length; c++) {
      const b = B[c]; let g = lambda * (jets.mdot[J[c]] / mdMax);
      for (let k = 0; k < 6; k++) g += 2 * w2[k] * b[k] * r[k];
      const nu = Math.min(1, Math.max(0, u[c] - g / H[c])), du = nu - u[c];
      if (du === 0) continue;
      u[c] = nu; for (let k = 0; k < 6; k++) r[k] += b[k] * du;
    }
  }
  for (let c = 0; c < J.length; c++) if (u[c] > 1e-6) out[J[c]] += u[c] * CYCLE;
  return out;
}
