// 10 Hz control (spec section 6). Translation: a = (v_des - v)/tau - g_CW per LVLH axis outside a deadband (burns:
// closed-loop trims with MIB-or-nothing pulses: pairs, then one aft/nose primary on X, verniers on Y and Z). Attitude: per-axis
// phase plane s = e + we|we|/(1.4 alpha), fire -sgn(s) when |s| > theta_db (20 % hysteresis), on-time |dw|/alpha in
// [MIB, 100 ms]; dw moves we to the target rate -sgn(s) min(w_max, w_lc + ks max(0, |s| - theta)) (w_lc: the slow
// limit-cycle rate of a hold, at least one MIB pulse); no pulse below one MIB; a rate deadband nulls faster drift.
// Errors are against the LVLH-referenced target: we = w - R(q)^T wL - w_ref (w_ref = 0: LVLH hold).
import { DEG, CYCLE, INERTIA, TRANS_SETS, LATERAL_SETS, VERNIER_TRANS } from './consts.js';
import { gCW } from './cw.js';
import { attError, omegaRel } from './rigid.js';
import { createJets, transOnTimes, attSelect, torqueOf, select6, F_REF } from './jets.js';
import { qInvRotate } from '../mathx.js';

// body axes [x roll, y yaw, z pitch]; alpha (rad/s^2) from the spec section 5 authorities
export const MODES = Object.freeze({
  P: Object.freeze({ db: 1 * DEG, wmax: 0.5 * DEG, wlc: 0.02 * DEG, ks: 0.5, rdb: [0.24 * DEG, 0.035 * DEG, 0.035 * DEG], alpha: [3.72 * DEG, 0.55 * DEG, 0.52 * DEG] }),
  V: Object.freeze({ db: 0.3 * DEG, wmax: 0.05 * DEG, wlc: 0.002 * DEG, ks: 0.05, rdb: [0.01 * DEG, 0.01 * DEG, 0.01 * DEG], alpha: [0.051 * DEG, 0.0061 * DEG, 0.0063 * DEG] }),
});
export const THETA_FINAL = 0.1 * DEG;
// burn completion per LVLH axis: 1.2 mm/s (the X trims are single primary pulses), Y and Z finish on verniers
export const BURN_TOL = 0.0012;
export const BURN_TOL_V = 0.0005;
// a vernier hold hands over to the primaries beyond its deadband + 0.7 deg, or 0.1 deg/s
export const ESCALATE = Object.freeze({ att: 0.7 * DEG, rate: 0.1 * DEG });
// requested pulses use the largest MIB the Monte Carlo draws (spec section 5), so they always fire in the cycle asked
export const MIB_REQ = Object.freeze({ P: 0.05, V: 0.1 });

const QREF = Float64Array.from([0, 0, 0, 1]);
export function createControl({ jets = createJets(), inertia = INERTIA } = {}) {
  const req = new Float64Array(jets.n), trans = new Float64Array(jets.n), att = new Float64Array(jets.n);
  const e = new Float64Array(3), we = new Float64Array(3), g = new Float64Array(3), aL = new Float64Array(3), aB = new Float64Array(3);
  const firing = [false, false, false], hot = [false, false, false], stats = { pulses: [0, 0, 0], escalations: 0, mode: 'V' };

  function burnTrim(xh, q, mass, vDes) {
    // LVLH velocity error -> body -> per body axis: primary pair, single primary (X), vernier (Y, Z), or nothing
    for (let i = 0; i < 3; i++) aL[i] = vDes[i] - xh[3 + i];
    qInvRotate(q, aL, aB);
    for (let ax = 0; ax < 3; ax++) {
      const dv = aB[ax], adv = Math.abs(dv), set = TRANS_SETS[2 * ax + (dv > 0 ? 0 : 1)].filter((i) => !jets.isFailed[i]);
      if (!set.length || adv < (ax === 0 ? BURN_TOL : BURN_TOL_V)) continue;
      const fOne = Math.abs(jets.F[set[0] * 3 + ax]), mib = MIB_REQ.P, pulse = (k) => (k * fOne * mib) / mass;
      let use = set, on = 0;
      if (ax === 0 && adv <= 2.2 * pulse(1)) { use = [set[0]]; on = adv > 0.5 * pulse(1) ? mib : 0; }
      // lateral (Y, Z) fine trims compare against the healthy pair's pulse, not the current set size, so a
      // pair degraded by a failed jet still routes a fine residual to the vernier instead of a lone primary
      else if (ax !== 0 && adv <= 0.5 * pulse(2)) {
        const v = VERNIER_TRANS[2 * (ax - 1) + (dv > 0 ? 0 : 1)], fv = Math.abs(jets.F[v * 3 + ax]);
        use = [v]; on = Math.min(CYCLE, (adv * mass) / fv); if (on < MIB_REQ.V) on = adv > (0.5 * fv * MIB_REQ.V) / mass ? MIB_REQ.V : 0;
      } else on = Math.max(mib, Math.min(CYCLE, (adv * mass) / (fOne * use.length)));
      for (const i of use) trans[i] += on;
    }
  }
  function track(xh, q, mass, cmd) {
    gCW(xh, g);
    const db = cmd.db ?? [0.005, 0.005], tau = cmd.tau ?? 5;
    for (let i = 0; i < 3; i++) {
      const dv = cmd.vDes[i] - xh[3 + i], lim = i === 0 ? db[0] : db[1];
      // hysteresis: start outside the deadband, keep firing down to 40 % of it
      if (Math.abs(dv) > lim) hot[i] = true; else if (Math.abs(dv) < 0.4 * lim) hot[i] = false;
      aL[i] = hot[i] ? dv / tau - g[i] : 0;
    }
    qInvRotate(q, aL, aB);
    transOnTimes(jets, aB, mass, trans);
  }
  function attitude(q, w, cmd, lop) {
    let mode = cmd.mode || 'V', esc = false;
    attError(q, cmd.qRef || QREF, e); omegaRel(q, w, we);
    if (mode === 'V' && !lop && (Math.max(...e.map(Math.abs)) > (cmd.thetaDb || MODES.V.db) + ESCALATE.att || Math.max(...we.map(Math.abs)) > ESCALATE.rate)) { mode = 'P'; esc = true; stats.escalations++; }
    stats.mode = mode;
    // escalated from a hold: primaries only bring the state back inside the vernier's reach (no primary limit cycle)
    const M = MODES[mode], th = lop ? (cmd.thetaDb ? Math.max(cmd.thetaDb, MODES.V.db) : MODES.P.db) : mode === 'V' && cmd.thetaDb ? cmd.thetaDb : M.db, tau = [0, 0, 0], mib = MIB_REQ[mode];
    const wlc = (i) => (esc ? 0 : Math.max(M.wlc, M.alpha[i] * mib)), rdb = (i) => (esc ? Math.min(M.rdb[i], 0.5 * ESCALATE.rate) : M.rdb[i]);
    for (let i = 0; i < 3; i++) {
      const a = M.alpha[i], s = e[i] + (we[i] * Math.abs(we[i])) / (1.4 * a);
      if (Math.abs(s) > th || (firing[i] && Math.abs(s) > 0.8 * th)) firing[i] = true; else firing[i] = false;
      let dw = 0;
      if (firing[i]) dw = -Math.sign(s) * Math.min(M.wmax, wlc(i) + M.ks * Math.max(0, Math.abs(s) - th)) - we[i];
      else if (Math.abs(we[i]) > rdb(i)) dw = -we[i];
      // outside the band round to the nearest pulse; in the hysteresis zone only whole pulses (no dithering)
      if (Math.abs(dw) < (Math.abs(s) > th ? 0.5 : 1) * a * mib) continue;
      const on = Math.min(CYCLE, Math.max(mib, Math.abs(dw) / a));
      tau[i] = (Math.sign(dw) * inertia[i] * a * on) / CYCLE;
      stats.pulses[i]++;
    }
    if (lop) return tau;
    // feed the translation jets' cross-torque forward, where it is at least half this mode's smallest pulse
    const tt = torqueOf(jets, trans);
    for (let i = 0; i < 3; i++) if (Math.abs(tt[i]) >= (0.5 * inertia[i] * M.alpha[i] * mib) / CYCLE) tau[i] -= tt[i];
    if (tau.some((v) => v !== 0)) attSelect(jets, mode, tau, att);
    return tau;
  }
  // a failed jet: in any cycle whose translation needs a Y or Z set that lost a jet ("lopsided"), the translation becomes
  // a body force and one joint 6-DOF select serves it with the attitude torque (wider deadband, no escalation); a lone
  // X jet stays on the fixed path (its torque is fed forward to force-free attitude pairs)
  const lopsided = LATERAL_SETS.filter((set) => set.some((i) => jets.isFailed[i]));
  function degradedForce(xh, q, mass, cmd, f) {
    f.fill(0);
    if (!cmd.vDes) return f;
    if (cmd.burn) {
      for (let i = 0; i < 3; i++) aL[i] = cmd.vDes[i] - xh[3 + i];
      qInvRotate(q, aL, aB);
      for (let ax = 0; ax < 3; ax++) f[ax] = Math.abs(aB[ax]) < BURN_TOL ? 0 : Math.max(-F_REF, Math.min(F_REF, (aB[ax] * mass) / CYCLE));
      return f;
    }
    trans.fill(0); track(xh, q, mass, cmd); trans.fill(0);
    for (let ax = 0; ax < 3; ax++) f[ax] = Math.max(-F_REF, Math.min(F_REF, aB[ax] * mass));
    return f;
  }
  const fB = new Float64Array(3);
  return {
    jets, stats, e, we,
    // xh: nav state; q, w: attitude and inertial body rate; cmd: { vDes, burn, tau, db, mode, thetaDb, qRef }
    update(xh, q, w, mass, cmd) {
      trans.fill(0); att.fill(0);
      if (cmd.vDes) { if (cmd.burn) burnTrim(xh, q, mass, cmd.vDes); else track(xh, q, mass, cmd); }
      const lop = lopsided.some((set) => set.some((i) => trans[i] > 0));
      if (lop) {
        trans.fill(0);
        const tau = attitude(q, w, cmd, true);
        select6(jets, degradedForce(xh, q, mass, cmd, fB), tau, att);
      } else attitude(q, w, cmd, false);
      for (let i = 0; i < jets.n; i++) req[i] = trans[i] + att[i];
      return req;
    },
    reset() { firing.fill(false); hot.fill(false); stats.pulses = [0, 0, 0]; stats.escalations = 0; },
  };
}
