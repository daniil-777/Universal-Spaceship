// 1 Hz guidance (spec sections 3 and 6): TRANSFER (departure, MCCs at T/3 and 2T/3, arrival) -> H1 ACQUISITION
// (X, Y box-kept while Z swings to 0, then the Zd null) -> H1 (>= 300 s, GO) -> CORRIDOR -> H2 (60 s, GO) -> FINAL
// -> contact (judged by the sim). BREAKOUT pre-empts any phase; from H1 acquisition to the 3 m commit point a Rule-P
// breakout must exist (Rule A), checked every 10 s. Output: the velocity command for the filter and the control.
import { DEG, H1_POINT, H2_RHO, GO, KOS_R, FINAL_SPEED, RULE_P_TRANSFER, BREAKOUT_V, SHIP_PORT, LATERAL_SETS, JET_INDEX, portRel, inCone } from './consts.js';
import { planTransfer, planMcc, planBreakout, ruleP, BREAKOUT_GRID, POSIGRADE_GRID } from './passive.js';
import { propagate } from './cw.js';
import { THETA_FINAL, BURN_TOL, BURN_TOL_V } from './control.js';
import { attError, omegaRel } from './rigid.js';
import { qRotate } from '../mathx.js';

export const PH = Object.freeze({ TRANSFER: 'TRANSFER', H1ACQ: 'H1 ACQ', H1: 'H1', CORRIDOR: 'CORRIDOR', H2: 'H2', FINAL: 'FINAL', BREAKOUT: 'BREAKOUT', DEPART: 'DEPART' });
export const RULE_A_PHASES = Object.freeze([PH.H1ACQ, PH.H1, PH.CORRIDOR, PH.H2, PH.FINAL]);
export const CORRIDOR_PHASES = Object.freeze([PH.CORRIDOR, PH.H2, PH.FINAL]);
// planning margin over Rule P for the departure and each MCC: in planning's Monte Carlo the executed state landed
// <= 5 m below its plan (nav velocity, +-1.2 mm/s trims); 8 m still lets the spec's nominal hop (249.2 m) fly
export const PLAN_MARGIN = Object.freeze({ depart: 8, mcc: 8 });
export const COAST_DB = 1 * DEG;
export const GUID = Object.freeze({ h1Hold: 300, h2Hold: 60, noGoMax: 1800, boEvery: 10, commit: 3, burnMax: 900, boxH1: 3, boxH2: 0.5 });
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const QREF = [0, 0, 0, 1];
// the far-start DEPART is a radial -Y burn (planned radial dv -0.30 to -0.92 m/s over the whole T search, never +Y).
// P1 is the only -Y primary at the nose: without it just the 107 N vernier V1 balances the aft -Y thrust, so the
// DEPART flies lopsided for minutes (155-276 s) and misses Rule P. A lost P3 is replaced by the aft P9/P11 pair
// (DEPART 5-17 s, Rule P kept), so it flies the transfer like any other failed lateral jet.
const DEPART_CRITICAL = Object.freeze([JET_INDEX.P1]);
// the same principle for the breakout from H1 (the NO-GO with a failed lateral jet): the plan's first-passing v+
// (0.02, 0.10, 0) flies lopsided for ~50 s with P2 or P4 failed, its radial lag slows the posigrade drift, and the worst
// drag can bring the ship back into the KOS. A posigrade v+ puts no load on a Y set (with P2 or P4 failed it burns
// 1-4 s on the +X set instead of ~50 s); the plan's grid stays behind it as the fallback
const HEALTHY_BREAKOUT_GRID = Object.freeze([...POSIGRADE_GRID, ...BREAKOUT_GRID]);

// failed: the jets FDIR has declared failed-off. Flight rule: no KOS entry without torque-balanced lateral translation.
export function createGuidance({ phase = PH.TRANSFER, t0 = 0, events = [], failed = [] } = {}) {
  const lateralOk = !LATERAL_SETS.some((set) => set.some((i) => failed.includes(i)));
  const departOk = !DEPART_CRITICAL.some((i) => failed.includes(i));
  const st = { phase, tPhase: t0, burn: null, plan: null, tDep: null, mcc: [false, false], boAvail: true, boT: -Infinity, committed: false, noGoSince: null, lastZ: null };
  const p = new Float64Array(3), e = new Float64Array(3), we = new Float64Array(3), vr = new Float64Array(3);
  const setPhase = (ph, t, why = '') => { st.phase = ph; st.tPhase = t; events.push({ t, kind: 'phase', phase: ph, why }); };
  // a burn aims at the planned post-burn trajectory: its velocity moves with the CW dynamics while the jets work
  const startBurn = (kind, v, t, xh) => {
    st.burn = { kind, t0: t, v0: Float64Array.from(xh.subarray(3, 6)), plan: Float64Array.from([xh[0], xh[1], xh[2], v[0], v[1], v[2]]) };
    st.burn.screened = [v[0] - xh[3], v[1] - xh[4], v[2] - xh[5]];
  };
  const _bx = new Float64Array(6), r = (x) => Math.hypot(x[0], x[1], x[2]);
  // box-keeping: back toward the hold point at |err| / 100 s (<= 5 cm/s) once outside the box, else stay put
  const box = (err, lim) => (Math.abs(err) > lim ? -clamp(err / 100, -0.05, 0.05) : 0);
  // GO: attitude, rates, nav, propellant, a Rule-P breakout, and the hold itself (inside its box, at rest)
  function goOk(nav, err, lim) {
    attError(nav.q, QREF, e); omegaRel(nav.q, nav.w, we);
    const still = Math.max(...err.map(Math.abs)) <= lim && Math.hypot(nav.x[3], nav.x[4], nav.x[5]) < 0.01;
    return Math.hypot(...e) < GO.att && Math.hypot(...we) < GO.rate && nav.valid && nav.fuel >= GO.propFrac && st.boAvail && still;
  }
  const inside = (x) => CORRIDOR_PHASES.includes(st.phase) || r(x) <= KOS_R;
  function breakout(t, why, nav) {
    // the at-once far breakout (from TRANSFER) flies the widest-margin v+ (planBreakout's widest); H1 and later fly the
    // first one that passes: at H1 the grid's rAfter saturates at the current range, so a wider v+ only lengthens a lopsided burn.
    // With a failed lateral jet, a breakout outside the KOS after TRANSFER tries the posigrade v+ first
    const into = inside(nav.x), far = st.phase === PH.TRANSFER;
    const plan = planBreakout(nav.x, { inside: into, widest: !into && far, grid: !lateralOk && !into && !far ? HEALTHY_BREAKOUT_GRID : BREAKOUT_GRID });
    events.push({ t, kind: 'abort', why, verified: !!plan });
    setPhase(PH.BREAKOUT, t, why);
    startBurn('BREAKOUT', plan ? plan.v : BREAKOUT_V, t, nav.x);
    st.burn.inside = inside(nav.x);
  }
  function burnDone(t, nav) {
    const b = st.burn, xh = nav.x; st.burn = null;
    events.push({ t, t0: b.t0, kind: b.kind, screened: b.screened, v0: Array.from(b.v0), v1: Array.from(xh.subarray(3, 6)), x: Array.from(xh), inside: b.inside });
    if (b.kind === 'DEPART') st.tDep = b.t0;
    else if (b.kind === 'ARRIVE') setPhase(PH.H1ACQ, t);
    else if (b.kind === 'ZNULL') setPhase(PH.H1, t);
    else if (b.kind === 'BREAKOUT') setPhase(PH.DEPART, t);
  }

  // nav: { x (6), q, w, mass, fuel (fraction of the propellant left), valid }
  function update(t, nav) {
    const xh = nav.x, cmd = { vDes: null, burn: false, tau: 5, db: [0.005, 0.005], mode: 'V', thetaDb: null, bypass: false, phase: st.phase };
    portRel(xh, nav.q, p);
    const axial = -p[0], lat = Math.hypot(p[1], p[2]);
    // the port's own velocity from the attitude motion (w_rel x port arm, in LVLH): lateral commands steer the port
    omegaRel(nav.q, nav.w, we);
    qRotate(nav.q, [we[1] * SHIP_PORT[2] - we[2] * SHIP_PORT[1], we[2] * SHIP_PORT[0] - we[0] * SHIP_PORT[2], we[0] * SHIP_PORT[1] - we[1] * SHIP_PORT[0]], vr);
    cmd.axial = axial;
    cmd.inCorridor = CORRIDOR_PHASES.includes(st.phase) && inCone(p);
    if (st.burn) {
      const b = st.burn, vt = propagate(b.plan, t - b.t0, 0, _bx).subarray(3, 6);
      const ok = Math.abs(vt[0] - xh[3]) <= BURN_TOL && Math.abs(vt[1] - xh[4]) <= BURN_TOL_V && Math.abs(vt[2] - xh[5]) <= BURN_TOL_V;
      if (ok || t - b.t0 > GUID.burnMax) burnDone(t, nav);
      else return Object.assign(cmd, { vDes: Float64Array.from(vt), burn: true, tau: 1, mode: 'P', bypass: b.kind === 'BREAKOUT', burnKind: b.kind });
    }
    if (RULE_A_PHASES.includes(st.phase) && !st.committed && t - st.boT >= GUID.boEvery) {
      st.boT = t; st.boAvail = !!planBreakout(xh, { inside: inside(xh) });
      if (!st.boAvail) { breakout(t, 'no Rule-P breakout (Rule A)', nav); return update(t, nav); }
    }
    switch (st.phase) {
      case PH.TRANSFER: {
        if (st.tDep === null) {
          // flight rule (deviation 11) at the start: a failed DEPART_CRITICAL jet would fly the departure lopsided
          // for minutes and miss Rule P, so that run breaks out before the transfer; any other failed lateral jet
          // flies the transfer and breaks out at H1 (NO-GO there)
          if (!departOk) { breakout(t, 'NO-GO: a lateral translation jet has failed', nav); return update(t, nav); }
          if (!st.plan) st.plan = planTransfer(xh, { rMin: RULE_P_TRANSFER + PLAN_MARGIN.depart });
          if (!st.plan) { events.push({ t, kind: 'nogo', why: 'no Rule-P transfer' }); return cmd; }
          startBurn('DEPART', st.plan.v, t, xh); events.push({ t, kind: 'plan', T: st.plan.T, dz: st.plan.dz, nominal: st.plan.nominal, rMin: st.plan.rMin });
          return update(t, nav);
        }
        const T = st.plan.T, el = t - st.tDep;
        for (let k = 0; k < 2; k++) {
          if (st.mcc[k] || el < ((k + 1) * T) / 3) continue;
          st.mcc[k] = true;
          const v = planMcc(xh, T - el), dv = Math.hypot(v[0] - xh[3], v[1] - xh[4], v[2] - xh[5]);
          const post = Float64Array.from([xh[0], xh[1], xh[2], v[0], v[1], v[2]]);
          if (dv > 0.005 && ruleP(post, RULE_P_TRANSFER + PLAN_MARGIN.mcc)) { startBurn('MCC', v, t, xh); return update(t, nav); }
          events.push({ t, kind: 'MCC skipped', dv });
        }
        if (el >= T) { startBurn('ARRIVE', [0, 0, xh[5]], t, xh); return update(t, nav); }
        cmd.thetaDb = COAST_DB;
        return cmd;
      }
      case PH.H1ACQ: {
        const z = xh[2];
        if ((st.lastZ !== null && Math.sign(z) !== Math.sign(st.lastZ)) || (Math.abs(z) < 2 && Math.abs(xh[5]) < 0.01)) { st.lastZ = null; startBurn('ZNULL', [xh[3], xh[4], 0], t, xh); return update(t, nav); }
        st.lastZ = z;
        cmd.vDes = [box(xh[0] - H1_POINT[0], GUID.boxH1), box(xh[1] - H1_POINT[1], GUID.boxH1), xh[5]];
        return cmd;
      }
      case PH.H1: {
        cmd.vDes = [box(xh[0] - H1_POINT[0], GUID.boxH1), box(xh[1] - H1_POINT[1], GUID.boxH1), box(xh[2] - H1_POINT[2], GUID.boxH1)];
        if (t - st.tPhase < GUID.h1Hold) return cmd;
        if (!lateralOk) { breakout(t, 'NO-GO: a lateral translation jet has failed', nav); return update(t, nav); }
        if (goOk(nav, [xh[0] - H1_POINT[0], xh[1] - H1_POINT[1], xh[2] - H1_POINT[2]], GUID.boxH1)) { st.noGoSince = null; setPhase(PH.CORRIDOR, t, 'GO'); return update(t, nav); }
        st.noGoSince ??= t;
        if (t - st.noGoSince > GUID.noGoMax) { breakout(t, 'NO-GO at H1', nav); return update(t, nav); }
        return cmd;
      }
      case PH.CORRIDOR: {
        if (r(xh) < KOS_R && !cmd.inCorridor) { breakout(t, 'outside the corridor cone', nav); return update(t, nav); }
        if (axial <= H2_RHO) { setPhase(PH.H2, t); return update(t, nav); }
        cmd.vDes = [clamp(axial / 1000, FINAL_SPEED, 0.2), -p[1] / 100 - vr[1], -p[2] / 100 - vr[2]];
        return cmd;
      }
      case PH.H2: {
        // the flight rule holds at H2 as at H1: a start at H2 (start=final) is already inside the KOS, so a failed lateral
        // jet breaks out before the hold instead of flying a degraded FINAL
        if (!lateralOk) { breakout(t, 'NO-GO: a lateral translation jet has failed', nav); return update(t, nav); }
        cmd.vDes = [-box(axial - H2_RHO, GUID.boxH2), box(p[1], GUID.boxH2) - vr[1], box(p[2], GUID.boxH2) - vr[2]];
        if (t - st.tPhase < GUID.h2Hold) return cmd;
        if (goOk(nav, [axial - H2_RHO, p[1], p[2]], GUID.boxH2)) { st.noGoSince = null; setPhase(PH.FINAL, t, 'GO'); return update(t, nav); }
        st.noGoSince ??= t;
        if (t - st.noGoSince > GUID.noGoMax) { breakout(t, 'NO-GO at H2', nav); return update(t, nav); }
        return cmd;
      }
      case PH.FINAL: {
        if (!st.committed && axial <= GUID.commit) {
          st.committed = true;
          attError(nav.q, QREF, e); omegaRel(nav.q, nav.w, we);
          if (lat > 0.08 || Math.hypot(...e) > 3 * DEG || Math.max(...we.map(Math.abs)) > 0.15 * DEG) { breakout(t, 'IDSS gate at the 3 m commit', nav); return update(t, nav); }
        }
        return Object.assign(cmd, { vDes: [FINAL_SPEED, -p[1] / 10 - vr[1], -p[2] / 10 - vr[2]], tau: 2, db: [0.004, 0.005], thetaDb: THETA_FINAL });
      }
      default:
        return cmd;
    }
  }
  return { st, events, update, breakout: (t, why, nav) => { if (st.phase !== PH.BREAKOUT && st.phase !== PH.DEPART) breakout(t, why, nav); } };
}
