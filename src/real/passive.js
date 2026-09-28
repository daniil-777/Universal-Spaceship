// Passive safety and the far-field plans (spec sections 3 and 6): the 24 h free-drift minimum over the drag bound,
// the breakout criterion, the TRANSFER targeting with its T search, and the BREAKOUT burn search.
// Positions under a constant differential drag ad are p(t) + ad * g(t): r^2 is quadratic in ad, so the worst ad in
// [-b, b] is exact per sample.
import { N, DAY, DRAG_B, KOS_R, T_ORB, H1_POINT, RULE_P_TRANSFER, BREAKOUT_V } from './consts.js';

const _p = new Float64Array(3), _g = new Float64Array(2);
function freePos(x, t) {
  const n = N, s = Math.sin(n * t), c = Math.cos(n * t);
  _p[0] = x[0] + 6 * (s - n * t) * x[1] + (4 * s - 3 * n * t) / n * x[3] - 2 * (1 - c) / n * x[4];
  _p[1] = (4 - 3 * c) * x[1] + 2 * (1 - c) / n * x[3] + s / n * x[4];
  _p[2] = c * x[2] + s / n * x[5];
  _g[0] = (4 / (n * n)) * (1 - c) - 1.5 * t * t;
  _g[1] = (2 / (n * n)) * (n * t - s);
}
// the smallest r at time t over ad in [-b, b]; sets _worst
let _worst = 0;
function minROverDrag(x, t, b) {
  freePos(x, t);
  const gg = _g[0] * _g[0] + _g[1] * _g[1];
  let a = gg > 0 ? -(_p[0] * _g[0] + _p[1] * _g[1]) / gg : 0;
  a = a < -b ? -b : a > b ? b : a;
  _worst = a;
  return Math.hypot(_p[0] + a * _g[0], _p[1] + a * _g[1], _p[2]);
}

// 24 h free-drift minimum distance to the station CoM over every constant ad in [-b, b], sampled every dt
export function passiveMin(x, horizon = DAY, b = DRAG_B, dt = 60) {
  let rMin = Infinity, tMin = 0, adWorst = 0;
  for (let t = 0; t <= horizon + 1e-9; t += dt) {
    const r = minROverDrag(x, t, b);
    if (r < rMin) { rMin = r; tMin = t; adWorst = _worst; }
  }
  return { rMin, tMin, adWorst };
}

// Rule P for TRANSFER burns and MCCs
export const ruleP = (x, rMin = RULE_P_TRANSFER, b = DRAG_B) => passiveMin(x, DAY, b).rMin >= rMin;

// BREAKOUT criterion (Rule P for BREAKOUT and CAM). inside = true (in the KOS or in the corridor after GO): out of the
// KOS by one orbit for every ad and never back in for 24 h; inside = false (outside the KOS): never enter it.
export function breakoutOk(x, { b = DRAG_B, dt = 10, inside = true } = {}) {
  let tLast = null, rAfter = Infinity;
  for (let t = 0; t <= DAY; t += dt) {
    const r = minROverDrag(x, t, b);
    if (r <= KOS_R) { tLast = t; rAfter = Infinity; if (!inside || t >= T_ORB) return { ok: false, tExit: null, rAfter: r }; }
    else rAfter = Math.min(rAfter, r);
  }
  return { ok: true, tExit: tLast === null ? 0 : tLast + dt, rAfter };
}

// In-plane CW targeting: the velocity (Xd, Yd) at x that puts (X, Y) at target after T (drag unknown: 0)
export function solveInPlane(x, T, target = H1_POINT) {
  const n = N, s = Math.sin(n * T), c = Math.cos(n * T);
  const a11 = (4 * s - 3 * n * T) / n, a12 = -2 * (1 - c) / n, a21 = 2 * (1 - c) / n, a22 = s / n;
  const r1 = target[0] - x[0] - 6 * (s - n * T) * x[1], r2 = target[1] - (4 - 3 * c) * x[1];
  const det = a11 * a22 - a12 * a21;
  return [(r1 * a22 - a12 * r2) / det, (a11 * r2 - a21 * r1) / det];
}

// velocity at time T of the free drift from x (no drag)
function velAt(x, T) {
  const n = N, s = Math.sin(n * T), c = Math.cos(n * T);
  return [6 * n * (c - 1) * x[1] + (4 * c - 3) * x[3] - 2 * s * x[4], 3 * n * s * x[1] + 2 * s * x[3] + c * x[4], -n * s * x[2] + c * x[5]];
}

// TRANSFER plan: the spec's nominal half-orbit radial hop when it passes Rule P; otherwise the least total dv over
// T in [0.35, 0.6] orbit x a cross-track component, among candidates that pass Rule P. null when none passes.
export const DZ_GRID = Object.freeze([0, -0.02, 0.02, -0.05, 0.05, -0.1, 0.1, -0.15, 0.15, -0.2, 0.2]);
export function planTransfer(x, { b = DRAG_B, rMin = RULE_P_TRANSFER } = {}) {
  const cand = (T, dz) => {
    const [vx, vy] = solveInPlane(x, T), xp = Float64Array.from([x[0], x[1], x[2], vx, vy, x[5] + dz]);
    const va = velAt(xp, T);
    const dv1 = Math.hypot(vx - x[3], vy - x[4], dz), dv2 = Math.hypot(va[0], va[1]);
    return { T, dz, v: [vx, vy, x[5] + dz], dv1, dv2, dv: dv1 + dv2, x: xp };
  };
  const nom = cand(T_ORB / 2, 0);
  nom.rMin = passiveMin(nom.x, DAY, b).rMin;
  if (nom.rMin >= rMin) return { ...nom, nominal: true };
  let best = null;
  for (let k = 0; k <= 25; k++) {
    for (const dz of DZ_GRID) {
      const c = cand((0.35 + 0.01 * k) * T_ORB, dz);
      if (best && c.dv >= best.dv) continue;
      c.rMin = passiveMin(c.x, DAY, b).rMin;
      if (c.rMin >= rMin) best = c;
    }
  }
  return best && { ...best, nominal: false };
}

// mid-course correction: re-target the remaining time (in-plane only); returns the new velocity
export function planMcc(x, tLeft) {
  const [vx, vy] = solveInPlane(x, tLeft);
  return [vx, vy, x[5]];
}

// BREAKOUT: target post-burn velocity v+ relative to the station; searched when the nominal fails the criterion
export const BREAKOUT_GRID = Object.freeze([
  BREAKOUT_V, [0.02, 0.15, 0], [0.04, 0.10, 0], [0.04, 0.15, 0], [0.0, 0.15, 0], [0.02, 0.20, 0], [0.06, 0.2, 0], [0.1, 0.2, 0],
]);
// posigrade-only v+: a breakout on the +X set (the aft P13/P14) alone, which no failed Y or Z primary touches
export const POSIGRADE_GRID = Object.freeze([[0.03, 0, 0], [0.05, 0, 0]]);
// widest: take the passing v+ with the largest 24 h minimum range (rAfter) instead of the first one. Outside the KOS a
// finite burn's radial offset dY drifts along-track at 6n dY, so a first-passing plan with a few metres of margin is
// not passively safe once flown. grid: the v+ candidates, in order of preference
export function planBreakout(x, { b = DRAG_B, inside = true, widest = false, grid = BREAKOUT_GRID } = {}) {
  let best = null;
  for (const v of grid) {
    const xp = Float64Array.from([x[0], x[1], x[2], v[0], v[1], v[2]]);
    const chk = breakoutOk(xp, { b, inside });
    if (!chk.ok) continue;
    const c = { v: Array.from(v), dv: Math.hypot(v[0] - x[3], v[1] - x[4], v[2] - x[5]), ...chk };
    if (!widest) return c;
    if (!best || c.rAfter > best.rAfter) best = c;
  }
  return best;
}
