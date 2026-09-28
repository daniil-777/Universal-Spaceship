// Safety filter, S1 classes (spec section 6): P1 = the station (KOS barrier outside the corridor, an axial approach-speed
// barrier inside it) and the 3 km keep-in; P4 = the speed ball. P2 (fragments) and P3 (corridor cone) arrive in S2
// through `extra`. Dykstra's method projects the commanded velocity onto the half-spaces and the ball.
// A half-space is { n: [3], b, cls } meaning n . v >= b. Priority P1 > P2 > P3 > P4; P1 is never dropped.
import { N, KOS_R, KEEPIN_R, A_AV } from './consts.js';

export const BRAKE_SPEED = 0.05;
export const barrier = (h, k = 0.01) => Math.min(k * h, Math.sqrt(A_AV * Math.max(h, 0)));
export const speedLimit = (r) => Math.min(2, 0.2 + 2 * N * (r - 10));
// the in-corridor station term: the closing speed along the docking axis stays below this at axial distance h (m)
export const axialLimit = (h) => Math.max(0.11, Math.min(0.25, 1.25 * h / 1000));

// ctx: { inCorridor, axial (m, station port plane to ship port), extra: [half-spaces] }
export function constraints(x, ctx = {}) {
  const r = Math.hypot(x[0], x[1], x[2]), u = [x[0] / r, x[1] / r, x[2] / r], list = [];
  if (ctx.inCorridor) list.push({ n: [-1, 0, 0], b: -axialLimit(Math.max(ctx.axial, 0)), cls: 'P1' });
  else list.push({ n: u, b: -barrier(r - KOS_R), cls: 'P1' });
  list.push({ n: [-u[0], -u[1], -u[2]], b: -barrier(KEEPIN_R - r), cls: 'P1' });
  for (const c of ctx.extra || []) list.push(c);
  return { r, u, list, ball: { R: speedLimit(r), cls: 'P4' } };
}

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
function projHalf(c, y, out) {
  const nn = dot(c.n, c.n), g = dot(c.n, y) - c.b;
  if (g >= 0) { out[0] = y[0]; out[1] = y[1]; out[2] = y[2]; return; }
  const k = g / nn; out[0] = y[0] - k * c.n[0]; out[1] = y[1] - k * c.n[1]; out[2] = y[2] - k * c.n[2];
}
function projBall(R, y, out) {
  const l = Math.hypot(y[0], y[1], y[2]), k = l > R ? R / l : 1;
  out[0] = y[0] * k; out[1] = y[1] * k; out[2] = y[2] * k;
}
const violation = (v, halves, ball) => Math.max(0, ...halves.map((c) => (c.b - dot(c.n, v)) / Math.sqrt(dot(c.n, c.n))), ball ? Math.hypot(...v) - ball.R : 0);

// Dykstra onto the intersection; returns { v, iters, residual }
export function dykstra(v0, halves, ball, { iters = 200, tol = 1e-7 } = {}) {
  const sets = halves.length + (ball ? 1 : 0), p = Array.from({ length: sets }, () => [0, 0, 0]);
  let x = [v0[0], v0[1], v0[2]]; const y = [0, 0, 0], z = [0, 0, 0]; let it = 0;
  for (; it < iters; it++) {
    let moved = 0;
    for (let s = 0; s < sets; s++) {
      y[0] = x[0] + p[s][0]; y[1] = x[1] + p[s][1]; y[2] = x[2] + p[s][2];
      if (s < halves.length) projHalf(halves[s], y, z); else projBall(ball.R, y, z);
      p[s][0] = y[0] - z[0]; p[s][1] = y[1] - z[1]; p[s][2] = y[2] - z[2];
      moved = Math.max(moved, Math.abs(z[0] - x[0]), Math.abs(z[1] - x[1]), Math.abs(z[2] - x[2]));
      x = [z[0], z[1], z[2]];
    }
    if (moved < tol && violation(x, halves, ball) < tol) break;
  }
  return { v: x, iters: it + 1, residual: violation(x, halves, ball) };
}

// Exact fallback for the rare sharp vertex where Dykstra has not reached 1e-7 in 200 iterations (one random case in
// 200 needed ~2,000): the projection is the nearest feasible KKT candidate (<= 3 active planes, or <= 2 plus the ball).
function solveEq(v0, cs) {
  const k = cs.length, G = [], r = [];
  for (let i = 0; i < k; i++) { G.push(cs.map((c) => dot(cs[i].n, c.n))); r.push(cs[i].b - dot(cs[i].n, v0)); }
  const lam = gauss(G, r);
  if (!lam) return null;
  const v = [v0[0], v0[1], v0[2]];
  for (let i = 0; i < k; i++) for (let a = 0; a < 3; a++) v[a] += lam[i] * cs[i].n[a];
  return v;
}
function gauss(A, b) {
  const n = b.length, M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c; for (let i = c + 1; i < n; i++) if (Math.abs(M[i][c]) > Math.abs(M[p][c])) p = i;
    if (Math.abs(M[p][c]) < 1e-12) return null;
    [M[c], M[p]] = [M[p], M[c]];
    for (let i = 0; i < n; i++) if (i !== c) { const f = M[i][c] / M[c][c]; for (let j = c; j <= n; j++) M[i][j] -= f * M[c][j]; }
  }
  return M.map((row, i) => row[n] / row[i]);
}
function* subsets(list, maxK, from = 0, cur = []) {
  yield cur;
  if (cur.length === maxK) return;
  for (let i = from; i < list.length; i++) yield* subsets(list, maxK, i + 1, [...cur, list[i]]);
}
export function projectExact(v0, halves, ball) {
  let best = null, bd = Infinity;
  const take = (v) => { if (!v || !(violation(v, halves, ball) <= 1e-9)) return; const d = Math.hypot(v[0] - v0[0], v[1] - v0[1], v[2] - v0[2]); if (d < bd) { bd = d; best = v; } };
  for (const cs of subsets(halves, 3)) take(cs.length ? solveEq(v0, cs) : [v0[0], v0[1], v0[2]]);
  if (ball) for (const cs of subsets(halves, 2)) {
    const p = cs.length ? solveEq([0, 0, 0], cs) : [0, 0, 0];
    if (!p) continue;
    const q = cs.length ? solveEq(v0, cs) : v0, d = [q[0] - p[0], q[1] - p[1], q[2] - p[2]], rr = ball.R * ball.R - dot(p, p);
    if (rr < 0) continue;
    if (cs.length === 2) {
      const u = [cs[0].n[1] * cs[1].n[2] - cs[0].n[2] * cs[1].n[1], cs[0].n[2] * cs[1].n[0] - cs[0].n[0] * cs[1].n[2], cs[0].n[0] * cs[1].n[1] - cs[0].n[1] * cs[1].n[0]];
      const l = Math.hypot(...u); if (l < 1e-12) continue;
      for (const sg of [1, -1]) { const t = (sg * Math.sqrt(rr)) / l; take([p[0] + t * u[0], p[1] + t * u[1], p[2] + t * u[2]]); }
    } else {
      const l = Math.hypot(...d); if (l < 1e-12) continue;
      const k = Math.sqrt(rr) / l; take([p[0] + k * d[0], p[1] + k * d[1], p[2] + k * d[2]]);
    }
  }
  return best;
}
export function project(v0, halves, ball) {
  const res = dykstra(v0, halves, ball);
  if (res.residual < 1e-7) return { ...res, method: 'dykstra' };
  const v = projectExact(v0, halves, ball);
  return v ? { v, iters: res.iters, residual: violation(v, halves, ball), method: 'exact' } : null;
}

const DROP_ORDER = ['P4', 'P3', 'P2'];
const finite3 = (a) => Number.isFinite(a[0]) && Number.isFinite(a[1]) && Number.isFinite(a[2]);
// Rejects non-finite input (NaN/Infinity) before it reaches Dykstra or the exact fallback.
// A NaN violation compares false against every threshold there, which would otherwise let the
// unmodified, unsafe command through instead of failing safe.
function validInput(vDes, x, ctx) {
  if (!finite3(vDes) || !finite3(x)) return false;
  if (ctx.inCorridor && !Number.isFinite(ctx.axial)) return false;
  for (const c of ctx.extra || []) {
    if (!finite3(c.n) || !Number.isFinite(c.b)) return false;
  }
  return true;
}
// The filter: v_des (LVLH, m/s) -> safe velocity. Logs every drop and brake into log (array) with time t.
export function filterVelocity(vDes, x, ctx = {}, log = null, t = 0) {
  if (!validInput(vDes, x, ctx)) {
    if (log) log.push({ t, kind: 'brake', cls: 'P1', why: 'invalid input' });
    return { v: [0, 0, 0], brake: true, dropped: [], changed: true };
  }
  const { r, u, list, ball } = constraints(x, ctx);
  if (!ctx.inCorridor && r <= KOS_R) {
    if (log) log.push({ t, kind: 'brake', cls: 'P1', why: 'inside KOS outside the corridor' });
    return { v: [BRAKE_SPEED * u[0], BRAKE_SPEED * u[1], BRAKE_SPEED * u[2]], brake: true, dropped: [], changed: true };
  }
  if (r >= KEEPIN_R) {
    if (log) log.push({ t, kind: 'brake', cls: 'P1', why: 'keep-in' });
    return { v: [-BRAKE_SPEED * u[0], -BRAKE_SPEED * u[1], -BRAKE_SPEED * u[2]], brake: true, dropped: [], changed: true };
  }
  let halves = list, useBall = true; const dropped = [];
  for (let k = 0; k <= DROP_ORDER.length; k++) {
    const res = project(vDes, halves, useBall ? ball : null);
    if (res) {
      const changed = Math.hypot(res.v[0] - vDes[0], res.v[1] - vDes[1], res.v[2] - vDes[2]) > 1e-9;
      return { v: res.v, brake: false, dropped, changed, residual: res.residual, method: res.method };
    }
    if (k === DROP_ORDER.length) break;
    const cls = DROP_ORDER[k];
    if (cls === 'P4') useBall = false; else halves = halves.filter((c) => c.cls !== cls);
    dropped.push(cls);
    if (log) log.push({ t, kind: 'drop', cls });
  }
  if (log) log.push({ t, kind: 'brake', cls: 'P1', why: 'P1 infeasible' });
  return { v: [BRAKE_SPEED * u[0], BRAKE_SPEED * u[1], BRAKE_SPEED * u[2]], brake: true, dropped, changed: true };
}
