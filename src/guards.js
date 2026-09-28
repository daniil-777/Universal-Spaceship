// The fly-by-wire's two helpers that take the stick from the pilot: the edge guard (turns away from a corridor edge)
// and the tunnel guide (threads a long world's tunnel bores). Moved out of env.js unchanged.
import { clamp, qInvRotate } from './mathx.js';
import { ENV, N_RAYS } from './envconst.js';

const _rel = new Float64Array(3), _tmp = new Float64Array(3);

// Edge guard: the corridor has edges — side walls everywhere, and in space a ceiling and a floor — that the beams see
// (sense()). When the ship would reach one within 1.7 s, the fly-by-wire blends in a turn away from it (a rotation of
// the nose toward the corridor, banking into the turn at a side wall), from nothing at 1.7 s to full authority at 0.6 s
// or within a unit of the edge; the pilot keeps full control everywhere else, and whenever its beams see terrain or
// traffic within ~15 units (it is busy dodging) the guard yields.
// Returns the blend 0..1 (also env.guard, for the flight board) and leaves the commands in env.guardCmd.
export function edgeGuard(env) {
  const s = env.ship, S = ENV.ship, lz = ENV.zHalf + S.wallClamp, top = (env.hf ? ENV.mountains.ceiling : ENV.yHalf) + S.wallClamp, bottom = -(ENV.yHalf + S.wallClamp);
  let best = 0, nx = 0, ny = 0, nz = 0;
  const edge = (dist, closing, ex, ey, ez) => {        // dist to the edge, speed toward it, the inward normal
    if (closing <= 0.05) return;
    const t = Math.max(0, dist) / closing, b = Math.max(Math.min(1, (1.7 - t) / 1.1), Math.min(1, (1.0 - dist) / 1.0));   // the velocity trails the nose by ~0.7 s: start early
    if (b > best) { best = b; nx = ex; ny = ey; nz = ez; }
  };
  edge(lz - s.p[2], s.v[2], 0, 0, -1); edge(lz + s.p[2], -s.v[2], 0, 0, 1);
  if (!env.hf) { edge(top - s.p[1], s.v[1], 0, -1, 0); edge(s.p[1] - bottom, -s.v[1], 0, 1, 0); }   // over terrain the thin-air ceiling is no edge to steer from: climbing to it over a ridge is fair
  if (best > 0) {                                      // the pilot is dodging something close (terrain, a rock, a plane): the guard steps back — a scrape along an edge is the lesser evil
    let threat = Infinity; for (let i = 0; i < N_RAYS; i++) if (!env.rayEdge[i] && env.rayHit[i] < threat) threat = env.rayHit[i];
    best *= Math.min(1, Math.max(0, (threat - 5) / 10));
  }
  env.guard = best; if (best <= 0) return 0;
  const f = s.f; let ax = f[1] * nz - f[2] * ny, ay = f[2] * nx - f[0] * nz, az = f[0] * ny - f[1] * nx, l = Math.hypot(ax, ay, az);   // turn the nose toward the corridor: about f × n
  if (l < 0.15) {                                    // nose straight at the edge: yaw (side wall) or pitch (ceiling, floor) away, whichever way is shorter
    const A = ny === 0 ? s.u : s.r, sg = ny === 0 ? (-(s.r[0] * nx + s.r[1] * ny + s.r[2] * nz) >= 0 ? 1 : -1) : ((s.u[0] * nx + s.u[1] * ny + s.u[2] * nz) >= 0 ? 1 : -1);
    ax = A[0] * sg; ay = A[1] * sg; az = A[2] * sg; l = 1;
  }
  const W = 1.5 / l, e = env.guardCmd; _rel[0] = ax * W; _rel[1] = ay * W; _rel[2] = az * W; qInvRotate(s.q, _rel, _tmp);   // body rates (roll, yaw, pitch)
  let roll = _tmp[0] / S.rateMax[0];
  if (ny === 0) roll += -1.6 * (Math.sign(ay) * 0.55 - s.r[1]);   // at a side wall bank into the turn (the right wing rises for a left turn)
  e[0] = clamp(_tmp[2] / S.rateMax[2], -1, 1); e[1] = clamp(_tmp[1] / S.rateMax[1], -1, 1); e[2] = clamp(roll, -1, 1);
  return best;
}

// Tunnel guide: a long world's tunnel rocks close the lane except for their bore, and the pilots never learned to thread
// one — so from 80 units before an entrance the fly-by-wire takes over (fully from 35 units, through the bore and a
// little past the exit): it pursues a point on the bore's centre line ahead of the ship, keeps the wings level and the
// speed at cruise. Returns the blend 0..1 (also env.guide) and leaves the commands in env.guideCmd.
export function tunnelGuide(env) {
  const hf = env.hf; env.guide = 0; if (!hf || !hf.tunnels || !hf.tunnels.length) return 0;
  const s = env.ship, S = ENV.ship, shift = hf.lap ? hf.lap.shift : 0, P = hf.PERIOD; let best = 0, T = null, X0 = 0;
  for (const t of hf.tunnels) {
    let x0 = t.x0 - shift, x1 = t.x1 - shift; const k = Math.round((s.p[0] - (x0 + x1) / 2) / P); x0 += k * P; x1 += k * P;
    const ahead = x0 - s.p[0], b = s.p[0] >= x0 - 1 && s.p[0] <= x1 + 8 ? 1 : ahead > 0 && ahead < 80 ? Math.min(1, (80 - ahead) / 45) : 0;
    if (b > best) { best = b; T = t; X0 = x0; }
  }
  if (!T) return 0;
  const ahead = X0 - s.p[0], L = ahead > 0 ? Math.max(10, Math.min(30, ahead * 0.6)) : 12;   // look further ahead while far out: a gentle line-up, then a tight hold
  let dx = L, dy = hf.y0 + T.y - s.p[1], dz = T.z - s.p[2]; const l = Math.hypot(dx, dy, dz); dx /= l; dy /= l; dz /= l;
  const f = s.f, ax = f[1] * dz - f[2] * dy, ay = f[2] * dx - f[0] * dz, az = f[0] * dy - f[1] * dx;   // turn the nose toward the aim point: about f × d
  _rel[0] = ax * 3; _rel[1] = ay * 3; _rel[2] = az * 3; qInvRotate(s.q, _rel, _tmp);
  const e = env.guideCmd; e[0] = clamp(_tmp[2] / S.rateMax[2], -1, 1); e[1] = clamp(_tmp[1] / S.rateMax[1], -1, 1);
  e[2] = clamp(_tmp[0] / S.rateMax[0] + 1.5 * s.r[1], -1, 1); e[3] = 0;   // wings level (positive roll lowers the right wing), cruise speed
  env.guide = best; return best;
}
