// Dubins paths — the shortest route between two poses for a vehicle that cannot turn tighter than R (Dubins 1957; the six
// words LSL, RSR, LSR, RSL, RLR, LRL in A. Walker's formulation). The approach planner joins wherever the ship is to the
// final approach course with one. Runway frame in and out (x forward, z right, heading = atan2(z, x), a right turn goes
// toward +z); internally the usual math frame (X = x, Y = −z, θ counter-clockwise = −heading).
const TWO_PI = 2 * Math.PI, mod2pi = (a) => a - TWO_PI * Math.floor(a / TWO_PI);
const WORDS = {
  LSL(a, b, d, sa, sb, ca, cb, cab) { const p2 = 2 + d * d - 2 * cab + 2 * d * (sa - sb); if (p2 < 0) return null; const t1 = Math.atan2(cb - ca, d + sa - sb); return [mod2pi(t1 - a), Math.sqrt(p2), mod2pi(b - t1)]; },
  RSR(a, b, d, sa, sb, ca, cb, cab) { const p2 = 2 + d * d - 2 * cab + 2 * d * (sb - sa); if (p2 < 0) return null; const t1 = Math.atan2(ca - cb, d - sa + sb); return [mod2pi(a - t1), Math.sqrt(p2), mod2pi(t1 - b)]; },
  LSR(a, b, d, sa, sb, ca, cb, cab) { const p2 = -2 + d * d + 2 * cab + 2 * d * (sa + sb); if (p2 < 0) return null; const p = Math.sqrt(p2), t0 = Math.atan2(-ca - cb, d + sa + sb) - Math.atan2(-2, p); return [mod2pi(t0 - a), p, mod2pi(t0 - mod2pi(b))]; },
  RSL(a, b, d, sa, sb, ca, cb, cab) { const p2 = -2 + d * d + 2 * cab - 2 * d * (sa + sb); if (p2 < 0) return null; const p = Math.sqrt(p2), t0 = Math.atan2(ca + cb, d - sa - sb) - Math.atan2(2, p); return [mod2pi(a - t0), p, mod2pi(b - t0)]; },
  RLR(a, b, d, sa, sb, ca, cb, cab) { const t = (6 - d * d + 2 * cab + 2 * d * (sa - sb)) / 8; if (Math.abs(t) > 1) return null; const ph = Math.atan2(ca - cb, d - sa + sb), p = mod2pi(TWO_PI - Math.acos(t)), t0 = mod2pi(a - ph + mod2pi(p / 2)); return [t0, p, mod2pi(a - b - t0 + mod2pi(p))]; },
  LRL(a, b, d, sa, sb, ca, cb, cab) { const t = (6 - d * d + 2 * cab + 2 * d * (sb - sa)) / 8; if (Math.abs(t) > 1) return null; const ph = Math.atan2(ca - cb, d + sa - sb), p = mod2pi(TWO_PI - Math.acos(t)), t0 = mod2pi(-a - ph + p / 2); return [t0, p, mod2pi(mod2pi(b) - a - t0 + mod2pi(p))]; },
};

export function dubins(from, to, R) {                     // → { word, params (in radii), length, from, R }
  const X0 = from.x, Y0 = -from.z, th0 = -from.hdg, dx = to.x - X0, dy = -to.z - Y0, D = Math.hypot(dx, dy), d = D / R;
  const th = d > 0 ? mod2pi(Math.atan2(dy, dx)) : 0, a = mod2pi(th0 - th), b = mod2pi(-to.hdg - th);
  const sa = Math.sin(a), sb = Math.sin(b), ca = Math.cos(a), cb = Math.cos(b), cab = Math.cos(a - b);
  let best = null;
  for (const [word, f] of Object.entries(WORDS)) { const p = f(a, b, d, sa, sb, ca, cb, cab); if (!p) continue; const len = (p[0] + p[1] + p[2]) * R; if (!best || len < best.length) best = { word, params: p, length: len }; }
  return { ...best, from: { X: X0, Y: Y0, th: th0 }, R };
}

function segment(t, q, type, out) {                        // advance a normalised pose by t along an L, S or R piece
  const st = Math.sin(q[2]), ct = Math.cos(q[2]);
  if (type === 'L') { out[0] = q[0] + Math.sin(q[2] + t) - st; out[1] = q[1] - Math.cos(q[2] + t) + ct; out[2] = q[2] + t; }
  else if (type === 'R') { out[0] = q[0] - Math.sin(q[2] - t) + st; out[1] = q[1] + Math.cos(q[2] - t) - ct; out[2] = q[2] - t; }
  else { out[0] = q[0] + ct * t; out[1] = q[1] + st * t; out[2] = q[2]; }
  return out;
}
export function poseAt(d, s, out = {}) {                   // the pose s metres along the path (clamped to its ends)
  const R = d.R, w = d.word, [t0, p1] = d.params, q0 = [0, 0, d.from.th], q1 = segment(t0, q0, w[0], [0, 0, 0]), q2 = segment(p1, q1, w[1], [0, 0, 0]), q = [0, 0, 0];
  const tp = Math.min(Math.max(s, 0), d.length) / R;
  if (tp < t0) segment(tp, q0, w[0], q); else if (tp < t0 + p1) segment(tp - t0, q1, w[1], q); else segment(tp - t0 - p1, q2, w[2], q);
  out.x = q[0] * R + d.from.X; out.z = -(q[1] * R + d.from.Y); out.hdg = Math.atan2(Math.sin(-q[2]), Math.cos(-q[2])); out.s = Math.min(Math.max(s, 0), d.length);
  out.turn = tp < t0 ? w[0] : tp < t0 + p1 ? w[1] : w[2]; return out;
}
export function samplePath(d, ds) {                        // points every ds metres (and the end), runway frame
  const n = Math.max(1, Math.ceil(d.length / ds)), pts = [];
  for (let i = 0; i <= n; i++) pts.push(poseAt(d, Math.min(i * ds, d.length), {}));
  return pts;
}
