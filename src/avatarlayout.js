// The Avatar valley, the China route's world: Meshy-generated Zhangjiajie formations — a colossal pillar, a needle
// cluster, a stone arch, a waterfall massif, a Tianzi ridge wall, vine-hung Hallelujah pillars, a pagoda summit, Guilin
// karst domes and the first two photo clusters — along a 960-unit valley that repeats only after eight corridor laps.
// A river winds down the middle (the flight lane) with formations on both banks and the odd one on an island, staggered
// canyon walls beyond the banks and, far out, massifs of two to six formations stacked together, growing into giants
// toward the horizon: about a thousand formations, ten times the old route's. Every formation grows out of the ground
// rather than standing on it: its rim sits below a foothill that the valley floor raises around its footprint
// (valleyGround), so rock, jungle skirt and forest floor meet like a real mountain's roots. Shared by the renderer, the
// ground and the offline bake of the collision grid. Units: corridor units (1 ≈ 19 m); map x ∈ [0, 960), periodic.
const TL = 'textures/meshy/tiles/', OLD = 'textures/meshy/', P = 960, TAU = Math.PI * 2;
// S: summit at scale 1; w: the footprint's (x, z) extent / height (skirt included); b: the pillar body's half-extents /
// height (what the flight lane must avoid); sink: how deep the rim goes below the ground / height; front: faces +z
export const AV_MODELS = [
  { n: 'av_monolith', S: 44, w: [0.93, 0.90], b: [0.25, 0.25], sink: 0.02 },
  { n: 'av_needles', S: 40, w: [1.09, 1.04], b: [0.42, 0.40], sink: 0.02 },
  { n: 'av_arch', S: 42, w: [1.00, 0.74], b: [0.40, 0.28], sink: 0.02, front: true },
  { n: 'av_falls', S: 32, w: [1.14, 1.22], b: [0.45, 0.48], sink: 0.025, front: true },
  { n: 'av_ridge', S: 30, w: [2.31, 0.96], b: [1.0, 0.36], sink: 0.03 },
  { n: 'av_hallelujah', S: 46, w: [1.01, 0.93], b: [0.35, 0.33], sink: 0.02 },
  { n: 'av_pagoda', S: 44, w: [0.88, 0.83], b: [0.30, 0.28], sink: 0.02, front: true },
  { n: 'av_karst', S: 28, w: [1.88, 1.72], b: [0.62, 0.58], sink: 0.03 },
  { n: 'pillars_a', S: 40, w: [1.30, 0.99], b: [0.6, 0.45], sink: 0.045, old: true },
  { n: 'pillars_b', S: 40, w: [1.36, 1.34], b: [0.62, 0.62], sink: 0.045, old: true },
  // tunnel rocks (placed by hand across the lane, turned so the bore runs along the flight path, their depth `sd` compressed):
  // tunnel = the bore through the model along its z axis, measured by the harness's tunnel_probe.mjs (fractions of the
  // height; u across = model x, v up): its box u0..u1 × v0..v1, centre (cu, cv), where the rock begins and ends (e0, e1), and its
  // profile `prof` — across the bore, the open run [u, v low, v high] that rays along the axis pass all the way through
  { n: 'av_gate', S: 84, w: [1.603, 1.186], b: [0.62, 0.45], sink: 0.02, sd: 0.6, tunnel: { u0: -0.11, u1: 0.09, v0: 0.269, v1: 0.494, cu: -0.012, cv: 0.377, e0: -0.157, e1: 0.211 , prof: [[-0.105,0.325,0.419],[-0.095,0.325,0.444],[-0.085,0.319,0.456],[-0.075,0.294,0.463],[-0.065,0.287,0.463],[-0.055,0.281,0.475],[-0.045,0.275,0.481],[-0.035,0.269,0.487],[-0.025,0.269,0.494],[-0.015,0.275,0.487],[-0.005,0.275,0.487],[0.005,0.275,0.481],[0.015,0.275,0.481],[0.025,0.275,0.481],[0.035,0.281,0.481],[0.045,0.287,0.469],[0.055,0.287,0.456],[0.065,0.294,0.425],[0.075,0.287,0.406],[0.085,0.313,0.35]] } },
  { n: 'av_rivercave', S: 90, w: [1.383, 1.404], b: [0.6, 0.6], sink: 0.02, sd: 0.5, tunnel: { u0: -0.052, u1: 0.13, v0: 0.019, v1: 0.25, cu: 0.036, cv: 0.126, e0: -0.609, e1: 0.63 , prof: [[-0.048,0.075,0.119],[-0.039,0.05,0.144],[-0.03,0.037,0.188],[-0.022,0.031,0.206],[-0.013,0.025,0.244],[-0.004,0.019,0.25],[0.004,0.019,0.244],[0.013,0.019,0.237],[0.022,0.019,0.25],[0.03,0.019,0.25],[0.039,0.019,0.25],[0.048,0.025,0.25],[0.056,0.025,0.194],[0.065,0.025,0.244],[0.073,0.025,0.219],[0.082,0.025,0.225],[0.091,0.025,0.2],[0.099,0.031,0.2],[0.108,0.037,0.188],[0.117,0.094,0.175],[0.125,0.138,0.163]] } },
  { n: 'av_fallsgate', S: 72, w: [1.707, 1.652], b: [0.7, 0.6], sink: 0.02, sd: 0.6, tunnel: { u0: -0.107, u1: 0.16, v0: 0.181, v1: 0.494, cu: 0.029, cv: 0.334, e0: -0.433, e1: 0.314 , prof: [[-0.101,0.281,0.35],[-0.091,0.237,0.369],[-0.08,0.231,0.4],[-0.069,0.225,0.456],[-0.059,0.212,0.463],[-0.048,0.2,0.463],[-0.037,0.194,0.469],[-0.027,0.194,0.475],[-0.016,0.194,0.481],[-0.005,0.188,0.494],[0.005,0.181,0.494],[0.016,0.188,0.494],[0.027,0.188,0.494],[0.037,0.188,0.487],[0.048,0.181,0.487],[0.059,0.181,0.487],[0.069,0.181,0.487],[0.08,0.188,0.494],[0.091,0.194,0.487],[0.101,0.194,0.481],[0.112,0.2,0.475],[0.123,0.206,0.469],[0.133,0.212,0.438],[0.144,0.219,0.369],[0.155,0.313,0.362]] } },
];
// waterfall lips per model (the harness's falls_probe.mjs): [azimuth in the model's frame, radius, lip height, landing
// height] as fractions of its height — where a sheer face drops a long way below a ledge (no overhang above)
export const AV_LIPS = [
  [[3.665,0.193,0.8,0.143],[4.974,0.164,0.86,0.307],[0.785,0.195,0.68,0.164],[1.833,0.186,0.68,0.182]],
  [[3.927,0.138,0.8,0.004],[2.88,0.293,0.92,0.19],[1.833,0.303,0.92,0.24]],
  [],
  [[1.571,0.26,0.86,0.016],[3.403,0.367,0.74,0.194],[4.974,0.42,0.68,0.194],[0.262,0.36,0.68,0.214]],
  [[4.189,0.132,0.86,0.297],[5.498,0.251,0.74,0.191],[2.356,0.136,0.8,0.255],[0.785,0.229,0.74,0.205]],
  [[1.309,0.101,0.92,0.095],[4.712,0.298,0.92,0.14],[2.88,0.446,0.74,0]],
  [],
  [[1.047,0.176,0.8,0.278]],
  [[4.189,0.148,0.92,0],[1.309,0.157,0.92,0.023],[2.618,0.163,0.92,0.539]],
  [[4.189,0.104,0.8,0],[1.571,0.109,0.62,0],[0.524,0.204,0.8,0.236]],
  [[1.047,0.259,0.74,0.178],[4.451,0.216,0.68,0.147],[6.021,0.497,0.62,0.187]],
  [[0.524,0.258,0.92,0.012],[2.88,0.226,0.92,0.012],[5.76,0.525,0.74,0.008],[3.927,0.57,0.56,0.179]],
  [[1.309,0.483,0.74,0],[4.974,0.525,0.74,0],[2.88,0.59,0.92,0.256],[0,0.604,0.8,0.154]],
];
export const AV_ZONES = ['Hallelujah Peaks', 'Tianzi Ridge', 'Golden Whip Stream', 'Karst Gardens'];   // 240 units each
const ZW = [                                               // model weights per zone (indices into AV_MODELS)
  [3, 2, 1, 0, 0, 4, 0, 0, 1, 1], [1, 3, 0, 0, 3, 0, 1, 0, 2, 2], [2, 1, 2, 3, 0, 1, 2, 0, 0, 0], [1, 1, 0, 1, 0, 1, 1, 4, 0, 0]];

export const riverZ = (x) => 6 * Math.sin(TAU * x / 240) + 2.5 * Math.sin(TAU * x / 96 + 1.3);   // the river's centre line (|z| ≤ 8.5)
export const riverW = (x) => 3.4 + 0.9 * Math.sin(TAU * x / 160 + 0.4);                          // half-width of the water
const laneHalf = (x) => 8.5 + 2.5 * (0.5 + 0.5 * Math.sin(TAU * x / 320 + 0.7));                  // formation bodies stay this far from the river's centre
const smooth = (a, b, v) => { const t = Math.min(1, Math.max(0, (v - a) / (b - a))); return t * t * (3 - 2 * t); };
function hash(i, j, s) { let h = Math.imul(i, 374761393) + Math.imul(j, 668265263) + Math.imul(s, 1442695041); h = Math.imul(h ^ (h >>> 13), 1274126177); return ((h ^ (h >>> 16)) >>> 0) / 4294967296; }
function vnoise(x, z, cell, s) {                           // value noise, periodic in x (the cell divides the period)
  const n = P / cell, fx = x / cell, fz = z / cell, ix = Math.floor(fx), iz = Math.floor(fz), tx = fx - ix, tz = fz - iz, sx = tx * tx * (3 - 2 * tx), sz = tz * tz * (3 - 2 * tz);
  const i0 = ((ix % n) + n) % n, i1 = (i0 + 1) % n, a = hash(i0, iz, s), b = hash(i1, iz, s), c = hash(i0, iz + 1, s), d = hash(i1, iz + 1, s);
  return a + (b - a) * sx + (c - a) * sz + (a - b - c + d) * sx * sz;
}
const wrap = (dx) => dx - P * Math.round(dx / P);
// the valley floor without the foothills: sides rising away from the river, rolling forest hills, a flat flood plain and the channel
function floor(x, z) {
  const az = Math.abs(z), dr = Math.abs(z - riverZ(x)), w = riverW(x);
  const hills = (1.0 + 2.8 * smooth(30, 220, az)) * (0.65 * vnoise(x, z, 96, 11) + 0.35 * vnoise(x, z, 40, 12));
  return 7 * Math.pow(smooth(40, 460, az), 1.5) + smooth(w + 2, w + 14, dr) * hills - 1.6 * (1 - smooth(w - 1.2, w + 1.6, dr));
}
// local frame of a placement (three.js turns local (x, z) by ry into world (c·x + s·z, −s·x + c·z))
const local = (p, dx, dz) => { const c = Math.cos(p.ry), s = Math.sin(p.ry); return [c * dx - s * dz, s * dx + c * dz]; };
const world = (p, lx, lz) => { const c = Math.cos(p.ry), s = Math.sin(p.ry); return [p.x + c * lx + s * lz, p.z - s * lx + c * lz]; };
const halfW = (p) => { const m = AV_MODELS[p.m]; return [m.w[0] / 2 * m.S * p.s, m.w[1] / 2 * m.S * p.s * (p.sd || 1)]; };   // the skirt's half-extents (sd: a squeezed depth)
const halfB = (p) => { const m = AV_MODELS[p.m]; return [m.b[0] * m.S * p.s, m.b[1] * m.S * p.s * (p.sd || 1)]; };         // the body's
function reach(p, [a, b], ux, uz) { const [lx, lz] = local(p, ux, uz); return 1 / Math.sqrt((lx / a) ** 2 + (lz / b) ** 2); }   // the ellipse's radius toward (ux, uz)
function ring(p, [a, b], n = 16) { const out = []; for (let k = 0; k < n; k++) { const t = TAU * k / n; out.push(world(p, a * Math.cos(t), b * Math.sin(t))); } return out; }

// The ground (height above y0 at map (x, z)): the floor plus, around every formation, a foothill — a plateau under its
// skirt that falls away over another 0.6 of the footprint — so the skirt's jungle slope continues into forest floor.
// valleySampler(set).sample(x, z, out) also gives out[1]: how close (x, z) is to a formation's foot (1 at the rim, 0 clear
// of it), for the contact shade and scree the ground shader lays around the rock.
export function valleySampler(set) {
  if (set.sampler) return set.sampler;
  const cell = 40, bins = new Map(), key = (i, j) => i * 4096 + j, nx = P / cell;
  for (const p of set.placements) {                        // spatial bins over each foothill's reach (wrapping in x)
    const [a, b] = halfW(p), r = 1.8 * Math.max(a, b);
    for (let i = Math.floor((p.x - r) / cell); i <= Math.floor((p.x + r) / cell); i++) for (let j = Math.floor((p.z - r) / cell); j <= Math.floor((p.z + r) / cell); j++) {
      const k = key(((i % nx) + nx) % nx, j + 2048); if (!bins.has(k)) bins.set(k, []); bins.get(k).push(p); }
  }
  const sample = (x, z, out) => {
    let h = 0, near = 0; const list = bins.get(key(((Math.floor(x / cell) % nx) + nx) % nx, Math.floor(z / cell) + 2048));
    if (list) for (const p of list) {
      const [a, b] = halfW(p), [lx, lz] = local(p, wrap(x - p.x), z - p.z), e = Math.sqrt((lx / a) ** 2 + (lz / b) ** 2); if (e >= 1.8) continue;
      const f = 1 - smooth(0.85, 1.6, e), v = 0.035 * AV_MODELS[p.m].S * p.s * f * Math.sqrt(f); if (v > h) h = v;
      const c = 1 - smooth(0.9, 1.8, e); if (c > near) near = c;
    }
    out[0] = floor(x, z) + h; out[1] = near; return out;
  };
  const tmp = [0, 0];
  set.sampler = { sample, height: (x, z) => sample(x, z, tmp)[0] };
  return set.sampler;
}
export const valleyGround = (set) => valleySampler(set).height;

export function avatarValley() {
  let seed = 20260927; const rng = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
  const placed = [], islands = [];
  const pick = (x, allow) => { const z = Math.floor(((((x + 40 * (rng() - 0.5)) % P) + P) % P) / 240), w = ZW[z].map((v, i) => (allow(i) ? v : 0)), t = w.reduce((a, v) => a + v, 0) * rng();
    let acc = 0; for (let i = 0; i < w.length; i++) { acc += w[i]; if (t < acc) return i; } return 0; };
  const clash = (p, k) => placed.some((q) => { const dx = wrap(p.x - q.x), dz = p.z - q.z, d = Math.hypot(dx, dz); if (d < 1e-6) return true;
    return d < k * (reach(p, halfB(p), -dx / d, -dz / d) + reach(q, halfB(q), dx / d, dz / d)); });
  const inSpawn = (p) => {                                 // the ship's spawn points (map x = 120k − 54): the first 26 units stay open
    for (let k = 0; k <= 8; k++) { const x0 = 120 * k - 58, x1 = 120 * k - 32;
      for (const [x, z] of ring(p, halfB(p)).concat([[p.x, p.z]])) { const xx = x + P * Math.round((x0 - x) / P); if (xx > x0 && xx < x1 && Math.abs(z) < 11) return true; }
      for (const [x, z] of ring(p, halfW(p))) { const xx = x + P * Math.round((x0 - x) / P); if (xx > x0 && xx < x1 && Math.abs(z) < 6) return true; } }
    return false; };
  const inLane = (p, side, extra = 0) => ring(p, halfB(p)).some(([x, z]) => (z - riverZ(x)) * side < laneHalf(x) + extra);
  const turn = (m, side) => (AV_MODELS[m].front ? (side > 0 ? Math.PI : 0) + 0.8 * (rng() - 0.5) : rng() * TAU);   // show the arch, the falls and the stair to the valley
  // the tunnel rocks first: one of each across the lane, the bore on the corridor's centre line (the river cave where the
  // river crosses it), turned a quarter so the model's z (the bore) runs along +x; the approach stays open
  const tunnels = [];
  for (const [k, m, dx] of [[1, 10, 20], [4, 11, 26], [6, 12, 20]]) {
    const M = AV_MODELS[m], t = M.tunnel; let x = 120 * k + dx;
    if (m === 11) { let best = x; for (let c = x - 12; c <= x + 12; c += 0.5) if (Math.abs(riverZ(c)) < Math.abs(riverZ(best))) best = c; x = best; }
    const p = { m, x, z: +(t.cu * M.S).toFixed(2), ry: Math.PI / 2, s: 1, sd: M.sd, zone: 'tunnel' }; placed.push(p); tunnels.push(p);
  }
  const inApproach = (p) => tunnels.some((q) => { const t = AV_MODELS[q.m].tunnel, H = AV_MODELS[q.m].S, x0 = q.x + t.e0 * H * q.sd - 75, x1 = q.x + t.e1 * H * q.sd + 30;
    return ring(p, halfB(p)).concat([[p.x, p.z]]).some(([x, z]) => { const xx = x + P * Math.round((q.x - x) / P); return xx > x0 && xx < x1 && Math.abs(z) < 17; }); });
  // islands: a small formation in the river where the banks open out
  for (const k of [0, 2, 3, 5, 7]) {                        // between the spawn points
    const x = 120 * k + 4 + 24 * rng(), m = [0, 6, 1, 5][Math.floor(rng() * 4)], p = { m, x, z: riverZ(x), ry: rng() * TAU, s: +(0.42 + 0.12 * rng()).toFixed(2), zone: 'island' };
    if (!inSpawn(p) && !inApproach(p)) { placed.push(p); islands.push(x); }
  }
  // the banks: a formation every 22–36 units on each bank, its body just outside the lane
  for (const side of [-1, 1]) for (let x = side > 0 ? 12 * rng() : 12 + 12 * rng(); x < P; x += 22 + 14 * rng()) {   // each bank on its own, staggered
    const wide = islands.some((ix) => Math.abs(wrap(x - ix)) < 28) ? 12 : 0;   // beside an island the banks step back: water on both sides of it
    for (let t = 0; t < 10; t++) {
      const m = pick(x, (i) => i !== 4 || rng() < 0.5), s = +(0.8 + 0.26 * rng()).toFixed(2), ry = m === 4 ? (rng() < 0.5 ? 0 : Math.PI) + 0.3 * (rng() - 0.5) : turn(m, side);
      const p = { m, x: x + 4 * (rng() - 0.5), z: 0, ry, s, zone: 'bank' }; const across = reach(p, halfB(p), 0, side);
      p.z = riverZ(p.x) + side * (laneHalf(p.x) + wide + across + 4 * rng());
      for (let k = 0; k < 12 && inLane(p, side, wide); k++) p.z += side * 1.5;   // where the river bends under the body, step back from it
      if (!inLane(p, side, wide) && !inSpawn(p) && !inApproach(p) && !clash(p, 0.95)) { placed.push(p); break; }
    }
  }
  // the canyon walls: three staggered rows each side
  for (const side of [-1, 1]) for (const [z0, z1, s0] of [[30, 46, 0.85], [54, 72, 1.0], [78, 98, 1.1]]) for (let x = 20 * rng(); x < P; x += 20 + 12 * rng()) {
    for (let t = 0; t < 10; t++) {
      const m = pick(x, (i) => z0 > 30 || (i !== 9 && i !== 3)), p = { m, x: x + 6 * (rng() - 0.5), z: side * (z0 + (z1 - z0) * rng()), ry: m === 4 ? 0.5 * (rng() - 0.5) + (rng() < 0.5 ? 0 : Math.PI) : turn(m, side), s: +(s0 + 0.3 * rng()).toFixed(2), zone: 'wall' };
      if (ring(p, halfB(p)).some(([, z]) => z * side < 14)) continue;   // the walls never close in on the corridor
      if (!inSpawn(p) && !inApproach(p) && !clash(p, 0.9)) { placed.push(p); break; }
    }
  }
  // the massifs: on a jittered grid out to the horizon, two to six formations stacked together, bigger with distance
  const WIDE = new Set([3, 4, 7, 8, 9]);                    // the broad formations stop growing sooner (a giant falls massif would fill the sky)
  for (const side of [-1, 1]) for (let zc = 112; zc < 600; zc += 50) for (let xc = 0; xc < P; xc += 56) {
    if (rng() < 0.08) continue;
    const cx = xc + 28 + 26 * (rng() - 0.5), cz = side * (zc + 22 * (rng() - 0.5)), sb = 1.05 + 1.2 * smooth(100, 480, Math.abs(cz)), n = 2 + Math.floor(rng() * 5), group = [];
    for (let k = 0; k < n; k++) for (let t = 0; t < 8; t++) {
      const m = pick(cx, () => true), p = { m, x: cx, z: cz, ry: rng() * TAU, s: +(Math.min(WIDE.has(m) ? 1.6 : 2.4, sb * (0.75 + 0.4 * rng()))).toFixed(2), zone: 'massif' };
      if (group.length) { const q = group[Math.floor(rng() * group.length)], a = rng() * TAU, d = (0.5 + 0.3 * rng()) * (Math.max(...halfB(q)) + Math.max(...halfB(p))); p.x = q.x + d * Math.cos(a); p.z = q.z + d * Math.sin(a); }
      if (Math.abs(p.z) - Math.max(...halfW(p)) < 62 || clash(p, group.length ? 0.45 : 0.65)) continue;
      placed.push(p); group.push(p); break;
    }
  }
  for (const p of placed) { p.x = ((p.x % P) + P) % P; p.z = +p.z.toFixed(2); p.x = +p.x.toFixed(2); p.ry = +p.ry.toFixed(3); }
  // preload: every model's finer levels stream in the background after the distant ones (13 models: all stay resident)
  const set = {
    models: AV_MODELS.map((m) => (m.old ? OLD : TL) + m.n + '.glb'), mid: AV_MODELS.map((m) => (m.old ? OLD : TL) + m.n + '_mid.glb'), bands: [20, 60, 190], range: 440, preload: true, maxNear: 16, far: AV_MODELS.map((m) => (m.old ? OLD : TL) + m.n + '_far.glb'), xfar: AV_MODELS.map((m) => (m.old ? OLD : TL) + m.n + '_xfar.glb'),
    fits: AV_MODELS.map((m) => m.S), halves: AV_MODELS.map((m) => [m.w[0] / 2 * m.S, m.w[1] / 2 * m.S]), placements: placed, period: P, zones: AV_ZONES, canyon: 0.12, valley: true, sky: 1.4,   // sky: more sky light on the rock (the low sun backlights the valley)
  };
  const g = valleyGround(set);                             // each formation's base: the lowest ground under its rim, less the rim's depth
  for (const p of placed) { const m = AV_MODELS[p.m]; p.y = +(Math.min(...ring(p, halfW(p), 20).map(([x, z]) => g(x, z))) - m.sink * m.S * p.s).toFixed(2); }
  // waterfalls: about one in four of the tall formations beside the corridor, and a few of the giants far out, pour from
  // the lip that faces the flight path; each: its lip (map x, height above y0, z), the outward direction, drop and width
  set.falls = [];
  for (const p of placed) {
    const M = AV_MODELS[p.m], H = M.S * p.s * (p.sy || 1), lips = AV_LIPS[p.m], near = p.zone === 'bank' || p.zone === 'wall', giant = p.zone === 'massif' && H > 70 && Math.abs(p.z) < 260;
    if (!lips.length || H < 28 || !(near ? rng() < 0.4 : giant && rng() < 0.2)) continue;
    let best = null, bs = -2;
    for (const [a, r, v, land] of lips) { const [ox, oz] = world({ ...p, x: 0, z: 0 }, Math.cos(a), Math.sin(a)), toward = -Math.sign(p.z) * oz; if (toward > bs) { bs = toward; best = [a, r, v, land, ox, oz]; } }
    if (!best || bs < -0.2) continue;
    const [a, r, v, land, ox, oz] = best, rr = r * M.S * p.s, [lx, lz] = world(p, rr * Math.cos(a), rr * Math.sin(a) * (p.sd || 1));
    set.falls.push({ x: +(((lx % P) + P) % P).toFixed(2), z: +lz.toFixed(2), y: +(p.y + v * H).toFixed(2), drop: +((v - land) * H).toFixed(2), w: +((0.05 + 0.04 * rng()) * M.S * p.s).toFixed(2), dir: +Math.atan2(oz, ox).toFixed(3) });
  }
  for (const p of tunnels) {                                // a curtain over the mouths of the Heaven's Gate and the waterfall gate: the ship bursts through it into the rock
    if (p.m === 11) continue; const t = AV_MODELS[p.m].tunnel, H = AV_MODELS[p.m].S * p.s, top = p.y + (t.v1 + 0.12) * H;
    set.falls.push({ x: +(p.x + t.e0 * H * p.sd - 1.2).toFixed(2), z: +(p.z - t.cu * H + 0.3 * (t.u1 - t.u0) * H).toFixed(2), y: +top.toFixed(2), drop: +(top - Math.max(0, p.y + t.v0 * H - 6)).toFixed(2), w: +(0.055 * H).toFixed(2), dir: +Math.PI.toFixed(3), mouth: 1 });   // off to one side of the mouth: the ship clips its edge
  }
  set.tunnels = tunnels.map((p) => { const t = AV_MODELS[p.m].tunnel, H = AV_MODELS[p.m].S * p.s;   // the bores in map space (ry = π/2: model z → +x, model x → −z)
    return { m: p.m, x0: +(p.x + t.e0 * H * p.sd).toFixed(2), x1: +(p.x + t.e1 * H * p.sd).toFixed(2), z: +(p.z - t.cu * H).toFixed(2), y: +(p.y + t.cv * H).toFixed(2), hw: +((t.u1 - t.u0) / 2 * H).toFixed(2), hh: +((t.v1 - t.v0) / 2 * H).toFixed(2) }; });
  return set;
}
