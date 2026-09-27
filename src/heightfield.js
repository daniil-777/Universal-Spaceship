// Procedural mountain range under the corridor (atmospheric flight), shared by the environment and the renderer.
// Periodic in x (period 2·xHalf), domain-warped ridged multi-fractal noise with a valley floor; heights in world units
// above the ground plane y0, scaled so the highest summit of every seed stands exactly `amplitude` above the floor. Sampled on a grid once (cheap bilinear lookups for the 55 sensor beams × 128 environments).
import { mulberry32 } from './mathx.js';
import { GRID as CHINA_GRID, SCALE as CHINA_SCALE } from './china_grid.js';

export const PERIOD = 120, ZSPAN = 90, NX = 240, NZ = 180;   // grid: 0.5-unit cells over x ∈ [−60, 60), z ∈ [−45, 45]
export function makeSampler(h, y0) {                       // bilinear world (x, z) → ground height (world y); periodic in x, floor beyond z
  return (x, z) => {
    let fx = ((x % PERIOD) + PERIOD) % PERIOD / PERIOD * NX, fz = (z / ZSPAN + 0.5) * (NZ - 1);
    if (fz <= 0) return y0 + h[0 * NX + Math.floor(fx) % NX]; if (fz >= NZ - 1) return y0;
    const i0 = Math.floor(fx) % NX, i1 = (i0 + 1) % NX, j0 = Math.floor(fz), j1 = Math.min(NZ - 1, j0 + 1), tx = fx - Math.floor(fx), tz = fz - j0;
    const a = h[j0 * NX + i0], b = h[j0 * NX + i1], c = h[j1 * NX + i0], d = h[j1 * NX + i1];
    return y0 + (a + (b - a) * tx) + ((c + (d - c) * tx) - (a + (b - a) * tx)) * tz;
  };
}
// A long city (the megacity): the grid is `nx` columns over `period` units, and `lap.shift` (a whole number of corridor
// periods, advanced by the environment each time the ship wraps) slides the 120-unit corridor along it.
export function makeLongSampler(h, y0, nx, period, lap) {
  return (x, z) => {
    let fx = (((x + lap.shift) % period) + period) % period / period * nx, fz = (z / ZSPAN + 0.5) * (NZ - 1);
    if (fz <= 0) return y0 + h[Math.floor(fx) % nx]; if (fz >= NZ - 1) return y0;
    const i0 = Math.floor(fx) % nx, i1 = (i0 + 1) % nx, j0 = Math.floor(fz), j1 = Math.min(NZ - 1, j0 + 1), tx = fx - Math.floor(fx), tz = fz - j0;
    const a = h[j0 * nx + i0], b = h[j0 * nx + i1], c = h[j1 * nx + i0], d = h[j1 * nx + i1];
    return y0 + (a + (b - a) * tx) + ((c + (d - c) * tx) - (a + (b - a) * tx)) * tz;
  };
}
// Tunnel roofs over a long world: the underside of the rock above each tunnel cell (Infinity where the sky is open);
// the lowest of the four cells around (x, z), so a roof never sags between cells into the ship's clearance
export function makeLongCeiling(c, y0, nx, period, lap) {
  return (x, z) => {
    const fx = (((x + lap.shift) % period) + period) % period / period * nx, fz = (z / ZSPAN + 0.5) * (NZ - 1);
    if (fz < 0 || fz > NZ - 1) return Infinity;
    const i0 = Math.floor(fx) % nx, i1 = (i0 + 1) % nx, j0 = Math.floor(fz), j1 = Math.min(NZ - 1, j0 + 1);
    const m = Math.min(c[j0 * nx + i0], c[j0 * nx + i1], c[j1 * nx + i0], c[j1 * nx + i1]); return m === Infinity ? Infinity : y0 + m;
  };
}
function makeNoise(seed, PX = 8, PZ = 16) {                // periodic-in-x value noise on a lattice of `PX` cells per period
  const rng = mulberry32(seed), table = new Float32Array(PX * PZ);
  for (let i = 0; i < table.length; i++) table[i] = rng();
  return (x, z) => {                                      // x in periods, z in cells
    const xi = Math.floor(x), zi = Math.floor(z), fx = x - xi, fz = z - zi, sx = fx * fx * (3 - 2 * fx), sz = fz * fz * (3 - 2 * fz);
    const X0 = ((xi % PX) + PX) % PX, X1 = (X0 + 1) % PX, Z0 = ((zi % PZ) + PZ) % PZ, Z1 = (Z0 + 1) % PZ;
    const a = table[Z0 * PX + X0], b = table[Z0 * PX + X1], c = table[Z1 * PX + X0], d = table[Z1 * PX + X1];
    return (a + (b - a) * sx) + ((c + (d - c) * sx) - (a + (b - a) * sx)) * sz;
  };
}
function fillPillars(h, seed) {                             // Zhangjiajie: sandstone pillars with irregular footprints and near-vertical flanks rising from low wooded hills
  const rng = mulberry32(seed * 31 + 7), nb = makeNoise(seed + 3, 10), nb2 = makeNoise(seed + 5, 24), P = [];
  for (let k = 0; k < 72; k++) {                          // most columns stay below the flight band; only the tallest sixth rises into it
    const r = 1.2 + Math.pow(rng(), 1.3) * 3.6, asp = 0.7 + rng() * 0.6, a = rng() * Math.PI;
    P.push({ x: rng() * PERIOD, z: (rng() * 2 - 1) * 36 * Math.sqrt(rng()), rx: r * asp, rz: r / asp, ca: Math.cos(a), sa: Math.sin(a), h: 0.25 + Math.pow(rng(), 2.4) * 0.75, p1: rng() * 6.283, p2: rng() * 6.283, reach: r * 1.7 });
  }
  for (let j = 0; j < NZ; j++) for (let i = 0; i < NX; i++) {
    const u = i / NX, x = u * PERIOD, z = (j / (NZ - 1) - 0.5) * ZSPAN;
    let m = 0.10 * nb(u * 10, z / 9 + 50) + 0.05 * nb2(u * 24, z / 4 + 20);
    for (const q of P) {
      let dx = x - q.x; if (dx > PERIOD / 2) dx -= PERIOD; else if (dx < -PERIOD / 2) dx += PERIOD;
      const dz = z - q.z; if (Math.abs(dx) > q.reach || Math.abs(dz) > q.reach) continue;
      const ex = (dx * q.ca + dz * q.sa) / q.rx, ez = (-dx * q.sa + dz * q.ca) / q.rz, d = Math.hypot(ex, ez), th = Math.atan2(ez, ex);
      const t = d / (1 + 0.22 * Math.sin(3 * th + q.p1) + 0.12 * Math.sin(5 * th + q.p2)); if (t >= 1.06) continue;
      const hp = q.h * (1 - 0.08 * t * t) * Math.min(1, (1.06 - t) / 0.24) * (0.92 + 0.08 * nb2(u * 24 + 3, z / 3));   // a domed top, flanks that drop within a quarter radius
      if (hp > m) m = hp;
    }
    h[j * NX + i] = m;
  }
}
export function decodeGrid(h, grid, scale, combine = false) {   // baked Meshy heights (base64 bytes, 255 = `scale` units) into h, optionally keeping the higher of the two
  if (!grid) return;
  const bin = atob(grid); for (let i = 0; i < h.length && i < bin.length; i++) { const v = bin.charCodeAt(i) / 255 * scale; if (!combine || v > h[i]) h[i] = v; }
}
function fillMeshy(h) { decodeGrid(h, CHINA_GRID, CHINA_SCALE); }   // the Meshy pillar clusters (see src/china_grid.js); empty until baked
export function createHeightField({ seed = 5, amplitude = 20, y0 = -26, style = 'alps' } = {}) {   // style: 'alps' (ridged range), 'pillars' (procedural Zhangjiajie) or 'meshy' (the baked Meshy clusters)
  const n1 = makeNoise(seed), n2 = makeNoise(seed + 7, 18), n3 = makeNoise(seed + 13, 40), nw = makeNoise(seed + 29, 10);
  const h = new Float32Array(NX * NZ);
  const ridge = (v) => 1 - Math.abs(2 * v - 1);
  if (style === 'pillars') fillPillars(h, seed); else if (style === 'meshy') fillMeshy(h); else for (let j = 0; j < NZ; j++) for (let i = 0; i < NX; i++) {
    const u = i / NX, z = (j / (NZ - 1) - 0.5) * ZSPAN;                                      // u: fraction of the period
    const uw = u + 0.045 * (nw(u * 10 + 5, z / 11 + 40) - 0.5), zw = z + 4.0 * (nw(u * 10 + 2, z / 11 + 3) - 0.5);   // domain warp: ridges bend instead of following the lattice
    const r1 = Math.pow(ridge(n1(uw * 8, zw / 14 + 50)), 1.4), r2 = ridge(n2(uw * 18, zw / 6 + 20)), r3 = n3(u * 40, z / 2.5 + 9);
    let m = 0.66 * r1 + 0.24 * r2 * r1 + 0.10 * r3 * (0.3 + 0.7 * r1);                          // rounded ridges, detail only on the high ground
    const range = 0.3 + 0.7 * n1(u * 4 + 3, 77);                                        // slow variation along the corridor: massifs and passes
    const belt = Math.exp(-(z * z) / (2 * 24 * 24));                                          // the range fades toward the corridor sides
    m = Math.max(0, m * range * (0.3 + 0.7 * belt) - 0.08);
    h[j * NX + i] = Math.pow(m, 1.3);                                                          // convex shaping: wide valleys, prominent massifs
  }
  let hmax = 0; for (let i = 0; i < h.length; i++) hmax = Math.max(hmax, h[i]);
  if (hmax > 0) for (let i = 0; i < h.length; i++) h[i] *= amplitude / hmax;                    // every world's summit reaches the same height
  let peak = 0; for (let i = 0; i < h.length; i++) peak = Math.max(peak, h[i]);
  return { height: makeSampler(h, y0), y0, amplitude, peak: y0 + peak, grid: h, NX, NZ, PERIOD, ZSPAN, style };
}
