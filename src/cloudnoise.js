// A tileable 3D noise for the clouds (N³ RGBA8, built once on the first descent): R = Perlin–Worley — a value-noise fbm
// remapped by an inverted Worley fbm, the billowy base of cumulus — and G = a Worley fbm for fine erosion and rain streaks.
// Every lattice wraps with period N, so the texture repeats without a seam.
import { mulberry32, clamp } from './mathx.js';

export function cloudNoiseData(N = 64, seed = 7) {
  const rng = mulberry32(seed), data = new Uint8Array(N * N * N * 4);
  const pts = (c) => Float32Array.from({ length: c * c * c * 3 }, () => rng()), lat = (c) => Float32Array.from({ length: c * c * c }, () => rng());
  const W4 = pts(4), W8 = pts(8), W16 = pts(16), W32 = pts(32), L4 = lat(4), L8 = lat(8), L16 = lat(16), m = (a, c) => ((a % c) + c) % c;
  const worley = (u, v, w, c, P) => {                      // distance to the nearest feature point of a c³ periodic grid
    const fx = u * c, fy = v * c, fz = w * c, ix = Math.floor(fx), iy = Math.floor(fy), iz = Math.floor(fz); let md = 9;
    for (let dz = -1; dz <= 1; dz++) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const cx = ix + dx, cy = iy + dy, cz = iz + dz, k = ((m(cz, c) * c + m(cy, c)) * c + m(cx, c)) * 3;
      const ex = cx + P[k] - fx, ey = cy + P[k + 1] - fy, ez = cz + P[k + 2] - fz, d = ex * ex + ey * ey + ez * ez; if (d < md) md = d; }
    return Math.min(1, Math.sqrt(md));
  };
  const vnoise = (u, v, w, c, G) => {                      // periodic value noise, smoothstep-interpolated
    const fx = u * c, fy = v * c, fz = w * c, ix = Math.floor(fx), iy = Math.floor(fy), iz = Math.floor(fz), tx = fx - ix, ty = fy - iy, tz = fz - iz;
    const sx = tx * tx * (3 - 2 * tx), sy = ty * ty * (3 - 2 * ty), sz = tz * tz * (3 - 2 * tz), at = (x, y, z) => G[(m(z, c) * c + m(y, c)) * c + m(x, c)], l = (a, b, t) => a + (b - a) * t;
    const row = (y, z) => l(at(ix, y, z), at(ix + 1, y, z), sx);
    return l(l(row(iy, iz), row(iy + 1, iz), sy), l(row(iy, iz + 1), row(iy + 1, iz + 1), sy), sz);
  };
  for (let z = 0; z < N; z++) for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const u = x / N, v = y / N, w = z / N, o = ((z * N + y) * N + x) * 4;
    const val = 0.5 * vnoise(u, v, w, 4, L4) + 0.3 * vnoise(u, v, w, 8, L8) + 0.2 * vnoise(u, v, w, 16, L16);
    const wf = 0.625 * (1 - worley(u, v, w, 4, W4)) + 0.25 * (1 - worley(u, v, w, 8, W8)) + 0.125 * (1 - worley(u, v, w, 16, W16));
    const det = 0.625 * (1 - worley(u, v, w, 8, W8)) + 0.25 * (1 - worley(u, v, w, 16, W16)) + 0.125 * (1 - worley(u, v, w, 32, W32));
    data[o] = Math.round(255 * clamp((val - (wf - 1)) / (2 - wf), 0, 1)); data[o + 1] = Math.round(255 * clamp(det, 0, 1)); data[o + 3] = 255;
  }
  return data;
}

// The noise's R channel as the GPU samples it (linear filter, repeat, texel centres; 0..1), for the physics' copy of the
// cloud lobes (src/weather.js) — the same 64³ texture the renderer draws with, built here on first use (~0.5 s).
let NOISE = null; const NN = 64;
export function cloudNoiseR(u, v, w) {
  const d = NOISE || (NOISE = cloudNoiseData(NN, 7)), m = (i) => ((i % NN) + NN) % NN;
  const fx = u * NN - 0.5, fy = v * NN - 0.5, fz = w * NN - 0.5, x0 = Math.floor(fx), y0 = Math.floor(fy), z0 = Math.floor(fz), tx = fx - x0, ty = fy - y0, tz = fz - z0;
  const xa = m(x0), xb = m(x0 + 1), ya = m(y0) * NN, yb = m(y0 + 1) * NN, za = m(z0) * NN * NN, zb = m(z0 + 1) * NN * NN, at = (x, y, z) => d[(z + y + x) * 4];
  const l = (a, b, t) => a + (b - a) * t, row = (y, z) => l(at(xa, y, z), at(xb, y, z), tx);
  return l(l(row(ya, za), row(yb, za), ty), l(row(ya, zb), row(yb, zb), ty), tz) / 255;
}
