// The weather of an atmospheric world (never in space — see env.atmosphere): a mean wind that grows with height, veers a
// little and swings with slow gust fronts; cloud cells laid along the world's loop and drifting with the wind — flat-based
// fair-weather cumulus under the thin-air ceiling, towering cumulus reaching down into the valley with violent cores and
// rain beneath, stratus banks hugging the valley floor — each with an updraft and the sinking ring around it; air rising
// up windward slopes and sinking in the lee, with rotor turbulence there; and the turbulence intensity σ that drives the
// Dryden filters (src/turbulence.js). Deterministic per seed. The cell list and cellShape()/rainShape() are mirrored in
// GLSL (src/weatherglsl.js), so the clouds on screen are the air the physics flies through. x is render space (the
// corridor's coordinate); a long world's lap shift maps it onto the loop.
import { mulberry32, randn, clamp } from './mathx.js';
import { makeLongSampler } from './heightfield.js';
import { cloudNoiseR } from './cloudnoise.js';

export const CELL = Object.freeze({ CUMULUS: 0, TOWERING: 1, STRATUS: 2 });
const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
export function cellShape(type, R, base, top, dx, y, dz) {              // flat base, domed top (a sheet twice as long in x for stratus)
  const h = (y - base) / (top - base); if (h < 0 || h > 1) return 0;
  const ez = type === CELL.STRATUS ? dz * 2 : dz, r = Math.sqrt(dx * dx + ez * ez) / R, shrink = type === CELL.STRATUS ? 1 : 1 - 0.55 * h * h, rr = r / Math.max(shrink, 0.05);
  return (1 - smooth(0.6, 1.0, rr)) * smooth(0, 0.06, h) * (1 - smooth(0.85, 1.0, h));
}
export function rainShape(type, R, base, groundY, dx, y, dz) {         // towering cells: a rain shaft under the base
  if (type !== CELL.TOWERING || y > base || y < groundY) return 0;
  const r = Math.sqrt(dx * dx + dz * dz) / (0.5 * R); return (1 - smooth(0.5, 1.0, r)) * smooth(groundY, groundY + 4, y);
}
// The lobes the renderer draws (src/clouds.js): a Perlin–Worley noise pushes each cell's edge out or in (cauliflower
// sides, a lumpy dome, the flat base kept; fog sheets get ragged edges). The physics uses the same shape, so the air is what you see.
export const LOBE = Object.freeze({ scale: 1 / 30, amount: 0.7, median: 0.62, boil: 0.004 });   // 1/30: periodic in 120 (lap slides are seamless)
export const lobeG = (r) => (clamp((r - 0.3) / 0.62, 0, 1) - LOBE.median) * LOBE.amount;         // r: the noise's R channel, 0..1
export function cellLobed(type, R, base, top, dx, y, dz, g) {
  const st = type === CELL.STRATUS, gi = st ? g * 0.7 : g; return cellShape(type, R * (1 + gi), base + (st ? 0 : gi * 0.8), top + gi * (st ? 0.15 : 0.3) * (top - base), dx, y, dz);   // fog sheets: ragged edges, a flat floor
}
const wrap = (d, P) => d - P * Math.round(d / P), BUCKET = 40, ZS = 3, EPS = 0.5;   // ZS, EPS: the outflow's scale height and ground-friction depth (Oseguera–Bowles)
// The ground low-passed (three box passes ≈ a Gaussian of σ ≈ 10 units), once per height grid: the wind follows the lie of
// the land, not every wall, pillar and roof (a raw 4-unit gradient made 3–5 u/s drafts in one step over a skyline).
const SMOOTH = new WeakMap();
function smoothGround(hf, floor) {
  if (!hf) return () => floor; if (!hf.grid) return (x, z) => hf.height(x, z);
  const nx = hf.NX, nz = hf.NZ, r = 20; let a = SMOOTH.get(hf.grid);
  if (!a) {
    a = Float32Array.from(hf.grid); const b = new Float32Array(nx * nz), w = 2 * r + 1, m = (i) => ((i % nx) + nx) % nx, cz = (j) => Math.min(nz - 1, Math.max(0, j));
    for (let pass = 0; pass < 3; pass++) {
      for (let j = 0; j < nz; j++) { const o = j * nx; let sum = 0; for (let k = -r; k <= r; k++) sum += a[o + m(k)]; for (let i = 0; i < nx; i++) { b[o + i] = sum / w; sum += a[o + m(i + r + 1)] - a[o + m(i - r)]; } }
      for (let i = 0; i < nx; i++) { let sum = 0; for (let k = -r; k <= r; k++) sum += b[cz(k) * nx + i]; for (let j = 0; j < nz; j++) { a[j * nx + i] = sum / w; sum += b[cz(j + r + 1) * nx + i] - b[cz(j - r) * nx + i]; } }
    }
    SMOOTH.set(hf.grid, a);
  }
  return makeLongSampler(a, hf.y0, nx, hf.PERIOD || 120, hf.lap || { shift: 0 });
}

// orographic: coherent lift up windward slopes (terrain; a skyline gets rotor turbulence only). carry: a previous sky of
// the same world whose clock, drift and gust front go on (the Playbox changing severity, wind or cover mid-flight).
export const SKIES = { clear: [0.2, 0.6, 0, 0.5], fair: [0.5, 1, 1, 1], cloudy: [0.55, 1, 2, 1], storm: [1, 1.6, 1.6, 2] };   // Playbox / ?sky= presets: [severity, wind, cover, turbulence]; fair = the default sky
export function createWeather({ seed = 1, severity = 0.5, wind = 1, cover = 1, turb = 1, period = 120, hf = null, ceiling = 12, floor = -26, zHalf = 18, windDir = null, orographic = true, carry = null } = {}) {
  const frng = mulberry32(seed + 7919), s = clamp(severity, 0, 1), cells = [];
  let rng = mulberry32(seed); const stream = (k) => { rng = mulberry32((seed ^ k) >>> 0); };   // one stream per kind of cell: a new severity adds or drops cells at the end, the rest stay
  const add = (type, R, base, top, W, sig, zSpan) => { const x = rng() * period, z = (rng() * 2 - 1) * zSpan; cells.push({ type, x, z, R, base, top, W, sig, seed: rng() }); };
  const nCu = cover > 0 ? Math.max(1, Math.round(period / 40 * (0.3 + 0.7 * s) * cover)) : 0;
  const nTw = Math.round(period / 200 * s * cover + (s >= 0.6 && period < 200 && cover > 0 ? 0.6 : 0));
  const nSt = hf ? Math.round(period / 150 * (0.3 + 0.5 * s) * cover) : 0;
  for (let i = 0; i < nCu; i++) { const R = 8 + 17 * rng(), base = ceiling - 2 - 4 * rng(), v = rng(); rng();   // one vigor per cumulus: a deeper cell has the stronger updraft and the rougher air (the radar can rank them); the spare draw keeps the cells where they were
    add(CELL.CUMULUS, R, base, base + (8 + 14 * v) * (0.6 + 0.4 * s), (0.5 + 1.5 * v) * (0.4 + 0.6 * s), (0.6 + 0.4 * s) * (0.7 + 0.6 * v), zHalf + 12); }
  stream(0x51ed27); for (let i = 0; i < nTw; i++) { const R = 10 + 8 * rng(), base = floor + 10 + 6 * rng(); add(CELL.TOWERING, R, base, ceiling + 40 + 30 * rng(), 3 + 3 * s, 2.5 + 1.5 * s, zHalf - 4); }
  stream(0x2545f491); for (let i = 0; i < nSt; i++) { const R = 20 + 25 * rng(), base = floor + 2 + 3 * rng(); add(CELL.STRATUS, R, base, base + 4 + 4 * rng(), 0, 0.15, zHalf); }
  const nDk = Math.round(period / 60 * (0.2 + 0.6 * s) * cover);   // a decorative deck above the thin-air ceiling: cumulus with no updraft, far out and high (looks only)
  stream(0x6c8e9cf5); for (let i = 0; i < nDk; i++) { const R = 20 + 25 * rng(), base = ceiling + 14 + 16 * rng(); add(CELL.CUMULUS, R, base, base + 6 + 8 * rng(), 0, 0.3, zHalf + 60); }
  const nB = Math.max(1, Math.ceil(period / BUCKET)), buckets = Array.from({ length: nB }, () => []);   // a cell sits in every bucket its reach (±2R) touches
  for (const c of cells) for (let b = Math.floor((c.x - 2 * c.R) / BUCKET); b <= Math.floor((c.x + 2 * c.R) / BUCKET); b++) { const k = ((b % nB) + nB) % nB; if (!buckets[k].includes(c)) buckets[k].push(c); }
  const dir = windDir ?? mulberry32((seed ^ 0x3c6ef372) >>> 0)() * 2 * Math.PI, U0 = (2 + 7 * s) * wind;
  let front = carry ? carry.front : 1, drift = carry ? carry.drift : 0, t = carry ? carry.time : 0, lead = 0, la = 0;   // lead, la: lookAhead's drift and time
  const shift = () => (hf && hf.lap ? hf.lap.shift : 0), ground = smoothGround(hf, floor);
  const near = (mx, dr) => buckets[Math.floor(((((mx - dr) % period) + period) % period) / BUCKET) % nB];
  const vprof = (c, y) => (y < c.base ? clamp((y - (c.base - 15)) / 15, 0, 1) : y > c.top ? 0 : 1 - 0.6 * (y - c.base) / (c.top - c.base));
  const sinkAt = (c, y, za) => (ZS * (1 - Math.exp(-za / ZS)) - EPS * (1 - Math.exp(-za / EPS))) / (ZS - EPS) * (1 - smooth(0.55, 0.9, (y - c.base) / (c.top - c.base)));   // the downdraft's profile: 0 at the ground, full a few units up, gone in the storm's upper half
  const slope = (x, z, out) => { const e = 3; out[0] = (ground(x + e, z) - ground(x - e, z)) / (2 * e); out[1] = (ground(x, z + e) - ground(x, z - e)) / (2 * e); return out; };
  const lobe = (x, y, z, dr, tt) => lobeG(cloudNoiseR((x - dr) * LOBE.scale + tt * LOBE.boil, y * LOBE.scale, z * LOBE.scale));   // the renderer's lobe at a point (it drifts with the cells)
  const _g = [0, 0], pool = [], _tb = {}; let gx = NaN, gz = NaN, gy = 0;
  const groundSlope = (x, z) => { if (x !== gx || z !== gz) { gx = x; gz = z; gy = ground(x, z); if (hf) slope(x, z, _g); } return gy; };   // windAt and sigmaAt ask about the same point
  return {
    cells, period, floor, ceiling, severity: s, dir, U0,
    get time() { return t; }, get drift() { return drift; }, get front() { return front; },
    advance(h) { t += h; front = clamp(front + h * (1 - front) / 60 + Math.sqrt(h) * 0.03 * randn(frng), 0.5, 1.5); drift = (drift + h * U0 * front * Math.cos(dir)) % period; },
    lookAhead(dt) { la = dt; lead = dt * U0 * front * Math.cos(dir); },   // windAt/turbAt see the cells as they will be dt from now (the physics samples a decision's end); 0 restores
    windAt(x, y, z, out) {
      const gs = groundSlope(x, z), za = Math.max(0, y - gs), hA = Math.max(1, za), prof = clamp(Math.log(hA / 0.5) / Math.log(40), 0, 1.3), veer = dir + 0.26 * clamp(hA / 40, 0, 1);
      const U = U0 * front * prof * (hf && orographic ? 1 + 0.5 * clamp((gs - floor) / 20, 0, 1) * Math.exp(-hA / 15) : 1);   // faster over the high ground
      out[0] = U * Math.cos(veer); out[2] = U * Math.sin(veer);
      let w = 0; const mx = x + shift(), dr = drift + lead;
      for (const c of near(mx, dr)) { if (c.type === CELL.STRATUS) continue;
        const ex = wrap(mx - dr - c.x, period), ez = z - c.z, d = Math.sqrt(ex * ex + ez * ez), r = d / c.R, v = vprof(c, y);
        if (c.type === CELL.TOWERING) {                   // a mature storm: an updraft ring round the rain shaft's downdraft, which spreads along the ground (Oseguera–Bowles)
          w += c.W * (v * (Math.exp(-(((r - 0.8) / 0.3) ** 2)) - 0.35 * Math.exp(-(((r - 1.4) / 0.35) ** 2))) - 0.9 * sinkAt(c, y, za) * Math.exp(-((r / 0.45) ** 2)));
          if (d > 1e-6) { const Rd = 0.45 * c.R, ur = 0.9 * c.W * Rd * Rd / (2 * d * (ZS - EPS)) * (1 - Math.exp(-((d / Rd) ** 2))) * (Math.exp(-za / ZS) - Math.exp(-za / EPS)); out[0] += ur * ex / d; out[2] += ur * ez / d; }
        } else if (v > 0) w += c.W * v * (Math.exp(-r * r) - 0.35 * Math.exp(-(((r - 1.4) / 0.35) ** 2))); }
      if (hf && orographic) w += clamp(out[0] * _g[0] + out[2] * _g[1], -4, 4) * Math.exp(-hA / 20);   // up the windward slope, down the lee
      out[1] = w; return out;
    },
    turbAt(x, y, z, out) {                                // the Dryden filters' input: σ_w, the height above the ground, the cloud's share of σ² and its length scale (0.4 R)
      const za = Math.max(0, y - groundSlope(x, z)), hA = Math.max(1, za), U = U0 * front, mx = x + shift();
      let sc = 0, Lc = 0; const dr = drift + lead, g = lobe(x, y, z, dr, t + la);
      for (const c of near(mx, dr)) { const ex = wrap(mx - dr - c.x, period), ez = z - c.z; let d = cellLobed(c.type, c.R, c.base, c.top, ex, y, ez, g) * c.sig;
        if (c.type === CELL.TOWERING) {                   // the shear round a storm's shaft is rough: 0.5·|∂w/∂r|·0.4R, at most the cloud's own
          const r = Math.sqrt(ex * ex + ez * ez) / c.R, v = vprof(c, y), e1 = Math.exp(-(((r - 0.8) / 0.3) ** 2)), e3 = Math.exp(-(((r - 1.4) / 0.35) ** 2));
          const dw = c.W * (v * (-2 * (r - 0.8) / 0.09 * e1 + 0.7 * (r - 1.4) / 0.1225 * e3) + 1.8 * sinkAt(c, y, za) * r / 0.2025 * Math.exp(-((r / 0.45) ** 2)));
          d = Math.hypot(d, Math.min(c.sig, 0.2 * Math.abs(dw))); }
        if (d > sc) { sc = d; Lc = 0.4 * c.R; } }
      let rotor = 0; if (hf && U > 0.1) { rotor = clamp(-(Math.cos(dir) * _g[0] + Math.sin(dir) * _g[1]), 0, 1.5) * 0.3 * U * Math.exp(-hA / 15); }
      const bg = (0.15 + 0.45 * s) * (1 - 0.5 * smooth(ceiling - 6, ceiling - 2, y));   // clear air, calmer above the cumulus bases
      const mech = 0.15 * U * Math.exp(-hA / 15), shear = clamp(1.5 * U / (hA * Math.log(40)), 0, 1.5), s2 = bg * bg + sc * sc + mech * mech + rotor * rotor + shear * shear;
      out.sigma = Math.sqrt(s2) * turb; out.hA = hA; out.share = sc * sc / s2; out.Lc = Lc; return out;   // turb: the turbulence knob (1 = as modelled)
    },
    sigmaAt(x, y, z) { return this.turbAt(x, y, z, _tb).sigma; },
    cloudAt(x, y, z) { const mx = x + shift(), g = lobe(x, y, z, drift, t); let d = 0; for (const c of near(mx, drift)) d = Math.max(d, cellLobed(c.type, c.R, c.base, c.top, wrap(mx - drift - c.x, period), y, z - c.z, g)); return d; },
    cellsNear(x, range, out = []) {
      out.length = 0; const mx = x + shift(), sh = shift();
      for (const c of cells) { const cx = c.x + drift, reach = range + 2 * c.R;
        for (let k = Math.ceil((mx - reach - cx) / period); k <= Math.floor((mx + reach - cx) / period); k++) {
          const o = pool[out.length] || (pool[out.length] = {}); o.type = c.type; o.x = cx + k * period - sh; o.z = c.z; o.R = c.R; o.base = c.base; o.top = c.top; o.sig = c.sig; o.W = c.W; o.seed = c.seed; out.push(o); } }
      return out;
    },
  };
}
