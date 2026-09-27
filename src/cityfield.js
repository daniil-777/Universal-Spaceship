// Skylines as obstacles. Three stylised cities — Dubai, New York, Moscow — built from axis-aligned building footprints
// (boxes, round and octagonal towers, tapered towers, setback stacks)
// (landmarks placed by hand, filler blocks from a seeded generator), rasterised into the same periodic height field the
// mountains use, so the environment's collision test, sensor beams and ground clearance work unchanged. The renderer
// draws the boxes themselves. Units: the corridor's (1 unit ≈ 8 m at the ship's scale); heights above the ground y0.
import { mulberry32 } from './mathx.js';
import { makeSampler, makeLongSampler, makeLongCeiling, decodeGrid, PERIOD, ZSPAN, NX, NZ } from './heightfield.js';
import { MESHY_SETS, inMeshyFootprint } from './meshylayout.js';
import { GRID as NY_GRID, SCALE as NY_SCALE } from './newyork_grid.js';
import { GRID as DUBAI_GRID, SCALE as DUBAI_SCALE } from './dubai_grid.js';
import { GRID as MOSCOW_GRID, SCALE as MOSCOW_SCALE } from './moscow_grid.js';
import { GRID as LONDON_GRID, SCALE as LONDON_SCALE } from './london_grid.js';
const MESHY_GRIDS = { newyork: [NY_GRID, NY_SCALE], dubai: [DUBAI_GRID, DUBAI_SCALE], moscow: [MOSCOW_GRID, MOSCOW_SCALE], london: [LONDON_GRID, LONDON_SCALE] };

export const T = {                                          // facade tints (linear RGB)
  glass: [0.55, 0.68, 0.82], teal: [0.48, 0.72, 0.72], white: [0.86, 0.87, 0.9], gold: [0.82, 0.68, 0.42], silver: [0.76, 0.78, 0.82],
  stone: [0.72, 0.66, 0.56], brick: [0.56, 0.36, 0.3], concrete: [0.64, 0.63, 0.6], copper: [0.72, 0.42, 0.28], red: [0.62, 0.2, 0.18],
  cream: [0.88, 0.82, 0.68], spire: [0.86, 0.86, 0.88], slab: [0.74, 0.72, 0.66], dark: [0.28, 0.32, 0.38],
};
const pick = (rng, arr) => arr[Math.floor(rng() * arr.length) % arr.length];

function builder(seed, chunks = null) {                   // chunks: the city's Meshy district set — no procedural boxes inside its footprints
  const rng = mulberry32(seed), B = [];
  const box = (x, z, w, d, h, tint = T.glass, kind = 0, y = 0, shape = 0, taper = 1) => {   // shape: 0 box, 1 round, 2 octagon, 3 tapered box (top = taper × footprint, snapped to 0.4 / 0.6 / 0.8)
    if (h > 0.2 && !inMeshyFootprint(chunks, x, z)) B.push({ x, z, w, d, h, y, tint, kind, shape, taper: shape === 3 ? (taper < 0.5 ? 0.4 : taper < 0.7 ? 0.6 : 0.8) : 1 }); };
  const tiered = (x, z, tiers, tint, kind = 1, spire = 0) => {   // stacked setbacks (Empire State, the Stalinist towers, Burj Khalifa)
    let y = 0; for (const [w, d, h] of tiers) { box(x, z, w, d, h, tint, kind, y); y += h; }
    if (spire) box(x, z, 0.7, 0.7, spire, T.spire, 3, y); return y + spire;
  };
  const blocks = (x0, x1, z0, z1, bx, bz, st, hFn, tints, kind = 0) => {   // a block grid with streets `st` wide, 1–2 buildings per block
    for (let bxi = x0; bxi + bx <= x1 + 1e-6; bxi += bx + st) for (let bzi = z0; bzi + bz <= z1 + 1e-6; bzi += bz + st) {
      const cx = bxi + bx / 2, cz = bzi + bz / 2, h = hFn(cx, cz, rng); if (h <= 0) continue;
      const k = h > 18 ? kind : 1, sv = rng(), shape = h > 18 ? (sv < 0.16 ? 1 : sv < 0.28 ? 2 : sv < 0.40 ? 3 : 0) : 0, taper = shape === 3 ? 0.55 + rng() * 0.3 : 1;   // tall towers: some round, octagonal or tapered
      if (rng() < 0.35) for (let i = 0; i < 2; i++) box(bxi + bx * (0.26 + 0.48 * i), cz, bx * 0.44, bz * (0.55 + 0.4 * rng()), i ? Math.max(3, h * (0.4 + 0.5 * rng())) : h, pick(rng, tints), k);
      else box(cx + (rng() - 0.5) * 0.6, cz, bx * (0.62 + 0.34 * rng()), bz * (0.62 + 0.34 * rng()), h, pick(rng, tints), k, 0, shape, taper);
    }
  };
  const roofs = () => {                                    // rooftop plant and masts on the taller towers (they are obstacles too)
    for (const q of B.slice()) { if (q.kind === 3 || q.y > 0 || q.h < 20 || q.w < 3) continue;
      const top = q.y + q.h; if (rng() < 0.7) box(q.x + (rng() - 0.5) * q.w * 0.4, q.z + (rng() - 0.5) * q.d * 0.4, q.w * 0.35, q.d * 0.35, 0.8 + rng() * 0.8, T.concrete, 1, top);
      if (rng() < 0.45) box(q.x + q.w * 0.3, q.z - q.d * 0.3, 0.35, 0.35, 2 + rng() * 4, T.spire, 3, top); }
  };
  return { rng, B, box, tiered, blocks, roofs };
}

const CITIES = {
  dubai(b) {                                                // Sheikh Zayed Road runs along the corridor: tower rows both sides, Downtown and the Marina beside it
    const { rng, box, tiered, blocks } = b, glass = [T.glass, T.teal, T.white, T.gold, T.silver];
    for (let x = -34; x < 58; x += 6.5 + rng() * 2) for (const side of [-1, 1]) {   // the strip: towers 18–34 tall lining the road; the canyon narrows through Downtown
      if (rng() < 0.18) continue; const w = 3.5 + rng() * 2.5, squeeze = x > 4 && x < 30 ? 2.5 : 0;
      box(x + rng() * 2, side * (5.5 + rng() * 5 - squeeze), w, w * (0.8 + rng() * 0.5), 18 + rng() * 16, pick(rng, glass), 0);
    }
    { let y = 0; for (const [w, hh] of [[7.4, 10], [5.8, 9], [4.4, 8], [3.0, 7], [1.9, 5]]) { box(15, 13, w, w, hh, T.silver, 4, y, 2); y += hh; } box(15, 13, 0.7, 0.7, 5, T.spire, 3, y); }   // Burj Khalifa: octagonal setbacks, 44 units, into the thin air
    box(9, 18, 16, 10, 4, T.stone, 1); box(22, 19, 4.5, 4.5, 28, T.gold, 0); box(26, 15, 4, 4, 26, T.glass, 0); box(6, 24, 4, 4, 24, T.white, 0);   // the mall and the Address towers
    for (let gx = -30; gx < -12; gx += 6.5) for (const gz of [5, 11.5, 18, -9, -15.5, -22]) { if (rng() < 0.25) continue; const w = 4 + rng() * 1.5; box(gx + rng(), gz + rng(), w, w, 22 + rng() * 14, pick(rng, glass), 0); }   // the Marina and JLT clusters either side of the road
    box(-20, 8, 4.8, 4.8, 38, T.teal, 0, 0, 1);              // Princess Tower: round
    box(-46, -24, 3.2, 1.6, 24, T.white, 4, 0, 3, 0.35);     // Burj Al Arab: the sail, tapering to its top
    blocks(-30, 58, 14, 30, 6, 4, 2, (x, z, r) => (r() < 0.55 ? 3 + r() * 5 : 0), [T.stone, T.cream, T.concrete]);     // villas and low-rise inland
    blocks(-30, 58, -30, -14, 6, 4, 2, (x, z, r) => (r() < 0.5 ? 3 + r() * 6 : 0), [T.stone, T.cream, T.white]);
    blocks(-58, -36, 12, 30, 7, 4, 2, (x, z, r) => (r() < 0.4 ? 3 + r() * 4 : 0), [T.stone, T.cream]);                  // the approach: low coast
  },
  newyork(b) {                                              // Manhattan: a street grid, Central Park as a gap, Midtown super-talls beside the path, Downtown ahead
    const { rng, box, tiered, blocks } = b, mid = [T.glass, T.silver, T.stone, T.concrete, T.white], old = [T.brick, T.stone, T.concrete];
    const park = (x, z) => x > -44 && x < -24 && Math.abs(z) < 7;
    const hMid = (x, z, r) => { if (park(x, z)) return 0; const core = Math.exp(-((x - 4) * (x - 4)) / 500) * Math.exp(-(z * z) / 260), down = Math.exp(-((x - 45) * (x - 45)) / 160) * Math.exp(-(z * z) / 120);
      const p = 0.75 * core + 0.7 * down, u = r(); return u < 0.55 ? 6 + r() * 8 : u < 0.55 + 0.35 * p ? 16 + r() * 12 : u < 0.55 + 0.45 * p ? 28 + r() * 8 : 6 + r() * 6; };
    blocks(-24, 58, -28, 28, 8, 4, 3, hMid, mid);
    blocks(-58, -26, -28, 28, 8, 4, 3, (x, z, r) => (park(x, z) ? 0 : 5 + r() * 8), old);                              // Upper West Side / Harlem: mid-rise
    tiered(2, 3, [[9, 5, 12], [6.5, 4, 16], [4, 3, 8]], T.stone, 1, 4);      // Empire State 40
    tiered(10, -6, [[6, 6, 22], [4.5, 4.5, 6], [3, 3, 4]], T.silver, 4, 3);   // Chrysler 35
    tiered(6, 8, [[6, 6, 26], [4, 4, 10]], T.glass, 0, 2);                    // One Vanderbilt 38
    box(-2, -2, 6, 6, 34, T.glass, 0); box(-8, 6, 3, 3, 38, T.white, 4); box(-16, 5, 4, 4, 40, T.silver, 4); box(-14, 1, 2.4, 2.4, 36, T.white, 4);   // Bank of America, 432 Park, Central Park Tower, Steinway
    box(45, -4, 6.4, 6.4, 38, T.glass, 4, 0, 3, 0.55); box(45, -4, 0.7, 0.7, 4, T.spire, 3, 38);   // One World Trade Center: tapering glass, 42
    box(50, 4, 5, 5, 26, T.glass, 0, 0, 2); box(41, 3, 4, 4, 22, T.stone, 1); box(38, -8, 4, 4, 24, T.silver, 0, 0, 1);
  },
  moscow(b) {                                               // Moscow: suburbs, the University, the Kremlin and old centre, the Seven Sisters, Moscow City ahead
    const { rng, box, tiered, blocks } = b, sov = [T.slab, T.concrete, T.cream], mc = [T.glass, T.silver, T.teal];
    blocks(-58, -42, -28, 28, 12, 3, 4, (x, z, r) => (r() < 0.7 ? 6 + r() * 4 : 0), sov, 1);                            // the approach: rows of Soviet slabs
    tiered(-38, -22, [[16, 10, 8], [10, 7, 9], [6, 5, 7], [3.5, 3.5, 5]], T.cream, 1, 6); box(-47, -22, 5, 4, 12, T.cream, 1); box(-29, -22, 5, 4, 12, T.cream, 1);   // Moscow State University 35
    for (const [x, z, w, d] of [[-17, 8, 18, 0.6], [-17, 2, 18, 0.6], [-26, 5, 0.6, 6], [-8, 5, 0.6, 6]]) box(x, z, w, d, 2.5, T.red, 1);   // the Kremlin walls
    for (const [x, z] of [[-26, 2], [-26, 8], [-8, 2], [-8, 8]]) { box(x, z, 1.6, 1.6, 6, T.red, 1); box(x, z, 0.5, 0.5, 2.5, T.gold, 3, 6); }
    box(-17, 5, 5, 4, 6, T.white, 1); box(-17, 5, 0.6, 0.6, 3, T.gold, 3, 6); box(-19, 6.5, 1.4, 1.4, 9, T.white, 1); box(-19, 6.5, 0.5, 0.5, 3, T.gold, 3, 9);   // cathedrals and the bell tower
    blocks(-34, 6, -16, 16, 7, 5, 2.5, (x, z, r) => (x > -27 && x < -7 && z > 1 && z < 9 ? 0 : 3 + r() * 3.5), [T.cream, T.stone, T.brick], 1);   // the old centre around it
    tiered(-12, 20, [[10, 7, 8], [6, 5, 10], [3.5, 3.5, 6]], T.cream, 1, 3);    // Foreign Ministry 27
    tiered(0, -16, [[10, 7, 9], [6.5, 5, 11], [3.5, 3.5, 6]], T.cream, 1, 4);   // Hotel Ukraina 30
    tiered(-18, -8, [[8, 6, 8], [5.5, 4.5, 9], [3, 3, 5]], T.cream, 1, 4);      // Kudrinskaya 26
    tiered(30, 12, [[8, 6, 7], [5, 4, 8], [3, 3, 4]], T.cream, 1, 3);           // Red Gates 22
    tiered(40, -16, [[7, 5, 6], [4.5, 4, 7], [3, 3, 4]], T.cream, 1, 3);        // Leningradskaya 20
    tiered(22, 22, [[10, 6, 8], [6, 4.5, 9], [3.5, 3.5, 5]], T.cream, 1, 4);    // Kotelnicheskaya 26
    box(18, -2, 7.4, 7.4, 42, T.glass, 4, 0, 3, 0.6); box(13, 4, 5.2, 5.2, 36, T.copper, 4, 0, 3, 0.7); box(24, -8, 5, 5, 38, T.dark, 0); box(24, -3, 4, 4, 30, T.dark, 0);   // Moscow City: Federation (tapering), Mercury (copper, tapering), OKO
    box(27, 2, 5, 5, 34, T.glass, 0); box(10, -7, 5.2, 5.2, 30, T.teal, 0, 0, 1); box(20, 6, 5, 5, 32, T.silver, 0, 0, 2); box(16, 8, 4, 4, 28, T.silver, 0); box(13, -11, 4, 4, 28, T.glass, 0);   // Eurasia, Evolution, City of Capitals, Imperia
    for (let i = 0; i < 3; i++) box(29 + i * 3.2, -6, 2.8, 4, 24 - i * 4, T.glass, 0);   // Naberezhnaya
    blocks(34, 58, -28, 28, 6, 4, 3, (x, z, r) => (Math.abs(z) < 14 && x > 36 ? (r() < 0.5 ? 10 + r() * 8 : 5 + r() * 4) : (r() < 0.6 ? 6 + r() * 4 : 0)), sov.concat(mc), 0);   // eastern residential towers
    blocks(-34, 6, 18, 30, 8, 3, 3, (x, z, r) => (r() < 0.6 ? 6 + r() * 4 : 0), sov, 1); blocks(-34, 6, -30, -18, 8, 3, 3, (x, z, r) => (r() < 0.6 ? 6 + r() * 4 : 0), sov, 1);
  },
  london(b) {                                               // the Thames runs along the corridor: Westminster, the City on the north bank, the Shard south, Tower Bridge, Canary Wharf downstream
    const { rng, box, tiered, blocks } = b, old = [T.brick, T.stone, T.cream], glass = [T.glass, T.silver, T.teal], river = (x) => 4 * Math.sin(x / 25);   // the river's centre line bends gently
    const land = (h) => (x, z, r) => (Math.abs(z - river(x)) < 4.5 ? 0 : h(x, z, r));     // nothing in the Thames
    blocks(-58, -40, -30, 30, 7, 4, 2.5, land((x, z, r) => (r() < 0.75 ? 3 + r() * 3 : 0)), old, 1);                     // Chelsea and Battersea terraces
    box(-50, 9, 10, 6, 8, T.brick, 1); for (const [dx, dz] of [[-4.2, -2.4], [4.2, -2.4], [-4.2, 2.4], [4.2, 2.4]]) box(-50 + dx, 9 + dz, 0.9, 0.9, 7, T.cream, 1, 8, 1);   // Battersea Power Station
    box(-35, -5.5, 2, 2, 16, T.stone, 1); box(-35, -5.5, 0.6, 0.6, 4, T.spire, 3, 16); box(-30, -7, 12, 4, 6, T.stone, 1); box(-24.5, -8, 3, 3, 14, T.stone, 1);   // Big Ben, the Houses of Parliament, Victoria Tower
    box(-38, -12, 8, 4, 8, T.stone, 1); box(-42.5, -12, 2, 2, 12, T.stone, 1); box(-33.5, -12, 2, 2, 12, T.stone, 1);   // Westminster Abbey
    blocks(-40, -10, -28, -6, 7, 4, 2.5, (x, z, r) => (Math.abs(z - river(x)) < 4.5 ? 0 : 4 + r() * 3.5), [T.stone, T.cream, T.brick], 1);   // Whitehall, Covent Garden, Holborn
    blocks(-40, 30, 6, 30, 7, 4, 2.5, land((x, z, r) => (r() < 0.8 ? 3 + r() * 4 : 0)), old, 1);                         // the south bank: Lambeth, Southwark, Bermondsey
    box(-12, -9, 5, 5, 10, T.stone, 1, 0, 1); box(-12, -9, 3.2, 3.2, 4, T.stone, 1, 10, 1); box(-12, -9, 0.9, 0.9, 3, T.gold, 3, 14);   // St Paul's: drum, dome, lantern
    box(-3, -12, 6, 6, 40, T.glass, 4); box(2, -14, 5, 5, 34, T.glass, 0); box(0, -8.5, 5, 5, 30, T.teal, 4, 0, 1); box(6, -9, 6, 5, 36, T.silver, 4, 0, 3, 0.4);   // 22 Bishopsgate, Heron, the Gherkin, the Cheesegrater
    box(10, -6.5, 6, 4, 28, T.glass, 0); box(-6, -16, 4, 4, 30, T.silver, 0); box(9, -13, 5, 5, 32, T.glass, 4, 0, 3, 0.6); box(-7, -7, 4, 4, 24, T.stone, 1);   // the Walkie-Talkie, Tower 42, the Scalpel, a stone block
    box(2, 7, 6, 6, 42, T.glass, 4, 0, 3, 0.4); box(2, 7, 0.7, 0.7, 3, T.spire, 3, 42);                                    // the Shard
    box(13, -2.6, 2.4, 2.4, 12, T.stone, 1); box(13, 2.6, 2.4, 2.4, 12, T.stone, 1); box(13, 0, 1.4, 6.5, 1.4, T.stone, 1, 5.5);   // Tower Bridge: two towers and the deck across the river
    blocks(-10, 28, -30, -6, 7, 4, 2.5, land((x, z, r) => (x < 14 && z > -20 ? 0 : r() < 0.7 ? 5 + r() * 6 : 0)), old.concat(glass), 1);   // the City fringe and the East End
    box(38, -3, 6, 6, 34, T.silver, 4); box(38, -3, 4, 4, 4, T.silver, 4, 34, 3, 0.4); box(44, -6, 5, 5, 30, T.glass, 0); box(44, 1, 5, 5, 30, T.glass, 0);   // Canary Wharf: One Canada Square (pyramid), 8 and 25 Canada Square
    box(32, -7, 5, 5, 36, T.glass, 4); box(33, 3, 5, 5, 34, T.white, 4, 0, 3, 0.6); box(48, -1, 5, 5, 30, T.teal, 0, 0, 1); box(41, 6, 4, 4, 26, T.silver, 0);   // Landmark Pinnacle, Newfoundland, One Park Drive
    blocks(28, 58, -30, 30, 6, 4, 3, land((x, z, r) => (Math.abs(z) < 12 && x > 30 && x < 50 ? 0 : r() < 0.6 ? 6 + r() * 6 : 0)), [T.slab, T.concrete, T.brick], 1);   // Docklands estates
  },
};
export const CITY_NAMES = { dubai: 'Dubai', newyork: 'New York', moscow: 'Moscow', london: 'London', mega: 'Megacity' };

// Long worlds (the megacity, the Avatar valley) bring their own baked grid, too big to import up front: the page loads the
// module when the route is chosen and registers it here; training and evaluation register it before creating their environments.
const LONG = {};
export function registerLongGrid(city, { GRID, SCALE, NX: nx, PERIOD: period, TUNNEL = '', TUNNELS = [] }) { LONG[city] = { GRID, SCALE, nx, period, TUNNEL, TUNNELS }; }
export const hasLongGrid = (city) => !!LONG[city];
export function createLongField(city, y0) {                // city: the world's name (its Meshy set)
  const L = LONG[city], h = new Float32Array(L.nx * NZ), lap = { shift: 0 }; decodeGrid(h, L.GRID, L.SCALE);
  let peak = 0; for (let i = 0; i < h.length; i++) peak = Math.max(peak, h[i]);
  let floor = h, ceil = null;                               // tunnels: `grid` keeps the rock's top (shadows); collision uses the bore's floor and its roof
  if (L.TUNNEL) { floor = h.slice(); ceil = new Float32Array(h.length).fill(Infinity); const b = atob(L.TUNNEL);
    for (let r = 0; r + 5 < b.length; r += 6) { const k = b.charCodeAt(r) | (b.charCodeAt(r + 1) << 8) | (b.charCodeAt(r + 2) << 16) | (b.charCodeAt(r + 3) << 24); floor[k] = b.charCodeAt(r + 4) / 255 * L.SCALE; ceil[k] = b.charCodeAt(r + 5) / 255 * L.SCALE; } }
  return { height: makeLongSampler(floor, y0, L.nx, L.period, lap), ceiling: ceil ? makeLongCeiling(ceil, y0, L.nx, L.period, lap) : null, tunnels: L.TUNNELS || [],
    y0, amplitude: peak, peak: y0 + peak, grid: h, NX: L.nx, NZ, PERIOD: L.period, ZSPAN, buildings: [], city, chunks: city, lap, segments: Math.round(L.period / PERIOD) };
}

export function createCityField({ city = 'dubai', seed = 1, y0 = -26 } = {}) {
  if (LONG[city]) return createLongField(city, y0);
  const meshy = MESHY_GRIDS[city] && MESHY_GRIDS[city][0] ? MESHY_SETS[city] : null;   // the Meshy chunks count only once their grid has been baked
  const b = builder(seed * 7919 + 17, meshy); (CITIES[city] || CITIES.dubai)(b); b.roofs();
  const h = new Float32Array(NX * NZ), cell = PERIOD / NX, zc = ZSPAN / (NZ - 1);
  for (const q of b.B) {                                    // rasterise: each column keeps the highest roof above it (the footprint shape decides which columns)
    const i0 = Math.round((q.x - q.w / 2) / cell), i1 = Math.round((q.x + q.w / 2) / cell), j0 = Math.max(0, Math.round((q.z - q.d / 2) / zc + (NZ - 1) / 2)), j1 = Math.min(NZ - 1, Math.round((q.z + q.d / 2) / zc + (NZ - 1) / 2));
    const hw = q.w / 2, hd = q.d / 2, sh = q.shape || 0, tp = q.taper ?? 1;
    for (let j = j0; j < j1; j++) for (let i = i0; i < i1; i++) {
      const ux = Math.abs(i * cell - q.x) / hw, uz = Math.abs(-ZSPAN / 2 + j * zc - q.z) / hd; let cover = 1;
      if (sh === 1) { if (ux * ux + uz * uz > 1) continue; }                                     // round
      else if (sh === 2) { if (ux + uz > 1.414) continue; }                                      // octagon: corners cut (|x|+|z| ≤ 0.707 of the footprint)
      else if (sh === 3) { const u = Math.max(ux, uz); cover = u <= tp ? 1 : (1 - u) / (1 - tp); if (cover <= 0.02) continue; }   // tapered: the column ends where the slope passes
      const k = j * NX + (((i % NX) + NX) % NX), top = q.y + q.h * cover; if (top > h[k]) h[k] = top;
    }
  }
  if (meshy) decodeGrid(h, MESHY_GRIDS[city][0], MESHY_GRIDS[city][1], true);        // the district chunks' baked heights join the field
  let peak = 0; for (let i = 0; i < h.length; i++) peak = Math.max(peak, h[i]);
  return { height: makeSampler(h, y0), y0, amplitude: peak, peak: y0 + peak, grid: h, NX, NZ, PERIOD, ZSPAN, buildings: b.B, city, chunks: meshy ? city : null };
}
