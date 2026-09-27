// Where the Meshy-generated Zhangjiajie pillar clusters stand along the periodic corridor — shared by the renderer
// (src/meshyworld.js) and the offline bake that turns the placed meshes into the environment's height grid, so the
// picture and the collision field are the same thing. Units: corridor units (period 120 in x, z ∈ [−45, 45]).
import { avatarValley } from './avatarlayout.js';
export const MESHY_MODELS = ['textures/meshy/pillars_a.glb', 'textures/meshy/pillars_b.glb'];
export const MESHY_SUMMIT = 40;                            // the tallest cluster's summit above the ground (= the Alps' and the towers' ceiling)

// model index, x, z, rotation about y (rad), scale relative to the tallest cluster; the lane |z| < 14 has four clusters
// to weave through, the flanks hold the rest, and the first 16 units of the period (the ship's spawn) stay clear
export const MESHY_PLACEMENTS = [
  { m: 0, x: -30, z: 6, ry: 0.4, s: 1.0 }, { m: 1, x: 0, z: -8, ry: 2.1, s: 0.9 }, { m: 0, x: 30, z: 4, ry: 3.6, s: 0.85 }, { m: 1, x: 55, z: -10, ry: 1.2, s: 0.95 },
  { m: 1, x: -50, z: -22, ry: 0.9, s: 0.8 }, { m: 0, x: -15, z: 24, ry: 2.8, s: 0.9 }, { m: 1, x: 15, z: -26, ry: 0.2, s: 0.75 }, { m: 0, x: 40, z: 22, ry: 1.7, s: 0.95 },
  { m: 1, x: -40, z: 20, ry: 4.1, s: 0.7 }, { m: 0, x: 50, z: -30, ry: 2.4, s: 0.85 }, { m: 1, x: -5, z: 34, ry: 0.6, s: 0.65 }, { m: 0, x: 20, z: -38, ry: 3.1, s: 0.7 },
];

// District sets for the city routes: square Meshy tiles (one prompt-generated diorama each, streets along all four
// edges) joined edge to edge on a street grid across the corridor — `grid` gives the tile size and the rows the seam
// streets run between — with bigger district plates from the user's photos beyond them. fits: { width } = a tile (the
// plate spans `width`; `inner`/`rot` for a plate generated inside a larger base at an angle) or a number = the summit.
// `far` = the distant level of detail. Placements: model, x, z, rotation, scale, optional vertical stretch `sy`.
const Q = Math.PI / 2, TL = 'textures/meshy/tiles/', R = (m, x, z, r, s = 1, sy) => ({ m, x, z, ry: r * Q, s, ...(sy ? { sy } : {}) });
export const MESHY_SETS = {
  china: { canyon: 0.12, preload: true, models: MESHY_MODELS, far: ['textures/meshy/pillars_a_far.glb', 'textures/meshy/pillars_b_far.glb'], xfar: ['textures/meshy/pillars_a_xfar.glb', 'textures/meshy/pillars_b_xfar.glb'], fits: [MESHY_SUMMIT, MESHY_SUMMIT], halves: [[26, 20], [27, 27]], placements: MESHY_PLACEMENTS },
  newyork: {                                               // 40-unit tiles: Lower Manhattan (One World Trade Center), Midtown East (Chrysler), Hudson Yards; the Midtown-from-the-Hudson plate lines both flanks, the Empire State and a Billionaires' Row pencil tower in its plaza
    models: [TL + 'ny_fidi.glb', TL + 'ny_midtown_east.glb', TL + 'ny_hudson_yards.glb', 'textures/meshy/ny_hudson.glb', 'textures/meshy/ny_midtown.glb', 'textures/meshy/ny_billionaires.glb'],
    far: [TL + 'ny_fidi_far.glb', TL + 'ny_midtown_east_far.glb', TL + 'ny_hudson_yards_far.glb', 'textures/meshy/ny_hudson_far.glb', 'textures/meshy/ny_midtown_far.glb', 'textures/meshy/ny_billionaires_far.glb'],
    xfar: [TL + 'ny_fidi_xfar.glb', TL + 'ny_midtown_east_xfar.glb', TL + 'ny_hudson_yards_xfar.glb', 'textures/meshy/ny_hudson_xfar.glb', 'textures/meshy/ny_midtown_xfar.glb', 'textures/meshy/ny_billionaires_xfar.glb'],
    fits: [{ width: 40 }, { width: 40 }, { width: 40 }, 32, 40, 42], halves: [[20, 20], [20, 20], [20, 20], [51.85, 51.73], [4.07, 2.05], [2.09, 1.71]],
    grid: { tile: 40, rows: [-40, 0, 40] }, preload: true, maxNear: 8,   // preload: every model's near level streams in the background and stays
    placements: [R(0, 20, 0, 0), R(1, 60, 0, 1), R(2, 100, 0, 2), R(2, 20, 40, 1), R(0, 60, 40, 2), R(1, 100, 40, 3), R(1, 20, -40, 2), R(2, 60, -40, 3), R(0, 100, -40, 1),
      R(3, 60, 120, 1, 1.154), R(3, 60, -120, -1, 1.154), { m: 4, x: 44, z: 129, ry: 0.2, s: 1 }, { m: 5, x: 53, z: 111, ry: 0.1, s: 1 }],
  },
  dubai: {                                                 // 60-unit tiles: Downtown (the Burj Khalifa: its city plate is 62 % of a model turned 45° on a sand plinth), Sheikh Zayed Road (Emirates Towers, Museum of the Future), Business Bay; the Marina plate on both flanks
    models: [TL + 'dubai_downtown.glb', TL + 'dubai_szr.glb', TL + 'dubai_business_bay.glb', 'textures/meshy/dubai_marina.glb'],
    far: [TL + 'dubai_downtown_far.glb', TL + 'dubai_szr_far.glb', TL + 'dubai_business_bay_far.glb', 'textures/meshy/dubai_marina_far.glb'],
    xfar: [TL + 'dubai_downtown_xfar.glb', TL + 'dubai_szr_xfar.glb', TL + 'dubai_business_bay_xfar.glb', 'textures/meshy/dubai_marina_xfar.glb'],
    fits: [{ width: 60, inner: 0.62, rot: Math.PI / 4 }, { width: 60 }, { width: 60 }, 36], halves: [[30, 30], [30, 30], [30, 30], [34.6, 34.97]],
    grid: { tile: 60, rows: [-60, 0, 60] }, preload: true, maxNear: 8,
    placements: [R(0, 30, 0, 0), R(1, 90, 0, 0), R(2, 30, 60, 1), R(1, 90, 60, 2), R(1, 30, -60, 3), R(2, 90, -60, 0),
      R(3, 30, 125, 0, 0.867), R(3, 90, 125, 2, 0.867), R(3, 30, -125, 1, 0.867), R(3, 90, -125, 3, 0.867)],
  },
  moscow: {                                                // 60-unit tiles: Moscow City (Evolution, Federation, Mercury), a Stalinist high-rise, Red Square with the Kremlin wall and St Basil's; two more rows beyond
    models: [TL + 'moscow_city.glb', TL + 'moscow_stalinist.glb', TL + 'moscow_red_square.glb'],
    far: [TL + 'moscow_city_far.glb', TL + 'moscow_stalinist_far.glb', TL + 'moscow_red_square_far.glb'],
    xfar: [TL + 'moscow_city_xfar.glb', TL + 'moscow_stalinist_xfar.glb', TL + 'moscow_red_square_xfar.glb'],
    fits: [{ width: 60 }, { width: 60 }, { width: 60 }], halves: [[30, 30], [30, 30], [30, 30]],
    grid: { tile: 60, rows: [-120, -60, 0, 60, 120] }, preload: true, maxNear: 8,
    placements: [R(0, 30, 0, 0), R(1, 90, 0, 1), R(1, 30, 60, 2), R(0, 90, 60, 3), R(2, 30, -60, 0), R(1, 90, -60, 3),
      R(0, 30, 120, 2), R(1, 90, 120, 0), R(1, 30, -120, 1), R(0, 90, -120, 1)],
  },
  london: {                                                // 60-unit tiles: Westminster (Big Ben), the City (the Gherkin, the Walkie-Talkie), the Shard quarter, their stone and Docklands twins, and the New York–London fusion; five rows
    models: ['london_westminster', 'london_city', 'london_shard', 'london_city_stone', 'london_shard_docklands', 'fusion_ny_london'].map((n) => TL + n + '.glb'), far: ['london_westminster', 'london_city', 'london_shard', 'london_city_stone', 'london_shard_docklands', 'fusion_ny_london'].map((n) => TL + n + '_far.glb'), xfar: ['london_westminster', 'london_city', 'london_shard', 'london_city_stone', 'london_shard_docklands', 'fusion_ny_london'].map((n) => TL + n + '_xfar.glb'),
    fits: Array.from({ length: 6 }, () => ({ width: 60 })), halves: Array.from({ length: 6 }, () => [30, 30]),
    grid: { tile: 60, rows: [-120, -60, 0, 60, 120] }, preload: true, maxNear: 8,
    placements: [R(1, 30, 0, 0), R(2, 90, 0, 1), R(0, 30, 60, 2), R(3, 90, 60, 3), R(4, 30, -60, 1), R(5, 90, -60, 0),
      R(3, 30, 120, 0), R(2, 90, 120, 2), R(1, 30, -120, 3), R(4, 90, -120, 2)],
  },
};
// The megacity: every district on one long map — New York, London, Moscow and Dubai zones in a loop, each border bridged
// by a fusion district that mixes the two cities' architecture — seven rows across (the corridor's three and two more on
// each flank, 420 units wide) and 960 units (eight corridor laps, over a minute of flight) before it repeats. Variety:
// Meshy restyled twins (the same geometry re-textured in another style), remixes (two districts cut along a street
// each and joined — the second slid so the streets meet), skyline stretches, and one Paris plaza per zone for the
// Eiffel Towers. Slots are filled by a seeded shuffle: no district twice in a row or column, landmarks at most twice.
const MEGA = ['ny_fidi', 'ny_midtown_east', 'ny_hudson_yards', 'london_westminster', 'london_city', 'london_shard', 'moscow_city', 'moscow_stalinist', 'moscow_red_square',
  'dubai_downtown', 'dubai_szr', 'dubai_business_bay', 'fusion_ny_london', 'fusion_london_moscow', 'fusion_moscow_dubai', 'fusion_dubai_ny',
  'ny_hudson_yards_bronze', 'ny_midtown_east_brick', 'london_city_stone', 'london_shard_docklands', 'moscow_city_gold', 'moscow_stalinist_brick', 'dubai_business_bay_sand', 'dubai_szr_marble', 'paris'];
const ZONES = [[0, 1, 2, 16, 17], [3, 4, 5, 18, 19], [6, 7, 8, 20, 21], [9, 10, 11, 22, 23]], FUSION = [12, 13, 14, 15], LANDMARK = new Set([0, 3, 8, 9]), PARIS = 24;   // landmarks: One WTC, Westminster, Red Square, the Burj
const TWIN = { 16: 2, 17: 1, 18: 4, 19: 5, 20: 6, 21: 7, 22: 11, 23: 10 };                 // a restyled twin shares its original's geometry
const BASE = (m) => TWIN[m] ?? m;
// street lines to cut on, in the tile's own frame (the harness's cuts_meshy.mjs: the lowest line within 10 units of the centre, kept where it is a street — under ~4 units)
const CUTS = { 0: ['z', 9.5], 1: ['z', -7], 2: ['x', 0.5], 3: ['z', 10], 6: ['z', 8.5], 8: ['x', 0], 9: ['x', -4], 10: ['z', -7.5], 11: ['z', -3], 13: ['x', 5.5], 14: ['z', 0], 15: ['x', 0] };
const pose = ([axis, c], flip) => (axis === 'z' ? (flip ? [2, -c] : [0, c]) : (flip ? [1, -c] : [3, c]));   // quarter turns that lay the street along world x, and where it then lies (world z)
const rectFor = (k, lo, hi) => (k === 0 ? [-1e5, lo, 1e5, hi] : k === 1 ? [-hi, -1e5, -lo, 1e5] : k === 2 ? [-1e5, -hi, 1e5, -lo] : [lo, -1e5, hi, 1e5]);   // world z ∈ [lo, hi] in the tile's own frame
function remix(a, b, x, z, sa, sb) {                       // a keeps the south of its street, b the north, slid so the two streets meet
  let best = null;
  for (const fa of [0, 1]) for (const fb of [0, 1]) {
    const [ka, wa] = pose(CUTS[BASE(a)], fa), [kb, wb] = pose(CUTS[BASE(b)], fb), d = wa - wb;
    if (Math.abs(wa) <= 16 && (!best || Math.abs(d) < Math.abs(best.d))) best = { ka, wa, kb, wb, d };
  }
  const { ka, wa, kb, wb, d } = best, Q = Math.PI / 2;
  return [{ m: a, x, z, ry: ka * Q, s: 1, sy: sa, clip: rectFor(ka, -1e5, wa) }, { m: b, x, z: z + d, ry: kb * Q, s: 1, sy: sb, clip: rectFor(kb, wb, 30 - d) }];
}
function megacity() {
  let seed = 20260926; const rng = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
  const rows = [-180, -120, -60, 0, 60, 120, 180], cols = 16, grid = [], placements = [], landmarks = new Map(), stretch = () => +(0.85 + 0.35 * rng()).toFixed(2);
  for (let c = 0; c < cols; c++) {
    const zone = Math.floor(c / 4), fusionCol = c % 4 === 3, next = (zone + 1) % 4; grid.push([]);
    for (let r = 0; r < rows.length; r++) {
      const x = 30 + 60 * c, z = rows[r];
      if (c % 4 === 1 && z === (zone % 2 ? -60 : 60)) { grid[c].push(PARIS); placements.push({ m: PARIS, x, z, ry: (zone % 2) * Math.PI, s: 1, plaza: zone }); continue; }   // the zone's Paris plaza (an Eiffel Tower stands in its lawn)
      let pool = fusionCol ? (r === 0 || r === rows.length - 1 ? ZONES[zone] : r % 2 ? [FUSION[zone]] : [FUSION[zone], ...ZONES[next]]) : ZONES[zone];
      const left = (grid[c - 1] || [])[r], up = grid[c][r - 1];
      pool = pool.filter((m) => m !== left && m !== up && !(LANDMARK.has(m) && ((landmarks.get(m) || 0) >= 2 || Math.abs(z) > 120)));
      if (!pool.length) { const plain = (fusionCol ? [FUSION[zone], ...ZONES[zone]] : ZONES[zone]).filter((m) => !LANDMARK.has(m)); pool = plain.filter((m) => m !== left && m !== up); if (!pool.length) pool = plain.filter((m) => m !== left); if (!pool.length) pool = plain; }
      const m = pool[Math.floor(rng() * pool.length)]; grid[c].push(m); if (LANDMARK.has(m)) landmarks.set(m, (landmarks.get(m) || 0) + 1);
      const cuttable = ZONES[zone].filter((q) => BASE(q) !== BASE(m) && !LANDMARK.has(q) && CUTS[BASE(q)] !== undefined);
      if (!fusionCol && !LANDMARK.has(m) && CUTS[BASE(m)] !== undefined && cuttable.length && rng() < 0.3) { placements.push(...remix(m, cuttable[Math.floor(rng() * cuttable.length)], x, z, stretch(), stretch())); continue; }
      placements.push({ m, x, z, ry: Math.floor(rng() * 4) * Math.PI / 2, s: 1, ...(LANDMARK.has(m) || fusionCol ? {} : { sy: stretch() }) });
    }
  }
  // Eiffel Towers (src/eiffel.js, built in the browser): each zone's Paris plaza gets one in the zone's style, and two giants
  // on Paris plazas in the outer rows stand over the skyline as landmarks from anywhere in the corridor
  const towers = [['chrome', 60], ['classic', 60], ['red', 60], ['ivory', 60], ['gold', 88], ['twisted', 88]], T0 = MEGA.length;
  for (const p of placements.filter((q) => q.plaza !== undefined)) { const k = p.ry ? -1 : 1; placements.push({ m: T0 + p.plaza, x: p.x + 2 * k, z: p.z + 2 * k, ry: p.ry, s: 1 }); }   // in the courtyard (its centre is 2 units off the tile's)
  for (const [i, x, z] of [[4, 330, 180], [5, 810, -180]]) {
    const at = placements.findIndex((q) => q.x === x && q.z === z); if (at >= 0) placements.splice(at, 1);
    placements.push({ m: PARIS, x, z, ry: 0, s: 1 }, { m: T0 + i, x: x + 2, z: z + 2, ry: 0, s: 1 });
  }
  const desc = towers.map(([variant, height]) => ({ proc: 'eiffel', variant, height }));
  return {
    models: MEGA.map((n) => TL + n + '.glb').concat(desc), far: MEGA.map((n) => TL + n + '_far.glb').concat(desc), xfar: MEGA.map((n) => TL + n + '_xfar.glb').concat(desc),
    fits: MEGA.map((n) => (n === 'dubai_downtown' ? { width: 60, inner: 0.62, rot: Math.PI / 4 } : { width: 60 })).concat(desc.map(() => null)), halves: MEGA.map(() => [30, 30]).concat(desc.map((d) => [d.height * 0.2, d.height * 0.2])),
    period: 960, grid: { tile: 60, rows }, placements, zones: ['New York', 'London', 'Moscow', 'Dubai'], maxNear: 12, prefetch: 170,   // 31 models: near levels by proximity, fetched early
  };
}
MESHY_SETS.mega = megacity();
// the China route's Avatar valley (src/avatarlayout.js): laid out on first use (~0.5 s), not on every page load
let avatar = null; Object.defineProperty(MESHY_SETS, 'avatar', { enumerable: false, get: () => avatar || (avatar = avatarValley()) });

export function inMeshyFootprint(set, x, z) {              // is (x, z) inside any placed chunk's footprint (periodic in x)? the box cities skip such spots
  if (!set) return false;
  for (const p of set.placements) { const [hw, hd] = set.halves[p.m]; let dx = x - p.x; dx -= 120 * Math.round(dx / 120); const dz = z - p.z, c = Math.cos(p.ry), s = Math.sin(p.ry);
    const lx = dx * c + dz * s, lz = -dx * s + dz * c; if (Math.abs(lx) < hw * p.s + 1 && Math.abs(lz) < hd * p.s + 1) return true; }
  return false;
}
