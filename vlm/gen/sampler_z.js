// vlm/gen/sampler_z.js — the split-aware Z view plan (spec §3.4, §8): level-3 blocks assigned 80/10/10 by expected view
// weight (level-5 tiles inherit), an OOD region (NE Oceania land plus New Guinea, buffered 300 km, 0.5 deg raster; a cell
// holding a polygon vertex counts too, so islets smaller than a cell keep their buffer), 200 locations x 4 views with
// range/tilt/heading/Sun re-drawn per view and redrawn until the widened footprint grid lies in the location's split.
// Node-side; the build re-checks the rule on the captured grid.
import { lonLatToTile, MAX_LAT } from '../../src/earthtiles.js';
import { mulberry32 } from '../../src/mathx.js';
import { isLand, pointInFeature, pointsOf, nameOf } from './geo/naturalearth.js';
import { poseCamera, matrixGrid, sunAngles, rMinKm } from './labels/zoom.js';
import { rangeBin, RANGE_EDGES_KM } from './schema.js';

const DEG = Math.PI / 180, NG_BOX = [130.8, -10.8, 150.9, -0.8];
export const SPLIT_SHARE = Object.freeze({ train: 0.8, val: 0.1, test: 0.1 });
// spec §3.4: range log-uniform in [r_min, 2500 km]; the split check samples 32x18 widened by one cell (spec §8)
export const RANGE_MAX_KM = 2500, CHECK_GRID = Object.freeze({ nx: 32, ny: 18, margin: 1 });
export function blockKey(lat, lon, z = 3) { const t = lonLatToTile(((lon + 540) % 360) - 180, Math.max(-MAX_LAT, Math.min(MAX_LAT, lat)), z), n = 2 ** z; return `${z}/${((Math.floor(t.x) % n) + n) % n}/${Math.min(n - 1, Math.max(0, Math.floor(t.y)))}`; }
// A block over 10 % of the weight may go only to train and is credited to it first. The others, in seeded order, go to
// the split with the largest relative shortfall against 80/10/10; val or test takes a block only while it stays within
// 1 point of its share (without that cap a mid-size block overshoots: on NE-like weights 3 of seeds 1-8 missed val or
// test by more than 2.5 points; with it, 0 of seeds 1-1000 did, worst 2.35).
export function assignBlocks(seed, weights) {
  const rng = mulberry32(seed), keys = [...weights.keys()].sort(), total = [...weights.values()].reduce((a, b) => a + b, 0), got = { train: 0, val: 0, test: 0 }, out = new Map();
  for (let i = keys.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [keys[i], keys[j]] = [keys[j], keys[i]]; }
  const big = (k) => weights.get(k) > 0.1 * total;
  for (const k of keys) if (big(k)) { out.set(k, 'train'); got.train += weights.get(k); }
  for (const k of keys) {
    if (big(k)) continue;
    const w = weights.get(k); let best = 'train', most = -Infinity;
    for (const s of ['val', 'test', 'train']) {
      if (s !== 'train' && got[s] + w > (SPLIT_SHARE[s] + 0.01) * total) continue;
      const short = 1 - got[s] / (SPLIT_SHARE[s] * total);
      if (short > most) { best = s; most = short; }
    }
    out.set(k, best); got[best] += w;
  }
  return out;
}
export function buildOodMask(ne, { cellDeg = 0.5, bufferKm = 300 } = {}) {
  const nx = Math.round(360 / cellDeg), ny = Math.round(180 / cellDeg), inside = new Uint8Array(nx * ny), out = new Uint8Array(nx * ny);
  const cell = (lat, lon) => Math.min(ny - 1, Math.max(0, Math.floor((90 - lat) / cellDeg))) * nx + ((Math.floor((lon + 180) / cellDeg) % nx) + nx) % nx;
  const inNg = (lon, lat) => lon >= NG_BOX[0] && lon <= NG_BOX[2] && lat >= NG_BOX[1] && lat <= NG_BOX[3];
  const oceania = (ne.layers.ne_10m_admin_0_countries || []).filter((f) => f.props.CONTINENT === 'Oceania'), land = ne.layers.ne_10m_land || [];
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const lon = -180 + (i + 0.5) * cellDeg, lat = 90 - (j + 0.5) * cellDeg;
    if ((inNg(lon, lat) && land.some((f) => pointInFeature(f, lon, lat))) || oceania.some((f) => pointInFeature(f, lon, lat))) inside[j * nx + i] = 1;
  }
  for (const f of oceania) for (const [lon, lat] of pointsOf(f.geom)) inside[cell(lat, lon)] = 1;
  for (const f of land) for (const [lon, lat] of pointsOf(f.geom)) if (inNg(lon, lat)) inside[cell(lat, lon)] = 1;
  const dy = Math.ceil(bufferKm / 111.2 / cellDeg);
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    if (!inside[j * nx + i]) continue;
    const lat = 90 - (j + 0.5) * cellDeg, dx = Math.ceil(bufferKm / (111.2 * Math.max(0.05, Math.cos(lat * DEG))) / cellDeg);
    for (let v = -dy; v <= dy; v++) for (let u = -dx; u <= dx; u++) { const jj = j + v; if (jj < 0 || jj >= ny || (u * u) / (dx * dx || 1) + (v * v) / (dy * dy || 1) > 1) continue; out[jj * nx + (((i + u) % nx) + nx) % nx] = 1; }
  }
  return { has: (lat, lon) => out[cell(lat, lon)] === 1 };
}
export function viewSplit(grid, blocks, ood) {
  const pts = grid.filter(Boolean); if (!pts.length) return null;
  const nOod = pts.filter((p) => ood.has(p.lat, p.lon)).length;
  if (nOod === pts.length) return 'ood';
  if (nOod) return null;
  const s = new Set(pts.map((p) => blocks.get(blockKey(p.lat, p.lon)) ?? 'train'));
  return s.size === 1 ? [...s][0] : null;
}
// The Sun (spec §3.4): the wanted elevation is uniform in [-18°, 70°] on a random day of 2026. The time is where the
// Sun crosses it, rising or setting by a coin flip (the other if the day has only one); a day that never reaches it
// gives the time of the closest attainable elevation (its highest or lowest Sun). 5-min samples, crossings interpolated.
export function drawSunUtc(rng, lat, lon) {
  const want = -18 + rng() * 88, day = Date.UTC(2026, 0, 1) + Math.floor(rng() * 365) * 86400e3, rising = rng() < 0.5, step = 300e3, e = [], up = [], down = [];
  for (let k = 0; k <= 288; k++) e.push(sunAngles(day + k * step, lat, lon).elev - want);
  for (let k = 0; k < 288; k++) {
    const a = e[k], b = e[k + 1], t = Math.min(day + 86399e3, day + (k + a / (a - b)) * step);
    if (a <= 0 && b > 0) up.push(t); else if (a > 0 && b <= 0) down.push(t);
  }
  const hits = rising ? (up.length ? up : down) : (down.length ? down : up);
  if (hits.length) return Math.round(hits[0] / 1000) * 1000;
  let kb = 0; for (let k = 1; k < 288; k++) if (Math.abs(e[k]) < Math.abs(e[kb])) kb = k;
  return day + kb * step;
}
// The place mix of spec §3.4. 'land' and 'coast' points are drawn area-uniformly with sin(lat) in [-0.93, 0.95], so lat
// in [-68.4°, 71.8°]: most of Antarctica and the high Arctic are out, while places, peaks and regions keep their own
// latitudes. The 10 % coast/ocean share draws coast-only sea points (NE sea with NE land 0.5° of longitude east or west),
// never open ocean. After 500 misses the place falls back to the Matterhorn.
function drawPlace(rng, ne, used) {
  const u = rng(), pick = (arr) => arr[Math.floor(rng() * arr.length)];
  if (u < 0.35) { const pp = (ne.layers.ne_10m_populated_places || []).filter((f) => (f.props.POP_MAX ?? 0) >= 50000); const w = pp.map((f) => 1 / (1 + (used.get(f.props.ADM0NAME) || 0))); let r = rng() * w.reduce((a, b) => a + b, 0); for (let i = 0; i < pp.length; i++) { r -= w[i]; if (r <= 0) return { kind: 'place', name: pp[i].props.NAME, country: pp[i].props.ADM0NAME, lon: pp[i].point[0], lat: pp[i].point[1] }; } }
  if (u < 0.55) { const f = pick(ne.layers.ne_10m_geography_regions_elevation_points || []); if (f) return { kind: 'peak', name: nameOf(f.props), lon: f.point[0], lat: f.point[1] }; }
  if (u < 0.70) { const f = pick(ne.layers.ne_10m_geography_regions_polys || []); if (f) return { kind: 'region', name: nameOf(f.props), lon: f.point[0], lat: f.point[1] }; }
  for (let i = 0; i < 500; i++) { const lat = Math.asin(-0.93 + 1.88 * rng()) / DEG, lon = -180 + 360 * rng(), land = isLand(ne, lon, lat); if (u < 0.9 ? land : !land && [1, -1].some((d) => isLand(ne, lon + d * 0.5, lat))) return { kind: u < 0.9 ? 'land' : 'coast', name: null, lon, lat }; }
  return { kind: 'land', name: null, lon: 7.658, lat: 45.976 };
}
// the log-uniform share of each range bin over [r_min, 2500 km]
export function rangeShares(rMin) {
  const e = [rMin, ...RANGE_EDGES_KM, RANGE_MAX_KM], L = Math.log(RANGE_MAX_KM / rMin);
  return [0, 1, 2, 3, 4].map((b) => Math.max(0, Math.log(e[b + 1] / Math.max(rMin, e[b]))) / L);
}
// Ranges are log-uniform in [r_min, 2500 km], stratified: per split, each view adds its location's bin shares to the
// split's expected counts and takes the bin with the largest deficit, drawing log-uniformly inside it. A bin that fails
// 50 redraws is counted in `failed` and the view tries the next bin. `unfilled` lists the split|bin cells still at least
// one whole view short of the quota at the end; `quota` holds the expected and filled counts per split.
export function planZoom({ seed, nLocations = 200, ne, blocks, ood, profile = 'open' }) {
  const rng = mulberry32(seed), Lmax = profile === 'open' ? 15 : 14, used = new Map(), locations = [], failed = {}, quota = {}, edges = [0, ...RANGE_EDGES_KM, RANGE_MAX_KM];
  for (let id = 0; id < nLocations; id++) {
    const place = drawPlace(rng, ne, used); used.set(place.country, (used.get(place.country) || 0) + 1);
    const split = ood.has(place.lat, place.lon) ? 'ood' : blocks.get(blockKey(place.lat, place.lon)) ?? 'train', views = [];
    const rMin = rMinKm(place.lat, Lmax), share = rangeShares(rMin), Q = (quota[split] ||= { expected: [0, 0, 0, 0, 0], filled: [0, 0, 0, 0, 0] });
    for (let v = 0; v < 4; v++) {
      share.forEach((x, b) => { Q.expected[b] += x; });
      const order = [0, 1, 2, 3, 4].filter((b) => share[b] > 0).sort((a, b) => Q.expected[b] - Q.filled[b] - (Q.expected[a] - Q.filled[a]) || a - b);
      let ok = null;
      for (const bin of order) {
        const lo = Math.max(rMin, edges[bin]), hi = edges[bin + 1];
        for (let t = 0; t < 50 && !ok; t++) {
          const view = { lat: place.lat, lon: place.lon, rangeKm: Math.exp(Math.log(lo) + rng() * Math.log(hi / lo)), tilt: rng() < 0.4 ? 0 : rng() * 60 * DEG, heading: rng() * 2 * Math.PI };
          if (viewSplit(matrixGrid(poseCamera(view), view, CHECK_GRID), blocks, ood) === split) ok = view;
        }
        if (ok) { Q.filled[bin]++; break; }
        failed[`${split}|${bin}`] = (failed[`${split}|${bin}`] || 0) + 1;
      }
      if (!ok) continue;
      const alwaysDay = rng() < 0.3; views.push({ ...ok, utcMs: alwaysDay ? Date.UTC(2026, 5, 21, 12) : drawSunUtc(rng, place.lat, place.lon), alwaysDay, range_bin: rangeBin(ok.rangeKm) });
    }
    locations.push({ id, split, place, views });
  }
  const unfilled = {};
  for (const [s, Q] of Object.entries(quota)) Q.expected.forEach((x, b) => { const d = Math.floor(x - Q.filled[b] + 1e-9); if (d >= 1) unfilled[`${s}|${b}`] = d; });
  return { locations, unfilled, failed, quota };
}
