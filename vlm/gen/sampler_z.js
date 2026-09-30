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
export function blockKey(lat, lon, z = 3) { const t = lonLatToTile(((lon + 540) % 360) - 180, Math.max(-MAX_LAT, Math.min(MAX_LAT, lat)), z), n = 2 ** z; return `${z}/${((Math.floor(t.x) % n) + n) % n}/${Math.min(n - 1, Math.max(0, Math.floor(t.y)))}`; }
export function assignBlocks(seed, weights) {
  const rng = mulberry32(seed), keys = [...weights.keys()].sort(), total = [...weights.values()].reduce((a, b) => a + b, 0), out = new Map();
  for (let i = keys.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [keys[i], keys[j]] = [keys[j], keys[i]]; }
  let acc = 0; for (const k of keys) { const s = acc < 0.1 * total ? 'val' : acc < 0.2 * total ? 'test' : 'train'; out.set(k, s); acc += weights.get(k); }
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
function drawUtc(rng, lat, lon) {
  for (let tries = 0; tries < 40; tries++) {
    const want = -18 + rng() * 88, day = Date.UTC(2026, 0, 1) + Math.floor(rng() * 365) * 86400e3, times = [];
    for (let m = 0; m < 1440; m += 10) { const e = sunAngles(day + m * 60e3, lat, lon).elev; times.push([Math.abs(e - want), day + m * 60e3]); }
    times.sort((a, b) => a[0] - b[0]); if (times[0][0] < 1) return times[Math.floor(rng() * Math.min(2, times.filter((t) => t[0] < 1).length))][1];
  }
  return Date.UTC(2026, 5, 21, 12);
}
function drawPlace(rng, ne, used) {
  const u = rng(), pick = (arr) => arr[Math.floor(rng() * arr.length)];
  if (u < 0.35) { const pp = (ne.layers.ne_10m_populated_places || []).filter((f) => (f.props.POP_MAX ?? 0) >= 50000); const w = pp.map((f) => 1 / (1 + (used.get(f.props.ADM0NAME) || 0))); let r = rng() * w.reduce((a, b) => a + b, 0); for (let i = 0; i < pp.length; i++) { r -= w[i]; if (r <= 0) return { kind: 'place', name: pp[i].props.NAME, country: pp[i].props.ADM0NAME, lon: pp[i].point[0], lat: pp[i].point[1] }; } }
  if (u < 0.55) { const f = pick(ne.layers.ne_10m_geography_regions_elevation_points || []); if (f) return { kind: 'peak', name: nameOf(f.props), lon: f.point[0], lat: f.point[1] }; }
  if (u < 0.70) { const f = pick(ne.layers.ne_10m_geography_regions_polys || []); if (f) return { kind: 'region', name: nameOf(f.props), lon: f.point[0], lat: f.point[1] }; }
  for (let i = 0; i < 500; i++) { const lat = Math.asin(-0.93 + 1.88 * rng()) / DEG, lon = -180 + 360 * rng(), land = isLand(ne, lon, lat); if (u < 0.9 ? land : !land && [1, -1].some((d) => isLand(ne, lon + d * 0.5, lat))) return { kind: u < 0.9 ? 'land' : 'coast', name: null, lon, lat }; }
  return { kind: 'land', name: null, lon: 7.658, lat: 45.976 };
}
export function planZoom({ seed, nLocations = 200, ne, blocks, ood, profile = 'open' }) {
  const rng = mulberry32(seed), Lmax = profile === 'open' ? 15 : 14, used = new Map(), locations = [], unfilled = {}, edges = [0, ...RANGE_EDGES_KM, 2500];
  for (let id = 0; id < nLocations; id++) {
    const place = drawPlace(rng, ne, used); used.set(place.country, (used.get(place.country) || 0) + 1);
    const split = ood.has(place.lat, place.lon) ? 'ood' : blocks.get(blockKey(place.lat, place.lon)) ?? 'train', views = [];
    for (let v = 0; v < 4; v++) {
      const rMin = rMinKm(place.lat, Lmax), bin = (id * 4 + v) % 5, lo = Math.max(rMin, edges[bin] || rMin), hi = Math.min(2500, edges[bin + 1]); let ok = null;
      for (let t = 0; t < 50 && !ok && lo < hi; t++) {
        const view = { lat: place.lat, lon: place.lon, rangeKm: Math.exp(Math.log(lo) + rng() * (Math.log(hi) - Math.log(lo))), tilt: rng() < 0.4 ? 0 : rng() * 60 * DEG, heading: rng() * 2 * Math.PI };
        if (viewSplit(matrixGrid(poseCamera(view), view, { margin: 1 }), blocks, ood) === split) ok = view;
      }
      if (!ok) { unfilled[`${split}|${bin}`] = (unfilled[`${split}|${bin}`] || 0) + 1; continue; }
      const alwaysDay = rng() < 0.3; views.push({ ...ok, utcMs: alwaysDay ? Date.UTC(2026, 5, 21, 12) : drawUtc(rng, place.lat, place.lon), alwaysDay, range_bin: rangeBin(ok.rangeKm) });
    }
    locations.push({ id, split, place, views });
  }
  return { locations, unfilled };
}
