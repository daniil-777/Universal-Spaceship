// vlm/gen/geo/naturalearth.js — Natural Earth 10m (public domain) for Z facts: loading with per-feature bboxes,
// antimeridian-safe point-in-polygon, features in the view projected through the captured camera, the nearest named
// place, and the gazetteer's names. Node only (fs).
import fs from 'node:fs';
import path from 'node:path';
import { createLocalFrame, R_KM } from '../../../src/earthtiles.js';
import { project, cameraPosition } from '../labels/camera.js';

export const NE_LAYERS = Object.freeze(['ne_10m_populated_places', 'ne_10m_admin_0_countries', 'ne_10m_admin_1_states_provinces', 'ne_10m_geography_regions_polys',
  'ne_10m_geography_regions_elevation_points', 'ne_10m_land', 'ne_10m_lakes', 'ne_10m_rivers_lake_centerlines', 'ne_10m_glaciated_areas']);
const DEG = Math.PI / 180, wrap = (d) => ((d + 540) % 360) - 180;
// NE v5.1.2 GeoJSON field case (checked on the download): populated places and regions use NAME/FEATURECLA; admin-0 uses
// NAME/CONTINENT with a lower-case featurecla; elevation points, lakes, rivers, glaciated areas and admin-1 use
// name/featurecla (glacier names are mostly null)
export const nameOf = (p) => p.NAME ?? p.name ?? null, claOf = (p) => p.FEATURECLA ?? p.featurecla ?? null;
const rings = (g) => (g.type === 'Polygon' ? [g.coordinates] : g.type === 'MultiPolygon' ? g.coordinates : []);
export const pointsOf = (g) => (g.type === 'Point' ? [g.coordinates] : g.type === 'LineString' ? g.coordinates : g.type === 'MultiLineString' ? g.coordinates.flat() : rings(g).flat(2));
// a feature with an edge across the seam (never in NE, which is cut at ±180) gets the whole longitude range
function bboxOf(g) {
  let a = 90, b = -90, lo = 180, hi = -180, seam = false, prev = null;
  for (const [x, y] of pointsOf(g)) { a = Math.min(a, y); b = Math.max(b, y); lo = Math.min(lo, x); hi = Math.max(hi, x); if (prev !== null && Math.abs(x - prev) > 180) seam = true; prev = x; }
  return seam ? [-180, a, 180, b] : [lo, a, hi, b];
}
// A label point: the vertex average, longitudes unwrapped around the first vertex. A feature with a polygon that spans
// 360° of longitude (a ring round the pole, cut at ±180: Antarctica, the Polar Plateau) has no meaningful average
// longitude that way; it is labelled at the vertex average of its largest polygon's outer ring (closing vertex dropped),
// longitudes unwrapped around that polygon's bbox centre (the identity for NE, which stays within ±180).
const ringArea = (r) => { let a = 0; for (let i = 0, j = r.length - 1; i < r.length; j = i++) a += (r[j][0] + r[i][0]) * (r[j][1] - r[i][1]); return Math.abs(a / 2); };
function centroid(g) {
  const p = pointsOf(g); if (g.type === 'Point') return p[0];
  const polys = rings(g);
  if (polys.some((poly) => { const m = metaOf(poly); return !m.seam && m.hi - m.lo >= 359; })) {
    const big = polys.reduce((a, b) => (ringArea(b[0]) > ringArea(a[0]) ? b : a)), r = big[0].slice(0, -1), m = metaOf(big), c = (m.lo + m.hi) / 2;
    let sx = 0, sy = 0; for (const [x, y] of r) { sx += Math.abs(x - c) <= 180 ? x : c + wrap(x - c); sy += y; }
    return [wrap(sx / r.length), sy / r.length];
  }
  const x0 = p[0][0]; let sx = 0, sy = 0; for (const [x, y] of p) { sx += x0 + wrap(x - x0); sy += y; } return [wrap(sx / p.length), sy / p.length];
}
// a feature without coordinates cannot be placed and is skipped (NE v5.1.2 rivers: the Loire is an empty MultiLineString)
export async function loadNaturalEarth(dir, files = NE_LAYERS) {
  const ne = { layers: {} };
  for (const f of files) for (const ft of JSON.parse(fs.readFileSync(path.join(dir, `${f}.geojson`), 'utf8')).features) {
    if (!ft.geometry || !pointsOf(ft.geometry).length) continue;
    const layer = ft.properties.layer || f; (ne.layers[layer] ||= []).push({ props: ft.properties, geom: ft.geometry, bbox: bboxOf(ft.geometry), point: centroid(ft.geometry) });
  }
  return ne;
}
// NE v5.1.2 is cut at ±180: no edge in any of the nine layers spans more than 180° of longitude. The planar crossing test
// at the query longitude in [-180, 180) is therefore exact for rings of any width (Afro-Eurasia spans 197.5°; Antarctica
// spans 360° and closes along ±180 and the pole), where unwrapping every vertex around the query point is not. -180 is
// also tried as +180, so a point on the seam finds the polygons on either side. An edge spanning more than 180° is read
// as crossing the seam the short way (not NE); such a ring is unwrapped around the query point instead, which is exact
// for rings narrower than 180°. Per-polygon boxes are cached, so a polygon far from the point costs one comparison.
const META = new WeakMap();
function metaOf(poly) {
  let m = META.get(poly);
  if (m) return m;
  let lo = 180, hi = -180, a = 90, b = -90, seam = false;
  for (const ring of poly) for (let i = 0; i < ring.length; i++) {
    const x = ring[i][0], y = ring[i][1]; lo = Math.min(lo, x); hi = Math.max(hi, x); a = Math.min(a, y); b = Math.max(b, y);
    if (i && Math.abs(x - ring[i - 1][0]) > 180) seam = true;
  }
  m = seam ? { lo: -180, hi: 180, a, b, seam } : { lo, hi, a, b, seam };
  META.set(poly, m);
  return m;
}
function crosses(x, lat, ring, unwrap) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const xi = unwrap ? x + wrap(ring[i][0] - x) : ring[i][0], yi = ring[i][1], xj = unwrap ? x + wrap(ring[j][0] - x) : ring[j][0], yj = ring[j][1];
    if ((yi > lat) !== (yj > lat) && x < ((xj - xi) * (lat - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
function inPoly(x, lat, poly, unwrap) {
  if (!crosses(x, lat, poly[0], unwrap)) return false;
  for (let k = 1; k < poly.length; k++) if (crosses(x, lat, poly[k], unwrap)) return false;
  return true;
}
export function pointInGeometry(lon, lat, g) {
  const x = wrap(lon);
  return rings(g).some((poly) => {
    const m = metaOf(poly);
    if (lat < m.a || lat > m.b) return false;
    if (m.seam) return inPoly(x, lat, poly, true);
    return (x >= m.lo && x <= m.hi && inPoly(x, lat, poly, false)) || (x === -180 && m.hi === 180 && inPoly(180, lat, poly, false));
  });
}
const inBox = (b, lon, lat) => lat >= b[1] - 1e-9 && lat <= b[3] + 1e-9 && (b[2] - b[0] >= 359 || Math.abs(wrap(lon - (b[0] + b[2]) / 2)) <= (b[2] - b[0]) / 2 + 1e-9);
export const pointInFeature = (f, lon, lat) => inBox(f.bbox, lon, lat) && pointInGeometry(lon, lat, f.geom);
export const featuresAt = (ne, layer, lon, lat) => (ne.layers[layer] || []).filter((f) => pointInFeature(f, lon, lat));
export const isLand = (ne, lon, lat) => featuresAt(ne, 'ne_10m_land', lon, lat).length > 0;
export const regionOf = (px, py, W, H) => `${['upper', 'middle', 'lower'][Math.min(2, Math.floor(3 * py / H))]} ${['left', 'centre', 'right'][Math.min(2, Math.floor(3 * px / W))]}`.replace('middle centre', 'centre');
export const compassOf = (deg) => ['north', 'north-east', 'east', 'south-east', 'south', 'south-west', 'west', 'north-west'][Math.round((((deg % 360) + 360) % 360) / 45) % 8];
const gc = (la1, lo1, la2, lo2) => { const a = Math.sin((la2 - la1) * DEG / 2) ** 2 + Math.cos(la1 * DEG) * Math.cos(la2 * DEG) * Math.sin((lo2 - lo1) * DEG / 2) ** 2; return 2 * R_KM * Math.asin(Math.sqrt(a)); };
const bearing = (la1, lo1, la2, lo2) => (Math.atan2(Math.sin((lo2 - lo1) * DEG) * Math.cos(la2 * DEG), Math.cos(la1 * DEG) * Math.sin(la2 * DEG) - Math.sin(la1 * DEG) * Math.cos(la2 * DEG) * Math.cos((lo2 - lo1) * DEG)) / DEG + 360) % 360;
// exclude holds features (e.g. the in-view places' .feature), not names: two places can share a name
export function nearestPlace(ne, lat, lon, { exclude = new Set(), minPop = 50000 } = {}) {
  let best = null;
  for (const f of ne.layers.ne_10m_populated_places || []) { if ((f.props.POP_MAX ?? 0) < minPop || exclude.has(f)) continue; const km = gc(lat, lon, f.point[1], f.point[0]); if (!best || km < best.km) best = { name: f.props.NAME, km, lat: f.point[1], lon: f.point[0] }; }
  if (best) { best.bearing = bearing(lat, lon, best.lat, best.lon); best.compass = compassOf(best.bearing); }
  return best;
}
const VIEW_LAYERS = { ne_10m_populated_places: 'place', ne_10m_geography_regions_elevation_points: 'peak', ne_10m_lakes: 'lake', ne_10m_rivers_lake_centerlines: 'river', ne_10m_geography_regions_polys: 'region', ne_10m_glaciated_areas: 'glacier' };
// cam.eye is set by poseCamera(); a captured camera without it uses the eye from its matrixWorldInverse. Each entry
// carries its NE feature as a non-enumerable .feature (for nearestPlace's exclude; never serialised into a record).
export function featuresInView(ne, cam, origin, { W = 896, H = 504 } = {}) {
  const frame = createLocalFrame(origin.lat, origin.lon), eye = cam.eye || cameraPosition(cam), out = [];
  for (const [layer, kind] of Object.entries(VIEW_LAYERS)) for (const f of ne.layers[layer] || []) {
    const [lon, lat] = f.point, p = frame.toLocal(lat, lon, 0), up = [p[0], p[1] + R_KM, p[2]];
    if ((eye[0] - p[0]) * up[0] + (eye[1] - p[1]) * up[1] + (eye[2] - p[2]) * up[2] <= 0) continue;
    const q = project(cam, p, W, H);
    if (q.front && q.x >= 0 && q.x < W && q.y >= 0 && q.y < H && (nameOf(f.props) || kind === 'glacier')) {
      const o = { layer, kind, name: nameOf(f.props), pop: f.props.POP_MAX ?? null, featurecla: claOf(f.props), px: q.x, py: q.y, region: regionOf(q.x, q.y, W, H) };
      out.push(Object.defineProperty(o, 'feature', { value: f, enumerable: false }));
    }
  }
  return out;
}
export function gazetteerNames(ne) { const s = new Set(); for (const fs of Object.values(ne.layers)) for (const f of fs) for (const k of ['NAME', 'name', 'NAME_EN', 'name_en', 'NAMEASCII', 'NAME_ALT', 'name_alt', 'ADMIN']) if (f.props[k]) s.add(String(f.props[k])); return [...s]; }
