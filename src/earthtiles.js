// earthtiles.js — pure geo maths for the Earth zoom view (no three.js, Node-testable): Web Mercator tiles, which open
// imagery source serves which level, Terrarium elevation decoding, the level of detail for a viewing distance and the
// rings' tile windows; then the local frame, the Sun in it, the telescope camera's pose, panning and the clip planes.
import { julianDay, gmst, sunEci } from './ephem.js';
export const R_KM = 6371, MAX_LAT = 85.0511287798066, MIN_LEVEL = 4, MAX_LEVEL = 18, RING_TILES = 8, HEIGHT_TILES = 4, MAX_HEIGHT_LEVEL = 13;
// From test/probe_earthtiles.mjs (2026-09-28): the newest Sentinel-2 cloudless mosaic, and Esri's placeholder for
// missing deep imagery (the same 2521-byte JPEG over every open-ocean tile at levels 17-18).
export const EOX_YEAR = 2025;
export const ESRI_BLANK = { bytes: 2521, sha1: '1660d86a87f57ef0ff580822e0e62f0feb48deee' };
const DEG = Math.PI / 180, KM_PER_PX0 = 156.54303392804097, EQUATOR_KM = 40075.016686;
export const mod = (a, n) => ((a % n) + n) % n;
const clampLat = (lat) => Math.max(-MAX_LAT, Math.min(MAX_LAT, lat));

export function lonLatToTile(lonDeg, latDeg, z) {
  const n = 2 ** z, r = clampLat(latDeg) * DEG;
  return { x: (lonDeg + 180) / 360 * n, y: (1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2 * n };
}
export function tileToLonLat(x, y, z) {
  const n = 2 ** z;
  return { lon: x / n * 360 - 180, lat: Math.atan(Math.sinh(Math.PI * (1 - 2 * y / n))) / DEG };
}
export const tileSizeKm = (z, latDeg) => EQUATOR_KM * Math.cos(clampLat(latDeg) * DEG) / 2 ** z;

export const SOURCES = [
  { maxLevel: 8, name: 'NASA Blue Marble', credit: 'Blue Marble: NASA Earth Observatory (GIBS)',
    url: (z, x, y) => `https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/BlueMarble_NextGeneration/default/GoogleMapsCompatible_Level8/${z}/${y}/${x}.jpeg` },
  { maxLevel: 14, name: 'Sentinel-2 cloudless', credit: `Sentinel-2 cloudless (s2maps.eu) by EOX IT Services GmbH, contains modified Copernicus Sentinel data ${EOX_YEAR}`,
    url: (z, x, y) => `https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-${EOX_YEAR}_3857/default/g/${z}/${y}/${x}.jpg` },
  { maxLevel: MAX_LEVEL, name: 'Esri World Imagery', credit: 'imagery © Esri, Maxar, Earthstar Geographics and the GIS user community',
    url: (z, x, y) => `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/${z}/${y}/${x}` },
];
export const HEIGHT_SOURCE = { name: 'AWS Terrain Tiles', credit: 'elevation: AWS Terrain Tiles (SRTM, GMTED2010, ETOPO1 and others)',
  url: (z, x, y) => `https://s3.amazonaws.com/elevation-tiles-prod/terrarium/${z}/${x}/${y}.png` };
export const ZOOM_CREDIT = [...SOURCES, HEIGHT_SOURCE].map((s) => s.credit).join(' · ');
export const sourceForLevel = (z) => SOURCES.find((s) => z <= s.maxLevel) || SOURCES[SOURCES.length - 1];
export const tileUrl = (src, z, x, y) => src.url(z, mod(x, 2 ** z), y);
export const heightLevel = (z) => Math.max(0, Math.min(z - 1, MAX_HEIGHT_LEVEL));
export const decodeTerrarium = (r, g, b) => Math.max(0, r * 256 + g + b / 256 - 32768);

// The level whose tile pixel matches the ground footprint of one screen pixel at this viewing distance.
export function levelFloat(rangeKm, latDeg, fovYDeg, viewportPx) {
  const kmPerScreenPx = rangeKm * 2 * Math.tan(fovYDeg * DEG / 2) / viewportPx;
  return Math.log2(KM_PER_PX0 * Math.cos(clampLat(latDeg) * DEG) / kmPerScreenPx);
}
// The inner ring sits at or finer than the ideal level (±0.3 of hysteresis stops it flickering between two levels). The
// rings step ONE level (each only 2× coarser than the next, so the screen's edges stay nearly as sharp as its centre and
// keep the same imagery source), and a fifth ring two levels further out reaches the horizon of a tilted view. A change
// of level keeps three of the five rings.
export const RING_COUNT = 5;
export function pickInnerLevel(zf, current = null) {
  let L = current ?? Math.ceil(zf);
  while (zf > L + 0.3 && L < MAX_LEVEL) L += 1;
  while (zf < L - 1.3 && L > MIN_LEVEL) L -= 1;
  return Math.max(MIN_LEVEL, Math.min(MAX_LEVEL, L));
}
export const ringLevels = (L0) => [L0, L0 - 1, L0 - 2, L0 - 3, L0 - 5].filter((L) => L >= 2);

// 8 × 8 tiles around the point; the origin sits on even tile indices so the 4 × 4 height tiles one level up line up.
export function ringWindow(lonDeg, latDeg, level) {
  const t = lonLatToTile(lonDeg, latDeg, level);
  return { level, x0: 2 * Math.floor(t.x / 2) - RING_TILES / 2, y0: 2 * Math.floor(t.y / 2) - RING_TILES / 2 };
}
// The height tiles under a ring and where the ring's uv lands in their (toroidal) atlas: atlasUv = off + uv * scale.
export function heightWindow(win) {
  const level = heightLevel(win.level), s = 2 ** (level - win.level), cx = (win.x0 + RING_TILES / 2) * s, cy = (win.y0 + RING_TILES / 2) * s;
  const x0 = Math.floor(cx) - HEIGHT_TILES / 2, y0 = Math.floor(cy) - HEIGHT_TILES / 2;
  return { level, x0, y0, off: [(win.x0 * s - x0) / HEIGHT_TILES, (win.y0 * s - y0) / HEIGHT_TILES], scale: RING_TILES * s / HEIGHT_TILES };
}
// The window's tiles, nearest the centre first: slot = the toroidal atlas cell, key = "z/x/y" with x wrapped.
export function windowTiles(win, size) {
  const n = 2 ** win.level, c = (size - 1) / 2, out = [];
  for (let j = 0; j < size; j++) for (let i = 0; i < size; i++) {
    const x = win.x0 + i, y = win.y0 + j;
    if (y < 0 || y >= n) continue;
    out.push({ x, y, slot: mod(y, size) * size + mod(x, size), key: `${win.level}/${mod(x, n)}/${y}`, d: (i - c) ** 2 + (j - c) ** 2 });
  }
  return out.sort((a, b) => a.d - b.d);
}

const wrap180 = (lon) => mod(lon + 180, 360) - 180;

// A frame on the sphere (R_KM) at the view's target, in kilometres: x East, y Up, z South (north is −z, as in
// src/terrain.js). Double precision: the ring vertices are placed relative to it, so no large coordinates reach the GPU.
export function createLocalFrame(lat0Deg, lon0Deg) {
  const la = lat0Deg * DEG, lo = lon0Deg * DEG, sla = Math.sin(la), cla = Math.cos(la), slo = Math.sin(lo), clo = Math.cos(lo);
  const e = [-slo, clo, 0], n = [-sla * clo, -sla * slo, cla], u = [cla * clo, cla * slo, sla], p0 = [R_KM * u[0], R_KM * u[1], R_KM * u[2]], t = [0, 0, 0];
  const ecef = (latDeg, lonDeg, r, out) => { const a = latDeg * DEG, b = lonDeg * DEG, c = Math.cos(a); out[0] = r * c * Math.cos(b); out[1] = r * c * Math.sin(b); out[2] = r * Math.sin(a); return out; };
  const local = (v, out) => {
    const x = v[0] * e[0] + v[1] * e[1] + v[2] * e[2], y = v[0] * u[0] + v[1] * u[1] + v[2] * u[2], z = -(v[0] * n[0] + v[1] * n[1] + v[2] * n[2]);
    out[0] = x; out[1] = y; out[2] = z; return out;
  };
  return {
    lat: lat0Deg, lon: lon0Deg,
    toLocal(latDeg, lonDeg, hKm, out = [0, 0, 0]) { ecef(latDeg, lonDeg, R_KM + hKm, t); t[0] -= p0[0]; t[1] -= p0[1]; t[2] -= p0[2]; return local(t, out); },
    upAt(latDeg, lonDeg, out = [0, 0, 0]) { return local(ecef(latDeg, lonDeg, 1, t), out); },
    northAt(latDeg, lonDeg, out = [0, 0, 0]) { const a = latDeg * DEG, b = lonDeg * DEG; t[0] = -Math.sin(a) * Math.cos(b); t[1] = -Math.sin(a) * Math.sin(b); t[2] = Math.cos(a); return local(t, out); },
    dirToLocal(v, out = [0, 0, 0]) { return local(v, out); },
  };
}
// The textured globe (src/earth.js; Greenwich on the sphere's +x, the north pole on +y, 90° E on −z): the local images
// of its three axes — the columns of the rotation that turns it into the frame.
export const globeAxes = (frame) => [frame.dirToLocal([1, 0, 0]), frame.dirToLocal([0, 0, 1]), frame.dirToLocal([0, -1, 0])];

export function sunLocal(utcMs, frame, out = [0, 0, 0]) {
  const jd = julianDay(utcMs), g = gmst(jd), s = sunEci(jd), c = Math.cos(g), si = Math.sin(g);
  return frame.dirToLocal([c * s[0] + si * s[1], -si * s[0] + c * s[1], s[2]], out);
}

// The telescope: the camera `rangeKm` from the target (on the ground at groundKm), tilted from straight down toward the
// horizon (tilt, rad) and facing `heading` (rad, 0 = north). The screen's up is the heading when looking straight down.
const _u = [0, 0, 0], _n = [0, 0, 0];
export function cameraPose(frame, v, out = { pos: [0, 0, 0], up: [0, 0, 0], look: [0, 0, 0], camLat: 0, camLon: 0, camAltKm: 0 }) {
  const g = v.groundKm || 0, T = frame.toLocal(v.lat, v.lon, g, out.look), U = frame.upAt(v.lat, v.lon, _u), N = frame.northAt(v.lat, v.lon, _n);
  const E = [N[1] * U[2] - N[2] * U[1], N[2] * U[0] - N[0] * U[2], N[0] * U[1] - N[1] * U[0]];
  const ch = Math.cos(v.heading), sh = Math.sin(v.heading), ct = Math.cos(v.tilt), st = Math.sin(v.tilt);
  for (let i = 0; i < 3; i++) {
    const f = N[i] * ch + E[i] * sh;
    out.pos[i] = T[i] + v.rangeKm * (U[i] * ct - f * st); out.up[i] = U[i] * st + f * ct;
  }
  const back = v.rangeKm * st;
  out.camLat = Math.max(-MAX_LAT, Math.min(MAX_LAT, v.lat - back * ch / R_KM / DEG));
  out.camLon = wrap180(v.lon - back * sh / (R_KM * Math.cos(v.lat * DEG)) / DEG);
  out.camAltKm = g + v.rangeKm * ct;
  return out;
}
export function panTarget(v, dRightKm, dFwdKm) {
  const ch = Math.cos(v.heading), sh = Math.sin(v.heading), dN = dFwdKm * ch - dRightKm * sh, dE = dFwdKm * sh + dRightKm * ch;
  v.lat = Math.max(-MAX_LAT, Math.min(MAX_LAT, v.lat + dN / R_KM / DEG));
  v.lon = wrap180(v.lon + dE / (R_KM * Math.cos(v.lat * DEG)) / DEG);
  return v;
}
// Near: a fraction of the camera's clearance above the ground (≥ 20 m, ≤ 50 km). Far: past the horizon, plus the
// atmosphere's limb from orbit or distant ridges from low down.
export function clipPlanes(clearKm, camAltKm) {
  const a = Math.max(0, camAltKm), horizon = Math.sqrt(a * (2 * R_KM + a));
  return { near: Math.min(50, Math.max(0.02, 0.3 * clearKm)), far: 1.2 * (horizon + (a < 300 ? 350 : 2000)) };
}
