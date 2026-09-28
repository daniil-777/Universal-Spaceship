// earthtiles.js — pure geo maths for the Earth zoom view (no three.js, Node-testable): Web Mercator tiles, which open
// imagery source serves which level, Terrarium elevation decoding, the level of detail for a viewing distance and the
// rings' tile windows. (Task 2 adds the local frame, the Sun, the camera pose, panning and the clip planes.)
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
// Rings sit on EVEN levels (the inner one at or finer than the ideal level): a change of level then keeps three of the
// four rings. ±0.3 of hysteresis stops the inner ring flickering between two levels.
export function pickInnerLevel(zf, current = null) {
  let L = current ?? 2 * Math.ceil(zf / 2);
  while (zf > L + 0.3 && L < MAX_LEVEL) L += 2;
  while (zf < L - 2.3 && L > MIN_LEVEL) L -= 2;
  return Math.max(MIN_LEVEL, Math.min(MAX_LEVEL, L));
}
export const ringLevels = (L0) => [L0, L0 - 2, L0 - 4, L0 - 6].filter((L) => L >= 2);

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
