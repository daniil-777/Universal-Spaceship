import { test } from 'node:test';
import assert from 'node:assert/strict';
import { lonLatToTile, tileToLonLat, tileSizeKm, SOURCES, EOX_YEAR, sourceForLevel, tileUrl, HEIGHT_SOURCE, heightLevel, decodeTerrarium, levelFloat, pickInnerLevel, ringLevels, ringWindow, heightWindow, windowTiles, RING_TILES, RING_COUNT, HEIGHT_TILES, MAX_LAT, R_KM, createLocalFrame, globeAxes, sunLocal, cameraPose, panTarget, clipPlanes } from '../src/earthtiles.js';

const close = (a, b, eps, msg = '') => assert.ok(Math.abs(a - b) <= eps, `${msg} ${a} vs ${b}`);

test('Web Mercator tiles: the probed Alps tile, round trips, the latitude limit, tile sizes', () => {
  const t = lonLatToTile(7.658, 45.976, 12); assert.deepEqual([Math.floor(t.x), Math.floor(t.y)], [2135, 1457]);
  for (const [lon, lat, z] of [[0, 0, 3], [-74.0, 40.7, 15], [179.99, -60, 10], [-179.99, 84, 18]]) {
    const p = lonLatToTile(lon, lat, z), q = tileToLonLat(p.x, p.y, z); close(q.lon, lon, 1e-9, 'lon'); close(q.lat, lat, 1e-9, 'lat');
  }
  close(tileToLonLat(0, 0, 0).lat, MAX_LAT, 1e-9, 'top edge'); close(lonLatToTile(0, 89.9, 4).y, 0, 1e-9, 'beyond 85.05° clamps to the top edge');
  close(tileSizeKm(0, 0), 40075.016686, 1e-6); close(tileSizeKm(12, 60), 40075.016686 / 2 / 4096, 1e-9);
});

test('sources by level; the URLs are the ones probed on 2026-09-28', () => {
  const eoxMax = SOURCES[1].maxLevel; assert.ok(eoxMax >= 13, 'EOX serves at least to level 13');
  assert.equal(sourceForLevel(8).name, 'NASA Blue Marble'); assert.equal(sourceForLevel(9).name, 'Sentinel-2 cloudless'); assert.equal(sourceForLevel(eoxMax).name, 'Sentinel-2 cloudless');
  assert.equal(sourceForLevel(eoxMax + 1).name, 'Esri World Imagery'); assert.equal(sourceForLevel(18).name, 'Esri World Imagery');
  assert.equal(tileUrl(sourceForLevel(8), 8, 133, 91), 'https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/BlueMarble_NextGeneration/default/GoogleMapsCompatible_Level8/8/91/133.jpeg');
  assert.equal(tileUrl(sourceForLevel(12), 12, 2135, 1457), `https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-${EOX_YEAR}_3857/default/g/12/1457/2135.jpg`);
  assert.equal(tileUrl(sourceForLevel(17), 17, 68336, 46636), 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/17/46636/68336');
  assert.equal(tileUrl(HEIGHT_SOURCE, 12, 2135, 1457), 'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/12/2135/1457.png');
  assert.equal(tileUrl(sourceForLevel(2), 2, -1, 1), tileUrl(sourceForLevel(2), 2, 3, 1), 'x wraps around the antimeridian');
});

test('Terrarium heights: metres from RGB; the sea floor is sea level; heights one level up, capped at 13', () => {
  assert.equal(decodeTerrarium(128, 0, 0), 0); assert.equal(decodeTerrarium(129, 0, 0), 256); assert.equal(decodeTerrarium(130, 44, 128), 556.5);
  assert.equal(decodeTerrarium(120, 0, 0), 0);
  assert.equal(heightLevel(18), 13); assert.equal(heightLevel(10), 9); assert.equal(heightLevel(0), 0);
});

test('level of detail: the screen pixel footprint picks the level; the inner ring is at or finer than it, with hysteresis; rings step one level', () => {
  close(levelFloat(420, 0, 45, 900), 8.661, 0.005, 'ISS altitude at the equator'); close(levelFloat(420, 60, 45, 900), 7.661, 0.005, 'a level coarser at 60°');
  assert.ok(levelFloat(1, 45, 45, 900) > levelFloat(10, 45, 45, 900));
  assert.equal(pickInnerLevel(15.4), 16); assert.equal(pickInnerLevel(16.2, 16), 16); assert.equal(pickInnerLevel(16.4, 16), 17);
  assert.equal(pickInnerLevel(14.8, 16), 16); assert.equal(pickInnerLevel(14.6, 16), 15);
  assert.equal(pickInnerLevel(30), 18); assert.equal(pickInnerLevel(-5), 4); assert.equal(pickInnerLevel(30, 16), 18);
  assert.equal(RING_COUNT, 5);
  assert.deepEqual(ringLevels(16), [16, 15, 14, 13, 11]); assert.deepEqual(ringLevels(5), [5, 4, 3, 2]); assert.deepEqual(ringLevels(4), [4, 3, 2]);
});

test('ring windows: 8 × 8 tiles on even indices around the point, 64 distinct toroidal slots, inner tiles first, poles and the antimeridian', () => {
  const w = ringWindow(7.658, 45.976, 12), t = lonLatToTile(7.658, 45.976, 12);
  assert.equal(w.x0 % 2, 0); assert.equal(w.y0 % 2, 0); assert.ok(t.x >= w.x0 + 4 && t.x < w.x0 + 6 && t.y >= w.y0 + 4 && t.y < w.y0 + 6);
  const tiles = windowTiles(w, RING_TILES); assert.equal(tiles.length, 64); assert.equal(new Set(tiles.map((q) => q.slot)).size, 64);
  assert.ok(tiles[0].d <= tiles[63].d); assert.equal(tiles[0].key.split('/')[0], '12');
  const polar = windowTiles(ringWindow(0, 85, 3), RING_TILES); assert.ok(polar.length < 64 && polar.every((q) => q.y >= 0 && q.y < 8), 'no rows beyond the poles');
  const wrap = windowTiles(ringWindow(179.9, 0, 3), RING_TILES); assert.ok(wrap.every((q) => +q.key.split('/')[1] < 8), 'x wraps in the key');
});

test('height windows: 4 × 4 Terrarium tiles one level up cover the ring (more than cover it where heights stop at 13)', () => {
  for (const [lon, lat] of [[7.658, 45.976], [-74, 40.7], [139.7, 35.7], [-150, -60]]) for (let L = 2; L <= 18; L += 1) {
    const w = ringWindow(lon, lat, L), h = heightWindow(w), s = 2 ** (h.level - L);
    close(h.x0 + h.off[0] * HEIGHT_TILES, w.x0 * s, 1e-9, 'ring start x'); close(h.y0 + h.off[1] * HEIGHT_TILES, w.y0 * s, 1e-9, 'ring start y');
    close(h.scale * HEIGHT_TILES, RING_TILES * s, 1e-12, 'ring width');
    assert.ok(w.x0 * s >= h.x0 && (w.x0 + RING_TILES) * s <= h.x0 + HEIGHT_TILES + 1e-9 && w.y0 * s >= h.y0 && (w.y0 + RING_TILES) * s <= h.y0 + HEIGHT_TILES + 1e-9, `covered at L${L}`);
  }
});

test('local frame: the target is the origin, 1 km east drops 0.0785 m, north is −z, up is +y, height goes straight up', () => {
  const f = createLocalFrame(45.976, 7.658), o = f.toLocal(45.976, 7.658, 0); o.forEach((c) => close(c, 0, 1e-9));
  const kmDeg = 1 / (R_KM * Math.PI / 180);
  const e = f.toLocal(45.976, 7.658 + kmDeg / Math.cos(45.976 * Math.PI / 180), 0); close(e[0], 1, 1e-4, 'east'); close(e[1], -1 / (2 * R_KM), 1e-7, 'drop'); close(e[2], 0, 2e-4);
  const n = f.toLocal(45.976 + kmDeg, 7.658, 0); close(n[2], -1, 1e-4, 'north = −z'); close(n[0], 0, 1e-6);
  const u = f.upAt(45.976, 7.658); close(u[0], 0, 1e-12); close(u[1], 1, 1e-12); close(u[2], 0, 1e-12);
  close(f.toLocal(45.976, 7.658, 4.478)[1], 4.478, 1e-9, 'the Matterhorn’s height');
});

test('the textured globe turned into the local frame: a rotation that puts the target straight up', () => {
  const f = createLocalFrame(-33.9, 151.2), [ax, ay, az] = globeAxes(f), lat = -33.9 * Math.PI / 180, lon = 151.2 * Math.PI / 180;
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  for (const a of [ax, ay, az]) close(Math.hypot(...a), 1, 1e-12);
  close(dot(ax, ay), 0, 1e-12); close(dot(ax, az), 0, 1e-12); close(dot(ay, az), 0, 1e-12);
  const p = [Math.cos(lat) * Math.cos(lon), Math.sin(lat), -Math.cos(lat) * Math.sin(lon)];
  const w = [0, 1, 2].map((k) => ax[k] * p[0] + ay[k] * p[1] + az[k] * p[2]); close(w[0], 0, 1e-12); close(w[1], 1, 1e-12); close(w[2], 0, 1e-12);
  const det = ax[0] * (ay[1] * az[2] - ay[2] * az[1]) - ax[1] * (ay[0] * az[2] - ay[2] * az[0]) + ax[2] * (ay[0] * az[1] - ay[1] * az[0]);
  close(det, 1, 1e-12, 'a rotation, not a mirror');
});

test('the Sun: overhead near the subsolar point at the June solstice noon UTC, straight below on the other side', () => {
  const t = Date.UTC(2026, 5, 21, 12), a = sunLocal(t, createLocalFrame(23.44, 0)), b = sunLocal(t, createLocalFrame(-23.44, 180));
  assert.ok(a[1] > 0.995, 'up ' + a[1]); assert.ok(b[1] < -0.995, 'down ' + b[1]); close(Math.hypot(...a), 1, 1e-12);
});

test('camera pose: straight down with north up; heading 90° puts east up; tilted it backs away south and stays level-headed', () => {
  const f = createLocalFrame(45.976, 7.658), v = { lat: 45.976, lon: 7.658, rangeKm: 50, tilt: 0, heading: 0, groundKm: 1 };
  let p = cameraPose(f, v); close(p.pos[0], 0, 1e-9); close(p.pos[1], 51, 1e-9); close(p.pos[2], 0, 1e-9); close(p.up[0], 0, 1e-9); close(p.up[2], -1, 1e-9); close(p.look[1], 1, 1e-9);
  p = cameraPose(f, { ...v, heading: Math.PI / 2 }); close(p.up[0], 1, 1e-9); close(p.up[2], 0, 1e-9);
  p = cameraPose(f, { ...v, tilt: Math.PI / 3 }); close(p.pos[1], 26, 1e-9); close(p.pos[2], 50 * Math.sin(Math.PI / 3), 1e-9, 'south of the target');
  assert.ok(p.camLat < v.lat); close(p.camLon, v.lon, 1e-9); close(p.camAltKm, 26, 1e-9); close(Math.hypot(...p.up), 1, 1e-12);
});

test('panning moves the target along the ground in the view’s axes; latitude clamps at the Mercator limit, longitude wraps', () => {
  const kmPerDeg = R_KM * Math.PI / 180;
  let v = panTarget({ lat: 0, lon: 0, heading: 0 }, 0, kmPerDeg); close(v.lat, 1, 1e-12); close(v.lon, 0, 1e-12);
  v = panTarget({ lat: 0, lon: 0, heading: Math.PI / 2 }, 0, kmPerDeg); close(v.lat, 0, 1e-12); close(v.lon, 1, 1e-12);
  v = panTarget({ lat: 0, lon: 0, heading: 0 }, kmPerDeg, 0); close(v.lon, 1, 1e-12);
  v = panTarget({ lat: 85, lon: 0, heading: 0 }, 0, 500); close(v.lat, MAX_LAT, 1e-12);
  v = panTarget({ lat: 0, lon: 179.5, heading: 0 }, kmPerDeg, 0); close(v.lon, -179.5, 1e-9);
});

test('clip planes: near follows the clearance, far reaches past the horizon, the depth range stays usable', () => {
  for (const alt of [0.3, 8, 420, 20000]) {
    const c = clipPlanes(alt, alt), horizon = Math.sqrt(alt * (2 * R_KM + alt));
    assert.ok(c.far > horizon, `far at ${alt} km`); assert.ok(c.near >= 0.02 && c.near <= 50); assert.ok(c.far / c.near < 1e5, `depth range at ${alt} km`);
  }
});

test('the rings fit a 140 MB GPU budget', () => {
  const colour = (RING_TILES * 256) ** 2 * 4 * 4 / 3, height = (HEIGHT_TILES * 256) ** 2 * 4, geometry = (129 * 129 * 8 + 128 * 128 * 6) * 4;
  assert.ok(RING_COUNT * (colour + height + geometry) <= 140 * 2 ** 20, `${(RING_COUNT * (colour + height + geometry) / 2 ** 20).toFixed(1)} MB`);
});
