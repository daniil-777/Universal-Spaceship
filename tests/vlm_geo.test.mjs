import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadNaturalEarth, featuresAt, pointInGeometry, regionOf, compassOf, nearestPlace, isLand, featuresInView } from '../vlm/gen/geo/naturalearth.js';
import { decodeTerrariumRaw, decodeTilePng, elevStats, createElevationReader, terrariumUrl } from '../vlm/gen/geo/terrarium.js';
import { poseCamera, matrixGrid, lightingOf, sunAngles, rMinKm, zoomTags, zoomFacts } from '../vlm/gen/labels/zoom.js';
import { pixelRay, project } from '../vlm/gen/labels/camera.js';
import { createLocalFrame, R_KM } from '../src/earthtiles.js';
import { blockKey, assignBlocks, viewSplit, buildOodMask } from '../vlm/gen/sampler_z.js';
import { validateRecord } from '../vlm/gen/schema.js';

const FIX = fileURLToPath(new URL('./vlm_fixtures/', import.meta.url)), ne = await loadNaturalEarth(FIX, ['ne_mini']);
test('point-in-polygon with a hole', () => {
  assert.deepEqual(featuresAt(ne, 'ne_10m_admin_0_countries', 2, 2).map((f) => f.props.NAME), ['Squareland']);
  assert.deepEqual(featuresAt(ne, 'ne_10m_admin_0_countries', 5, 5), []);
  assert.deepEqual(featuresAt(ne, 'ne_10m_admin_0_countries', 11, 5), []);
});
test('REVIEW FOCUS 2: antimeridian view: grid, point-in-polygon and split block wrap', () => {
  for (const lon of [179.5, -179.5, 180, -180]) assert.deepEqual(featuresAt(ne, 'ne_10m_admin_0_countries', lon, 0).map((f) => f.props.NAME), ['Fijiish'], `lon ${lon}`);
  assert.equal(blockKey(0, 180), blockKey(0, -180)); assert.equal(blockKey(84, 10), blockKey(85.05, 10));
  const view = { lat: 0, lon: 179.95, rangeKm: 300, tilt: 0, heading: 0 }, grid = matrixGrid(poseCamera(view), view, { nx: 16, ny: 9 });
  const lons = grid.filter(Boolean).map((g) => g.lon);
  assert.ok(lons.some((l) => l > 179) && lons.some((l) => l < -179) && lons.every((l) => l >= -180 && l <= 180));
  const mk = (east, west) => new Map([1, -1].flatMap((la) => [[blockKey(la, 179.9), east], [blockKey(la, -179.9), west]])), noOod = { has: () => false };
  assert.equal(viewSplit(grid, mk('val', 'val'), noOod), 'val');
  assert.equal(viewSplit(grid, mk('val', 'train'), noOod), null);
  assert.equal(viewSplit(grid, mk('val', 'val'), { has: (lat, lon) => lon < 0 }), null, 'a footprint partly in the OOD region is redrawn');
  assert.equal(viewSplit(grid, mk('val', 'val'), { has: () => true }), 'ood', 'a footprint entirely in the OOD region is OOD');
  assert.equal(nearestPlace(ne, 0, 179.9, { exclude: new Set() }).name, 'Beta');
});
test('Terrarium raw decode (sea kept below 0) on the fixture', async () => {
  assert.equal(decodeTerrariumRaw(127, 156, 0), -100); assert.equal(decodeTerrariumRaw(131, 232, 128), 1000.5);
  const t = await decodeTilePng(fs.readFileSync(FIX + 'terrarium_4x4.png'));
  assert.deepEqual([t.w, t.h], [4, 4]); assert.deepEqual([...t.elev.slice(0, 4)], [-100, 0, 1000.5, 4807]);
  const s = elevStats([{ elev: -100, sea: true, px: 0, py: 0 }, { elev: 4807, sea: false, px: 1, py: 0 }, { elev: 1000.5, sea: false, px: 1, py: 1 }, { elev: null, sea: false, px: 0, py: 1 }], { W: 2, H: 2 });
  assert.deepEqual([s.min, s.max, s.relief], [-100, 4807, 4907]); assert.equal(s.n, 3);
});
test('the 16x9 grid from camera matrices round-trips within 1e-6 rad; margin widens the grid', () => {
  for (const v of [{ lat: 45.976, lon: 7.658, rangeKm: 50, tilt: 0.6, heading: 1.1 }, { lat: -33.9, lon: 151.2, rangeKm: 2400, tilt: 0, heading: 0 }]) {
    const cam = poseCamera(v), frame = createLocalFrame(v.lat, v.lon), g = matrixGrid(cam, v, { nx: 16, ny: 9 });
    for (const p of g.filter(Boolean)) {
      const a = pixelRay(cam, p.px, p.py, 896, 504), q = frame.toLocal(p.lat, p.lon, 0), d = q.map((x, i) => x - a.origin[i]), l = Math.hypot(...d);
      assert.ok(Math.acos(Math.min(1, (d[0] * a.dir[0] + d[1] * a.dir[1] + d[2] * a.dir[2]) / l)) < 1e-6);
    }
    assert.equal(matrixGrid(cam, v, { nx: 16, ny: 9, margin: 1 }).length, 18 * 11);
  }
});
test('lighting, r_min and tags', () => {
  assert.deepEqual(lightingOf({ mode: 'always_day' }), { mode: 'always_day', sun_elev_deg: 66.7, sun_az_deg: 54.5, class: 'day' });
  const s = sunAngles(Date.parse('2026-06-21T12:00:00Z'), 0, 0); assert.ok(s.elev > 60);
  assert.equal(lightingOf({ mode: 'utc', utcMs: Date.parse('2026-06-21T00:00:00Z'), lat: 0, lon: 0 }).class, 'night');
  assert.ok(Math.abs(rMinKm(0, 15) - 0.011533 * 504 * 0.5 * 1.15) < 1e-9);
  assert.deepEqual(zoomTags({ geo: { sea_frac: 0.7, relief: 100, mean: 50, cellReliefFrac: {}, urban: false, desert: false, ice: false }, view: { rangeKm: 50, lat: 10 }, lighting: { class: 'night' } }).sort(), ['COASTLINE', 'FLAT', 'NIGHT', 'WATER_DOMINANT']);
  assert.deepEqual(zoomTags({ geo: { sea_frac: 0, relief: 1800, mean: 2500, cellReliefFrac: {}, urban: false, desert: false, ice: false }, view: { rangeKm: 50, lat: 46 }, lighting: { class: 'day' } }).sort(), ['HIGH_TERRAIN', 'MOUNTAINS']);
});
test('block assignment comes out near 80/10/10 by weight and is seeded', () => {
  const w = new Map(Array.from({ length: 64 }, (_, i) => [`3/${i % 8}/${i >> 3}`, 1])), a = assignBlocks(7, w), b = assignBlocks(7, w);
  assert.deepEqual([...a], [...b]); const n = (s) => [...a.values()].filter((x) => x === s).length;
  assert.ok(n('val') >= 6 && n('val') <= 8 && n('test') >= 6 && n('test') <= 8, `${n('train')}/${n('val')}/${n('test')}`);
});

// Beyond the brief's fixture: NE v5.1.2 is cut at ±180 and dense (no edge spans more than 180° of longitude), but its land
// layer holds a 197.5°-wide Afro-Eurasia ring and a 360° Antarctica ring that closes along ±180 and the pole.
test('REVIEW FOCUS 2: rings wider than 180° and the polar cap stay exact on both sides of the seam', () => {
  const wide = { type: 'Polygon', coordinates: [[[-18, 0], [81, 0], [180, 0], [180, 60], [81, 60], [-18, 60], [-18, 0]]] };
  const cap = { type: 'Polygon', coordinates: [[[-180, -60], [-90, -61], [0, -62], [90, -61], [180, -60], [180, -90], [90, -90], [0, -90], [-90, -90], [-180, -90], [-180, -60]]] };
  const land = { layers: { ne_10m_land: [{ props: {}, geom: wide, bbox: [-18, 0, 180, 60] }, { props: {}, geom: cap, bbox: [-180, -90, 180, -60] }] } };
  const cases = [[170, 30, true], [179.9, 30, true], [-10, 30, true], [100, 30, true], [-20, 30, false], [-170, 30, false], [0, -80, true], [-179, -89, true], [179.9, -70, true], [-90, -60.5, false], [0, -61, false]];
  for (const [lon, lat, want] of cases) assert.equal(isLand(land, lon, lat), want, `${lon},${lat}`);
  assert.equal(pointInGeometry(170 - 360, 30, wide), true, 'the query longitude is normalised');
  const across = { type: 'Polygon', coordinates: [[[179, -1], [-179, -1], [-179, 1], [179, 1], [179, -1]]] };
  assert.ok([179.5, -179.5, 180, -180].every((lon) => pointInGeometry(lon, 0, across)) && !pointInGeometry(178.5, 0, across), 'a ring with an uncut seam edge is unwrapped');
});
test('REVIEW FOCUS 2: a view over the pole: lat/lon stay in range, blocks clamp to the top rows, the grid round-trips', () => {
  const view = { lat: 84, lon: 179.9, rangeKm: 900, tilt: 0.9, heading: 0 }, cam = poseCamera(view), frame = createLocalFrame(view.lat, view.lon);
  const g = matrixGrid(cam, view, { nx: 16, ny: 9, margin: 1 }).filter(Boolean);
  assert.ok(g.length > 100 && g.every((p) => p.lat <= 90 && p.lat >= 60 && p.lon >= -180 && p.lon < 180), `${g.length} ground points`);
  assert.ok(g.some((p) => p.lat > 85.06) && g.some((p) => p.lon > 0) && g.some((p) => p.lon < 0), 'the footprint crosses the pole and the Mercator limit');
  assert.ok(g.every((p) => /^3\/[0-7]\/[01]$/.test(blockKey(p.lat, p.lon))));
  const north = new Map(Array.from({ length: 16 }, (_, i) => [`3/${i % 8}/${i >> 3}`, 'test']));
  assert.equal(viewSplit(g, north, { has: () => false }), 'test');
  north.set(blockKey(89, 0), 'val'); assert.equal(viewSplit(g, north, { has: () => false }), null);
  for (const p of g) {
    const a = pixelRay(cam, p.px, p.py, 896, 504), q = frame.toLocal(p.lat, p.lon, 0), d = q.map((x, i) => x - a.origin[i]), l = Math.hypot(...d);
    assert.ok(Math.acos(Math.min(1, (d[0] * a.dir[0] + d[1] * a.dir[1] + d[2] * a.dir[2]) / l)) < 1e-6);
  }
});
test('OOD mask: Oceania land buffered by 300 km wraps across ±180; islets smaller than a cell are kept', () => {
  const ood = buildOodMask(ne);
  const cases = [[0, 179.9, true], [0, -180, true], [0, 180, true], [0.5, -177.5, true], [0, 177, true], [0, 175, false], [0, -175, false], [5, 5, false], [89.9, 0, false], [-90, 180, false]];
  for (const [lat, lon, want] of cases) assert.equal(ood.has(lat, lon), want, `${lat},${lon}`);
  const islet = { type: 'Polygon', coordinates: [[[179.05, -8.55], [179.2, -8.55], [179.2, -8.45], [179.05, -8.45], [179.05, -8.55]]] };
  const o2 = buildOodMask({ layers: { ne_10m_admin_0_countries: [{ props: { CONTINENT: 'Oceania' }, geom: islet, bbox: [179.05, -8.55, 179.2, -8.45] }] } });
  assert.ok(o2.has(-8.5, 179.1) && o2.has(-8.5, -178.5) && !o2.has(-8.5, 175));
});
test('featuresInView projects NE points through the camera across the seam, with or without cam.eye', () => {
  const view = { lat: 0, lon: 179.95, rangeKm: 300, tilt: 0, heading: 0 }, cam = poseCamera(view), f = featuresInView(ne, cam, view);
  const beta = f.find((x) => x.name === 'Beta'), gamma = f.find((x) => x.name === 'Gamma');
  assert.ok(beta && gamma && gamma.px > beta.px, 'east is to the right at heading 0');
  assert.ok(!f.some((x) => x.name === 'Alpha'), 'the far side of the globe is culled');
  const p = project(cam, createLocalFrame(0, 179.95).toLocal(0, 179.5, 0), 896, 504); assert.ok(Math.abs(p.x - beta.px) < 1e-9);
  delete cam.eye; assert.deepEqual(featuresInView(ne, cam, view), f);
  assert.equal(nearestPlace(ne, 0, 179.9).compass, 'west'); assert.equal(nearestPlace(ne, 0, 179.9, { exclude: new Set(['Beta']) }).name, 'Alpha');
  assert.equal(nearestPlace(ne, 0, 179.9, { exclude: new Set(['Beta', 'Alpha']) }), null, 'Gamma is below the 50k population floor');
  assert.deepEqual([regionOf(0, 0, 896, 504), regionOf(448, 252, 896, 504), regionOf(895, 503, 896, 504)], ['upper left', 'centre', 'lower right']);
  assert.deepEqual([0, 350, 90, 225].map(compassOf), ['north', 'north', 'east', 'south-west']);
});
test('zoomFacts: every Z fact is visual, the 16x9 grid is stored, and the record validates', () => {
  const view = { lat: 45.976, lon: 7.658, rangeKm: 50, tilt: 0.6, heading: 1.1 }, cam = poseCamera(view), rec = JSON.parse(fs.readFileSync(FIX + 'record_Z.json', 'utf8'));
  const info = { ...view, camAltKm: 41.3, clearKm: 38.9, L0: 13 }, lighting = lightingOf({ mode: 'always_day' });
  rec.facts = zoomFacts(info, cam, { origin: view, lighting, rings: [{ level: 13, layer_id: 's2cloudless_3857' }] });
  assert.equal(rec.facts['grid.latlon'].v.length, 144); assert.ok(Object.values(rec.facts).every((x) => x.obs === 'visual'));
  assert.deepEqual([rec.facts['view.range_bin'].v, rec.facts['sun.class'].v, rec.facts['view.tilt_deg'].v], [1, 'day', 34.3775]);
  assert.ok(rec.facts['grid.approx_km'].v > 20 && Math.abs(Math.hypot(...rec.facts['camera.eye_km'].v) - 50) < 1e-3);
  assert.deepEqual(validateRecord(rec), { ok: true, errors: [] });
});
test('the elevation reader decodes cached Terrarium tiles once, wraps longitude, and is null without a cached tile', async () => {
  const png = fs.readFileSync(FIX + 'terrarium_4x4.png'), asked = [], root = terrariumUrl(0, 0, 0);
  assert.equal(root, 'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/0/0/0.png');
  const reader = createElevationReader({ get: (url) => { asked.push(url); return url === root ? { body: png, ctype: 'image/png' } : null; } });
  const got = await Promise.all([[80, -170], [80, 100], [80, 180], [0, 0], [80, -190]].map(([la, lo]) => reader.at(la, lo, 0)));
  assert.deepEqual(got, [-100, 4807, -100, 0, 4807]); assert.equal(asked.filter((u) => u === root).length, 1, 'decoded once');
  assert.equal(await reader.at(10, 10, 3), null);
  assert.equal(R_KM, 6371);
});
test('loadNaturalEarth skips features without coordinates (NE v5.1.2 rivers hold an empty MultiLineString for the Loire)', async () => {
  const d = fs.mkdtempSync(path.join(os.tmpdir(), 'apv-ne-')), f = (geometry, name) => ({ type: 'Feature', properties: { name }, geometry });
  fs.writeFileSync(path.join(d, 'ne_10m_rivers_lake_centerlines.geojson'), JSON.stringify({ type: 'FeatureCollection', features: [f({ type: 'MultiLineString', coordinates: [] }, 'Loire'), f(null, 'Nowhere'), f({ type: 'MultiLineString', coordinates: [[[7, 46], [8, 46.5]]] }, 'Rhone')] }));
  const r = await loadNaturalEarth(d, ['ne_10m_rivers_lake_centerlines']); fs.rmSync(d, { recursive: true, force: true });
  assert.deepEqual(r.layers.ne_10m_rivers_lake_centerlines.map((x) => [x.props.name, x.point]), [['Rhone', [7.5, 46.25]]]);
});
