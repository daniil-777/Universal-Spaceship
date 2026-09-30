import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { boxResize } from '../vlm/gen/boxresize.js';
import { PALETTE, imageFacts, imageFactsOf } from '../vlm/gen/imagefacts.js';
import { seedSplit, splitOf, groupOf, assertGroupsDisjoint, zRecheck } from '../vlm/gen/build/split.js';
import { hamming, crossSplitDrops } from '../vlm/gen/build/dedupe.js';
import { targetsOf, REG } from '../vlm/gen/build/targets.js';
import { contextClassIds, narratorRows, eyeExcluded } from '../vlm/gen/build/export.js';
import { classWeights } from '../vlm/gen/build/balance.js';
import { textTally, distinctN } from '../vlm/gen/build/stats.js';
import { datasheet } from '../vlm/gen/build/datasheet.js';
import { zoomGeo } from '../vlm/gen/build/zoomgeo.js';
import { PALETTE_NAMES, TEXT_FACTS, TAG_WORDS, validateTextFacts, REASONS } from '../vlm/gen/schema.js';
import { EYE } from '../vlm/gen/safety.js';
import { loadNaturalEarth } from '../vlm/gen/geo/naturalearth.js';
import { poseCamera } from '../vlm/gen/labels/zoom.js';
// sharp is installed under vlm/ (vlm/package.json), not at the repo root
const sharp = createRequire(new URL('../vlm/package.json', import.meta.url))('sharp');

test('boxResize is an exact area average (constant stays constant; a 2x2 block averages)', () => {
  const c = new Uint8Array(896 * 504 * 4).fill(200), out = boxResize(c, 896, 504); assert.equal(out.length, 160 * 96 * 3); assert.ok(out.every((v) => v === 200));
  const s = new Uint8Array(4 * 4 * 4); for (let i = 0; i < 16; i++) s.set([i < 8 ? 0 : 100, 0, 0, 255], i * 4);
  assert.deepEqual([...boxResize(s, 4, 4, 2, 2)].filter((_, i) => i % 3 === 0), [0, 0, 100, 100]);
});
test('splits are by seed group; A moscow/cloudy and D far go to OOD; twins share their original split and group', () => {
  assert.deepEqual([0, 7, 8, 9, 10, 18].map(seedSplit), ['train', 'train', 'val', 'test', 'train', 'val']);
  const r = (family, seed, url, extra = {}) => ({ family, key: `${family}_r_${seed}`, provenance: { seed, page_url: url, twin_of: null, ...extra }, facts: {} });
  assert.equal(splitOf(r('A', 3, '/index.html?atmo=1&route=moscow&sky=fair')), 'ood'); assert.equal(splitOf(r('A', 3, '/index.html?atmo=1&route=alps&sky=cloudy')), 'ood');
  assert.equal(splitOf(r('D', 18, '/index.html?start=far&seed=18')), 'ood'); assert.equal(splitOf(r('L', 18, '/index.html?start=final&seed=18')), 'val');
  const recs = [r('S', 8, '/x'), r('S', 8, '/x', { twin_of: 'S_r_8' })]; assert.doesNotThrow(() => assertGroupsDisjoint(recs, recs.map((x) => splitOf(x))));
  assert.throws(() => assertGroupsDisjoint(recs, ['val', 'test']));
  // a twin whose own seed differs lands in its original's group and split
  const orig = r('L', 9, '/index.html?start=final'), twin = { ...r('L', 17, '/index.html?start=final', { twin_of: orig.key }), key: 'L_r_twin' }, byKey = new Map([[orig.key, orig], [twin.key, twin]]);
  assert.equal(splitOf(twin, { byKey }), 'test'); assert.equal(groupOf(twin, byKey), 'L:9');
  const zplan = { locations: [{ id: 4, split: 'val' }], spares: [{ id: 15, split: 'test' }] };
  assert.equal(splitOf({ ...r('Z', 15, '/vlm/capture/zoom.html?lat=1&lon=2'), family: 'Z' }, { zplan }), 'test');
});
test('the Z split re-check reads the captured camera on the 32x18 grid widened by one cell', () => {
  const view = { lat: 46, lon: 7.5, rangeKm: 30, tilt: 0, heading: 0 }, cam = poseCamera(view), rec = { key: 'Z_k', family: 'Z', frames: ['raw/r/Z/Z_k.f0.png'], cameras: { 'Z_k.f0.png': cam },
    facts: { 'view.lat_deg': { v: 46 }, 'view.lon_deg': { v: 7.5 } }, provenance: { seed: 0, page_url: '/vlm/capture/zoom.html?rs=1&licence=open&lat=46&lon=7.5' } };
  const all = (s) => ({ get: () => s }), none = { has: () => false }, cut = { get: (k) => (k === '3/4/2' ? 'val' : 'train') };
  assert.equal(zRecheck(rec, { blocks: all('val'), ood: none }), 'val');
  assert.equal(zRecheck(rec, { blocks: all('train'), ood: { has: () => true } }), 'ood');
  const far = { ...rec, cameras: { 'Z_k.f0.png': poseCamera({ ...view, lat: 45.1, lon: 0.01, rangeKm: 400 }) }, provenance: { seed: 0, page_url: '/z?lat=45.1&lon=0.01' } };
  assert.equal(zRecheck(far, { blocks: cut, ood: none }), null, 'a footprint across two blocks of different splits breaks its split');
});
test('dedupe: Hamming distance on 64-bit dHashes; cross-split near-duplicates leave val/test, never train', () => {
  assert.equal(hamming(0n, 0xffn), 8); assert.equal(hamming(0x8000000000000001n, 1n), 1); assert.equal(hamming(0xffffffffffffffffn, 0n), 64);
  const h = new Map([['a', 0n], ['b', 0x7n], ['c', 0xffffn], ['d', 0x1n]]), s = new Map([['a', 'train'], ['b', 'val'], ['c', 'test'], ['d', 'train']]);
  assert.deepEqual([...crossSplitDrops(h, s)].sort(), ['b']);
});
test('Pilot Eye targets and masks follow the family and observability rules', () => {
  const eye = { verdict: 'UNSAFE', severity: 3, reasons: ['HAZARD_AHEAD'], safe_actions: ['CLIMB'], p_ref: 0.75, ttc_s: 1.5 };
  const S = targetsOf({ family: 'S', safety_eye: eye, facts: { 'clearance.min_u': { v: 2 } }, zoom: null });
  assert.equal(S.targets.verdict, 2); assert.equal(S.masks.p_ref, 1); assert.deepEqual(S.masks.reg, [1, 1, 0, 0, 0, 0]); assert.equal(S.masks.zoom_tags, 0);
  const D = targetsOf({ family: 'D', safety_eye: { ...eye, safe_actions: [], p_ref: 1 }, facts: { rho_m: { v: 60 }, closing_cms: { v: 20 } }, zoom: null });
  assert.equal(D.masks.p_ref, 0); assert.equal(D.masks.reg[REG.indexOf('closing')], 0, 'closing is masked beyond the eye scope');
  assert.equal(targetsOf({ family: 'D', safety_eye: eye, facts: { rho_m: { v: EYE.closingRhoM }, closing_cms: { v: 20 } }, zoom: null }).masks.reg[REG.indexOf('closing')], 1);
  for (const r of ['LOW_FUEL', 'JET_FAILURE', 'NO_BREAKOUT_AVAILABLE']) assert.equal(D.masks.reasons[REASONS.indexOf(r)], 0);
  const L = targetsOf({ family: 'L', safety_eye: { ...eye, reasons: [] }, facts: {}, zoom: null });
  for (const r of ['TAILWIND', 'GLIDESLOPE_DEVIATION', 'SPEED_OUT_OF_BAND']) assert.equal(L.masks.reasons[REASONS.indexOf(r)], 0, r);
  for (const r of ['HIGH_SINK_RATE', 'RUNWAY_EDGE', 'CANNOT_STOP', 'LOCALIZER_DEVIATION']) assert.equal(L.masks.reasons[REASONS.indexOf(r)], 1, r);
  const Z = targetsOf({ family: 'Z', safety_eye: null, facts: {}, zoom: { tags: ['COASTLINE'], range_bin: 2 } }); assert.equal(Z.masks.verdict, 0); assert.equal(Z.targets.range_bin, 2);
});
test('D records inside 0.5 m axial are excluded from the Pilot Eye export only (ruling T10-g)', () => {
  assert.ok(eyeExcluded({ family: 'D', facts: { axial_m: { v: 0.42 } } })); assert.equal(eyeExcluded({ family: 'D', facts: { axial_m: { v: 0.5 } } }), null);
  assert.equal(eyeExcluded({ family: 'S', facts: {} }), null);
});
test('export drops items that need a context-class fact the row does not supply (rowContext is the one rule)', () => {
  const rec = { family: 'S', key: 'k', narrator_frame: 'raw/r/S/k.f2.png', facts: { 'hazard.0.kind': { v: 'rock', obs: 'visual' }, 'hazard.0.dist_u': { v: 3, obs: 'context', unit: 'u' }, 'ship.speed_m_s': { v: 266, obs: 'context', unit: 'm/s' } },
    safety_eye: { verdict: 'SAFE', severity: 0, reasons: [], best_action: 'CONTINUE', p_ref: 0, ttc_s: null }, row_monitor: { verdict: 'SAFE', severity: 0, reasons: [], action: 'CONTINUE', p_ref: 0, ttc_bin: 'none', clr_bin: '>40 u' }, texts: [
      { task: 'vqa', prompt: 'What is it?', answer: 'A rock.', fact_ids: ['hazard.0.kind'] }, { task: 'vqa', prompt: 'How far?', answer: 'About 50 m.', fact_ids: ['hazard.0.dist_u'] },
      { task: 'vqa', prompt: 'How fast?', answer: 'About 250 m/s.', fact_ids: ['ship.speed_m_s'] }, { task: 'safety', prompt: 'Safe?', answer: 'The monitor rates this SAFE.', fact_ids: ['safety.verdict'] },
      { task: 'vqa', prompt: 'Cause?', answer: 'Nothing.', fact_ids: ['safety.cause'], context_facts: ['safety.cause'], needsContext: true }] };
  assert.ok(contextClassIds(rec).has('hazard.0.dist_u') && contextClassIds(rec).has('safety.verdict'));
  const rows = narratorRows(rec, { rng: () => 0.9, split: 'train', mode: 'gt+noise' }), tasks = rows.map((r) => r.messages[1].content[0].text);
  assert.ok(!tasks.includes('About 50 m.'), 'the distance is not supplied by any Context line'); assert.ok(tasks.includes('The monitor rates this SAFE.') && tasks.includes('About 250 m/s.') && tasks.includes('Nothing.'));
  for (const r of rows) if (r.fact_ids.some((f) => f.startsWith('safety.') || f === 'ship.speed_m_s')) { assert.match(r.messages[0].content[1].text, /^Context: telemetry: speed 266 m\/s; monitor: SAFE/); assert.equal(r.context_src, 'gt+noise'); }
  const plain = rows.find((r) => r.fact_ids[0] === 'hazard.0.kind'); assert.equal(plain.context, false); assert.equal(plain.messages[0].content[1].text, 'What is it?');
  const low = narratorRows(rec, { rng: () => 0.1, split: 'train', mode: 'gt+noise' }).find((r) => r.fact_ids[0] === 'hazard.0.kind'); assert.equal(low.context, true, 'a visual row carries the Context with p = 0.5');
});
test('class-balanced weights per family x verdict sum to the record count', () => {
  const recs = [...Array(9)].map((_, i) => ({ key: `a${i}`, family: 'S', safety_eye: { verdict: 'SAFE' } })).concat([{ key: 'b', family: 'S', safety_eye: { verdict: 'UNSAFE' } }]);
  const w = classWeights(recs); assert.ok(Math.abs([...w.values()].reduce((a, b) => a + b, 0) - 10) < 1e-9); assert.ok(w.get('b') > w.get('a0'));
});
test('image facts: the 12 PALETTE_NAMES, brightness bins on mean luminance in [0, 1], values that pass validateTextFacts', async () => {
  assert.deepEqual(Object.keys(PALETTE), [...PALETTE_NAMES]);
  const flat = (rgb, W = 20, H = 10) => { const d = new Uint8Array(W * H * 3); for (let i = 0; i < W * H; i++) d.set(rgb, i * 3); return d; };
  assert.equal(imageFactsOf(flat([40, 40, 40]), 20, 10)['image.brightness_bin'].v, 'dark');
  assert.equal(imageFactsOf(flat([110, 110, 110]), 20, 10)['image.brightness_bin'].v, 'medium');
  assert.equal(imageFactsOf(flat([200, 200, 200]), 20, 10)['image.brightness_bin'].v, 'bright');
  const sea = imageFactsOf(flat([12, 32, 70]), 20, 10); assert.equal(sea['image.palette_0'].v, 'blue', 'a dark sea is blue, not black'); assert.equal(sea['image.edge_bin'].v, 'smooth');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'apv-if-')), file = path.join(dir, 'x.png'), W = 896, H = 504, px = Buffer.alloc(W * H * 3);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) px.set(((x >> 4) + (y >> 4)) % 2 ? [230, 130, 30] : [15, 15, 15], (y * W + x) * 3);
  await sharp(px, { raw: { width: W, height: H, channels: 3 } }).png().toFile(file);
  const F = await imageFacts(file); fs.rmSync(dir, { recursive: true, force: true });
  assert.deepEqual(new Set([F['image.palette_0'].v, F['image.palette_1'].v]), new Set(['orange', 'black'])); assert.equal(F['image.edge_bin'].v, 'highly textured');
  assert.ok(Object.values(F).every((f) => f.obs === 'visual')); assert.ok(F['image.mean_lum'].v > 0 && F['image.mean_lum'].v < 1);
  const rec = { family: 'S', facts: F }; assert.deepEqual(validateTextFacts(rec), { ok: true, errors: [] });
});
test('Z geo facts: land-only elevation, water_frac with lakes, admin-1 at the centre, TEXT_FACTS shapes, terrain words from TAG_WORDS', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'apv-zg-')), sq = (x0, y0, x1, y1) => [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]], F = (layer, props, geometry) => ({ type: 'Feature', properties: { layer, ...props }, geometry });
  fs.writeFileSync(path.join(dir, 'mini.geojson'), JSON.stringify({ type: 'FeatureCollection', features: [
    F('ne_10m_land', {}, { type: 'Polygon', coordinates: sq(0, 40, 7.45, 50) }), F('ne_10m_admin_0_countries', { NAME: 'Westland', CONTINENT: 'Europe' }, { type: 'Polygon', coordinates: sq(0, 40, 7.45, 50) }),
    F('ne_10m_admin_1_states_provinces', { name: 'Upper Province' }, { type: 'Polygon', coordinates: sq(7, 40, 7.45, 50) }), F('ne_10m_lakes', { name: 'Big Lake', scalerank: 3 }, { type: 'Polygon', coordinates: sq(7.2, 45.6, 7.45, 46.4) }),
    F('ne_10m_populated_places', { NAME: 'Centreville', POP_MAX: 200000, SCALERANK: 2 }, { type: 'Point', coordinates: [7.45, 46.02] }), F('ne_10m_populated_places', { NAME: 'Farburg', POP_MAX: 90000 }, { type: 'Point', coordinates: [5, 44] })] }));
  const ne = await loadNaturalEarth(dir, ['mini']); fs.rmSync(dir, { recursive: true, force: true });
  const view = { lat: 46, lon: 7.4, rangeKm: 60, tilt: 0, heading: 0 }, cam = poseCamera(view);
  const { matrixGrid } = await import('../vlm/gen/labels/zoom.js'), grid = matrixGrid(cam, view).map((g) => (g ? [g.lat, g.lon] : null));
  const rec = { key: 'Z_t', family: 'Z', frames: ['raw/r/Z/Z_t.f0.png'], cameras: { 'Z_t.f0.png': cam }, provenance: { page_url: '/vlm/capture/zoom.html?lat=46&lon=7.4' }, zoom: { tags: null, range_bin: 1, lighting: { class: 'day' } },
    facts: { 'grid.latlon': { v: grid }, 'view.rings': { v: [11, 10] }, 'view.range_km': { v: 60 }, 'view.lat_deg': { v: 46 }, 'view.lon_deg': { v: 7.4 } } };
  await zoomGeo.apply(rec, { ne, elev: { at: async (lat, lon) => (lon < 7.45 ? 1800 + 1000 * (lat - 45.8) : -3000) } });
  const f = (id) => rec.facts[id].v;
  assert.ok(f('geo.sea_frac') > 0.3 && f('geo.sea_frac') < 0.7, `sea_frac ${f('geo.sea_frac')}`); assert.ok(f('geo.water_frac') > f('geo.sea_frac'), 'lakes count as water');
  assert.ok(f('geo.elev_min_m') >= 1000, 'the seabed (-3000 m) never enters the land elevation stats'); assert.equal(f('geo.coast_side'), 'right');
  assert.equal(f('place.country'), 'Westland'); assert.equal(f('place.admin1'), 'Upper Province'); assert.equal(f('place.nearest').name, 'Farburg');
  assert.ok(f('place.in_view').some((x) => x.name === 'Centreville' && x.kind === 'place')); assert.ok(rec.zoom.tags.includes('COASTLINE'));
  assert.equal(rec.zoom.tags.includes('WATER_DOMINANT'), f('geo.water_frac') > 0.6);
  const words = Object.values(TAG_WORDS).map((w) => w[0]); assert.ok(words.some((w) => f('geo.terrain').includes(w)));
  for (const id of Object.keys(TEXT_FACTS).filter((k) => TEXT_FACTS[k].families.includes('Z') && !k.startsWith('image.'))) assert.ok(rec.facts[id], id);
  Object.assign(rec.facts, imageFactsOf(new Uint8Array(30).fill(90), 5, 2)); assert.deepEqual(validateTextFacts(rec), { ok: true, errors: [] });
});
test('text stats: reject rates per family and per template component; distinct-n', () => {
  const t = textTally(); t.add('S', { texts: [{ template_id: 'a+b' }, { template_id: 'a' }], rejected: [{ item: { template_id: 'a' }, errors: ['x'], parserOk: false }] });
  t.add('L', { texts: [{ template_id: 'c' }], rejected: [] }); const s = t.summary();
  assert.equal(s.byFamily.S.rejectRate, +(1 / 3).toFixed(4)); assert.equal(s.byFamily.L.rejectRate, 0); assert.equal(s.byTemplate.a.rejected, 1); assert.equal(s.byTemplate.a.total, 3);
  assert.equal(s.parserRejects, 1); assert.equal(distinctN(['a b c', 'a b d'], 1), 4 / 6);
});
test('the datasheet has the 7 sections and the amendment B records', () => {
  const md = datasheet({ counts: {}, text: { rejectRate: 0.01, parserFalseReject: 0 }, upstream: {}, naturalMix: {}, pairs: {} }, { name: 'x', date: 'd', runs: ['r'], git_sha: 's', licence: 'open', capture_mode: 'clock', sizes: null, c_near: 2.5, eye: EYE, playwright: '1.63.0' });
  for (let i = 1; i <= 7; i++) assert.match(md, new RegExp(`^## ${i} `, 'm'));
  for (const s of [...EYE.hidden, 'closingRhoM 11', '1600-frame slot', '80,000', '12 u', 'T_VIS 0.1', 'axial < 0.5 m', 'scene.in_cloud']) assert.ok(md.includes(s), s);
});
test('a record of a page without .done is left out; an A twin needs its own .done and its original page\'s, not the twin page start', async () => {
  const { missingDone } = await import('../vlm/gen/build/split.js'), A = (key, pv) => ({ key, family: 'A', provenance: { twin_of: null, ...pv } });
  const done = new Set([30, 31, 35, 50035]);
  assert.equal(missingDone(A('A_r_00035_000053', { page_episode: 30 }), done), null);
  assert.equal(missingDone(A('A_r_50035_000053', { page_episode: 50030, twin_of: 'A_r_00035_000053' }), done), null, 'the twin page start 50030 never gets a .done');
  assert.equal(missingDone(A('A_r_00033_000010', { page_episode: 30 }), done), 33);
  assert.equal(missingDone(A('A_r_00031_000010', { page_episode: 29 }), done), 29, 'a segment of a page whose first episode is unfinished');
  assert.equal(missingDone(A('A_r_50035_000053', { page_episode: 50030, twin_of: 'A_r_00035_000053' }), new Set([35, 50035])), 30);
  assert.equal(missingDone({ key: 'S_r_50002_000100', family: 'S', provenance: { twin_of: 'S_r_00002_000100' } }, new Set([50002])), 2);
});
