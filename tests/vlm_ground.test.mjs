// v1 grounding labels (ruling V1-2): the build's ground.* facts from the snapshot / v0 facts / cameras, the box, point and
// lat/lon claims and their support, and the grounding text items, each verified like every template item.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { clipBox, normBox, groundHazards, cvApproach, project, groundRunway, addGroundFacts } from '../vlm/gen/build/ground.js';
import { parseClaims, makeGazetteer, BASE_NAMES, verifyTemplateItem, verifyFreeText } from '../vlm/gen/text/verify.js';
import { boxHolds, pointHolds, iou } from '../vlm/gen/text/verify_ground.js';
import { groundItems, GROUND_ANSWERS } from '../vlm/gen/text/ground_items.js';
import { defaultBank } from '../vlm/gen/text/paraphrase.js';
import { mulberry32 } from '../src/mathx.js';
import { countryName } from '../vlm/gen/build/zoomgeo.js';

const bank = defaultBank(), GAZ = makeGazetteer([...BASE_NAMES, 'Lake Geneva', 'Switzerland', 'Bern']);
const V = (v, obs = 'visual', unit = null) => ({ v, unit, obs });
const H = (i, kind, box, dist, extra = {}) => ({ i, kind, r: 1, c: [dist, 0, 0], v: [-2, 0, 0], q: null, in_frame: true, box_px: box, visible: true, fogged: false, in_cloud: false, occluded: null, cam_dist: dist, ...extra });
function recS(hazards, facts = {}) {
  return { key: 'S_t_00000_000001', family: 'S', narrator_frame: 'raw/t/S/S_t_00000_000001.f2.png', frames: [], cameras: {}, safety: null,
    facts: { 'hazards.count_in_frame': V(hazards.length), 'kinds_in_frame': V([...new Set(hazards.map((h) => h.kind))]), ...facts },
    snapshot: { v: 1, scene: null, frames: { 'S_t_00000_000001.f2.png': { ship: { p: [0, 0, 0], v: [10, 0, 0], q: [0, 0, 0, 1] }, cloud_at_ship: 0, fog: null, hazards } } } };
}

test('ground facts: boxes clipped to the frame and normalised to 0-100 (x first); nearest first; the fog/visibility gate and the 2 px² floor', () => {
  assert.deepEqual(clipBox([-10, 20, 30, 600]), [0, 20, 30, 504]); assert.equal(clipBox([900, 10, 950, 20]), null); assert.deepEqual(normBox([0, 0, 896, 504]), [0, 0, 100, 100]);
  const r = recS([H(0, 'rock', [448, 252, 538, 302], 30), H(1, 'comet', [100, 50, 120, 70], 12), H(2, 'rock', [10, 10, 11, 11], 5), H(3, 'satellite', [600, 300, 640, 340], 8, { visible: false, fogged: true })]);
  const g = groundHazards(r);
  assert.equal(g.source, 'snapshot'); assert.equal(g.complete, true);
  assert.deepEqual(g.hazards.map((h) => h.kind), ['comet', 'rock'], 'nearest first; the 1 px² rock and the fogged satellite are left out');
  assert.deepEqual(g.hazards[1].box, [50, 50, 60, 60]); assert.deepEqual(g.hazards[1].pt, [55, 55]);
  assert.ok(g.hazards.every((h) => h.alone));
  const occ = groundHazards(recS([H(0, 'rock', [100, 100, 200, 200], 10), H(1, 'rock', [120, 120, 190, 190], 20)])).hazards;
  assert.deepEqual(occ.map((h) => h.alone), [true, false], 'a box more than half covered by a nearer one is not counted');
});
test('ground facts: a v0 record (no snapshot) lists the six nearest hazards from its facts; complete only when that is every hazard in frame', () => {
  const f = (i, kind, box) => ({ [`hazard.${i}.kind`]: V(kind), [`hazard.${i}.in_frame`]: V(!!box), [`hazard.${i}.box_px`]: V(box) });
  const rec = { family: 'A', narrator_frame: 'raw/x/A/k.f2.png', facts: { 'hazards.count_in_frame': V(2), ...f(0, 'airliner', [800, 0, 1000, 200]), ...f(1, 'rock', null), ...f(2, 'birds', [0, 0, 90, 50]) } };
  const g = groundHazards(rec); assert.equal(g.source, 'facts'); assert.equal(g.complete, true); assert.deepEqual(g.hazards.map((h) => h.box), [[89, 0, 100, 40], [0, 0, 10, 10]]);
  rec.facts['hazards.count_in_frame'] = V(3); assert.equal(groundHazards(rec).complete, false);
  addGroundFacts(rec); assert.equal(rec.facts['ground.hazards'].obs, 'visual'); assert.equal(rec.facts['ground.cv_sim'], undefined, 'no velocities in v0 facts');
});
test('constant-velocity closest approach (cv_sim, sim units): head-on, passing and receding', () => {
  assert.deepEqual(cvApproach([0, 0, 0], [0, 0, 0], [10, 0, 0], [-2, 0, 0], 1), { cpa_cv_sim_u: -1, tca_cv_sim_s: 5, ttc_cv_sim_s: 4.5 });
  const pass = cvApproach([0, 0, 0], [0, 0, 0], [10, 3, 0], [-2, 0, 0], 1); assert.equal(pass.tca_cv_sim_s, 5); assert.equal(pass.cpa_cv_sim_u, 2); assert.equal(pass.ttc_cv_sim_s, null);
  assert.equal(cvApproach([0, 0, 0], [0, 0, 0], [10, 0, 0], [2, 0, 0], 1).tca_cv_sim_s, 0, 'receding: the closest approach is now');
  const r = recS([H(0, 'rock', [448, 252, 538, 302], 30)]); addGroundFacts(r);
  assert.equal(r.facts['ground.cv_sim'].obs, 'context', 'velocities are not visual: never Narrator text'); assert.equal(r.facts['ground.cv_sim'].v[0].tca_cv_sim_s, 2.5);
});
test('the runway outline from the chase camera: projected corners, a point at the near threshold; null behind the camera or in fog', () => {
  const cam = { matrixWorldInverse: [0, 0, -1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, -100, -1000, 1], projectionMatrix: [1.2, 0, 0, 0, 0, 2.14, 0, 0, 0, 0, -1, -1, 0, 0, -0.2, 0] };
  const p = project(cam, [0, 0, 0]); assert.ok(p && p[0] > 440 && p[0] < 456 && p[1] > 252, `${p}`);
  const rec = { family: 'L', narrator_frame: 'raw/x/L/k.f2.png', cameras: { 'k.f2.png': cam }, facts: {},
    snapshot: { scene: { runway: { corners: [[0, 0, -30], [0, 0, 30], [4000, 0, 30], [4000, 0, -30]] } }, frames: { 'k.f2.png': { aircraft: { p: [-900, 90, 0] }, fog_density: 2e-5 } } } };
  const g = groundRunway(rec); assert.ok(g && g.box.every((x) => x >= 0 && x <= 100) && g.pt[1] >= g.box[1] && g.pt[1] <= g.box[3], JSON.stringify(g));
  assert.equal(groundRunway({ ...rec, snapshot: { ...rec.snapshot, frames: { 'k.f2.png': { fog_density: 0.003 } } } }), null, 'fogged (exp² transmittance < 0.1 at the threshold)');
  assert.equal(groundRunway({ ...rec, facts: { 'scene.in_cloud': V(true) } }), null, 'inside the cloud deck');
});
test('claims: boxes, points and lat/lon pairs are parsed before numbers; a box holds at IoU >= 0.5 or within 2 units, a point inside or near', () => {
  const c = parseClaims('The nearest one is a rock at [48,44,54,55]. Here: (51,50). The view is centred near 46° N, 7° E, in Switzerland.', GAZ);
  assert.deepEqual(c.filter((x) => ['box', 'point', 'latlon'].includes(x.type)).map((x) => [x.type, x.value]), [['box', [48, 44, 54, 55]], ['point', [51, 50]], ['latlon', [46, 7]]]);
  assert.ok(!c.some((x) => x.type === 'number'), 'no digit of a box, point or coordinate is read as a number');
  assert.deepEqual(c.find((x) => x.type === 'box').subject, { kind: 'hazard', value: 'rock' });
  assert.deepEqual(parseClaims('Near 33° S, 70° W.', GAZ).find((x) => x.type === 'latlon').value, [-33, -70]);
  assert.ok(boxHolds([10, 10, 20, 20], [11, 10, 21, 20]) && boxHolds([10, 10, 12, 12], [12, 12, 14, 14]) && !boxHolds([10, 10, 20, 20], [30, 30, 40, 40]));
  assert.ok(iou([0, 0, 10, 10], [0, 0, 10, 10]) === 1 && pointHolds([5, 5], [0, 0, 4, 4], null) && !pointHolds([9, 9], [0, 0, 4, 4], null));
});
test('verifier: grounding claims are checked against the ground facts, with their subject; a wrong box, point, count or coordinate is rejected', () => {
  const r = recS([H(0, 'rock', [448, 252, 538, 302], 30), H(1, 'comet', [100, 50, 120, 70], 12)]); addGroundFacts(r);
  const ok = (t) => verifyFreeText(t, r, { gaz: GAZ }).verified;
  assert.ok(ok('The nearest one is a comet at [11,10,13,14].') && ok('The rock is at [50,50,60,60].') && ok('It is at (55,55).'));
  assert.ok(!ok('The rock is at [11,10,13,14].'), 'the box of the comet, said of the rock'); assert.ok(!ok('The comet is at [70,70,90,90].')); assert.ok(!ok('It is at (80,90).'));
  assert.ok(ok('I can point to two hazards: (12,12) and (55,55).') && !ok('I can point to three hazards: (12,12), (55,55) and (60,60).'));
  const z = { family: 'Z', facts: { 'view.lat_deg': V(46.4), 'view.lon_deg': V(7.2), 'place.country': V('Switzerland'), 'ground.features': V([{ name: 'Lake Geneva', kind: 'lake', pt: [30, 60] }]) }, zoom: { tags: [], range_bin: 1 } };
  const zok = (t) => verifyFreeText(t, z, { gaz: GAZ }).verified;
  assert.ok(zok('The view is centred near 46° N, 7° E, in Switzerland.') && !zok('The view is centred near 48° N, 7° E.') && !zok('Near 46° S, 7° E.'));
  assert.ok(zok('You can find Lake Geneva at (32,58).') && !zok('You can find Lake Geneva at (70,20).'));
  const sea = { ...z, facts: { ...z.facts, 'ground.features': V([{ name: 'Southend-on-Sea', kind: 'place', pt: [50, 50] }, { name: 'Novy Port', kind: 'place', pt: [19, 55] }]) } };
  const g2 = makeGazetteer([...BASE_NAMES, 'Southend-on-Sea', 'Novy Port']), sok = (t) => verifyFreeText(t, sea, { gaz: g2 }).verified;
  assert.ok(sok('In this view, Southend-on-Sea is at (50,50).') && sok('The frame places Novy Port at (19,55).'), 'a feature whose name holds a sea or port noun');
  assert.ok(!sok('In this view, Southend-on-Sea is at (90,10).'), 'the point must still sit on a feature');
});
test('grounding items: every one verifies on its record, the grounding templates all fire, and a tampered answer fails', () => {
  const S = recS([H(0, 'rock', [448, 252, 538, 302], 30), H(1, 'comet', [100, 50, 120, 70], 12), H(2, 'satellite', [700, 300, 760, 360], 40)]); addGroundFacts(S);
  const Z = { key: 'Z_t', family: 'Z', facts: { 'view.lat_deg': V(46.4), 'view.lon_deg': V(7.2), 'place.country': V('Switzerland'), 'ground.features': V([{ name: 'Lake Geneva', kind: 'lake', pt: [30, 60] }, { name: 'Bern', kind: 'place', pt: [60, 40] }]) }, zoom: { tags: [], range_bin: 1 } };
  const Zsea = { ...Z, facts: { ...Z.facts, 'place.country': V(null), 'view.lat_deg': V(-40.2), 'view.lon_deg': V(-150.6) } };
  const cam = { matrixWorldInverse: [0, 0, -1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, -100, -1000, 1], projectionMatrix: [1.2, 0, 0, 0, 0, 2.14, 0, 0, 0, 0, -1, -1, 0, 0, -0.2, 0] };
  const L = { key: 'L_t', family: 'L', narrator_frame: 'raw/x/L/k.f2.png', cameras: { 'k.f2.png': cam }, facts: {}, snapshot: { scene: { runway: { corners: [[0, 0, -30], [0, 0, 30], [4000, 0, 30], [4000, 0, -30]] } }, frames: { 'k.f2.png': { fog_density: 2e-5 } } } };
  addGroundFacts(L);
  const aids = new Set();
  for (const [rec, k] of [[S, 1], [S, 2], [Z, 1], [Zsea, 1], [L, 1], [S, 3], [Z, 4]]) for (const split of ['train', 'test']) for (const it of groundItems(rec, { bank, rng: mulberry32(k), split })) {
    const c = verifyTemplateItem(it, rec, { gaz: GAZ }); assert.ok(c.verified, `${it.template_id}: ${it.prompt} -> ${it.answer}: ${c.errors}`); assert.equal(it.task, 'grounding');
    aids.add(it.template_id.split('+')[1]);
  }
  assert.deepEqual(GROUND_ANSWERS.filter((a) => !aids.has(a)), [], 'every grounding answer template is produced');
  const it = groundItems(S, { bank, rng: mulberry32(1), split: 'train' }).find((x) => x.family_q === 'ground_box');
  assert.equal(verifyTemplateItem({ ...it, answer: it.answer.replace(/\[[\d,]+\]/, '[1,1,3,3]') }, S, { gaz: GAZ }).verified, false);
});
const SMOKE = '/Volumes/LaCie/astro-pilot/vlm/raw/v1-smoke';
test('real v1 smoke records: every grounding item of S, A and L verifies (skipped without the smoke run)', { skip: !fs.existsSync(SMOKE) && 'no raw/v1-smoke' }, () => {
  let n = 0;
  for (const f of ['S', 'A', 'L']) for (const name of fs.readdirSync(`${SMOKE}/${f}`).filter((x) => /^[SAL]_.*\.json$/.test(x)).slice(0, 60)) {
    const r = JSON.parse(fs.readFileSync(`${SMOKE}/${f}/${name}`, 'utf8')); r.narrator_frame = `raw/v1-smoke/${f}/${r.narrator_frame}`; addGroundFacts(r);
    for (const it of groundItems(r, { bank, rng: mulberry32(n), split: 'train' })) { const c = verifyTemplateItem(it, r, { gaz: GAZ }); assert.ok(c.verified, `${name} ${it.answer}: ${c.errors}`); n++; }
  }
  assert.ok(n > 50, `${n} items`);
});
test('Z country names: the full ADMIN name where Natural Earth abbreviates (the verifier reads "Dem." as a sentence end)', () => {
  assert.equal(countryName({ NAME: 'Dem. Rep. Congo', ADMIN: 'Democratic Republic of the Congo' }), 'Democratic Republic of the Congo');
  assert.equal(countryName({ NAME: 'Fr. Polynesia', ADMIN: 'French Polynesia' }), 'French Polynesia'); assert.equal(countryName({ NAME: 'Chile', ADMIN: 'Chile' }), 'Chile');
  assert.equal(countryName({ NAME: 'Cyprus U.N. Buffer Zone', ADMIN: 'Cyprus No Mans Area' }), 'Cyprus No Mans Area'); assert.equal(countryName({ name: 'X' }), 'X');
  const t = verifyFreeText('The centre lies near 1° S, 29° E, in Democratic Republic of the Congo.', { family: 'Z', facts: { 'view.lat_deg': V(-1.2), 'view.lon_deg': V(29.1), 'place.country': V('Democratic Republic of the Congo') }, zoom: { tags: [] } }, { gaz: makeGazetteer([...BASE_NAMES, 'Democratic Republic of the Congo']) });
  assert.ok(t.verified, t.errors.join('; '));
});
test('Z compass claims: "Port Elizabeth ... to the north-west" binds to the place, not to a docking port (v1)', () => {
  const z = { family: 'Z', facts: { 'place.nearest': V({ name: 'Port Elizabeth', km: 5000, bearing: 315, compass: 'north-west' }) }, zoom: { tags: [] } };
  const gaz = makeGazetteer([...BASE_NAMES, 'Port Elizabeth']), ok = (t) => verifyFreeText(t, z, { gaz }).verified;
  assert.ok(ok('To the north-west, about 5,000 km away, lies Port Elizabeth.') && ok('The closest large town is Port Elizabeth, about 5,000 km away to the north-west.'));
  assert.ok(!ok('The closest large town is Port Elizabeth, about 5,000 km away to the south-east.'), 'the bearing still has to hold');
  const d = { family: 'D', facts: {} }; assert.equal(verifyFreeText('The port lies to the north-west.', d, { gaz }).verified, false, 'a docking port is never placed by a compass word');
});
test('a snapshot of an unexpected shape costs the record its grounding facts, never the build', () => {
  const r = recS([H(0, 'rock', [448, 252, 538, 302], 30, { c: null })]); addGroundFacts(r); assert.equal(r.facts['ground.hazards'].v.length, 1); assert.equal(r.facts['ground.cv_sim'], undefined, 'no velocity, no cv_sim');
  const bad = recS([]); bad.snapshot.frames['S_t_00000_000001.f2.png'].hazards = [{ get kind() { throw new Error('broken'); }, visible: true, in_frame: true, box_px: [1, 1, 50, 50], cam_dist: 1 }];
  const seen = []; addGroundFacts(bad, { onError: (x, e) => seen.push(e.message) });
  assert.deepEqual(seen, ['broken']); assert.equal(bad.facts['ground.hazards'], undefined);
});
