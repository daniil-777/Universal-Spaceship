import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { validateRecord, recordKey, rangeBin, REASONS, ACTIONS, FAMILY_ACTIONS, fact } from '../vlm/gen/schema.js';

const load = (f) => JSON.parse(fs.readFileSync(new URL(`./vlm_fixtures/${f}`, import.meta.url), 'utf8'));
const S = () => load('record_S.json'), Z = () => load('record_Z.json');
const rejects = (rec, re) => { const r = validateRecord(rec); assert.equal(r.ok, false, 'expected a rejection'); assert.ok(r.errors.some((e) => re.test(e)), `no error matching ${re}: ${r.errors}`); };

test('the fixtures validate', () => { for (const r of [S(), Z()]) assert.deepEqual(validateRecord(r), { ok: true, errors: [] }); });
test('enums and helpers', () => {
  assert.equal(REASONS.length, 26); assert.equal(ACTIONS.length, 11);
  assert.deepEqual(FAMILY_ACTIONS.D, ['CONTINUE', 'BREAKOUT', 'HOLD_POSITION', 'SLOW_DOWN']);
  assert.equal(recordKey('S', 'apv0', 12, 345), 'S_apv0_00012_000345');
  assert.deepEqual([5, 20, 99, 400, 1499, 1500, 2400].map(rangeBin), [0, 1, 1, 3, 3, 4, 4]);
  assert.deepEqual(fact(1, 'm', 'visual'), { v: 1, unit: 'm', obs: 'visual' });
});
test('rejects non-finite numbers and facts without obs', () => {
  const a = S(); a.facts['ship.speed_u_s'].v = Infinity; rejects(a, /non-finite/);
  const b = S(); b.safety.p_ref = NaN; rejects(b, /non-finite/);
  const c = S(); delete c.facts['hazard.0.kind'].obs; rejects(c, /has no obs/);
});
test('rejects imagery without licence or layer, any Esri source, EOX 2018-2025 in open', () => {
  const a = Z(); delete a.render.imagery[0].licence; rejects(a, /missing licence or layer_id/);
  const b = Z(); b.render.imagery[0].source = 'Esri World Imagery'; rejects(b, /Esri/);
  const c = Z(); c.render.imagery[0].layer_id = 's2cloudless-2025_3857'; rejects(c, /2018-2025 in the open profile/);
  const d = Z(); d.render.licence_profile = 'nc'; d.render.imagery[0].layer_id = 's2cloudless-2025_3857'; d.render.imagery[0].licence = 'CC BY-NC-SA 4.0'; assert.equal(validateRecord(d).ok, true);
});
test('rejects wrong frame counts and a D record without chase.png', () => {
  const a = S(); a.frames.pop(); rejects(a, /frames.length must be 3/);
  const b = Z(); b.frames.push('x.png'); rejects(b, /frames.length must be 1/);
  const d = S(); d.family = 'D'; d.key = 'D_fix_00001_000123'; d.safety.macro = null; d.safety_eye.macro = null; rejects(d, /D record without chase.png/);
});
test('rejects renderScale or dpr != 1, and an L record not in chase or with the path drawn', () => {
  const a = S(); a.render.renderScale = 0.85; rejects(a, /renderScale and dpr must be 1/);
  const b = S(); b.render.dpr = 2; rejects(b, /renderScale and dpr must be 1/);
  const l = S(); l.family = 'L'; l.render.view.eye = 'cinema'; l.render.path = false; rejects(l, /L eye view must be chase/);
  const m = S(); m.family = 'L'; m.render.path = null; rejects(m, /L path must be false/);
});
test('rejects A below 0.995 atmosphere, an L timeout, an L seed < 1', () => {
  const a = S(); a.family = 'A'; a.provenance.atmosphere = 0.99; rejects(a, /atmosphere < 0.995/);
  const l = S(); l.family = 'L'; l.render.path = false; l.safety.action_outcome = { CONTINUE: 'timeout' }; rejects(l, /timeout/);
  const s = S(); s.family = 'L'; s.render.path = false; s.provenance.seed = 0; rejects(s, /L seed must be >= 1/);
});
test('rejects a runtime injection applied less than 2 frame_dt + 1 step before f2', () => {
  const a = S(); a.provenance.injection = { kind: 'collision_course', params: {}, step: 117, sim_t_s: 7.8 }; rejects(a, /injection/);
  const b = S(); b.provenance.injection = { kind: 'collision_course', params: {}, step: 116, sim_t_s: 7.73 }; assert.equal(validateRecord(b).ok, true);
  const c = S(); c.provenance.injection = { kind: 'hflare', params: {}, step: null, sim_t_s: null }; assert.equal(validateRecord(c).ok, true);
});
test('rejects enum violations in the safety block', () => {
  const a = S(); a.safety.reasons = ['NOT_A_REASON']; rejects(a, /reason/);
  const b = S(); b.safety.safe_actions = ['GO_AROUND']; rejects(b, /safe_actions/);
});
