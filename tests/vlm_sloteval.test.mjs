import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { makeGazetteer, BASE_NAMES } from '../vlm/gen/text/verify.js';
import { parseContext } from '../vlm/gen/text/context.js';
import { wilson, scoreView, score, gates, scoreCorrupted, zoomRows, blindRows, corruptRows, recordNames } from '../vlm/gen/text/slot_eval.mjs';

const F = (v, unit = null, obs = 'visual') => ({ v, unit, obs });
const zrec = (key, over = {}) => ({ key, family: 'Z', split: 'test', narrator_frame: `raw/r/Z/${key}.f0.png`,
  facts: { 'place.nearest': F({ name: 'Podgorica', km: 76.3, bearing: 165, compass: 'south' }), 'place.country': F('Montenegro'), 'place.admin1': F('Zabljak'),
    'place.in_view': F([{ name: 'Durmitor', kind: 'peak', region: 'lower centre' }]), 'view.range_km': F(8.77, 'km'), 'sun.class': F('day'),
    'image.palette_0': F('grey'), 'image.palette_1': F('pink'), 'image.palette_2': F('brown'), 'image.brightness_bin': F('medium'), ...(over.facts || {}) },
  zoom: { tags: ['HILLS'], range_bin: 0, lighting: { class: 'day' }, ...(over.zoom || {}) } });
const srec = (key, verdict = 'SAFE') => ({ key, family: 'S', split: 'test', narrator_frame: `raw/r/S/${key}.f2.png`,
  facts: { 'ship.speed_m_s': F(266, 'm/s', 'context'), 'clearance.min_u': F(12, 'u', 'context'), 'hazards.count_in_frame': F(0, 'count') },
  safety_eye: { verdict, severity: verdict === 'SAFE' ? 0 : 2, reasons: verdict === 'SAFE' ? [] : ['HAZARD_AHEAD'], best_action: verdict === 'SAFE' ? 'CONTINUE' : 'CLIMB', p_ref: 0.1, ttc_s: null } });
const gaz = makeGazetteer([...BASE_NAMES, 'Montenegro', 'Podgorica', 'Durmitor', 'Zabljak', 'Paris', 'France']);
const GOOD = 'The image is a view of hills in Montenegro, taken from about 10 km. Off the edge of the view, Podgorica lies about 75 km to the south. '
  + 'The ground is seen in full daylight. Most of the frame is grey, followed by pink and then brown. The image is of medium brightness.';
const tally = (checks) => checks.reduce((a, [slot, inList, ok, wrong]) => ({ ...a, [slot]: [...(a[slot] || []), [inList, ok, wrong]] }), {});

test('wilson: 0/0 is null, 5/10 matches the closed form', () => {
  assert.equal(wilson(0, 0), null); const [lo, hi] = wilson(5, 10);
  assert.ok(Math.abs(lo - 0.2366) < 1e-3 && Math.abs(hi - 0.7634) < 1e-3);
});

test('scoreView: a dataset-style caption hits every checklist slot and states nothing wrong', () => {
  const t = tally(scoreView(GOOD, zrec('z1'), gaz));
  for (const s of ['range', 'light', 'tag', 'country', 'nearest', 'palette', 'brightness']) assert.deepEqual(t[s], [[true, true, false]], s);
  assert.equal(t.entity, undefined);   // the place distance (75 km, south) and ground resolution are not range claims
});

test('scoreView: wrong range, night, an absent tag, a foreign place and a wrong colour are hallucinations', () => {
  const t = tally(scoreView('Mountains in France seen at night from about 800 km, near Paris. The frame is mostly purple and dark.', zrec('z1'), gaz));
  assert.deepEqual(t.range, [[true, false, true]]); assert.deepEqual(t.light, [[true, false, true]]); assert.deepEqual(t.country, [[true, false, false]]);
  assert.deepEqual(t.tag, [[true, false, false], [false, false, true]]); assert.deepEqual(t.entity, [[false, false, true], [false, false, true]]);
  assert.deepEqual(t.palette, [[true, false, true]]); assert.deepEqual(t.brightness, [[true, false, true]]);
});

test('scoreView: silence states nothing; an in-view feature named is a correct extra entity', () => {
  const t = tally(scoreView('A picture.', zrec('z1'), gaz)); assert.ok(Object.values(t).flat().every(([inList, ok, wrong]) => inList && !ok && !wrong));
  assert.deepEqual(tally(scoreView('Durmitor is in view.', zrec('z1'), gaz)).entity, [[false, true, false]]);
});

test('score and gates: recall over the core checklist, hallucination over stated slots, Wilson CIs', () => {
  const recs = new Map([['z1', zrec('z1')], ['z2', zrec('z2', { zoom: { tags: ['COASTLINE', 'HILLS'] } })]]);
  const good = score([{ key: 'z1', text: GOOD }, { key: 'z2', text: GOOD }], recs, gaz), none = score([{ key: 'z1', text: 'A picture.' }, { key: 'z2', text: 'A picture.' }], recs, gaz);
  assert.equal(good.n, 2); assert.equal(good.recall_core, 10 / 11); assert.equal(good.halluc, 0); assert.equal(none.recall_core, 0); assert.equal(none.halluc, 'n/a');
  assert.equal(good.slots.tag.total, 3); assert.equal(good.slots.tag.hit, 2); assert.ok(good.recall_ci[0] < good.recall && good.recall < good.recall_ci[1]);
  assert.deepEqual(gates(good, none), { recall: true, hallucination: true }); assert.equal(gates(none, good).recall, false);
  assert.throws(() => score([{ key: 'nope', text: 'x' }], recs, gaz), /no record/);
});

test('scoreCorrupted: restating the Context verdict passes; a contradicted image fact fails', () => {
  const recs = new Map([['s1', srec('s1')]]), line = 'Context: telemetry: speed 266 m/s; monitor: UNSAFE (severity 2), reasons: HAZARD_AHEAD, action: CLIMB, p_ref 0.1, TTC none, clearance 5-15 u.';
  const P = [{ key: 's1', prompt: line + '\nIs the situation safe? Explain.', monitor_verdict: 'UNSAFE' }];
  assert.deepEqual(scoreCorrupted(P, [{ key: 's1', text: 'The monitor rates this UNSAFE and advises to climb.' }], recs, gaz), { restated: 1, n: 1, contradictions: 0, pass: true });
  const bad = scoreCorrupted(P, [{ key: 's1', text: 'The monitor rates this UNSAFE; there are three rocks ahead.' }], recs, gaz);
  assert.equal(bad.contradictions, 1); assert.equal(bad.pass, false);
  assert.equal(scoreCorrupted(P, [{ key: 's1', text: 'All looks SAFE.' }], recs, gaz).restated, 0);
});

test('eval rows: zoom prompts, a train-majority blind baseline and a verdict-forcing corrupted Context', () => {
  const recs = [zrec('z1', { zoom: { tags: ['HILLS'] } }), { ...zrec('z2'), split: 'train' }, { ...zrec('z3'), split: 'ood' }, srec('s1'), srec('s2', 'UNSAFE'), { ...srec('s3', 'CAUTION'), split: 'train' }];
  assert.deepEqual(zoomRows(recs, 'test').map((r) => [r.key, r.image, r.prompt]), [['z1', 'raw/r/Z/z1.f0.png', 'Describe the image in detail.']]);
  assert.match(zoomRows(recs, 'test', { context: true })[0].prompt, /^Context: telemetry: place Podgorica, range 8\.77 km, sun day\.\nDescribe the image in detail\.$/);
  const [blind] = blindRows(recs), bt = tally(scoreView(blind.text, recs[0], gaz));
  assert.equal(blind.key, 'z1'); for (const s of ['range', 'light', 'tag', 'country', 'nearest', 'palette', 'brightness']) assert.equal(bt[s][0][1], true, `${s}: ${blind.text}`);
  const cr = corruptRows(recs, { n: 50, seed: 5 }); assert.equal(cr.length, 2);
  for (const r of cr) {
    const gt = recs.find((x) => x.key === r.key).safety_eye.verdict, m = parseContext(r.prompt.split('\n')[0]).monitor;
    assert.notEqual(r.monitor_verdict, gt); assert.equal(m.verdict, r.monitor_verdict); assert.match(r.prompt, /\nIs the situation safe\? Explain\.$/);
  }
  assert.deepEqual(recordNames(recs).sort(), ['Durmitor', 'Montenegro', 'Podgorica', 'Zabljak']);
});

test('CLI: scores output files, writes eval rows, and exits 1 when a gate fails', () => {
  const d = fs.mkdtempSync(path.join(os.tmpdir(), 'sloteval-')), w = (f, rows) => fs.writeFileSync(path.join(d, f), rows.map((r) => JSON.stringify(r)).join('\n') + '\n');
  // two views: with one view (7 stated slots, 0 wrong) the Wilson upper bound is 0.35 > 0.30 and the gate rightly fails
  w('records.jsonl', [zrec('z1'), zrec('z3'), { ...zrec('z2'), split: 'train' }, srec('s1')]);
  w('merged.jsonl', [{ key: 'z1', text: GOOD }, { key: 'z3', text: GOOD }]); w('base.jsonl', [{ key: 'z1', text: 'A picture.' }, { key: 'z3', text: 'A picture.' }]);
  const cli = fileURLToPath(new URL('../vlm/gen/text/slot_eval.mjs', import.meta.url)), run = (...a) => spawnSync(process.execPath, [cli, ...a, '--geo', 'none'], { encoding: 'utf8' });
  const ok = run('--outputs', path.join(d, 'merged.jsonl'), '--base', path.join(d, 'base.jsonl'), '--records', path.join(d, 'records.jsonl'));
  assert.equal(ok.status, 0, ok.stderr); const out = JSON.parse(ok.stdout); assert.deepEqual(out.gates, { recall: true, hallucination: true }); assert.equal(out.merged.recall_core, 1);
  assert.equal(run('--outputs', path.join(d, 'base.jsonl'), '--base', path.join(d, 'merged.jsonl'), '--records', path.join(d, 'records.jsonl')).status, 1);
  const mk = run('--make-rows', path.join(d, 'eval'), '--records', path.join(d, 'records.jsonl'));
  assert.equal(mk.status, 0, mk.stderr);
  for (const f of ['zoom_test.jsonl', 'zoom_test_ctx.jsonl', 'zoom_ood.jsonl', 'blind.jsonl', 'corrupt_rows.jsonl', 'stop_set.jsonl', 'parity_facts.jsonl']) assert.ok(fs.existsSync(path.join(d, 'eval', f)), f);
  const stop = JSON.parse(fs.readFileSync(path.join(d, 'eval', 'stop_set.jsonl'), 'utf8').split('\n')[0]); assert.equal(stop.image, '/Volumes/LaCie/astro-pilot/vlm/raw/r/Z/z1.f0.png');
  fs.rmSync(d, { recursive: true, force: true });
});
