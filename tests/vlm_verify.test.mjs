import { test } from 'node:test';
import assert from 'node:assert/strict';
import { roundNice, fmtSlot, parseClaims, makeGazetteer, verifyFreeText, verifyTemplateItem } from '../vlm/gen/text/verify.js';
import { makeNegative } from '../vlm/gen/text/vqa.js';

const gaz = makeGazetteer(['Zermatt', 'Switzerland', 'Italy', 'Tokyo', 'New York', 'Matterhorn']);
const F = (v, unit, obs = 'context') => ({ v, unit, obs });
const rec = { family: 'S', facts: { 'hazard.0.kind': F('rock', null, 'visual'), 'hazard.0.dist_u': F(12.3, 'u'), 'hazard.0.clock': F(2, 'clock', 'visual'), 'hazards.count_in_frame': F(2, 'count', 'visual'),
  'kinds_in_frame': F(['rock'], null, 'visual'), 'hazard.0.side': F('right', null, 'visual'), 'hazard.0.box_px': F([700, 100, 760, 160], 'px', 'visual') },
  safety: { verdict: 'UNSAFE', reasons: ['HAZARD_AHEAD'], best_action: 'CLIMB', action_outcome: { CONTINUE: 'crash_possible' } } };
const zrec = { family: 'Z', facts: { 'place.nearest': F({ name: 'Zermatt', km: 4, compass: 'north-east', bearing: 40 }, null, 'visual'), 'place.country': F('Switzerland', null, 'visual'),
  'place.in_view': F([{ name: 'Matterhorn', region: 'upper left' }], null, 'visual'), 'view.range_km': F(50, 'km', 'visual') }, safety: null };

test('one rounding table: nice steps 1, 2, 2.5, 5; "about 20 m" covers 17.5-22.5', () => {
  assert.deepEqual(roundNice(19.3), { value: 20, lo: 17.5, hi: 22.5 }); assert.deepEqual(roundNice(3.2), { value: 3, lo: 2.5, hi: 3.5 });
  const s = fmtSlot('hazard.0.dist_u', rec.facts['hazard.0.dist_u']); assert.equal(s.text, 'about 250 m'); assert.ok(s.lo <= 12.3 && s.hi >= 12.3);
  assert.equal(fmtSlot('hazard.0.dist_u', rec.facts['hazard.0.dist_u'], { system: 'imperial' }).text, 'about 750 ft');
});
test('parser: numbers with units, number words, clock, region, compass, verdict/action/reason words, entities', () => {
  const c = parseClaims('Two rocks at 3 o\'clock, about 250 m away in the upper left; the monitor rates this UNSAFE: climb. Zermatt lies north-east.', gaz);
  const t = (type) => c.filter((x) => x.type === type).map((x) => x.value);
  assert.deepEqual(t('count'), [2]); assert.deepEqual(t('clock'), [3]); assert.deepEqual(t('region'), ['upper left']); assert.deepEqual(t('compass'), ['north-east']);
  assert.deepEqual(t('verdict'), ['UNSAFE']); assert.deepEqual(t('action'), ['CLIMB']); assert.deepEqual(t('entity'), ['Zermatt']);
  assert.equal(c.find((x) => x.type === 'number').value, 250);
});
test('verifier catches wrong numbers, wrong places and unknown capitalised names; accepts the rounding interval', () => {
  assert.equal(verifyFreeText('A rock about 250 m ahead.', rec, { gaz }).verified, true);
  assert.equal(verifyFreeText('A rock about 400 m ahead.', rec, { gaz }).verified, false);
  assert.equal(verifyFreeText('This is Tokyo, seen from about 50 km.', zrec, { gaz }).verified, false);
  assert.equal(verifyFreeText('Near Zermatt, seen from about 50 km.', zrec, { gaz }).verified, true);
  assert.equal(verifyFreeText('A rock near Zorblax Station.', rec, { gaz }).verified, false);
  assert.equal(verifyFreeText('The rock is at 9 o\'clock.', rec, { gaz }).verified, false);
  assert.equal(verifyFreeText('Zermatt is to the south.', zrec, { gaz }).verified, false);
});
test('template text is checked twice: slot intervals contain the facts, and the parser recovers exactly the slot set', () => {
  const slot = fmtSlot('hazard.0.dist_u', rec.facts['hazard.0.dist_u']);
  const item = { task: 'vqa', prompt: 'How far is the nearest rock?', answer: `The nearest rock is ${slot.text} away.`, slots: [slot], fact_ids: ['hazard.0.dist_u'] };
  assert.deepEqual(verifyTemplateItem(item, rec, { gaz }), { verified: true, parserOk: true, errors: [] });
  const bad = { ...item, slots: [{ ...slot, lo: 20, hi: 30 }] }; assert.equal(verifyTemplateItem(bad, rec, { gaz }).verified, false);
  const extra = { ...item, answer: `${item.answer} It is 3 o'clock.` }; const r = verifyTemplateItem(extra, rec, { gaz }); assert.equal(r.parserOk, false);
});
test('negatives: absent hazard, wrong number and wrong place pass; a negative whose premise is true is rejected', () => {
  const rng = () => 0.3;
  for (const [type, r] of [['absent', rec], ['number', rec], ['place', zrec]]) { const n = makeNegative(r, type, rng, gaz); assert.ok(n && n.verified, `${type}: ${JSON.stringify(n)}`); assert.ok(n.false_premise); }
  const lie = makeNegative(rec, 'absent', rng, gaz, { forceKind: 'rock' }); assert.equal(lie, null);
});

// ---- the verifier beyond the brief's cases ----
import { runTeacher } from '../vlm/gen/text/teacher.js';
const t2 = (text, g = gaz) => parseClaims(text, g).map((c) => [c.type, c.type === 'range' ? [c.lo, c.hi, c.unit] : c.value]);
const drec = { family: 'D', facts: { station_distance_bin: F('2-20 m', null, 'visual'), rho_m: F(8.2, 'm'), jets_failed: F(['P6']), papi_whites_cam: F(null, 'count', 'visual') }, safety: { verdict: 'UNSAFE', reasons: ['KOS_VIOLATION'], best_action: 'NONE_SAFE', safe_actions: [] } };
test('ranges: bins read as intervals; a monitor-attributed bin must equal the Context bin; string bin facts count', () => {
  assert.deepEqual(t2('between 95 and 285 m, under 1 second, over 1,500 km, 1-3 s'), [['range', [95, 285, 'm']], ['range', [0, 1, 's']], ['range', [1500, Infinity, 'km']], ['range', [1, 3, 's']]]);
  const mon = { monitor: { verdict: 'UNSAFE', severity: 3, reasons: ['HAZARD_AHEAD'], action: 'CLIMB', p_ref: 1, ttc_bin: '1-3 s', clr_bin: '5-15 u' } };
  assert.equal(verifyFreeText('The monitor expects contact in between 1 and 3 seconds.', rec, { gaz, context: mon }).verified, true);
  assert.equal(verifyFreeText('The monitor expects contact in between 3 and 6 seconds.', rec, { gaz, context: mon }).verified, false);
  assert.equal(verifyFreeText('The monitor reports a clearance of between 95 and 285 m.', rec, { gaz, context: mon }).verified, true);
  assert.equal(verifyFreeText('The station is between 2 and 20 m away.', drec, { gaz }).verified, true);
  assert.equal(verifyFreeText('The station is between 20 and 100 m away.', drec, { gaz }).verified, false);
});
test('§5.7 obsRule: without Context only visual facts; a Context supplies its telemetry ids and, with a monitor, safety.*', () => {
  assert.equal(verifyFreeText('A rock about 250 m ahead.', rec, { gaz, obsRule: true }).verified, false);
  assert.equal(verifyFreeText('A rock at 2 o\'clock in the image.', rec, { gaz, obsRule: true }).verified, true);
  assert.equal(verifyFreeText('A rock about 250 m ahead.', rec, { gaz, obsRule: true, context: { telemetry: [['dist', '12.3 u', 'hazard.0.dist_u']], monitor: null } }).verified, true);
  assert.equal(verifyFreeText('The monitor rates this UNSAFE.', rec, { gaz }).verified, false, 'a flight row without Context');
  assert.equal(verifyFreeText('The monitor rates this SAFE.', rec, { gaz, context: { monitor: { verdict: 'SAFE', reasons: [], action: 'CONTINUE' } } }).verified, true, 'monitor-attributed');
});
test('parser: "no safe action" is an action, not SAFE; reason words count only beside a verdict, a cause or the monitor', () => {
  assert.deepEqual(t2('The monitor rates this UNSAFE because of a keep-out sphere violation and finds no safe action.'), [['verdict', 'UNSAFE'], ['reason', 'KOS_VIOLATION'], ['action', 'NONE_SAFE']]);
  assert.equal(verifyFreeText('The monitor rates this UNSAFE because of a keep-out sphere violation and finds no safe action.', drec, { gaz, context: { monitor: { verdict: 'UNSAFE', reasons: ['KOS_VIOLATION'], action: 'NONE_SAFE' } } }).verified, true);
  assert.deepEqual(t2('A tailwind of about 5 kt and severe turbulence.'), [['number', 5]]);
  assert.deepEqual(t2('It is unsafe because of a tailwind.').map((c) => c[0]), ['verdict', 'reason']);
});
test('parser: acronyms and bodies are terms, not places; a sentence-initial common word is not a place; your N o\'clock is the pilot bearing', () => {
  const g = makeGazetteer(['PAPI', 'Earth', 'Split', 'Orange', 'Zermatt']);
  assert.deepEqual(t2('The PAPI shows two white and two red lights; the Earth is in view.', g), [['count', 2], ['count', 2]]);
  assert.deepEqual(t2('Split the wind into components. Orange and blue dominate. The view is near Split.', g), [['entity', 'Split']]);
  assert.deepEqual(t2('A comet at your 1 o\'clock; the rock sits at 2 o\'clock in the image.'), [['bearing_clock', 1], ['clock', 2]]);
  const brec = { ...rec, facts: { ...rec.facts, 'hazard.0.bearing_clock': F(1, 'clock') } };
  assert.equal(verifyFreeText('The rock is at your 1 o\'clock and in the upper right of the image.', brec, { gaz }).verified, true, 'box_px 730,130 lies in the upper right');
  assert.equal(verifyFreeText('One jet failed: P6. No white lights show.', drec, { gaz }).verified, false, 'papi unknown: no count claim about it holds');
  assert.equal(verifyFreeText('One jet failed: P6.', drec, { gaz }).verified, true);
});
test('teacher: only visual facts are sent, output is verified as a Context-free row, the key is redacted from errors', async () => {
  const sent = [], logs = [], send = async (req) => { sent.push(req); return { text: req.facts['hazard.0.dist_u'] ? 'x' : 'A rock at 2 o\'clock in the image, about 250 m away.' }; };
  const out = await runTeacher([rec], { send, model: 'm', priceIn: 3, priceOut: 15, yesSpend: true, env: { ANTHROPIC_API_KEY: 'sk-k' }, log: (s) => logs.push(s), gaz });
  assert.ok(sent.length === 1 && !sent[0].facts['hazard.0.dist_u'] && sent[0].facts['hazard.0.clock']); assert.equal(out.length, 0, 'the distance is context-class');
  const ok = await runTeacher([rec], { send: async () => ({ text: 'A rock at 2 o\'clock in the image.' }), model: 'm', priceIn: 3, priceOut: 15, yesSpend: true, env: { ANTHROPIC_API_KEY: 'sk-k' }, log: () => {}, gaz });
  assert.equal(ok.length, 1); assert.equal(ok[0].generator.model, 'm'); assert.match(ok[0].generator.prompt_sha256, /^[0-9a-f]{64}$/);
  await runTeacher([rec], { send: async () => { throw new Error('401 for key sk-k'); }, model: 'm', priceIn: 3, priceOut: 15, yesSpend: true, env: { ANTHROPIC_API_KEY: 'sk-k' }, log: (s) => logs.push(s), gaz });
  assert.ok(!logs.join('\n').includes('sk-k') && logs.some((l) => l.includes('[redacted]')));
});
