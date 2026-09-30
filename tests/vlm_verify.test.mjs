import { test } from 'node:test';
import assert from 'node:assert/strict';
import { roundNice, fmtSlot, parseClaims, makeGazetteer, verifyFreeText, verifyTemplateItem, derive } from '../vlm/gen/text/verify.js';
import { aboutHolds } from '../vlm/gen/text/verify_numbers.js';
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
// T10-v adversarial: the fog/cloud visibility gate (landing.js) nulls papi_whites_cam and windsock.from_deg once the
// chase camera is inside cloud; the verifier must reject a claim about either on such a record, while still accepting
// the in-cloud claim itself and a claim about the aircraft's own visual facts (cfg.gear stays visible, close in)
const F2 = (v, unit, obs) => ({ v, unit, obs });
const lrec = { family: 'L', facts: { papi_whites_cam: F2(null, 'count', 'visual'), 'windsock.from_deg': F2(null, 'deg', 'visual'), 'scene.in_cloud': F2(true, null, 'visual'), 'cfg.gear': F2('down', null, 'visual') }, safety: null };
test('T10-v: an in-cloud L record rejects a PAPI or windsock claim (their facts are null) and accepts the in-cloud claim', () => {
  assert.equal(verifyFreeText('The PAPI shows two white and two red lights, placing the aircraft on the glide path.', lrec, { gaz }).verified, false);
  assert.equal(verifyFreeText('A windsock by the runway indicates wind from about 250 degrees.', lrec, { gaz }).verified, false);
  assert.equal(verifyFreeText('The camera is inside cloud.', lrec, { gaz }).verified, true);
  assert.equal(verifyFreeText('The landing gear is down.', lrec, { gaz }).verified, true, "cfg.gear stays a real visual fact close to the chase camera");
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
import { corruptMonitor } from '../vlm/gen/text/context.js';
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
  assert.deepEqual(t2('A tailwind of about 5 kt and severe turbulence.'), [['category', 'tail'], ['number', 5], ['category', 'severe']], 'wind and turbulence facts, not reasons');
  assert.deepEqual(t2('It is unsafe because of a tailwind.').map((c) => c[0]), ['verdict', 'reason']);
});
test('parser: acronyms and bodies are terms, not places; a sentence-initial common word is not a place; your N o\'clock is the pilot bearing', () => {
  const g = makeGazetteer(['PAPI', 'Earth', 'Split', 'Orange', 'Zermatt']);
  const papi = t2('The PAPI shows two white and two red lights; the Earth is in view.', g);
  assert.deepEqual(papi, [['presence', 'papi'], ['count', 2], ['count', 2], ['presence', 'earth']]);
  assert.deepEqual(t2('Split the wind into components. Orange and blue dominate. The view is near Split.', g).filter((c) => c[0] !== 'category'), [['entity', 'Split']]);
  assert.deepEqual(t2('A comet at your 1 o\'clock; the rock sits at 2 o\'clock in the image.'), [['kind', 'comet'], ['bearing_clock', 1], ['kind', 'rock'], ['clock', 2]]);
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
test("monitor attribution is per sentence; a monitor-attributed action must be the monitor's (GT safe_actions do not leak in)", () => {
  const r = { family: 'S', facts: rec.facts, safety: { ...rec.safety, safe_actions: ['CLIMB', 'TURN_LEFT'] } }, mon = { monitor: { verdict: 'SAFE', reasons: [], action: 'CONTINUE' } };
  assert.equal(verifyFreeText('The monitor rates this SAFE. The situation is UNSAFE.', r, { gaz, context: mon }).verified, true, 'second sentence is checked against the safety block');
  assert.equal(verifyFreeText('The monitor rates this SAFE. The situation is SAFE.', r, { gaz, context: mon }).verified, false);
  assert.equal(verifyFreeText('The monitor rates this SAFE. It advises the pilot to continue.', r, { gaz, context: mon }).verified, true, 'a continuation sentence stays the monitor\'s');
  assert.equal(verifyFreeText('The monitor advises the pilot to turn left.', r, { gaz, context: mon }).verified, false, 'TURN_LEFT is a GT safe action, not the monitor\'s');
  assert.equal(verifyFreeText('The monitor advises the pilot to continue.', r, { gaz, context: { monitor: { ...mon.monitor, action: 'CLIMB' } } }).verified, false, 'continue after an advice cue is an action claim');
});
test('outcomes, categories and R9 descriptors are claims; a number binds to the quantity noun of its clause', () => {
  const s = { family: 'S', facts: { ...rec.facts, 'hazard.0.size_bin': F('medium', null, 'visual'), 'sun.lit': F(1, null, 'visual'), 'image.palette_0': F('blue', null, 'visual'), 'image.palette_1': F('white', null, 'visual') }, safety: rec.safety };
  const ok = (t, r = s, o = {}) => verifyFreeText(t, r, { gaz, ...o }).verified;
  assert.ok(ok('A medium-sized rock in direct sunlight; the image is mostly blue and white.') && !ok('A tiny rock.') && !ok('The rock is in shadow.') && !ok('The image is mostly red.'));
  assert.ok(!ok('A badly damaged, burning rock.') && !ok('A stunning view of a rock.'));
  const mon = (p_ref) => ({ monitor: { verdict: 'UNSAFE', reasons: ['HAZARD_AHEAD'], action: 'CLIMB', p_ref, ttc_bin: 'none', clr_bin: 'none' } });
  assert.ok(ok('The monitor expects that, if nothing changes, a crash is certain.', s, { context: mon(1) }) && !ok('The monitor expects that, if nothing changes, a crash is certain.', s, { context: mon(0.5) }));
  assert.ok(!ok('If nothing changes, the aircraft touches down and completes the landing.', s, { context: mon(1) }), 'an L outcome on an S record');
  const l = { family: 'L', facts: { ias_kt: F(60, 'kt'), 'wind.head_kt': F(-11, 'kt'), 'wind.cross_kt': F(3, 'kt') }, safety: { verdict: 'UNSAFE', reasons: [], best_action: 'NONE_SAFE', action_outcome: { CONTINUE: 'overrun' } } };
  assert.ok(ok('A tailwind of about 10 kt with about 3 kt of crosswind from the right.', l) && !ok('A tailwind of about 60 kt with about 3 kt of crosswind from the right.', l), 'about 60 kt is the IAS, not the tailwind');
  assert.ok(!ok('If nothing changes, the aircraft overruns the end of the runway.', l, { context: { monitor: { verdict: 'SAFE', reasons: [], action: 'CONTINUE', p_ref: null } }, obsRule: true }), 'an outcome the monitor verdict contradicts');
});
test('corruptMonitor copies the donor tuple whole (p_ref included) and refuses a pool of another family', () => {
  const m = { verdict: 'UNSAFE', severity: 3, reasons: ['HAZARD_AHEAD'], action: 'CLIMB', p_ref: 1, ttc_bin: '1-3 s', clr_bin: '<5 u' }, pool = [{ family: 'S', verdict: 'SAFE', severity: 0, reasons: [], action: 'CONTINUE', p_ref: 0, ttc_bin: 'none', clr_bin: 'none' }];
  const c = corruptMonitor(m, { rng: () => 0.01, confusion: null, pool, family: 'S' }); assert.deepEqual([c.verdict, c.action, c.p_ref], ['SAFE', 'CONTINUE', 0]);
  assert.throws(() => corruptMonitor(m, { rng: () => 0.01, confusion: null, pool, family: 'A' }), /pool entry of family S/);
});

// ---- fix round 2 ----
const ok = (t, r, o = {}) => verifyFreeText(t, r, { gaz, ...o }).verified;
const sRec = { family: 'S', facts: { ...rec.facts, 'hazard.0.size_bin': F('medium', null, 'visual'), 'hazard.0.closing_u_s': F(21.5, 'u/s'),
  'hazard.1.kind': F('comet'), 'hazard.1.in_frame': F(false, null, 'visual'), 'hazard.1.clock': F(null, 'clock', 'visual'),
  'hazard.1.closing_u_s': F(-3, 'u/s'), moon_in_frame: F(false, null, 'visual') }, safety: { ...rec.safety, cause: 'rock' } };
const sMon = { monitor: { verdict: 'UNSAFE', severity: 3, reasons: ['HAZARD_AHEAD'], action: 'CLIMB', p_ref: 1, ttc_bin: 'none', clr_bin: 'none' } };
test('round 2: a benign L/D outcome never stands next to an UNSAFE monitor', () => {
  const l = { family: 'L', facts: { 'cfg.gear': F('down', null, 'visual') },
    safety: { verdict: 'CAUTION', reasons: [], best_action: 'CONTINUE', action_outcome: { CONTINUE: 'landed' } } };
  const mon = (verdict) => ({ monitor: { verdict, severity: 3, reasons: ['CANNOT_STOP'], action: 'GO_AROUND', p_ref: null, ttc_bin: 'none',
    clr_bin: 'none' } });
  const say = 'If nothing changes, the aircraft touches down and completes the landing.';
  assert.equal(ok(say, l, { context: mon('UNSAFE'), obsRule: true }), false);
  assert.equal(ok(say, l, { context: mon('SAFE'), obsRule: true }), true);
});
test('round 2: no false rejects for closing in D and L, world tags, METAR clouds, the binding window and common openers', () => {
  const d = { family: 'D', facts: { closing_cms: F(12.1, 'cm/s'), speed_limit_cms: F(30, 'cm/s'), phase: F('FINAL', null, 'visual') }, safety: null };
  assert.ok(ok('The spacecraft is approaching the station.', d) && !ok('The spacecraft is moving away from the station.', d));
  assert.ok(ok('The spacecraft is approaching the station.', { ...d, facts: { phase: d.facts.phase } }), 'no closing rate: the phase');
  assert.ok(ok('The spacecraft must not exceed about 30 cm/s.', d), 'a limit, not a negated speed');
  const l = { family: 'L', facts: { vert_mode: F('GS'), wow: F(false), 'scene.clouds': F('-RA SCT030', null, 'visual'), ias_kt: F(142.3, 'kt'),
    'wind.head_kt': F(12.4, 'kt') }, safety: null };
  assert.ok(ok('The aircraft approaches the runway under scattered clouds in the rain.', l));
  assert.ok(!ok('The aircraft approaches the runway.', { ...l, facts: { ...l.facts, vert_mode: F('ROLLOUT'), wow: F(true) } }));
  assert.ok(ok('The aircraft is at about 140 kt with a headwind of about 12 kt.', l) && !ok('The sky is overcast.', l));
  assert.deepEqual(['SCT030', 'OVC003', '-RA BKN012', 'NSC', 'broken', 'few'].map((x) => derive(['cloud_code'], x)),
    ['SCT', 'OVC', 'BKN', 'NSC', 'BKN', 'FEW']);
  const a = { family: 'A', facts: { world: F('mountains', null, 'visual'), route: F('alps', null, 'visual') }, safety: null };
  assert.ok(ok('The aircraft flies over the mountains and hilly terrain.', a) && !ok('The aircraft flies over city areas.', a));
  for (const o of ['Watch', 'Keep', 'Floating', 'Bathed', 'Glinting', 'Built-up']) assert.ok(ok(`${o}, a rock sits at 2 o'clock in the image.`, rec), o);
  assert.ok(!ok('Zorblax lies ahead.', rec) && !ok('It is near Zorblax town.', rec));
  const g = makeGazetteer(['Split']);
  assert.deepEqual(parseClaims('Split lies on the coast.', g).map((c) => c.type), ['entity']);
  assert.deepEqual(parseClaims('Split the view in two.', g).map((c) => c.type), []);
});
test('round 2: position claims bind to their subject; kinds, causes and disagreements are claims', () => {
  assert.ok(ok("The rock is at 2 o'clock in the image.", sRec) && !ok("The Moon is at 2 o'clock in the image.", sRec));
  assert.ok(!ok("The comet is at 2 o'clock in the image.", sRec) && !ok('The Moon is on the right side of the image.', sRec));
  assert.ok(ok('The rock is on the right side of the image.', sRec) && !ok('An airliner is on the right side of the image.', sRec));
  assert.ok(ok('The Matterhorn sits in the upper left of the image.', zrec) && !ok('Zermatt sits in the upper left of the image.', zrec));
  assert.ok(ok('There is a rock in the frame.', sRec) && !ok('There is a comet in the frame.', sRec), 'the comet is out of the frame');
  assert.ok(ok('The rock is closing in.', sRec) && !ok('The rock is moving away.', sRec), 'the receding hazard is the comet');
  assert.ok(!ok('The comet is moving away.', sRec), 'a kind named with an article must be in the frame');
  assert.ok(ok('The biggest threat is the rock.', sRec, { context: sMon }) && !ok('The biggest threat is the satellite.', sRec, { context: sMon }));
  assert.ok(!ok('The biggest threat is the rock.', sRec, { context: { monitor: { ...sMon.monitor, reasons: ['STALL'] } } }), 'not a monitor reason');
  assert.ok(!ok('The monitor rates this UNSAFE. But the monitor is wrong here.', sRec, { context: sMon }));
  assert.ok(!ok('Despite the monitor, the flight is fine.', sRec, { context: sMon }) && !ok('The monitor is quiet.', sRec), 'no Context');
});
test('round 2: a negated claim holds when its positive does not; spelled-out numbers are numbers', () => {
  assert.ok(!ok('The rock is not medium-sized.', sRec) && ok('The rock is not large.', sRec));
  assert.ok(ok('There is not a comet in the frame.', sRec) && !ok('There is not a rock in the frame.', sRec));
  assert.ok(!ok("The rock is not at 2 o'clock in the image.", sRec) && ok("The rock is not at 5 o'clock in the image.", sRec));
  assert.ok(!ok('It does not show up on the right side of the image.', sRec));
  const c = parseClaims('It is about nine hundred and fifty metres away, closing at twenty-one knots, under five seconds out.', gaz);
  assert.deepEqual(c.map((x) => [x.type, x.value ?? [x.lo, x.hi], x.unit]), [['number', 950, 'm'], ['number', 21, 'kt'], ['range', [0, 5], 's']]);
  assert.ok(ok('The nearest hazard is about two hundred and fifty metres away.', rec) && !ok('The nearest hazard is about seven hundred metres away.', rec));
});
test('round 2: roundNice is symmetric about zero; "about X" holds within half a human step of X', () => {
  assert.deepEqual(roundNice(-19.3), { value: -20, lo: -22.5, hi: -17.5 }); assert.equal(roundNice(-3.5).value, -4);
  for (const v of [0.66, 3.5, 6.6, 12.4, 142.3, 233.7]) assert.equal(roundNice(-v).value, -roundNice(v).value, String(v));
  assert.ok(aboutHolds(6.6, 7) && aboutHolds(0.66, 0.6) && aboutHolds(142.3, 140) && aboutHolds(12.4, 12));
  assert.ok(!aboutHolds(6.6, 5) && !aboutHolds(233.7, 300) && !aboutHolds(21.5, 30));
});

// ---- fix round 3 ----
const lRec = { family: 'L', facts: { 'cfg.gear': F('down', null, 'visual'), papi_whites_cam: F(1, 'count', 'visual'),
  'windsock.from_deg': F(null, 'deg', 'visual') }, safety: null };
test('round 3: common nouns open sentences; kind words used as modifiers are not hazard claims', () => {
  for (const o of ['Runway', 'Birds', 'Station', 'Ocean', 'Tarmac', 'Glaciers', 'Thrusters']) {
    assert.ok(ok(`${o} dominates the view as the gear is down.`, lRec), o);
  }
  assert.ok(!ok('Zorblax dominates the view as the gear is down.', lRec), 'an unknown name');
  assert.ok(!ok('Runway town lies ahead.', lRec), 'a common word in a name frame');
  const a = { family: 'A', facts: { world: F('pillars', null, 'visual'), kinds_in_frame: F([], null, 'visual') }, safety: null };
  assert.ok(ok('The aircraft weaves between the rock pillars.', a) && ok('A rock face rises ahead of the aircraft.', a));
  assert.ok(!ok('There is a rock ahead.', a) && ok("A bird's-eye view shows the Matterhorn in the upper left.", zrec));
});
test('round 3: negation binds only to a predicate of its clause; idioms are not negations; two negations cancel', () => {
  assert.ok(ok('The ship is not far from the rock.', sRec) && ok("It is not hard to spot the rock at 2 o'clock in the image.", sRec));
  assert.ok(!ok('The ship is not far from the comet.', sRec), 'no comet in the frame');
  assert.ok(!ok('The frame shows not only a comet but also a rock.', sRec), 'no comet in the frame');
  assert.ok(ok('There is no sign of a comet in the frame.', sRec) && ok('Nowhere in the frame is there a comet.', sRec));
  assert.ok(ok('It is not the case that the rock is large.', sRec) && !ok('It is not the case that the rock is medium-sized.', sRec));
  assert.ok(!ok('The rock is not not large.', sRec) && !ok('It is not true that the rock is not large.', sRec));
});
test('round 3: gear state in its common forms; presence of bodies and objects', () => {
  assert.ok(ok('With the gear down, the aircraft approaches the runway.', lRec) && ok('The landing gear is extended.', lRec));
  assert.ok(!ok('The aircraft has its gear up.', lRec) && !ok('The undercarriage is retracted.', lRec));
  assert.ok(!ok('The gear is not down.', lRec) && ok('The PAPI shows one white and three red lights.', lRec));
  assert.ok(!ok('The windsock is visible.', lRec) && !ok('The station is visible in the frame.', lRec));
  const s = { ...sRec, facts: { ...sRec.facts, earth_in_frame: F(true, null, 'visual') } };
  assert.ok(ok('The Moon is not in view.', s) && !ok('The Moon is visible.', s));
  assert.ok(ok('Also in view is the Earth.', s) && !ok('The Earth is not in view.', s));
});
test('round 3: bins, storm clouds, grouped numbers and names written without diacritics', () => {
  const d = { family: 'D', facts: { station_distance_bin: F('2-20 m', null, 'visual'), phase: F('FINAL', null, 'visual') }, safety: null };
  assert.ok(ok('The station is visible and less than 20 m away.', d) && !ok('The station is less than 2 m away.', d));
  const a = (preset) => ({ family: 'A', facts: { 'weather.preset': F(preset, null, 'visual') }, safety: null });
  assert.ok(ok('Storm clouds loom ahead.', a('storm')) && !ok('Storm clouds loom ahead.', a('clear')));
  for (const t of ['about 1 250 m/s', "about 1'250 m/s", 'about 1 250 m/s', 'about 1,250 m/s']) {
    assert.equal(parseClaims(t, gaz)[0].value, 1250, t);
  }
  const g = makeGazetteer(['Zürich', 'Zermatt']);
  assert.equal(g.canonical('Zurich'), 'Zürich'); assert.equal(g.has('Zérmatt'), false);
});
