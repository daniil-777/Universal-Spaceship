import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { checkBank, loadBank, render, pickForm, heldOut, slotsOf, formId } from '../vlm/gen/text/paraphrase.js';
import { QFAMILIES, balanceAnswers, exportable } from '../vlm/gen/text/vqa.js';
import { GROUND_ANSWERS } from '../vlm/gen/text/ground_items.js';
import { renderContext, parseContext, corruptMonitor, monitorOf } from '../vlm/gen/text/context.js';
import { runTeacher, estimateCost, TEACHER_PROMPT } from '../vlm/gen/text/teacher.js';
import { CATS, COMMON_LOWER } from '../vlm/gen/text/verify_words.js';
import { mulberry32 } from '../src/mathx.js';

const bank = loadBank(fileURLToPath(new URL('../vlm/gen/text/bank/', import.meta.url)));
test('the bank protects slots, has 10-20 forms per template and at least 45 question families', () => {
  assert.deepEqual(checkBank(bank), []);
  assert.ok(QFAMILIES.length >= 45); assert.equal(new Set(QFAMILIES.map((q) => q.id)).size, QFAMILIES.length);
  assert.deepEqual(checkBank({ t: { kind: 'vqa_a', families: ['S'], slots: ['kind', 'dist'], forms: Array(10).fill('A {kind} ahead.') } }).length > 0, true);
});
test('rendering never leaves undefined, NaN or empty slots; 20 % of paraphrase ids are held out for test', () => {
  for (const [id, t] of Object.entries(bank)) for (const f of t.forms) { const out = render(f, Object.fromEntries(slotsOf(f).map((k) => [k, 'x']))); assert.ok(!/undefined|NaN|\{|\}/.test(out), `${id}: ${out}`); }
  assert.throws(() => render('A {kind}.', { kind: undefined })); assert.throws(() => render('A {kind}.', { kind: '' }));
  const ids = Object.keys(bank).flatMap((id) => bank[id].forms.map((_, i) => heldOut(id, i))), share = ids.filter(Boolean).length / ids.length;
  assert.ok(share > 0.12 && share < 0.28, `held-out share ${share}`);
  const tid = Object.keys(bank).find((id) => bank[id].forms.some((_, i) => heldOut(id, i))), rng = mulberry32(3);
  for (let i = 0; i < 50; i++) { assert.ok(heldOut(tid, pickForm(bank, tid, rng, 'test').paraphrase_id)); assert.ok(!heldOut(tid, pickForm(bank, tid, rng, 'train').paraphrase_id)); }
});
test('VQA answer balance: no answer above the family cap after rejection sampling', () => {
  const items = Array.from({ length: 100 }, (_, i) => ({ family_q: 'presence_kind', answerKey: i < 80 ? 'no' : 'yes' }));
  const kept = balanceAnswers(items, { maxShare: 0.5 }), no = kept.filter((x) => x.answerKey === 'no').length;
  assert.ok(no / kept.length <= 0.5 + 1e-9); assert.equal(kept.filter((x) => x.answerKey === 'yes').length, 20);
});
test('Context line renders and parses back; gt+noise copies a same-family tuple', () => {
  const m = monitorOf({ verdict: 'UNSAFE', severity: 3, reasons: ['HAZARD_AHEAD'], best_action: 'CLIMB', p_ref: 0.75, ttc_s: 2.1 }, { 'clearance.min_u': { v: 3.1 } }, 'S');
  const line = renderContext({ telemetry: [['speed', '270 m/s', 'ship.speed_m_s']], monitor: m });
  assert.match(line, /^Context: /); assert.deepEqual(parseContext(line).monitor, m);
  const pool = [{ verdict: 'SAFE', severity: 0, reasons: [], action: 'CONTINUE', p_ref: 0, ttc_bin: 'none', clr_bin: '>40 u' }];
  const c = corruptMonitor(m, { rng: () => 0.01, confusion: null, pool }); assert.equal(c.verdict, 'SAFE'); assert.equal(c.action, 'CONTINUE');
});
test('teacher gating: no call without --yes-spend or ANTHROPIC_API_KEY, the cost printed first, the key in no output', async () => {
  const calls = [], logs = [], send = async (req) => { calls.push(req); return { text: 'A rock ahead.' }; };
  const recs = [{ key: 'k', family: 'S', facts: {}, narrator_frame: 'x.png' }];
  await runTeacher(recs, { send, model: 'm', priceIn: 3, priceOut: 15, yesSpend: false, env: { ANTHROPIC_API_KEY: 'sk-secret' }, log: (s) => logs.push(s) }); assert.equal(calls.length, 0);
  await runTeacher(recs, { send, model: 'm', priceIn: 3, priceOut: 15, yesSpend: true, env: {}, log: (s) => logs.push(s) }); assert.equal(calls.length, 0);
  const out = await runTeacher(recs, { send, model: 'm', priceIn: 3, priceOut: 15, yesSpend: true, env: { ANTHROPIC_API_KEY: 'sk-secret' }, log: (s) => logs.push(s) });
  assert.equal(calls.length, 1); assert.match(logs[0], /estimated cost/); assert.ok(!JSON.stringify([out, logs]).includes('sk-secret'));
  assert.ok(estimateCost({ n: 1000, tokensIn: 1500, tokensOut: 200, priceIn: 3, priceOut: 15 }) > 0);
});

// ---- integration: synthetic records with the label modules' exact fact ids, one or two per family ----
import fs from 'node:fs';
import { spawnSync } from 'node:child_process';
import { recordTexts, visualSlots } from '../vlm/gen/text/items.js';
import { makeGazetteer, BASE_NAMES, SENTENCE_WORDS, ROUTE_NAMES, verifyTemplateItem, verifyFreeText } from '../vlm/gen/text/verify.js';
import { telemetryOf, contextSupplies, rowContext, REASON_TEXT, ACTION_TEXT } from '../vlm/gen/text/context.js';
import { REASONS, ACTIONS, OUTCOMES, CAUSES, HAZARD_KINDS, ZOOM_TAGS, TAG_WORDS, PALETTE_NAMES, SAFETY_TEXT_IDS, validateTextFacts } from '../vlm/gen/schema.js';
import { BRIGHTNESS_BINS, EDGE_BINS, COAST_SIDES } from '../vlm/gen/schema.js';
import { W, KIND_A, CAUSE_REASONS } from '../vlm/gen/text/vqa.js';
import { parseClaims, prefOutcome, outcomeAgrees } from '../vlm/gen/text/verify.js';
const V = (v, unit = null) => ({ v, unit, obs: 'visual' }), C = (v, unit = null) => ({ v, unit, obs: 'context' });
const IMG = { 'image.palette_0': V('blue'), 'image.palette_1': V('white'), 'image.palette_2': V('grey'), 'image.brightness_bin': V('medium'), 'image.edge_bin': V('textured') };
const SEV = { SAFE: 0, CAUTION: 1, UNSAFE: 3 };
const saf = (verdict, reasons, best, outcome, cause = null) => ({ verdict, severity: SEV[verdict], reasons, cause, action_outcome: { CONTINUE: outcome }, p_ref: verdict === 'UNSAFE' ? 1 : 0, safe_actions: best === 'NONE_SAFE' ? [] : [best], best_action: best, ttc_s: 2.1 });
const hz = (i, kind, inFrame, o = {}) => ({ [`hazard.${i}.kind`]: (inFrame ? V : C)(kind), [`hazard.${i}.in_frame`]: V(inFrame), [`hazard.${i}.box_px`]: V(inFrame ? [600, 180, 660, 240] : null, 'px'), [`hazard.${i}.side`]: V(inFrame ? 'right' : null),
  [`hazard.${i}.clock`]: V(inFrame ? 2 : null, 'clock'), [`hazard.${i}.bearing_clock`]: C(1, 'clock'), [`hazard.${i}.size_bin`]: V(inFrame ? 'medium' : null), [`hazard.${i}.dist_u`]: C(o.d ?? 12.3, 'u'),
  [`hazard.${i}.closing_u_s`]: C(o.c ?? 21.5, 'u/s'), [`hazard.${i}.ttc_s`]: C(o.t === undefined ? 0.57 : o.t, 's'), [`hazard.${i}.cpa_u`]: C(1.2, 'u') });
const ship = (p = 8.4) => ({ 'ship.speed_u_s': C(14.2, 'u/s'), 'ship.speed_m_s': C(269.8, 'm/s'), 'ship.pitch_deg': C(p, 'deg'), 'clearance.min_u': C(3.1, 'u'), 'edges.ceiling_u': C(9.5, 'u') });
const REC = [
  { key: 'S_t_1', family: 'S', facts: { ...ship(), ...hz(0, 'rock', true), ...hz(1, 'comet', false, { c: -3, t: null }), 'hazards.count_in_frame': V(3, 'count'), 'kinds_in_frame': V(['comet', 'rock']), 'clearance.ground_u': C(null, 'u'),
    'sun.lit': V(1), 'earth_in_frame': V(true), 'moon_in_frame': V(false), 'weather.preset': V('space'), 'world': V('space'), 'in_tunnel': V(false), ...IMG }, safety: saf('UNSAFE', ['HAZARD_AHEAD', 'HAZARD_CLOSING_FAST'], 'CLIMB', 'crash_possible', 'rock') },
  { key: 'S_t_2', family: 'S', facts: { ...ship(-1), ...hz(0, 'satellite', false, { c: -2, t: null }), 'hazards.count_in_frame': V(0, 'count'), 'kinds_in_frame': V([]), 'sun.lit': V(0), 'earth_in_frame': V(false), 'moon_in_frame': V(true), ...IMG },
    safety: saf('SAFE', [], 'CONTINUE', 'clear') },
  { key: 'A_t_1', family: 'A', facts: { ...ship(-6), ...hz(0, 'birds', true, { d: 30 }), 'hazards.count_in_frame': V(2, 'count'), 'kinds_in_frame': V(['birds']), 'clearance.ground_u': C(7.2, 'u'), 'air.stall_margin_deg': C(6.6, 'deg'), 'air.n_g': C(1.22, 'g'),
    'air.turbulence': C('SEVERE'), 'air.speed_band': C('normal'), 'air.wind_u_s': C([-2.3, 0, 0.4], 'u/s'), 'air.in_cloud': V(0.8), 'weather.cells': C([{ type: 1, dist_u: 22.5, bearing_deg: 40 }, { type: 0, dist_u: 5, bearing_deg: 0 }]),
    'weather.preset': V('storm'), 'world': V('mountains'), 'route': V('alps'), 'in_tunnel': V(false), ...IMG }, safety: saf('CAUTION', ['SEVERE_TURBULENCE', 'STORM_CELL'], 'DESCEND', 'crash_possible', 'terrain') },
  { key: 'A_t_2', family: 'A', facts: { ...ship(0.5), 'hazards.count_in_frame': V(0, 'count'), 'kinds_in_frame': V([]), 'air.turbulence': C('LIGHT'), 'air.n_g': C(1.0, 'g'), 'air.in_cloud': V(0), 'weather.cells': C([]),
    'weather.preset': V('clear'), 'world': V('newyork'), 'route': V('newyork'), 'in_tunnel': V(true), ...IMG }, safety: saf('SAFE', [], 'CONTINUE', 'clear') },
  { key: 'L_t_1', family: 'L', facts: { vert_mode: C('GS'), ias_kt: C(142.3, 'kt'), 'ils.loc_dots': C(0.3, 'dots'), 'ils.gs_dots': C(-1.6, 'dots'), papi_whites_cam: V(1, 'count'), 'windsock.from_deg': V(250, 'deg'), 'cfg.gear': V('down'),
    'cfg.spoilers': V(0), 'wind.head_kt': C(12.4, 'kt'), 'wind.cross_kt': C(-6.1, 'kt'), gates: C({ lateral: true, vertical: true, loc: true, gs: false, speed: true, vs: true, gear: true }), wow: C(false),
    'scene.time': V('day'), 'scene.vis': V('cavok'), 'scene.clouds': V('SCT030'), 'scene.rain': V(true), 'scene.in_cloud': V(false), ...IMG }, safety: saf('CAUTION', ['GLIDESLOPE_DEVIATION'], 'CLIMB', 'landed') },
  { key: 'L_t_2', family: 'L', facts: { vert_mode: C('ROLLOUT'), ias_kt: C(60, 'kt'), 'ils.loc_dots': C(null, 'dots'), 'ils.gs_dots': C(null, 'dots'), papi_whites_cam: V(null, 'count'), 'windsock.from_deg': V(null, 'deg'), 'cfg.gear': V('down'),
    'cfg.spoilers': V(0.9), 'wind.head_kt': C(-11, 'kt'), 'wind.cross_kt': C(3, 'kt'), wow: C(true), 'scene.time': V('night'), 'scene.vis': V('fog'), 'scene.clouds': V('OVC003'), 'scene.in_cloud': V(false), ...IMG }, safety: saf('UNSAFE', ['CANNOT_STOP'], 'NONE_SAFE', 'overrun', 'runway') },
  { key: 'D_t_1', family: 'D', facts: { phase: V('FINAL'), station_distance_bin: V('2-20 m'), rho_m: C(8.2, 'm'), closing_cms: C(12.1, 'cm/s'), corridor_limit_cms: C(8, 'cm/s'), speed_limit_cms: C(30, 'cm/s'), in_cone: C(true),
    att_err_deg: C(0.6, 'deg'), jets_failed: C(['P6']), fuel_frac: C(0.62), breakout_available: C(true), 'sun.lit': V(1), earth_in_frame: V(false), ...IMG }, safety: saf('CAUTION', ['CLOSING_TOO_FAST'], 'SLOW_DOWN', 'capture') },
  { key: 'D_t_2', family: 'D', facts: { phase: V('H1'), station_distance_bin: V('100-400 m'), rho_m: C(212, 'm'), closing_cms: C(-0.5, 'cm/s'), corridor_limit_cms: C(null, 'cm/s'), speed_limit_cms: C(74.6, 'cm/s'), in_cone: C(false),
    att_err_deg: C(3.4, 'deg'), jets_failed: C([]), fuel_frac: C(0.35), breakout_available: C(false), 'sun.lit': V(0), earth_in_frame: V(true), ...IMG }, safety: saf('UNSAFE', ['KOS_VIOLATION', 'NO_BREAKOUT_AVAILABLE'], 'NONE_SAFE', 'fail', 'station') },
  { key: 'Z_t_1', family: 'Z', facts: { 'view.range_km': V(50, 'km'), 'view.gsd_m': V(38.2, 'm'), 'sun.class': V('day'), 'place.country': V('Switzerland'), 'place.admin1': V('Valais'),
    'place.nearest': V({ name: 'Zermatt', km: 4.2, bearing: 40, compass: 'north-east' }), 'place.in_view': V([{ name: 'Matterhorn', kind: 'peak', region: 'upper left' }, { name: 'Split', kind: 'place', region: 'centre' }]),
    'geo.sea_frac': V(0.3), 'geo.coast_side': V('left'), ...IMG }, zoom: { tags: ['MOUNTAINS', 'COASTLINE', 'ICE'], range_bin: 1 }, safety: null },
  { key: 'Z_t_2', family: 'Z', facts: { 'view.range_km': V(1200, 'km'), 'view.gsd_m': V(1250, 'm'), 'sun.class': V('night'), 'place.country': V(null), 'place.nearest': V({ name: 'Apia', km: 2480, bearing: 275, compass: 'west' }),
    'place.in_view': V([]), 'geo.sea_frac': V(1), 'geo.coast_side': V(null), ...IMG }, zoom: { tags: ['WATER_DOMINANT', 'NIGHT'], range_bin: 3 }, safety: null },
  { key: 'L_t_3', family: 'L', facts: { vert_mode: C('GS'), ias_kt: C(150, 'kt'), 'ils.loc_dots': C(-1.8, 'dots'), 'ils.gs_dots': C(0.4, 'dots'), papi_whites_cam: V(2, 'count'), 'cfg.gear': V('up'), 'cfg.spoilers': V(0),
    'wind.head_kt': C(5, 'kt'), 'wind.cross_kt': C(9, 'kt'), gates: C({ lateral: true, vertical: true, loc: true, gs: true, speed: true, vs: true, gear: true }), wow: C(false), 'scene.time': V('dusk'), 'scene.vis': V('haze'), 'scene.in_cloud': V(false), ...IMG },
    safety: saf('UNSAFE', ['LOCALIZER_DEVIATION', 'UNSTABLE_APPROACH'], 'GO_AROUND', 'go_around') },
  { key: 'D_t_3', family: 'D', facts: { phase: V('CORRIDOR'), station_distance_bin: V('20-100 m'), rho_m: C(55, 'm'), closing_cms: C(3.2, 'cm/s'), corridor_limit_cms: C(8, 'cm/s'), speed_limit_cms: C(20, 'cm/s'), in_cone: C(false),
    att_err_deg: C(1.1, 'deg'), jets_failed: C([]), fuel_frac: C(0.9), breakout_available: C(true), ...IMG }, safety: saf('CAUTION', ['LATERAL_MISALIGNMENT'], 'HOLD_POSITION', 'breakout') },
  { key: 'Z_t_3', family: 'Z', facts: { 'view.range_km': V(250, 'km'), 'view.gsd_m': V(300, 'm'), 'sun.class': V('twilight'), 'place.country': V('Switzerland'), 'place.nearest': V({ name: 'Zermatt', km: 60, bearing: 190, compass: 'south' }),
    'place.in_view': V([{ name: 'Valais', kind: 'region', region: 'lower right' }]), 'geo.sea_frac': V(0), 'geo.coast_side': V(null), ...IMG }, zoom: { tags: ['HILLS', 'URBAN'], range_bin: 2 }, safety: null },
  // T10-v: an in-cloud L record — the fog/cloud visibility gate nulls papi_whites_cam and windsock.from_deg (far from
  // the chase camera), while cfg.gear stays visible (close to it); scene.in_cloud true picks the cap_l_cloud lead
  { key: 'L_t_4', family: 'L', facts: { vert_mode: C('GS'), ias_kt: C(138, 'kt'), 'ils.loc_dots': C(0.2, 'dots'), 'ils.gs_dots': C(-0.4, 'dots'), papi_whites_cam: V(null, 'count'), 'windsock.from_deg': V(null, 'deg'), 'cfg.gear': V('down'),
    'cfg.spoilers': V(0), 'wind.head_kt': C(8, 'kt'), 'wind.cross_kt': C(-4, 'kt'), gates: C({ lateral: true, vertical: true, loc: true, gs: true, speed: true, vs: true, gear: true }), wow: C(false),
    'scene.time': V('night'), 'scene.vis': V('fog'), 'scene.clouds': V('OVC003'), 'scene.rain': V(true), 'scene.in_cloud': V(true), ...IMG }, safety: saf('CAUTION', ['GLIDESLOPE_DEVIATION'], 'CLIMB', 'landed') }];
const GAZ = makeGazetteer([...BASE_NAMES, 'Switzerland', 'Valais', 'Zermatt', 'Matterhorn', 'Apia', 'Tokyo', 'Paris', 'Split', 'Point', 'Orange', 'Wind', 'Mobile']);
const SA_ = (r) => r.family === 'S' || r.family === 'A';
const monOf = (r, verdict) => ({ verdict: verdict || r.safety.verdict, severity: SEV[verdict || r.safety.verdict], reasons: verdict ? ['HAZARD_AHEAD'] : [...r.safety.reasons], action: verdict ? 'CONTINUE' : r.safety.best_action, p_ref: SA_(r) ? r.safety.p_ref : null, ttc_bin: 'none', clr_bin: 'none' });
const ctxOf = (r, verdict) => ({ telemetry: telemetryOf(r.facts, r.family), monitor: r.safety ? monOf(r, verdict) : null });

test('every form of every template a family or caption can use verifies on the synthetic records (parser recovers exactly the slots)', () => {
  const fails = []; let n = 0;
  for (const r of REC) {
    const ctx = ctxOf(r);
    for (const q of QFAMILIES) {
      if (!q.families.includes(r.family)) continue; const a = q.ask(r, mulberry32(5), ctx); if (!a) continue;
      const at = Object.fromEntries(Object.entries(a.a).map(([k, s]) => [k, s.text])), qt = Object.fromEntries(Object.entries(a.q).map(([k, s]) => [k, typeof s === 'string' ? s : s.text]));
      bank[a.aid].forms.forEach((af, i) => { n++; const it = { prompt: render(bank[a.qid].forms[i % bank[a.qid].forms.length], qt), answer: render(af, at), slots: Object.values(a.a), checks: a.checks };
        const c = verifyTemplateItem(it, r, { gaz: GAZ, context: ctx }); if (!c.verified) fails.push(`${r.key} ${q.id}: ${it.answer} ${c.errors}`); });
    }
    const S = visualSlots(r);
    for (const [id, t] of Object.entries(bank)) if (t.kind.startsWith('caption') && t.families.includes(r.family) && t.slots.every((x) => S[x])) for (const f of t.forms) {
      n++; const answer = render(f, Object.fromEntries(Object.entries(S).map(([k, s]) => [k, s.text]))), c = verifyTemplateItem({ answer, slots: t.slots.map((x) => S[x]) }, r, { gaz: GAZ }); if (!c.verified) fails.push(`${r.key} ${id}: ${answer} ${c.errors}`);
    }
  }
  assert.ok(n > 1500, `${n} forms`); assert.deepEqual(fails.slice(0, 5), []);
});
test('recordTexts: nothing rejected, captions within bounds, one negative, VQA and negatives near 9:1, test split uses held-out forms only', () => {
  let vqa = 0, neg = 0;
  for (const r of REC) for (const split of ['train', 'test']) for (let s = 1; s <= 8; s++) {
    const ctx = ctxOf(r), out = recordTexts(r, { bank, gaz: GAZ, rng: mulberry32(s), split, context: ctx });
    assert.deepEqual(out.rejected.map((x) => `${x.item.template_id}: ${x.errors}`), [], r.key); assert.deepEqual(out.skipped.filter((x) => x.task !== 'safety'), [], r.key);
    const words = (t) => out.texts.find((x) => x.task === t).answer.split(/\s+/).length;
    assert.ok(words('caption_short') <= 20 && words('caption_detail') >= 40 && words('caption_detail') <= 90, r.key);
    assert.equal(out.texts.filter((t) => t.task === 'negative').length, 1); vqa += out.texts.filter((t) => t.task === 'vqa').length; neg += out.texts.filter((t) => t.task === 'negative').length;
    for (const t of out.texts) {
      assert.ok(t.verified && Array.isArray(t.fact_ids) && t.fact_ids.length && t.generator === 'template' && 'false_premise' in t, `${r.key} ${t.template_id}`);
      if (split === 'test') String(t.paraphrase_id).split('+').forEach((p, i) => { const tid = t.template_id.split('+')[i]; if (bank[tid] && bank[tid].forms.some((_, j) => heldOut(tid, j))) assert.ok(heldOut(tid, p), `${tid}#${p}`); });
    }
  }
  assert.ok(neg / (vqa + neg) > 0.08 && neg / (vqa + neg) < 0.15, `negative share ${neg / (vqa + neg)}`);
});
test('§5.7: an item without needsContext states only visual facts; a needsContext item rests on what its Context supplies; the monitor is quoted, contradictions are said', () => {
  for (const r of REC) {
    const ctx = ctxOf(r), out = recordTexts(r, { bank, gaz: GAZ, rng: mulberry32(9), split: 'train', context: ctx });
    for (const t of out.texts) {
      if (!t.needsContext) assert.ok(verifyFreeText(t.answer, r, { gaz: GAZ, obsRule: true }).verified, `${r.key} ${t.template_id}: ${t.answer}`);
      else if (contextSupplies(t.context_facts, ctx)) assert.ok(verifyFreeText(t.answer, r, { gaz: GAZ, obsRule: true, context: ctx }).verified, `${r.key} ${t.answer}`);
      if (/\b(UNSAFE|SAFE|CAUTION)\b/.test(t.answer)) { assert.ok(t.needsContext && /monitor/i.test(t.answer), t.answer); assert.match(t.answer, new RegExp(ctx.monitor.verdict)); }
      if (t.task.startsWith('caption')) assert.equal(t.needsContext, false);
    }
  }
  const s2 = REC[1], ctx = ctxOf(s2, 'UNSAFE'), out = recordTexts(s2, { bank, gaz: GAZ, rng: mulberry32(2), split: 'train', context: ctx }), chain = out.texts.find((t) => t.task === 'safety');
  assert.match(chain.answer, /UNSAFE/); assert.match(chain.answer, /no hazards/i);
  const noCtx = recordTexts(REC[0], { bank, gaz: GAZ, rng: mulberry32(2), split: 'train', context: null });
  assert.ok(!noCtx.texts.some((t) => t.task === 'safety' || ['is_safe', 'why', 'what_to_do', 'if_nothing_changes', 'most_dangerous'].includes(t.family_q)));
});
test('sentence-initial words of the bank and of slot words never read as places; route names match the page', async () => {
  const missing = new Set();
  for (const t of Object.values(bank)) for (const f of t.forms) for (const s of f.split(/(?<=[.!?:;])\s+/)) { const w = s.trim().split(/\s+/)[0]; if (w && !w.startsWith('{')) { const c = w.replace(/[^\p{L}'-]/gu, '').toLowerCase(); if (!SENTENCE_WORDS.has(c)) missing.add(c); } }
  for (const m of Object.values(W)) for (const p of Array.isArray(m) ? m : Object.values(m)) { const c = String(p).split(/\s+/)[0].toLowerCase(); if (!SENTENCE_WORDS.has(c)) missing.add(c); }
  for (const p of Object.values(KIND_A)) if (!SENTENCE_WORDS.has(p.split(' ')[0])) missing.add(p);
  assert.deepEqual([...missing], []);
  const { ROUTES } = await import('../vlm/gen/labels/worlds.js'); assert.deepEqual(Object.fromEntries(Object.entries(ROUTES).map(([k, r]) => [k, r.name])), ROUTE_NAMES);
});
test('the fact ids the text layer reads are the ones the label modules write', () => {
  const src = ['corridor_facts', 'landing', 'docking', 'zoom'].map((m) => fs.readFileSync(new URL(`../vlm/gen/labels/${m}.js`, import.meta.url), 'utf8')).join('\n');
  const ids = new Set(REC.filter((r) => r.family !== 'Z').flatMap((r) => Object.keys(r.facts)).filter((id) => !id.startsWith('image.')));
  for (const id of ['view.range_km', 'view.gsd_m', 'sun.class']) ids.add(id);
  const loop = (id) => { const m = /^scene\.(\w+)$/.exec(id); return !!m && src.includes('put(`scene.${k}`') && src.includes(`'${m[1]}'`); };
  const missing = [...ids].filter((id) => !src.includes(`'${id}'`) && !src.includes(`\`${id.replace(/^hazard\.\d\./, 'hazard.${i}.')}\``) && !loop(id));
  assert.deepEqual(missing, []);
});
test('verify_cli: stdin {text, record, context, names} gives {verified, errors}', () => {
  const run = (text) => JSON.parse(spawnSync(process.execPath, [fileURLToPath(new URL('../vlm/gen/text/verify_cli.mjs', import.meta.url))], { input: JSON.stringify({ text, record: REC[8], names: ['Zermatt', 'Tokyo'] }), encoding: 'utf8' }).stdout);
  assert.deepEqual(run('Near Zermatt, about 50 km away.'), { verified: true, errors: [] }); assert.equal(run('This is Tokyo.').verified, false);
});
test('bank rules: Z detail leads carry {range} and {terrain}; monitor answers name the monitor; caption, safety and family minimums', () => {
  const T = Object.entries(bank);
  for (const [id, t] of T) if (t.kind === 'caption_detail' && t.families.includes('Z')) assert.ok(t.slots.includes('range') && t.slots.includes('terrain'), id);
  for (const [id, t] of T) if (t.slots.includes('clock')) for (const f of t.forms) assert.match(f, /image|frame|picture|screen|view/i, `${id}: the image clock is screen-relative: ${f}`);
  for (const [id, t] of T) if (/^(is_safe|why|what_to_do)(_\w+)?_a$/.test(id) || t.kind === 'safety') for (const f of t.forms) assert.match(f, /monitor/i, `${id}: ${f}`);
  const CUE = /^(?:it|its|so|best action|advice|recommended|reason|the recommended action|the pilot should|the crew should|the spacecraft should)\b/i;
  for (const [id, t] of T) for (const f of t.forms) { let seen = false; for (const x of f.split(/(?<=[.!?;])\s+/)) { const mon = /monitor/i.test(x); seen ||= mon; if (/\{(verdict|reason|action)\}|\{outcome\}/.test(x) && t.kind !== 'vqa_q' && !(mon || (seen && CUE.test(x))) && !(id === 'what_if_a')) assert.fail(`${id}: "${x}" states a monitor field without attributing it`); } }
  for (const fam of ['S', 'A', 'L', 'D', 'Z']) {
    assert.ok(T.filter(([, t]) => t.families.includes(fam) && t.kind.startsWith('caption_') && t.kind !== 'caption_part').length >= 2, fam);
    if (fam !== 'Z') assert.ok(T.some(([, t]) => t.kind === 'safety' && t.families.includes(fam)), fam);
  }
  const fired = new Set(), aids = new Set();
  for (const r of REC) for (let k = 1; k <= 8; k++) for (const q of QFAMILIES) if (q.families.includes(r.family)) { const a = q.ask(r, mulberry32(k), { ...ctxOf(r), ...(r.safety ? { monitor: { ...monOf(r), ttc_bin: '1-3 s', clr_bin: '5-15 u' } } : {}) }); if (a) { fired.add(q.id); aids.add(a.aid); } }
  assert.deepEqual(QFAMILIES.map((q) => q.id).filter((id) => !fired.has(id)), [], 'every family fires on some record');
  assert.deepEqual(T.filter(([id, t]) => t.kind === 'vqa_a' && !aids.has(id) && !GROUND_ANSWERS.includes(id)).map(([id]) => id), [], 'every answer template is produced (grounding: tests/vlm_ground.test.mjs)');
});
// the L/D outcomes a text may state next to each monitor verdict, written out (not outcomeAgrees): a benign outcome never
// stands next to UNSAFE
const BENIGN = ['landed', 'capture', 'clear'];
const AGREES = { SAFE: BENIGN, CAUTION: [...BENIGN, 'go_around', 'breakout', 'crash_possible'],
  UNSAFE: ['crash_possible', 'crash_certain', 'go_around', 'hard', 'excursion', 'overrun', 'short', 'tailstrike', 'crash', 'breakout', 'fail'] };
test('outcomeAgrees matches the written-out table for every outcome and verdict', () => {
  for (const v of Object.keys(AGREES)) for (const o of Object.keys(W.outcome)) assert.equal(outcomeAgrees(o, v), AGREES[v].includes(o), `${o} / ${v}`);
});
test("§5.7 under a corrupted monitor: the prediction is the monitor's p_ref (S/A) or agrees with its verdict (L/D); a cause is one of its reasons", () => {
  let n = 0;
  for (const r of REC.filter((x) => x.safety)) for (const verdict of ['SAFE', 'CAUTION', 'UNSAFE']) for (const p_ref of SA_(r) ? [0, 0.5, 1] : [null]) for (const reasons of [[], ['HAZARD_AHEAD'], ['TERRAIN_CLOSE', 'CANNOT_STOP', 'KOS_VIOLATION']]) {
    const m = { verdict, severity: SEV[verdict], reasons, action: verdict === 'SAFE' ? 'CONTINUE' : 'NONE_SAFE', p_ref, ttc_bin: 'none', clr_bin: 'none' }, ctx = { telemetry: telemetryOf(r.facts, r.family), monitor: m };
    for (let s = 1; s <= 3; s++) {
      const out = recordTexts(r, { bank, gaz: GAZ, rng: mulberry32(s), split: 'train', context: ctx });
      assert.deepEqual(out.rejected.map((x) => `${x.item.template_id}: ${x.errors}`), [], `${r.key} ${verdict}`);
      for (const t of out.texts) {
        for (const c of parseClaims(t.answer, GAZ).filter((x) => x.type === 'outcome')) {
          n++;
          if (SA_(r)) { assert.equal(c.value, prefOutcome(p_ref), `${r.key} ${t.answer}`); assert.match(t.answer.split(/(?<=[.!?])\s+/).find((x) => x.includes(W.outcome[c.value])), /monitor/i); }
          else assert.ok(AGREES[verdict].includes(c.value), `${r.key} ${verdict}: ${t.answer}`);
        }
        if (t.family_q === 'most_dangerous') assert.ok(CAUSE_REASONS[r.safety.cause].some((x) => reasons.includes(x)), `${r.key}: ${t.answer}`);
        if (rowContext(t, ctx, () => 0).keep && t.needsContext) assert.ok(verifyFreeText(t.answer, r, { gaz: GAZ, obsRule: true, context: ctx }).verified, `${r.key} ${verdict}: ${t.answer}`);
      }
    }
  }
  assert.ok(n > 50, `${n} outcome claims`);
});
test('the export rule is one function: contextSupplies(ids, context) and rowContext(item, context, rng)', () => {
  const r = REC[4], ctx = ctxOf(r), line = `Context: telemetry: IAS 142.3 kt, GS dots -1.6 dots; monitor: CAUTION (severity 1), reasons: GLIDESLOPE_DEVIATION, action: CLIMB, TTC none, clearance none.`;
  assert.ok(contextSupplies(['ias_kt', 'ils.gs_dots', ...SAFETY_TEXT_IDS], ctx) && contextSupplies(['ias_kt', 'safety.p_ref'], line) && contextSupplies(ctx, 'ias_kt'));
  assert.ok(!contextSupplies(['vert_mode'], ctx) && !contextSupplies(['safety.verdict'], { telemetry: ctx.telemetry, monitor: null }) && !contextSupplies(['safety.unknown'], ctx) && !contextSupplies(['ias_kt'], null));
  assert.deepEqual(rowContext({ needsContext: true, context_facts: ['vert_mode'] }, ctx, () => 0.9), { keep: false, withCtx: true });
  assert.deepEqual(rowContext({ needsContext: false, context_facts: [] }, null, () => 0.9), { keep: true, withCtx: false });
  assert.deepEqual(rowContext({ needsContext: false, context_facts: [] }, ctx, () => 0.1), { keep: true, withCtx: true });
  const ids = new Set(); for (const x of REC) for (let s = 1; s <= 6; s++) for (const t of recordTexts(x, { bank, gaz: GAZ, rng: mulberry32(s), context: ctxOf(x) }).texts) t.fact_ids.filter((f) => f.startsWith('safety.')).forEach((f) => ids.add(f));
  assert.deepEqual([...ids].filter((f) => !SAFETY_TEXT_IDS.includes(f)), []);
});
test('shared text constants: the text tables cover the schema enums, TEXT_FACTS validate, every template has a held-out form', () => {
  const same = (a, b, what) => assert.deepEqual([...a].sort(), [...b].sort(), what);
  same(Object.keys(REASON_TEXT), REASONS, 'REASON_TEXT'); same(Object.keys(ACTION_TEXT), ACTIONS, 'ACTION_TEXT'); same(Object.keys(W.outcome), [...new Set(Object.values(OUTCOMES).flat())], 'W.outcome');
  same(Object.keys(W.cause), CAUSES, 'W.cause'); same(Object.keys(KIND_A), HAZARD_KINDS, 'KIND_A'); same(Object.keys(TAG_WORDS), ZOOM_TAGS, 'TAG_WORDS'); same(Object.keys(CAUSE_REASONS), CAUSES, 'CAUSE_REASONS');
  assert.ok(PALETTE_NAMES.includes('teal') && PALETTE_NAMES.length === 12);
  const z = { ...REC[8], zoom: { tags: ['MOUNTAINS'], range_bin: 1 } };
  assert.deepEqual(validateTextFacts(z), { ok: true, errors: [] }); assert.deepEqual(validateTextFacts(REC[0]), { ok: true, errors: [] });
  const bad = { ...z, facts: { ...z.facts, 'image.brightness_bin': V('dim'), 'geo.coast_side': V('north') } }; delete bad.facts['place.admin1'];
  assert.deepEqual(validateTextFacts(bad).errors.length, 3);
  for (const [id, t] of Object.entries(bank)) assert.ok(t.forms.some((_, i) => heldOut(id, i)), id);
  const mod = { t: { ...bank.sky_a, forms: ['A new first form about {sky} skies.', ...bank.sky_a.forms.slice(1)] } }, f0 = pickForm(mod, 't', () => 0.99, 'train').paraphrase_id;
  assert.equal(pickForm(bank, 'sky_a', () => 0.99, 'train').paraphrase_id, f0, 'a paraphrase id is the hash of its form, not its position');
  const a = balanceAnswers(Array.from({ length: 40 }, (_, i) => ({ family_q: 'q', answerKey: i < 30 ? 'a' : 'b', i })), { maxShare: 0.5, rng: mulberry32(1) }), b = balanceAnswers(Array.from({ length: 40 }, (_, i) => ({ family_q: 'q', answerKey: i < 30 ? 'a' : 'b', i })), { maxShare: 0.5, rng: mulberry32(2) });
  assert.equal(a.length, 20); assert.notDeepEqual(a.map((x) => x.i), b.map((x) => x.i));
});

// ---- fix round 2 ----
test('round 2: the word tables follow the schema enums; scene.clouds is a text fact; the bank vocabulary is whole; the teacher prompt is plain', () => {
  assert.deepEqual(Object.keys(W.bright), [...BRIGHTNESS_BINS]); assert.deepEqual(Object.keys(W.coast), [...COAST_SIDES]);
  assert.deepEqual(CATS.filter(([d]) => d === 'texture').map(([, v]) => v).sort(), [...EDGE_BINS].sort());
  const l = REC.find((r) => r.key === 'L_t_1');
  assert.deepEqual(validateTextFacts(l), { ok: true, errors: [] });
  assert.equal(validateTextFacts({ ...l, facts: { ...l.facts, 'scene.clouds': V('cloudy') } }).ok, false);
  const words = new Set(Object.values(bank).flatMap((t) => t.forms.flatMap((f) => f.replace(/\{[a-z_]+\}/g, ' ').toLowerCase()
    .match(/\p{L}[\p{L}'-]*/gu) || []).map((x) => x.replace(/'s$/, ''))));
  assert.deepEqual([...words].filter((x) => x.length > 1 && !COMMON_LOWER.has(x)), []);
  assert.doesNotMatch(TEACHER_PROMPT, /vivid/i);
});
test('round 2: held-out forms are stable under appends; every template keeps a train and a held form; exportable is contextSupplies', () => {
  for (const [id, t] of Object.entries(bank)) {
    const held = t.forms.filter((f) => parseInt(formId(f), 16) % 5 === 0).length;
    assert.ok(held >= 1 && held < t.forms.length, id);
  }
  const t = bank.sky_a, before = t.forms.map((f) => heldOut('sky_a', formId(f)));
  pickForm({ sky_a: { ...t, forms: [...t.forms, 'Yet another form about {sky} skies.'] } }, 'sky_a', () => 0, 'train');
  assert.deepEqual(t.forms.map((f) => heldOut('sky_a', formId(f))), before, 'appending a form moves no split');
  pickForm(bank, 'sky_a', () => 0, 'train');
  const heldForm = t.forms.find((f) => parseInt(formId(f), 16) % 5 === 0);
  assert.throws(() => pickForm({ solo: { ...t, forms: [heldForm] } }, 'solo', () => 0, 'train'), /no train form/);
  const l = REC.find((r) => r.key === 'L_t_1'), ctx = ctxOf(l);
  for (const ids of [['ias_kt'], ['vert_mode'], ['safety.verdict'], ['cfg.gear', 'safety.cause'], ['cfg.gear']]) {
    const cited = ids.filter((id) => id.startsWith('safety.') || l.facts[id].obs !== 'visual');
    for (const c of [ctx, null, { telemetry: ctx.telemetry, monitor: null }]) assert.equal(exportable(ids, l, c), contextSupplies(cited, c), `${ids} ${!!c}`);
  }
});
test('round 2: most_dangerous names a hazard only when it is the one kind in the frame', () => {
  const md = QFAMILIES.find((q) => q.id === 'most_dangerous'), s1 = REC.find((r) => r.key === 'S_t_1');
  assert.equal(md.ask(s1, mulberry32(1), ctxOf(s1)), null, 'comet and rock in view');
  const one = { ...s1, facts: { ...s1.facts, kinds_in_frame: V(['rock']) } };
  assert.ok(md.ask(one, mulberry32(1), ctxOf(one)));
});

test('round 3: bank wording — jet counts agree, the nearest-hazard answers say "in view"', () => {
  assert.ok(!bank.jets_some_a.forms.some((f) => /which are/.test(f)));
  assert.ok(bank.nearest_kind_a.forms.includes('The closest hazard in view is {a_kind}.'));
  assert.ok(bank.nearest_side_a.forms.includes('The nearest hazard in view is {side_at}.'));
  assert.ok(bank.nearest_clock_a.forms.includes('The nearest hazard in view sits at {clock} in the image.'));
  for (const id of ['nearest_kind_a', 'nearest_side_a', 'nearest_clock_a']) {
    assert.ok(!bank[id].forms.some((f) => /^The (?:nearest|closest) hazard (?:is|sits)/.test(f)), id);
  }
});

// T10-v: an in-cloud L record — the papi/sock caption parts and VQA never fire (their facts are null), the detail
// caption uses cap_l_cloud (richer than cap_l_lead, so richest() always picks it once scene.in_cloud is true), and the
// in_cloud VQA family answers yes for L too
test('T10-v: an in-cloud L record states no PAPI or windsock text and its detail caption carries the in-cloud lead', () => {
  const l4 = REC.find((r) => r.key === 'L_t_4'), ctx = ctxOf(l4);
  const S = visualSlots(l4);
  assert.equal(S.papi_white, undefined); assert.equal(S.papi_red, undefined); assert.equal(S.papi_path, undefined); assert.equal(S.sock, undefined);
  assert.ok(S.in_cloud, 'the in_cloud slot is set');
  for (let s = 1; s <= 8; s++) {
    const out = recordTexts(l4, { bank, gaz: GAZ, rng: mulberry32(s), split: 'train', context: ctx });
    assert.deepEqual(out.rejected.map((x) => `${x.item.template_id}: ${x.errors}`), []);
    const detail = out.texts.find((t) => t.task === 'caption_detail');
    assert.match(detail.template_id, /^cap_l_cloud\+/, `seed ${s}: ${detail.template_id}`);
    assert.match(detail.answer, /inside cloud/); assert.match(detail.answer, /hidden|obscured/);
    assert.ok(!out.texts.some((t) => /papi|windsock/i.test(t.answer)), `seed ${s}: a text mentions the PAPI or the windsock while in cloud`);
    const cloudVqa = out.texts.find((t) => t.family_q === 'in_cloud');
    if (cloudVqa) { assert.equal(cloudVqa.answerKey, 'yes'); assert.match(cloudVqa.answer, /cloud/i); assert.match(cloudVqa.template_id, /^cloud_l_/); }
  }
});
