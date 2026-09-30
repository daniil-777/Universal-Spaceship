import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { checkBank, loadBank, render, pickForm, heldOut, slotsOf } from '../vlm/gen/text/paraphrase.js';
import { QFAMILIES, balanceAnswers } from '../vlm/gen/text/vqa.js';
import { renderContext, parseContext, corruptMonitor, monitorOf } from '../vlm/gen/text/context.js';
import { runTeacher, estimateCost } from '../vlm/gen/text/teacher.js';
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
import { telemetryOf, contextSupplies } from '../vlm/gen/text/context.js';
import { W, KIND_A } from '../vlm/gen/text/vqa.js';
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
    'scene.time': V('day'), 'scene.vis': V('cavok'), 'scene.clouds': V('SCT030'), 'scene.rain': V(true), ...IMG }, safety: saf('CAUTION', ['GLIDESLOPE_DEVIATION'], 'CLIMB', 'landed') },
  { key: 'L_t_2', family: 'L', facts: { vert_mode: C('ROLLOUT'), ias_kt: C(60, 'kt'), 'ils.loc_dots': C(null, 'dots'), 'ils.gs_dots': C(null, 'dots'), papi_whites_cam: V(null, 'count'), 'windsock.from_deg': V(null, 'deg'), 'cfg.gear': V('down'),
    'cfg.spoilers': V(0.9), 'wind.head_kt': C(-11, 'kt'), 'wind.cross_kt': C(3, 'kt'), wow: C(true), 'scene.time': V('night'), 'scene.vis': V('fog'), 'scene.clouds': V('OVC003'), ...IMG }, safety: saf('UNSAFE', ['CANNOT_STOP'], 'NONE_SAFE', 'overrun', 'runway') },
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
    'wind.head_kt': C(5, 'kt'), 'wind.cross_kt': C(9, 'kt'), gates: C({ lateral: true, vertical: true, loc: true, gs: true, speed: true, vs: true, gear: true }), wow: C(false), 'scene.time': V('dusk'), 'scene.vis': V('haze'), ...IMG },
    safety: saf('UNSAFE', ['LOCALIZER_DEVIATION', 'UNSTABLE_APPROACH'], 'GO_AROUND', 'go_around') },
  { key: 'D_t_3', family: 'D', facts: { phase: V('CORRIDOR'), station_distance_bin: V('20-100 m'), rho_m: C(55, 'm'), closing_cms: C(3.2, 'cm/s'), corridor_limit_cms: C(8, 'cm/s'), speed_limit_cms: C(20, 'cm/s'), in_cone: C(false),
    att_err_deg: C(1.1, 'deg'), jets_failed: C([]), fuel_frac: C(0.9), breakout_available: C(true), ...IMG }, safety: saf('CAUTION', ['LATERAL_MISALIGNMENT'], 'HOLD_POSITION', 'breakout') },
  { key: 'Z_t_3', family: 'Z', facts: { 'view.range_km': V(250, 'km'), 'view.gsd_m': V(300, 'm'), 'sun.class': V('twilight'), 'place.country': V('Switzerland'), 'place.nearest': V({ name: 'Zermatt', km: 60, bearing: 190, compass: 'south' }),
    'place.in_view': V([{ name: 'Valais', kind: 'region', region: 'lower right' }]), 'geo.sea_frac': V(0), 'geo.coast_side': V(null), ...IMG }, zoom: { tags: ['HILLS', 'URBAN'], range_bin: 2 }, safety: null }];
const GAZ = makeGazetteer([...BASE_NAMES, 'Switzerland', 'Valais', 'Zermatt', 'Matterhorn', 'Apia', 'Tokyo', 'Paris', 'Split', 'Point', 'Orange', 'Wind', 'Mobile']);
const monOf = (r, verdict) => ({ verdict: verdict || r.safety.verdict, severity: SEV[verdict || r.safety.verdict], reasons: verdict ? ['HAZARD_AHEAD'] : [...r.safety.reasons], action: verdict ? 'CONTINUE' : r.safety.best_action, p_ref: null, ttc_bin: 'none', clr_bin: 'none' });
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
  for (const r of REC) for (const verdict of [null, 'X']) for (const split of ['train', 'test']) for (let s = 1; s <= 6; s++) {
    const ctx = verdict && r.safety ? ctxOf(r, r.safety.verdict === 'SAFE' ? 'UNSAFE' : 'SAFE') : ctxOf(r), out = recordTexts(r, { bank, gaz: GAZ, rng: mulberry32(s), split, context: ctx });
    assert.deepEqual(out.rejected.map((x) => `${x.item.template_id}: ${x.errors}`), [], r.key); assert.deepEqual(out.skipped.filter((x) => x.task !== 'safety'), [], r.key);
    const words = (t) => out.texts.find((x) => x.task === t).answer.split(/\s+/).length;
    assert.ok(words('caption_short') <= 20 && words('caption_detail') >= 40 && words('caption_detail') <= 90, r.key);
    assert.equal(out.texts.filter((t) => t.task === 'negative').length, 1); vqa += out.texts.filter((t) => t.task === 'vqa').length; neg += out.texts.filter((t) => t.task === 'negative').length;
    for (const t of out.texts) {
      assert.ok(t.verified && Array.isArray(t.fact_ids) && t.fact_ids.length && t.generator === 'template' && 'false_premise' in t, `${r.key} ${t.template_id}`);
      if (split === 'test') String(t.paraphrase_id).split('+').forEach((p, i) => { const tid = t.template_id.split('+')[i]; if (bank[tid] && bank[tid].forms.some((_, j) => heldOut(tid, j))) assert.ok(heldOut(tid, +p), `${tid}#${p}`); });
    }
  }
  assert.ok(neg / (vqa + neg) > 0.08 && neg / (vqa + neg) < 0.15, `negative share ${neg / (vqa + neg)}`);
});
test('§5.7: an item without needsContext states only visual facts; a needsContext item rests on what its Context supplies; the monitor is quoted, contradictions are said', () => {
  for (const r of REC) {
    const ctx = ctxOf(r), out = recordTexts(r, { bank, gaz: GAZ, rng: mulberry32(9), split: 'train', context: ctx });
    for (const t of out.texts) {
      if (!t.needsContext) assert.ok(verifyFreeText(t.answer, r, { gaz: GAZ, obsRule: true }).verified, `${r.key} ${t.template_id}: ${t.answer}`);
      else if (t.context_facts.every((id) => contextSupplies(ctx, id))) assert.ok(verifyFreeText(t.answer, r, { gaz: GAZ, obsRule: true, context: ctx }).verified, `${r.key} ${t.answer}`);
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
  for (const fam of ['S', 'A', 'L', 'D', 'Z']) {
    assert.ok(T.filter(([, t]) => t.families.includes(fam) && t.kind.startsWith('caption_') && t.kind !== 'caption_part').length >= 2, fam);
    if (fam !== 'Z') assert.ok(T.some(([, t]) => t.kind === 'safety' && t.families.includes(fam)), fam);
  }
  const fired = new Set(), aids = new Set();
  for (const r of REC) for (let k = 1; k <= 8; k++) for (const q of QFAMILIES) if (q.families.includes(r.family)) { const a = q.ask(r, mulberry32(k), { ...ctxOf(r), ...(r.safety ? { monitor: { ...monOf(r), ttc_bin: '1-3 s', clr_bin: '5-15 u' } } : {}) }); if (a) { fired.add(q.id); aids.add(a.aid); } }
  assert.deepEqual(QFAMILIES.map((q) => q.id).filter((id) => !fired.has(id)), [], 'every family fires on some record');
  assert.deepEqual(T.filter(([id, t]) => t.kind === 'vqa_a' && !aids.has(id)).map(([id]) => id), [], 'every answer template is produced');
});
