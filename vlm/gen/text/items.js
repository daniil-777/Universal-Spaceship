// vlm/gen/text/items.js — every text item of one record (spec §5.1): 2 captions (visual facts only), 1 safety explanation
// (flight; DriveLM chain, stated as the monitor's, with the Context line), 8-10 VQA and one negative, each rendered from the
// bank and verified; only verified items are kept (the rest go to the rejection log; `skipped` lists what the facts could
// not support). Every item carries needsContext and context_facts: the context-class facts it cites (§5.7: its row must
// carry the Context line, and export.js drops it when that Context does not supply them). VQA picks the families the
// record's Context supplies first. The Z geo and image facts are the build's (zoomgeo.js, imagefacts.js): place.nearest
// {name, km, bearing, compass}, place.country, place.admin1, place.in_view [{name, kind, region}], geo.sea_frac,
// geo.coast_side (left/right/top/bottom), image.palette_0..2, image.brightness_bin (dark/medium/bright), image.edge_bin.
import { pickForm, render, polish } from './paraphrase.js';
import { verifyTemplateItem, factsOf, catSlot, entSlot, countSlot, clockSlot, regionSlot, compassSlot, fmtSlot, ROUTE_NAMES } from './verify.js';
import { QFAMILIES, makeNegative, NEGATIVE_TYPES, exportable, stationSlot, W, KIND_A, KIND_N, reasonsText } from './vqa.js';
import { ACTION_TEXT, normContext } from './context.js';

const list = (xs) => (xs.length < 2 ? xs.join('') : `${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}`);
const pick = (xs, rng) => xs[Math.floor(rng() * xs.length)];
const BRIGHT = { dark: 'dark', medium: 'of medium brightness', bright: 'bright' };
const TAG_BASE = [['WATER_DOMINANT', 'mostly open water'], ['MOUNTAINS', 'mountains'], ['HILLS', 'hills'], ['FLAT', 'flat land']];
const TAG_EXTRA = [['COASTLINE', 'a coastline'], ['MOUNTAINS', 'mountains'], ['HIGH_TERRAIN', 'high ground'], ['URBAN', 'built-up areas'], ['DESERT', 'desert'], ['ICE', 'ice']];
// every zoom tag but NIGHT (the light slot carries it) in one phrase: "mountains with high ground and ice"
export function terrainPhrase(tags) {
  const t = new Set(tags || []), base = TAG_BASE.find(([k]) => t.has(k)), extra = TAG_EXTRA.filter(([k]) => t.has(k) && (!base || k !== base[0])).map(([, w]) => w);
  return base ? (extra.length ? `${base[1]} with ${list(extra)}` : base[1]) : extra.length ? list(extra) : null;
}
// the caption slots of a record, from obs:visual facts only
export function visualSlots(rec) {
  const F = factsOf(rec), S = {}, put = (k, s) => { if (s) S[k] = s; };
  const v = (id) => (F[id] && F[id].obs === 'visual' && F[id].v !== null && F[id].v !== undefined ? F[id].v : null);
  const n = v('hazards.count_in_frame'), kinds = v('kinds_in_frame');
  if (n !== null) put('hazards', countSlot('hazards.count_in_frame', n, 'hazard'));
  if (Array.isArray(kinds) && kinds.length > 1) put('kinds_list', catSlot('kinds_in_frame', kinds, list(kinds.map((k) => KIND_N[k]))));
  const i = [0, 1, 2, 3, 4, 5].find((j) => v(`hazard.${j}.in_frame`) === true);
  if (i !== undefined && KIND_A[v(`hazard.${i}.kind`)]) {
    const k = v(`hazard.${i}.kind`), c = v(`hazard.${i}.clock`), s = v(`hazard.${i}.side`), z = v(`hazard.${i}.size_bin`);
    put('a_kind', catSlot(`hazard.${i}.kind`, k, KIND_A[k])); put('kind', catSlot(`hazard.${i}.kind`, k, KIND_N[k]));
    if (c) put('clock', clockSlot(`hazard.${i}.clock`, c)); if (W.side[s]) put('side_at', catSlot(`hazard.${i}.side`, s, W.side[s])); if (W.size[z]) put('size', catSlot(`hazard.${i}.size_bin`, z, W.size[z]));
  }
  if (v('earth_in_frame') === true) put('earth', catSlot('earth_in_frame', true, 'the Earth'));
  if (v('moon_in_frame') === true) put('moon', catSlot('moon_in_frame', true, 'the Moon'));
  const lit = v('sun.lit'); if (lit !== null && (lit >= 0.9 || lit <= 0.1)) put('light_state', catSlot('sun.lit', lit >= 0.9, lit >= 0.9 ? 'in direct sunlight' : 'in shadow', ['gt', 0.5]));
  const route = v('route'), world = v('world'), sky = v('weather.preset'), cloud = v('air.in_cloud');
  if (ROUTE_NAMES[route]) put('route', entSlot('route', ROUTE_NAMES[route], ['map', ROUTE_NAMES]));
  if (W.world[world]) put('terrain', catSlot('world', world, W.world[world])); if (W.sky[sky]) put('sky', catSlot('weather.preset', sky, W.sky[sky]));
  if (v('in_tunnel') === true) put('tunnel', catSlot('in_tunnel', true, 'inside a tunnel')); if (cloud !== null && cloud >= 0.5) put('cloud', catSlot('air.in_cloud', true, 'inside cloud', ['gt', 0.25]));
  const gear = v('cfg.gear'), w = v('papi_whites_cam'), t = v('scene.time'), vis = v('scene.vis');
  if (gear) put('gear', catSlot('cfg.gear', gear, gear === 'down' ? 'down' : gear === 'up' ? 'up' : 'in transit'));
  if (w !== null) { put('papi_white', countSlot('papi_whites_cam', w, 'white')); put('papi_red', countSlot('papi_whites_cam', 4 - w, 'red', ['papi_red'])); put('papi_path', catSlot('papi_whites_cam', w, W.papi[w])); }
  if (v('windsock.from_deg') !== null) put('sock', fmtSlot('windsock.from_deg', F['windsock.from_deg']));
  if (W.time[t]) put('light', catSlot('scene.time', t, W.time[t])); if (W.vis[vis]) put('visibility', catSlot('scene.vis', vis, W.vis[vis]));
  const cc = v('scene.clouds') === null ? null : (/\b(FEW|SCT|BKN|OVC|NSC|SKC|NONE)\b/.exec(String(v('scene.clouds'))) || [])[1]; if (W.clouds[cc]) put('clouds', catSlot('scene.clouds', cc, W.clouds[cc], ['cloud_code']));
  if (v('scene.rain') === true) put('rain', catSlot('scene.rain', true, 'rain')); if (v('cfg.spoilers') !== null && v('cfg.spoilers') >= 0.5) put('spoilers', catSlot('cfg.spoilers', true, 'deployed', ['gt', 0.25]));
  const ph = v('phase'), sb = v('station_distance_bin');
  if (W.phase[ph]) put('dphase', catSlot('phase', ph, W.phase[ph]));
  if (typeof sb === 'string' && F.station_distance_bin.obs === 'visual') put('station_dist', stationSlot(F));
  if (rec.family === 'Z') {
    const tags = v('zoom.tags'), tp = terrainPhrase(tags), p = v('place.nearest'), c = v('place.country'), sea = v('geo.sea_frac'), cs = v('geo.coast_side'), sc = v('sun.class');
    if (v('view.range_km') > 0) put('range', fmtSlot('view.range_km', F['view.range_km']));
    if (tp) put('terrain', catSlot('zoom.tags', [...tags].sort().join(','), tp, ['sorted']));
    if (typeof c === 'string') put('country', entSlot('place.country', c));
    if (p && p.name && p.km > 0 && p.compass) { put('place', entSlot('place.nearest', p.name, ['field', 'name'])); put('place_km', { ...fmtSlot('place.nearest', { v: p.km, unit: 'km' }), derive: ['field', 'km'] }); put('compass', compassSlot('place.nearest', p.compass, ['field', 'compass'])); }
    const iv = v('place.in_view'), fi = Array.isArray(iv) ? iv.findIndex((x) => x.name && x.region && iv.filter((y) => y.name === x.name).length === 1) : -1;
    if (fi >= 0) { put('feature', entSlot('place.in_view', iv[fi].name, ['idx', fi, 'name'])); put('feature_region', regionSlot('place.in_view', iv[fi].region, ['idx', fi, 'region'])); }
    if (W.daylight[sc]) put('daylight', catSlot('sun.class', sc, W.daylight[sc])); if (W.coast[cs]) put('coast_side', catSlot('geo.coast_side', cs, W.coast[cs]));
    if (sea !== null && sea >= 0.02 && sea <= 0.98) put('sea', fmtSlot('geo.sea_frac', F['geo.sea_frac'])); if (v('view.gsd_m') > 0) put('gsd', fmtSlot('view.gsd_m', F['view.gsd_m']));
  }
  const c0 = v('image.palette_0'), c1 = v('image.palette_1'), br = v('image.brightness_bin'), ed = v('image.edge_bin');
  const c2 = v('image.palette_2');
  if (c0 && c1 && c0 !== c1) { put('colour_a', catSlot('image.palette_0', c0, c0)); put('colour_b', catSlot('image.palette_1', c1, c1)); if (c2 && c2 !== c0 && c2 !== c1) put('colour_c', catSlot('image.palette_2', c2, c2)); }
  if (BRIGHT[br] && ed) { put('brightness', catSlot('image.brightness_bin', br, BRIGHT[br])); put('texture', catSlot('image.edge_bin', ed, ed)); }
  return S;
}
const textOf = (slots) => Object.fromEntries(Object.entries(slots).map(([k, s]) => [k, typeof s === 'string' ? s : s.text]));
const usable = (bank, kind, family, S, part) => Object.keys(bank).filter((id) => bank[id].kind === kind && bank[id].families.includes(family) && (part === undefined || bank[id].part === part) && bank[id].slots.every((x) => S[x]));
const richest = (ids, bank, rng) => { const m = Math.max(...ids.map((id) => bank[id].slots.length)); return pick(ids.filter((id) => bank[id].slots.length === m), rng); };
const words = (s) => s.split(/\s+/).filter(Boolean).length;
// the detailed caption: a lead sentence (for Z it carries {range} and {terrain}) and then the family's parts in order
export const PLAN = Object.freeze({ S: ['hazard', 'side', 'kinds', 'bodies', 'light', 'palette', 'tone'], A: ['count', 'hazard', 'side', 'kinds', 'tunnel', 'cloud', 'palette', 'tone'],
  L: ['gear', 'papi', 'sock', 'clouds', 'rain', 'spoilers', 'palette', 'tone'], D: ['bodies', 'light', 'palette', 'tone'], Z: ['place', 'daylight', 'feature', 'coast', 'sea', 'palette', 'detail'] });
function composeDetail(rec, S, bank, rng, split) {
  const leads = usable(bank, 'caption_detail', rec.family, S); if (!leads.length) return null;
  const segs = [], add = (id) => { const f = pickForm(bank, id, rng, split); segs.push({ id, pid: f.paraphrase_id, text: polish(render(f.form, textOf(S))), slots: bank[id].slots.map((x) => S[x]) }); };
  add(richest(leads, bank, rng));
  for (const part of PLAN[rec.family] || []) {
    const ids = usable(bank, 'caption_part', rec.family, S, part); if (!ids.length) continue;
    const id = richest(ids, bank, rng), f = pickForm(bank, id, rng, split), text = polish(render(f.form, textOf(S)));
    if (words(segs.map((s) => s.text).join(' ')) + words(text) <= 90) segs.push({ id, pid: f.paraphrase_id, text, slots: bank[id].slots.map((x) => S[x]) });
  }
  return { answer: segs.map((s) => s.text).join(' '), slots: segs.flatMap((s) => s.slots), template_id: segs.map((s) => s.id).join('+'), paraphrase_id: segs.map((s) => s.pid).join('+') };
}
const idsOf = (slots, extra = []) => [...new Set([...slots.map((s) => s.fact_id), ...extra])];
// slots and checks served the verification; the stored item is the §4.4 text item (plus needsContext, context_facts and,
// for VQA, family_q and answerKey for balanceAnswers), so they stay on the object but out of its JSON
const hide = (it) => { for (const k of ['slots', 'checks']) if (k in it) Object.defineProperty(it, k, { value: it[k], enumerable: false }); return it; };
function contextFacts(ids, rec) { const F = factsOf(rec); return ids.filter((id) => id.startsWith('safety.') || (F[id] && F[id].obs === 'context')); }
// the monitor names a hazard ahead but none is in the frame: the target says so (§5.7)
const contradicts = (rec, m) => ['S', 'A'].includes(rec.family) && m.reasons.some((r) => r === 'HAZARD_AHEAD' || r === 'HAZARD_CLOSING_FAST') && factsOf(rec)['hazards.count_in_frame'] && factsOf(rec)['hazards.count_in_frame'].v === 0;
function contraSeg(bank, rng, split) { const s = countSlot('hazards.count_in_frame', 0, 'hazard'), f = pickForm(bank, 'contra', rng, split); return { text: polish(render(f.form, { none_seen: s.text })), slots: [s], pid: f.paraphrase_id }; }
const PERC = { S: [['perc_hazard', ['a_kind', 'clock']], ['perc_count', ['hazards']]], A: [['perc_hazard', ['a_kind', 'clock']], ['perc_count', ['hazards']]], L: [['perc_gear', ['gear']]], D: [['perc_docking', ['dphase', 'station_dist']]] };
function safetyChain(rec, S, m, bank, rng, split) {
  const o = rec.safety.action_outcome && rec.safety.action_outcome.CONTINUE, perc = (PERC[rec.family] || []).find(([, need]) => need.every((k) => S[k]));
  if (!perc || !W.outcome[o]) return null;
  const none = m.action === 'NONE_SAFE', chain = m.reasons.length ? (none ? 'safety_chain_nonesafe' : `safety_chain_${rec.family === 'S' || rec.family === 'A' ? 'sa' : rec.family.toLowerCase()}`) : none ? null : 'safety_chain_noreason';
  if (!chain || !bank[chain]) return null;
  const pf = pickForm(bank, perc[0], rng, split), outcome = catSlot('safety.action_outcome', o, W.outcome[o], ['field', 'CONTINUE']), of = pickForm(bank, 'what_if_a', rng, split), cf = pickForm(bank, chain, rng, split);
  const words_ = { perception: polish(render(pf.form, textOf(S))), prediction: polish(render(of.form, { outcome: outcome.text })), verdict: m.verdict, reason: reasonsText(m.reasons), action: ACTION_TEXT[m.action] };
  const slots = [...perc[1].map((k) => S[k]), outcome], pids = [pf.paraphrase_id, of.paraphrase_id, cf.paraphrase_id];
  let answer = polish(render(cf.form, words_));
  const contra = contradicts(rec, m); if (contra) { const c = contraSeg(bank, rng, split); answer = `${answer} ${c.text}`; slots.push(...c.slots); pids.push(c.pid); }
  return { answer, slots, template_id: `${perc[0]}+what_if_a+${chain}${contra ? '+contra' : ''}`, paraphrase_id: pids.join('+') };
}
export function recordTexts(rec, { bank, gaz, rng, split = 'train', context = null }) {
  const texts = [], rejected = [], skipped = [], ctx = normContext(context), m = ctx && ctx.monitor, S = visualSlots(rec);
  const keep = (it, extraIds = []) => {
    it.fact_ids = idsOf([...(it.slots || []), ...(it.checks || [])], [...(it.fact_ids || []), ...extraIds]); it.context_facts = contextFacts(it.fact_ids, rec); it.needsContext = it.context_facts.length > 0;
    const chk = verifyTemplateItem(it, rec, { gaz, context }); it.verified = chk.verified; hide(it);
    if (chk.verified) texts.push(it); else rejected.push({ item: it, errors: chk.errors, parserOk: chk.parserOk });
  };
  const base = { generator: 'template', false_premise: null };
  for (const kind of ['caption_short', 'caption_detail']) {
    if (kind === 'caption_short') {
      const ids = usable(bank, 'caption_short', rec.family, S); if (!ids.length) { skipped.push({ task: kind, why: 'no caption template fits the visual facts' }); continue; }
      const id = pick(ids, rng), f = pickForm(bank, id, rng, split), answer = polish(render(f.form, textOf(S)));
      if (words(answer) > 20) { skipped.push({ task: kind, why: `${words(answer)} words`, answer }); continue; }
      keep({ task: kind, prompt: 'Describe the image in one sentence.', answer, slots: bank[id].slots.map((x) => S[x]), template_id: id, paraphrase_id: f.paraphrase_id, ...base });
    } else {
      const d = composeDetail(rec, S, bank, rng, split); if (!d) { skipped.push({ task: kind, why: 'no lead sentence fits the visual facts' }); continue; }
      if (words(d.answer) < 40 || words(d.answer) > 90) { skipped.push({ task: kind, why: `${words(d.answer)} words`, answer: d.answer }); continue; }
      keep({ task: kind, prompt: 'Describe the image in detail.', ...d, ...base });
    }
  }
  if (rec.safety && m) {
    const c = safetyChain(rec, S, m, bank, rng, split);
    if (c) keep({ task: 'safety', prompt: 'Is the situation safe? Explain.', ...c, ...base }, ['safety.verdict', 'safety.reasons', 'safety.best_action', 'safety.action_outcome']);
    else skipped.push({ task: 'safety', why: 'no perception or chain template fits' });
  }
  const cands = [];
  for (const q of QFAMILIES) {
    if (!q.families.includes(rec.family)) continue;
    let a; try { a = q.ask(rec, rng, context); } catch (e) { rejected.push({ item: { task: 'vqa', family_q: q.id }, errors: [`ask threw: ${e.message}`] }); continue; }
    if (a && bank[a.qid] && bank[a.aid]) cands.push({ q, a, x: rng(), ok: exportable([...a.fact_ids, ...Object.values(a.a).map((s) => s.fact_id)], rec, context) });
  }
  cands.sort((p, r) => (p.ok === r.ok ? p.x - r.x : p.ok ? -1 : 1));
  for (const { q, a } of cands.slice(0, 8 + Math.floor(rng() * 3))) {
    const qf = pickForm(bank, a.qid, rng, split), af = pickForm(bank, a.aid, rng, split), slots = Object.values(a.a);
    let answer = polish(render(af.form, textOf(a.a))), pid = `${qf.paraphrase_id}+${af.paraphrase_id}`, tid = `${a.qid}+${a.aid}`;
    if (m && (q.id === 'is_safe' || q.id === 'why') && contradicts(rec, m)) { const c = contraSeg(bank, rng, split); answer = `${answer} ${c.text}`; slots.push(...c.slots); pid += `+${c.pid}`; tid += '+contra'; }
    keep({ task: 'vqa', prompt: polish(render(qf.form, textOf(a.q))), answer, slots, checks: a.checks, template_id: tid, paraphrase_id: pid, family_q: q.id, answerKey: a.answerKey, ...base }, a.fact_ids);
  }
  const negs = (NEGATIVE_TYPES[rec.family] || []).map((t) => makeNegative(rec, t, rng, gaz, { bank, split, context })).filter(Boolean);
  for (const n of negs) { n.context_facts = contextFacts(n.fact_ids, rec); n.needsContext = n.context_facts.length > 0; hide(n); if (!n.verified) rejected.push({ item: n, errors: n.errors, parserOk: !n.errors.some((e) => e.startsWith('parser')) }); }
  const good = negs.filter((n) => n.verified).map((n) => ({ n, x: rng(), ok: exportable(n.fact_ids, rec, context) })).sort((p, r) => (p.ok === r.ok ? p.x - r.x : p.ok ? -1 : 1));
  if (good.length) texts.push(good[0].n); else skipped.push({ task: 'negative', why: 'no negative type fits the facts' });
  return { texts, rejected, skipped };
}
