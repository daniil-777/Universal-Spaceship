// vlm/gen/text/slot_eval.mjs — the §11.5 slot evaluation of Narrator outputs on Z views, the blind baseline and the
// corrupted-Context gate (Task 15; amendment B: the T7 parser/verifier and the schema.js text constants, so the evaluation
// reads the words the dataset writes).
//   node vlm/gen/text/slot_eval.mjs --outputs merged.jsonl --base base.jsonl --records records.jsonl [--geo <NE dir>|none]
//        [--corrupted corrupt_rows.jsonl --corrupted-out outputs.jsonl]           -> JSON; exit 1 when a gate fails
//   node vlm/gen/text/slot_eval.mjs --make-rows <dir> --records records.jsonl [--n-corrupt 50] [--seed 5]
//        -> zoom_test(.._ctx).jsonl, zoom_ood.jsonl, blind.jsonl, corrupt_rows.jsonl, stop_set.jsonl, parity_facts.jsonl
// Checklist per Z view (all visual facts, scored without a Context line): range_bin, day/night (sun.class), each zoom tag of the
// view (NIGHT is the light slot), country, nearest named place — the §11.5 core, which the recall gate uses — plus the dominant
// colour (image.palette_0) and the brightness bin, reported alongside. Stated = a slot said right or wrong, a tag the view lacks,
// or any place name; hallucination = stated slots contradicting the facts, tags the view lacks and places outside its place
// facts / stated. Wilson 95 % CIs (0/0 -> null, the rate 'n/a'). Gates: core recall >= zero-shot + 0.20; hallucination upper
// bound <= 0.30. The §5.4 verifier is a conservative screen on free text, so the hallucination rate is a lower bound.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseClaims, makeGazetteer, verifyFreeText, UNITS, BASE_NAMES, W } from './verify.js';
import { norm } from './verify_claims.js';
import { corruptMonitor, monitorOf, renderContext, telemetryOf } from './context.js';
import { rangeBin, TAG_WORDS, PALETTE_NAMES, BRIGHTNESS_BINS, VERDICTS } from '../schema.js';
import { mulberry32 } from '../../../src/mathx.js';

export const CORE = Object.freeze(['range', 'light', 'tag', 'country', 'nearest']);
export const DETAIL_PROMPT = 'Describe the image in detail.', SAFETY_PROMPT = 'Is the situation safe? Explain.';
const LACIE = '/Volumes/LaCie/astro-pilot/vlm', GEO = `${LACIE}/geo`;
const LIGHT = new Set(['day', 'golden', 'twilight', 'dusk', 'night', 'lit']);
// a length in a place-distance clause ("Podgorica lies about 75 km to the south") or the ground resolution ("15 m per pixel")
// is not the viewing range
const NOT_RANGE = /\bpixels?\b|\blies\b|\b(?:north|south|east|west)(?:-?(?:east|west))?\b/;
// verifier errors about image facts (the corrupted-Context gate): numbers, counts, positions, places, kinds, presence
const IMAGE_ERR = /^(?:negated )?(?:number|range|count|clock|region|compass|entity|unknown entity|kind|presence)\b/;
const RANGE_TEXT = ['about 10 km', 'about 50 km', 'about 200 km', 'about 800 km', 'about 2,000 km'];

export const wilson = (k, n) => {
  if (!n) return null;
  const p = k / n, z = 1.96, d = 1 + z * z / n, c = p + z * z / (2 * n), h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n));
  return [(c - h) / d, (c + h) / d];
};
const fv = (r, id) => (r.facts[id] ? r.facts[id].v : null);
export const placeNames = (r) => [fv(r, 'place.country'), fv(r, 'place.admin1'), fv(r, 'place.nearest')?.name, ...(fv(r, 'place.in_view') || []).map((x) => x.name)].filter(Boolean);
export const recordNames = (recs) => [...new Set(recs.filter((r) => r.family === 'Z').flatMap(placeNames))];

// one output against its Z record -> [[slot, inChecklist, statedRight, statedWrong]]
export function scoreView(text, r, gaz) {
  const cl = parseClaims(text, gaz).filter((c) => !c.negated), cat = (dim) => cl.filter((c) => c.type === 'category' && c.dim === dim).map((c) => c.value);
  const out = [];
  const kms = cl.filter((c) => c.type === 'number' && c.dim === 'len' && !NOT_RANGE.test(c.clause || '')).map((c) => (c.value * UNITS[c.unit][1]) / 1000);
  const inBin = kms.some((k) => rangeBin(k) === r.zoom.range_bin);
  out.push(['range', true, inBin, kms.length > 0 && !inBin]);
  const sun = fv(r, 'sun.class') ?? r.zoom.lighting?.class, lights = cat('light').map((v) => [].concat(v).filter((x) => LIGHT.has(x))).filter((v) => v.length);
  const litOk = (vs) => vs.includes(sun) || (sun !== 'night' && vs.includes('lit'));
  out.push(['light', true, lights.some(litOk), lights.some((vs) => !litOk(vs))]);
  const tags = (r.zoom.tags || []).filter((t) => t !== 'NIGHT' && TAG_WORDS[t]), said = new Set(cat('tag'));
  for (const t of tags) out.push(['tag', true, said.has(t), false]);
  for (const t of said) if (!tags.includes(t)) out.push(['tag', false, false, true]);
  const ents = [...new Set(cl.filter((c) => c.type === 'entity' && c.value).map((c) => c.value))], places = new Set(placeNames(r).map(norm));
  const country = fv(r, 'place.country'), nearest = fv(r, 'place.nearest')?.name, said1 = (n) => !!n && ents.some((e) => norm(e) === norm(n));
  if (country) out.push(['country', true, said1(country), false]);
  if (nearest) out.push(['nearest', true, said1(nearest), false]);
  for (const e of ents) if (!(country && norm(e) === norm(country)) && !(nearest && norm(e) === norm(nearest))) out.push(['entity', false, places.has(norm(e)), !places.has(norm(e))]);
  const pal = [0, 1, 2].map((i) => fv(r, `image.palette_${i}`)).filter((c) => PALETTE_NAMES.includes(c)), cols = cat('colour').map((v) => [].concat(v).filter((c) => PALETTE_NAMES.includes(c))).filter((v) => v.length);
  if (pal.length) out.push(['palette', true, cols.some((v) => v.includes(pal[0])), cols.some((v) => !v.some((c) => pal.includes(c)))]);
  const b = fv(r, 'image.brightness_bin'), bs = cat('bright').filter((v) => BRIGHTNESS_BINS.includes(v));
  if (BRIGHTNESS_BINS.includes(b)) out.push(['brightness', true, bs.includes(b), bs.some((v) => v !== b)]);
  return out;
}

export function score(rows, recs, gaz) {
  const slots = {}, S = (k) => (slots[k] ||= { hit: 0, total: 0, stated: 0, bad: 0 });
  for (const o of rows) {
    const r = recs.get(o.key);
    if (!r || r.family !== 'Z') throw new Error(`no record for ${o.key} (or not a Z view)`);
    for (const [k, inList, ok, wrong] of scoreView(o.text, r, gaz)) { const s = S(k); if (inList) { s.total++; if (ok) s.hit++; } if (ok || wrong) s.stated++; if (wrong) s.bad++; }
  }
  const sum = (ks, f) => ks.reduce((a, k) => a + (slots[k] ? slots[k][f] : 0), 0), all = Object.keys(slots);
  const [hit, total, ch, ct, stated, bad] = [sum(all, 'hit'), sum(all, 'total'), sum(CORE, 'hit'), sum(CORE, 'total'), sum(all, 'stated'), sum(all, 'bad')];
  return { n: rows.length, recall_core: ct ? ch / ct : null, recall_core_ci: wilson(ch, ct), recall: total ? hit / total : null, recall_ci: wilson(hit, total),
    halluc: stated ? bad / stated : 'n/a', halluc_ci: wilson(bad, stated), slots };
}
export const gates = (merged, base) => ({ recall: merged.recall_core !== null && merged.recall_core >= (base.recall_core || 0) + 0.2, hallucination: merged.halluc_ci ? merged.halluc_ci[1] <= 0.3 : true });

// the corrupted-Context gate: the output restates the (corrupted) monitor verdict of its Context line (>= 90 %), and no
// image-fact claim is rejected against the record's facts
export function scoreCorrupted(prompts, outputs, recs, gaz) {
  const P = new Map(prompts.map((p) => [p.key, p])); let ok = 0, contra = 0;
  for (const o of outputs) {
    const p = P.get(o.key), r = recs.get(o.key);
    if (!p || !r) throw new Error(`no corrupted prompt or record for ${o.key}`);
    if (parseClaims(o.text, gaz).some((c) => c.type === 'verdict' && !c.negated && c.value === p.monitor_verdict)) ok++;
    const line = p.prompt.split('\n')[0], ctx = line.startsWith('Context:') ? line : { telemetry: [], monitor: { verdict: p.monitor_verdict, reasons: [], action: 'CONTINUE' } };
    if (verifyFreeText(o.text, r, { gaz, context: ctx }).errors.some((e) => IMAGE_ERR.test(e))) contra++;
  }
  const n = outputs.length, restated = n ? ok / n : 0;
  return { restated, n, contradictions: contra, pass: n > 0 && restated >= 0.9 && contra === 0 };
}

// ---- evaluation rows ----
export const zoomRows = (recs, split, { context = false, prompt = DETAIL_PROMPT } = {}) => recs.filter((r) => r.family === 'Z' && r.split === split)
  .map((r) => ({ key: r.key, image: r.narrator_frame, prompt: context ? `${renderContext({ telemetry: telemetryOf(r.facts, 'Z'), monitor: null })}\n${prompt}` : prompt }));
const majority = (xs) => { const c = new Map(); for (const x of xs) if (x !== null && x !== undefined) c.set(x, (c.get(x) || 0) + 1); return [...c].sort((a, b) => b[1] - a[1])[0]?.[0] ?? null; };
// the blind baseline: every test view gets the train-split majority value of every slot, in the dataset's words
export function blindRows(recs, split = 'test') {
  const z = recs.filter((r) => r.family === 'Z'), tr = z.some((r) => r.split === 'train') ? z.filter((r) => r.split === 'train') : z;
  const bin = majority(tr.map((r) => r.zoom.range_bin)), tag = majority(tr.flatMap((r) => r.zoom.tags || []).filter((t) => t !== 'NIGHT')), sun = majority(tr.map((r) => fv(r, 'sun.class')));
  const country = majority(tr.map((r) => fv(r, 'place.country'))), place = majority(tr.map((r) => fv(r, 'place.nearest')?.name)), pal = majority(tr.map((r) => fv(r, 'image.palette_0'))), b = majority(tr.map((r) => fv(r, 'image.brightness_bin')));
  const text = [`A view of ${tag ? TAG_WORDS[tag][0] : 'the ground'}${country ? ` in ${country}` : ''} from ${RANGE_TEXT[bin ?? 0]} ${W.daylight[sun] || 'in full daylight'}${place ? `, near ${place}` : ''}.`,
    pal ? `Most of the frame is ${pal}.` : '', b ? `The image is ${W.bright[b]}.` : ''].filter(Boolean).join(' ');
  return z.filter((r) => r.split === split).map((r) => ({ key: r.key, text }));
}
// flight test rows whose Context carries a monitor verdict forced away from the ground truth (a whole same-family tuple of
// that verdict when the pool has one, else the ground-truth tuple with the verdict swapped)
export function corruptRows(recs, { n = 50, seed = 5, split = 'test' } = {}) {
  const rng = mulberry32(seed), flight = recs.filter((r) => r.family !== 'Z' && r.safety_eye);
  return flight.filter((r) => r.split === split).slice(0, n).map((r) => {
    const gt = monitorOf(r.safety_eye, r.facts, r.family), others = VERDICTS.filter((v) => v !== gt.verdict);
    const pool = flight.filter((x) => x.family === r.family).map((x) => ({ ...monitorOf(x.safety_eye, x.facts, x.family), family: x.family }));
    let m = corruptMonitor(gt, { rng, confusion: { [gt.verdict]: Object.fromEntries(VERDICTS.map((v) => [v, v === gt.verdict ? 0 : 0.5])) }, pool, family: r.family });
    if (m.verdict === gt.verdict) m = { ...m, verdict: others[Math.floor(rng() * others.length)] };
    return { key: r.key, image: r.narrator_frame, prompt: `${renderContext({ telemetry: telemetryOf(r.facts, r.family), monitor: m })}\n${SAFETY_PROMPT}`, monitor_verdict: m.verdict, gt_verdict: gt.verdict };
  });
}
const abs = (p) => (p.startsWith('/') ? p : `${LACIE}/${p}`);
export function makeRows(recs, dir, { nCorrupt = 50, seed = 5 } = {}) {
  const test = zoomRows(recs, 'test'), flight = recs.filter((r) => r.family !== 'Z' && r.split === 'test'), pf = (flight.length ? flight : recs).slice(0, 3);
  const files = { 'zoom_test.jsonl': test, 'zoom_test_ctx.jsonl': zoomRows(recs, 'test', { context: true }), 'zoom_ood.jsonl': zoomRows(recs, 'ood'), 'blind.jsonl': blindRows(recs),
    'corrupt_rows.jsonl': corruptRows(recs, { n: nCorrupt, seed }), 'stop_set.jsonl': test.map((r) => ({ image: abs(r.image), prompt: r.prompt })),
    'parity_facts.jsonl': pf.map((r) => ({ image: abs(r.narrator_frame), prompt: 'Describe the image in one sentence.' })) };
  fs.mkdirSync(dir, { recursive: true });
  for (const [f, rows] of Object.entries(files)) fs.writeFileSync(path.join(dir, f), rows.map((x) => JSON.stringify(x)).join('\n') + (rows.length ? '\n' : ''));
  return Object.fromEntries(Object.entries(files).map(([f, rows]) => [f, rows.length]));
}

// ---- CLI ----
export const readJsonl = (f) => fs.readFileSync(f, 'utf8').split('\n').filter((l) => l.trim()).map((l) => JSON.parse(l));
async function gazetteer(geo, recs) {
  let names = [];
  if (geo !== 'none') {
    const { loadNaturalEarth, gazetteerNames, NE_LAYERS } = await import('../geo/naturalearth.js');
    names = gazetteerNames(await loadNaturalEarth(geo, NE_LAYERS.filter((l) => l !== 'ne_10m_land')));
  }
  return makeGazetteer([...BASE_NAMES, ...names, ...recordNames(recs)]);
}
async function main(argv) {
  const arg = (k) => { const i = argv.indexOf('--' + k); return i >= 0 ? argv[i + 1] : null; };
  if (!arg('records')) throw new Error('--records is required');
  const all = readJsonl(arg('records')), recs = new Map(all.map((r) => [r.key, r]));
  if (arg('make-rows')) { console.log(JSON.stringify(makeRows(all, arg('make-rows'), { nCorrupt: +(arg('n-corrupt') || 50), seed: +(arg('seed') || 5) }))); return 0; }
  if (!arg('outputs') || !arg('base')) throw new Error('--outputs and --base are required');
  const gaz = await gazetteer(arg('geo') || GEO, all), merged = score(readJsonl(arg('outputs')), recs, gaz), base = score(readJsonl(arg('base')), recs, gaz);
  const out = { merged, zero_shot: base, gates: gates(merged, base) };
  if (arg('corrupted')) {
    if (!arg('corrupted-out')) throw new Error('--corrupted needs --corrupted-out');
    out.corrupted = scoreCorrupted(readJsonl(arg('corrupted')), readJsonl(arg('corrupted-out')), recs, gaz);
  }
  console.log(JSON.stringify(out, null, 1));
  return out.gates.recall && out.gates.hallucination && (!out.corrupted || out.corrupted.pass) ? 0 : 1;
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) process.exitCode = await main(process.argv.slice(2));
