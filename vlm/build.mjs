// vlm/build.mjs — raw records -> dataset (spec §4.4, §5, §8, §9).
// node vlm/build.mjs --run <run>[,<run>…] --out <name> [--families S,A,L,D,Z] [--sizes sizes.json] [--context gt+noise|oof]
//   [--context-from preds.jsonl] [--confusion file]
// -> $LACIE/datasets/<name>/{records.jsonl, pilot_eye/*.jsonl, narrator/*.jsonl, llava/*.json, coco/captions_*.json,
//    cache/pilot_eye_<split>.u8 + cache/index.json, labels.json, stats.json, datasheet.md, ATTRIBUTION.txt, rejections.jsonl}
// Records are read from raw/<run>/<F>/ (^[SALDZ]_.*\.json$ only; ._* AppleDouble, .done, plan/zorder/checkpoint/stop, ledgers
// and logs are skipped). A record whose page has no .done (an interrupted drive) is left out. Each run's Z records use that
// run's own Z/plan.json. The teacher is off (controller ruling): no paid API is ever called here.
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { validateRecord, validateTextFacts, REASONS, ACTIONS, VERDICTS, ZOOM_TAGS } from './gen/schema.js';
import { imageFacts } from './gen/imagefacts.js';
import { splitOf, zRecheck, groupOf, assertGroupsDisjoint, queryOf } from './gen/build/split.js';
import { dhash, crossSplitDrops } from './gen/build/dedupe.js';
import { classWeights } from './gen/build/balance.js';
import { targetsOf, REG, REASON_SETS } from './gen/build/targets.js';
import { narratorRows, rowContextOf, eyeExcluded, llavaOf, cocoOf, writeCache, writeJsonl, EYE_SIZE, EYE_MIN_AXIAL_M } from './gen/build/export.js';
import { textTally, datasetStats, upstreamOf } from './gen/build/stats.js';
import { datasheet } from './gen/build/datasheet.js';
import { zoomGeo } from './gen/build/zoomgeo.js';
import { recordTexts } from './gen/text/items.js';
import { makeGazetteer, BASE_NAMES } from './gen/text/verify.js';
import { loadBank } from './gen/text/paraphrase.js';
import { balanceAnswers } from './gen/text/vqa.js';
import { monitorOf, corruptMonitor, telemetryOf } from './gen/text/context.js';
import { buildOodMask } from './gen/sampler_z.js';
import { mulberry32 } from '../src/mathx.js';
import { EYE, EYE_REMOVED } from './gen/safety.js';

const L = '/Volumes/LaCie/astro-pilot/vlm', REPO = fileURLToPath(new URL('../', import.meta.url));
const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
if (!arg('run') || !arg('out')) throw new Error('usage: node vlm/build.mjs --run <run>[,<run>…] --out <name> [--families S,A,L,D,Z] [--sizes sizes.json] [--context gt+noise|oof --context-from preds.jsonl]');
if (arg('teacher', 'off') !== 'off') throw new Error('the teacher is off for this dataset (controller ruling): no paid API is called by the build');
const runs = arg('run').split(','), name = arg('out'), out = path.join(L, 'datasets', name), fams = arg('families', 'S,A,L,D,Z').split(',');
const mode = arg('context', arg('context-from') ? 'oof' : 'gt+noise'); if (!['gt+noise', 'oof'].includes(mode) || (mode === 'oof' && !arg('context-from'))) throw new Error(`--context ${mode}: gt+noise, or oof with --context-from`);
const t0 = Date.now(), say = (s) => console.log(`[build ${((Date.now() - t0) / 1000).toFixed(0)} s] ${s}`), rng = mulberry32(20260929);
async function mapLimit(xs, k, fn) { let i = 0; await Promise.all(Array.from({ length: Math.min(k, xs.length) }, async () => { while (i < xs.length) { const j = i++; await fn(xs[j], j); } })); }
fs.rmSync(out, { recursive: true, force: true }); fs.mkdirSync(path.join(out, 'cache'), { recursive: true });

// ---- 1. load: validated records of finished pages only
const epOf = (key) => +/_(\d{5})_\d{6}$/.exec(key)[1];
function missingDone(r, done) {
  const need = [epOf(r.key)], pv = r.provenance;
  if (r.family === 'A' && Number.isInteger(pv.page_episode)) need.push(pv.page_episode);
  if (pv.twin_of) { need.push(epOf(pv.twin_of)); if (r.family === 'A' && Number.isInteger(pv.page_episode) && pv.page_episode >= 50000) need.push(pv.page_episode - 50000); }
  return need.find((e) => !done.has(e)) ?? null;
}
const recs = [], invalid = [], incomplete = [], plans = new Map();
for (const run of runs) for (const f of fams) {
  const d = path.join(L, 'raw', run, f); if (!fs.existsSync(d)) continue;
  const names = fs.readdirSync(d).filter((n) => !n.startsWith('._')).sort(), done = new Set(names.map((n) => /^episode_(\d{5})\.done$/.exec(n)).filter(Boolean).map((m) => +m[1]));
  if (f === 'Z' && fs.existsSync(path.join(d, 'plan.json'))) plans.set(run, JSON.parse(fs.readFileSync(path.join(d, 'plan.json'), 'utf8')));
  for (const n of names.filter((x) => /^[SALDZ]_.*\.json$/.test(x))) {
    const r = JSON.parse(fs.readFileSync(path.join(d, n), 'utf8')), miss = missingDone(r, done);
    if (miss !== null) { incomplete.push({ key: r.key, run, episode: miss }); continue; }
    const v = validateRecord(r); if (!v.ok) { invalid.push({ key: r.key, run, errors: v.errors }); continue; }
    r.frames = r.frames.map((x) => `raw/${run}/${f}/${x}`); r.narrator_frame = `raw/${run}/${f}/${r.narrator_frame}`; r.run = run; recs.push(r);
  }
}
recs.sort((a, b) => a.key.localeCompare(b.key));
say(`loaded ${recs.length} records from ${runs.join(',')} (${invalid.length} invalid, ${incomplete.length} of pages without .done)`);

// ---- 2. image facts (every record) and Z geo facts
const geo = await zoomGeo.load();
await mapLimit(recs, 6, async (r) => { Object.assign(r.facts, await imageFacts(path.join(L, r.narrator_frame))); });
for (const r of recs.filter((x) => x.family === 'Z')) await zoomGeo.apply(r, geo);
say('image and geo facts done');

// ---- 3. splits (Z: the run's plan; the captured view re-checked on the widened 32x18 grid, a broken view discarded)
const zDiscards = [], zrecs = recs.filter((r) => r.family === 'Z'), ood = zrecs.length ? buildOodMask(geo.ne) : null;
for (const r of zrecs) {
  const plan = plans.get(r.run); if (!plan) throw new Error(`${r.key}: raw/${r.run}/Z/plan.json is missing`);
  const want = splitOf(r, { zplan: plan }), got = zRecheck(r, { blocks: new Map(Object.entries(plan.blocks || {})), ood });
  if (got !== want) { zDiscards.push({ key: r.key, location: r.provenance.seed, want, got }); r.zDiscard = true; }
}
const live = recs.filter((r) => !r.zDiscard), byKey = new Map(live.map((r) => [r.key, r]));
const splits = live.map((r) => splitOf(r, { zplan: plans.get(r.run), byKey })); assertGroupsDisjoint(live, splits, byKey);
const splitOfKey = new Map(live.map((r, i) => [r.key, splits[i]]));
say(`splits done (${zDiscards.length} Z views discarded by the re-check)`);

// ---- 4. dedupe (dHash of f2) and pixel-identical minimal pairs
const hashes = new Map(); await mapLimit(live, 6, async (r) => { hashes.set(r.key, await dhash(path.join(L, r.frames[r.frames.length - 1]))); });
const dropped = crossSplitDrops(hashes, splitOfKey), kept0 = live.filter((r) => !dropped.has(r.key)), keptKey = new Map(kept0.map((r) => [r.key, r]));
const same = (a, b) => a.frames.length === b.frames.length && a.frames.every((f, i) => fs.readFileSync(path.join(L, f)).equals(fs.readFileSync(path.join(L, b.frames[i]))));
for (const r of kept0) { const t = r.provenance.twin_of && keptKey.get(r.provenance.twin_of); if (t && same(r, t)) r.pixel_identical_pair = t.pixel_identical_pair = true; }
say(`dedupe dropped ${dropped.size}`);

// ---- 5. texts: the row monitor (gt+noise, or oof), recordTexts against the row's Context, the verifier tally
const gaz = makeGazetteer([...BASE_NAMES, ...geo.names]), bank = loadBank(fileURLToPath(new URL('./gen/text/bank/', import.meta.url)));
const preds = mode === 'oof' ? new Map(fs.readFileSync(arg('context-from'), 'utf8').trim().split('\n').map((l) => JSON.parse(l)).map((p) => [p.key, p.monitor])) : null;
const confusion = arg('confusion') ? JSON.parse(fs.readFileSync(arg('confusion'), 'utf8')) : null, pools = {};
for (const r of kept0) if (r.safety_eye) (pools[r.family] ||= []).push({ ...monitorOf(r.safety_eye, r.facts, r.family), family: r.family });
const tally = textTally(), textFactFails = [], textErrors = [], rejFile = path.join(out, 'rejections.jsonl'), rejLines = [];
for (const r of kept0) {
  const split = splitOfKey.get(r.key), tf = validateTextFacts(r);
  if (!tf.ok) { textFactFails.push({ key: r.key, errors: tf.errors }); continue; }
  const gt = r.safety_eye ? monitorOf(r.safety_eye, r.facts, r.family) : null;
  r.row_monitor = gt ? (mode === 'oof' ? preds.get(r.key) ?? null : corruptMonitor(gt, { rng, confusion, pool: pools[r.family], family: r.family })) : null;
  let res;
  try { res = recordTexts(r, { bank, gaz, rng, split, context: rowContextOf(r) }); } catch (e) { textErrors.push({ key: r.key, error: String(e && e.message) }); res = { texts: [], rejected: [] }; }
  r.texts = res.texts; tally.add(r.family, res);
  for (const x of res.rejected) rejLines.push(JSON.stringify({ key: r.key, family: r.family, split, task: x.item.task, template_id: x.item.template_id ?? null, family_q: x.item.family_q ?? null, prompt: x.item.prompt ?? null, answer: x.item.answer ?? null, errors: x.errors, parserOk: x.parserOk ?? null }));
}
fs.writeFileSync(rejFile, rejLines.join('\n') + (rejLines.length ? '\n' : ''));
const failed = new Set(textFactFails.map((x) => x.key)), kept = kept0.filter((r) => !failed.has(r.key));
const vqa = kept.flatMap((r) => r.texts.filter((t) => t.task === 'vqa').map((t) => ({ ...t, key: r.key }))), keepVqa = new Set(balanceAnswers(vqa).map((t) => `${t.key}|${t.template_id}|${t.prompt}`));
for (const r of kept) r.texts = r.texts.filter((t) => t.task !== 'vqa' || keepVqa.has(`${r.key}|${t.template_id}|${t.prompt}`));
const textSummary = tally.summary();
say(`texts: ${textSummary.items} kept, ${textSummary.rejected} rejected (${textSummary.rejectRate}); ${textFactFails.length} records failed validateTextFacts, ${textErrors.length} text errors`);

// ---- 6. Pilot Eye targets and weights (D contact range excluded, ruling T10-g), build-time fields
const eyeRecs = kept.filter((r) => !eyeExcluded(r)), weights = classWeights(eyeRecs);
for (const r of kept) {
  const { targets, masks } = targetsOf(r), ex = eyeExcluded(r);
  Object.assign(r, { split: splitOfKey.get(r.key), group: groupOf(r, keptKey), natural: !r.provenance.injection && !r.provenance.twin_of, weight: ex ? null : weights.get(r.key), targets, masks, pilot_eye_excluded: ex,
    context_gt: r.safety_eye ? { monitor: monitorOf(r.safety_eye, r.facts, r.family), telemetry: telemetryOf(r.facts, r.family) } : null });
}
writeJsonl(path.join(out, 'records.jsonl'), kept);

// ---- 7. exports per split
const index = {}, allRows = [], exportDrops = {}, SPLITS = ['train', 'val', 'test', 'ood'];
for (const s of SPLITS) {
  const rs = kept.filter((r) => r.split === s), eye = rs.filter((r) => !r.pilot_eye_excluded);
  index[s] = await writeCache(out, s, eye);
  writeJsonl(path.join(out, 'pilot_eye', `${s}.jsonl`), eye.map((r) => { const q = queryOf(r); return { key: r.key, family: r.family, frames: r.frames, frame_dt_s: r.frame_dt_s, targets: r.targets, masks: r.masks, weight: r.weight, natural: r.natural,
    sampler_weight: r.provenance.sampler_weight ?? 1, group: r.group, twin_of: r.provenance.twin_of, injection: r.provenance.injection ? r.provenance.injection.kind : null, meta: { route: q.get('route'), sky: q.get('sky'), start: q.get('start'), phase: r.facts.phase ? r.facts.phase.v : null } }; }));
  const rows = rs.flatMap((r) => narratorRows(r, { rng, split: s, mode, gaz, drops: exportDrops })); allRows.push(...rows);
  writeJsonl(path.join(out, 'narrator', `${s}.jsonl`), rows);
  fs.mkdirSync(path.join(out, 'llava'), { recursive: true }); fs.writeFileSync(path.join(out, 'llava', `${s}.json`), JSON.stringify(llavaOf(rows)));
  fs.mkdirSync(path.join(out, 'coco'), { recursive: true }); fs.writeFileSync(path.join(out, 'coco', `captions_${s}.json`), JSON.stringify(cocoOf(rs)));
}
fs.writeFileSync(path.join(out, 'cache', 'index.json'), JSON.stringify({ size: EYE_SIZE, filter: 'box (exact area average, vlm/gen/boxresize.js)', rows_per_record: 3, index }));
say('exports written');

// ---- 8. labels, stats, datasheet, attribution
const cal = JSON.parse(fs.readFileSync(new URL('./gen/calibration.json', import.meta.url), 'utf8')), buildSha = (() => { try { return execFileSync('git', ['rev-parse', '--short', 'HEAD'], { cwd: REPO }).toString().trim(); } catch { return null; } })();
fs.writeFileSync(path.join(out, 'labels.json'), JSON.stringify({ c_near: cal.c_near, verdicts: VERDICTS, reasons: REASONS, actions: ACTIONS, zoom_tags: ZOOM_TAGS, reg: REG, reason_masks: REASON_SETS, eye: EYE, eye_hidden: EYE.hidden, eye_closing_rho_m: EYE.closingRhoM, eye_removed: EYE_REMOVED,
  eye_export_excludes: { D_axial_below_m: EYE_MIN_AXIAL_M, ruling: 'T10-g' }, nominal_frame_dt: { S: 0.2, A: 0.2, L: 0.25, D: 2 }, input: EYE_SIZE, resize: 'box', context_mode: mode }, null, 1));
const eyeStats = { records: Object.fromEntries(SPLITS.map((s) => [s, Object.keys(index[s]).length])), excluded_d_contact: kept.filter((r) => r.pilot_eye_excluded).length };
const stats = { build: { name, runs, families: fams, mode, build_sha: buildSha, seconds: Math.round((Date.now() - t0) / 1000) },
  ...datasetStats({ kept, recs, splitOfKey, dropped, invalid, zDiscards, textSummary, rows: allRows, exportDrops, eye: eyeStats, upstream: upstreamOf(path.join(L, 'raw'), runs), textFactFails }), incomplete: incomplete.length, textErrors };
fs.writeFileSync(path.join(out, 'stats.json'), JSON.stringify(stats, null, 1));
fs.writeFileSync(path.join(out, 'ATTRIBUTION.txt'), ['EOxCloudless https://cloudless.eox.at by EOX IT Services GmbH (Contains modified Copernicus Sentinel data 2016 & 2017)', 'Blue Marble: NASA Earth Observatory (GIBS), public domain',
  'AWS Terrain Tiles: SRTM, GMTED2010, ETOPO1 and others (see https://github.com/tilezen/joerd/blob/master/docs/attribution.md)', 'Made with Natural Earth. Free vector and raster map data @ naturalearthdata.com', 'NASA textures: see textures/CREDITS.txt in the Astro Pilot repository',
  'Meshy models (CC BY 4.0 credit to Meshy if made on the free plan; the user confirms the plan)'].join('\n') + '\n');
const sizes = arg('sizes') && fs.existsSync(arg('sizes')) ? JSON.parse(fs.readFileSync(arg('sizes'), 'utf8')).sizes ?? null : null, first = kept[0];
fs.writeFileSync(path.join(out, 'datasheet.md'), datasheet(stats, { name, date: new Date().toISOString(), runs, git_sha: [...new Set(kept.map((r) => r.provenance.git_sha))].join('/'), build_sha: buildSha, licence: first ? first.render.licence_profile : 'open',
  capture_mode: first ? first.render.capture_mode : 'clock', sizes, playwright: '1.63.0', c_near: cal.c_near, eye: EYE, eye_removed: EYE_REMOVED }));
console.log(`built ${name}: ${kept.length} records (${invalid.length} invalid, ${incomplete.length} without .done, ${dropped.size} deduped, ${zDiscards.length} Z views discarded, ${textFactFails.length} text-fact failures), ${textSummary.items} text items, ${allRows.length} narrator rows, reject rate ${textSummary.rejectRate}, parser false-reject ${textSummary.parserFalseReject}`);
