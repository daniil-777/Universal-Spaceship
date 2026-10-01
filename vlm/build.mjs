// vlm/build.mjs — raw records -> dataset (spec §4.4, §5, §8, §9).
// node vlm/build.mjs --run <run>[,<run>…] --out <name> [--families S,A,L,D,Z] [--sizes sizes.json] [--context gt+noise|oof] [--licence open|nc]
//   [--context-from preds.jsonl] [--confusion file] [--workers N] [--no-cache] [--cache-dir dir]
// -> $LACIE/datasets/<name>/{records.jsonl, pilot_eye/*.jsonl, narrator/*.jsonl, llava/*.json, coco/captions_*.json,
//    cache/pilot_eye_<split>.u8 + cache/index.json, labels.json, stats.json, datasheet.md, ATTRIBUTION.txt, rejections.jsonl}
// Records are read from raw/<run>/<F>/ (^[SALDZ]_.*\.json$ only; ._* AppleDouble, .done, plan/zorder/checkpoint/stop, ledgers
// and logs are skipped). A record whose page has no .done (an interrupted drive) is left out. Each run's Z records use that
// run's own Z/plan.json. The teacher is off (controller ruling): no paid API is ever called here.
// v1 scale-up (the v0 output, tests/vlm_build_scale.test.mjs): records are read concurrently; each record's PNGs are decoded
// once in a worker-thread pool (frames.js: image facts, dHash, Pilot Eye rows); the image facts, dHash and Z geo facts are
// cached per record key + code sha + record stamp in $LACIE/cache/build (factcache.js), so a rebuild is incremental; the
// dedupe is multi-index hashing (dedupe.js); JSONL/JSON outputs are written in pieces. Only the texts stay sequential: they
// draw from one seeded stream in record order.
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import { execFileSync, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { validateRecord, validateTextFacts, REASONS, ACTIONS, VERDICTS, ZOOM_TAGS } from './gen/schema.js';
import { splitOf, zRecheck, groupOf, assertGroupsDisjoint, queryOf, missingDone } from './gen/build/split.js';
import { crossSplitDrops } from './gen/build/dedupe.js';
import { splitClassWeights } from './gen/build/balance.js';
import { targetsOf, REG, REASON_SETS } from './gen/build/targets.js';
import { narratorRows, rowContextOf, eyeExcluded, llavaOf, cocoOf, writeCacheParallel, writeJsonl, writeJsonArray, EYE_SIZE, EYE_MIN_AXIAL_M, datasetNameOk } from './gen/build/export.js';
import { textTally, datasetStats, upstreamOf } from './gen/build/stats.js';
import { datasheet } from './gen/build/datasheet.js';
import { licenceOf, attributionOf, oofPreds } from './gen/build/licence.js';
import { zoomGeo } from './gen/build/zoomgeo.js';
import { createPool, poolSize } from './gen/build/pool.js';
import { openFactCache, codeShaOf, stampOf } from './gen/build/factcache.js';
import { addGroundFacts } from './gen/build/ground.js';
import { recordTexts } from './gen/text/items.js';
import { makeGazetteer, BASE_NAMES } from './gen/text/verify.js';
import { loadBank } from './gen/text/paraphrase.js';
import { balanceAnswers } from './gen/text/vqa.js';
import { monitorOf, corruptMonitor, telemetryOf } from './gen/text/context.js';
import { buildOodMask } from './gen/sampler_z.js';
import { mulberry32 } from '../src/mathx.js';
import { EYE, EYE_REMOVED } from './gen/safety.js';

const L = '/Volumes/LaCie/astro-pilot/vlm', REPO = fileURLToPath(new URL('../', import.meta.url));
const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; }, flag = (k) => process.argv.includes('--' + k);
if (arg('workers')) process.env.APV_BUILD_WORKERS = arg('workers');
// sharp decodes on the process's libuv threadpool, which every worker thread shares and which is sized once, before its first
// use: without UV_THREADPOOL_SIZE the build runs itself again with the threadpool sized to the workers
if (!process.env.UV_THREADPOOL_SIZE) {
  const r = spawnSync(process.execPath, process.argv.slice(1), { stdio: 'inherit', env: { ...process.env, UV_THREADPOOL_SIZE: String(poolSize() + 4) } });
  process.exit(r.status ?? 1);
}
// the commit whose code this process loaded: read at startup (a merge during a long build must not relabel it)
const buildSha = (() => { try { return execFileSync('git', ['rev-parse', '--short', 'HEAD'], { cwd: REPO, stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim(); } catch { return null; } })();
if (!arg('run') || !arg('out')) throw new Error('usage: node vlm/build.mjs --run <run>[,<run>…] --out <name> [--families S,A,L,D,Z] [--sizes sizes.json] [--context gt+noise|oof --context-from preds.jsonl]');
if (arg('teacher', 'off') !== 'off') throw new Error('the teacher is off for this dataset (controller ruling): no paid API is called by the build');
if (!datasetNameOk(arg('out'))) throw new Error(`--out ${JSON.stringify(arg('out'))}: a dataset name is one plain path segment ([A-Za-z0-9][A-Za-z0-9._-]*)`);
const runs = arg('run').split(','), name = arg('out'), out = path.join(L, 'datasets', name), fams = arg('families', 'S,A,L,D,Z').split(',');
const mode = arg('context', arg('context-from') ? 'oof' : 'gt+noise'); if (!['gt+noise', 'oof'].includes(mode) || (mode === 'oof' && !arg('context-from'))) throw new Error(`--context ${mode}: gt+noise, or oof with --context-from`);
const t0 = Date.now(), say = (s) => console.log(`[build ${((Date.now() - t0) / 1000).toFixed(0)} s] ${s}`), rng = mulberry32(20260929);
async function mapLimit(xs, k, fn) { let i = 0; await Promise.all(Array.from({ length: Math.min(k, xs.length) }, async () => { while (i < xs.length) { const j = i++; await fn(xs[j], j); } })); }
const yieldNow = () => new Promise((r) => setImmediate(r));
fs.rmSync(out, { recursive: true, force: true }); fs.mkdirSync(path.join(out, 'cache'), { recursive: true });
const pool = createPool(new URL('./gen/build/frames.js', import.meta.url));
// the modules whose code decides the cached facts (a change to any of them starts a new cache)
const CODE = ['gen/imagefacts.js', 'gen/boxresize.js', 'gen/schema.js', 'gen/build/dedupe.js', 'gen/build/frames.js', 'gen/build/zoomgeo.js', 'gen/build/split.js', 'gen/geo/naturalearth.js',
  'gen/geo/terrarium.js', 'gen/geo/sources.json', 'gen/labels/zoom.js', 'gen/text/items.js', 'gen/build/ground.js', 'capture/tilecache.mjs', '../src/earthtiles.js'].map((f) => fileURLToPath(new URL(`./${f}`, import.meta.url)));
const codeSha = codeShaOf(CODE), cache = flag('no-cache') ? null : openFactCache(arg('cache-dir', path.join(L, 'cache', 'build')), codeSha);

// ---- 1. load: validated records of finished pages only (the files are read concurrently, then taken in the v0 order)
const recs = [], invalid = [], incomplete = [], plans = new Map(), todo = [], stamps = new Map();
for (const run of runs) for (const f of fams) {
  const d = path.join(L, 'raw', run, f); if (!fs.existsSync(d)) continue;
  const names = fs.readdirSync(d).filter((n) => !n.startsWith('._')).sort(), done = new Set(names.map((n) => /^episode_(\d{5})\.done$/.exec(n)).filter(Boolean).map((m) => +m[1]));
  if (f === 'Z' && fs.existsSync(path.join(d, 'plan.json'))) plans.set(run, JSON.parse(fs.readFileSync(path.join(d, 'plan.json'), 'utf8')));
  for (const n of names.filter((x) => /^[SALDZ]_.*\.json$/.test(x))) todo.push({ run, f, d, n, done });
}
const raws = new Array(todo.length);
await mapLimit(todo, 16, async (t, i) => { raws[i] = await fsp.readFile(path.join(t.d, t.n), 'utf8'); });
for (const [i, { run, f, d, n, done }] of todo.entries()) {
  const r = JSON.parse(raws[i]), miss = missingDone(r, done); raws[i] = null;
  if (miss !== null) { incomplete.push({ key: r.key, run, episode: miss }); continue; }
  const v = validateRecord(r); if (!v.ok) { invalid.push({ key: r.key, run, errors: v.errors }); continue; }
  r.frames = r.frames.map((x) => `raw/${run}/${f}/${x}`); r.narrator_frame = `raw/${run}/${f}/${r.narrator_frame}`; r.run = run; recs.push(r);
  if (cache) stamps.set(r.key, stampOf(path.join(d, n)));
}
recs.sort((a, b) => a.key.localeCompare(b.key));
// one licence profile per dataset (spec R2): a build mixing open and nc records, or one whose --licence differs, is refused
const licence = licenceOf(recs, arg('licence', null));
say(`loaded ${recs.length} records from ${runs.join(',')} (${invalid.length} invalid, ${incomplete.length} of pages without .done)`);

// ---- 2. image facts, dHash, frame-file hashes and Pilot Eye rows (worker pool: every PNG read and decoded once, in key order,
// which is the capture's order on the disk) and Z geo facts; the cache holds all but the rows, which a cached record re-reads
const geo = await zoomGeo.load(), hashOf = new Map(), derived = new Map(), eyeRows = new Map(), groundErrors = [];
for (const r of recs) { const c = cache && cache.get(r.key, stamps.get(r.key)); if (c) derived.set(r.key, c); }
const fresh = recs.filter((r) => !derived.has(r.key)), hits = recs.length - fresh.length;
await mapLimit(fresh, 4 * pool.size, async (r) => {
  const j = await pool.run({ root: L, narrator: r.narrator_frame, frames: r.frames, want: { facts: true, hash: true, files: true, eye: true }, eyeSize: EYE_SIZE });
  derived.set(r.key, { image: j.facts, hash: j.hash, files: j.files }); eyeRows.set(r.key, j.eye);
});
// then the Z geo facts (main thread): what apply() adds or changes, in its order, so a cached entry replays it exactly
const geoOf = new Map();
for (const r of fresh.filter((x) => x.family === 'Z')) {
  const old = JSON.parse(JSON.stringify(r.facts)), probe = { ...r, facts: JSON.parse(JSON.stringify(r.facts)), zoom: { ...r.zoom } };
  await zoomGeo.apply(probe, geo);
  geoOf.set(r.key, { facts: Object.fromEntries(Object.entries(probe.facts).filter(([k, x]) => JSON.stringify(old[k]) !== JSON.stringify(x))), tags: probe.zoom.tags });
}
for (const r of fresh) { const c = derived.get(r.key); if (r.family === 'Z') c.geo = geoOf.get(r.key); if (cache) cache.put(r.key, stamps.get(r.key), c); }
if (cache) cache.flush();
for (const r of recs) {
  const c = derived.get(r.key);
  Object.assign(r.facts, c.image);
  if (r.family === 'Z') { for (const [k, x] of Object.entries(c.geo.facts)) r.facts[k] = x; r.zoom.tags = c.geo.tags; }
  hashOf.set(r.key, BigInt(`0x${c.hash}`));
  addGroundFacts(r, { onError: (x, e) => groundErrors.push({ key: x.key, error: String(e && e.message) }) });   // v1 grounding facts (cheap, not cached)
}
say(`image and geo facts done (${hits} of ${recs.length} from the cache ${codeSha}; ${pool.size} workers)${groundErrors.length ? `; ${groundErrors.length} records without grounding facts (first: ${groundErrors[0].key}: ${groundErrors[0].error})` : ''}`);

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

// ---- 4. dedupe (dHash of f2, multi-index hashing) and pixel-identical minimal pairs
const hashes = new Map(live.map((r) => [r.key, hashOf.get(r.key)]));
const dropped = crossSplitDrops(hashes, splitOfKey), kept0 = live.filter((r) => !dropped.has(r.key)), keptKey = new Map(kept0.map((r) => [r.key, r]));
// the same frame files byte for byte (SHA-256 of each file, taken when it was read)
const fileHashes = (r) => derived.get(r.key).files, same = (a, b) => a.frames.length === b.frames.length && fileHashes(a).every((h, i) => h === fileHashes(b)[i]);
for (const r of kept0) { const t = r.provenance.twin_of && keptKey.get(r.provenance.twin_of); if (t && same(r, t)) r.pixel_identical_pair = t.pixel_identical_pair = true; }
say(`dedupe dropped ${dropped.size}`);

// ---- 5. texts: the row monitor (gt+noise, or oof), recordTexts against the row's Context, the verifier tally
const gaz = makeGazetteer([...BASE_NAMES, ...geo.names]), bank = loadBank(fileURLToPath(new URL('./gen/text/bank/', import.meta.url)));
const preds = mode === 'oof' ? oofPreds(fs.readFileSync(arg('context-from'), 'utf8'), { groupOf: (k) => (keptKey.get(k) ? groupOf(keptKey.get(k), keptKey) : null), splitOf: (k) => splitOfKey.get(k) }) : null;
const confusion = arg('confusion') ? JSON.parse(fs.readFileSync(arg('confusion'), 'utf8')) : null, pools = {};
// the gt+noise pools per split x family, so a val/test monitor tuple never reaches a train Context line (review item 6)
const poolKey = (r) => `${splitOfKey.get(r.key)}|${r.family}`;
for (const r of kept0) if (r.safety_eye) (pools[poolKey(r)] ||= []).push({ ...monitorOf(r.safety_eye, r.facts, r.family), family: r.family });
const tally = textTally(), textFactFails = [], textErrors = [], rejFile = path.join(out, 'rejections.jsonl'), rejLines = [];
for (const r of kept0) {
  const split = splitOfKey.get(r.key), tf = validateTextFacts(r);
  // a record whose text facts fail keeps its Pilot Eye export and gets no Narrator texts (review item 5)
  if (!tf.ok) { textFactFails.push({ key: r.key, errors: tf.errors }); r.texts = []; r.row_monitor = null; continue; }
  const gt = r.safety_eye ? monitorOf(r.safety_eye, r.facts, r.family) : null;
  r.row_monitor = gt ? (mode === 'oof' ? preds.get(r.key) ?? null : corruptMonitor(gt, { rng, confusion, pool: pools[poolKey(r)], family: r.family })) : null;
  let res;
  try { res = recordTexts(r, { bank, gaz, rng, split, context: rowContextOf(r) }); } catch (e) { textErrors.push({ key: r.key, error: String(e && e.message) }); res = { texts: [], rejected: [] }; }
  r.texts = res.texts; tally.add(r.family, res);
  for (const x of res.rejected) rejLines.push(JSON.stringify({ key: r.key, family: r.family, split, task: x.item.task, template_id: x.item.template_id ?? null, family_q: x.item.family_q ?? null, prompt: x.item.prompt ?? null, answer: x.item.answer ?? null, errors: x.errors, parserOk: x.parserOk ?? null }));
}
fs.writeFileSync(rejFile, rejLines.join('\n') + (rejLines.length ? '\n' : ''));
const kept = kept0;
const vqa = kept.flatMap((r) => r.texts.filter((t) => t.task === 'vqa').map((t) => ({ ...t, key: r.key }))), keepVqa = new Set(balanceAnswers(vqa).map((t) => `${t.key}|${t.template_id}|${t.prompt}`));
for (const r of kept) r.texts = r.texts.filter((t) => t.task !== 'vqa' || keepVqa.has(`${r.key}|${t.template_id}|${t.prompt}`));
const textSummary = tally.summary();
say(`texts: ${textSummary.items} kept, ${textSummary.rejected} rejected (${textSummary.rejectRate}); ${textFactFails.length} records failed validateTextFacts, ${textErrors.length} text errors`);

// ---- 6. Pilot Eye targets and weights (D contact range excluded, ruling T10-g), build-time fields
for (const r of kept) r.split = splitOfKey.get(r.key);
const weights = splitClassWeights(kept.filter((r) => !eyeExcluded(r, r.split)));
for (const r of kept) {
  const { targets, masks } = targetsOf(r), ex = eyeExcluded(r, r.split);
  Object.assign(r, { group: groupOf(r, keptKey), natural: !r.provenance.injection && !r.provenance.twin_of, weight: ex ? null : weights.get(r.key), targets, masks, pilot_eye_excluded: ex,
    context_gt: r.safety_eye ? { monitor: monitorOf(r.safety_eye, r.facts, r.family), telemetry: telemetryOf(r.facts, r.family) } : null });
}
writeJsonl(path.join(out, 'records.jsonl'), kept);

// ---- 7. exports per split: the four Pilot Eye caches fill from the pool while the Narrator rows are made (in the v0 order)
const index = {}, rowStats = [], exportDrops = {}, SPLITS = ['train', 'val', 'test', 'ood'];
const caches = SPLITS.map((s) => writeCacheParallel(out, s, kept.filter((r) => r.split === s && !r.pilot_eye_excluded), pool, { rows: eyeRows }).then((x) => { index[s] = x; }));
for (const s of SPLITS) {
  const rs = kept.filter((r) => r.split === s), eye = rs.filter((r) => !r.pilot_eye_excluded), rows = [];
  writeJsonl(path.join(out, 'pilot_eye', `${s}.jsonl`), eye.map((r) => { const q = queryOf(r); return { key: r.key, family: r.family, frames: r.frames, frame_dt_s: r.frame_dt_s, targets: r.targets, masks: r.masks, weight: r.weight, natural: r.natural,
    sampler_weight: r.provenance.sampler_weight ?? 1, group: r.group, twin_of: r.provenance.twin_of, injection: r.provenance.injection ? r.provenance.injection.kind : null, pixel_identical_pair: !!r.pixel_identical_pair, meta: { route: q.get('route'), sky: q.get('sky'), start: q.get('start'), phase: r.facts.phase ? r.facts.phase.v : null } }; }));
  for (const [i, r] of rs.entries()) { rows.push(...narratorRows(r, { rng, split: s, mode, gaz, drops: exportDrops })); if (i % 10 === 9) await yieldNow(); }
  for (const x of rows) rowStats.push({ family: x.family, context: x.context, task: x.task });
  writeJsonl(path.join(out, 'narrator', `${s}.jsonl`), rows);
  writeJsonArray(path.join(out, 'llava', `${s}.json`), llavaOf(rows));
  fs.mkdirSync(path.join(out, 'coco'), { recursive: true }); fs.writeFileSync(path.join(out, 'coco', `captions_${s}.json`), JSON.stringify(cocoOf(rs)));
}
await Promise.all(caches); await pool.close(); eyeRows.clear();
fs.writeFileSync(path.join(out, 'cache', 'index.json'), JSON.stringify({ size: EYE_SIZE, filter: 'box (exact area average, vlm/gen/boxresize.js)', rows_per_record: 3, index: Object.fromEntries(SPLITS.map((s) => [s, index[s]])) }));
say('exports written');

// ---- 8. labels, stats, datasheet, attribution
const cal = JSON.parse(fs.readFileSync(new URL('./gen/calibration.json', import.meta.url), 'utf8'));
fs.writeFileSync(path.join(out, 'labels.json'), JSON.stringify({ c_near: cal.c_near, verdicts: VERDICTS, reasons: REASONS, actions: ACTIONS, zoom_tags: ZOOM_TAGS, reg: REG, reason_masks: REASON_SETS, eye: EYE, eye_hidden: EYE.hidden, eye_closing_rho_m: EYE.closingRhoM, eye_removed: EYE_REMOVED,
  eye_export_excludes: { D_axial_below_m: EYE_MIN_AXIAL_M, ruling: 'T10-g' }, nominal_frame_dt: { S: 0.2, A: 0.2, L: 0.25, D: 2 }, input: EYE_SIZE, resize: 'box', context_mode: mode }, null, 1));
const eyeStats = { records: Object.fromEntries(SPLITS.map((s) => [s, Object.keys(index[s]).length])), excluded_d_contact: kept.filter((r) => /T10-g/.test(r.pilot_eye_excluded || '')).length, excluded_pixel_identical_train: kept.filter((r) => /T11-a/.test(r.pilot_eye_excluded || '')).length };
const stats = { build: { name, runs, families: fams, mode, licence, build_sha: buildSha, seconds: Math.round((Date.now() - t0) / 1000) },
  ...datasetStats({ kept, recs, splitOfKey, dropped, invalid, zDiscards, textSummary, rows: rowStats, exportDrops, eye: eyeStats, upstream: upstreamOf(path.join(L, 'raw'), runs), textFactFails }), incomplete: incomplete.length, textErrors };
fs.writeFileSync(path.join(out, 'stats.json'), JSON.stringify(stats, null, 1));
fs.writeFileSync(path.join(out, 'ATTRIBUTION.txt'), attributionOf(kept, licence));
const sizes = arg('sizes') && fs.existsSync(arg('sizes')) ? JSON.parse(fs.readFileSync(arg('sizes'), 'utf8')).sizes ?? null : null, first = kept[0];
fs.writeFileSync(path.join(out, 'datasheet.md'), datasheet(stats, { name, date: new Date().toISOString(), runs, git_sha: [...new Set(kept.map((r) => r.provenance.git_sha))].join('/'), build_sha: buildSha, licence,
  capture_mode: first ? first.render.capture_mode : 'clock', sizes, mode, captured: kept.reduce((m, r) => ({ ...m, [r.family]: (m[r.family] || 0) + 1 }), {}), playwright: '1.63.0', c_near: cal.c_near, eye: EYE, eye_removed: EYE_REMOVED }));
console.log(`built ${name}: ${kept.length} records (${invalid.length} invalid, ${incomplete.length} without .done, ${dropped.size} deduped, ${zDiscards.length} Z views discarded, ${textFactFails.length} text-fact failures), ${textSummary.items} text items, ${rowStats.length} narrator rows, reject rate ${textSummary.rejectRate}, parser false-reject ${textSummary.parserFalseReject}`);
