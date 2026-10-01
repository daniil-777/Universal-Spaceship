// vlm/tools/colab_pack.mjs — the Colab package of a built dataset (rulings V1-6, V1-8), for upload to Google Drive:
//   shards/narrator-<split>-<nnnnn>.tar   WebDataset shards of about --shard-mb: <key>.jpg (the narrator frame, JPEG q90, 896x504)
//                                         and <key>.json (the record: facts, targets, masks, texts)
//   cache/index.json + cache/parts/pilot_eye_<split>.u8.<nnn>   the Pilot Eye uint8 cache in parts of at most --shard-mb
//   pilot_eye/*.jsonl, narrator/*.jsonl  (images[0] rewritten to the shard-relative '<shard>/<key>.jpg'), records.jsonl (the same
//   narrator_frame rewrite), labels.json, stats.json, splits.json, datasheet.md, ATTRIBUTION.txt, code/ (vlm/train, the vlm/gen
//   files slot_eval.mjs needs, the G2 processor files, the G2 parity baseline), APV_train_all.ipynb, README.md, manifest.json, SHA256SUMS
// The package is capped at --max-gb (8): the JPEG quality steps down from --quality when the estimate would not fit.
//   node vlm/tools/colab_pack.mjs --dataset apv-open-v1 --out /Volumes/LaCie/astro-pilot/vlm/colab/apv-open-v1/ [--max-gb 8]
//        [--shard-mb 500] [--quality 90] [--sample N --seed 1]   (--sample: a family x split stratified subset, for dry runs)
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import sharp from 'sharp';
import { createTar } from './tar.mjs';
import { mulberry32 } from '../../src/mathx.js';

const LACIE = '/Volumes/LaCie/astro-pilot/vlm', REPO = fileURLToPath(new URL('../../', import.meta.url));
const G2 = path.join(LACIE, 'models', 'narrator-base-g2-fused'), SPLITS = ['train', 'val', 'test', 'ood'], ROW = 3 * 160 * 96 * 3;
const readJsonl = (f) => (fs.existsSync(f) ? fs.readFileSync(f, 'utf8').split('\n').filter((l) => l.trim()).map((l) => JSON.parse(l)) : []);
const sha256 = (buf) => crypto.createHash('sha256').update(buf).digest('hex');
// the v1 raw snapshot (per-frame state for later build-side labels) stays in the LaCie dataset, not in the Colab package
const slim = ({ snapshot, ...r }) => r;
async function mapOrdered(xs, k, fn, each) { const q = []; let next = 0; for (let i = 0; i < xs.length; i++) { while (next < xs.length && next < i + k) { q.push(fn(xs[next], next)); next++; } await each(await q.shift(), i); } }

// a family x split stratified sample of n records (at least one per cell while n allows), deterministic
export function sampleKeys(recs, n, seed = 1) {
  const rng = mulberry32(seed), cells = new Map();
  for (const r of recs) { const c = `${r.family}|${r.split}`; if (!cells.has(c)) cells.set(c, []); cells.get(c).push(r.key); }
  const keys = [...cells.keys()].sort(), pick = new Set();
  for (const c of keys) { const xs = cells.get(c); for (let i = xs.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [xs[i], xs[j]] = [xs[j], xs[i]]; } }
  for (let round = 0; pick.size < Math.min(n, recs.length); round++) for (const c of keys) { const xs = cells.get(c); if (round < xs.length && pick.size < n) pick.add(xs[round]); }
  return pick;
}
// the files code/ carries: vlm/train (no tests), and the relative-import closure of the Node tools the notebook runs
export function codeFiles(repo = REPO, entries = ['vlm/gen/text/slot_eval.mjs']) {
  const walk = (d) => fs.readdirSync(path.join(repo, d), { withFileTypes: true }).flatMap((e) => (e.name === '__pycache__' || e.name === 'tests' || e.name.startsWith('.') ? [] : e.isDirectory() ? walk(path.join(d, e.name)) : [path.join(d, e.name)]));
  const out = new Set(walk('vlm/train').filter((f) => /\.(py|txt|json)$/.test(f))), todo = [...entries];
  while (todo.length) {
    const f = todo.pop(); if (out.has(f)) continue; out.add(f);
    const src = fs.readFileSync(path.join(repo, f), 'utf8');
    for (const m of src.matchAll(/(?:^|\n)\s*(?:import|export)\s[^'"]*?from\s+['"](\.[^'"]+)['"]|import\(\s*['"](\.[^'"]+)['"]\s*\)/g)) {
      const t = path.normalize(path.join(path.dirname(f), m[1] || m[2])); if (fs.existsSync(path.join(repo, t))) todo.push(t);
    }
  }
  for (const f of [...out]) if (f.endsWith('.js') && f.startsWith('vlm/gen/text/')) for (const b of fs.readdirSync(path.join(repo, 'vlm/gen/text/bank'))) if (b.endsWith('.json')) out.add(`vlm/gen/text/bank/${b}`);
  return [...out, 'vlm/package.json'].filter((f, i, a) => a.indexOf(f) === i).sort();
}
export const README = (m) => `# ${m.dataset} for Google Colab

${m.records} records (${Object.entries(m.counts).map(([s, n]) => `${s} ${n}`).join(', ')}), ${(m.total_bytes / 1e9).toFixed(2)} GB, package id ${m.id}.

## 1. Upload
Upload this whole folder to the top of your Google Drive, so that it is **My Drive/${m.dataset}/** (drag the folder onto
drive.google.com, or use Google Drive for desktop). Do not rename it. Wait until every file has finished uploading.

## 2. Run
Open **${m.dataset}/APV_train_all.ipynb** in Google Colab (double-click it on Drive, then "Open with Google Colaboratory").
Runtime > Change runtime type > **GPU** (A100 best, then L4; T4 works, slower), then **Runtime > Run all** and allow the
Drive access. The defaults train Pilot Eye for 30 min and the Narrator for 3 h, then export, evaluate and write
**My Drive/${m.dataset}/apv-models-v1.zip**. If Colab disconnects, open the notebook again and Run all: every stage resumes
from its last checkpoint on Drive (runs/v1/), finished stages are skipped.

## 3. Install
Download apv-models-v1.zip from Drive, then in the astro-pilot worktree on the Mac:

    node vlm/tools/install_models.mjs ~/Downloads/apv-models-v1.zip
    node vlm/web/serve.mjs --port 8795

The site's Narrator card and vlm/web/demo.html read /__vlm/models/current.json, which the installer writes.

## Contents
shards/ (WebDataset tars: <key>.jpg narrator frame at JPEG q${m.jpeg_quality}, <key>.json record), cache/ (Pilot Eye uint8 cache in parts,
joined by the notebook), pilot_eye/ and narrator/ (training rows; a narrator row's image is '<shard>/<key>.jpg'), records.jsonl,
labels.json, stats.json, splits.json, datasheet.md, ATTRIBUTION.txt (licence ${m.licence}), code/ (the training code), SHA256SUMS.
`;

export async function pack({ dataset, out, maxGb = 8, shardMb = 500, quality = 90, sample = null, seed = 1, root = LACIE, g2 = G2, repo = REPO, notebook = path.join(REPO, 'vlm/colab/APV_train_all.ipynb'), log = console.log }) {
  const src = path.join(root, 'datasets', dataset), shardBytes = shardMb * 1e6, maxBytes = maxGb * 1e9, sums = new Map();
  if (!fs.existsSync(path.join(src, 'records.jsonl'))) throw new Error(`${src}/records.jsonl not found: build the dataset first`);
  if (fs.existsSync(out) && fs.readdirSync(out).some((f) => !f.startsWith('.'))) throw new Error(`${out} is not empty: give a new folder`);
  fs.mkdirSync(out, { recursive: true });
  const write = (rel, data) => { const p = path.join(out, rel); fs.mkdirSync(path.dirname(p), { recursive: true }); fs.writeFileSync(p, data); sums.set(rel, sha256(Buffer.isBuffer(data) ? data : Buffer.from(data))); };
  let recs = readJsonl(path.join(src, 'records.jsonl'));
  const all = recs.length, keep = sample ? sampleKeys(recs, sample, seed) : null; if (keep) recs = recs.filter((r) => keep.has(r.key));
  const bySplit = Object.fromEntries(SPLITS.map((s) => [s, recs.filter((r) => r.split === s)]));
  // the JPEG quality that fits the cap: measured on up to 40 frames spread over the records
  const probe = recs.filter((_, i) => i % Math.max(1, Math.floor(recs.length / 40)) === 0).slice(0, 40), idx = JSON.parse(fs.readFileSync(path.join(src, 'cache/index.json'), 'utf8'));
  const size = (f) => (fs.existsSync(path.join(src, f)) ? fs.statSync(path.join(src, f)).size : 0);
  const fixed = recs.length * ROW + (recs.length / Math.max(1, all)) * ['records.jsonl', ...SPLITS.flatMap((s) => [`narrator/${s}.jsonl`, `pilot_eye/${s}.jsonl`])].reduce((a, f) => a + size(f), 0) + 20e6;
  const avgJson = recs.reduce((a, r) => a + JSON.stringify(r).length, 0) / Math.max(1, recs.length);
  let q = quality, est = 0;
  for (; q >= 60; q -= 5) {
    let jb = 0; for (const r of probe) jb += (await sharp(path.join(root, r.narrator_frame)).jpeg({ quality: q }).toBuffer()).length;
    est = fixed + recs.length * (jb / Math.max(1, probe.length) + avgJson + 2048);
    if (est <= maxBytes) break;
  }
  if (est > maxBytes) throw new Error(`the package would be about ${(est / 1e9).toFixed(1)} GB even at JPEG q60 (cap ${maxGb} GB): pack a --sample or raise --max-gb`);
  if (q !== quality) log(`JPEG quality ${quality} -> ${q} to stay under ${maxGb} GB (estimate ${(est / 1e9).toFixed(2)} GB)`);
  // shards: records in split order; each tar rolls over at shardBytes
  const shards = [], where = new Map();
  for (const s of SPLITS) {
    let tar = null, name = null, n = 0;
    const roll = () => { if (tar) { const { bytes, sha256: h } = tar.close(); shards.push({ name, path: `shards/${name}.tar`, split: s, n, bytes }); sums.set(`shards/${name}.tar`, h); } tar = null; };
    await mapOrdered(bySplit[s], 8, async (r) => sharp(path.join(root, r.narrator_frame)).jpeg({ quality: q }).toBuffer(), async (jpg, i) => {
      const r = bySplit[s][i];
      if (!tar || tar.bytes >= shardBytes) { roll(); name = `narrator-${s}-${String(shards.filter((x) => x.split === s).length).padStart(5, '0')}`; fs.mkdirSync(path.join(out, 'shards'), { recursive: true }); tar = createTar(path.join(out, 'shards', `${name}.tar`)); n = 0; }
      const img = `${name}/${r.key}.jpg`; where.set(r.key, img);
      tar.add(`${r.key}.jpg`, jpg); tar.add(`${r.key}.json`, JSON.stringify({ ...slim(r), narrator_frame: img, narrator_frame_raw: r.narrator_frame })); n++;
    });
    roll();
  }
  log(`${shards.length} shards (${shards.reduce((a, x) => a + x.bytes, 0) / 1e9} GB)`);
  // the Pilot Eye cache in parts (a sample rebuilds its split files from the source rows)
  const eyeParts = {}, index = { ...idx, index: {} };
  for (const s of SPLITS) {
    const rows = readJsonl(path.join(src, 'pilot_eye', `${s}.jsonl`)).filter((r) => !keep || keep.has(r.key)), srcFile = path.join(src, 'cache', `pilot_eye_${s}.u8`);
    write(`pilot_eye/${s}.jsonl`, rows.map((r) => JSON.stringify(r)).join('\n') + (rows.length ? '\n' : ''));
    index.index[s] = {}; eyeParts[s] = [];
    const fd = rows.length ? fs.openSync(srcFile, 'r') : null, per = Math.max(1, Math.floor(shardBytes / ROW)), buf = Buffer.alloc(ROW);
    for (let p = 0; p * per < rows.length; p++) {
      const part = `cache/parts/pilot_eye_${s}.u8.${String(p).padStart(3, '0')}`, chunk = rows.slice(p * per, (p + 1) * per), data = Buffer.alloc(chunk.length * ROW);
      chunk.forEach((r, j) => { fs.readSync(fd, buf, 0, ROW, (idx.index[s][r.key] / 3) * ROW); buf.copy(data, j * ROW); index.index[s][r.key] = 3 * (p * per + j); });
      write(part, data); eyeParts[s].push(part);
    }
    if (fd !== null) fs.closeSync(fd);
  }
  write('cache/index.json', JSON.stringify(index));
  // narrator rows and records with the shard-relative frame
  for (const s of SPLITS) {
    const rows = readJsonl(path.join(src, 'narrator', `${s}.jsonl`)).filter((r) => !keep || keep.has(r.key)).map((r) => ({ ...r, images: [where.get(r.key)] }));
    if (rows.some((r) => !r.images[0])) throw new Error(`narrator/${s}.jsonl names a record without a packed frame`);
    write(`narrator/${s}.jsonl`, rows.map((r) => JSON.stringify(r)).join('\n') + (rows.length ? '\n' : ''));
  }
  write('records.jsonl', recs.map((r) => JSON.stringify({ ...slim(r), narrator_frame: where.get(r.key), narrator_frame_raw: r.narrator_frame })).join('\n') + '\n');
  for (const f of ['labels.json', 'stats.json', 'datasheet.md', 'ATTRIBUTION.txt']) if (fs.existsSync(path.join(src, f))) write(f, fs.readFileSync(path.join(src, f)));
  write('splits.json', JSON.stringify(Object.fromEntries(SPLITS.map((s) => [s, bySplit[s].map((r) => r.key)]))));
  // code/: the training code, the Node slot evaluation and what it imports, the G2 processor files and parity baseline
  for (const f of codeFiles(repo)) write(`code/${f}`, fs.readFileSync(path.join(repo, f)));
  write('code/package.json', JSON.stringify({ private: true, type: 'module' }, null, 1) + '\n');
  for (const f of ['config.json', 'processor_config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.json', 'preprocessor_config.json', 'generation_config.json']) write(`code/processor/${f}`, fs.readFileSync(path.join(g2, f)));
  write('code/baselines/g2_parity.json', JSON.stringify({ c: Object.fromEntries(Object.entries(JSON.parse(fs.readFileSync(path.join(g2, 'parity.json'), 'utf8')).c).map(([d, c]) => [d, { ours: c.ours, published: c.published }])), source: 'narrator-base-g2-fused/parity.json' }, null, 1));
  if (fs.existsSync(notebook)) write('APV_train_all.ipynb', fs.readFileSync(notebook));
  const stats = fs.existsSync(path.join(src, 'stats.json')) ? JSON.parse(fs.readFileSync(path.join(src, 'stats.json'), 'utf8')) : {};
  const files = [...sums.keys()].filter((f) => !f.startsWith('shards/') && !f.startsWith('cache/parts/')).sort();
  const id = sha256(Buffer.from([...sums].sort().map(([f, h]) => `${h}  ${f}`).join('\n'))).slice(0, 16);
  const man = { format: 'apv-colab/1', id, dataset: path.basename(out.replace(/\/+$/, '')), source: dataset, created: new Date().toISOString(), sample: sample || null, jpeg_quality: q, licence: stats.build ? stats.build.licence : null,
    records: recs.length, counts: Object.fromEntries(SPLITS.map((s) => [s, bySplit[s].length])), shards, eye_parts: eyeParts, files, total_bytes: 0 };
  write('README.md', README({ ...man, total_bytes: [...sums.keys()].reduce((a, f) => a + fs.statSync(path.join(out, f)).size, 0) }));
  man.files = [...sums.keys()].filter((f) => !f.startsWith('shards/') && !f.startsWith('cache/parts/')).sort().concat('manifest.json');
  man.total_bytes = [...sums.keys()].reduce((a, f) => a + fs.statSync(path.join(out, f)).size, 0);
  write('manifest.json', JSON.stringify(man, null, 1));
  fs.writeFileSync(path.join(out, 'SHA256SUMS'), [...sums].sort().map(([f, h]) => `${h}  ${f}`).join('\n') + '\n');
  // macOS writes an AppleDouble ._<name> beside every file on exFAT (the provenance xattr): left in, a browser upload copies
  // them all to Drive, so they go (nothing in the package lists or reads them)
  const dropDouble = (d) => { for (const e of fs.readdirSync(d, { withFileTypes: true })) { const p = path.join(d, e.name); if (e.name.startsWith('._')) fs.rmSync(p, { force: true }); else if (e.isDirectory()) dropDouble(p); } };
  dropDouble(out);
  if (man.total_bytes > maxBytes) throw new Error(`the package is ${(man.total_bytes / 1e9).toFixed(2)} GB, over the ${maxGb} GB cap`);
  log(`packed ${recs.length} records into ${out}: ${(man.total_bytes / 1e9).toFixed(2)} GB, ${shards.length} shards, JPEG q${q}, id ${id}`);
  return man;
}

if (process.argv[1] && pathToFileURL(fs.realpathSync(process.argv[1])).href === import.meta.url) {
  const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
  if (!arg('dataset') || !arg('out')) { console.error('usage: node vlm/tools/colab_pack.mjs --dataset <name> --out <folder> [--max-gb 8] [--shard-mb 500] [--quality 90] [--sample N --seed 1]'); process.exit(2); }
  await pack({ dataset: arg('dataset'), out: path.resolve(arg('out')), maxGb: +arg('max-gb', 8), shardMb: +arg('shard-mb', 500), quality: +arg('quality', 90), sample: arg('sample') ? +arg('sample') : null, seed: +arg('seed', 1) });
}
