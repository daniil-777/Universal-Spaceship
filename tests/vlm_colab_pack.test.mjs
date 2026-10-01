// The Colab package (V1-6, V1-8): the ustar shard writer, the stratified sample, the code/ snapshot, and pack() on a small
// synthetic dataset (shards, JPEG frames, rewritten image keys, cache parts that join back, SHA256SUMS, manifest, cap).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { createTar, readTar } from '../vlm/tools/tar.mjs';
import { sampleKeys, codeFiles, pack } from '../vlm/tools/colab_pack.mjs';
const sharp = createRequire(new URL('../vlm/package.json', import.meta.url))('sharp');
const tmp = (p) => fs.mkdtempSync(path.join(os.tmpdir(), p));
const sha = (b) => crypto.createHash('sha256').update(b).digest('hex');

test('tar: flat members round-trip, the shard hash is the file hash, and bsdtar/GNU tar list it', () => {
  const d = tmp('apv-tar-'), f = path.join(d, 's.tar'), t = createTar(f);
  t.add('k1.jpg', Buffer.from('abc')); t.add('k1.json', '{"a":1}'); t.add('k2.jpg', Buffer.alloc(1025, 7));
  const { bytes, sha256 } = t.close(), buf = fs.readFileSync(f);
  assert.equal(bytes, buf.length); assert.equal(sha256, sha(buf)); assert.equal(buf.length % 512, 0);
  assert.deepEqual(readTar(f).map((m) => [m.name, m.data.length]), [['k1.jpg', 3], ['k1.json', 7], ['k2.jpg', 1025]]);
  const ls = spawnSync('tar', ['-tf', f], { encoding: 'utf8' }); assert.equal(ls.status, 0, ls.stderr); assert.deepEqual(ls.stdout.trim().split('\n'), ['k1.jpg', 'k1.json', 'k2.jpg']);
  assert.throws(() => createTar(path.join(d, 'x.tar')).add('a/b.jpg', 'x'), /flat names/);
  fs.rmSync(d, { recursive: true, force: true });
});
test('sample: every family x split cell is represented while n allows, deterministically', () => {
  const recs = []; for (const f of 'SALDZ') for (const s of ['train', 'val', 'test']) for (let i = 0; i < (s === 'train' ? 20 : 3); i++) recs.push({ key: `${f}_${s}_${i}`, family: f, split: s });
  const k = sampleKeys(recs, 30, 1); assert.equal(k.size, 30); assert.deepEqual(k, sampleKeys(recs, 30, 1));
  for (const f of 'SALDZ') for (const s of ['train', 'val', 'test']) assert.ok([...k].some((x) => x.startsWith(`${f}_${s}_`)), `${f}|${s}`);
});
test('code/: the training code without its tests, and every module slot_eval.mjs imports', () => {
  const c = codeFiles();
  for (const f of ['vlm/train/colab/narrator.py', 'vlm/train/narrator/export_decoder.py', 'vlm/gen/text/slot_eval.mjs', 'vlm/gen/text/verify_ground.js', 'vlm/gen/schema.js', 'src/mathx.js', 'vlm/gen/text/bank/grounding.json', 'vlm/package.json']) assert.ok(c.includes(f), f);
  assert.ok(!c.some((f) => f.includes('/tests/') || f.includes('__pycache__')));
});
async function dataset(root) {
  const ds = path.join(root, 'datasets', 'tiny'), raw = path.join(root, 'raw', 'r', 'S'); fs.mkdirSync(raw, { recursive: true });
  const recs = [['S_r_00000_000001', 'train'], ['S_r_00000_000002', 'train'], ['S_r_00001_000001', 'val'], ['S_r_00002_000001', 'test']].map(([key, split], i) => ({ key, family: 'S', split, narrator_frame: `raw/r/S/${key}.f2.png`, frames: [`raw/r/S/${key}.f2.png`], texts: [], facts: {}, snapshot: { v: 1 } }));
  for (const [i, r] of recs.entries()) await sharp({ create: { width: 896, height: 504, channels: 4, background: { r: 40 * i, g: 90, b: 160, alpha: 1 } } }).png().toFile(path.join(root, r.narrator_frame));
  const w = (f, s) => { fs.mkdirSync(path.dirname(path.join(ds, f)), { recursive: true }); fs.writeFileSync(path.join(ds, f), s); };
  w('records.jsonl', recs.map((r) => JSON.stringify(r)).join('\n') + '\n');
  const index = { train: {}, val: {}, test: {}, ood: {} }, ROW = 3 * 160 * 96 * 3;
  for (const s of ['train', 'val', 'test', 'ood']) {
    const rs = recs.filter((r) => r.split === s); rs.forEach((r, j) => { index[s][r.key] = 3 * j; });
    w(`cache/pilot_eye_${s}.u8`, Buffer.concat(rs.map((r, j) => Buffer.alloc(ROW, 10 * j + s.length))));
    w(`pilot_eye/${s}.jsonl`, rs.map((r) => JSON.stringify({ key: r.key, family: 'S' })).join('\n') + (rs.length ? '\n' : ''));
    w(`narrator/${s}.jsonl`, rs.map((r) => JSON.stringify({ images: [r.narrator_frame], key: r.key, family: 'S', task: 'vqa', messages: [] })).join('\n') + (rs.length ? '\n' : ''));
  }
  w('cache/index.json', JSON.stringify({ size: [160, 96], index })); w('labels.json', '{}'); w('stats.json', JSON.stringify({ build: { licence: 'open' } })); w('datasheet.md', '# x\n'); w('ATTRIBUTION.txt', 'x\n');
  const g2 = path.join(root, 'g2'); fs.mkdirSync(g2);
  for (const f of ['config.json', 'processor_config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.json', 'preprocessor_config.json', 'generation_config.json']) fs.writeFileSync(path.join(g2, f), '{}');
  fs.writeFileSync(path.join(g2, 'parity.json'), JSON.stringify({ c: { q4f16: { ours: 0.63, published: 0.63, ours_per_sample: [1] } } }));
  return { recs, g2, ROW };
}
test('pack: shards with JPEG frames and records, shard-relative rows, cache parts that join, SHA256SUMS and the manifest', async () => {
  const root = tmp('apv-pack-'), out = path.join(root, 'pkg'), { recs, g2, ROW } = await dataset(root);
  const man = await pack({ dataset: 'tiny', out, root, g2, shardMb: 0.0015, log: () => {} });
  assert.deepEqual(man.counts, { train: 2, val: 1, test: 1, ood: 0 }); assert.equal(man.jpeg_quality, 90);
  assert.equal(man.shards.filter((s) => s.split === 'train').length, 2, 'a shard rolls over at --shard-mb');
  const m0 = readTar(path.join(out, man.shards[0].path));
  assert.deepEqual(m0.map((m) => m.name), ['S_r_00000_000001.jpg', 'S_r_00000_000001.json']);
  const meta = await sharp(m0[0].data).metadata(); assert.deepEqual([meta.format, meta.width, meta.height], ['jpeg', 896, 504]);
  assert.equal(JSON.parse(m0[1].data).narrator_frame, `${man.shards[0].name}/S_r_00000_000001.jpg`); assert.equal(JSON.parse(m0[1].data).snapshot, undefined, 'the raw snapshot stays on LaCie');
  const row = JSON.parse(fs.readFileSync(path.join(out, 'narrator/train.jsonl'), 'utf8').split('\n')[1]); assert.equal(row.images[0], `${man.shards[1].name}/S_r_00000_000002.jpg`);
  const joined = Buffer.concat(man.eye_parts.train.map((p) => fs.readFileSync(path.join(out, p)))); assert.equal(joined.length, 2 * ROW);
  assert.ok(joined.equals(fs.readFileSync(path.join(root, 'datasets/tiny/cache/pilot_eye_train.u8'))), 'the parts join back to the cache');
  const sums = fs.readFileSync(path.join(out, 'SHA256SUMS'), 'utf8').trim().split('\n').map((l) => l.split(/\s+/));
  for (const [h, f] of sums) assert.equal(sha(fs.readFileSync(path.join(out, f))), h, f);
  for (const f of ['manifest.json', 'README.md', 'code/vlm/train/colab/pack_io.py', 'code/processor/tokenizer.json', 'code/baselines/g2_parity.json', 'code/package.json', 'splits.json']) assert.ok(fs.existsSync(path.join(out, f)), f);
  assert.deepEqual(JSON.parse(fs.readFileSync(path.join(out, 'code/baselines/g2_parity.json'), 'utf8')).c, { q4f16: { ours: 0.63, published: 0.63 } });
  assert.ok(man.files.includes('narrator/train.jsonl') && !man.files.some((f) => f.startsWith('shards/')));
  await assert.rejects(pack({ dataset: 'tiny', out, root, g2, log: () => {} }), /not empty/);
  await assert.rejects(pack({ dataset: 'tiny', out: path.join(root, 'p2'), root, g2, maxGb: 1e-6, log: () => {} }), /GB/);
  const s = await pack({ dataset: 'tiny', out: path.join(root, 'p3'), root, g2, sample: 2, log: () => {} }); assert.equal(s.records, 2);
  fs.rmSync(root, { recursive: true, force: true });
});
