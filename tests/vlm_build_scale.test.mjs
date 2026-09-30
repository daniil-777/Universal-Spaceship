// The build scale-up (v1): multi-index-hash dedupe, the one-pass frame job in worker threads, the per-record derived-facts
// cache, streamed JSONL/JSON writers and the parallel Pilot Eye cache. Each must give the same bytes as the v0 code path.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { createHash } from 'node:crypto';
import { crossSplitDrops, chunksOf, hamming, dhash, dhashOf } from '../vlm/gen/build/dedupe.js';
import { imageFacts } from '../vlm/gen/imagefacts.js';
import { boxResize } from '../vlm/gen/boxresize.js';
import { frameJob } from '../vlm/gen/build/frames.js';
import { createPool } from '../vlm/gen/build/pool.js';
import { openFactCache, codeShaOf } from '../vlm/gen/build/factcache.js';
import { writeJsonl, writeJsonArray, writeCache, writeCacheParallel } from '../vlm/gen/build/export.js';
import { mulberry32 } from '../src/mathx.js';
const sharp = createRequire(new URL('../vlm/package.json', import.meta.url))('sharp');
const tmp = (p) => fs.mkdtempSync(path.join(os.tmpdir(), p));

// the v0 all-pairs scan, kept here as the reference
function bruteDrops(hashes, splits, maxDist = 4) {
  const keys = [...hashes.keys()], drop = new Set();
  for (let i = 0; i < keys.length; i++) for (let j = i + 1; j < keys.length; j++) {
    const a = keys[i], b = keys[j];
    if (splits.get(a) === splits.get(b) || hamming(hashes.get(a), hashes.get(b)) > maxDist) continue;
    for (const k of [a, b]) if (splits.get(k) !== 'train') drop.add(k);
  }
  return drop;
}
const rand64 = (rng) => (BigInt(Math.floor(rng() * 2 ** 32)) << 32n) | BigInt(Math.floor(rng() * 2 ** 32));
const flip = (h, rng, n) => { for (let i = 0; i < n; i++) h ^= 1n << BigInt(Math.floor(rng() * 64)); return h; };

test('dedupe: multi-index hashing drops exactly what the all-pairs scan drops (clusters, exact duplicates, every distance)', () => {
  for (const seed of [1, 2, 3, 4, 5, 6]) for (const maxDist of [0, 2, 4, 7]) {
    const rng = mulberry32(seed * 101 + maxDist), hashes = new Map(), splits = new Map(), SPL = ['train', 'train', 'train', 'val', 'test', 'ood'];
    const centres = [...Array(12)].map(() => rand64(rng));
    for (let i = 0; i < 400; i++) {
      const k = `k${i}`, c = centres[Math.floor(rng() * centres.length)];
      hashes.set(k, rng() < 0.15 ? rand64(rng) : flip(c, rng, Math.floor(rng() * 9)));
      splits.set(k, SPL[Math.floor(rng() * SPL.length)]);
    }
    assert.deepEqual([...crossSplitDrops(hashes, splits, { maxDist })].sort(), [...bruteDrops(hashes, splits, maxDist)].sort(), `seed ${seed} maxDist ${maxDist}`);
  }
  const h = new Map([['a', 0n], ['b', 0x7n], ['c', 0xffffn], ['d', 0x1n]]), s = new Map([['a', 'train'], ['b', 'val'], ['c', 'test'], ['d', 'train']]);
  assert.deepEqual([...crossSplitDrops(h, s)].sort(), ['b'], 'the v0 case');
  const same = new Map([...Array(50)].map((_, i) => [`x${i}`, 42n])), ss = new Map([...same.keys()].map((k, i) => [k, i === 0 ? 'train' : i % 2 ? 'val' : 'test']));
  assert.equal(crossSplitDrops(same, ss).size, 49, 'a flood of identical frames: every non-train member leaves');
});
test('dedupe: the chunks cover all 64 bits once', () => {
  for (const parts of [1, 3, 5, 8]) {
    const h = 0xfedcba9876543210n, w = Math.ceil(64 / parts), c = chunksOf(h, parts);
    assert.equal(c.length, parts); assert.equal(c.reduce((acc, v, i) => acc | (BigInt(v) << BigInt(i * w)), 0n), h);
  }
});

async function frameFiles(dir, channels = 4) {
  const W = 896, H = 504, files = [];
  for (let f = 0; f < 4; f++) {
    const px = Buffer.alloc(W * H * channels);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { const o = (y * W + x) * channels; px[o] = (x * 7 + f * 31) & 255; px[o + 1] = ((x >> 3) + (y >> 2) * 5) & 255; px[o + 2] = (y * 3 + f * 17) & 255; if (channels === 4) px[o + 3] = 255; }
    const p = path.join(dir, `k.f${f}.png`); await sharp(px, { raw: { width: W, height: H, channels } }).png().toFile(p); files.push(p);
  }
  return files;
}
const eyeOld = async (file) => { const { data, info } = await sharp(file).ensureAlpha().raw().toBuffer({ resolveWithObject: true }); return boxResize(data, info.width, info.height, 160, 96, info.channels); };

test('frame job: one decode per PNG gives the v0 image facts, dHash and Pilot Eye rows (RGBA and RGB frames)', async () => {
  for (const ch of [4, 3]) {
    const dir = tmp('apv-fj-'), [f0, f1, f2, chase] = await frameFiles(dir, ch), rel = (p) => path.basename(p);
    const r = await frameJob({ root: dir, narrator: rel(chase), frames: [rel(f0), rel(f1), rel(f2)], want: { facts: true, hash: true, eye: true, files: true }, eyeSize: [160, 96] });
    const sha = (p) => createHash('sha256').update(fs.readFileSync(p)).digest('hex');
    assert.deepEqual(r.files, [sha(f0), sha(f1), sha(f2)], 'the SHA-256 of each frame file, for the pixel-identical pair check');
    assert.deepEqual(r.facts, await imageFacts(chase), `facts (${ch} channels)`);
    assert.equal(r.hash, (await dhash(f2)).toString(16), `dHash of the last frame (${ch} channels)`);
    const want = Buffer.concat([await eyeOld(f0), await eyeOld(f1), await eyeOld(f2)]);
    assert.ok(Buffer.from(r.eye).equals(want), `eye rows (${ch} channels)`);
    const z = await frameJob({ root: dir, narrator: rel(f0), frames: [rel(f0)], want: { eye: true }, eyeSize: [160, 96] }), one = await eyeOld(f0);
    assert.ok(Buffer.from(z.eye).equals(Buffer.concat([one, one, one])), 'a one-frame (Z) record repeats its frame to 3 rows');
    assert.equal(z.facts, undefined); assert.equal(z.hash, undefined);
    const { data, info } = await sharp(f2).raw().toBuffer({ resolveWithObject: true });
    assert.equal(await dhashOf(sharp(data, { raw: { width: info.width, height: info.height, channels: info.channels } })), await dhash(f2));
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
test('pool: jobs run in worker threads, results come back in order, a failing job rejects', async () => {
  const dir = tmp('apv-pool-'), mod = path.join(dir, 'job.mjs');
  fs.writeFileSync(mod, "import { threadId } from 'node:worker_threads';\nexport async function job(t) { if (t.bad) throw new Error('bad ' + t.i); await new Promise((r) => setTimeout(r, (7 * t.i) % 5)); const b = new Uint8Array([t.i]); return { result: { i: t.i * 2, thread: threadId, b }, transfer: [b.buffer] }; }\n");
  const pool = createPool(mod, 3);
  try {
    const out = await Promise.all([...Array(20)].map((_, i) => pool.run({ i })));
    assert.deepEqual(out.map((o) => o.i), [...Array(20)].map((_, i) => i * 2)); assert.ok(out.every((o) => o.thread > 0), 'not the main thread');
    assert.deepEqual(out.map((o) => o.b[0]), [...Array(20)].map((_, i) => i), 'a transferred buffer arrives');
    await assert.rejects(pool.run({ i: 3, bad: true }), /bad 3/);
    assert.equal((await pool.run({ i: 4 })).i, 8, 'the pool keeps working after a failed job');
  } finally { await pool.close(); fs.rmSync(dir, { recursive: true, force: true }); }
});
test('fact cache: a hit needs the same key, code sha and record stamp; entries survive a reopen; the last write wins', () => {
  const dir = tmp('apv-fc-');
  let c = openFactCache(dir, 'abc123');
  assert.equal(c.get('S_r_00001_000010', '100:5'), undefined);
  c.put('S_r_00001_000010', '100:5', { hash: 'ff', facts: { 'image.mean_lum': { v: 0.5, unit: null, obs: 'visual' } } }); c.put('S_r_00001_000011', '7:7', { hash: '1' });
  c.put('S_r_00001_000011', '7:8', { hash: '2' }); c.flush();
  c = openFactCache(dir, 'abc123');
  assert.equal(c.get('S_r_00001_000010', '100:5').hash, 'ff'); assert.equal(c.get('S_r_00001_000010', '100:6'), undefined, 'a rewritten record misses');
  assert.equal(c.get('S_r_00001_000011', '7:8').hash, '2'); assert.equal(c.size, 2);
  assert.equal(openFactCache(dir, 'other').get('S_r_00001_000010', '100:5'), undefined, 'new build code misses');
  const a = path.join(dir, 'a.js'), b = path.join(dir, 'b.js'); fs.writeFileSync(a, 'x'); fs.writeFileSync(b, 'y');
  const s1 = codeShaOf([a, b]); fs.writeFileSync(b, 'z'); assert.notEqual(codeShaOf([a, b]), s1); assert.match(s1, /^[0-9a-f]{12}$/);
  fs.rmSync(dir, { recursive: true, force: true });
});
test('streamed writers: JSONL and a JSON array are byte-identical to the v0 whole-string writes', () => {
  const dir = tmp('apv-wr-'), rows = [...Array(2503)].map((_, i) => ({ i, s: 'é "q"', a: [i, { x: null }] }));
  writeJsonl(path.join(dir, 'a.jsonl'), rows); assert.equal(fs.readFileSync(path.join(dir, 'a.jsonl'), 'utf8'), rows.map((r) => JSON.stringify(r)).join('\n') + '\n');
  writeJsonl(path.join(dir, 'e.jsonl'), []); assert.equal(fs.readFileSync(path.join(dir, 'e.jsonl'), 'utf8'), '');
  writeJsonArray(path.join(dir, 'b.json'), rows); assert.equal(fs.readFileSync(path.join(dir, 'b.json'), 'utf8'), JSON.stringify(rows));
  writeJsonArray(path.join(dir, 'c.json'), []); assert.equal(fs.readFileSync(path.join(dir, 'c.json'), 'utf8'), '[]');
  fs.rmSync(dir, { recursive: true, force: true });
});
test('parallel Pilot Eye cache: the same bytes and index as the sequential v0 writer', async () => {
  const dir = tmp('apv-pc-'), files = await frameFiles(dir, 4), rel = (p) => path.basename(p);
  const recs = [{ key: 'a', frames: files.slice(0, 3).map(rel) }, { key: 'b', frames: [rel(files[3])] }, { key: 'c', frames: [rel(files[2]), rel(files[0]), rel(files[1])] }];
  const i1 = await writeCache(path.join(dir, 'v0'), 'train', recs, { root: dir, size: [160, 96] });
  const pool = createPool(new URL('../vlm/gen/build/frames.js', import.meta.url), 2);
  try {
    const i2 = await writeCacheParallel(path.join(dir, 'v1'), 'train', recs, pool, { root: dir, size: [160, 96], window: 2 });
    assert.deepEqual(i2, i1); assert.ok(fs.readFileSync(path.join(dir, 'v1/cache/pilot_eye_train.u8')).equals(fs.readFileSync(path.join(dir, 'v0/cache/pilot_eye_train.u8'))));
    const have = new Map([['b', (await frameJob({ root: dir, frames: recs[1].frames, want: { eye: true } })).eye]]);
    const i3 = await writeCacheParallel(path.join(dir, 'v2'), 'train', recs, pool, { root: dir, size: [160, 96], rows: have });
    assert.deepEqual(i3, i1); assert.ok(fs.readFileSync(path.join(dir, 'v2/cache/pilot_eye_train.u8')).equals(fs.readFileSync(path.join(dir, 'v0/cache/pilot_eye_train.u8'))), 'rows given in a Map are written as they are');
  } finally { await pool.close(); fs.rmSync(dir, { recursive: true, force: true }); }
});
