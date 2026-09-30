import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRing } from '../vlm/capture/probe/ring.js';
import { fnv1a32, renderSeed, scanRun, writeEpisode } from '../vlm/capture/drive.mjs';
import { shrink, SIZES } from '../vlm/capture/report.mjs';

test('REVIEW FOCUS 3: ring drops frames across a reset or a crash', () => {
  const r = createRing(3); assert.equal(r.push({ step: 10 }), null); assert.equal(r.push({ step: 13 }), null);
  assert.equal(r.push({ step: 2 }), null, 'a smaller step (env.reset after a crash) clears the ring'); assert.equal(r.size, 1);
  r.push({ step: 5 }); assert.equal(r.push({ step: 8, reset: true }), null); assert.equal(r.size, 1);
  r.push({ step: 11 }); const clip = r.push({ step: 14 }); assert.deepEqual(clip.map((f) => f.step), [8, 11, 14]);
});
test('render_seed depends only on run, family and episode seed (FNV-1a 32)', () => {
  assert.equal(fnv1a32('a'), 0xe40c292c); assert.equal(renderSeed('apv0', 'S', 101), renderSeed('apv0', 'S', 101)); assert.notEqual(renderSeed('apv0', 'S', 101), renderSeed('apv0', 'S', 102));
});
test('REVIEW FOCUS 4: resume scan removes unfinished episodes and counts only finished ones', () => {
  const d = fs.mkdtempSync(path.join(os.tmpdir(), 'apv-raw-')), png = Buffer.from('89504e47', 'hex');
  writeEpisode(d, [{ key: 'S_r_00001_000100', frames: ['S_r_00001_000100.f0.png', 'S_r_00001_000100.f1.png', 'S_r_00001_000100.f2.png'] }], { 'S_r_00001_000100.f0.png': png, 'S_r_00001_000100.f1.png': png, 'S_r_00001_000100.f2.png': png }, 1);
  fs.writeFileSync(path.join(d, 'S_r_00002_000050.f0.png'), png); fs.writeFileSync(path.join(d, 'S_r_00002_000050.json'), '{}');
  const s = scanRun(d);
  assert.equal(s.samples, 1); assert.deepEqual([...s.episodes], [1]); assert.deepEqual(s.removed.sort(), ['S_r_00002_000050.f0.png', 'S_r_00002_000050.json']);
  assert.ok(!fs.existsSync(path.join(d, 'S_r_00002_000050.f0.png')));
});
test('R17 shrink: within budget the sizes stay; over budget the over-share families shrink first, never below 50 %', () => {
  const fast = { Z: 1, S: 1, A: 1, L: 1, D: 1 }, a = shrink(fast, 12);
  assert.deepEqual(a.sizes, SIZES); assert.ok(a.hours < 12);
  const slowZ = { Z: 40, S: 5, A: 5, L: 5, D: 5 }, b = shrink(slowZ, 12);
  assert.ok(b.hours <= 12, `projected ${b.hours} h`); assert.ok(b.sizes.Z < SIZES.Z && b.sizes.Z >= SIZES.Z / 2); assert.equal(b.sizes.S, SIZES.S); assert.equal(b.sizes.A, SIZES.A);
  const hopeless = { Z: 600, S: 600, A: 600, L: 600, D: 600 }, c = shrink(hopeless, 12);
  for (const f of Object.keys(SIZES)) assert.equal(c.sizes[f], Math.ceil(SIZES[f] / 2)); assert.ok(c.hours > 12);
});
test('Z steady state: the zoom rings\' per-frame easing (earthrings.js, dt = 16 ms) ends at one fixed point from any start', () => {
  const e = 1 - Math.exp(-(16 / 1000) * 3), settle = (x) => { for (let n = 0; n < 5000; n++) { const y = x + (1 - x) * e; if (y === x) return x; x = y; } return NaN; };
  const ends = [0, 0.123, 0.5, 0.9, 0.99, 0.995, 0.9999999].map(settle);
  assert.ok(ends.every((x) => x === ends[0]), 'a cold and a warm run reach the same blend and relief factors'); assert.equal(ends[0], 1 - 10 * 2 ** -53);
});
