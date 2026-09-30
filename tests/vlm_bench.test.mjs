// vlm/web/bench.mjs (spec §10.2 (b)/(d), §10.3; G2): helpers and gates, the CLI guard, and — when the exported G2 folder
// is on LaCie — transformers.js loading that folder and generating a few tokens on the CPU.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { maxAbsDiff, perToken, meanRowCos, gateFailures, visionVariants, loadNarrator, prepare } from '../vlm/web/bench.mjs';

const ROOT = fileURLToPath(new URL('..', import.meta.url));
const FOLDER = process.env.APV_NARRATOR_DIR || '/Volumes/LaCie/astro-pilot/vlm/models/narrator-base-g2-fused';

test('maxAbsDiff is the largest absolute difference and rejects a length mismatch', () => {
  assert.equal(maxAbsDiff(new Float32Array([0, 1, -2]), new Float32Array([0.5, 1, -1.75])), 0.5);
  assert.throws(() => maxAbsDiff(new Float32Array(3), new Float32Array(4)), /length/);
});

test('perToken spreads the n-token time after the first token over the n - 1 decode steps', () => {
  assert.equal(perToken({ ttft: 250, total: 640, n: 40 }), 10);
});

test('meanRowCos averages the cosine of each d-wide row', () => {
  const a = new Float32Array([1, 0, 0, 1]), b = new Float32Array([2, 0, 1, 0]);
  assert.equal(meanRowCos(a, b, 2), 0.5);
});

test('gates: per-token latency on WebGPU only, image features at mean cos >= 0.99 on any device, and NaN never passes', () => {
  const ok = { ms_per_token: 11.9, device: 'webgpu', maxMsPerToken: 12, vision_cos: 0.999, minCos: 0.99 };
  assert.deepEqual(gateFailures(ok), []);
  assert.equal(gateFailures({ ...ok, ms_per_token: 50.2 }).length, 1);
  assert.deepEqual(gateFailures({ ...ok, ms_per_token: 50.2, device: 'cpu' }), []);
  assert.match(gateFailures({ ...ok, vision_cos: 0.47 })[0], /vision/);
  assert.equal(gateFailures({ ...ok, ms_per_token: NaN }).length, 1);
  assert.equal(gateFailures({ ...ok, vision_cos: NaN }).length, 1);
});

test('visionVariants lists the vision_encoder dtypes present, in transformers.js dtype names', () => {
  const names = ['vision_encoder_q4.onnx', 'vision_encoder_quantized.onnx', 'vision_encoder_fp16.onnx', 'vision_encoder.onnx', 'decoder_model_merged_q4.onnx', '._vision_encoder_int8.onnx'];
  assert.deepEqual(visionVariants(names), ['fp32', 'fp16', 'q8', 'q4']);
});

test('the CLI runs only as the entry script: run from the repo without a command it prints usage and exits 2', () => {
  const r = spawnSync(process.execPath, ['vlm/web/bench.mjs'], { cwd: ROOT, encoding: 'utf8' });
  assert.equal(r.status, 2); assert.match(r.stdout, /usage: bench\.mjs narrator .*--decoder q4f16\|q4\|fp16\|fp32.*--vision q8\|q4\|int8\|fp16\|all/);
});

test('the exported Narrator folder loads in transformers.js and generates 4 tokens (CPU, default dtypes)', { skip: !fs.existsSync(path.join(FOLDER, 'parity.json')) && `no exported folder at ${FOLDER}`, timeout: 180000 }, async () => {
  const { tf, processor, model, parity } = await loadNarrator({ modelDir: FOLDER, device: 'cpu' });
  try {
    const { inputs, ids } = await prepare(tf, processor, FOLDER, parity.samples[0]);
    assert.equal(ids.length, parity.d.prompt_len[0]);
    const out = await model.generate({ ...inputs, max_new_tokens: 4, min_new_tokens: 4, do_sample: false, eos_token_id: 49279 });
    assert.equal(out.dims.at(-1) - ids.length, 4);
  } finally {
    await model.dispose();
  }
});
