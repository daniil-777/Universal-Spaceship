// vlm/web/bench.mjs helpers (spec §10.2 (b)/(d), §10.3; G2 round 2): pixel_values agree within 1e-2, decode ms/token from
// TTFT and an n-token run, image features compared per row on the device, and the bench gates.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { maxAbsDiff, perToken, meanRowCos, gateFailures } from '../vlm/web/bench.mjs';

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

test('gates: per-token latency on WebGPU only, and image features at mean cos >= 0.99 on any device', () => {
  const ok = { ms_per_token: 11.9, device: 'webgpu', maxMsPerToken: 12, vision_cos: 0.999, minCos: 0.99 };
  assert.deepEqual(gateFailures(ok), []);
  assert.equal(gateFailures({ ...ok, ms_per_token: 50.2 }).length, 1);
  assert.deepEqual(gateFailures({ ...ok, ms_per_token: 50.2, device: 'cpu' }), []);
  assert.match(gateFailures({ ...ok, vision_cos: 0.47 })[0], /vision/);
});
