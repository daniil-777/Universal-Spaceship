import { test } from 'node:test';
import assert from 'node:assert/strict';
import { cloudNoiseData } from '../src/cloudnoise.js';

test('cloud noise: in range, varied, and seamless across the tile edges', () => {
  const N = 16, d = cloudNoiseData(N, 7), at = (x, y, z, c) => d[((((z + N) % N) * N + ((y + N) % N)) * N + ((x + N) % N)) * 4 + c];
  let min = 255, max = 0, inner = 0, seam = 0, n = 0;
  for (let z = 0; z < N; z++) for (let y = 0; y < N; y++) { for (let x = 0; x < N; x++) { const v = at(x, y, z, 0); min = Math.min(min, v); max = Math.max(max, v); }
    inner += Math.abs(at(1, y, z, 0) - at(0, y, z, 0)); seam += Math.abs(at(0, y, z, 0) - at(N - 1, y, z, 0)); n++; }
  assert.ok(max - min > 100, `range ${min}..${max}`); assert.ok(seam / n < 2.5 * (inner / n) + 2, `seam ${seam / n} vs inner ${inner / n}`);
});
