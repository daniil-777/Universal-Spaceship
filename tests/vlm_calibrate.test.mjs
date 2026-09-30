import { test } from 'node:test';
import assert from 'node:assert/strict';
import { chooseCNear, mixAt } from '../vlm/gen/calibrate.mjs';

const st = (group, pRef, medClr, extra = {}) => ({ group, pRef, medClr, unsafeNow: false, turbSevere: false, ...extra });
test('mixAt applies the corridor rules to stored rollout summaries', () => {
  const s = [st('S', 0, 10), st('S', 0, 2), st('S', 0.25, 10), st('S', 0.5, 10), st('S', 0, 10, { unsafeNow: true })];
  assert.deepEqual(mixAt(s, 2.5), { SAFE: 1, CAUTION: 2, UNSAFE: 2, n: 5 });
  assert.deepEqual(mixAt(s, 1.5), { SAFE: 2, CAUTION: 1, UNSAFE: 2, n: 5 });
});
test('chooseCNear picks the admissible value closest to 2.5, or null when none is admissible', () => {
  const mk = (g, clrs) => clrs.map((c) => st(g, 0, c));
  const states = [...mk('S', [1, 2, 3, 3.5, 5, 6, 7, 8, 9, 10]), ...mk('A_search', [1.6, 2.2, 3.2, 3.9, 6, 7, 8, 9, 10, 11])];
  const r = chooseCNear(states); assert.ok(r.c_near >= 1.5 && r.c_near <= 4); assert.equal(r.grid.length, 11);
  const row = r.grid.find((g) => g.c === r.c_near); assert.ok(row.S >= 0.15 && row.S <= 0.45 && row.A_search >= 0.15 && row.A_search <= 0.45);
  assert.equal(chooseCNear(mk('S', [0.1, 0.2]).concat(mk('A_search', [0.1, 0.2]))).c_near, null);
});
