import { test } from 'node:test';
import assert from 'node:assert/strict';
import { chooseCNear, mixAt } from '../vlm/gen/calibrate.mjs';
import { corridorSafety, CORRIDOR_ACTIONS } from '../vlm/gen/safety.js';

const st = (group, pRef, medClr, extra = {}) => ({ group, pRef, medClr, unsafeNow: false, turbSevere: false, ...extra });
test('mixAt applies the corridor rules to stored rollout summaries', () => {
  const s = [st('S', 0, 10), st('S', 0, 2), st('S', 0.25, 10), st('S', 0.5, 10), st('S', 0, 10, { unsafeNow: true }), st('S', 0, 10, { turbSevere: true })];
  assert.deepEqual(mixAt(s, 2.5), { SAFE: 1, CAUTION: 3, UNSAFE: 2, n: 6 });
  assert.deepEqual(mixAt(s, 1.5), { SAFE: 2, CAUTION: 2, UNSAFE: 2, n: 6 });
});
test('chooseCNear picks the admissible value closest to 2.5, or null when none is admissible', () => {
  const mk = (g, clrs) => clrs.map((c) => st(g, 0, c));
  const states = [...mk('S', [1, 2, 3, 3.5, 5, 6, 7, 8, 9, 10]), ...mk('A_search', [1.6, 2.2, 3.2, 3.9, 6, 7, 8, 9, 10, 11])];
  const r = chooseCNear(states); assert.ok(r.c_near >= 1.5 && r.c_near <= 4); assert.equal(r.grid.length, 11);
  const row = r.grid.find((g) => g.c === r.c_near); assert.ok(row.S >= 0.15 && row.S <= 0.45 && row.A_search >= 0.15 && row.A_search <= 0.45);
  assert.equal(chooseCNear(mk('S', [0.1, 0.2]).concat(mk('A_search', [0.1, 0.2]))).c_near, null);
});

// mixAt (calibrate.mjs) is a stripped, stored-summary reduction of the same rule as corridorSafety (safety.js): it only
// keeps CONTINUE's crash rate and median clearance plus a collapsed now-flag, not the other 6 actions' branches or the
// full now object corridorSafety's input contract needs. Deriving mixAt from corridorSafety directly is therefore
// impractical; this test instead pins that the two verdict rules agree on every case mixAt distinguishes.
function branchesFor(pRef, medClr) {
  const b = {};
  for (const a of CORRIDOR_ACTIONS) b[a] = { k: 4, crashes: 0, medClr: 1e9, minClr: 1e9, causes: [] };
  b.CONTINUE = { k: 4, crashes: Math.round(pRef * 4), medClr, minClr: medClr, causes: [] };
  return b;
}
function corridorInputFor(state) {
  return { branches: branchesFor(state.pRef, state.medClr), now: { overstressed: !!state.unsafeNow, stalled: false, pullUp: false, turbSevere: !!state.turbSevere } };
}
function mixAtVerdict(state, c) {
  const m = mixAt([state], c);
  return m.SAFE ? 'SAFE' : m.CAUTION ? 'CAUTION' : 'UNSAFE';
}
test('mixAt classification agrees with corridorSafety.verdict on shared fixtures', () => {
  const c = 2.5;
  const cases = [st('S', 0, 2), st('S', 0, 10), st('S', 0.25, 10), st('S', 0.5, 10), st('S', 0, 10, { unsafeNow: true }), st('S', 0, 10, { turbSevere: true })];
  for (const state of cases) assert.equal(mixAtVerdict(state, c), corridorSafety(corridorInputFor(state), { cNear: c }).verdict, JSON.stringify(state));
});
test('chooseCNear breaks a distance tie by preferring the lower admissible c (stable sort order)', () => {
  const states = [st('S', 0, 5), st('A_search', 0, 5)];
  const opts = { lo: 2, hi: 3, step: 0.5, target: [0, 1] };
  assert.equal(chooseCNear(states, { ...opts, start: 2.25 }).c_near, 2);
  assert.equal(chooseCNear(states, { ...opts, start: 2.75 }).c_near, 2.5);
});
