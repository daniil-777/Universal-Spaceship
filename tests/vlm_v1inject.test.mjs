import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mulberry32 } from '../src/mathx.js';
import { drawInjection, INJECT_RATE, INJECTIONS, INJECT_WEIGHTS } from '../vlm/gen/inject.js';
import { freeSlots, PRE_KICK } from '../vlm/capture/probe/frame.js';

const keyOf = (d) => d.kind + (d.params.band ? `:${d.params.band}` : '');
test('V1-3: L and D inject most episodes (UNSAFE and CAUTION >= 15 % each needs them); S and A keep the v0 rates', () => {
  assert.ok(INJECT_RATE.L >= 0.8 && INJECT_RATE.D >= 0.9); assert.equal(INJECT_RATE.S, 0.25); assert.equal(INJECT_RATE.A, 0.10);
  for (const f of ['L', 'D']) assert.equal(INJECT_WEIGHTS[f].length, INJECTIONS[f].length, `${f}: one weight per catalogue entry`);
});
test('V1-3: the kind draw follows INJECT_WEIGHTS (UNSAFE-producing kinds weighted up) with one rng draw, as in v0', () => {
  for (const f of ['L', 'D']) {
    const rng = mulberry32(3), seen = {}, W = INJECT_WEIGHTS[f], sum = W.reduce((a, b) => a + b, 0); let n = 0;
    for (let i = 0; i < 40000; i++) { const d = drawInjection(f, rng); if (d) { n++; seen[keyOf(d)] = (seen[keyOf(d)] || 0) + 1; } }
    INJECTIONS[f].forEach((e, i) => { const k = e.kind + (e.band ? `:${e.band}` : ''), got = (seen[k] || 0) / n; assert.ok(Math.abs(got - W[i] / sum) < 0.015, `${f} ${k}: ${got.toFixed(3)} vs ${(W[i] / sum).toFixed(3)}`); });
  }
  assert.ok(INJECT_WEIGHTS.L[INJECTIONS.L.findIndex((e) => e.kind === 'hflare')] >= 3 && INJECT_WEIGHTS.L[INJECTIONS.L.findIndex((e) => e.kind === 'lateral_25')] >= 3);
  assert.ok(INJECT_WEIGHTS.D[INJECTIONS.D.findIndex((e) => e.kind === 'closing_plus_0.2')] >= 3);
  // S/A: one entry, the same draw sequence as v0 (rate draw, then the index draw)
  const a = mulberry32(5), b = mulberry32(5); for (let i = 0; i < 200; i++) { const d = drawInjection('S', a); const hit = b() < INJECT_RATE.S; if (hit) b(); assert.equal(!!d, hit); if (d) b(); }
});
test('V1-3: while a runtime kick is pending, L keeps at most 4 clean samples before it and D none; otherwise the v0 slots', () => {
  assert.equal(freeSlots(12, 0, false), 12); assert.equal(freeSlots(12, 3, true), 8, 'v0: one slot kept back for the kick');
  assert.equal(freeSlots(12, 0, true, PRE_KICK.L), 4); assert.equal(freeSlots(12, 4, true, PRE_KICK.L), 0);
  assert.equal(freeSlots(12, 0, true, PRE_KICK.D), 0); assert.equal(freeSlots(12, 1, false, PRE_KICK.D), 11, 'after the kick the whole run is open again');
});
