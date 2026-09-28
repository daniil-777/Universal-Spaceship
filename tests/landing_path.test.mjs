import { test } from 'node:test';
import assert from 'node:assert/strict';
import { dubins, samplePath } from '../src/landing/dubins.js';
import { mulberry32 } from '../src/mathx.js';

const DEG = Math.PI / 180, wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));

test('Dubins: the path reaches the goal pose exactly, whatever the start (runway frame: heading atan2(z, x))', () => {
  const rng = mulberry32(5);
  for (let i = 0; i < 300; i++) {
    const a = { x: (rng() - 0.5) * 30000, z: (rng() - 0.5) * 30000, hdg: (rng() * 2 - 1) * Math.PI }, b = { x: (rng() - 0.5) * 30000, z: (rng() - 0.5) * 30000, hdg: (rng() * 2 - 1) * Math.PI };
    const R = 800 + rng() * 2000, d = dubins(a, b, R), pts = samplePath(d, 5), e = pts[pts.length - 1];
    assert.ok(Math.hypot(e.x - b.x, e.z - b.z) < 1 && Math.abs(wrap(e.hdg - b.hdg)) < 0.2 * DEG, `${d.word}: end off by ${Math.hypot(e.x - b.x, e.z - b.z).toFixed(2)} m, ${(wrap(e.hdg - b.hdg) / DEG).toFixed(2)}°`);
    assert.ok(Math.abs(pts[pts.length - 1].s - d.length) < 1e-6 && d.length >= Math.hypot(b.x - a.x, b.z - a.z) - 1e-6);
    for (let k = 1; k < pts.length; k++) assert.ok(Math.hypot(pts[k].x - pts[k - 1].x, pts[k].z - pts[k - 1].z) <= 5 + 1e-6);   // continuous
  }
});

test('Dubins: aligned poses give the straight line; a U-turn gives a half-circle; turns never tighter than R', () => {
  const s = dubins({ x: 0, z: 0, hdg: 0 }, { x: 5000, z: 0, hdg: 0 }, 1000); assert.ok(Math.abs(s.length - 5000) < 1e-6, s.word);
  const u = dubins({ x: 0, z: 0, hdg: 0 }, { x: 0, z: 2000, hdg: Math.PI }, 1000); assert.ok(Math.abs(u.length - Math.PI * 1000) < 1e-6, u.word + ' ' + u.length);
  const pts = samplePath(dubins({ x: 0, z: 0, hdg: 0 }, { x: 3000, z: -4000, hdg: 2 }, 1200), 2);
  for (let k = 2; k < pts.length; k++) { const dh = Math.abs(wrap(pts[k].hdg - pts[k - 1].hdg)), ds = pts[k].s - pts[k - 1].s; assert.ok(dh <= ds / 1200 + 1e-6, 'curvature'); }
});

test('Dubins: a right turn goes toward +z (the runway frame\'s right)', () => {
  const d = dubins({ x: 0, z: 0, hdg: 0 }, { x: 0, z: 2000, hdg: Math.PI }, 1000), pts = samplePath(d, 10), mid = pts[Math.floor(pts.length / 2)];
  assert.ok(mid.z > 500 && mid.x > 500, `${d.word}: the middle of the U-turn at (${mid.x.toFixed(0)}, ${mid.z.toFixed(0)})`); assert.ok(pts.every((p) => p.z > -1));
});
