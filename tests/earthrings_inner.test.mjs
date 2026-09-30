import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import { innerWindow, innerState, ringWindow, ringLevels, RING_TILES } from '../src/earthtiles.js';

// Fix A (diagnosis 2026-09-29 §1(a)): a coarser ring never depth-competes with the next finer ring. The window maths
// (where the finer ring sits in the coarser ring's uv) moved out of earthrings.js into a pure function; these are the
// values earthrings.js computed inline before.
const close = (a, b, eps, msg = '') => assert.ok(Math.abs(a - b) <= eps, `${msg} ${a} vs ${b}`);
const oldFormula = (i, o) => { const s = 2 ** (i.level - o.level); return { scale: s, min: [(i.x0 / s - o.x0) / RING_TILES, (i.y0 / s - o.y0) / RING_TILES] }; };

test('innerWindow: the finer ring in the coarser ring\'s uv, as the old inline formula', () => {
  const cases = [
    [{ level: 15, x0: 1000, y0: 500 }, { level: 14, x0: 497, y0: 246 }, 2, [0.375, 0.5]],
    [{ level: 15, x0: 1001, y0: 503 }, { level: 14, x0: 497, y0: 246 }, 2, [0.4375, 0.6875]],
    [{ level: 16, x0: 2000, y0: 1000 }, { level: 14, x0: 497, y0: 246 }, 4, [0.375, 0.5]],
  ];
  for (const [inner, outer, scale, min] of cases) {
    const w = innerWindow(inner, outer), o = oldFormula(inner, outer);
    assert.equal(w.scale, scale); assert.equal(w.scale, o.scale);
    close(w.min[0], min[0], 1e-12, 'min x'); close(w.min[1], min[1], 1e-12, 'min y');
    close(w.min[0], o.min[0], 1e-12, 'old x'); close(w.min[1], o.min[1], 1e-12, 'old y');
  }
});

// At level 2 the world is 4 tiles wide but a ring is 8: its mesh wraps the globe twice, and the copy outside the finer
// ring's uv window must still count as under it (else it keeps its true depth and hides the pushed rings: 3000 km bands).
test('innerWindow: the world\'s width in the coarser ring\'s uv, so the under test wraps in longitude', () => {
  // ringWindow at tile x 2.5 of level 2 (x0 −2) and of level 4 (x0 6): the finer ring spans uv 0.4375..0.6875 and again
  // one world (0.5) to the west
  const w = innerWindow({ level: 4, x0: 6, y0: 6 }, { level: 2, x0: -2, y0: -2 }), iu = (u) => (((u - w.min[0]) % w.wrap) + w.wrap) % w.wrap * w.scale;
  assert.equal(w.wrap, 0.5); assert.equal(w.scale, 4); close(w.min[0], 0.4375, 1e-12);
  assert.equal(innerWindow({ level: 5, x0: 12, y0: 2 }, { level: 3, x0: 2, y0: -2 }).wrap, 1);
  assert.equal(innerWindow({ level: 15, x0: 1000, y0: 500 }, { level: 14, x0: 497, y0: 246 }).wrap, 2 ** 14 / RING_TILES);
  for (const u of [0.5, 0.55, 0.6]) {
    close(iu(u), iu(u - w.wrap), 1e-12, 'both copies of the level-2 ring');
    assert.ok(iu(u) > 0 && iu(u) < 1 && iu(u - w.wrap) > 0 && iu(u - w.wrap) < 1, 'both copies lie under the finer ring');
  }
  assert.ok(iu(0.3) > 1, 'west of the finer ring is not under it');
});

// The FRAG's under test along x, with the wrap (GLSL mod) and without it (the brief's formula): whether a coarse-ring uv
// is under the finer ring, and where (iu.x, which also decides the interior hole).
const glslMod = (a, b) => a - b * Math.floor(a / b);
const underX = (u, w, wrap) => { const x = (wrap ? glslMod(u - w.min[0], w.wrap) : u - w.min[0]) * w.scale; return x > 0 && x < 1 ? x : null; };
test('innerWindow: from level 3 up the longitude wrap changes no under / footprint result (ringWindow pairs, antimeridian)', () => {
  const lons = [-179.999, -179.9, -120, -0.001, 0, 0.001, 45, 139.76, 179.9, 179.999], lats = [-84, -45, 0, 35.68, 45.976, 84];
  const pairs = [];
  for (let c = 3; c <= 17; c++) for (const f of [c + 1, c + 2]) if (f <= 18) pairs.push([c, f]);
  for (let L0 = 4; L0 <= 18; L0++) { const ls = ringLevels(L0); for (const c of ls) { const fin = ls.filter((l) => l > c); if (fin.length) pairs.push([c, Math.min(...fin)]); } }
  let checked = 0, level2Differs = 0;
  for (const [c, f] of pairs) for (const lon of lons) for (const lat of lats) {
    const w = innerWindow(ringWindow(lon, lat, f), ringWindow(lon, lat, c));
    for (let k = 0; k <= 256; k++) {
      const u = k / 256, a = underX(u, w, true), b = underX(u, w, false);
      if (c < 3) { if (a !== b) level2Differs++; continue; }
      checked++;
      if (a === null || b === null) assert.equal(a, b, `level ${c}->${f} lon ${lon} lat ${lat} u ${u}: under ${a !== null} with the wrap, ${b !== null} without`);
      else close(a, b, 1e-9, `level ${c}->${f} lon ${lon} lat ${lat} u ${u}: iu.x`);
    }
  }
  assert.ok(checked > 100000, `checked ${checked} samples`);
  assert.ok(level2Differs > 0, 'the same check does see the wrap at level 2 (the test can fail)');
});

test('innerState: on while the finer ring is drawn; done only once its imagery AND its relief are fully in', () => {
  assert.deepEqual(innerState({ vis: 0.5, hk: 0 }), { on: true, done: false });
  assert.deepEqual(innerState({ vis: 1, hk: 0.5 }), { on: true, done: false }, 'K2: no hole before the relief is up');
  assert.deepEqual(innerState({ vis: 1, hk: 1 }), { on: true, done: true });
  assert.deepEqual(innerState({ vis: 0.001, hk: 1 }), { on: false, done: false });
});

test('earthrings.js: the fragment shader writes gl_FragDepth and has uInnerDone; the vis > 0.98 hole gate is gone', () => {
  const src = fs.readFileSync(fileURLToPath(new URL('../src/earthrings.js', import.meta.url)), 'utf8');
  const frag = (src.match(/const FRAG = \/\* glsl \*\/`([\s\S]*?)`;/) || [])[1];
  assert.ok(frag, 'FRAG found');
  assert.match(frag, /gl_FragDepth\s*=/); assert.match(frag, /uniform[^;]*\buInnerDone\b/); assert.match(frag, /uInnerDone\s*>\s*0\.5/);
  assert.doesNotMatch(src, /inner\.vis\s*>\s*0\.98/);
  assert.match(src, /innerWindow\(/); assert.match(src, /innerState\(/);
  assert.match(src, /uInnerDone:\s*\{\s*value:\s*0\s*\}/, 'makeRing declares the uniform');
  assert.match(frag, /mod\(vUv\.x - uInnerMin\.x, uInnerWrap\)/, 'the under test wraps in longitude');
  assert.match(src, /uInnerWrap\.value = w\.wrap/, 'the update loop sets the wrap');
  assert.match(frag, /gl_FragDepth = !under \? gl_FragCoord\.z : \(uInnerDone > 0\.5 \? mix\(gl_FragCoord\.z, 1\.0, 0\.2\) : 0\.99999\);/, 'far push while the finer ring fades in, a fifth of the way once it is done (the sail)');
});
