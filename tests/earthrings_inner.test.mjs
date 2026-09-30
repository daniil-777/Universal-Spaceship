import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { register } from 'node:module';
import { fileURLToPath } from 'node:url';
import { innerWindow, innerState, ringWindow, ringLevels, createLocalFrame, tileLevelOf, RING_TILES, MAX_LEVEL } from '../src/earthtiles.js';

// src/earthrings.js imports the bare specifier 'three', which Node only resolves from tests/ (tests/node_modules/three,
// r170). This inline resolve hook re-resolves 'three' and 'three/...' as if this test file had imported them.
register('data:text/javascript,' + encodeURIComponent(`export async function resolve(s, c, next) {
  return s === 'three' || s.startsWith('three/') ? next(s, { ...c, parentURL: ${JSON.stringify(import.meta.url)} }) : next(s, c); }`));

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
  for (let c = 3; c <= MAX_LEVEL - 1; c++) for (const f of [c + 1, c + 2]) if (f <= MAX_LEVEL) pairs.push([c, f]);
  for (let L0 = 4; L0 <= MAX_LEVEL; L0++) { const ls = ringLevels(L0); for (const c of ls) { const fin = ls.filter((l) => l > c); if (fin.length) pairs.push([c, Math.min(...fin)]); } }
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
  assert.match(src, /uInnerDone:\s*\{\s*value:\s*0\s*\}/, 'makeRing declares the uniform');
  assert.match(frag, /mod\(vUv\.x - uInnerMin\.x, uInnerWrap\)/, 'the under test wraps in longitude');
  assert.match(frag, /gl_FragDepth = !under \? gl_FragCoord\.z : \(uInnerDone > 0\.5 \? mix\(gl_FragCoord\.z, 1\.0, 0\.2\) : 0\.99999\);/, 'far push while the finer ring fades in, a fifth of the way once it is done (the sail)');
});

// --- The update loop's inner-ring wiring, driven for real (final review I1 + M3; replaces the regex pins on update()):
// createEarthRings in Node with a fake renderer and a fake loader that answers at once for the URLs it is told to.
function makeFakeRenderer() {
  const props = new Map(), ensure = (t) => { let p = props.get(t); if (!p) { p = { __webglTexture: {} }; props.set(t, p); } return p; };
  return { initTexture: ensure, copyTextureToTexture: (s, d) => { ensure(d); }, getContext: () => ({ TEXTURE_2D: 0x0de1, generateMipmap() {} }),
    properties: { get: ensure }, state: { bindTexture() {}, unbindTexture() {} } };
}
const isHeight = (url) => url.includes('terrarium');
function harness(lat, lon) {
  let delivers = () => true;
  const loader = { inFlight: 0, queued: 0, stats: {}, failRate: 0, request(job) { if (job.wanted() && delivers(job.url)) job.done({}); }, prune() {}, suspend() {}, resume() {} };
  return (async () => {
    const { createEarthRings } = await import('../src/earthrings.js');
    const meshes = [], rings = createEarthRings({ add: (m) => meshes.push(m) }, makeFakeRenderer(), { loader });
    rings.rebase(createLocalFrame(lat, lon));
    const u = (L) => meshes.find((m) => m.renderOrder - 2 === L).material.uniforms;
    return { rings, u, setDelivers: (f) => { delivers = f; } };
  })();
}
// The rule, as a model: each used ring defers to the NEAREST finer used ring that is on (vis > 0.003), and every inner
// uniform comes from that same ring; with none on, the ring is not under anything.
function checkWiring(h, view, frame) {
  const want = ringLevels(view.L0), vis = (L) => h.u(L).uVis.value / view.vis, hk = (L) => h.u(L).uHeightK.value;
  for (const L of want) {
    const u = h.u(L), on = want.filter((F) => F > L && innerState({ vis: vis(F), hk: hk(F) }).on).sort((a, b) => a - b), F = on[0], tag = `frame ${frame}, L${L}`;
    if (F === undefined) { assert.equal(u.uInnerOn.value, 0, `${tag}: nothing finer on`); assert.equal(u.uInnerVis.value, 0, tag); continue; }
    const st = innerState({ vis: vis(F), hk: hk(F) }), w = innerWindow(ringWindow(view.lon, view.lat, F), ringWindow(view.lon, view.lat, L)), f = h.u(F);
    assert.equal(u.uInnerOn.value, 1, `${tag}: under L${F}`); assert.equal(u.uInnerDone.value, st.done ? 1 : 0, `${tag}: done as L${F}`);
    assert.equal(u.uInnerVis.value, vis(F), `${tag}: uInnerVis = L${F}'s fade-in`); assert.equal(u.tInner.value, f.tMask.value, `${tag}: tInner = L${F}'s mask`);
    assert.ok(u.uInnerCOff.value.equals(f.uCOff.value), `${tag}: uInnerCOff = L${F}'s atlas offset`);
    assert.equal(u.uInnerScale.value, w.scale, `${tag}: scale`); assert.equal(u.uInnerWrap.value, w.wrap, `${tag}: wrap`);
    assert.ok(Math.abs(u.uInnerMin.value.x - w.min[0]) < 1e-9 && Math.abs(u.uInnerMin.value.y - w.min[1]) < 1e-9, `${tag}: uInnerMin`);
  }
}

test('update(): every used ring\'s inner uniforms come from the nearest finer used ring that is on, per ring per frame', async () => {
  const lat = 35.68, lon = 139.76, h = await harness(lat, lon), view = { L0: 15, lat, lon, vis: 1, sun: { x: 0, y: 1, z: 0 }, hazeK: 0.55, hazeL: 40 };
  for (let f = 0; f < 150; f++) { h.rings.update(0.05, view); checkWiring(h, view, f); }
  for (const L of ringLevels(15)) assert.ok(h.u(L).uVis.value > 0.99, `L${L} fully in before the step`);
  // a 2-level zoom step (L0 15 -> 17): the loader brings the finest ring (17) in, the one between (16) stays empty
  view.L0 = 17; h.setDelivers((url) => isHeight(url) || tileLevelOf(url) !== 16);
  let deferred = 0;
  for (let f = 0; f < 150; f++) {
    h.rings.update(0.05, view); checkWiring(h, view, 150 + f);
    if (h.u(17).uVis.value > 0.003 && h.u(16).uVis.value <= 0.003) deferred++;
  }
  assert.ok(deferred > 20, `frames with 17 on and 16 empty: ${deferred}`);
  assert.equal(h.u(16).uVis.value, 0, 'the ring between never showed');
});

test('update(): after a 2-level step the coarse ring defers to the finest ring while the ring between is empty (no (a) blobs)', async () => {
  const lat = 30.05, lon = 31.23, h = await harness(lat, lon), view = { L0: 15, lat, lon, vis: 1, sun: { x: 0, y: 1, z: 0 }, hazeK: 0.55, hazeL: 40 };
  for (let f = 0; f < 150; f++) h.rings.update(0.05, view);
  view.L0 = 17; h.setDelivers((url) => isHeight(url) || tileLevelOf(url) !== 16);
  for (let f = 0; f < 40; f++) { h.rings.update(0.05, view); if (h.u(17).uVis.value > 0.2) break; }
  const c = h.u(15), fine = h.u(17), w = innerWindow(ringWindow(lon, lat, 17), ringWindow(lon, lat, 15));
  assert.ok(fine.uVis.value > 0.2 && h.u(16).uVis.value === 0, 'the finest ring is fading in, the one between is empty');
  assert.equal(c.uInnerOn.value, 1, 'L15 is behind L17 (pushed), not competing with it in depth');
  assert.equal(c.tInner.value, fine.tMask.value, 'L15 reads L17\'s tile mask');
  assert.equal(c.uInnerScale.value, 4, 'the window of a ring two levels finer'); assert.equal(w.scale, 4);
  assert.ok(Math.abs(c.uInnerMin.value.x - w.min[0]) < 1e-12 && Math.abs(c.uInnerMin.value.y - w.min[1]) < 1e-12, 'at L17\'s place in L15\'s uv');
  assert.equal(c.uInnerVis.value, fine.uVis.value, 'the one-layer blend uses L17\'s fade-in');
});
