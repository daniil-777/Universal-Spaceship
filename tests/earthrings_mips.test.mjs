import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import { createLocalFrame } from '../src/earthtiles.js';   // pure geo maths, no three.js (see its own header)

// Task 6 (diagnosis 2026-09-29 §3 #9, K6): three r170's copyTextureToTexture regenerates a texture's whole mip chain
// on every level-0 copy when dstTexture.generateMipmaps is true (confirmed in three.module.js: `if (level === 0 &&
// dstTexture.generateMipmaps) _gl.generateMipmap(glTarget)`). flush() could copy up to UPLOADS_PER_FRAME (6) colour
// tiles a frame, so an atlas' mip chain was rebuilt up to 6 times a frame instead of once. Fix: copy() turns the
// destination atlas' generateMipmaps off for the GPU copy and back on right after, flush() collects every colour
// atlas it touched in a Set, and rebuilds each one's mips exactly once at the end, by hand, via the GL call the
// diagnosis confirmed (bypassing copyTextureToTexture's own, now-suppressed, per-copy regeneration).
const SRC = fs.readFileSync(fileURLToPath(new URL('../src/earthrings.js', import.meta.url)), 'utf8');

// --- source-level checks: these hold regardless of whether createEarthRings can be driven in Node (it can, below,
// with a fake renderer and a fake loader that resolve tile requests synchronously) --------------------------------

test('earthrings.js: colour atlases are still created with generateMipmaps = true (the mip chain is allocated)', () => {
  assert.match(SRC, /atlas\(RING_TILES \* T, THREE\.SRGBColorSpace, true\)/, 'the colour atlas keeps mips = true at creation');
  assert.match(SRC, /atlas\(HEIGHT_TILES \* T, THREE\.NoColorSpace, false\)/, 'the height atlas is unaffected (mips = false, unchanged)');
});

test('earthrings.js: copy() turns the destination\'s generateMipmaps off around the GPU copy and restores it after', () => {
  const fn = (SRC.match(/function copy\([^)]*\)\s*\{[\s\S]*?\n  \}/) || [])[0];
  assert.ok(fn, 'copy() found');
  const iOff = fn.search(/generateMipmaps\s*=\s*false/), iCopy = fn.search(/copyTextureToTexture\(/), iOn = fn.search(/generateMipmaps\s*=\s*true/);
  assert.ok(iOff >= 0, 'copy() sets generateMipmaps = false');
  assert.ok(iCopy > iOff, 'the GPU copy runs after generateMipmaps is turned off');
  assert.ok(iOn > iCopy, 'generateMipmaps is restored to true after the GPU copy');
});

test('earthrings.js: flush() collects touched colour atlases in a Set and rebuilds each one\'s mips once, after the upload loop', () => {
  const fn = (SRC.match(/function flush\(\)\s*\{[\s\S]*?\n  \}/) || [])[0];
  assert.ok(fn, 'flush() found');
  assert.match(fn, /new Set\(\)/, 'flush() collects touched atlases in a Set');
  const iLoopEnd = fn.lastIndexOf('done++;'), iMip = fn.search(/generateMipmap\(/);
  assert.ok(iLoopEnd >= 0 && iMip > iLoopEnd, 'the manual mip rebuild runs after the upload loop, not inside it');
  // the exact GL path the diagnosis confirmed in three.module.js: get the context, look up the WebGL texture, bind
  // it, and call generateMipmap by hand (copyTextureToTexture's own per-copy regeneration is suppressed by copy()).
  assert.match(fn, /renderer\.getContext\(\)/); assert.match(fn, /renderer\.properties\.get\(/);
  assert.match(fn, /renderer\.state\.bindTexture\(/); assert.match(fn, /__webglTexture/); assert.match(fn, /gl\.generateMipmap\(/);
});

// --- behavioural check: attempted, but createEarthRings CANNOT be driven in Node here. earthrings.js does
// `import * as THREE from 'three'`, and that bare specifier only resolves under tests/ (tests/node_modules/three,
// r170.0 — used by this file's own harness, e.g. earthloader.test.mjs's fakes) because that is the nearest
// node_modules above it; src/earthrings.js has no node_modules above *it* (src/ or the repo root), so importing it
// throws ERR_MODULE_NOT_FOUND before a single line of createEarthRings runs — confirmed with
// `node -e "import('./src/earthrings.js')"` from the repo root. Per the brief ("if it cannot, say so and keep the
// source test"): the dynamic import below is attempted first and the test records why it was skipped, so a future
// change to module resolution (e.g. a src/node_modules or an import map for Node) would make this test start
// exercising createEarthRings for real instead of silently staying a no-op. ------------------------------------

function makeFakeRenderer() {
  const props = new Map();
  const ensure = (t) => { let p = props.get(t); if (!p) { p = { __webglTexture: {} }; props.set(t, p); } return p; };
  const calls = { generateMipmap: 0, bindTexture: 0 };
  const gl = { TEXTURE_2D: 0x0de1, generateMipmap() { calls.generateMipmap++; } };
  return {
    calls,
    initTexture: (t) => ensure(t),
    // r170's own rule, reproduced (diagnosis.md §3 #9): a level-0 copy into a texture whose generateMipmaps is true
    // regenerates its whole mip chain right there. copy() must have turned this off before calling us.
    copyTextureToTexture(src, dst, srcRegion, dstPosition, level = 0) {
      ensure(dst);
      if (level === 0 && dst.generateMipmaps) calls.generateMipmap++;
    },
    getContext: () => gl,
    properties: { get: ensure },
    state: { bindTexture: () => { calls.bindTexture++; } },
  };
}
// Resolves every job the moment it is requested (still wanted), so a whole ring's colour and height tiles land in
// earthrings.js's internal `uploads` array within one synchronous update() call, the same way a pile of already-
// cached tiles would.
function makeFakeLoader(bmp = {}) {
  return { inFlight: 0, queued: 0, stats: { requested: 0, loaded: 0, failed: 0, blank: 0, deduped: 0 }, failRate: 0,
    request(job) { if (job.wanted()) job.done(bmp); }, prune() {}, suspend() {}, resume() {} };
}

test('createEarthRings + flush(): 6 uploads into one colour atlas give exactly 1 generateMipmap call, once a frame', async (t) => {
  let createEarthRings;
  try { ({ createEarthRings } = await import('../src/earthrings.js')); }
  catch (e) {
    t.skip(`createEarthRings cannot be driven in Node: ${e.message} (src/earthrings.js's own 'three' import only `
      + `resolves under tests/, not under src/ or the repo root; kept as the source-level checks above instead)`);
    return;
  }
  const scene = { add() {} }, renderer = makeFakeRenderer(), loader = makeFakeLoader();
  const rings = createEarthRings(scene, renderer, { loader });
  rings.rebase(createLocalFrame(45, 8));
  const view = { L0: 15, lat: 45, lon: 8, vis: 1, sun: { x: 0, y: 1, z: 0 }, hazeK: 0, hazeL: 60 };

  // Frame 1: every ring is placed for the first time; the fake loader resolves all of ring 0's (the finest, placed
  // first) ~64 colour tiles synchronously, so flush()'s UPLOADS_PER_FRAME = 6 cap means the first 6 uploads it drains
  // are all colour tiles into the SAME ring's SAME atlas — the "6 uploads into one atlas" case the brief asks for.
  rings.update(0.016, view);
  assert.equal(renderer.calls.generateMipmap, 1, `frame 1: expected 1 generateMipmap call for the one touched colour atlas, got ${renderer.calls.generateMipmap}`);
  assert.ok(renderer.calls.bindTexture >= 1, 'the manual rebuild binds the atlas before calling generateMipmap');

  // Frame 2: ring 0 still has far more than 6 undelivered colour tiles queued, so this frame's flush() again drains
  // 6 uploads into that same atlas — one more manual rebuild, not six, and not zero.
  rings.update(0.016, view);
  assert.equal(renderer.calls.generateMipmap, 2, `frame 2: expected 2 cumulative generateMipmap calls (1 more), got ${renderer.calls.generateMipmap}`);
});
