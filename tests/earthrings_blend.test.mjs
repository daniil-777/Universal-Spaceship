import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { register } from 'node:module';
import { fileURLToPath } from 'node:url';
import { createLocalFrame, ringLevels } from '../src/earthtiles.js';   // pure geo maths, no three.js

// src/earthrings.js imports the bare specifier 'three', which Node only resolves from tests/ (tests/node_modules/three,
// r170). This inline resolve hook re-resolves 'three' and 'three/...' as if this test file had imported them.
register('data:text/javascript,' + encodeURIComponent(`export async function resolve(s, c, next) {
  return s === 'three' || s.startsWith('three/') ? next(s, { ...c, parentURL: ${JSON.stringify(import.meta.url)} }) : next(s, c); }`));

// Task 7 (2026-09-30): in the far fade (2500-4000 km, uFar < 1) every finer ring's box showed as a soft rectangle over
// the globe. Diagnosis (test/zoom_seam_check.mjs, per-hypothesis variants): each ring's alpha carries V = uFar × day,
// so where a coarse ring C is still drawn under a finer ring F (F's edge band always, its whole footprint until F is
// fully in) two "over" blends let the globe G through (1 − V)² instead of (1 − V). The error is
// V (1 − V) cF cC (C − G): zero with opaque rings (V = 1), 19 % of the ring/globe difference at 3000 km (V = 0.74).
// Fix: where uFar < 1, under F, C's alpha is scaled by (1 − cF) / (1 − V cF), cF = F's own cover (its fade-in × its
// edge fade), so the pair composites as V (cF F + (1 − cF) cC C) + (1 − V (cF + (1 − cF) cC)) G. Below 2500 km the
// factor is off, even at dusk (relief parallax would open holes there; follow-up round).
const SRC = fs.readFileSync(fileURLToPath(new URL('../src/earthrings.js', import.meta.url)), 'utf8');
const FRAG = (SRC.match(/const FRAG = \/\* glsl \*\/`([\s\S]*?)`;/) || [])[1];
const grab = (re, what) => { const m = FRAG && FRAG.match(re); assert.ok(m, `FRAG: ${what} not found`); return m.slice(1); };

// The shader's own expressions, read from the source and evaluated as JS (its float arithmetic and ?: read the same in
// JS), so these tests follow the shader and not a copy of it. edge(iu) is the finer ring's edge fade at this fragment,
// edge(vUv) the ring's own.
function shaderAlpha() {
  const [cExpr, vExpr] = grab(/float c = ([^,;]+), V = ([^;]+);/, 'the finer ring\'s cover c and the stack\'s far fade V');
  const [stackExpr] = grab(/float stack = ([^;]+);/, 'the stack factor');
  const [alphaExpr] = grab(/gl_FragColor = vec4\(col, (.+?)\); \}/, 'the output alpha');
  return new Function('uVis', 'uFar', 'uInnerVis', 'day', 'under', 'eOwn', 'eInner',
    `const iu = 'iu', vUv = 'vUv', edge = (u) => (u === iu ? eInner : eOwn); const c = ${cExpr}, V = ${vExpr}; const stack = ${stackExpr}; return ${alphaExpr};`);
}
const oldAlpha = (uVis, uFar, uInnerVis, day, under, eOwn) => uVis * eOwn * day;   // HEAD 8d2f54f: uVis * e.x * e.y * day

const over = (dst, src, a) => dst.map((d, i) => d + (src[i] - d) * a);
// one pixel: the globe G, then the coarse ring C (drawn first: renderOrder 2 + level), then the finer ring F on top
function composite({ V0, day, visF, visC, eF, eC, F, C, G }, alpha) {
  const aC = alpha(visC * V0, V0, visF, day, true, eC, eF);
  const aF = alpha(visF * V0, V0, 0, day, false, eF, 0);
  return over(over(G, C, aC), F, aF);
}
// the intended picture: the rings are ONE layer over the globe, F over C inside it, the whole stack faded by V
function oneLayer({ V0, day, visF, visC, eF, eC, F, C, G }) {
  const V = V0 * day, cF = visF * eF, cC = visC * eC;
  return G.map((g, i) => V * cF * F[i] + V * (1 - cF) * cC * C[i] + (1 - V * cF - V * (1 - cF) * cC) * g);
}
let seed = 12345;
const rnd = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; };
const sample = (fixed = {}) => ({ V0: rnd(), day: rnd(), visF: rnd(), visC: rnd(), eF: rnd(), eC: rnd(), F: [rnd(), rnd(), rnd()], C: [rnd(), rnd(), rnd()], G: [rnd(), rnd(), rnd()], ...fixed });
const maxErr = (a, b) => Math.max(...a.map((x, i) => Math.abs(x - b[i])));

test('FRAG: the coarse ring\'s alpha carries the stack factor, from the finer ring\'s own cover (uInnerVis × edge(iu)) and uFar', () => {
  assert.match(FRAG, /uniform float [^;]*\buInnerVis\b[^;]*\buFar\b/, 'uInnerVis and uFar are declared');
  assert.match(FRAG, /float edge\(vec2 u\) \{ vec2 e = smoothstep\(vec2\(0\.0\), vec2\(0\.08\), u\) \* smoothstep\(vec2\(0\.0\), vec2\(0\.08\), 1\.0 - u\); return e\.x \* e\.y; \}/,
    'one edge fade for the ring itself and for the finer ring under it (the same 8 % smoothstep as before)');
  const [cExpr, vExpr] = grab(/float c = ([^,;]+), V = ([^;]+);/, 'c and V');
  assert.match(cExpr, /^under \? uInnerVis \* edge\(iu\) : 0\.0$/, 'c is the finer ring\'s cover, and only where the ring is under it');
  assert.equal(vExpr, 'uFar * day', 'V is the part of every ring\'s alpha they share: the far fade and the day side');
  assert.match(grab(/float stack = ([^;]+);/, 'stack')[0], /^uFar < 1\.0 && /, 'the factor acts only in the far fade (uFar < 1); below 2500 km it is off, even at dusk');
  const [alphaExpr] = grab(/gl_FragColor = vec4\(col, (.+?)\); \}/, 'alpha');
  assert.match(alphaExpr, /\bstack\b/, 'the output alpha is scaled by the stack factor');
});

test('stack blend: over 20 000 random pixels (far fade, day, both rings\' fade-ins and edge fades, colours) the pair composites as ONE layer over the globe', () => {
  const alpha = shaderAlpha();
  let worst = 0;
  for (let k = 0; k < 20000; k++) { const s = sample(); worst = Math.max(worst, maxErr(composite(s, alpha), oneLayer(s))); }
  // the V·c ≥ 0.9999 branch (a near-opaque stack, factor 1) leaves at most V (1 − V) ≤ 1e-4 of the colour difference
  assert.ok(worst < 1.1e-4, `worst colour error ${worst}`);
});

test('stack blend: the old shader (HEAD 8d2f54f) fails the same check by V (1 − V) cF cC (C − G) (the seam), so the check can tell', () => {
  const s = sample({ V0: 0.741, day: 1, visF: 1, visC: 1, eF: 1, eC: 1, F: [0.3, 0.3, 0.3], C: [0.3, 0.3, 0.3], G: [0.8, 0.8, 0.8] });
  const err = composite(s, oldAlpha)[0] - oneLayer(s)[0], expect = 0.741 * (1 - 0.741) * (0.3 - 0.8);
  assert.ok(Math.abs(err - expect) < 1e-12, `old error ${err} vs V(1 − V)(C − G) = ${expect}`);
  assert.ok(Math.abs(err) > 0.09, 'a 10 % step: the soft box at 3000 km');
  assert.ok(maxErr(composite(s, shaderAlpha()), oneLayer(s)) < 1e-12, 'the fixed shader has none in the same pixel');
});

test('stack blend: below 2500 km (uFar = 1) every ring\'s alpha is exactly the old one, at any time of day (dusk included)', () => {
  const alpha = shaderAlpha();
  for (let k = 0; k < 5000; k++) {
    const uVis = rnd(), uInnerVis = rnd() < 0.2 ? 1 : rnd(), eOwn = rnd(), eInner = rnd() < 0.2 ? 1 : rnd(), under = rnd() < 0.7, day = rnd() < 0.3 ? 1 : rnd();
    assert.equal(alpha(uVis, 1, uInnerVis, day, under, eOwn, eInner), oldAlpha(uVis, 1, uInnerVis, day, under, eOwn), `sample ${k} (day ${day})`);
  }
});

test('stack blend: where the finer ring fully covers (its cover c = 1) and the stack is see-through (V < 1), the coarse ring adds nothing', () => {
  const alpha = shaderAlpha();
  for (const V0 of [0.1, 0.5, 0.741, 0.99]) assert.equal(alpha(0.8 * V0, V0, 1, 1, true, 1, 1), 0, `V0 = ${V0}`);
  assert.ok(alpha(0.741, 0.741, 1, 1, false, 1, 1) > 0.7, 'not under a finer ring: the coarse ring is drawn as before');
});

function makeFakeRenderer() {
  const props = new Map(), ensure = (t) => { let p = props.get(t); if (!p) { p = { __webglTexture: {} }; props.set(t, p); } return p; };
  return { initTexture: ensure, copyTextureToTexture: (s, d) => { ensure(d); }, getContext: () => ({ TEXTURE_2D: 0x0de1, generateMipmap() {} }),
    properties: { get: ensure }, state: { bindTexture() {}, unbindTexture() {} } };
}
const fakeLoader = { inFlight: 0, queued: 0, stats: {}, failRate: 0, request(job) { if (job.wanted()) job.done({}); }, prune() {}, suspend() {}, resume() {} };

test('createEarthRings.update(): uFar = the view\'s far fade on every ring, uInnerVis = the next finer used ring\'s own fade-in (0 on the finest)', async () => {
  const { createEarthRings } = await import('../src/earthrings.js');
  const meshes = [], rings = createEarthRings({ add: (m) => meshes.push(m) }, makeFakeRenderer(), { loader: fakeLoader });
  rings.rebase(createLocalFrame(48, 10));
  const view = { L0: 15, lat: 48, lon: 10, vis: 0.5, sun: { x: 0, y: 1, z: 0 }, hazeK: 0.55, hazeL: 40 }, want = ringLevels(view.L0);
  const byLevel = (L) => meshes.find((m) => m.renderOrder - 2 === L).material.uniforms;
  let partial = 0;
  for (let f = 0; f < 120; f++) {
    rings.update(0.05, view);
    for (const L of want) {
      const u = byLevel(L), finer = want.filter((x) => x > L);
      assert.equal(u.uFar.value, 0.5, `frame ${f}, L${L}: uFar`);
      const expect = finer.length ? byLevel(Math.min(...finer)).uVis.value / view.vis : 0;
      assert.equal(u.uInnerVis.value, expect, `frame ${f}, L${L}: uInnerVis = the finer ring's fade-in`);
      if (u.uInnerVis.value > 0.01 && u.uInnerVis.value < 0.99) partial++;
    }
  }
  assert.ok(partial > 0, 'some frame caught a finer ring mid fade-in (not only 0 or 1)');
});
