import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// Task 5: the zoom composer gets its own 4x MSAA render target (diagnosis.md §2 found no black bands on ANGLE/Metal,
// contradicting app.js's comment on the *flight* composer — a separate target, so that comment is untouched here),
// and app.js stops feeding the zoom's frame rate into adaptQuality while the zoom is open (it already lowers L0 on
// its own via the pixel ratio adaptQuality controls).
const SRC = fileURLToPath(new URL('../src', import.meta.url));

test('earthzoom.js gives its composer a 4x MSAA render target', () => {
  const src = fs.readFileSync(path.join(SRC, 'earthzoom.js'), 'utf8');
  assert.match(src, /new THREE\.WebGLRenderTarget\(1,\s*1,\s*\{\s*type:\s*THREE\.HalfFloatType,\s*samples:\s*4\s*\}\)/);
  assert.match(src, /new EffectComposer\(renderer,\s*rt\)/);
});

test('app.js stays at 499 lines and only calls adaptQuality while the zoom is closed', () => {
  const src = fs.readFileSync(path.join(SRC, 'app.js'), 'utf8');
  const lineCount = (src.match(/\n/g) || []).length;
  assert.equal(lineCount, 499);
  const lines = src.split('\n');
  const guarded = lines.some((l) => l.includes('if (!zoom || !zoom.active) adaptQuality(state.fps);'));
  assert.ok(guarded, 'expected a line gating adaptQuality on the zoom being closed');
  const unguarded = lines.some((l) => /[^!]adaptQuality\(state\.fps\);/.test(l) && !l.includes('zoom.active)'));
  assert.ok(!unguarded, 'adaptQuality(state.fps) must not be called without the zoom guard');
});

// Task 6 addendum (Task 5 review): pin the ?zoomaa=0 escape so a later edit cannot make the MSAA target unconditional,
// or move the flag's read off module scope (it must be read once, never mid-session, per the comment above it).
test('earthzoom.js: ZOOMAA is a module-scope flag that still gates the MSAA target with a ternary', () => {
  const src = fs.readFileSync(path.join(SRC, 'earthzoom.js'), 'utf8');
  assert.match(src, /^const ZOOMAA = new URLSearchParams\(location\.search\)\.get\('zoomaa'\) !== '0';$/m,
    'ZOOMAA must stay a module-scope const (column 0), not read inside a function');
  assert.match(src, /const rt = ZOOMAA \? new THREE\.WebGLRenderTarget\(1,\s*1,\s*\{[^}]*samples:\s*4[^}]*\}\)\s*:\s*undefined;/,
    'the MSAA target must stay behind the ?zoomaa=0 ternary, not become unconditional');
});
