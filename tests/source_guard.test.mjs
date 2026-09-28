import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// A `// comment` appended mid-line silently turns the rest of the line into comment (it happened three times on
// 2026-09-28: a lost OutputPass, a chase camera's lerp/lookAt and a closing brace, and space.js's lights/root/ready;
// a fourth time the swallowed statements carried their own trailing comment, so each piece is checked).
// Flag comments that END like code — a call statement `…);`, a closing `; }`, or an object property `key: value,` —
// in the comment or in any piece of it before a nested ` // `.
const SRC = fileURLToPath(new URL('../src', import.meta.url));
function commentAt(line) {                                   // the first // outside quotes and template literals
  let q = null;
  for (let i = 0; i < line.length - 1; i++) {
    const c = line[i];
    if (q) { if (c === '\\') i++; else if (c === q) q = null; }
    else if (c === '"' || c === "'" || c === '`') q = c;
    else if (c === '/' && line[i + 1] === '/' && line[i - 1] !== ':') return i;
  }
  return -1;
}
const CODE_END = [/[\w)\]]\s*\)\s*;\s*$/, /;\s*\}\s*$/, /\b[A-Za-z_$][\w$]*\s*:\s*[\w$.]+\s*,\s*$/];
test('no code is swallowed by a comment appended mid-line', () => {
  const walk = (d) => fs.readdirSync(d, { withFileTypes: true }).flatMap((e) => (e.isDirectory() ? walk(path.join(d, e.name)) : e.name.endsWith('.js') ? [path.join(d, e.name)] : []));
  const bad = [];
  const blankTemplates = (src) => src.replace(/`(?:\\[\s\S]|[^`\\])*`/g, (m) => m.replace(/[^\n]/g, ' '));   // shader strings: GLSL comments are not JS
  for (const f of walk(SRC)) blankTemplates(fs.readFileSync(f, 'utf8')).split('\n').forEach((line, n) => {
    const k = commentAt(line); if (k < 0 || !line.slice(0, k).trim()) return;
    const tail = line.slice(k + 2).trim(), pieces = tail.split(/\s\/\/\s/).map((p) => p.trim());
    if (pieces.some((p) => CODE_END.some((r) => r.test(p)))) bad.push(`${path.relative(SRC, f)}:${n + 1} …${tail.slice(-90)}`);
  });
  assert.deepEqual(bad, []);
});
