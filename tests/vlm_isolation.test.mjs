// vlm isolation (spec §3.1, §11.1): browser-loadable vlm/gen modules import only relative paths inside vlm/gen or src,
// every vlm file stays under 500 lines, and no vlm code line is swallowed by an appended // comment.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('..', import.meta.url)), VLM = path.join(ROOT, 'vlm');
const SKIP = new Set(['node_modules', '__pycache__']);
const walk = (d) => (fs.existsSync(d) ? fs.readdirSync(d, { withFileTypes: true }).flatMap((e) => (SKIP.has(e.name) ? [] : e.isDirectory() ? walk(path.join(d, e.name)) : [path.join(d, e.name)])) : []);
const rel = (f) => path.relative(ROOT, f).split(path.sep).join('/');
const importsOf = (src) => [...src.matchAll(/(?:^|\n)\s*(?:import|export)\s[^'"]*?from\s+['"]([^'"]+)['"]|import\(\s*['"]([^'"]+)['"]\s*\)/g)].map((m) => m[1] || m[2]);
const BROWSER = new Set(['vlm/gen/schema.js', 'vlm/gen/safety.js', 'vlm/gen/inject.js', 'vlm/gen/boxresize.js', 'vlm/gen/text/context.js']);
const isBrowser = (r) => BROWSER.has(r) || /^vlm\/gen\/labels\/[^/]+\.js$/.test(r);

test('browser-loadable vlm/gen modules import only relative paths inside vlm/gen/ or src/', () => {
  for (const f of walk(path.join(VLM, 'gen'))) {
    const r = rel(f); if (!isBrowser(r)) continue;
    for (const s of importsOf(fs.readFileSync(f, 'utf8'))) {
      assert.ok(s.startsWith('./') || s.startsWith('../'), `${r} imports the bare or node: specifier ${s}`);
      const target = rel(path.resolve(path.dirname(f), s));
      assert.ok(target.startsWith('vlm/gen/') || target.startsWith('src/'), `${r} imports ${s} -> ${target}`);
      assert.ok(fs.existsSync(path.resolve(path.dirname(f), s)), `${r}: ${s} does not exist`);
    }
  }
});
test('every file under vlm/ and tests/vlm_* stays under 500 lines', () => {
  const files = [...walk(VLM), ...fs.readdirSync(path.join(ROOT, 'tests')).filter((n) => n.startsWith('vlm_')).flatMap((n) => { const p = path.join(ROOT, 'tests', n); return fs.statSync(p).isDirectory() ? walk(p) : [p]; })];
  const long = files.filter((f) => /\.(m?js|py|json|html|txt|md)$/.test(f)).map((f) => { const t = fs.readFileSync(f, 'utf8'); return [rel(f), t.split('\n').length - (t.endsWith('\n') ? 1 : 0)]; }).filter(([, n]) => n >= 500);
  assert.deepEqual(long, []);
});
function commentAt(line) {
  let q = null;
  for (let i = 0; i < line.length - 1; i++) {
    const c = line[i];
    if (q) { if (c === '\\') i++; else if (c === q) q = null; } else if (c === '"' || c === "'" || c === '`') q = c;
    else if (c === '/' && line[i + 1] === '/' && line[i - 1] !== ':') return i;
  }
  return -1;
}
const CODE_END = [/[\w)\]]\s*\)\s*;\s*$/, /;\s*\}\s*$/, /\b[A-Za-z_$][\w$]*\s*:\s*[\w$.]+\s*,\s*$/];
test('no vlm code is swallowed by a // comment appended mid-line', () => {
  const bad = [], blank = (src) => src.replace(/`(?:\\[\s\S]|[^`\\])*`/g, (m) => m.replace(/[^\n]/g, ' '));
  for (const f of walk(VLM).filter((x) => /\.m?js$/.test(x))) blank(fs.readFileSync(f, 'utf8')).split('\n').forEach((line, n) => {
    const k = commentAt(line); if (k < 0 || !line.slice(0, k).trim()) return;
    const pieces = line.slice(k + 2).trim().split(/\s\/\/\s/).map((p) => p.trim());
    if (pieces.some((p) => CODE_END.some((re) => re.test(p)))) bad.push(`${rel(f)}:${n + 1}`);
  });
  assert.deepEqual(bad, []);
});
