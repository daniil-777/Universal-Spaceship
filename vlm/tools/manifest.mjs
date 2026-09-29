// vlm/tools/manifest.mjs — SHA-256 of every file tracked at a base commit, and a check of the working tree against it
// (spec §11.7): pre-existing files byte-identical, .gitignore = base + the three vlm lines, new files only under vlm/
// and tests/vlm_*, and src/app.js still 499 lines.
//   node vlm/tools/manifest.mjs record --base a473c91 --out /Volumes/LaCie/astro-pilot/vlm/logs/manifest_a473c91.json
//   node vlm/tools/manifest.mjs check --manifest /Volumes/LaCie/astro-pilot/vlm/logs/manifest_a473c91.json
import { execFileSync } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const git = (...a) => execFileSync('git', a, { cwd: ROOT, maxBuffer: 1 << 30 });
const sha = (buf) => crypto.createHash('sha256').update(buf).digest('hex');
export const GITIGNORE_ADDS = ['__pycache__/', '*.onnx', '*.safetensors'];
export const allowedNew = (rel) => rel.startsWith('vlm/') || /^tests\/vlm_/.test(rel);

export function record(base) {
  const files = git('ls-tree', '-r', '--name-only', base).toString().split('\n').filter(Boolean), out = {};
  for (const f of files) out[f] = sha(git('show', `${base}:${f}`));
  return { base, files: out };
}

export function check(man) {
  const bad = [];
  for (const [f, h] of Object.entries(man.files)) {
    const p = path.join(ROOT, f);
    if (!fs.existsSync(p)) { bad.push(`${f}: missing`); continue; }
    if (f === '.gitignore') {
      const want = git('show', `${man.base}:.gitignore`).toString() + GITIGNORE_ADDS.join('\n') + '\n';
      if (fs.readFileSync(p, 'utf8') !== want) bad.push('.gitignore: expected the base plus exactly the three vlm lines');
      continue;
    }
    if (sha(fs.readFileSync(p)) !== h) bad.push(`${f}: changed`);
  }
  for (const f of git('ls-files').toString().split('\n').filter(Boolean)) if (!(f in man.files) && !allowedNew(f)) bad.push(`${f}: new file outside vlm/ and tests/vlm_*`);
  const appLines = fs.readFileSync(path.join(ROOT, 'src/app.js'), 'utf8').split('\n').length - 1;
  if (appLines !== 499) bad.push(`src/app.js: ${appLines} lines, expected 499`);
  return bad;
}

if (process.argv[1] && process.argv[1].endsWith('manifest.mjs')) {
  const arg = (k) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : null; };
  if (process.argv[2] === 'record') {
    const m = record(arg('base'));
    fs.writeFileSync(arg('out'), JSON.stringify(m, null, 1));
    console.log(`recorded ${Object.keys(m.files).length} files at ${m.base} -> ${arg('out')}`);
  } else if (process.argv[2] === 'check') {
    const bad = check(JSON.parse(fs.readFileSync(arg('manifest'), 'utf8')));
    for (const b of bad) console.log('ISOLATION', b);
    console.log(bad.length ? `FAIL ${bad.length}` : 'OK isolation manifest clean');
    process.exit(bad.length ? 1 : 0);
  } else { console.log('usage: manifest.mjs record --base <sha> --out <file> | check --manifest <file>'); process.exit(2); }
}
