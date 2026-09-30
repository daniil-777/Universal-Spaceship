// node vlm/capture/determinism.mjs --family S --seed0 101 --case S [--rs-check] [--force-when cloud] [--plan-run <run with Z/plan.json>] [--mode clock|freeze]
// Two cold runs of the same page URL (§7.5) in separate processes (a fresh browser each): the same seed and one fixed
// --rs-override, so the run name never reaches the render seed; for A and Z the first with an empty cache dir, the second warm;
// every run starts from an emptied run folder, never a resumed one. Every record of the episode is compared (the key sets must
// match): facts JSON (facts, safety, safety_eye, zoom, cameras with key-relative names, provenance.clock) byte for byte, and
// PNGs: identical for S/L/D, >= 99.5 % identical pixels and mean |d| <= 0.5/255 for A/Z; in freeze mode the pixel numbers are
// reported, not gated. Writes $LACIE/logs/determinism_<case>.json.
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import sharp from 'sharp';
import { fnv1a32 } from './records.mjs';
const REPO = fileURLToPath(new URL('../../', import.meta.url));
const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; }, fam = arg('family'), cs = arg('case', fam), L = '/Volumes/LaCie/astro-pilot/vlm', mode = arg('mode', 'clock');
const tmpCache = path.join(L, 'tmp', `detcache_${cs}_${Date.now()}`), rs = String(fnv1a32(`det|${cs}|${arg('seed0', '101')}`)), extra = [...(arg('force-when') ? ['--force-when', arg('force-when')] : []), '--mode', mode];
const runOnce = (tag, rsUse = rs) => {
  const run = `det_${cs}_${tag}`, dir = path.join(L, 'raw', run, fam); fs.rmSync(path.join(L, 'raw', run), { recursive: true, force: true });
  if (fam === 'Z') { fs.mkdirSync(dir, { recursive: true }); fs.copyFileSync(path.join(L, 'raw', arg('plan-run', 'g1'), 'Z', 'plan.json'), path.join(dir, 'plan.json')); }
  execFileSync('node', ['vlm/capture/drive.mjs', '--family', fam, '--n', '1', '--run', run, '--seed0', arg('seed0', '101'), '--rs-override', rsUse, ...(fam === 'A' || fam === 'Z' ? ['--cache-dir', tmpCache] : []), ...extra], { stdio: 'inherit', cwd: REPO }); return dir;
};
const a = runOnce('a'), b = runOnce('b'), keysOf = (d) => fs.readdirSync(d).filter((f) => /^[SALDZ]_.*\.json$/.test(f)).sort();
const rel = (d) => keysOf(d).map((k) => k.replace(/^[SALDZ]_.*_(\d{5}_\d{6})\.json$/, '$1'));
const pick = (r) => JSON.stringify([r.facts, r.safety, r.safety_eye, r.zoom, Object.entries(r.cameras).map(([k, v]) => [k.slice(r.key.length), v]), r.provenance.clock]);
async function cmp(fa, fb) { const [x, y] = await Promise.all([fa, fb].map((f) => sharp(f).removeAlpha().raw().toBuffer())); let same = 0, sum = 0; for (let i = 0; i < x.length; i++) { const d = Math.abs(x[i] - y[i]); sum += d; } for (let i = 0; i < x.length; i += 3) if (x[i] === y[i] && x[i + 1] === y[i + 1] && x[i + 2] === y[i + 2]) same++; return { identical: Buffer.compare(fs.readFileSync(fa), fs.readFileSync(fb)) === 0, pixSame: same / (x.length / 3), mad: sum / x.length }; }
const strict = ['S', 'L', 'D'].includes(fam), ka = keysOf(a), kb = keysOf(b), records = [];
for (const [i, k] of ka.entries()) {
  if (!kb[i] || rel(a)[i] !== rel(b)[i]) break;
  const ra = JSON.parse(fs.readFileSync(path.join(a, k))), rb = JSON.parse(fs.readFileSync(path.join(b, kb[i]))), px = [];
  for (const f of ra.frames.concat(ra.narrator_frame.endsWith('.chase.png') ? [ra.narrator_frame] : [])) px.push({ f: f.slice(ra.key.length), ...(await cmp(path.join(a, f), path.join(b, f.replace(ra.key, rb.key)))) });
  records.push({ key: rel(a)[i], facts_identical: pick(ra) === pick(rb), frames: px, pixels_pass: px.every((p) => (strict ? p.identical : p.pixSame >= 0.995 && p.mad <= 0.5)) });
}
const sameKeys = ka.length > 0 && JSON.stringify(rel(a)) === JSON.stringify(rel(b)), factsOk = records.every((r) => r.facts_identical), pxOk = records.every((r) => r.pixels_pass);
const out = { case: cs, family: fam, mode, rs: +rs, keys_a: rel(a), keys_b: rel(b), same_keys: sameKeys, facts_identical: factsOk, pixels_pass: pxOk, records, pass: sameKeys && factsOk && (mode === 'freeze' || pxOk) };
if (process.argv.includes('--rs-check')) { const c = runOnce('c', String((+rs + 1) >>> 0)), kc = keysOf(c), ra = JSON.parse(fs.readFileSync(path.join(a, ka[0]))), rc = JSON.parse(fs.readFileSync(path.join(c, kc[0]))); out.rs_changes_png = !(await cmp(path.join(a, ra.frames[ra.frames.length - 1]), path.join(c, rc.frames[rc.frames.length - 1]))).identical; out.pass = out.pass && out.rs_changes_png; }
fs.rmSync(tmpCache, { recursive: true, force: true }); fs.writeFileSync(path.join(L, 'logs', `determinism_${cs}.json`), JSON.stringify(out, null, 1)); console.log(JSON.stringify({ case: cs, mode, same_keys: sameKeys, facts_identical: factsOk, pixels_pass: pxOk, n: records.length, pass: out.pass }));
process.exit(out.pass ? 0 : 1);
