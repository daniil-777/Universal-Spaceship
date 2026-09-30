// vlm/capture/drive.mjs — the capture CLI (spec §3, §7): one headless browser from the harness launch(), a fresh context per
// page load, the tile cache and politeness limiter, episodes until --n samples, a checkpoint every 50 samples, resume at the
// episode boundary, StopDrive (403/429/budget) checkpoints and stops.
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { scanRun, writeEpisode } from './records.mjs';
import { nextEpisode } from './families.mjs';
import { ledgerCounts } from './routes.mjs';
export { fnv1a32, renderSeed, scanRun, writeEpisode, assemble } from './records.mjs';
const REPO = fileURLToPath(new URL('../../', import.meta.url)), LACIE = '/Volumes/LaCie/astro-pilot/vlm';
const sh = (cmd, args) => { try { return execFileSync(cmd, args).toString(); } catch (e) { return ''; } };
// §7.7 / plan resource rules: >= 25 % free, no other Chromium (the idle hyperframes preview browser is not ours), no Python
// training or export. Another agent may hold the one browser, so the check waits up to waitS seconds before giving up.
function resources() {
  const free = +(/free percentage: (\d+)%/.exec(sh('memory_pressure', [])) || [0, 0])[1];
  const chrome = sh('pgrep', ['-fl', 'Chrome for Testing|chromium|chrome-headless-shell']).split('\n').filter((l) => l && !l.includes('/.cache/hyperframes/')).join('\n').trim();
  const train = sh('pgrep', ['-fl', 'vlm/train/|train\\.py|lora_train|export_onnx|export_decoder']).trim();
  return { free, chrome, train, ok: free >= 25 && !chrome && !train };
}
// Two clean checks 10 s apart are required, so a gap between another agent's back-to-back browser runs is not taken for free.
async function preflight(waitS) {
  const t0 = Date.now(); let clean = 0;
  for (let r = resources(); ; r = resources()) {
    clean = r.ok ? clean + 1 : 0;
    if (clean >= 2) return r.free;
    if (Date.now() - t0 > waitS * 1000) throw new Error(`preflight failed after ${waitS} s: ${r.free} % free (needs >= 25 %)${r.chrome ? `\nanother Chromium is running:\n${r.chrome}` : ''}${r.train ? `\nPython training or export is running:\n${r.train}` : ''}`);
    await new Promise((res) => setTimeout(res, 10000));
  }
}
if (process.argv[1] && process.argv[1].endsWith('drive.mjs')) {
  const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; }, family = arg('family'), n = +arg('n', 1), run = arg('run');
  if (!['S', 'Z'].includes(family) || !run) throw new Error('usage: drive.mjs --family S|Z --n N --run <name> (A, L, D come with Task 10)');
  process.env.AP_SITE ||= REPO.replace(/\/$/, ''); const free0 = await preflight(+arg('wait-browser', 1800));
  const { serve, launch } = await import('/Volumes/LaCie/astro-pilot/test/shot.mjs'), { createUpstream, StopDrive } = await import('./tilecache.mjs');
  const calFile = path.join(REPO, 'vlm/gen/calibration.json'), cal = fs.existsSync(calFile) ? JSON.parse(fs.readFileSync(calFile, 'utf8')) : { c_near: 2.5 };
  const dir = path.join(LACIE, 'raw', run, family), scan = scanRun(dir), { server, port } = await serve(), { browser, page } = await launch({ w: 896, h: 504 }); await page.context().close();
  const D = { browser, port, run, family, mode: arg('mode', 'clock'), licence: arg('licence', 'open'), cNear: +arg('c-near', cal.c_near ?? 2.5), ledger: [], stop: {}, sizes: arg('sizes', null), measure: process.argv.includes('--measure'),
    gitSha: execFileSync('git', ['rev-parse', '--short', 'HEAD'], { cwd: REPO }).toString().trim(), siteDirty: !!execFileSync('git', ['status', '--porcelain', '--', 'src', 'index.html', 'textures', 'model'], { cwd: REPO }).toString().trim(),
    log: (o) => fs.appendFileSync(path.join(dir, 'drive.log.jsonl'), JSON.stringify({ t: Date.now(), ...o }) + '\n') };
  // the 80,000 EOX budget is per run (§6), across families and resumes: the used count persists in raw/<run>/eox_budget.json
  // and seeds the cache's `used` count (its default EOX budget is the 80,000)
  const EOX = 'tiles.maps.eox.at', budgetFile = path.join(LACIE, 'raw', run, 'eox_budget.json'), eoxUsed0 = fs.existsSync(budgetFile) ? JSON.parse(fs.readFileSync(budgetFile, 'utf8')).used : 0;
  D.upstream = createUpstream({ dir: arg('cache-dir', path.join(LACIE, 'tilecache')), contact: process.env.APV_CONTACT || '', used: { [EOX]: eoxUsed0 } });
  const eoxUsed = () => (D.upstream.stats().byHost[EOX] || {}).used ?? eoxUsed0, saveBudget = () => { fs.mkdirSync(path.dirname(budgetFile), { recursive: true }); fs.writeFileSync(budgetFile, JSON.stringify({ used: eoxUsed() })); };
  D.routes = () => ({ port, profile: D.licence, upstream: D.upstream, ledger: D.ledger, stop: D.stop });
  fs.mkdirSync(dir, { recursive: true }); D.log({ start: { family, n, mode: D.mode, licence: D.licence, measure: D.measure, resumed: scan.samples, removed: scan.removed.length, eox_used: eoxUsed0, free_pct: free0, git: D.gitSha, site_dirty: D.siteDirty } });
  let samples = scan.samples, since = 0;
  const ckpt = (e) => { saveBudget(); fs.writeFileSync(path.join(dir, 'checkpoint.json'), JSON.stringify({ samples, episode: e, t: Date.now(), upstream: D.upstream.stats(), eox_used_run: eoxUsed() })); };
  try {
    // twins are episodes e + 50000 (Task 10), so primary episodes stay below 50000; twins count toward the family's size (§8)
    for (let e = 0; samples < n && e < 50000; e++) {
      if (scan.episodes.has(e)) continue;
      const res = await nextEpisode(D, { episode: e, seed0: +arg('seed0', 1), rsOverride: arg('rs-override', null), forceWhen: arg('force-when', null) });
      if (res && res.exhausted) { D.log({ exhausted: e }); break; }
      if (!res) { D.ledger.length = 0; saveBudget(); continue; }
      const kept = res.records.filter((r) => !r.error), tw = res.twin ? res.twin.records.filter((r) => !r.error) : [];
      for (const r of res.records.concat(res.twin ? res.twin.records : []).filter((x) => x.error)) D.log({ drop: r.rec.key, why: r.error });
      writeEpisode(dir, kept.map((r) => r.rec), Object.assign({}, ...kept.map((r) => r.files)), e);
      if (res.twin) writeEpisode(dir, tw.map((r) => r.rec), Object.assign({}, ...tw.map((r) => r.files)), res.twin.episode);
      const counts = ledgerCounts(D.ledger);
      fs.writeFileSync(path.join(dir, `ledger_${String(e).padStart(5, '0')}.jsonl`), D.ledger.map((x) => JSON.stringify(x)).join('\n')); D.ledger.length = 0;
      // measure_extra_s: a --measure run's encode-only passes, which report.mjs takes off the wall time
      fs.appendFileSync(path.join(dir, 'throughput.jsonl'), JSON.stringify({ episode: e, samples: kept.length + tw.length, twins: tw.length, wall_s: res.wallMs / 1000, measure_extra_s: D.measure ? (res.stats.encMs || 0) / 1000 : 0, ...res.stats, ledger: counts }) + '\n');
      samples += kept.length + tw.length; since += kept.length + tw.length; saveBudget(); if (since >= 50) { ckpt(e); since = 0; }
      console.log(`episode ${e}: ${kept.length} kept, ${res.records.length - kept.length} dropped, ${(res.wallMs / 1000).toFixed(1)} s; ${samples}/${n} samples`);
    }
  } catch (err) { D.log({ stop: String(err.message), kind: err instanceof StopDrive ? 'StopDrive' : 'error' }); if (!(err instanceof StopDrive)) throw err; }
  finally { ckpt(null); D.log({ end: { samples, upstream: D.upstream.stats() } }); await browser.close(); server.close(); D.upstream.close(); }
  console.log(`done ${family}: ${samples} samples in ${dir}`); process.exit(0);
}
