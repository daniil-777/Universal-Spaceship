// vlm/capture/drive.mjs — the capture CLI (spec §3, §7): one headless browser from the harness launch(), a fresh context per
// page load, the tile cache and politeness limiter, episodes until --n samples, a checkpoint every 50 samples, resume at the
// episode boundary, StopDrive (403/429/budget) checkpoints and stops.
// Exit codes: 0 done (also at the EOX budget, which persists in eox_budget.json); 3 (STOP_EXIT) a 403/429 stop, persisted to
// raw/<run>/stop.json, or a run that refuses to start because that file exists; 4 (RESOURCE_EXIT) free memory fell below 25 %
// between episodes; 1 any other error. `&&` chains therefore halt on a ban or a resource stop.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { scanRun, writeEpisode, startGate, stopOf, writeStop, exitCodeOf, pickStop, freeSpan, freshRecords, STOP_EXIT, RESOURCE_EXIT } from './records.mjs';
import { MAX_RESETS } from './episode_corridor.mjs';
import { nextEpisode, closeEpisode } from './families.mjs';
import { ledgerCounts } from './routes.mjs';
export { fnv1a32, renderSeed, scanRun, writeEpisode, assemble, startGate, stopOf, writeStop, readStop, exitCodeOf, STOP_EXIT } from './records.mjs';
const REPO = fileURLToPath(new URL('../../', import.meta.url)), LACIE = '/Volumes/LaCie/astro-pilot/vlm';
const sh = (cmd, args) => { try { return execFileSync(cmd, args).toString(); } catch (e) { return ''; } };
const freePct = () => +(/free percentage: (\d+)%/.exec(sh('memory_pressure', [])) || [0, 0])[1];
// §7.7 / plan resource rules: >= 25 % free, no other Chromium (the idle hyperframes preview browser is not ours), no Monte
// Carlo (real_mc, landing_mc), no Python training or export. Another agent may hold the one browser, so the check waits up
// to waitS seconds before giving up.
function resources() {
  const free = freePct(), chrome = sh('pgrep', ['-fl', 'Chrome for Testing|chromium|chrome-headless-shell']).split('\n').filter((l) => l && !l.includes('/.cache/hyperframes/')).join('\n').trim();
  const train = sh('pgrep', ['-fl', 'vlm/train/|train\\.py|lora_train|export_onnx|export_decoder']).trim(), mc = sh('pgrep', ['-fl', 'real_mc|landing_mc']).trim();
  return { free, chrome, train, mc, ok: free >= 25 && !chrome && !train && !mc };
}
// Two clean checks 10 s apart are required, so a gap between another agent's back-to-back browser runs is not taken for free.
async function preflight(waitS) {
  const t0 = Date.now(); let clean = 0;
  for (let r = resources(); ; r = resources()) {
    clean = r.ok ? clean + 1 : 0;
    if (clean >= 2) return r.free;
    if (Date.now() - t0 > waitS * 1000) throw new Error(`preflight failed after ${waitS} s: ${r.free} % free (needs >= 25 %)${r.chrome ? `\nanother Chromium is running:\n${r.chrome}` : ''}${r.train ? `\nPython training or export is running:\n${r.train}` : ''}${r.mc ? `\na Monte Carlo is running:\n${r.mc}` : ''}`);
    await new Promise((res) => setTimeout(res, 10000));
  }
}
if (process.argv[1] && process.argv[1].endsWith('drive.mjs')) {
  const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; }, family = arg('family'), n = +arg('n', 1), run = arg('run');
  if (!['S', 'A', 'L', 'D', 'Z'].includes(family) || !run) throw new Error('usage: drive.mjs --family S|A|L|D|Z --n N --run <name> [--seed0 K] [--force-when cloud|china|city|cloudy|below_base]');
  // APV_RAW_ROOT is a test hook only (the start-gate tests): it relocates raw/ and ends the process right after the gate (exit 2
  // when the gate passes), so a test can never reach the preflight, a browser or the network; captures always use $LACIE/raw
  const RAW = process.env.APV_RAW_ROOT || path.join(LACIE, 'raw'), runDir = path.join(RAW, run), gate = startGate(runDir);
  if (gate.refuse) { console.error(`refusing to start: ${gate.why}`); process.exit(gate.code); }
  if (process.env.APV_RAW_ROOT) { console.error('APV_RAW_ROOT (test hook): the start gate passed; stopping before the preflight'); process.exit(2); }
  // temp files (Playwright's browser profile among them) go to LaCie, never the Mac disk
  const TMP = path.join(LACIE, 'tmp'); fs.mkdirSync(TMP, { recursive: true }); process.env.TMPDIR = TMP;
  process.env.AP_SITE ||= REPO.replace(/\/$/, ''); const free0 = await preflight(+arg('wait-browser', 1800));
  const { serve, launch } = await import('/Volumes/LaCie/astro-pilot/test/shot.mjs'), { createUpstream, StopDrive } = await import('./tilecache.mjs');
  const calFile = path.join(REPO, 'vlm/gen/calibration.json'), cal = fs.existsSync(calFile) ? JSON.parse(fs.readFileSync(calFile, 'utf8')) : { c_near: 2.5 };
  // provenance.policy_sha: the served belt policy (S; the page asserts it flies it, probe setup) or atmo policy (A-PPO episodes),
  // hashed once per drive; L (autoland) and D (GNC) fly code, identified by git_sha
  const polFile = { S: 'model/policy.json', A: 'model/policy_atmo.json' }[family], polBuf = polFile ? fs.readFileSync(path.join(REPO, polFile)) : null, policy = polBuf ? { sha: crypto.createHash('sha256').update(polBuf).digest('hex'), steps: JSON.parse(polBuf).steps } : null;
  const dir = path.join(runDir, family), scan = scanRun(dir), { server, port } = await serve(), { browser, page } = await launch({ w: 896, h: 504 }); await page.context().close();
  const D = { browser, port, run, family, dir, policy, mode: arg('mode', 'clock'), licence: arg('licence', 'open'), cNear: +arg('c-near', cal.c_near ?? 2.5), ledger: [], stop: {}, sizes: arg('sizes', null), measure: process.argv.includes('--measure'),
    gitSha: execFileSync('git', ['rev-parse', '--short', 'HEAD'], { cwd: REPO }).toString().trim(), siteDirty: !!execFileSync('git', ['status', '--porcelain', '--', 'src', 'index.html', 'textures', 'model', 'vlm'], { cwd: REPO }).toString().trim(),
    log: (o) => fs.appendFileSync(path.join(dir, 'drive.log.jsonl'), JSON.stringify({ t: Date.now(), ...o }) + '\n') };
  // the 80,000 EOX budget is per run (§6), across families and resumes: the used count persists in raw/<run>/eox_budget.json
  // and seeds the cache's `used` count (its default EOX budget is the 80,000)
  const EOX = 'tiles.maps.eox.at', budgetFile = path.join(runDir, 'eox_budget.json'), eoxUsed0 = fs.existsSync(budgetFile) ? JSON.parse(fs.readFileSync(budgetFile, 'utf8')).used : 0;
  D.upstream = createUpstream({ dir: arg('cache-dir', path.join(LACIE, 'tilecache')), contact: process.env.APV_CONTACT || '', used: { [EOX]: eoxUsed0 } });
  const eoxUsed = () => (D.upstream.stats().byHost[EOX] || {}).used ?? eoxUsed0, saveBudget = () => { fs.mkdirSync(path.dirname(budgetFile), { recursive: true }); fs.writeFileSync(budgetFile, JSON.stringify({ used: eoxUsed() })); };
  D.routes = () => ({ port, profile: D.licence, upstream: D.upstream, ledger: D.ledger, stop: D.stop });
  fs.mkdirSync(dir, { recursive: true }); D.log({ start: { family, n, mode: D.mode, licence: D.licence, measure: D.measure, resumed: scan.samples, removed: scan.removed.length, eox_used: eoxUsed0, free_pct: free0, git: D.gitSha, site_dirty: D.siteDirty, policy_sha: policy && policy.sha, tmpdir: TMP } });
  let samples = scan.samples, since = 0, code = 0, fatal = null;
  const ckpt = (e) => { saveBudget(); fs.writeFileSync(path.join(dir, 'checkpoint.json'), JSON.stringify({ samples, episode: e, t: Date.now(), upstream: D.upstream.stats(), eox_used_run: eoxUsed() })); };
  // Z: a discarded location (or one with no view kept) is closed with a 0-record .done, so a resume never reruns it, and a
  // spare of the same split takes its place; closeEpisode writes zorder.json before the .done (families.mjs, zplan.mjs)
  try {
    // twins are episodes e + 50000 (Task 10), so primary episodes stay below 50000; twins count toward the family's size (§8)
    for (let e = 0; samples < n && e < 50000; e++) {
      if (scan.episodes.has(e)) continue;
      const free = freePct();
      if (free < 25) { D.log({ stop: `memory_pressure shows ${free} % free (< 25 %) before episode ${e}`, kind: 'resources' }); code = RESOURCE_EXIT; break; }
      // an A page samples across crash resets over at most the free indices from e (N-1): it stops before a finished episode
      const span = freeSpan(scan.episodes, e, MAX_RESETS + 1);
      const res = await nextEpisode(D, { episode: e, seed0: +arg('seed0', 1), rsOverride: arg('rs-override', null), forceWhen: arg('force-when', null), maxSpan: span });
      if (res && res.exhausted) { D.log({ exhausted: e }); break; }
      // a discarded page of any family is closed with a 0-record .done (N-1), so a resume never reruns it and the next page
      // starts at the same index (and seed) as in the original run
      if (!res) { D.ledger.length = 0; saveBudget(); if (D.stop.error) throw D.stop.error; closeEpisode(D, e, [], {}, { why: 'discarded' }); continue; }
      // an A page that sampled across crash resets spans episodes e .. e + resets (families.mjs); every other page is episode e
      const eps = res.episodes || [{ episode: e, records: res.records }], ok = (x) => !x.error, kept = eps.flatMap((x) => x.records).filter(ok), tw = res.twin ? res.twin.records.filter(ok) : [];
      if (eps.length > span || eps.some((x) => scan.episodes.has(x.episode))) throw new Error(`page ${e} spans ${eps.length} episodes; only ${span} are free (refusing to overwrite a finished episode)`);
      const fresh = freshRecords(dir, kept).length + freshRecords(dir, tw).length;
      for (const r of eps.flatMap((x) => x.records).concat(res.twin ? res.twin.records : []).filter((x) => x.error)) D.log({ drop: r.rec.key, why: r.error });
      // the twin is written before its original's .done (M-1), and a page's first episode's .done last: a resume that finds it
      // finds the whole page (a partial page is rerun from its first episode and rewrites the same files)
      if (res.twin) writeEpisode(dir, tw.map((r) => r.rec), Object.assign({}, ...tw.map((r) => r.files)), res.twin.episode);
      for (const x of eps.slice().reverse()) { const k = x.records.filter(ok); closeEpisode(D, x.episode, k.map((r) => r.rec), Object.assign({}, ...k.map((r) => r.files)), { why: 'no view kept' }); }
      const counts = ledgerCounts(D.ledger);
      fs.writeFileSync(path.join(dir, `ledger_${String(e).padStart(5, '0')}.jsonl`), D.ledger.map((x) => JSON.stringify(x)).join('\n')); D.ledger.length = 0;
      // measure_extra_s: a --measure run's encode-only passes, which report.mjs takes off the wall time
      fs.appendFileSync(path.join(dir, 'throughput.jsonl'), JSON.stringify({ episode: e, samples: kept.length + tw.length, twins: tw.length, wall_s: res.wallMs / 1000, measure_extra_s: D.measure ? (res.stats.encMs || 0) / 1000 : 0, ...res.stats, ledger: counts }) + '\n');
      // M-a: only newly written records count toward --n (a rerun that rewrites a twin written before a crash adds nothing)
      samples += fresh; since += fresh; saveBudget(); if (since >= 50) { ckpt(e); since = 0; }
      console.log(`episode ${e}${eps.length > 1 ? `-${e + eps.length - 1}` : ''}: ${kept.length} kept, ${res.records.length - kept.length} dropped, ${(res.wallMs / 1000).toFixed(1)} s; ${samples}/${n} samples`);
      e = eps[eps.length - 1].episode;
      // a stop latched during a kept episode (its failed requests already dropped their records) ends the drive here
      if (D.stop.error) throw D.stop.error;
    }
  } catch (err0) {
    const err = pickStop(err0, D.stop.error); if (err !== err0) D.log({ error: String(err0 && err0.message), superseded_by: String(err.message) });
    if (err instanceof StopDrive) {
      const s = stopOf(err); D.log({ stop: String(err.message), kind: 'StopDrive', ...s });
      code = exitCodeOf(err); if (code === STOP_EXIT) { writeStop(runDir, s); console.error(`STOP ${s.status} from ${s.host}: persisted to ${runDir}/stop.json`); }
    } else { D.log({ stop: String(err.message), kind: 'error' }); fatal = err; }
  } finally { ckpt(null); D.log({ end: { samples, code, upstream: D.upstream.stats() } }); await browser.close(); server.close(); D.upstream.close(); }
  if (fatal) throw fatal;
  console.log(`done ${family}: ${samples} samples in ${dir}${code ? ` (exit ${code})` : ''}`); process.exit(code);
}
