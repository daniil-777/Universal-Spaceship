// vlm/tools/overnight.mjs — the APV v1 overnight capture (plan rulings V1-1b, V1-4, V1-5, V1-7): two browser lanes over one run,
// each claiming whole families in its own order (lane SALD: A, S, L, D; lane Z: Z, then D, L, S), one drive per claimed family
// run to its target (drive.mjs resumes a family safely at the episode boundary), a watchdog that restarts a drive that died
// (never after the 403/429 stop latch, exit 3), a stall guard, an on-disk size cap (du -sk of raw/<run>, the budget ruling)
// and a hard stop at a wall-clock time (SIGINT to each drive's process group; the next start resumes). On finish it writes
// raw/<run>/CAPTURE_DONE.json. Progress: $LOGS/v1_overnight.log (events and a status line every 2 min), drive output in
// $LOGS/v1_lane<name>.log. Run it detached under caffeinate -dimsu (vlm/tools/detach.py). Ruling T10-i: no process this
// starts carries the Chromium preflight pattern (or the training one) in its command line; check it with cmdlineClean().
import fs from 'node:fs';
import path from 'node:path';
import { spawn, execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

export const LACIE = '/Volumes/LaCie/astro-pilot/vlm', STOP_EXIT = 3, RESOURCE_EXIT = 4, FAMS = ['S', 'A', 'L', 'D', 'Z'];
export const LANES = Object.freeze({ SALD: ['A', 'S', 'L', 'D'], Z: ['Z', 'D', 'L', 'S'] });
// drive.mjs preflight patterns (Chromium, training/export, Monte Carlo): a watcher whose own command line matched would be
// counted as a browser or a training job by the lanes' preflight
const PREFLIGHT = [/Chrome for Testing|chromium|chrome-headless-shell/i, /vlm\/train\/|train\.py|lora_train|export_onnx|export_decoder/, /real_mc|landing_mc/];
export const cmdlineClean = (args) => !PREFLIGHT.some((re) => re.test(args.join(' ')));
export const parseKV = (s, num = true) => Object.fromEntries(String(s || '').split(',').filter(Boolean).map((kv) => { const [k, v] = kv.split('='); return [k.trim(), num ? +v : v]; }));
// the next local wall-clock HH:MM strictly after now (the 07:00 stop of a run started the evening before)
export function nextStopTime(now, hhmm) {
  const [h, m] = hhmm.split(':').map(Number), t = new Date(now); t.setHours(h, m, 0, 0);
  if (t.getTime() <= now) t.setDate(t.getDate() + 1);
  return t.getTime();
}
// a family's finished records, without touching the directory (drive.mjs's scanRun deletes unfinished episodes: never here)
export function countFamily(dir) {
  if (!fs.existsSync(dir)) return { records: 0, episodes: 0, twins: 0 };
  const names = fs.readdirSync(dir).filter((n) => !n.startsWith('._')), done = new Set();
  for (const n of names) { const m = /^episode_(\d{5})\.done$/.exec(n); if (m) done.add(+m[1]); }
  let records = 0, twins = 0;
  for (const n of names) { const m = /^[SALDZ]_.+_(\d{5})_\d{6}\.json$/.exec(n); if (m && done.has(+m[1])) { records++; if (+m[1] >= 50000) twins++; } }
  return { records, episodes: done.size, twins };
}
// the next family a lane takes: the first in its order that is neither done nor held by another lane
export const pickFamily = (order, st) => order.find((f) => st.targets[f] > 0 && !st.done[f] && !st.held[f]) ?? null;
// what the lane does after a drive exits: done (the family reached its target, or a clean exit added nothing: the plan or the
// EOX budget is exhausted), stop (the 403/429 latch, or the launcher is stopping), wait-restart (free memory fell below the
// floor), restart (a crash; given up after maxFails in a row that each added nothing)
export function decide({ code, signal, added, count, target, stopping, fails, maxFails = 6 }) {
  if (stopping) return 'stop';
  if (code === STOP_EXIT) return 'stop-latch';
  if (code === 0) return count >= target || added === 0 ? 'done' : 'rerun';
  if (code === RESOURCE_EXIT) return 'wait-restart';
  return fails + 1 >= maxFails && added === 0 ? 'failed' : 'restart';
}
// orphans: a main browser (no --type= flag) with our LaCie temp profile whose parent died (ppid 1), left by a drive killed
// before Playwright could close it; it would count as a lane's browser in every later preflight. Never another profile
// (the hyperframes preview browser keeps its profile under /var/folders).
export const orphanBrowsers = (psOut, tmp = `${LACIE}/tmp/`) => String(psOut).split('\n').map((l) => /^\s*(\d+)\s+(\d+)\s+(.*)$/.exec(l)).filter(Boolean)
  .filter(([, , ppid, cmd]) => +ppid === 1 && cmd.includes(`--user-data-dir=${tmp}`) && !/\s--type=/.test(cmd)).map(([, pid]) => +pid);
const sh = (cmd, args) => { try { return execFileSync(cmd, args, { maxBuffer: 1 << 26 }).toString(); } catch (e) { return String(e.stdout || ''); } };
export const freePct = () => +(/free percentage: (\d+)%/.exec(sh('memory_pressure', [])) || [0, 0])[1];
export const duKB = (dir) => +(sh('du', ['-sk', dir]).split('\t')[0] || 0), duApparentKB = (dir) => +(sh('du', ['-A', '-sk', dir]).split('\t')[0] || 0);

if (process.argv[1] && process.argv[1].endsWith('overnight.mjs')) {
  const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
  const REPO = fileURLToPath(new URL('../../', import.meta.url)), run = arg('run'), logs = arg('logs', path.join(LACIE, 'logs')), prefix = arg('log-prefix', 'v1');
  if (!run) throw new Error('usage: overnight.mjs --run <name> --targets A=2500,S=3200,L=2000,D=1500,Z=800 --seeds S=200000,A=300000,L=380000,D=500000 --stop-at 07:00 --cap-gb 38 [--zplan-seed 41 --zplan-n 200 --blocks-from apv-pilot] [--lanes SALD,Z] [--stall-min 30]');
  const runDir = path.join(LACIE, 'raw', run), targets = parseKV(arg('targets', 'A=2500,S=3200,L=2000,D=1500,Z=800')), seeds = parseKV(arg('seeds', 'S=200000,A=300000,L=380000,D=500000'));
  const stopAt = arg('stop-in-min') ? Date.now() + +arg('stop-in-min') * 60000 : nextStopTime(Date.now(), arg('stop-at', '07:00')), capKB = +arg('cap-gb', 38) * 1e9 / 1024, stallMs = +arg('stall-min', 30) * 60000;
  const lanes = arg('lanes', 'SALD,Z').split(','), t0 = Date.now(), gitSha = sh('git', ['-C', REPO, 'rev-parse', '--short', 'HEAD']).trim();
  fs.mkdirSync(runDir, { recursive: true }); fs.mkdirSync(logs, { recursive: true });
  const logFile = path.join(logs, `${prefix}_overnight.log`), iso = (t = Date.now()) => new Date(t).toLocaleString('sv-SE');
  const log = (msg) => { const line = `${iso()} ${msg}`; fs.appendFileSync(logFile, line + '\n'); console.log(line); };
  const st = { targets, done: {}, held: {}, status: {}, stopping: false, why: null, latched: false, kids: new Map() };
  for (const f of FAMS) if (!(targets[f] > 0)) st.done[f] = true;
  const counts = () => Object.fromEntries(FAMS.map((f) => [f, countFamily(path.join(runDir, f))]));
  log(`start run ${run} at ${gitSha}: targets ${JSON.stringify(targets)} seeds ${JSON.stringify(seeds)} stop ${iso(stopAt)} cap ${arg('cap-gb', 38)} GB (du -sk) lanes ${lanes.join('+')} free ${freePct()} %`);
  // the Z plan (V1-4): once per run, the block splits from --blocks-from; a resume keeps the plan on disk
  if (targets.Z > 0 && !fs.existsSync(path.join(runDir, 'Z', 'plan.json'))) {
    const za = ['vlm/capture/zplan.mjs', '--run', run, '--seed', String(arg('zplan-seed', 41)), '--n', String(arg('zplan-n', 200)), '--licence', 'open'].concat(arg('blocks-from') ? ['--blocks-from', arg('blocks-from')] : []);
    if (!cmdlineClean(za)) throw new Error('zplan command line matches the preflight pattern');
    log(`zplan: ${sh('node', za).trim()}`);
  }
  const stopAll = (why) => {
    if (st.stopping) return; st.stopping = true; st.why = why; log(`STOP: ${why}; SIGINT to ${st.kids.size} drive(s)`);
    for (const [pid] of st.kids) { try { process.kill(-pid, 'SIGINT'); } catch { /* gone */ } }
    setTimeout(() => { for (const [pid] of st.kids) { try { process.kill(-pid, 'SIGKILL'); log(`SIGKILL drive group ${pid} (still alive 60 s after SIGINT)`); } catch { /* gone */ } } }, 60000).unref();
  };
  // one drive: its own process group (detached), so a stop reaches node and its browser; the lane log gets its output
  function drive(lane, fam) {
    const args = ['vlm/capture/drive.mjs', '--family', fam, '--n', String(targets[fam]), '--run', run, '--licence', 'open'].concat(fam === 'Z' ? [] : ['--seed0', String(seeds[fam])]);
    if (!cmdlineClean(args)) throw new Error(`drive command line matches the preflight pattern: ${args.join(' ')}`);
    const out = fs.openSync(path.join(logs, `${prefix}_lane${lane}.log`), 'a');
    fs.writeSync(out, `${iso()} lane ${lane}: node ${args.join(' ')}\n`);
    const p = spawn('node', args, { cwd: REPO, detached: true, stdio: ['ignore', out, out], env: { ...process.env, APV_BROWSER_LANES: String(lanes.length) } });
    st.kids.set(p.pid, { lane, fam, t0: Date.now() }); fs.closeSync(out);
    return new Promise((res) => p.on('exit', (code, signal) => { st.kids.delete(p.pid); res({ code, signal, pid: p.pid }); }));
  }
  const sleep = (ms) => new Promise((r) => { const t = setTimeout(r, ms); if (st.stopping) clearTimeout(t), r(); });
  async function laneLoop(lane) {
    const order = LANES[lane];
    for (let fam = pickFamily(order, st); fam && !st.stopping; fam = pickFamily(order, st)) {
      st.held[fam] = lane; let fails = 0;
      for (;;) {
        const before = countFamily(path.join(runDir, fam)).records, t = Date.now();
        log(`lane ${lane}: ${fam} start (${before}/${targets[fam]})`);
        const r = await drive(lane, fam), after = countFamily(path.join(runDir, fam)).records, added = after - before;
        for (const pid of orphanBrowsers(sh('ps', ['-axo', 'pid=,ppid=,command=']))) { try { process.kill(pid, 'SIGKILL'); log(`killed orphaned lane browser ${pid}`); } catch { /* gone */ } }
        const d = decide({ code: r.code, signal: r.signal, added, count: after, target: targets[fam], stopping: st.stopping, fails });
        log(`lane ${lane}: ${fam} exit ${r.code ?? r.signal} after ${((Date.now() - t) / 60000).toFixed(1)} min, +${added} (${after}/${targets[fam]}) -> ${d}`);
        if (d === 'done') { st.done[fam] = true; st.status[fam] = after >= targets[fam] ? 'target' : 'exhausted'; break; }
        if (d === 'stop') { st.status[fam] = 'stopped'; break; }
        if (d === 'stop-latch') { st.status[fam] = 'stop-latch'; st.latched = true; stopAll(`drive ${fam} exited ${STOP_EXIT} (403/429 stop latch: raw/${run}/stop.json); no restart`); break; }
        if (d === 'failed') { st.done[fam] = true; st.status[fam] = 'failed'; log(`lane ${lane}: ${fam} given up after ${fails + 1} failed starts in a row`); break; }
        fails = added > 0 ? 0 : fails + 1;
        if (d === 'wait-restart') await sleep(120000); else if (d === 'restart') await sleep(15000);
        if (st.stopping) { st.status[fam] = 'stopped'; break; }
      }
      delete st.held[fam];
    }
    log(`lane ${lane}: end`);
  }
  // watchers: status + size cap every 2 min, the wall-clock stop, the stall guard (no episode written for stall-min)
  let lastDu = 0;
  const tick = () => {
    const kb = duKB(runDir), c = counts(); lastDu = kb;
    log(`status: ${FAMS.map((f) => `${f} ${c[f].records}/${targets[f] || 0}`).join(' ')} | ${(kb * 1024 / 1e9).toFixed(2)} GB on disk | free ${freePct()} % | drives ${[...st.kids.values()].map((k) => `${k.lane}:${k.fam}`).join(' ') || '-'}`);
    if (kb >= capKB) stopAll(`raw/${run} is ${(kb * 1024 / 1e9).toFixed(2)} GB on disk (cap ${arg('cap-gb', 38)} GB)`);
    for (const [pid, k] of st.kids) {
      const dir = path.join(runDir, k.fam), mt = Math.max(k.t0, ...['throughput.jsonl', 'drive.log.jsonl', 'checkpoint.json'].map((n) => { try { return fs.statSync(path.join(dir, n)).mtimeMs; } catch { return 0; } }));
      if (Date.now() - mt > stallMs) { log(`stall: lane ${k.lane} ${k.fam} wrote nothing for ${Math.round((Date.now() - mt) / 60000)} min; SIGINT group ${pid} (the lane restarts it)`); try { process.kill(-pid, 'SIGINT'); } catch { /* gone */ } k.t0 = Date.now(); }
    }
  };
  const iv = setInterval(tick, 120000), stopT = setTimeout(() => stopAll(`hard stop at ${iso(stopAt)}`), Math.max(0, stopAt - Date.now()));
  process.on('SIGINT', () => stopAll('SIGINT to the launcher')); process.on('SIGTERM', () => stopAll('SIGTERM to the launcher'));
  tick();
  // lanes start 5 s apart, so their preflights (10 s period) do not run in step
  await Promise.all(lanes.map((l, i) => sleep(5000 * i).then(() => laneLoop(l))));
  clearInterval(iv); clearTimeout(stopT);
  const c = counts(), kb = duKB(runDir), akb = duApparentKB(runDir), t1 = Date.now();
  const done = { run, git_sha: gitSha, started: new Date(t0).toISOString(), stopped: new Date(t1).toISOString(), started_local: iso(t0), stopped_local: iso(t1), wall_h: +((t1 - t0) / 3.6e6).toFixed(2),
    why: st.why || 'all targets reached', stop_latch: st.latched, targets, seeds, counts: c, records_total: FAMS.reduce((a, f) => a + c[f].records, 0), status: Object.fromEntries(FAMS.map((f) => [f, st.status[f] || (targets[f] > 0 ? 'unfinished' : 'no target')])),
    bytes_on_disk: kb * 1024, bytes_apparent: akb * 1024, gb_on_disk: +(kb * 1024 / 1e9).toFixed(2), cap_gb: +arg('cap-gb', 38), eox: (() => { try { return JSON.parse(fs.readFileSync(path.join(runDir, 'eox_budget.json'), 'utf8')); } catch { return null; } })() };
  fs.writeFileSync(path.join(runDir, 'CAPTURE_DONE.json'), JSON.stringify(done, null, 1));
  log(`done: ${JSON.stringify({ why: done.why, counts: Object.fromEntries(FAMS.map((f) => [f, c[f].records])), gb_on_disk: done.gb_on_disk })} -> ${path.join(runDir, 'CAPTURE_DONE.json')}`);
  process.exit(0);
}
