// vlm/gen/calibrate.mjs — G3: collect >= N natural corridor states per behaviour policy x world in Node (S belt PPO at
// density 10/25/40; A every route x {clear, storm} x {atmo PPO, search pilot}), store their rollout summaries, and fix
// c_near in [1.5, 4] u so that CAUTION is 15-45 % of non-UNSAFE states in S and in A-search (spec §4.3).
//   node vlm/gen/calibrate.mjs --n 200 --out vlm/gen/calibration.json --log /Volumes/LaCie/astro-pilot/vlm/logs/calibration_states.jsonl
import fs from 'node:fs';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { pageWorld, registerLongGrids, ROUTES } from './labels/worlds.js';
import { actionSearch, nowState, ppoPilot, createSearchPilot } from './labels/corridor.js';

export function mixAt(states, c) {
  const m = { SAFE: 0, CAUTION: 0, UNSAFE: 0, n: states.length };
  for (const s of states) m[s.pRef >= 0.5 || s.unsafeNow ? 'UNSAFE' : s.pRef > 0 || (s.pRef === 0 && s.medClr <= c) || s.turbSevere ? 'CAUTION' : 'SAFE']++;
  return m;
}
const cautionShare = (states, c) => { const m = mixAt(states, c); return m.SAFE + m.CAUTION ? m.CAUTION / (m.SAFE + m.CAUTION) : 0; };
export function chooseCNear(states, { lo = 1.5, hi = 4, step = 0.25, target = [0.15, 0.45], start = 2.5 } = {}) {
  const S = states.filter((s) => s.group === 'S'), A = states.filter((s) => s.group === 'A_search'), grid = [];
  for (let c = lo; c <= hi + 1e-9; c += step) grid.push({ c: +c.toFixed(2), S: +cautionShare(S, c).toFixed(4), A_search: +cautionShare(A, c).toFixed(4) });
  const ok = grid.filter((g) => g.S >= target[0] && g.S <= target[1] && g.A_search >= target[0] && g.A_search <= target[1]).sort((a, b) => Math.abs(a.c - start) - Math.abs(b.c - start));
  return { c_near: ok.length ? ok[0].c : null, grid };
}

async function collect(cell, n, log) {
  const out = []; let seed = 1000 * (cell.idx + 1);
  while (out.length < n && seed < 1000 * (cell.idx + 1) + 400) {
    const e = pageWorld({ route: cell.route, sky: cell.sky, seed: seed++, density: cell.density ?? 25 }), act = cell.policy === 'search_v1' ? createSearchPilot() : null; let k = 0;
    for (let i = 1; i <= 900 && k < 12 && out.length < n; i++) {
      const r = e.step(act ? act.act(e) : cell.pilot(e)); if (r.done) break;
      if (i >= 23 && i % 23 === 0) {
        const b = actionSearch(e, { K: 4 }), now = nowState(e), st = { group: cell.group, cell: cell.key, pRef: b.CONTINUE.crashes / 4, medClr: Number.isFinite(b.CONTINUE.medClr) ? b.CONTINUE.medClr : 1e9, unsafeNow: now.overstressed || now.stalled || now.pullUp, turbSevere: now.turbSevere };
        out.push(st); fs.appendFileSync(log, JSON.stringify(st) + '\n'); k++;
      }
    }
  }
  return out;
}

if (process.argv[1] && process.argv[1].endsWith('calibrate.mjs')) {
  const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
  const n = +arg('n', 200), outFile = arg('out', 'vlm/gen/calibration.json'), log = arg('log', '/Volumes/LaCie/astro-pilot/vlm/logs/calibration_states.jsonl');
  await registerLongGrids();
  const { PPOAgent } = await import('../../src/ppo.js'), load = (f) => PPOAgent.fromJSON(JSON.parse(fs.readFileSync(fileURLToPath(new URL(`../../model/${f}`, import.meta.url)), 'utf8')));
  const belt = ppoPilot(load('policy.json')), atmo = ppoPilot(load('policy_atmo.json')), cells = [];
  for (const density of [10, 25, 40]) cells.push({ key: `belt_ppo|space|${density}`, group: 'S', route: null, sky: 'fair', density, policy: 'belt_ppo', pilot: belt });
  for (const route of Object.keys(ROUTES)) for (const sky of ['clear', 'storm']) for (const policy of ['atmo_ppo', 'search_v1']) cells.push({ key: `${policy}|${route}|${sky}`, group: policy === 'search_v1' ? 'A_search' : 'A_ppo', route, sky, policy, pilot: atmo });
  cells.forEach((c, i) => { c.idx = i; });
  const all = [];
  for (const c of cells) { const t0 = Date.now(), s = await collect(c, n, log); all.push(...s); console.log(`${c.key}: ${s.length} states in ${((Date.now() - t0) / 1000).toFixed(0)} s`); if (s.length < n) console.log(`WARNING ${c.key} short: ${s.length} < ${n}`); }
  const { c_near, grid } = chooseCNear(all), mixes = {};
  for (const c of cells) mixes[c.key] = mixAt(all.filter((s) => s.cell === c.key), c_near ?? 2.5);
  const res = { c_near, grid, mixes, n_per_cell: n, git_sha: execFileSync('git', ['rev-parse', '--short', 'HEAD']).toString().trim(), date: new Date().toISOString() };
  fs.writeFileSync(outFile, JSON.stringify(res, null, 1) + '\n');
  console.log(c_near === null ? 'G3 FAIL: no c_near in [1.5, 4] gives 15-45 % CAUTION in S and A-search' : `G3 c_near = ${c_near}`);
  process.exit(c_near === null ? 1 : 0);
}
