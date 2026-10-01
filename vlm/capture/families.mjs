// vlm/capture/families.mjs — per-family episode plans for drive.mjs: page URL, seeds, render seed, policy, then the episode
// driver and record assembly. nextEpisode(D, {episode, seed0, rsOverride, forceWhen}) -> {records, wallMs, stats} | null
// (a discarded episode) | {exhausted: true} (a Z episode past the end of the plan and of the spares taken).
import fs from 'node:fs';
import path from 'node:path';
import { renderSeed, assemble, capSeverity4, writeEpisode } from './records.mjs';
import { runCorridorEpisode, aWarmup } from './episode_corridor.mjs';
import { runZoomLocation } from './episode_zoom.mjs';
import { runLanding } from './episode_landing.mjs';
import { runDocking } from './episode_docking.mjs';
import { Discard } from './session.mjs';
import { PRE_KICK } from './probe/frame.js';
import { viewSplit, buildOodMask } from '../gen/sampler_z.js';
import { loadNaturalEarth } from '../gen/geo/naturalearth.js';
import { nextSpare } from './zplan.mjs';
import { planEpisode, aCells, landingSchedule, dockingSchedule, rngOf } from '../gen/sampler.js';
import { landingSafety, dockingSafety, eyeView } from '../gen/safety.js';
const UTC0 = Date.UTC(2026, 5, 21, 12), LACIE = '/Volumes/LaCie/astro-pilot/vlm';
// --force-when cases (G4 determinism, §7.5): A in cloud, china, a city route, cloudy; L night + rain + fog below the base
const FORCE = { cloud: { route: 'alps', sky: 'storm' }, china: { route: 'china', sky: 'fair' }, city: { route: 'newyork', sky: 'fair' }, cloudy: { route: 'alps', sky: 'cloudy' }, below_base: { time: 'night', rain: 1, vis: 'fog' } };
const lab = (fam, b) => ({ facts: b.facts, safety: (fam === 'L' ? landingSafety : dockingSafety)({ now: b.now, outcome: b.outcome }), safety_eye: eyeView(fam, { now: b.now, outcome: b.outcome }) });
// A cells (§3.4): the drive fills each route x sky cell toward its quota. On a resume the primary record counts are rebuilt from
// the finished records on disk (the resume scan has already removed unfinished episodes) and the tries (pages) per cell from
// throughput.jsonl (finished pages) plus the discards in drive.log.jsonl (M-7), so neither the cell order nor the dead-cell
// guard restarts.
const readLines = (f) => (fs.existsSync(f) ? fs.readFileSync(f, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l)) : []);
function aCounts(D) {
  if (D.cellCounts) return D.cellCounts;
  D.cellCounts = {}; D.cellTries = {};
  for (const n of fs.existsSync(D.dir) ? fs.readdirSync(D.dir).filter((x) => /^A_.+\.json$/.test(x)) : []) {
    const r = JSON.parse(fs.readFileSync(path.join(D.dir, n), 'utf8')), q = new URLSearchParams(r.provenance.page_url.split('?')[1]), k = `${q.get('route')}|${q.get('sky')}`;
    if (!r.provenance.twin_of) D.cellCounts[k] = (D.cellCounts[k] || 0) + 1;
  }
  for (const o of readLines(path.join(D.dir, 'throughput.jsonl')).concat(readLines(path.join(D.dir, 'drive.log.jsonl')).filter((x) => x.discard !== undefined)))
    if (o.cell) D.cellTries[o.cell] = (D.cellTries[o.cell] || 0) + 1;
  return D.cellCounts;
}
// --force-when cell:<route>|<sky> forces one A cell (throughput measurements); as every forced episode, it has no injection
const forceOf = (w) => (!w ? {} : w.startsWith('cell:') ? { route: w.slice(5).split('|')[0], sky: w.slice(5).split('|')[1] } : FORCE[w] || {});
// a cell whose episodes keep ending before any record (a storm cell the search pilot cannot fly: Task 3's pilot crashed 16/16
// Node alps storm episodes within 7.3 s) is set aside after DEAD_TRIES tries without one, so the fill loop cannot spin on it
export const DEAD_TRIES = 30;
export function pickCell(counts, tries = {}) {
  const key = (c) => `${c.route}|${c.sky}`, live = aCells(1600).filter((c) => !((tries[key(c)] || 0) >= DEAD_TRIES && !counts[key(c)]));
  return live.length ? live.map((c) => ({ c, left: (c.quota - (counts[key(c)] || 0)) / c.quota })).sort((a, b) => b.left - a.left)[0].c : null;
}
function plan(D, family, e, seed0, rsOverride, forceWhen = null) {
  let cell = null; const force = forceOf(forceWhen); D.lastCell = null;
  if (family === 'A') {
    const n = aCounts(D); cell = forceWhen ? { ...force, quota: 1 } : pickCell(n, D.cellTries);
    if (!cell) throw new Discard('every A cell is set aside (no record in its first 30 episodes)');
    const k = `${cell.route}|${cell.sky}`; D.cellTries[k] = (D.cellTries[k] || 0) + 1; D.lastCell = k;
  }
  const seed = seed0 + e, rs = rsOverride ? +rsOverride : renderSeed(D.run, family, seed), p = planEpisode(family, seed, { rs, cell, force }), rngSeed = seed ^ 0xa5a5, rng = rngOf(rngSeed);
  const ep = { ...p, episode: e, renderSeed: rs, utcMs: UTC0 + seed * 60000, rng, rngSeed, cell, force: forceWhen, policySha: null };
  // S/A: the pilot the probe asserts (S: the belt policy's step count) and hashes (drive.mjs: policy.json for S, policy_atmo.json for A)
  if (family === 'S' || family === 'A') { ep.params = { ...p.params, measure: !!D.measure, policySteps: family === 'S' && D.policy ? D.policy.steps : null }; ep.policySha = (family === 'S' || p.policyId === 'atmo_ppo') && D.policy ? D.policy.sha : null; }
  if (forceWhen) ep.inject = null;
  if (family === 'A') ep.warmup = (S) => aWarmup(S, D, ep);
  if (family === 'L' || family === 'D') ep.preKick = PRE_KICK[family];
  if (family === 'L') ep.schedule = landingSchedule(rng);
  if (family === 'D') ep.schedule = dockingSchedule(rng, p.params.start);
  if (ep.inject && ep.inject.at !== 'runtime' && family !== 'L') ep.inject = family === 'D' && ep.inject.kind === 'failed_p6' ? ep.inject : null;
  return ep;
}
async function runFamily(D, ep) {
  if (ep.family === 'A' || ep.family === 'S') {
    // at most 2 severity-4 samples per episode, i.e. per segment of an A page (families: segment k is episode e + k)
    const r = await runCorridorEpisode(D, ep), bySeg = new Map();
    for (const s of r.samples) { const k = s.seg ?? 0; if (!bySeg.has(k)) bySeg.set(k, []); bySeg.get(k).push(s); }
    return { ...r, samples: [...bySeg.values()].flatMap((x) => capSeverity4(x)) };
  }
  const r = ep.family === 'L' ? await runLanding(D, ep) : await runDocking(D, ep); for (const s of r.samples) s.label = lab(ep.family, s.label); return r;
}
// A twin sample must repeat its original's page-time history (I-2, M-4): the same step, and per frame the same step and page
// clock; one that does not is dropped (logged), never recorded as a minimal pair
export const sameHistory = (t, o) => !!o && t.step === o.step && t.frames.length === o.frames.length
  && t.frames.every((f, i) => f.step === o.frames[i].step && f.clock.perf_ms === o.frames[i].clock.perf_ms && f.clock.date_ms === o.frames[i].clock.date_ms);
// One S/A/L/D page and, when it has injected samples, its minimal-pair twin (§8; episode index + 50000). An A page samples across
// the page's own crash resets (ruling T10-a): segment k is episode e + k, so {episodes} lists e .. e + resets. S/A twins run the
// original's candidate loop with no collision course (episode_corridor.mjs); L/D twins replay the original's page-time ops with
// no kick armed (L: the twin URL has no wind injection and hFlare is off; D failed-P6: the probe-owned sim with no failed jet).
// A twin that is discarded yields no records; the injected records stay (the build counts unpaired ones).
export async function nextFlightEpisode(D, e, { seed0, rsOverride, forceWhen = null, maxSpan = 1 }) {
  const ep = { ...plan(D, D.family, e, seed0, rsOverride, forceWhen), maxSegments: Math.max(1, maxSpan) }, res = await runFamily(D, ep), eye = ep.family === 'D' ? 'centreline' : 'chase', seg = (s) => s.seg ?? 0;
  if (ep.inject && ep.inject.at !== 'runtime') for (const s of res.samples) s.injection = { kind: ep.inject.kind, params: ep.inject.params || {}, step: null, sim_t_s: null };
  const span = (res.resets ?? 0) + 1, recOf = (s) => assemble(D, { ...ep, episode: e + seg(s), pageEpisode: e }, s, { eyeView: eye });
  const episodes = Array.from({ length: span }, (_, k) => ({ episode: e + k, records: res.samples.filter((s) => seg(s) === k).map(recOf) })), recs = episodes.flatMap((x) => x.records);
  const stats = { crashed: res.crashed, resets: res.resets ?? 0, policy: ep.policyId, cell: ep.cell ? `${ep.cell.route}|${ep.cell.sky}` : null, inject: ep.inject ? ep.inject.kind : null, ...res.stats };
  // only records not on disk yet count toward the cell (M-a: aCounts has already counted the finished ones)
  if (ep.family === 'A') for (const r of recs) if (!r.error && !fs.existsSync(path.join(D.dir, `${r.rec.key}.json`))) D.cellCounts[`${ep.cell.route}|${ep.cell.sky}`] = (D.cellCounts[`${ep.cell.route}|${ep.cell.sky}`] || 0) + 1;
  const injected = res.samples.filter((s) => s.injection);
  if (!ep.inject || !injected.length) return { episodes, records: recs, wallMs: res.wallMs, stats };
  const flight = ep.family === 'L' || ep.family === 'D', k0 = seg(injected[0]);
  const tw = { ...ep, twin: true, twinOf: e + k0, episode: e + k0 + 50000, pageEpisode: e + 50000, url: ep.twinUrl ?? ep.url, params: { ...ep.params, hflare: false }, ops: flight ? res.ops : null, keep: new Set(injected.map((s) => s.idx)), rng: rngOf(ep.rngSeed), warmup: null };
  if (ep.family === 'A') tw.warmup = (S) => aWarmup(S, D, tw);
  let tr; try { tr = await runFamily(D, tw); } catch (err) { if (!(err instanceof Discard)) throw err; D.log({ discard: 'twin', episode: e, why: err.message }); tr = { samples: [], wallMs: 0, stats: {} }; }
  const orig = (s) => (flight ? res.samples.find((o) => o.idx === s.idx) : res.samples.find((o) => o.ci === s.ci && seg(o) === seg(s)));
  const good = tr.samples.filter((s) => { const ok = sameHistory(s, orig(s)); if (!ok) D.log({ drop: `twin sample ${s.step}`, episode: e, why: 'the twin page-time history differs from the original' }); return ok; });
  // M-c: a twin's sampler_weight is its original's (the original's may carry the severity-4 cap factor of its own segment)
  for (const s of good) s.weight = orig(s).weight;
  const twinRecs = good.map((s) => assemble(D, tw, s, { eyeView: eye }));
  return { episodes, records: recs, twin: { episode: tw.episode, records: twinRecs }, wallMs: res.wallMs + tr.wallMs, stats: { ...stats, twin: twinRecs.length, twin_dropped: tr.samples.length - good.length, twin_wall_s: tr.wallMs / 1000, twin_stats: tr.stats } };
}
// The split re-check on the captured 16x9 grid (§8, ruling for T9/T11): the plan's level-3 blocks and the OOD mask rebuilt
// from the same Natural Earth layers; a captured view that breaks its location's split is discarded (logged), not failed.
async function splitCheck(D) {
  if (!D.splitCheck) {
    const ne = await loadNaturalEarth(`${LACIE}/geo`, ['ne_10m_admin_0_countries', 'ne_10m_land']), blocks = new Map(Object.entries(D.zplan.blocks || {})), ood = buildOodMask(ne);
    D.splitCheck = (facts) => viewSplit(facts['grid.latlon'].v.map((p) => (p ? { lat: p[0], lon: p[1] } : null)), blocks, ood);
  }
  return D.splitCheck;
}
// Z episode e -> plan location: episodes 0..n-1 are plan.locations; a spare (plan.spares, zplan.mjs) runs as the next free
// episode only after a location is discarded. zorder.json (indexes into locations ++ spares) persists the order for a resume.
function zOrder(D) {
  if (!D.zorder) { const f = path.join(D.dir, 'zorder.json'); D.zorder = fs.existsSync(f) ? JSON.parse(fs.readFileSync(f, 'utf8')) : { order: D.zplan.locations.map((_, i) => i), spareFor: {} }; }
  return D.zorder;
}
export function zLocation(D, e) { const all = D.zplan.locations.concat(D.zplan.spares || []), i = zOrder(D).order[e]; return i === undefined ? null : all[i]; }
// after episode e's location was discarded (no record kept): queue the next spare, preferring the same split; returns the
// spare index or null when none is left
export function replaceDiscarded(D, e) {
  const N = D.zplan.locations.length, spares = D.zplan.spares || [], Z = zOrder(D), loc = zLocation(D, e);
  if (!loc || Z.spareFor[e] !== undefined) return null;
  const s = nextSpare(spares, new Set(Z.order.filter((i) => i >= N).map((i) => i - N)), loc.split);
  if (s === null) return null;
  Z.order.push(N + s); Z.spareFor[e] = s; fs.mkdirSync(D.dir, { recursive: true }); fs.writeFileSync(path.join(D.dir, 'zorder.json'), JSON.stringify(Z));
  return s;
}
// Closes episode e: a Z location that kept no view first queues its spare (zorder.json), then gets its 0-record .done, so a
// crash between the two never leaves a closed location without its replacement (step 0 of Task 10); returns the spare
export function closeEpisode(D, e, recs, files, { why = null, write = writeEpisode } = {}) {
  let spare = null;
  if (D.family === 'Z' && !recs.length) { spare = replaceDiscarded(D, e); D.log({ spare, for_episode: e, why }); }
  write(D.dir, recs, files, e);
  return spare;
}
function zStats(res, kept) {
  const n = res.views.length, sum = (f) => res.views.reduce((a, x) => a + f(x), 0), png = (x) => Math.floor(((x.frame.png.length - x.frame.png.indexOf(',') - 1) * 3) / 4);
  return { views: n, dropped: res.drops, split_dropped: kept.splitDropped, boot_ms: res.bootMs, frames: res.frames, frame_ms: res.frameMs, settle_s: res.views.map((x) => x.settle_s), settle_frames: res.views.map((x) => x.settle_frames), gate_frames: res.views.map((x) => x.gate_frames), steady_frames: res.views.map((x) => x.steady_frames),
    tiles: res.views.map((x) => x.tiles), cap_host_ms: n ? sum((x) => x.capHostMs) / n : null, png_bytes: n ? sum(png) / n : null, ground_src: res.views.map((x) => x.label.ground.src) };
}
export async function nextEpisode(D, { episode, seed0, rsOverride, forceWhen = null, maxSpan = 1 }) {
  try {
    if (D.family === 'Z') {
      if (!D.zplan) D.zplan = JSON.parse(fs.readFileSync(`${LACIE}/raw/${D.run}/Z/plan.json`, 'utf8'));
      const loc = zLocation(D, episode); if (!loc) return { exhausted: true };
      const rs = rsOverride ? +rsOverride : renderSeed(D.run, 'Z', loc.id), ep = { family: 'Z', episode, seed: loc.id, renderSeed: rs, utcMs: UTC0, url: `/vlm/capture/zoom.html?rs=${rs}&licence=${D.licence}&lat=${loc.place.lat}&lon=${loc.place.lon}`, id: loc.id, views: loc.views };
      const res = await runZoomLocation(D, ep), check = await splitCheck(D), kept = { splitDropped: [] };
      const views = res.views.filter((x) => { const got = check(x.label.facts); if (got === loc.split) return true; kept.splitDropped.push(x.index); D.log({ drop: 'Z view', loc: loc.id, view: x.index, why: `split re-check: ${got} is not ${loc.split}` }); return false; });
      // the record step is the view's index in the plan, so a dropped view never renumbers the others (stable keys across runs)
      const recs = views.map((x) => assemble(D, ep, { step: x.index, frame: x.frame, label: x.label, v: x.v, ledger0: x.ledger0 }));
      return { records: recs, wallMs: res.wallMs, stats: zStats(res, kept) };
    }
    // the await keeps a rejected Discard inside this try, so a discarded episode returns null instead of ending the drive
    return await nextFlightEpisode(D, episode, { seed0, rsOverride, forceWhen, maxSpan });
  } catch (err) { if (err instanceof Discard) { D.log({ discard: episode, why: err.message, ...(D.lastCell ? { cell: D.lastCell } : {}) }); return null; } throw err; }
}
