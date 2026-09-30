// vlm/capture/families.mjs — per-family episode plans for drive.mjs: page URL, seeds, render seed, policy, then the episode
// driver and record assembly. nextEpisode(D, {episode, seed0, rsOverride, forceWhen}) -> {records, wallMs, stats} | null
// (a discarded episode) | {exhausted: true} (a Z episode past the end of the plan and of the spares taken).
import fs from 'node:fs';
import path from 'node:path';
import { mulberry32 } from '../../src/mathx.js';
import { renderSeed, assemble, capSeverity4, writeEpisode } from './records.mjs';
import { runCorridorEpisode } from './episode_corridor.mjs';
import { runZoomLocation } from './episode_zoom.mjs';
import { Discard } from './session.mjs';
import { viewSplit, buildOodMask } from '../gen/sampler_z.js';
import { loadNaturalEarth } from '../gen/geo/naturalearth.js';
import { nextSpare } from './zplan.mjs';
const rngOf = (seed) => { const r = mulberry32(seed); return { float: r, int: (lo, hi) => lo + Math.floor(r() * (hi - lo + 1)), pick: (a) => a[Math.floor(r() * a.length)] }; };
const UTC0 = Date.UTC(2026, 5, 21, 12), LACIE = '/Volumes/LaCie/astro-pilot/vlm';
function planS(D, e, seed0, rsOverride) {
  const seed = seed0 + e, rng = rngOf(seed ^ 0x5a5a), density = rng.pick([10, 25, 40]), rs = rsOverride ? +rsOverride : renderSeed(D.run, 'S', seed);
  // policy: the pretrained belt pilot, asserted in the page (probe setup) against model/policy.json's step count and hashed
  return { family: 'S', episode: e, seed, renderSeed: rs, utcMs: UTC0 + seed * 60000, rng, policyId: 'belt_ppo', policySha: D.policy ? D.policy.sha : null, params: { density, measure: !!D.measure, policySteps: D.policy ? D.policy.steps : null },
    url: `/index.html?hud=0&lowpass=0&seed=${seed}&density=${density}&rs=${rs}` };
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
export async function nextEpisode(D, { episode, seed0, rsOverride }) {
  try {
    if (D.family === 'S') {
      const ep = planS(D, episode, seed0, rsOverride), res = await runCorridorEpisode(D, ep), recs = capSeverity4(res.samples).map((s) => assemble(D, ep, s));
      return { records: recs, wallMs: res.wallMs, stats: { crashed: res.crashed, ...res.stats } };
    }
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
    throw new Error(`family ${D.family} is added in Task 10`);
  } catch (err) { if (err instanceof Discard) { D.log({ discard: episode, why: err.message }); return null; } throw err; }
}
