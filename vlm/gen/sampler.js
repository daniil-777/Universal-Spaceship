// vlm/gen/sampler.js — seeded S/A/L/D episode plans (spec §3.2-§3.4, §8): page URLs, behaviour policy, the one injection,
// A route x sky cells with the OOD quotas (moscow 8 %, cloudy on the other six routes 7 %), L seeds >= 1.
import { mulberry32 } from '../../src/mathx.js';
import { drawInjection, P6_SEEDS } from './inject.js';
import { ROUTES } from './labels/worlds.js';
export const SKIES = Object.freeze(['clear', 'fair', 'cloudy', 'storm']);
export const rngOf = (seed) => { const r = mulberry32((seed * 2654435761) >>> 0); return { float: r, int: (lo, hi) => lo + Math.floor(r() * (hi - lo + 1)), pick: (a) => a[Math.floor(r() * a.length)] }; };
export function aCells(total = 1600) {
  const routes = Object.keys(ROUTES), others = routes.filter((r) => r !== 'moscow'), cells = [];
  for (const route of routes) for (const sky of SKIES) {
    const quota = route === 'moscow' ? 0.08 * total / 4 : sky === 'cloudy' ? 0.07 * total / others.length : 0.85 * total / (others.length * 3);
    cells.push({ route, sky, quota: Math.round(quota), ood: route === 'moscow' || sky === 'cloudy' });
  }
  return cells;
}
export function planEpisode(family, seed, { rs, cell = null, force = {} }) {
  const rng = rngOf(seed + 17), inject = drawInjection(family, rng.float);
  if (family === 'S') { const density = rng.pick([10, 25, 40]); return { family, seed, policyId: 'belt_ppo', params: { density }, inject: inject && { ...inject, candidate: rng.int(1, 8) }, url: `/index.html?hud=0&lowpass=0&seed=${seed}&density=${density}&rs=${rs}` }; }
  if (family === 'A') {
    const city = !!ROUTES[cell.route].city, policyId = cell.route === 'alps' ? 'search_v1' : rng.float() < 0.5 ? 'atmo_ppo' : 'search_v1';
    return { family, seed, policyId, params: { route: cell.route, sky: cell.sky, policy: policyId }, inject: inject && { ...inject, candidate: rng.int(1, 8) }, url: `/index.html?hud=0&atmo=1&route=${cell.route}&sky=${cell.sky}&seed=${seed}${city ? '&skyline=1' : ''}&rs=${rs}` };
  }
  if (family === 'L') {
    const s = 1 + (seed % 99999), start = rng.pick(['final', 'random']), time = force.time ?? rng.pick(['day', 'day', 'dusk', 'night']), vis = force.vis ?? rng.pick(['cavok', 'cavok', 'haze', 'fog']), clouds = rng.pick(['FEW', 'SCT', 'BKN', 'OVC', 'NONE']), rain = force.rain ?? (rng.float() < 0.2 ? 1 : 0);
    const base = `/index.html?scenario=landing&view=chase&path=0&start=${start}&seed=${s}&time=${time}&vis=${vis}&clouds=${clouds}&rain=${rain}`, wind = inject && inject.url ? `&wind=${inject.url.wind}${inject.url.turb ? '&turb=' + inject.url.turb : ''}` : '';
    return { family, seed: s, policyId: 'autoland', params: { start, time, vis, clouds, rain, hflare: !!(inject && inject.kind === 'hflare') }, inject, url: `${base}${wind}&rs=${rs}`, twinUrl: `${base}&rs=${rs}` };
  }
  const start = rng.pick(['near', 'final', 'final', 'far']), s = inject && inject.kind === 'failed_p6' ? inject.params.seed : seed;
  return { family: 'D', seed: s, policyId: 'gnc', params: { start }, inject, url: `/index.html?scenario=real&start=${start}&seed=${s}&nav=noisy&filter=1&rs=${rs}` };
}
// Landing times cover a whole run (up to the sim's 900 s limit: a random start lands at 400-490 s, which 60 times 3-8 s apart
// never reach) and are thinned by the drive by height (above 1000 ft kept with p = 0.35 and at most 6, below with p = 1, at
// most 12 per run), which oversamples below 1000 ft (§3.4); docking keeps rho > 30 m with p = 0.4 (episode_docking.mjs).
export function landingSchedule(rng, t0 = 0) { const out = []; for (let t = t0 + 4 + 6 * rng.float(); t < 900; t += 3 + 5 * rng.float()) out.push(+t.toFixed(3)); return out; }
export function dockingSchedule(rng, start) { const [t0, t1, lo, hi] = { far: [60, 9000, 240, 600], near: [20, 3000, 40, 120], final: [5, 600, 8, 20] }[start]; const out = []; for (let t = t0 + lo * rng.float(); t < t1 && out.length < 60; t += lo + (hi - lo) * rng.float()) out.push(+t.toFixed(1)); return out; }
