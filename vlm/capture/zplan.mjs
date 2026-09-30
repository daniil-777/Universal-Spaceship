// node vlm/capture/zplan.mjs --run <name> --seed 7 --n 200 [--licence open]  -> $LACIE/raw/<run>/Z/plan.json (+ blocks, unfilled bins)
// The plan draws n + 15 % locations in one planZoom pass (so the spares share the range-bin stratification) and keeps the
// extra ones as `spares`: the drive runs a spare, in order, only when a location is discarded, preferring a spare of the
// discarded location's split so the per-split quotas still fill (fix round 1).
import fs from 'node:fs';
import { loadNaturalEarth } from '../gen/geo/naturalearth.js';
import { planZoom, assignBlocks, buildOodMask, blockKey } from '../gen/sampler_z.js';
export const SPARE_FRAC = 0.15;
export function withSpares(locations, n) { return { locations: locations.slice(0, n), spares: locations.slice(n) }; }
// the index (into plan.spares) of the next spare for a discarded location of `split`: the first unused one of that split,
// else the first unused one of any split, else null; `used` holds the spare indexes already taken
export function nextSpare(spares, used, split) {
  const free = spares.map((s, i) => i).filter((i) => !used.has(i));
  return free.find((i) => spares[i].split === split) ?? free[0] ?? null;
}
if (process.argv[1] && process.argv[1].endsWith('zplan.mjs')) {
  const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; }, run = arg('run'), seed = +arg('seed', 7), n = +arg('n', 200);
  const ne = await loadNaturalEarth('/Volumes/LaCie/astro-pilot/vlm/geo'), w = new Map();
  for (const f of ne.layers.ne_10m_populated_places) if ((f.props.POP_MAX ?? 0) >= 50000) { const k = blockKey(f.point[1], f.point[0]); w.set(k, (w.get(k) || 0) + 1); }
  for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) if (!w.has(`3/${x}/${y}`)) w.set(`3/${x}/${y}`, 0.01);
  const blocks = assignBlocks(seed, w), ood = buildOodMask(ne), plan = planZoom({ seed, nLocations: Math.ceil(n * (1 + SPARE_FRAC)), ne, blocks, ood, profile: arg('licence', 'open') });
  const { locations, spares } = withSpares(plan.locations, n);
  fs.mkdirSync(`/Volumes/LaCie/astro-pilot/vlm/raw/${run}/Z`, { recursive: true });
  fs.writeFileSync(`/Volumes/LaCie/astro-pilot/vlm/raw/${run}/Z/plan.json`, JSON.stringify({ seed, blocks: Object.fromEntries(blocks), ...plan, locations, spares }));
  console.log(`plan: ${locations.length} locations (+${spares.length} spares), ${locations.reduce((a, l) => a + l.views.length, 0)} views, unfilled ${JSON.stringify(plan.unfilled)}`);
}
