// node vlm/capture/zplan.mjs --run <name> --seed 7 --n 200 [--licence open]  -> $LACIE/raw/<run>/Z/plan.json (+ blocks, unfilled bins)
import fs from 'node:fs';
import { loadNaturalEarth } from '../gen/geo/naturalearth.js';
import { planZoom, assignBlocks, buildOodMask, blockKey } from '../gen/sampler_z.js';
const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; }, run = arg('run'), seed = +arg('seed', 7), n = +arg('n', 200);
const ne = await loadNaturalEarth('/Volumes/LaCie/astro-pilot/vlm/geo'), w = new Map();
for (const f of ne.layers.ne_10m_populated_places) if ((f.props.POP_MAX ?? 0) >= 50000) { const k = blockKey(f.point[1], f.point[0]); w.set(k, (w.get(k) || 0) + 1); }
for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) if (!w.has(`3/${x}/${y}`)) w.set(`3/${x}/${y}`, 0.01);
const blocks = assignBlocks(seed, w), ood = buildOodMask(ne), plan = planZoom({ seed, nLocations: n, ne, blocks, ood, profile: arg('licence', 'open') });
fs.mkdirSync(`/Volumes/LaCie/astro-pilot/vlm/raw/${run}/Z`, { recursive: true });
fs.writeFileSync(`/Volumes/LaCie/astro-pilot/vlm/raw/${run}/Z/plan.json`, JSON.stringify({ seed, blocks: Object.fromEntries(blocks), ...plan }));
console.log(`plan: ${plan.locations.length} locations, ${plan.locations.reduce((a, l) => a + l.views.length, 0)} views, unfilled ${JSON.stringify(plan.unfilled)}`);
