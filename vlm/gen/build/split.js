// vlm/gen/build/split.js — group splits (spec §8): S/A by episode (page) seed, L/D by run seed (mod 10: 0-7 train, 8 val,
// 9 test), Z by its plan location (level-3 block, spares included); OOD: A moscow and A cloudy, D far, Z OOD region.
// A twin takes its original's split and group (anchorOf): twins replay their original's page, so the seeds agree, but the
// build does not rely on it. zRecheck() re-reads a captured Z view on the 32x18 grid widened by one cell (CHECK_GRID);
// a view whose re-check breaks its location's split is discarded by the build, never a build failure (amendment B).
import path from 'node:path';
import { viewSplit, CHECK_GRID } from '../sampler_z.js';
import { matrixGrid } from '../labels/zoom.js';

export const seedSplit = (seed) => { const m = ((seed % 10) + 10) % 10; return m <= 7 ? 'train' : m === 8 ? 'val' : 'test'; };
export const queryOf = (rec) => new URLSearchParams(((rec.provenance && rec.provenance.page_url) || '').split('?')[1] || '');
// the record whose seed and page decide the split: the twin's original when the build loaded it, else the record itself
export const anchorOf = (rec, byKey = null) => (rec.provenance && rec.provenance.twin_of && byKey && byKey.get(rec.provenance.twin_of)) || rec;
// a Z seed is a location id of its run's own plan (two runs' plans reuse ids), so a Z group carries the run; a flight seed
// is the scenario itself, the same in every run
export const groupOf = (rec, byKey = null) => { const a = anchorOf(rec, byKey); return a.family === 'Z' && a.run ? `Z:${a.run}:${a.provenance.seed}` : `${a.family}:${a.provenance.seed}`; };
export const zLocation = (zplan, id) => (zplan.locations || []).concat(zplan.spares || []).find((l) => l.id === id) || null;
export function splitOf(rec, { zplan = null, byKey = null } = {}) {
  const a = anchorOf(rec, byKey), p = queryOf(a);
  if (a.family === 'A' && (p.get('route') === 'moscow' || p.get('sky') === 'cloudy')) return 'ood';
  if (a.family === 'D' && p.get('start') === 'far') return 'ood';
  if (a.family !== 'Z') return seedSplit(a.provenance.seed);
  const loc = zplan && zLocation(zplan, a.provenance.seed);
  if (!loc) throw new Error(`${rec.key}: location ${a.provenance.seed} is not in the Z plan`);
  return loc.split;
}
// the view's local-frame origin: the page's lat/lon (the plan's exact values), else the rounded view facts
function originOf(rec) {
  const q = queryOf(rec), lat = parseFloat(q.get('lat')), lon = parseFloat(q.get('lon'));
  if (Number.isFinite(lat) && Number.isFinite(lon)) return { lat, lon };
  return { lat: rec.facts['view.lat_deg'].v, lon: rec.facts['view.lon_deg'].v };
}
// the split the captured camera's widened 32x18 ground grid lies in: 'ood' / a block split, or null when it straddles
// splits (blocks: Map or {get}; ood: {has(lat, lon)}, sampler_z.js buildOodMask)
export function zRecheck(rec, { blocks, ood }) {
  const cam = rec.cameras[path.basename(rec.frames[0])];
  if (!cam) throw new Error(`${rec.key}: no captured camera for ${rec.frames[0]}`);
  return viewSplit(matrixGrid(cam, originOf(rec), CHECK_GRID), blocks, ood);
}
export function assertGroupsDisjoint(recs, splits, byKey = null) {
  const g = new Map();
  recs.forEach((r, i) => { const k = groupOf(r, byKey), s = splits[i]; if (g.has(k) && g.get(k) !== s) throw new Error(`group ${k} is in ${g.get(k)} and ${s}`); g.set(k, s); });
}
