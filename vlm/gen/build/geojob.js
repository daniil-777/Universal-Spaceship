// vlm/gen/build/geojob.js — the Z geo facts of one view in a worker (build.mjs runs a small pool of these beside the frame
// pool: about 0.35 s of point-in-polygon work per view, so ~1000 views no longer hold the main thread for 6-8 minutes).
// Each worker loads Natural Earth and opens the tile cache once (about 0.7 GB). The result is what zoomGeo.apply() adds or
// changes, in its order, so the build (and a cached entry) replays it exactly; geoDiff() is the same in-process.
import { zoomGeo } from './zoomgeo.js';

let G = null;
export async function geoDiff(rec, geo) {
  const old = JSON.parse(JSON.stringify(rec.facts)), probe = { ...rec, facts: JSON.parse(JSON.stringify(rec.facts)), zoom: { ...rec.zoom } };
  await zoomGeo.apply(probe, geo);
  return { facts: Object.fromEntries(Object.entries(probe.facts).filter(([k, x]) => JSON.stringify(old[k]) !== JSON.stringify(x))), tags: probe.zoom.tags };
}
// task: {rec: the view's key, family, facts, cameras, frames, provenance, zoom}, optional {geoDir, tileDir}
export async function job({ rec, geoDir, tileDir }) {
  G ||= await zoomGeo.load({ ...(geoDir ? { geoDir } : {}), ...(tileDir ? { tileDir } : {}) });
  return { result: await geoDiff(rec, G) };
}
